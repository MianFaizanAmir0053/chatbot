import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import { GUARDRAIL_CONFIG } from "../config";
import { getAuxModels } from "../models";
import { structuredInvoke } from "../structured";
import type { RankedDocument } from "../retrieval/rerank";

/**
 * Only the numeric score and the unsupported claims are requested.
 *
 * An earlier version also asked for a `verdict` enum, which models returned as
 * free text ("Fully supported", "Not Grounded") and Zod rejected — failing the
 * whole guardrail over a redundant field. The verdict is derived from the score
 * instead, which cannot drift.
 */
const GroundednessSchema = z.object({
  groundedness: z
    .number()
    .min(0)
    .max(1)
    .describe("Fraction of the answer's factual claims directly supported by the excerpts"),
  unsupportedClaims: z
    .array(z.string())
    .describe("Claims stated in the answer that the excerpts do not support"),
});

function verdictFor(score: number): GroundednessReport["verdict"] {
  if (score >= 0.8) return "grounded";
  if (score >= GUARDRAIL_CONFIG.MIN_GROUNDEDNESS) return "partially_grounded";
  return "unsupported";
}

export interface GroundednessReport {
  score: number;
  verdict: "grounded" | "partially_grounded" | "unsupported";
  unsupportedClaims: string[];
  passed: boolean;
}

/**
 * Passages sent to the groundedness judge.
 *
 * Enough to cover a well-cited answer plus a little context, far short of a
 * delegating turn's full evidence set.
 */
const GROUNDEDNESS_MAX_EXCERPTS = 12;

/**
 * LLM-as-judge faithfulness check on the generated answer.
 *
 * This is the last line of defence against the failure mode that matters most in
 * RAG: a fluent answer that the documents do not actually support. It runs after
 * generation and gates what the user sees.
 *
 * Fails open — if the judge itself errors we return the answer rather than
 * blocking on a broken guardrail, but mark it unverified.
 */
export async function checkGroundedness(
  answer: string,
  documents: RankedDocument[],
): Promise<GroundednessReport> {
  // Nothing to check against, or an explicit refusal — no claim is being made.
  if (documents.length === 0 || answer.trim().length < 20) {
    return { score: 1, verdict: "grounded", unsupportedClaims: [], passed: true };
  }

  // Judge against the passages the answer actually cites, not the whole run's
  // evidence.
  //
  // Delegated research changed the scale of this. A single-agent turn gathered
  // a handful of passages; a fan-out of six researchers gathers around thirty,
  // and sending all of them meant a very large prompt to the judge for every
  // answer — serial, after the user is already reading, and mostly made of
  // passages the answer never mentions.
  //
  // Restricting to cited passages is also more accurate, not merely cheaper:
  // the question is whether each claim is supported by what it cites. A few
  // top-ranked uncited passages are kept as well, so that a claim citing
  // nothing can still be recognised as supported rather than scored as
  // unsupported by construction.
  const citedOrdinals = new Set(
    [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])).filter((n) => n >= 1),
  );

  const numbered = documents.map((ranked, i) => ({ ranked, ordinal: i + 1 }));
  const selected = numbered.filter((d) => citedOrdinals.has(d.ordinal));

  for (const candidate of numbered) {
    if (selected.length >= GROUNDEDNESS_MAX_EXCERPTS) break;
    if (!citedOrdinals.has(candidate.ordinal)) selected.push(candidate);
  }

  // Ordinals must survive the filter. Renumbering the excerpts here would make
  // the judge's view of "[7]" a different passage from the answer's, and every
  // citation would look unsupported.
  const excerpts = selected
    .sort((a, b) => a.ordinal - b.ordinal)
    .map(
      ({ ranked, ordinal }) =>
        `[${ordinal}] ${String(ranked.doc.metadata?.originalText ?? ranked.doc.pageContent)}`,
    )
    .join("\n\n");

  try {
        const report = await structuredInvoke(getAuxModels("fast"), GroundednessSchema, [
      new SystemMessage(
        "You verify whether an answer is supported by source excerpts.\n" +
          "Score groundedness from 0 to 1 as the fraction of factual claims in the answer " +
          "that are directly supported by the excerpts.\n" +
          "Ignore hedging, citations and conversational framing — judge only factual claims. " +
          "An answer that correctly says the information is not available is fully grounded.",
      ),
      new HumanMessage(`EXCERPTS:\n${excerpts}\n\nANSWER:\n${answer}`),
    ], { name: "groundedness", description: "Groundedness assessment of the answer" });

    // Clamp defensively: models occasionally return a percentage rather than a fraction.
    const score = Math.max(0, Math.min(1, report.groundedness));

    return {
      score,
      verdict: verdictFor(score),
      unsupportedClaims: report.unsupportedClaims ?? [],
      passed: score >= GUARDRAIL_CONFIG.MIN_GROUNDEDNESS,
    };
  } catch (error) {
    console.error("[guardrails] groundedness check failed, passing through:", error);
    return { score: -1, verdict: "grounded", unsupportedClaims: [], passed: true };
  }
}

/**
 * Repair citation markers the model wrote in a near-miss format.
 *
 * Validation only recognises `[n]`, and anything else slips through every check
 * as though the answer cited nothing at all: no refs are found, so none can be
 * invalid, so the answer is "valid" and the malformed marker is handed to the
 * user unresolved. A measured run produced `[1†L1-L3]` — a citation style from
 * a different vendor's file-search tooling — and the pipeline reported the
 * answer as having no citations while displaying that text verbatim.
 *
 * Silence is what makes this worth fixing rather than merely prompting against.
 * A fabricated number is caught and stripped; a marker in the wrong *shape* is
 * not caught at all, and the failure is invisible precisely when the model is
 * trying hardest to attribute a claim.
 *
 * Normalising rather than deleting keeps the intent. `[1†L1-L3]` plainly means
 * passage 1, and once it is `[1]` the ordinary validator can confirm or reject
 * it on the merits.
 */
export function normaliseCitations(answer: string): string {
  return (
    answer
      // Unicode brackets, as in 【1†source】 or 【W1†L1-L3】.
      //
      // The optional letter covers web citations, which `web_search` numbers
      // [W1], [W2] to keep them in a separate namespace from document
      // passages. The first version of this matched digits only, so document
      // markers were repaired while web markers were left mangled — and they
      // are the ones a user sees most, since any question the documents cannot
      // answer is cited entirely this way.
      .replace(/【\s*([A-Za-z]?\d+)[^】]*】/g, "[$1]")
      // Several numbers in one bracket: [1, 2] and [1;2] become [1][2].
      //
      // Must precede the suffix rule below, which would otherwise match `[1, 3]`
      // first — reading the comma as the start of a suffix and discarding every
      // number after the first. Losing a citation silently is exactly the class
      // of failure this function exists to stop, so the order is load-bearing.
      .replace(/\[\s*(\d+(?:\s*[,;]\s*\d+)+)\s*\]/g, (_, group: string) =>
        group
          .split(/[,;]/)
          .map((n) => `[${n.trim()}]`)
          .join(""),
      )
      // A bracketed reference carrying a suffix: [1†L1-L3], [W2†source],
      // [1:page 4], [1 - policy].
      .replace(/\[\s*([A-Za-z]?\d+)\s*[^\]\d\s][^\]]*\]/g, "[$1]")
      // Stray whitespace inside an otherwise well-formed marker.
      .replace(/\[\s+(\d+)\s*\]|\[\s*(\d+)\s+\]/g, (_, a: string, b: string) => `[${a ?? b}]`)
  );
}

/**
 * Verify that every [n] citation in the answer points at a document that was
 * actually retrieved. Catches fabricated citation numbers, which read as
 * authoritative but reference nothing.
 */
export function validateCitations(
  answer: string,
  documents: RankedDocument[],
): { valid: boolean; invalidRefs: number[] } {
  const refs = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const invalidRefs = [...new Set(refs.filter((n) => n < 1 || n > documents.length))];
  return { valid: invalidRefs.length === 0, invalidRefs };
}

/** Strip any citation markers that don't resolve, rather than showing broken refs. */
export function stripInvalidCitations(answer: string, documents: RankedDocument[]): string {
  return answer.replace(/\[(\d+)\]/g, (match, n) => {
    const idx = Number(n);
    return idx >= 1 && idx <= documents.length ? match : "";
  });
}
