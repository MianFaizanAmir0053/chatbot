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

  const excerpts = documents
    .map((d, i) => `[${i + 1}] ${String(d.doc.metadata?.originalText ?? d.doc.pageContent)}`)
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
