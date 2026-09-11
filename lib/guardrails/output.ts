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
  /**
   * Fraction of claims supported, or -1 for "not verified".
   *
   * -1 is not a low score. It means no judgement was made — the judge was
   * unavailable, or there was no evidence to judge against — and the interface
   * shows no badge at all rather than implying a verified result.
   */
  score: number;
  verdict: "grounded" | "partially_grounded" | "unsupported";
  unsupportedClaims: string[];
  passed: boolean;
}

/** A web source, with enough of its text for the judge to check a claim against. */
export interface WebEvidence {
  title: string;
  url: string;
  content?: string;
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
  web: WebEvidence[] = [],
): Promise<GroundednessReport> {
  // Nothing to check against, or an explicit refusal — no claim is being made.
  //
  // Reported as unverified (-1) rather than as a perfect score. The two are not
  // the same thing, and conflating them was actively misleading: a turn that
  // researched entirely on the web passed no documents here, took this branch,
  // and was labelled "Grounded in sources · 100%" without a single claim having
  // been checked. A measured run produced a confident comparison of two
  // versions of a web page when only one had ever been fetched, and wore that
  // badge while doing it.
  if (answer.trim().length < 20) {
    return { score: -1, verdict: "grounded", unsupportedClaims: [], passed: true };
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
  const documentExcerpts = selected
    .sort((a, b) => a.ordinal - b.ordinal)
    .map(
      ({ ranked, ordinal }) =>
        `[${ordinal}] ${String(ranked.doc.metadata?.originalText ?? ranked.doc.pageContent)}`,
    );

  // Web sources are judged too, in their own `[Wn]` numbering.
  //
  // They were previously invisible here: the judge only ever saw documents, so
  // a web-researched answer was scored against nothing. That is the case most
  // in need of checking, not least — a retrieved passage is at least something
  // the user uploaded, whereas a fetched page is whatever a model decided to
  // read, and a page that failed to load leaves the model free to write what it
  // expected to find.
  const webExcerpts = web
    .filter((w) => w.content?.trim())
    .slice(0, GROUNDEDNESS_MAX_EXCERPTS)
    .map((w, i) => `[W${i + 1}] ${w.title} — ${w.url}\n${w.content}`);

  const excerpts = [...documentExcerpts, ...webExcerpts].join("\n\n");

  // Everything gathered was unusable — a failed fetch, or an empty search. The
  // answer cannot be checked, and saying so is the whole point of this branch.
  if (excerpts.trim().length === 0) {
    console.warn("[guardrails] no evidence to check the answer against — reporting unverified");
    return { score: -1, verdict: "grounded", unsupportedClaims: [], passed: true };
  }

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
 * Keys belonging to this system's own auxiliary schemas.
 *
 * An object whose keys are drawn only from this set was produced by a planner
 * or a judge, not written for the user.
 */
const AUX_SCHEMA_KEYS = new Set([
  "variants",
  "keywords",
  "hypotheticalAnswer",
  "groundedness",
  "unsupportedClaims",
  "isInjection",
  "confidence",
  "rationale",
]);

/** The end index of the JSON object starting at 0, or -1 if it is not one. */
function jsonObjectEnd(text: string): number {
  if (text[0] !== "{") return -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    else if (ch === "}" && --depth === 0) return i + 1;
  }
  return -1;
}

/**
 * Remove an auxiliary call's output that was prepended to an answer.
 *
 * For a while, a query planner invoked inside a retrieval tool had its JSON
 * streamed into the answer — `{"variants":[…],"hypotheticalAnswer":"…"}` ahead
 * of the real reply, sometimes three of them. That leak is closed at the stream,
 * but the answers written while it was open are still in the conversation store,
 * so they are shown in the transcript and replayed into the model's context on
 * every later turn of those threads.
 *
 * Matched by parsing rather than by pattern, and only when every key belongs to
 * one of this system's own auxiliary schemas. An answer that legitimately opens
 * with a JSON object — a reply about configuration, say — has keys this does not
 * recognise and is left alone.
 */
export function stripLeakedAuxJson(text: string): string {
  let out = text.trimStart();
  for (;;) {
    const end = jsonObjectEnd(out);
    if (end < 0) return out === text.trimStart() ? text : out;
    let parsed: unknown;
    try {
      parsed = JSON.parse(out.slice(0, end));
    } catch {
      return out === text.trimStart() ? text : out;
    }
    const keys = Object.keys((parsed ?? {}) as Record<string, unknown>);
    if (keys.length === 0 || !keys.every((k) => AUX_SCHEMA_KEYS.has(k))) {
      return out === text.trimStart() ? text : out;
    }
    out = out.slice(end).trimStart();
  }
}

/**
 * Tool-call markup that arrived as prose instead of as a tool call.
 *
 * Not every model emits calls in the wire format its gateway parses. Several
 * families have their own template — `<tool_call>`, `<|python_tag|>`,
 * `<function=name>`, a ```tool_code fence — and a fine-tune renames it again;
 * one observed here writes `<uncensored_tool_call>search_documents<arg_key>
 * query</arg_key><arg_value>…</arg_value>`. When the gateway does not translate
 * its template, the text lands in `content` and the call never happens.
 *
 * Matched on the shape rather than on any one vendor's tag, because the set of
 * tags is open-ended and a miss here is silent: the markup is streamed to the
 * user as the answer.
 */
const TOOL_CALL_MARKUP =
  /<\|?\/?[a-z_]*tool[_-]?call\|?>|<\/?arg_(?:key|value)>|<\|python_tag\|>|<\/?function(?:_call)?[=>\s]|```tool_(?:code|call)/i;

/**
 * True when the model wrote a tool call out as text.
 *
 * Worth a check of its own because of what it implies. This is not cosmetic
 * damage to an answer — it means the model tried to call a tool, the call was
 * never executed, and whatever else it wrote was composed without the result it
 * was asking for. Treating the remaining prose as an answer would present an
 * ungrounded guess as a researched one.
 */
export function hasToolCallMarkup(text: string): boolean {
  return TOOL_CALL_MARKUP.test(text);
}

/**
 * Remove an unexecuted tool call from answer text.
 *
 * Removal rather than repair: there is nothing to recover. The call cannot be
 * run after the fact, and the arguments are only a record of what the model
 * wanted to look up. Callers must decide separately whether what survives is
 * still an answer — see the route, which falls through to the no-answer path
 * when it is not.
 */
export function stripToolCallMarkup(text: string): string {
  return (
    text
      // A complete template, opening tag through closing tag.
      .replace(/<\|?([a-z_]*tool[_-]?call)\|?>[\s\S]*?<\/\|?\1\|?>/gi, "")
      // An unterminated one, which is the usual case when generation stops
      // while the model waits for a result that will never arrive.
      .replace(/<\|?[a-z_]*tool[_-]?call\|?>[\s\S]*$/i, "")
      .replace(/<\|python_tag\|>[\s\S]*$/i, "")
      .replace(/```tool_(?:code|call)[\s\S]*?(?:```|$)/gi, "")
      // Leftover argument wrappers from a partially-matched template.
      .replace(/<\/?arg_(?:key|value)>/gi, " ")
      .replace(/<\/?function(?:_call)?[^>]*>/gi, "")
      .replace(/[ \t]{2,}/g, " ")
      .trim()
  );
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
): { valid: boolean; invalidRefs: number[]; refs: number[] } {
  const refs = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const invalidRefs = [...new Set(refs.filter((n) => n < 1 || n > documents.length))];
  // `refs` is returned as well as the verdict, because "no invalid citations"
  // and "cited its sources" are different claims and only the first is checked
  // here. An answer with no markers at all is trivially valid, and the caller
  // needs to be able to tell that apart from one that cited correctly.
  return { valid: invalidRefs.length === 0, invalidRefs, refs };
}

/**
 * Strip any citation markers that don't resolve, rather than showing broken refs.
 *
 * Removing a marker leaves the punctuation that framed it behind, and citations
 * are usually written in runs — "SAE 10W-40 grade oil [3], [4]." strips to
 * "grade oil , .", which reads as a typo the model made rather than as
 * something removed here. So the wreckage is tidied too.
 *
 * The tidying runs only when a marker was actually removed. Applied
 * unconditionally it would edit answers nothing was stripped from, and a rule
 * like "close the space before a comma" is a reasonable thing to do to damage
 * this function caused but not to prose a model deliberately wrote.
 */
export function stripInvalidCitations(answer: string, documents: RankedDocument[]): string {
  let removed = 0;
  const stripped = answer.replace(/\[(\d+)\]/g, (match, n) => {
    const idx = Number(n);
    if (idx >= 1 && idx <= documents.length) return match;
    removed++;
    return "";
  });

  if (removed === 0) return answer;

  return (
    stripped
      // The space that used to sit between the text and its marker.
      .replace(/[ \t]+([,;:.!?])/g, "$1")
      // A separator whose other side is gone: "oil , ." and "oil ,," .
      .replace(/([,;])[ \t]*(?=[.!?])/g, "")
      .replace(/([,;])(?:[ \t]*[,;])+/g, "$1")
      // A run of markers mid-sentence collapses several spaces into one.
      .replace(/[ \t]{2,}/g, " ")
      // A marker at the end of a line leaves the line trailing whitespace.
      .replace(/[ \t]+$/gm, "")
  );
}
