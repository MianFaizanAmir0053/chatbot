/**
 * The four defects a LangSmith trace exposed, pinned down.
 *
 * The trace looked healthy. A delegating turn fetched a web page, wrote a long
 * formatted comparison of two versions of that page, scored itself and finished
 * — and every guardrail reported success. Only one version had ever been
 * fetched; the first attempt had failed on a daily token limit. The "previous
 * version" column was assembled from the résumé in the document store, and the
 * answer carried a "Grounded in sources · 100%" badge while doing it.
 *
 * Nothing threw. Each check that should have caught it passed for a different
 * structural reason:
 *
 *  1. Groundedness was handed `documents` only. Web sources lived in a separate
 *     field, so a web-researched turn reached the judge with no evidence at all
 *     — read as "nothing to dispute" and scored 1.
 *  2. Citation validation looks for markers and checks each resolves, so an
 *     answer with no markers has nothing invalid and is "valid".
 *  3. The failed first attempt wrote nothing to the transcript, leaving a
 *     question with no reply; the user asked again, and the thread accumulated
 *     the same question three times.
 *  4. Answers written while the planner leaked into the stream still carry its
 *     JSON in the store, so it is replayed into context on every later turn.
 *
 * Usage: npx tsx --env-file=.env scripts/trace-audit-test.ts
 */

import {
  checkGroundedness,
  looksLikeInfrastructureExcuse,
  stripLeakedAuxJson,
  validateCitations,
} from "../lib/guardrails/output";
import type { RankedDocument } from "../lib/retrieval/rerank";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

/** The answer from the trace, shortened but the same shape: long and uncited. */
const UNCITED_ANSWER =
  "Here's a detailed comparison between the previous and current versions of the portfolio. " +
  "The previous version used a generic Full Stack Developer title, while the current version " +
  "leads with Senior Software Engineer. Project metrics moved from general stats to specific " +
  "figures such as £31K+ revenue and 95% accuracy. The tech stack is now organised into " +
  "sub-categories with emerging technology emphasised, and new About and Reach sections were " +
  "added along with an availability badge. Senior positioning improved from 6/10 to 8/10.";

async function groundednessGap() {
  banner("1. Groundedness must not score what it never saw");

  // The reported case: web-only evidence, so `documents` was empty.
  const webOnly = await checkGroundedness(UNCITED_ANSWER, [], [
    { title: "Faizan Amir", url: "https://faizan-dev.vercel.app", content: "Senior Software Engineer | React, Next.js & AI." },
  ]);
  check(
    "a web-only answer is actually judged",
    webOnly.score !== 1 || webOnly.unsupportedClaims.length > 0,
    `score ${webOnly.score}, verdict ${webOnly.verdict}`,
  );

  // No evidence of any kind. Previously this returned a perfect score.
  const nothing = await checkGroundedness(UNCITED_ANSWER, [], []);
  check(
    "with no evidence at all it reports unverified, not perfect",
    nothing.score === -1,
    `score ${nothing.score} (-1 means the badge is hidden)`,
  );

  // A web source that failed to load carries no text. An answer written anyway
  // is exactly the fabrication case, and must not be scored as supported.
  const emptyFetch = await checkGroundedness(UNCITED_ANSWER, [], [
    { title: "unreachable", url: "https://example.com", content: "" },
  ]);
  check(
    "a failed fetch counts as no evidence",
    emptyFetch.score === -1,
    `score ${emptyFetch.score}`,
  );

  // A genuine refusal still needs no badge — there is no claim to ground.
  const refusal = await checkGroundedness("Not in the documents.", [], []);
  check("a short refusal stays unverified rather than scored", refusal.score === -1);
}

function citationGap() {
  banner("2. 'No invalid citations' is not 'cited its sources'");

  const docs = [{}, {}] as unknown as RankedDocument[];

  const uncited = validateCitations(UNCITED_ANSWER, docs);
  check("an uncited answer still reports no invalid refs", uncited.valid, "as before");
  check(
    "but now reports that it cited nothing",
    uncited.refs.length === 0,
    "which is what the route warns on",
  );

  const cited = validateCitations("The capacity is 3.4 litres [1] and the grade is 10W-40 [2].", docs);
  check("a cited answer reports its refs", cited.refs.length === 2 && cited.valid, `${cited.refs.join(",")}`);

  // The researcher in the trace cited `[0]`, which resolves to no passage.
  const zero = validateCitations("The site lists three projects [0].", docs);
  check(
    "a [0] marker is caught as invalid",
    !zero.valid && zero.invalidRefs.includes("0"),
    "ordinals are 1-based",
  );

  // Web markers were previously invisible to the validator, so a fabricated one
  // passed untouched — the namespace least deserving of trust was the only one
  // taken on trust.
  const web = validateCitations("According to the site [W1], and also [W7].", docs, 3);
  check(
    "a web marker beyond what was retrieved is caught",
    !web.valid && web.invalidRefs.includes("W7") && !web.invalidRefs.includes("W1"),
    `invalid: ${web.invalidRefs.join(", ") || "none"}`,
  );
  check(
    "and a web-cited answer counts as cited",
    validateCitations("Per the site [W1].", [], 2).refs.length === 1,
    "web citations are citations",
  );
  check(
    "with no web sources retrieved, any [Wn] is invalid",
    !validateCitations("Per the site [W1].", docs, 0).valid,
    "nothing was fetched",
  );
}

/**
 * The retry middleware's failure text, exactly as it formats it.
 *
 * `Model call failed after ${n} ${n === 1 ? "attempt" : "attempts"} with ...`
 * — reproduced here so a wording change upstream shows up as a failing check
 * rather than as a branch that reports success while returning an error.
 */
function retryFailureText(attempts: number): string {
  return (
    `Model call failed after ${attempts} ${attempts === 1 ? "attempt" : "attempts"} ` +
    `with MiddlewareError: 403 Free quota exhausted.`
  );
}

/** The guard both the delegation tool and the route apply to that text. */
const EXHAUSTED = /^Model call failed after \d+ attempts?\b/i;

function retryFailureDetection() {
  banner("3. A branch that returns the provider's error instead of a finding");

  // `modelRetryMiddleware` runs with onFailure "continue", so a branch that
  // never reached a model resolves *successfully* carrying this text. Unless it
  // is recognised, the supervisor reads a provider error as research.
  check(
    "the four-attempt wording is caught",
    EXHAUSTED.test(retryFailureText(4)),
    "the case the guard was written for",
  );
  // The regression. Making an unretryable error fail immediately changed the
  // count to 1 and the noun to the singular, and a pattern matching only
  // "attempts" stopped matching — silently, since the branch still resolved.
  check(
    "and so is the single-attempt wording",
    EXHAUSTED.test(retryFailureText(1)),
    "the one that slipped through",
  );
  check("as is two", EXHAUSTED.test(retryFailureText(2)));

  // It must stay anchored: an answer that merely discusses failures is not one.
  check(
    "an answer describing a failure is not caught",
    !EXHAUSTED.test("The retry logic reports that a model call failed after 3 attempts [1]."),
    "anchored to the start",
  );
  check("nor an ordinary answer", !EXHAUSTED.test("The oil capacity is 3.4 litres [1]."));
}

function excuses() {
  banner("4. An answer that blames the system while holding the evidence");

  // Verbatim from a live delegated turn whose branches all returned ok, with
  // no provider error anywhere in the run.
  for (const excuse of [
    "I encountered rate limit issues while trying to research your questions about engine oil " +
      "capacity. Unfortunately, I cannot retrieve this information at this time.",
    "I encountered rate limits and quota issues while trying to research the documents. This " +
      "prevents me from completing the comparison.",
    "Unable to complete the research due to API rate limiting.",
  ]) {
    check(
      `flagged: ${JSON.stringify(excuse.slice(0, 46))}`,
      looksLikeInfrastructureExcuse(excuse),
    );
  }

  // The important half. An answer may legitimately be *about* rate limits —
  // that is a normal thing for a technical document to cover — and flagging it
  // would put a warning on a perfectly good answer.
  for (const fine of [
    "The API enforces a rate limit of 200 requests per day [1]. Exceeding it returns HTTP 429 [2].",
    "The documents describe a quota of 1000 calls per month for trial accounts [1].",
    "The engine oil capacity is 3.4 litres including the filter [1].",
    "The documents do not cover valve clearance intervals. I searched for the term and for " +
      "'tappet' and neither appears.",
    "Rate limiting is implemented with a token bucket, and the quota resets at midnight UTC [3].",
  ]) {
    check(
      `not flagged: ${JSON.stringify(fine.slice(0, 46))}`,
      !looksLikeInfrastructureExcuse(fine),
    );
  }
}

function leakedJson() {
  banner("5. Planner output stored inside an answer");

  // Verbatim from the trace.
  const one =
    '{"variants":["What is the URL for Faizan\'s portfolio website?","Provide the web address."],' +
    '"keywords":["Faizan","portfolio website"],"hypotheticalAnswer":"His site is example.com."}' +
    "Faizan Amir's portfolio website is **faizan-dev.vercel.app** [1].";
  check(
    "one leaked object is removed",
    stripLeakedAuxJson(one) === "Faizan Amir's portfolio website is **faizan-dev.vercel.app** [1].",
    JSON.stringify(stripLeakedAuxJson(one).slice(0, 60)),
  );

  // The same trace had three concatenated before one answer.
  const three =
    '{"hypotheticalAnswer":"a"}{"variants":["b"],"keywords":["c"],"hypotheticalAnswer":"d"}' +
    '{"variants":["e"]}Here is the audit.';
  check("several in a row are all removed", stripLeakedAuxJson(three) === "Here is the audit.", stripLeakedAuxJson(three));

  // The judge and the injection classifier have schemas too.
  check(
    "a judge's output is recognised as well",
    stripLeakedAuxJson('{"groundedness":0.9,"unsupportedClaims":[]}The answer.') === "The answer.",
  );

  banner("6. And an answer that legitimately starts with JSON is left alone");

  for (const intact of [
    '{"name":"chatbot","version":"1.0"} is the package manifest.',
    '{"threshold": 10} is the configured value [1].',
    "The capacity is 3.4 litres [1].",
    '{"variants": broken json',
    "{}",
  ]) {
    check(
      `unchanged: ${JSON.stringify(intact.slice(0, 44))}`,
      stripLeakedAuxJson(intact) === intact,
      stripLeakedAuxJson(intact) === intact ? "" : `became ${JSON.stringify(stripLeakedAuxJson(intact))}`,
    );
  }
}

async function main() {
  await groundednessGap();
  citationGap();
  retryFailureDetection();
  excuses();
  leakedJson();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
