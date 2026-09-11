/**
 * Do the citation repairs leave an answer a person would want to read?
 *
 * Citations are the whole claim this system makes — that an answer can be
 * traced to a passage — so the code around them runs on every reply, and its
 * mistakes are read by the user directly.
 *
 * Two repairs, failing in opposite directions.
 *
 * Normalisation exists because a marker in the wrong *shape* is invisible to
 * validation: no refs are found, so none can be invalid, so a mangled `[1†L1-L3]`
 * passes every check and is shown verbatim. The rules have a load-bearing
 * order, and the tests below pin it down.
 *
 * Stripping exists because a marker pointing at nothing is worse than no marker
 * at all. But removing one leaves the punctuation that framed it, and citations
 * come in runs — so "grade oil [3], [4]." became "grade oil , .", which reads
 * as the model's own typo rather than as damage done here. The tidy-up must fix
 * that without touching answers nothing was stripped from.
 *
 * Usage: npx tsx --env-file=.env scripts/citation-repair-test.ts
 */

import { normaliseCitations, stripInvalidCitations } from "../lib/guardrails/output";
import type { RankedDocument } from "../lib/retrieval/rerank";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

function equals(name: string, got: string, want: string) {
  check(name, got === want, got === want ? "" : `got ${JSON.stringify(got)}`);
}

/** Two resolvable passages, so [1] and [2] are valid and anything above is not. */
const TWO_DOCS = [{}, {}] as unknown as RankedDocument[];

function normalising() {
  banner("1. Repairing a marker written in the wrong shape");

  equals("unicode brackets", normaliseCitations("As stated 【1†source】."), "As stated [1].");
  equals(
    "a web marker keeps its namespace",
    normaliseCitations("Per the web 【W2†L1-L3】."),
    "Per the web [W2].",
  );
  equals("a suffix is dropped", normaliseCitations("See [1†L1-L3]."), "See [1].");
  equals("a page suffix too", normaliseCitations("See [2:page 4]."), "See [2].");

  // Load-bearing order: the suffix rule would match "[1, 3]" first, read the
  // comma as the start of a suffix, and silently discard the 3.
  equals("several numbers in one bracket split", normaliseCitations("Both [1, 3] agree."), "Both [1][3] agree.");
  equals("and with a semicolon", normaliseCitations("Both [1;3] agree."), "Both [1][3] agree.");
  equals("stray space inside a marker", normaliseCitations("See [ 1 ]."), "See [1].");

  const clean = "The capacity is 3.4 litres [1][2].";
  equals("a well-formed answer is untouched", normaliseCitations(clean), clean);
}

function stripping() {
  banner("2. Removing a marker that points at nothing");

  // The case observed in a live answer.
  equals(
    "a run of unresolvable markers takes its punctuation with it",
    stripInvalidCitations("The manual recommends SAE 10W-40 grade oil [3], [4].", TWO_DOCS),
    "The manual recommends SAE 10W-40 grade oil.",
  );
  equals(
    "a single one before a full stop",
    stripInvalidCitations("Rear pressure is 250 kPa [7].", TWO_DOCS),
    "Rear pressure is 250 kPa.",
  );
  equals(
    "a valid neighbour survives",
    stripInvalidCitations("The figure is 250 kPa [1], [9].", TWO_DOCS),
    "The figure is 250 kPa [1].",
  );
  equals(
    "a run mid-sentence does not leave a double space",
    stripInvalidCitations("Both [7] and [8] agree on this.", TWO_DOCS),
    "Both and agree on this.",
  );
  equals(
    "one at the end of a line leaves no trailing space",
    stripInvalidCitations("Ends the line [8]\nNext paragraph.", TWO_DOCS),
    "Ends the line\nNext paragraph.",
  );

  banner("3. Answers nothing was stripped from are returned untouched");

  // The tidy-up is repair of this function's own damage. Running it on an
  // answer that needed no repair would edit prose the model deliberately wrote.
  for (const intact of [
    "The capacity is 3.4 litres [1][2].",
    "Both apply [1], [2] here.",
    "A sentence with an odd space before the comma , which the model wrote.",
    "Rear is 250 kPa [1]. Front is 225 kPa [2].",
  ]) {
    equals(`unchanged: ${JSON.stringify(intact.slice(0, 42))}`, stripInvalidCitations(intact, TWO_DOCS), intact);
  }
}

function main() {
  normalising();
  stripping();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
