/**
 * Does an unexecuted tool call reach the user as an answer?
 *
 * Some models write their tool calls in a template their gateway does not
 * translate. Two things then go wrong, and neither announces itself:
 *
 *  - the template arrives in the message content and is streamed to the user as
 *    though it were the answer, and
 *  - the call never ran, so anything written around it was composed without the
 *    evidence it was reaching for.
 *
 * The second is the dangerous half. Stripping the markup and showing what
 * remains would turn a visible failure into an invisible one: a confident,
 * ungrounded reply. So the stripping is paired with a flag, and the route falls
 * through to the no-answer path when nothing survives.
 *
 * The repair half is tested for its boundaries rather than its successes. A
 * matcher that is too eager routes a call to the wrong tool, and the model is
 * never told — it gets an answer from a tool it did not ask for.
 *
 * Usage: npx tsx --env-file=.env scripts/tool-call-leak-test.ts
 */

import { repairToolName } from "../lib/agents/middleware";
import { hasToolCallMarkup, stripToolCallMarkup } from "../lib/guardrails/output";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

/** The exact text observed in a reply, before any of this existed. */
const OBSERVED =
  "<uncensored_tool_call>search_documents<arg_key>query</arg_key>" +
  "<arg_value>tyre pressure rear</arg_value>";

function detection() {
  banner("1. Recognising a call that arrived as prose");

  check("the observed leak is detected", hasToolCallMarkup(OBSERVED), "the reported case");
  check(
    "a standard tool_call template too",
    hasToolCallMarkup("<tool_call>{\"name\": \"search_documents\"}</tool_call>"),
  );
  check("a python_tag call", hasToolCallMarkup('<|python_tag|>search_documents(query="oil")'));
  check("a function tag", hasToolCallMarkup("<function=search_documents>{}</function>"));
  check("a tool_code fence", hasToolCallMarkup("```tool_code\nsearch_documents(q)\n```"));

  // The false positives that would matter: ordinary answers that happen to
  // discuss tools, code or markup. Retracting one of these would delete a
  // correct answer in front of the user.
  check(
    "an ordinary answer is untouched",
    !hasToolCallMarkup("The engine oil capacity is 3.4 litres including the filter [1]."),
  );
  check(
    "so is prose about tools",
    !hasToolCallMarkup("I searched the documents using the tool call described in section 4."),
    "the word alone is not the markup",
  );
  check(
    "so is an HTML example in an answer",
    !hasToolCallMarkup("The manual shows <b>250 kPa</b> in the table on page 12."),
  );
  check(
    "so is a normal code fence",
    !hasToolCallMarkup("```python\nprint('hello')\n```"),
  );
}

function stripping() {
  banner("2. Removing it, and knowing nothing is left");

  check("the observed leak strips to nothing", stripToolCallMarkup(OBSERVED) === "", "empty");
  check(
    "a closed template strips to nothing",
    stripToolCallMarkup("<tool_call>search_documents</tool_call>") === "",
  );

  // Generation usually stops at the call, so the tag is left open. Anything
  // after it is the model's unanswered wait, not an answer.
  const trailing = "Let me look that up. <uncensored_tool_call>search_documents<arg_key>q";
  check(
    "an unterminated template takes the rest with it",
    stripToolCallMarkup(trailing) === "Let me look that up.",
    JSON.stringify(stripToolCallMarkup(trailing)),
  );

  // The whole point of the flag: what survives may look like an answer while
  // having been written without the evidence it asked for.
  const survives = stripToolCallMarkup(`The rear tyre is 250 kPa [1]. ${OBSERVED}`);
  check(
    "prose before the call survives the strip",
    survives === "The rear tyre is 250 kPa [1].",
    JSON.stringify(survives),
  );
  check(
    "but the turn is still flagged as unexecuted",
    hasToolCallMarkup(`The rear tyre is 250 kPa [1]. ${OBSERVED}`),
    "detection is independent of what survives",
  );

  const clean = "The engine oil capacity is 3.4 litres [1].";
  check("a clean answer is returned unchanged", stripToolCallMarkup(clean) === clean);
}

function repair() {
  banner("3. Repairing a name, and refusing to guess");

  const tools = ["search_documents", "list_documents", "web_search", "fetch_url", "write_todos"];

  check(
    "the observed mangled name resolves",
    repairToolName("uncensored_tool_call>list_documents", tools) === "list_documents",
    "the reported case",
  );
  check(
    "so does a namespaced one",
    repairToolName("functions.search_documents", tools) === "search_documents",
  );
  check(
    "and a tag-wrapped one",
    repairToolName("<tool_call>web_search", tools) === "web_search",
  );

  check("an already-correct name is left alone", repairToolName("web_search", tools) === null);
  check(
    "an unknown tool is not invented",
    repairToolName("delete_everything", tools) === null,
    "no nearest-match guessing",
  );

  // The dangerous cases. A name that merely contains a tool name, or that
  // extends one, must not be rerouted: the model would receive a result from a
  // tool it never asked for and would have no way to notice.
  check(
    "a name that only contains a tool name is refused",
    repairToolName("search_documents_v2", tools) === null,
    "must end at the real name",
  );
  check(
    "a prefix is not a boundary",
    repairToolName("xweb_search", tools) === null,
    "word characters do not delimit",
  );
  check(
    "the longest match wins over a shorter suffix",
    repairToolName("call>search_documents", [...tools, "documents"]) === "search_documents",
    "not `documents`",
  );
  check(
    "regex metacharacters in a tool name are literal",
    repairToolName("wrap>a.b", ["a.b"]) === "a.b" && repairToolName("wrap>axb", ["a.b"]) === null,
    "escaped, not matched as a pattern",
  );
}

function main() {
  detection();
  stripping();
  repair();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
