/**
 * Can a conversation become impossible to continue?
 *
 * It could. The client replayed the whole transcript on every turn as
 * `history`, and each entry was validated against MAX_INPUT_CHARS — a ceiling
 * meant for what a user types. The first time the agent wrote a reply longer
 * than that, the transcript carrying it failed validation on every subsequent
 * turn, and the thread was finished: 400 on every send, with no way back except
 * starting over. The assistant answering thoroughly was enough to break the
 * conversation permanently.
 *
 * These checks pin down that the transcript is no longer part of the contract,
 * that a client still sending one is tolerated rather than refused, and that
 * removing it did not loosen the limit that does matter — the message itself.
 *
 * Usage: npx tsx --env-file=.env scripts/request-shape-test.ts
 */

import { GUARDRAIL_CONFIG } from "../lib/config";
import { ChatRequestSchema } from "../lib/guardrails/input";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

const LIMIT = GUARDRAIL_CONFIG.MAX_INPUT_CHARS;
/** An answer longer than the ceiling — the thing that used to be fatal. */
const LONG_ANSWER = "The engine oil capacity is 3.4 litres. ".repeat(Math.ceil(LIMIT / 30));

function theReportedFailure() {
  banner("1. The turn that used to break the conversation");

  const asOldClientSends = {
    message: "and what about the filter?",
    threadId: "thread-1",
    mode: "agentic",
    webSearch: false,
    thinking: "standard",
    deepAgents: false,
    // Exactly what the old client built: every prior turn, verbatim, including
    // an assistant answer past the ceiling.
    history: [
      { role: "user", content: "what is the engine oil capacity" },
      { role: "assistant", content: LONG_ANSWER },
    ],
  };

  check(
    "the oversized answer is longer than the old ceiling",
    LONG_ANSWER.length > LIMIT,
    `${LONG_ANSWER.length} chars vs ${LIMIT}`,
  );

  const parsed = ChatRequestSchema.safeParse(asOldClientSends);
  check(
    "the request is accepted",
    parsed.success,
    parsed.success ? "no longer fatal" : JSON.stringify(parsed.error.flatten().fieldErrors),
  );
  check(
    "and the transcript is dropped rather than carried",
    parsed.success && !("history" in parsed.data),
    parsed.success ? Object.keys(parsed.data).join(", ") : "—",
  );
}

function theNewShape() {
  banner("2. What the client sends now");

  const current = ChatRequestSchema.safeParse({
    message: "what is the engine oil capacity",
    mode: "agentic",
    webSearch: false,
    thinking: "standard",
    deepAgents: false,
  });
  check("a turn with no transcript is valid", current.success);
  // A first message legitimately has no thread yet, and that must not be an
  // error — it is how every conversation starts.
  check(
    "and needs no threadId",
    current.success && !current.data.threadId,
    "absent means new conversation",
  );

  const resumed = ChatRequestSchema.safeParse({
    message: "and the filter?",
    threadId: "abc-123",
    mode: "agentic",
    webSearch: false,
    thinking: "standard",
    deepAgents: false,
  });
  check("a resumed turn carries only its thread id", resumed.success);
}

function stillGuarded() {
  banner("3. The limit that still matters");

  // Dropping the transcript's ceiling must not drop the real one. This is the
  // text the user typed, and the cap is a guard against prompt-stuffing.
  const overlong = ChatRequestSchema.safeParse({ message: "x".repeat(LIMIT + 1) });
  check("an over-long message is still refused", !overlong.success, `> ${LIMIT} chars`);

  const atLimit = ChatRequestSchema.safeParse({ message: "x".repeat(LIMIT) });
  check("one exactly at the limit is allowed", atLimit.success, `${LIMIT} chars`);

  const empty = ChatRequestSchema.safeParse({ message: "" });
  check("an empty message is refused", !empty.success);

  const missing = ChatRequestSchema.safeParse({ threadId: "abc" });
  check("a missing message is refused", !missing.success);
}

function main() {
  theReportedFailure();
  theNewShape();
  stillGuarded();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
