/**
 * Does conversation storage actually hold a conversation?
 *
 * Exercises the whole contract against whichever driver is configured, so the
 * same checks prove MongoDB and the JSON-file fallback behave identically. That
 * matters because the fallback engages silently when a database is unreachable,
 * which is exactly when nobody is watching.
 *
 * Two properties get particular attention. Appends must be atomic — the file
 * driver reads, mutates and writes, so concurrent turns could lose one, and the
 * point of moving to a database is that they cannot. And a fork must copy the
 * transcript without sharing message objects, or appending to one thread would
 * silently rewrite history in another.
 *
 * Usage: npx tsx --env-file=.env scripts/conversation-store-test.ts
 */

import {
  appendMessages,
  conversationStoreInfo,
  deleteConversation,
  forkConversation,
  getConversation,
  listConversations,
  renameConversation,
  scopeChainFor,
} from "../lib/conversations/store";

let failures = 0;
const created: string[] = [];

function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}

function turn(role: "user" | "assistant", content: string) {
  return { role, content, createdAt: new Date().toISOString() };
}

const ID = `test-store-${Date.now().toString(36)}`;

async function roundTrip() {
  console.log("\n[round trip] a conversation survives a write and a read");

  await appendMessages(ID, [turn("user", "What is the receipt itemisation threshold?")]);
  created.push(ID);

  const afterQuestion = await getConversation(ID);
  check("created on the first turn", afterQuestion !== null, afterQuestion ? "stored" : "missing");
  // The question is recorded before the answer exists, because a run that is
  // abandoned still has to leave the conversation findable in the list.
  check(
    "titled from the opening question",
    Boolean(afterQuestion?.title?.includes("receipt itemisation")),
    afterQuestion?.title ?? "no title",
  );

  await appendMessages(ID, [turn("assistant", "It is 10 GBP.")]);
  const afterAnswer = await getConversation(ID);
  check(
    "appends rather than replaces",
    afterAnswer?.messages.length === 2,
    `${afterAnswer?.messages.length ?? 0} messages`,
  );

  const listed = await listConversations(50);
  const summary = listed.find((c) => c.id === ID);
  check("appears in the list", Boolean(summary), `${listed.length} conversations listed`);
  check(
    "the list reports the message count without the transcript",
    summary?.messages === 2,
    `${summary?.messages ?? 0}`,
  );
  check(
    "and derives a status from the last turn",
    summary?.status === "completed",
    summary?.status ?? "none",
  );
}

async function concurrentAppends() {
  console.log("\n[atomicity] concurrent turns do not lose each other");

  const id = `${ID}-race`;
  created.push(id);
  await appendMessages(id, [turn("user", "seed")]);

  // Ten appends at once. Read-mutate-write loses some of these; an atomic push
  // keeps all of them. This is the specific failure the database driver exists
  // to remove, so it is worth asserting rather than assuming.
  await Promise.all(
    Array.from({ length: 10 }, (_, i) => appendMessages(id, [turn("assistant", `reply ${i}`)])),
  );

  const stored = await getConversation(id);
  check(
    "every concurrent append survived",
    stored?.messages.length === 11,
    `${stored?.messages.length ?? 0}/11 messages`,
  );
}

async function renaming() {
  console.log("\n[rename] a deliberate name is not overwritten");

  const renamed = await renameConversation(ID, "  Expense policy questions  ");
  check("renames and trims", renamed?.title === "Expense policy questions", renamed?.title ?? "—");

  // The auto-title rule fires for conversations still called "New conversation".
  // A locked title must survive a later turn, or a rename would silently revert.
  await appendMessages(ID, [turn("user", "And the meal cap?")]);
  const after = await getConversation(ID);
  check(
    "survives a later turn",
    after?.title === "Expense policy questions",
    after?.title ?? "—",
  );

  const empty = await renameConversation(ID, "   ");
  check("refuses an empty name", empty === null, "blank rejected");
}

async function forking() {
  console.log("\n[fork] a copy that shares no state with its origin");

  const source = await getConversation(ID);
  const fork = await forkConversation(ID, { upTo: 2 });
  if (!fork) {
    check("forked", false, "fork returned null");
    return;
  }
  created.push(fork.id);

  check("truncates to the requested length", fork.messages.length === 2, `${fork.messages.length}`);
  check("records its parent", fork.forkedFrom === ID, fork.forkedFrom ?? "none");
  check(
    "names itself after the original",
    fork.title.includes("Expense policy questions"),
    fork.title,
  );

  // The chain is what retrieval uses to let a fork read its parent's documents.
  const chain = await scopeChainFor(fork.id);
  check(
    "the scope chain reaches the parent",
    chain[0] === fork.id && chain.includes(ID),
    chain.join(" -> "),
  );

  // Deep copy, not a shared reference: appending to the fork must not touch the
  // original's transcript.
  await appendMessages(fork.id, [turn("user", "only in the fork")]);
  const original = await getConversation(ID);
  check(
    "appending to the fork leaves the original alone",
    original?.messages.length === source?.messages.length,
    `original has ${original?.messages.length ?? 0}, had ${source?.messages.length ?? 0}`,
  );

  const summaries = await listConversations(50);
  check(
    "the parent reports its fork count",
    (summaries.find((c) => c.id === ID)?.forks ?? 0) >= 1,
    `${summaries.find((c) => c.id === ID)?.forks ?? 0} fork(s)`,
  );
}

async function deleting() {
  console.log("\n[delete] removal is real");

  const removed = await deleteConversation(ID);
  check("reports the removal", removed, "deleted");

  const gone = await getConversation(ID);
  check("the conversation is gone", gone === null, gone ? "still present" : "absent");

  const listed = await listConversations(50);
  check("and it leaves the list", !listed.some((c) => c.id === ID), `${listed.length} remaining`);

  // Deleting a parent must not delete the fork's own record — only its access to
  // the parent's documents, which is a scoping consequence, not a cascade.
  const forkStillThere = listed.some((c) => c.forkedFrom === ID);
  check(
    "a fork of it survives as a conversation",
    forkStillThere,
    forkStillThere ? "fork intact" : "fork vanished with the parent",
  );
}

async function main() {
  const info = await conversationStoreInfo();
  console.log(`store: ${info.driver}${info.durable ? " (durable)" : " (NOT durable)"}`);

  try {
    await roundTrip();
    await concurrentAppends();
    await renaming();
    await forking();
    await deleting();
  } finally {
    for (const id of created) {
      try {
        await deleteConversation(id);
      } catch {
        /* best effort */
      }
    }
    console.log("[fixture] removed");
  }

  console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
