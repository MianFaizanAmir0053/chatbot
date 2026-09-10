/**
 * Documents belong to a conversation, and only to it.
 *
 * Every check here is a leak or a loss that would be invisible in use. A
 * document reachable from the wrong conversation still produces a fluent,
 * confidently-cited answer — the citation resolves, the passage is real, and
 * nothing marks it as belonging to a different chat. A fork that cannot read
 * its parent's documents reports that they do not cover the question, which is
 * indistinguishable from the documents genuinely being silent. And embeddings
 * left behind by a deleted conversation are unreachable by definition, so
 * nothing can retrieve them to reveal they are still there.
 *
 * Runs against the configured vector store with its own fixtures, and removes
 * them afterwards.
 *
 * Usage: npx tsx --env-file=.env scripts/conversation-scope-test.ts
 */

import { Document } from "@langchain/core/documents";
import { getVectorStore } from "../lib/vectorstore";
import { knowledgeBaseStatus, removeThreadDocuments } from "../lib/ingest/pipeline";
import { invalidateSparseIndex } from "../lib/retrieval/hybrid";

let failures = 0;

function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}

const ALPHA = "test-thread-alpha";
const BETA = "test-thread-beta";
const FORK = "test-thread-fork";

function chunk(source: string, threadId: string | undefined, text: string, i: number): Document {
  return new Document({
    pageContent: text,
    metadata: { source, threadId, originalText: text, chunkIndex: i },
  });
}

async function seed() {
  const store = await getVectorStore();
  await store.addDocuments([
    chunk("alpha-policy.txt", ALPHA, "The alpha reimbursement ceiling is 250 GBP.", 0),
    chunk("alpha-policy.txt", ALPHA, "Alpha claims are approved by a team lead.", 1),
    chunk("beta-policy.txt", BETA, "The beta reimbursement ceiling is 900 GBP.", 0),
    chunk("shared-legacy.txt", undefined, "Legacy corpus document with no owner.", 0),
  ]);
  invalidateSparseIndex();
  console.log("[fixture] seeded alpha, beta and one unscoped document");
}

async function cleanup() {
  const store = await getVectorStore();
  await Promise.all([
    store.deleteByThread(ALPHA),
    store.deleteByThread(BETA),
    store.deleteByThread(FORK),
  ]);
  await store.deleteBySource("shared-legacy.txt");
  invalidateSparseIndex();
}

async function isolation() {
  console.log("\n[isolation] a conversation sees only its own documents");

  const alpha = await knowledgeBaseStatus(ALPHA);
  const names = alpha.documents.map((d) => d.source).sort();

  check(
    "sees its own document",
    names.includes("alpha-policy.txt"),
    names.join(", ") || "none",
  );
  check(
    "does NOT see another conversation's document",
    !names.includes("beta-policy.txt"),
    names.includes("beta-policy.txt") ? "LEAK: beta is visible from alpha" : "beta not visible",
  );
  // Documents predating scoping have no owner and must stay reachable, or
  // shipping this feature would have silently emptied every existing corpus.
  check(
    "still sees unscoped legacy documents",
    names.includes("shared-legacy.txt"),
    "legacy corpus reachable",
  );

  const corpus = await knowledgeBaseStatus();
  const all = corpus.documents.map((d) => d.source);
  check(
    "the unscoped view still sees everything",
    all.includes("alpha-policy.txt") && all.includes("beta-policy.txt"),
    `${all.length} documents corpus-wide`,
  );
}

async function forkInheritance() {
  console.log("\n[fork] a fork reads its parent's documents");

  // The chain is what the chat route passes: the fork, then its ancestors.
  const inherited = await knowledgeBaseStatus([FORK, ALPHA]);
  const names = inherited.documents.map((d) => d.source);

  check(
    "inherits the parent's documents",
    names.includes("alpha-policy.txt"),
    names.join(", ") || "none",
  );
  check(
    "without inheriting an unrelated conversation's",
    !names.includes("beta-policy.txt"),
    names.includes("beta-policy.txt") ? "LEAK: beta visible through the fork" : "beta not visible",
  );

  // A fork on its own — before anything is uploaded to it — must not appear to
  // own the parent's documents, or deleting the fork would take them with it.
  const ownOnly = await knowledgeBaseStatus(FORK);
  check(
    "but does not own them",
    !ownOnly.documents.some((d) => d.source === "alpha-policy.txt"),
    "parent's documents are inherited, not owned",
  );
}

async function deletionPurges() {
  console.log("\n[delete] deleting a conversation removes its documents");

  const before = (await knowledgeBaseStatus()).totalChunks;
  const removed = await removeThreadDocuments(BETA);

  check("reports what it removed", removed > 0, `${removed} chunk(s)`);

  const after = await knowledgeBaseStatus();
  check(
    "the documents are gone from the corpus",
    !after.documents.some((d) => d.source === "beta-policy.txt"),
    `${before} → ${after.totalChunks} chunks`,
  );

  // The whole point of scoping the purge: a delete must not reach documents
  // another conversation is still using.
  const alpha = await knowledgeBaseStatus(ALPHA);
  check(
    "another conversation's documents survive",
    alpha.documents.some((d) => d.source === "alpha-policy.txt"),
    "alpha intact",
  );
  check(
    "and unscoped documents survive",
    after.documents.some((d) => d.source === "shared-legacy.txt"),
    "legacy corpus intact",
  );
}

async function main() {
  try {
    await cleanup();
    await seed();
    await isolation();
    await forkInheritance();
    await deletionPurges();
  } finally {
    await cleanup();
    console.log("[fixture] removed");
  }

  console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
