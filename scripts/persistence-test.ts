/**
 * Persistence test.
 *
 * Queries the vector store WITHOUT indexing anything first. If this passes in a
 * cold process, the documents genuinely survived in Qdrant rather than living in
 * process memory — the whole reason for using a real vector database.
 *
 *   npx tsx --env-file=.env scripts/persistence-test.ts
 */
import { Document } from "@langchain/core/documents";
import { retrieve } from "../lib/retrieval/pipeline";
import { knowledgeBaseStatus } from "../lib/ingest/pipeline";
import { getVectorStore } from "../lib/vectorstore";

let failures = 0;
function assert(ok: boolean, msg: string) {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${msg}`);
  if (!ok) failures += 1;
}

async function main() {
  const store = await getVectorStore();
  console.log(`\n  driver: ${store.name} (persistent: ${store.persistent})`);
  assert(store.persistent, "the active driver is persistent");

  const status = await knowledgeBaseStatus();
  console.log(`  documents: ${status.documents.map((d) => `${d.source} (${d.chunks})`).join(", ") || "(none)"}`);
  assert(status.totalChunks > 0, `chunks survived from a previous process (${status.totalChunks})`);

  // No indexing in this process — retrieval must come entirely from Qdrant.
  const result = await retrieve("engine oil capacity", { topK: 3 });
  const text = result.documents
    .map((d) => String(d.doc.metadata.originalText ?? d.doc.pageContent))
    .join(" ");
  console.log(`  top score: ${result.topScore.toFixed(3)}, docs: ${result.documents.length}`);
  assert(result.documents.length > 0, "retrieval returns results in a cold process");
  assert(text.includes("3.4"), "retrieved content is correct (3.4 litres)");

  const retrievers = [...new Set(result.documents.flatMap((d) => d.retrievers))];
  console.log(`  fusion sources: ${retrievers.join(", ")}`);
  assert(retrievers.some((r) => r.startsWith("sparse")), "BM25 index rebuilt from Qdrant scroll");

  // deleteBySource must work against the payload index we created.
  //
  // Deletes a document this test creates, rather than one it hopes is present.
  // It previously deleted "motorcycle-manual.txt", which no longer exists in
  // the collection — the store holds "manual.txt" — so the call removed nothing
  // and the check failed while reporting a broken delete. Nothing was broken;
  // the assertion was aimed at a document that had been renamed out from under
  // it, and it would have gone on failing for as long as the name was wrong
  // while masking a genuine delete regression behind an already-red check.
  console.log("\n  --- delete-by-source ---");
  const fixture = `__persistence-fixture-${Date.now()}.txt`;
  await store.addDocuments([
    new Document({
      pageContent: `${fixture} > Fixture\n\nA disposable chunk written solely to be deleted.`,
      metadata: { source: fixture, chunkIndex: 0, originalText: "A disposable chunk." },
    }),
  ]);

  const before = await store.count();
  await store.deleteBySource(fixture);
  const after = await store.count();
  console.log(`  points: ${before} -> ${after}`);
  assert(after < before, "deleteBySource removed the document's chunks");

  // And it must remove only what it was asked to.
  const remaining = await store.listSources();
  assert(
    !remaining.some((s) => s.source === fixture),
    "the deleted source is gone from the index",
  );
  assert(
    remaining.some((s) => s.source === "manual.txt"),
    "unrelated documents survived the delete",
  );

  console.log(failures === 0 ? "\n  All persistence checks passed." : `\n  ${failures} failed.`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
