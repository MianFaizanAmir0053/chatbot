/**
 * Persistence test.
 *
 * Queries the vector store WITHOUT indexing anything first. If this passes in a
 * cold process, the documents genuinely survived in Qdrant rather than living in
 * process memory — the whole reason for using a real vector database.
 *
 *   npx tsx --env-file=.env scripts/persistence-test.ts
 */
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
  console.log("\n  --- delete-by-source ---");
  const before = await store.count();
  await store.deleteBySource("motorcycle-manual.txt");
  const after = await store.count();
  console.log(`  points: ${before} -> ${after}`);
  assert(after < before, "deleteBySource removed the document's chunks");

  console.log(failures === 0 ? "\n  All persistence checks passed." : `\n  ${failures} failed.`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
