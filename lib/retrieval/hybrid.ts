import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { Document } from "@langchain/core/documents";
import { RETRIEVAL_CONFIG } from "../config";
import { getVectorStore } from "../vectorstore";

/**
 * Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.
 *
 * Dense embeddings capture meaning but miss rare literal tokens — part numbers,
 * error codes, proper nouns. BM25 nails exact terms but is blind to paraphrase.
 * Fusing both is the single biggest retrieval-quality win available, and is why
 * this is the default rather than an option.
 */

let sparseIndex: BM25Retriever | null = null;
let sparseIndexSize = 0;

/** Called after any write to the vector store so the sparse index is rebuilt lazily. */
export function invalidateSparseIndex(): void {
  sparseIndex = null;
  sparseIndexSize = 0;
}

/**
 * BM25 is an in-memory index built from the full corpus.
 *
 * It is rebuilt only when invalidated by an ingest, not per query — building it
 * scans every chunk, which would otherwise dominate query latency.
 */
async function getSparseIndex(k: number): Promise<BM25Retriever | null> {
  if (sparseIndex) {
    sparseIndex.k = k;
    return sparseIndex;
  }
  const store = await getVectorStore();
  const docs = await store.getAllDocuments();
  if (docs.length === 0) return null;

  sparseIndex = BM25Retriever.fromDocuments(docs, { k });
  sparseIndexSize = docs.length;
  return sparseIndex;
}

export function sparseIndexStats() {
  return { built: sparseIndex !== null, documents: sparseIndexSize };
}

/** Stable identity for a chunk, used to align the same document across rankers. */
function docKey(doc: Document): string {
  const src = String(doc.metadata?.source ?? "");
  const idx = doc.metadata?.chunkIndex;
  if (idx !== undefined && idx !== null) return `${src}#${idx}`;
  return `${src}#${doc.pageContent.slice(0, 120)}`;
}

export interface FusedDocument {
  doc: Document;
  /** Combined RRF score across all ranked lists. */
  score: number;
  /** Which retrievers surfaced this document — useful for debugging recall. */
  retrievers: string[];
}

/**
 * Reciprocal Rank Fusion.
 *
 * RRF combines ranked lists without needing their scores to be comparable —
 * cosine similarity and BM25 scores live on completely different scales, so
 * naive score addition would let one ranker dominate. Rank position is the only
 * signal used: score = Σ 1 / (k + rank).
 */
export function reciprocalRankFusion(
  rankedLists: Array<{ name: string; docs: Document[] }>,
  k: number = RETRIEVAL_CONFIG.RRF_K,
): FusedDocument[] {
  const table = new Map<string, FusedDocument>();

  for (const list of rankedLists) {
    list.docs.forEach((doc, rank) => {
      const key = docKey(doc);
      const contribution = 1 / (k + rank + 1);
      const existing = table.get(key);
      if (existing) {
        existing.score += contribution;
        if (!existing.retrievers.includes(list.name)) existing.retrievers.push(list.name);
      } else {
        table.set(key, { doc, score: contribution, retrievers: [list.name] });
      }
    });
  }

  return [...table.values()].sort((a, b) => b.score - a.score);
}

/**
 * Run dense and sparse retrieval for every query variant, then fuse everything
 * into a single ranked candidate pool.
 */
export async function hybridSearch(
  queries: string[],
  limit: number = RETRIEVAL_CONFIG.FUSION_TOP_K,
): Promise<FusedDocument[]> {
  const store = await getVectorStore();
  const lists: Array<{ name: string; docs: Document[] }> = [];

  const dense = await Promise.all(
    queries.map(async (q, i) => {
      try {
        const hits = await store.similaritySearchWithScore(q, RETRIEVAL_CONFIG.DENSE_TOP_K);
        return { name: `dense:${i}`, docs: hits.map(([doc]) => doc) };
      } catch (error) {
        console.error("[hybrid] dense search failed:", error);
        return { name: `dense:${i}`, docs: [] as Document[] };
      }
    }),
  );
  lists.push(...dense);

  const bm25 = await getSparseIndex(RETRIEVAL_CONFIG.SPARSE_TOP_K);
  if (bm25) {
    const sparse = await Promise.all(
      queries.map(async (q, i) => {
        try {
          const docs = await bm25.invoke(q);
          return { name: `sparse:${i}`, docs: docs as Document[] };
        } catch (error) {
          console.error("[hybrid] sparse search failed:", error);
          return { name: `sparse:${i}`, docs: [] as Document[] };
        }
      }),
    );
    lists.push(...sparse);
  }

  return reciprocalRankFusion(lists).slice(0, limit);
}
