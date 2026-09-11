import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { Document } from "@langchain/core/documents";
import { RETRIEVAL_CONFIG } from "../config";
import { getVectorStore, scopeIds, type ThreadScope } from "../vectorstore";

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
async function getSparseIndex(): Promise<BM25Retriever | null> {
  if (sparseIndex) return sparseIndex;

  const store = await getVectorStore();
  const docs = await store.getAllDocuments();
  if (docs.length === 0) return null;

  sparseIndex = BM25Retriever.fromDocuments(docs, { k: SPARSE_FETCH_K });
  sparseIndexSize = docs.length;
  return sparseIndex;
}

/**
 * How many candidates the sparse index returns before scoping and trimming.
 *
 * Fixed at build time rather than set per call. The previous version assigned
 * `sparseIndex.k` on every request, which was safe while requests were serial
 * and is not now: delegated research runs several branches at once against this
 * one shared index, so one branch's assignment silently changed how many
 * results another branch got back.
 *
 * Generous because results are filtered afterwards. BM25 cannot filter by
 * conversation, so a scoped search takes the top matches corpus-wide and then
 * discards the ones belonging elsewhere — fetching exactly the number wanted
 * would leave a conversation whose documents are a small slice of the corpus
 * with almost nothing.
 */
const SPARSE_FETCH_K = 120;

export function sparseIndexStats() {
  return { built: sparseIndex !== null, documents: sparseIndexSize };
}

/**
 * Narrow documents to one conversation, keeping unscoped ones visible.
 *
 * Mirrors the rule the vector store enforces natively: a conversation sees its
 * own documents plus any that predate scoping. Applied here because BM25 has no
 * filter of its own — which is also why the sparse fetch is deliberately wide,
 * so narrowing does not starve the fused pool.
 */
function scopeDocs(docs: Document[], threadId?: ThreadScope): Document[] {
  const ids = scopeIds(threadId);
  if (ids.length === 0) return docs;
  return docs.filter((d) => {
    const owner = d.metadata?.threadId;
    return !owner || ids.includes(String(owner));
  });
}

/** Stable identity for a chunk, used to align the same document across rankers. */
/**
 * Failures that are the network's, not the query's.
 *
 * A connect timeout, a reset, a DNS hiccup, a gateway error — none of these say
 * anything about whether the search would succeed a moment later. A rejected
 * filter or a missing collection does, and retrying that only doubles the wait
 * before the same failure.
 */
function isTransientSearchFailure(error: unknown): boolean {
  const e = error as { code?: string; message?: string; status?: number; cause?: unknown };
  const code = String(e?.code ?? (e?.cause as { code?: string })?.code ?? "");
  if (/TIMEOUT|ECONNRESET|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|UND_ERR/i.test(code)) return true;
  if (typeof e?.status === "number" && e.status >= 500) return true;
  return /fetch failed|timeout|socket hang up|network/i.test(String(e?.message ?? ""));
}

/**
 * One dense search, retried once when the failure was the network's.
 *
 * Worth the extra attempt because of how quietly the alternative fails. The
 * catch below returns an empty result, so a single dropped connection removes
 * dense retrieval from the fusion entirely and the answer is built from BM25
 * alone — with no error, no warning, and an answer that still looks
 * well-formed. That was observed here as a 10s connect timeout to Qdrant while
 * the cluster was reachable before and after.
 *
 * Only once, and only for transient failures: the caller is a user waiting on a
 * search, and a second attempt at something that cannot work just adds the
 * whole timeout again.
 */
async function denseSearch(
  store: Awaited<ReturnType<typeof getVectorStore>>,
  query: string,
  threadId?: ThreadScope,
): Promise<[Document, number][]> {
  try {
    return await store.similaritySearchWithScore(query, RETRIEVAL_CONFIG.DENSE_TOP_K, threadId);
  } catch (error) {
    if (!isTransientSearchFailure(error)) throw error;
    console.warn("[hybrid] dense search hit a transient failure, retrying once");
    return store.similaritySearchWithScore(query, RETRIEVAL_CONFIG.DENSE_TOP_K, threadId);
  }
}

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
  threadId?: ThreadScope,
): Promise<FusedDocument[]> {
  const store = await getVectorStore();

  // The two retrievers are independent, so they run together rather than one
  // after the other. They were sequential, which meant every search paid the
  // dense path — an embedding call and a Qdrant round trip — before the sparse
  // path started, and on a first search also the corpus load that builds the
  // BM25 index. Nothing in either needs the other's result.
  const [dense, sparse] = await Promise.all([
    Promise.all(
      queries.map(async (q, i) => {
        try {
          const hits = await denseSearch(store, q, threadId);
          return { name: `dense:${i}`, docs: hits.map(([doc]) => doc) };
        } catch (error) {
          // Degrading to BM25 alone keeps the search alive, but it is a real
          // loss of quality — dense retrieval is what survives a vocabulary
          // mismatch between the question and the document — and nothing
          // downstream can tell it happened. Said plainly for that reason.
          console.error(
            `[hybrid] dense search failed twice; answering from BM25 alone for this query:`,
            error,
          );
          return { name: `dense:${i}`, docs: [] as Document[] };
        }
      }),
    ),
    (async () => {
      const bm25 = await getSparseIndex();
      if (!bm25) return [];
      return Promise.all(
        queries.map(async (q, i) => {
          try {
            const docs = await bm25.invoke(q);
            // The sparse index spans the whole corpus and is rebuilt only on
            // ingest, so scoping is applied to its results rather than to the
            // index. Keeping one shared index is deliberate: a per-conversation
            // index would be rebuilt on every switch and would hold the corpus
            // once per open conversation.
            //
            // Trimmed after scoping, not before, or the trim would spend its
            // budget on documents this conversation cannot see.
            const scoped = scopeDocs(docs as Document[], threadId).slice(
              0,
              RETRIEVAL_CONFIG.SPARSE_TOP_K,
            );
            return { name: `sparse:${i}`, docs: scoped };
          } catch (error) {
            console.error("[hybrid] sparse search failed:", error);
            return { name: `sparse:${i}`, docs: [] as Document[] };
          }
        }),
      );
    })(),
  ]);

  return reciprocalRankFusion([...dense, ...sparse]).slice(0, limit);
}
