import { CohereRerank } from "@langchain/cohere";
import type { Document } from "@langchain/core/documents";
import { COHERE_API_KEYS, RERANK_CONFIG, RETRIEVAL_CONFIG } from "../config";
import type { FusedDocument } from "./hybrid";

let rerankers: CohereRerank[] | null = null;

/**
 * One reranker per configured key.
 *
 * Rerank is the most aggressively metered Cohere endpoint on a trial key, and
 * losing it costs precision for the whole request — the pipeline falls back to
 * fusion order. Holding every key lets a 429 move to the next one instead.
 */
function getRerankers(): CohereRerank[] {
  if (!rerankers) {
    rerankers = COHERE_API_KEYS.map(
      (apiKey) => new CohereRerank({ apiKey, model: RERANK_CONFIG.model }),
    );
  }
  return rerankers;
}

export interface RankedDocument {
  doc: Document;
  /** Cross-encoder relevance in [0,1], or the fused RRF score when rerank is unavailable. */
  score: number;
  retrievers: string[];
  /** False when reranking was skipped or failed and fusion order was used instead. */
  reranked: boolean;
}

function fusionFallback(candidates: FusedDocument[], topN: number): RankedDocument[] {
  return candidates.slice(0, topN).map((c) => ({
    doc: c.doc,
    score: c.score,
    retrievers: c.retrievers,
    reranked: false,
  }));
}

const isRateLimit = (error: unknown): boolean => {
  const message = error instanceof Error ? error.message : String(error);
  return /429|rate.?limit|too many requests/i.test(message);
};

/**
 * True for failures another key could survive.
 *
 * Wider than isRateLimit deliberately: a revoked or exhausted key reports 401,
 * 403 or a quota message rather than 429, and treating only 429 as recoverable
 * would drop the whole request to fusion order while a perfectly good second
 * key sat unused.
 */
const isKeyProblem = (error: unknown): boolean => {
  const message = error instanceof Error ? error.message : String(error);
  return isRateLimit(error) || /401|403|quota|invalid api key|unauthor/i.test(message);
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Select which reranked results to keep.
 *
 * A fixed score cutoff is unreliable: rerank models do not share a score scale.
 * Measured on the same query and documents, rerank-v4.0-pro scores an irrelevant
 * passage ~0.28 where rerank-v3.5 scores it ~0.03 — one absolute threshold
 * cannot be correct for both.
 *
 * So the cut is relative to the best hit, with an absolute floor as a backstop,
 * and the single best document is always kept. Returning nothing when plausible
 * candidates existed is the worse failure: the agent reports "not found" and
 * stops, rather than reasoning over a weak-but-real passage.
 */
function applyAdaptiveCutoff(
  ranked: RankedDocument[],
  topN: number,
): RankedDocument[] {
  if (ranked.length === 0) return [];

  const best = ranked[0].score;
  const cutoff = Math.max(
    RETRIEVAL_CONFIG.RERANK_SCORE_FLOOR,
    best * RETRIEVAL_CONFIG.RERANK_RELATIVE_RATIO,
  );

  const kept = ranked.filter((r) => r.score >= cutoff);
  return (kept.length > 0 ? kept : ranked.slice(0, 1)).slice(0, topN);
}

/**
 * Cross-encoder reranking of the fused candidate pool.
 *
 * Bi-encoder retrieval scores query and document independently, so it can only
 * approximate relevance. A cross-encoder reads the pair together and is far more
 * accurate — but too slow to run over a whole corpus. Retrieving ~50 candidates
 * cheaply and reranking those is the standard production shape, and it is what
 * turns decent recall into high precision.
 *
 * Failure is non-fatal: we degrade to fusion order rather than lose the answer.
 */
export async function rerankDocuments(
  query: string,
  candidates: FusedDocument[],
  topN: number = RETRIEVAL_CONFIG.FINAL_TOP_K,
): Promise<RankedDocument[]> {
  if (candidates.length === 0) return [];

  const clients = getRerankers();
  if (clients.length === 0) return fusionFallback(candidates, topN);

  // Rerank endpoints are aggressively rate limited on trial keys, and both a
  // retry and a different key are much cheaper than losing precision for the
  // whole request. Keys are tried first — moving on costs nothing, where a
  // backoff costs latency — and the backoff only applies once they are all
  // rate limited, which means the quota is genuinely gone rather than one key
  // being briefly hot.
  const MAX_ROUNDS = 3;
  for (let round = 1; round <= MAX_ROUNDS; round += 1) {
    for (const [index, cohere] of clients.entries()) {
      try {
        const results = await cohere.rerank(
          candidates.map((c) => c.doc.pageContent),
          query,
          { model: RERANK_CONFIG.model, topN: Math.min(topN * 2, candidates.length) },
        );

        const ranked: RankedDocument[] = results.map((r) => ({
          doc: candidates[r.index].doc,
          score: r.relevanceScore,
          retrievers: candidates[r.index].retrievers,
          reranked: true,
        }));

        return applyAdaptiveCutoff(ranked, topN);
      } catch (error) {
        if (!isKeyProblem(error)) {
          console.error("[rerank] failed, falling back to fusion order:", error);
          return fusionFallback(candidates, topN);
        }
        console.warn(
          `[rerank] key ${index + 1}/${clients.length} unusable (round ${round}/${MAX_ROUNDS})`,
        );
      }
    }

    if (round < MAX_ROUNDS) {
      const backoff = 500 * 2 ** (round - 1);
      console.warn(`[rerank] every key rate limited, retrying in ${backoff}ms`);
      await sleep(backoff);
    }
  }

  console.error("[rerank] all keys exhausted, falling back to fusion order");
  return fusionFallback(candidates, topN);
}
