import { CohereRerank } from "@langchain/cohere";
import type { Document } from "@langchain/core/documents";
import { RERANK_CONFIG, RETRIEVAL_CONFIG, env, features } from "../config";
import type { FusedDocument } from "./hybrid";

let reranker: CohereRerank | null = null;

function getReranker(): CohereRerank | null {
  if (!features.cohere) return null;
  if (!reranker) {
    reranker = new CohereRerank({
      apiKey: env.COHERE_API_KEY,
      model: RERANK_CONFIG.model,
    });
  }
  return reranker;
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

  const cohere = getReranker();
  if (!cohere) return fusionFallback(candidates, topN);

  // Rerank endpoints are aggressively rate limited on trial keys, and a retry
  // is much cheaper than losing precision for the whole request.
  const MAX_ATTEMPTS = 3;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
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
      if (isRateLimit(error) && attempt < MAX_ATTEMPTS) {
        const backoff = 500 * 2 ** (attempt - 1);
        console.warn(`[rerank] rate limited, retrying in ${backoff}ms (${attempt}/${MAX_ATTEMPTS})`);
        await sleep(backoff);
        continue;
      }
      console.error("[rerank] failed, falling back to fusion order:", error);
      return fusionFallback(candidates, topN);
    }
  }

  return fusionFallback(candidates, topN);
}
