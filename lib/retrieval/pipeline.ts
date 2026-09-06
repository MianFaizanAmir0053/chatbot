import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import { RETRIEVAL_PROFILES, SUBAGENT_CONFIG, type ThinkingMode } from "../config";
import { getAuxModels } from "../models";
import { structuredInvoke } from "../structured";
import { hybridSearch } from "./hybrid";
import { rerankDocuments, type RankedDocument } from "./rerank";

const QueryPlanSchema = z.object({
  variants: z
    .array(z.string())
    .describe("Alternative phrasings of the question for retrieval"),
  keywords: z
    .array(z.string())
    .describe("Rare or technical literal terms that must match exactly"),
  hypotheticalAnswer: z
    .string()
    .describe("A short plausible answer, used as a HyDE retrieval probe"),
});

export interface RetrievalResult {
  documents: RankedDocument[];
  /** Formatted, citation-numbered context ready to hand to a model. */
  context: string;
  queriesUsed: string[];
  /** Highest reranker score seen — the primary confidence signal for CRAG. */
  topScore: number;
}

/**
 * Expand one question into several retrieval probes.
 *
 * Three techniques at once, because they fail in different ways:
 *  - multi-query: paraphrases beat vocabulary mismatch between question and doc
 *  - keyword extraction: preserves rare literal tokens that paraphrase destroys
 *  - HyDE: embeds a hypothetical *answer*, which sits closer in vector space to
 *    real answer passages than the question does
 *
 * Runs on the FAST tier — it is a high-volume call and does not need the pro model.
 */
async function planQueries(
  query: string,
  history: string,
  profile: (typeof RETRIEVAL_PROFILES)[ThinkingMode],
): Promise<string[]> {
  try {
    
    const plan = await structuredInvoke(getAuxModels("fast"), QueryPlanSchema, [
      new SystemMessage(
        "You rewrite user questions into effective document-retrieval queries.\n" +
          `Produce exactly ${profile.QUERY_VARIANTS} distinct paraphrases that a ` +
          "technical document might use, the rare literal terms worth matching exactly, " +
          "and one short hypothetical answer (1-2 sentences) as if quoting the document. " +
          "Resolve pronouns and references using the conversation history.",
      ),
      new HumanMessage(
        `${history ? `Conversation so far:\n${history}\n\n` : ""}Question: ${query}`,
      ),
    ], { name: "query_plan", description: "Retrieval queries derived from the question" });

    const queries = [
      query,
      // Smaller models ignore the requested count — one returned 45 variants,
      // which would have fanned out into 90 searches. Cap it at the source.
      ...(plan.variants ?? []).slice(0, profile.QUERY_VARIANTS),
      ...(plan.keywords?.length ? [plan.keywords.slice(0, 8).join(" ")] : []),
      ...(plan.hypotheticalAnswer ? [plan.hypotheticalAnswer] : []),
    ];

    return [...new Set(queries.map((q) => q.trim()).filter(Boolean))].slice(
      0,
      profile.MAX_QUERIES,
    );
  } catch (error) {
    console.error("[retrieval] query planning failed, using raw query:", error);
    return [query];
  }
}

/**
 * Render ranked chunks as numbered, attributed excerpts.
 *
 * Citation markers are assigned here so the generating model can reference
 * sources as [1], [2] and the UI can map them back to real documents and pages.
 * `originalText` is preferred so quotes never contain the synthetic context
 * header added at ingest time.
 */
/**
 * Render passages as numbered, citable excerpts.
 *
 * `ordinals` supplies the citation number for each passage. Pass it whenever
 * the numbers have to mean the same thing across more than one call, and let it
 * default to positional numbering only for one-shot formatting.
 *
 * The distinction matters more than it looks. Numbering each call from 1 makes the
 * second search's `[3]` a different passage from the first search's `[3]`, while
 * citation validation resolves both against the run's accumulated evidence — so
 * a marker that passes validation can still point at the wrong source, which is
 * worse than a broken one because nothing reports it. Delegated research turns
 * that from an edge case into the normal path: several subagents each retrieve
 * and cite, and their findings are concatenated into one answer.
 */
export function formatContext(docs: RankedDocument[], ordinals?: number[]): string {
  return docs
    .map((d, i) => {
      const meta = d.doc.metadata ?? {};
      const parts = [`source: ${meta.source ?? "unknown"}`];
      if (meta.section) parts.push(`section: ${meta.section}`);
      if (meta.page) parts.push(`page: ${meta.page}`);
      const body = String(meta.originalText ?? d.doc.pageContent).trim();
      return `[${ordinals?.[i] ?? i + 1}] (${parts.join(", ")}, relevance: ${d.score.toFixed(3)})\n${body}`;
    })
    .join("\n\n---\n\n");
}

/**
 * The full advanced-RAG retrieval path:
 * plan queries → hybrid dense+sparse search → RRF fusion → cross-encoder rerank.
 */
export async function retrieve(
  query: string,
  options: { history?: string; topK?: number; expand?: boolean; mode?: ThinkingMode } = {},
): Promise<RetrievalResult> {
  return withRetrievalSlot(async () => {
    const { history = "", expand = true, mode = "standard" } = options;
    const profile = RETRIEVAL_PROFILES[mode];
    const topK = options.topK ?? profile.FINAL_TOP_K;

    const queries = expand ? await planQueries(query, history, profile) : [query];
    // The profile's fusion width has to be passed: `hybridSearch` defaults to
    // the global RETRIEVAL_CONFIG value, so omitting it silently pinned every
    // mode to fifty candidates. Deep mode has been advertising a pool of eighty
    // and reranking fifty — the extra breadth it exists to buy was configured,
    // documented, and never reached the reranker.
    const fused = await hybridSearch(queries, profile.FUSION_TOP_K);
    const ranked = await rerankDocuments(query, fused, topK);

    return {
      documents: ranked,
      context: formatContext(ranked),
      queriesUsed: queries,
      topScore: ranked[0]?.score ?? 0,
    };
  });
}

/* ------------------------------------------------------------------ *
 * Concurrency control
 * ------------------------------------------------------------------ */

/**
 * Bound how many retrieval pipelines run at once.
 *
 * Model calls scale out across rotating provider keys; retrieval does not.
 * Embeddings and reranking both go to Cohere, which is the single shared
 * dependency here, so a fan-out of research branches multiplies the load on one
 * quota by the width of the fan-out. Measured with three concurrent branches
 * and no limit, every one of the eight Cohere keys was exhausted in turn and
 * the rotation spent the run failing over — making the fan-out slower than
 * running the same branches sequentially.
 *
 * A queue rather than a rate limiter because the constraint is concurrency, not
 * a schedule: work waits for a slot and starts the instant one frees, so the
 * limit costs nothing when fewer branches are running than there are slots,
 * which is every non-delegating request.
 */
let activeRetrievals = 0;
const retrievalQueue: Array<() => void> = [];

async function withRetrievalSlot<T>(run: () => Promise<T>): Promise<T> {
  if (activeRetrievals >= SUBAGENT_CONFIG.RETRIEVAL_CONCURRENCY) {
    await new Promise<void>((resolve) => retrievalQueue.push(resolve));
  }
  activeRetrievals++;
  try {
    return await run();
  } finally {
    activeRetrievals--;
    // Released in `finally` so a failed retrieval frees its slot too; leaking
    // one would shrink the pool permanently and eventually deadlock the queue.
    retrievalQueue.shift()?.();
  }
}
