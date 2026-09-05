import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import { RETRIEVAL_CONFIG } from "../config";
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
async function planQueries(query: string, history: string): Promise<string[]> {
  try {
    
    const plan = await structuredInvoke(getAuxModels("fast"), QueryPlanSchema, [
      new SystemMessage(
        "You rewrite user questions into effective document-retrieval queries.\n" +
          `Produce exactly ${RETRIEVAL_CONFIG.QUERY_VARIANTS} distinct paraphrases that a ` +
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
      ...(plan.variants ?? []).slice(0, RETRIEVAL_CONFIG.QUERY_VARIANTS),
      ...(plan.keywords?.length ? [plan.keywords.slice(0, 8).join(" ")] : []),
      ...(plan.hypotheticalAnswer ? [plan.hypotheticalAnswer] : []),
    ];

    return [...new Set(queries.map((q) => q.trim()).filter(Boolean))].slice(
      0,
      RETRIEVAL_CONFIG.MAX_QUERIES,
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
export function formatContext(docs: RankedDocument[]): string {
  return docs
    .map((d, i) => {
      const meta = d.doc.metadata ?? {};
      const parts = [`source: ${meta.source ?? "unknown"}`];
      if (meta.section) parts.push(`section: ${meta.section}`);
      if (meta.page) parts.push(`page: ${meta.page}`);
      const body = String(meta.originalText ?? d.doc.pageContent).trim();
      return `[${i + 1}] (${parts.join(", ")}, relevance: ${d.score.toFixed(3)})\n${body}`;
    })
    .join("\n\n---\n\n");
}

/**
 * The full advanced-RAG retrieval path:
 * plan queries → hybrid dense+sparse search → RRF fusion → cross-encoder rerank.
 */
export async function retrieve(
  query: string,
  options: { history?: string; topK?: number; expand?: boolean } = {},
): Promise<RetrievalResult> {
  const { history = "", topK = RETRIEVAL_CONFIG.FINAL_TOP_K, expand = true } = options;

  const queries = expand ? await planQueries(query, history) : [query];
  const fused = await hybridSearch(queries);
  const ranked = await rerankDocuments(query, fused, topK);

  return {
    documents: ranked,
    context: formatContext(ranked),
    queriesUsed: queries,
    topScore: ranked[0]?.score ?? 0,
  };
}
