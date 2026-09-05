import { z } from "zod";

/**
 * Central, validated configuration.
 *
 * Everything the system needs is declared here once, parsed once, and typed.
 * Optional integrations degrade gracefully instead of throwing at import time —
 * a missing TAVILY_API_KEY should downgrade web search, not crash the server.
 */

const EnvSchema = z.object({
  // --- Reasoning models (DeepSeek V4) ---
  DEEPSEEK_API_KEY: z.string().optional(),
  DEEPSEEK_BASE_URL: z.string().default("https://api.deepseek.com"),

  // --- Fallback / auxiliary provider ---
  OPENAI_API_KEY: z.string().optional(),

  // --- Embeddings + reranking ---
  COHERE_API_KEY: z.string().optional(),

  // --- Vector database ---
  QDRANT_URL: z.string().optional(),
  QDRANT_API_KEY: z.string().optional(),
  QDRANT_COLLECTION: z.string().default("chatbot_documents"),

  // --- Object storage ---
  AWS_REGION: z.string().default("us-east-1"),
  AWS_ACCESS_KEY_ID: z.string().optional(),
  AWS_SECRET_ACCESS_KEY: z.string().optional(),
  AWS_S3_BUCKET_NAME: z.string().optional(),

  // --- Web search ---
  TAVILY_API_KEY: z.string().optional(),

  // --- Observability ---
  LANGSMITH_TRACING: z.string().optional(),
  LANGSMITH_API_KEY: z.string().optional(),
  LANGSMITH_PROJECT: z.string().default("chatbot-agentic-rag"),

  NODE_ENV: z.string().default("development"),
});

function loadEnv() {
  const parsed = EnvSchema.safeParse(process.env);
  if (!parsed.success) {
    // Never hard-crash the server on config: report and fall back to defaults so
    // /api/health can explain precisely what is misconfigured.
    console.error("[config] Invalid environment:", parsed.error.flatten().fieldErrors);
    return EnvSchema.parse({});
  }
  return parsed.data;
}

export const env = loadEnv();

/**
 * Model tiers.
 *
 * Agentic RAG fans out into many small LLM calls (grading, routing, rewriting)
 * plus a few high-stakes ones (planning, synthesis). Routing them to different
 * tiers is where most of the cost saving lives: FAST handles the 10-30x volume
 * at roughly a third of PRO's price, PRO handles the calls that decide quality.
 */
export const MODEL_TIERS = {
  /** Supervisor, planning, and final grounded synthesis. */
  PRO: process.env.DEEPSEEK_MODEL_PRO || "deepseek-v4-pro",
  /** High-volume graders, routers, rewriters, compressors. */
  FAST: process.env.DEEPSEEK_MODEL_FAST || "deepseek-v4-flash",
} as const;

/** OpenAI models used only when DeepSeek is unavailable, and for moderation. */
export const FALLBACK_MODELS = {
  PRO: process.env.OPENAI_MODEL_PRO || "gpt-5.6-terra",
  FAST: process.env.OPENAI_MODEL_FAST || "gpt-5.6-luna",
} as const;

/**
 * Cohere chat models — the last link in the provider chain.
 *
 * Cohere is already a hard requirement for embeddings and reranking, so keeping
 * a chat tier here means the system can run end to end on a single key.
 * command-a supports tool calling, which the agent loop requires.
 */
export const COHERE_MODELS = {
  PRO: process.env.COHERE_MODEL_PRO || "command-a-03-2025",
  FAST: process.env.COHERE_MODEL_FAST || "command-r7b-12-2024",
} as const;

export const EMBEDDING_CONFIG = {
  model: "embed-v4.0",
  /** embed-v4.0 supports Matryoshka output dims: 256 / 512 / 1024 / 1536. */
  dimensions: 1536,
} as const;

export const RERANK_CONFIG = {
  /** Cross-encoder rerank. v4.0-pro is the highest-accuracy tier available. */
  model: process.env.COHERE_RERANK_MODEL || "rerank-v4.0-pro",
} as const;

/**
 * Retrieval tuning.
 *
 * Defaults follow the 2026 production consensus: retrieve wide with hybrid
 * search, rerank a large candidate pool with a cross-encoder, then hand a small
 * high-precision set to the model.
 */
export const RETRIEVAL_CONFIG = {
  /** Dense (vector) candidates pulled per query variant. */
  DENSE_TOP_K: 25,
  /** Sparse (BM25) candidates pulled per query variant. */
  SPARSE_TOP_K: 25,
  /** Candidates surviving RRF fusion and handed to the reranker. */
  FUSION_TOP_K: 50,
  /** Final documents returned to the agent after reranking. */
  FINAL_TOP_K: 8,
  /**
   * Absolute reranker-score backstop. Deliberately low, because rerank models
   * do not share a score scale — see applyAdaptiveCutoff in retrieval/rerank.ts.
   */
  RERANK_SCORE_FLOOR: 0.02,
  /** Keep documents scoring at least this fraction of the best hit's score. */
  RERANK_RELATIVE_RATIO: 0.3,
  /** Query rewrites requested for multi-query fan-out. */
  QUERY_VARIANTS: 3,
  /**
   * Hard ceiling on retrieval probes per search, whatever the planner returns.
   * Each probe costs a dense *and* a sparse search, so an unbounded planner
   * response turns one question into dozens of round-trips.
   */
  MAX_QUERIES: 6,
  /** RRF smoothing constant. 60 is the value from the original RRF paper. */
  RRF_K: 60,
  CHUNK_SIZE: 900,
  CHUNK_OVERLAP: 180,
} as const;

/** Hard ceilings that bound cost and stop runaway agent loops. */
export const GUARDRAIL_CONFIG = {
  MAX_TOOL_CALLS_PER_RUN: 25,
  MAX_MODEL_CALLS_PER_RUN: 20,
  MAX_AGENT_ITERATIONS: 30,
  MAX_INPUT_CHARS: 8000,
  MAX_FILE_BYTES: 25 * 1024 * 1024,
  /** Minimum groundedness score (0-1) an answer must reach to be returned. */
  MIN_GROUNDEDNESS: 0.5,
  RATE_LIMIT_WINDOW_MS: 60_000,
  RATE_LIMIT_MAX_REQUESTS: 20,
} as const;

/** Which optional subsystems are actually usable with the current env. */
export const features = {
  deepseek: Boolean(env.DEEPSEEK_API_KEY),
  openai: Boolean(env.OPENAI_API_KEY),
  cohere: Boolean(env.COHERE_API_KEY),
  qdrant: Boolean(env.QDRANT_URL),
  s3: Boolean(env.AWS_S3_BUCKET_NAME && env.AWS_ACCESS_KEY_ID),
  tavily: Boolean(env.TAVILY_API_KEY),
  moderation: Boolean(env.OPENAI_API_KEY),
  tracing: env.LANGSMITH_TRACING === "true" && Boolean(env.LANGSMITH_API_KEY),
} as const;

/**
 * True when a provider capable of driving the agent loop is configured.
 *
 * Cohere deliberately doesn't count: it can serve auxiliary single-shot calls
 * but not `createAgent`. See ModelOptions.allowCohere in models.ts.
 */
export function hasReasoningProvider(): boolean {
  return features.deepseek || features.openai;
}
