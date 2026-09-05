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
  /**
   * Override for the OpenAI-compatible endpoint.
   *
   * Set this to route the OpenAI path — fallback generation, the moderation
   * middleware and OpenAI web search — through a gateway. Left unset it talks
   * to api.openai.com. It must be honoured everywhere OPENAI_API_KEY is used,
   * or a gateway key would be sent to OpenAI itself.
   */
  OPENAI_BASE_URL: z.string().optional(),

  // --- Additional OpenAI-compatible gateways, chained for availability ---
  GROQ_API_KEY: z.string().optional(),
  GROQ_BASE_URL: z.string().default("https://api.groq.com/openai/v1"),

  BLUESMINDS_API_KEY: z.string().optional(),
  BLUESMINDS_BASE_URL: z.string().default("https://api.bluesminds.com/v1"),

  BAZAARLINK_API_KEY: z.string().optional(),
  BAZAARLINK_BASE_URL: z.string().default("https://api.bazaarlink.ai/v1"),

  MISTRAL_API_KEY: z.string().optional(),
  MISTRAL_BASE_URL: z.string().default("https://api.mistral.ai/v1"),

  OPENROUTER_API_KEY: z.string().optional(),
  OPENROUTER_BASE_URL: z.string().default("https://openrouter.ai/api/v1"),
  /** Attribution shown on OpenRouter's app leaderboard. */
  OPENROUTER_SITE_URL: z.string().default("http://localhost:3000"),
  OPENROUTER_SITE_NAME: z.string().default("Agentic RAG"),

  /**
   * Comma-separated provider order, best availability first.
   *
   * Overrides the built-in order so a provider can be promoted or demoted after
   * a quota change without editing code.
   */
  LLM_PROVIDER_ORDER: z.string().optional(),

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
  /**
   * Folder inside the bucket that this application owns.
   *
   * The bucket is shared with other apps, so every object written here is
   * scoped to one prefix. That keeps chatbot data from colliding with anything
   * else and makes a full cleanup a prefix delete rather than a filename hunt.
   */
  AWS_S3_PREFIX: z.string().default("chatbot/"),

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

/* ------------------------------------------------------------------ *
 * Provider registry
 * ------------------------------------------------------------------ */

export interface LlmProvider {
  name: string;
  apiKey: string;
  baseURL: string;
  pro: string;
  fast: string;
  /** Extra headers the gateway requires, for attribution or routing. */
  headers?: Record<string, string>;
}

/**
 * Every OpenAI-compatible provider, in default availability order.
 *
 * Free and trial keys fail in two ways that look identical to the agent — a
 * 429 burst limit and a 402 exhausted balance — and a single-provider setup
 * turns either into a dead chatbot. Declaring them as one ordered list lets the
 * agent rotate its primary across providers and fall through the rest on
 * failure, so one exhausted key degrades throughput instead of stopping work.
 *
 * Order below reflects measured behaviour: acceptance of a six-call burst
 * first, then latency. Override it with LLM_PROVIDER_ORDER.
 */
const PROVIDER_CATALOGUE: LlmProvider[] = [
  {
    name: "groq",
    apiKey: env.GROQ_API_KEY ?? "",
    baseURL: env.GROQ_BASE_URL,
    pro: process.env.GROQ_MODEL_PRO || "openai/gpt-oss-120b",
    fast: process.env.GROQ_MODEL_FAST || "openai/gpt-oss-20b",
  },
  {
    name: "bluesminds",
    apiKey: env.BLUESMINDS_API_KEY ?? "",
    baseURL: env.BLUESMINDS_BASE_URL,
    pro: process.env.BLUESMINDS_MODEL_PRO || "gpt-5.6-terra",
    fast: process.env.BLUESMINDS_MODEL_FAST || "gpt-5.6-luna",
  },
  {
    name: "openrouter",
    apiKey: env.OPENROUTER_API_KEY ?? "",
    baseURL: env.OPENROUTER_BASE_URL,
    pro: process.env.OPENROUTER_MODEL_PRO || "openai/gpt-4o",
    fast: process.env.OPENROUTER_MODEL_FAST || "openai/gpt-4o-mini",
    // OpenRouter attributes traffic to a site by these headers and shows it on
    // the app leaderboard; they are optional but cheap, and requests without
    // them are treated as anonymous.
    headers: {
      "HTTP-Referer": env.OPENROUTER_SITE_URL,
      "X-Title": env.OPENROUTER_SITE_NAME,
    },
  },
  {
    name: "mistral",
    apiKey: env.MISTRAL_API_KEY ?? "",
    baseURL: env.MISTRAL_BASE_URL,
    // The ministral line, not mistral-small: the free tier rate-limits the
    // small and magistral models on the very first call, while these answer
    // reliably and still support tool calling, which the agent loop requires.
    pro: process.env.MISTRAL_MODEL_PRO || "ministral-14b-latest",
    fast: process.env.MISTRAL_MODEL_FAST || "ministral-8b-latest",
  },
  {
    name: "bazaarlink",
    apiKey: env.BAZAARLINK_API_KEY ?? "",
    baseURL: env.BAZAARLINK_BASE_URL,
    pro: process.env.BAZAARLINK_MODEL_PRO || "deepseek-v4-pro",
    fast: process.env.BAZAARLINK_MODEL_FAST || "deepseek-v4-flash",
  },
  {
    name: "deepseek",
    apiKey: env.DEEPSEEK_API_KEY ?? "",
    baseURL: env.DEEPSEEK_BASE_URL,
    pro: MODEL_TIERS.PRO,
    fast: MODEL_TIERS.FAST,
  },
  {
    name: "openai",
    apiKey: env.OPENAI_API_KEY ?? "",
    baseURL: env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
    pro: FALLBACK_MODELS.PRO,
    fast: FALLBACK_MODELS.FAST,
  },
];

/**
 * Providers that actually have a key, in the configured order.
 *
 * Two providers pointed at the same endpoint with the same key would retry an
 * outage against itself, so duplicates are collapsed.
 */
export const LLM_PROVIDERS: LlmProvider[] = (() => {
  const preferred = env.LLM_PROVIDER_ORDER?.split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

  const usable = PROVIDER_CATALOGUE.filter((p) => p.apiKey);

  // When the order is given it is an allowlist, not just a ranking: a provider
  // is dropped by removing it from the list, without deleting its key. Appending
  // the unlisted ones instead would silently keep a known-bad gateway — one that
  // answers, but slowly enough to consume the whole request budget — as the last
  // hop, which is the case this setting exists to prevent.
  const ordered = preferred?.length
    ? preferred.flatMap((name) => usable.filter((p) => p.name === name))
    : usable;

  const seen = new Set<string>();
  return ordered.filter((p) => {
    const identity = `${p.baseURL}|${p.apiKey}`;
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  });
})();

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
  /**
   * Content moderation needs OpenAI's dedicated moderation endpoint
   * (omni-moderation-latest), which OpenAI-compatible gateways generally do not
   * serve — they answer /chat/completions and 503 everything else. Enabling it
   * against a gateway meant every model call carried a guardrail that could only
   * fail, so it is on only when talking to OpenAI proper.
   */
  moderation: Boolean(env.OPENAI_API_KEY) && !env.OPENAI_BASE_URL,
  tracing: env.LANGSMITH_TRACING === "true" && Boolean(env.LANGSMITH_API_KEY),
} as const;

/**
 * True when a provider capable of driving the agent loop is configured.
 *
 * Cohere deliberately doesn't count: it can serve auxiliary single-shot calls
 * but not `createAgent`. See ModelOptions.allowCohere in models.ts.
 */
export function hasReasoningProvider(): boolean {
  return LLM_PROVIDERS.length > 0;
}
