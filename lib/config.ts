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

  SILICONFLOW_API_KEY: z.string().optional(),
  SILICONFLOW_BASE_URL: z.string().default("https://api.siliconflow.com/v1"),

  REQUESTY_API_KEY: z.string().optional(),
  REQUESTY_BASE_URL: z.string().default("https://router.requesty.ai/v1"),

  LLM7_API_KEY: z.string().optional(),
  LLM7_BASE_URL: z.string().default("https://api.llm7.io/v1"),

  AIONLABS_API_KEY: z.string().optional(),
  AIONLABS_BASE_URL: z.string().default("https://api.aionlabs.ai/v1"),

  VENICE_API_KEY: z.string().optional(),
  VENICE_BASE_URL: z.string().default("https://api.venice.ai/api/v1"),

  POLLINATIONS_API_KEY: z.string().optional(),
  POLLINATIONS_BASE_URL: z.string().default("https://text.pollinations.ai/openai"),

  HUGGINGFACE_API_KEY: z.string().optional(),
  HUGGINGFACE_BASE_URL: z.string().default("https://router.huggingface.co/v1"),

  TOGETHER_API_KEY: z.string().optional(),
  TOGETHER_BASE_URL: z.string().default("https://api.together.ai/v1"),

  GEMINI_API_KEY: z.string().optional(),
  /**
   * Google's OpenAI-compatible surface, not the native generateContent API.
   *
   * The native endpoint takes an X-goog-api-key header and a `contents` body
   * that this registry cannot speak; this path accepts a Bearer token and the
   * standard chat-completions shape, so Gemini needs no special-casing.
   */
  GEMINI_BASE_URL: z.string().default("https://generativelanguage.googleapis.com/v1beta/openai"),

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

  /**
   * Providers a research subagent is allowed to run on.
   *
   * A narrower allowlist than LLM_PROVIDER_ORDER, and narrower on purpose. The
   * tail of the main chain exists to keep a single request alive when
   * everything better is rate-limited, and a slow-but-alive gateway is the right
   * answer there. It is the wrong answer for a fan-out: several subagents run
   * concurrently and the supervisor cannot synthesise until the slowest returns,
   * so one degraded provider sets the latency of the whole turn, and a provider
   * that answers without calling tools strands its branch entirely.
   *
   * Defaults to the providers measured as sustaining tool calls under load.
   */
  SUBAGENT_PROVIDER_ORDER: z.string().optional(),

  /**
   * Embeddings and reranking. Accepts a comma-separated list of keys.
   *
   * Cohere is the one hard single point of failure here: it is the only source
   * of both embeddings and reranking, so exhausting its quota does not degrade
   * retrieval, it stops it. Trial keys meter rerank especially tightly. Listing
   * several lets a rate-limited call move to the next key instead of dropping
   * the request to fusion order.
   */
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
    name: "gemini",
    apiKey: env.GEMINI_API_KEY ?? "",
    baseURL: env.GEMINI_BASE_URL,
    // The flash line, not gemini-pro-latest: pro returns 429 on this key's
    // quota. Note that flash spends its budget on thinking tokens before any
    // visible text, so a small max_tokens comes back empty rather than short.
    pro: process.env.GEMINI_MODEL_PRO || "gemini-flash-latest",
    fast: process.env.GEMINI_MODEL_FAST || "gemini-flash-lite-latest",
  },
  {
    name: "aionlabs",
    apiKey: env.AIONLABS_API_KEY ?? "",
    baseURL: env.AIONLABS_BASE_URL,
    // aion-3.0 is the only model from the 2026-09-06 provider trial that
    // completed a real agent turn. aion-3.0-mini returned tool calls in one
    // trial and answered without them in another, so it is not used.
    pro: process.env.AIONLABS_MODEL_PRO || "aion-labs/aion-3.0",
    fast: process.env.AIONLABS_MODEL_FAST || "aion-labs/aion-3.0",
  },
  {
    name: "siliconflow",
    apiKey: env.SILICONFLOW_API_KEY ?? "",
    baseURL: env.SILICONFLOW_BASE_URL,
    // Answered three trial calls with working tool calls and then returned 402
    // for the rest — the balance covered exactly those calls. Left out of the
    // chain until the account is funded; the models themselves are capable.
    pro: process.env.SILICONFLOW_MODEL_PRO || "deepseek-ai/DeepSeek-V4-Pro",
    fast: process.env.SILICONFLOW_MODEL_FAST || "zai-org/GLM-5.3-Flash",
  },
  {
    name: "requesty",
    apiKey: env.REQUESTY_API_KEY ?? "",
    baseURL: env.REQUESTY_BASE_URL,
    // Single tool calls succeed and sustain load, but a full agent turn fails
    // with 400 "enable tool_config.include_server_side_tool_invocations" —
    // a Requesty-side requirement this client does not send.
    pro: process.env.REQUESTY_MODEL_PRO || "google/gemma-4-31b-it",
    fast: process.env.REQUESTY_MODEL_FAST || "google/gemma-4-31b-it",
  },
  {
    name: "llm7",
    apiKey: env.LLM7_API_KEY ?? "",
    baseURL: env.LLM7_BASE_URL,
    // Its Claude and GPT models 402; codestral answers but only 2 of 5 calls
    // under load, the rest 429 "model is temporarily busy".
    pro: process.env.LLM7_MODEL_PRO || "codestral-latest",
    fast: process.env.LLM7_MODEL_FAST || "codestral-latest",
  },
  {
    name: "venice",
    apiKey: env.VENICE_API_KEY ?? "",
    baseURL: env.VENICE_BASE_URL,
    pro: process.env.VENICE_MODEL_PRO || "zai-org-glm-5-2",
    fast: process.env.VENICE_MODEL_FAST || "z-ai-glm-5-3-flash",
  },
  {
    name: "pollinations",
    apiKey: env.POLLINATIONS_API_KEY ?? "",
    baseURL: env.POLLINATIONS_BASE_URL,
    pro: process.env.POLLINATIONS_MODEL_PRO || "openai-fast",
    fast: process.env.POLLINATIONS_MODEL_FAST || "openai-fast",
  },
  {
    name: "huggingface",
    apiKey: env.HUGGINGFACE_API_KEY ?? "",
    baseURL: env.HUGGINGFACE_BASE_URL,
    // Tool support on the router is per model, not per account: the 8B Llama
    // in HF's own example rejects a tools array outright with
    // INVALID_REQUEST_BODY "model features", which would strand the agent loop.
    // These three accept tools.
    pro: process.env.HUGGINGFACE_MODEL_PRO || "deepseek-ai/DeepSeek-V3-0324",
    fast: process.env.HUGGINGFACE_MODEL_FAST || "meta-llama/Llama-3.3-70B-Instruct",
  },
  {
    name: "together",
    apiKey: env.TOGETHER_API_KEY ?? "",
    baseURL: env.TOGETHER_BASE_URL,
    // gpt-oss is the same family groq serves, so a working Together account
    // would be a like-for-like substitute rather than a quality step down.
    pro: process.env.TOGETHER_MODEL_PRO || "openai/gpt-oss-120b",
    fast: process.env.TOGETHER_MODEL_FAST || "openai/gpt-oss-20b",
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

  // A provider's key field may hold several comma-separated keys. Each becomes
  // its own entry against the same endpoint, so a 429 on one key falls to
  // another key at the same provider before leaving for a slower one — the
  // cheapest possible failover, and the one that matters most on the primary,
  // where free-tier per-key limits are what actually bite.
  const usable = PROVIDER_CATALOGUE.flatMap((p) =>
    splitKeys(p.apiKey).map((apiKey) => ({ ...p, apiKey })),
  );

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

/**
 * Providers judged reliable enough to run a delegated research branch.
 *
 * These are the gateways that passed the whole assessment ladder — a real tool
 * call, then five sustained under load, then a complete agent turn — not merely
 * the ones that returned a completion once. The distinction is what this list
 * is for: a gateway that answers three calls and then returns 402, or that
 * accepts a tools array and replies in prose without using it, looks healthy to
 * a connectivity check and silently produces an empty branch here.
 *
 * Falls back to the head of the main chain when nothing is configured and
 * nothing matches, so a renamed or removed provider degrades to "the best
 * available" rather than to an empty list, which would leave subagents with no
 * model at all.
 */
export const SUBAGENT_PROVIDERS: LlmProvider[] = (() => {
  const requested = (env.SUBAGENT_PROVIDER_ORDER ?? "groq,gemini,aionlabs")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

  const matched = requested.flatMap((name) => LLM_PROVIDERS.filter((p) => p.name === name));
  return matched.length > 0 ? matched : LLM_PROVIDERS.slice(0, 8);
})();

/**
 * Every configured Cohere key, in order.
 *
 * Callers rotate through these on a rate limit. Duplicates are dropped so a key
 * pasted twice does not make the retry budget look larger than it is.
 */
/**
 * Split a comma-separated credential value into individual keys.
 *
 * Quotes are stripped per key rather than around the whole value. A single
 * quoted key is unwrapped by the env loader, but the moment a second is
 * appended the value stops being one quoted string and the quotes survive — so
 * the first key silently becomes `"abc` and every call with it returns 401
 * while the rest quietly carry the load.
 */
export function splitKeys(raw: string | undefined): string[] {
  return [
    ...new Set(
      (raw ?? "")
        .split(",")
        .map((k) => k.trim().replace(/^["']|["']$/g, "").trim())
        .filter(Boolean),
    ),
  ];
}

export const COHERE_API_KEYS: string[] = splitKeys(env.COHERE_API_KEY);

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
/**
 * Retrieval depth per thinking mode.
 *
 * One `search_documents` call is not one request: it plans queries with a model,
 * then runs a dense and a sparse probe per query variant, then reranks the pool.
 * So the variant count multiplies embedding calls and the rerank payload, and is
 * the single biggest lever on both latency and spend.
 *
 * Standard is deliberately tighter than the previous fixed settings — most
 * questions are answered by the first two probes, and the extra four bought
 * little beyond latency. Deep spends that budget and more, on purpose: it is
 * for the questions where breadth of evidence is the point.
 */
export const RETRIEVAL_PROFILES = {
  standard: {
    QUERY_VARIANTS: 2,
    MAX_QUERIES: 3,
    FINAL_TOP_K: 6,
    FUSION_TOP_K: 40,
  },
  deep: {
    QUERY_VARIANTS: 5,
    MAX_QUERIES: 9,
    FINAL_TOP_K: 14,
    FUSION_TOP_K: 80,
  },
  /**
   * The profile a research subagent retrieves with.
   *
   * Query expansion is skipped entirely for this profile — `search_documents`
   * passes `expand: false` — so `QUERY_VARIANTS` and `MAX_QUERIES` are inert
   * here and kept only so the profile shape stays uniform. Expansion exists so
   * that a lone agent can attack a question from several angles; a supervisor
   * that has decomposed the question has already done that at a higher level,
   * and each branch re-deriving its own angles repeats the work — onto Cohere,
   * the one dependency a fan-out cannot spread across keys.
   *
   * The top-k figures do apply, and are a little richer than `standard`: a
   * branch has one job and a clean context, so it can afford to read a couple
   * more passages than a supervisor juggling the whole question.
   */
  focused: {
    QUERY_VARIANTS: 2,
    MAX_QUERIES: 3,
    FINAL_TOP_K: 8,
    FUSION_TOP_K: 50,
  },
} as const;

export type ThinkingMode = keyof typeof RETRIEVAL_PROFILES;

export const GUARDRAIL_CONFIG = {
  /* --- What actually bounds cost --- */
  MAX_TOOL_CALLS_PER_RUN: 25,
  /**
   * The real spend ceiling. Enforced by middleware with exitBehavior "end", so
   * hitting it returns the best answer so far instead of throwing it away.
   */
  MAX_MODEL_CALLS_PER_RUN: 20,

  /**
   * LangGraph recursion limit — graph steps, not model calls, and the two are
   * nowhere near the same number.
   *
   * Every middleware contributes its own node per hook, so one model turn costs
   * roughly a dozen steps before any tool runs. At 30 the budget was exhausted
   * after three or four turns, and a question needing several lookups died with
   * "Recursion limit reached" and no answer at all — while the model-call limit
   * that is supposed to bound cost had barely been touched.
   *
   * Raising this does not raise spend: MAX_MODEL_CALLS_PER_RUN still caps the
   * calls and ends the run gracefully. This only stops the step counter from
   * pre-empting that limit and turning a completable question into an error.
   */
  MAX_AGENT_ITERATIONS: 250,
  MAX_INPUT_CHARS: 8000,
  MAX_FILE_BYTES: 25 * 1024 * 1024,
  /** Minimum groundedness score (0-1) an answer must reach to be returned. */
  MIN_GROUNDEDNESS: 0.5,
  RATE_LIMIT_WINDOW_MS: 60_000,
  RATE_LIMIT_MAX_REQUESTS: 20,
} as const;

/**
 * Budgets for delegated research (Deep Agents mode).
 *
 * The obvious design — a ceiling on the supervisor plus a ceiling inside each
 * subagent — does not work, and fails in the worst possible way. The limit
 * middlewares keep their tallies in agent *state*, and a subagent inherits the
 * parent's state, so a branch begins counting from whatever the supervisor has
 * already spent. Measured with a per-branch limit of six, every delegation
 * returned "run level call limit reached with 6 model calls" before its
 * researcher made a single call of its own, and the supervisor concluded the
 * documents did not contain the answer. Nothing errored.
 *
 * So the budgets here are per *turn*, enforced once on the supervisor and
 * counting the branches too through that same shared state. A branch is bounded
 * separately by `BRANCH_RECURSION_LIMIT`, which is passed in the config of its
 * own invocation and therefore cannot be inherited.
 *
 * Worst case for a delegating turn is TURN_MODEL_CALLS model calls and
 * TURN_TOOL_CALLS tool calls across the supervisor and every researcher, with
 * at most MAX_BRANCHES_PER_ROUND * MAX_DELEGATION_ROUNDS branches ever started.
 */
export const SUBAGENT_CONFIG = {
  /**
   * Researchers one `delegate_research` call may start, running concurrently.
   *
   * Also the fan-out width the supervisor is asked to aim for, and the cap the
   * tool enforces on the array it receives — a model that ignores the schema
   * and sends twenty sub-questions gets the first six researched rather than
   * twenty subagents.
   */
  MAX_BRANCHES_PER_ROUND: 6,

  /**
   * How many times one turn may call `delegate_research`.
   *
   * Separate from the width because the two bound different things and
   * conflating them understates the total badly: one number serving as both a
   * per-call cap and a per-turn cap reads like a limit of six branches while
   * actually permitting thirty-six.
   *
   * Two, not three, and the reason is behavioural rather than arithmetic. Once
   * verification became automatic the opening round no longer needed a round of
   * its own to follow it, and a supervisor given three rounds used all of them:
   * a measured turn delegated four researchers, then two, then one, then hit
   * the cap and finished with an empty answer after 344 seconds. Each round
   * chased a smaller gap than the last, and none of them was worth the turn's
   * remaining budget. One round of research plus one to close a named gap is
   * the shape the workflow actually needs; the prompt says so too, and both
   * saying it is deliberate.
   */
  MAX_DELEGATION_ROUNDS: 2,

  /**
   * Model calls for a whole delegating turn — supervisor and branches together.
   *
   * One shared budget rather than a supervisor ceiling plus a per-branch
   * ceiling, because the counters are shared whether or not that is wanted.
   * `modelCallLimitMiddleware` keeps its tally in agent *state*, and a subagent
   * inherits the parent's state, so a branch begins counting from whatever the
   * supervisor has already spent.
   *
   * Giving branches their own limiter therefore does not bound them — it
   * measures the supervisor's usage against the branch's allowance and refuses
   * the delegation outright. Measured: with a per-branch limit of 6, every
   * single `task` call returned "run level call limit reached with 6 model
   * calls" before the researcher made even one call of its own, and the
   * supervisor reported that it could not find the documents. The feature
   * looked like it was running and produced nothing.
   *
   * The same sharing is what makes a single budget sufficient: the supervisor's
   * limiter sees the branches' calls too, so this is a real ceiling on the
   * whole turn rather than on the supervisor's share of it.
   */
  TURN_MODEL_CALLS: 45,

  /**
   * Tool calls for a whole delegating turn, for the same shared-counter reason.
   *
   * Has to be well above the non-delegating ceiling: every researcher's
   * searches are counted here too, so six branches running three searches each
   * is eighteen calls before the supervisor's own delegations are added. The
   * ordinary limit of 25 would stop a fan-out partway and leave the supervisor
   * synthesising from branches that never finished.
   */
  TURN_TOOL_CALLS: 60,

  /**
   * Graph steps one research branch may take.
   *
   * The only ceiling that actually binds a branch. It is passed in the config
   * of the branch's own `invoke`, so unlike the limit middlewares — whose
   * counters live in state a subagent inherits from its parent — it is scoped
   * to that single call and cannot be pre-consumed by the supervisor.
   *
   * Sized against the straggler, not the average. Per-branch timing showed the
   * fan-out itself was healthy — three branches started together and two
   * finished in 13.8s and 16.3s — while the third took 85.6s and set the whole
   * round's duration. It was researching a sub-question the documents do not
   * answer, and the instruction to try "genuinely different phrasings" before
   * reporting an absence has no natural stopping point: it searched until the
   * budget stopped it.
   *
   * A branch is one narrow sub-question, and roughly four searches settle it
   * either way. Capping it converts a straggler into a prompt, honest "the
   * documents do not address this" — which is the same answer the long version
   * reached, several minutes sooner.
   *
   * Forty-eight rather than thirty-six: at the tighter figure a branch hit the
   * ceiling mid-search after eighty-one seconds, and back then that cost the
   * whole branch. Exhaustion is now recoverable — the branch streams, so
   * whatever it had found is returned as a partial finding — which makes the
   * ceiling a safety net rather than a cliff, and makes a little more headroom
   * cheap. It still bounds a genuinely stuck branch well inside the turn's
   * shared model-call budget.
   */
  BRANCH_RECURSION_LIMIT: 48,

  /**
   * Full retrieval pipelines allowed to run at once, across all branches.
   *
   * Model calls parallelise across eight rotating provider keys; retrieval does
   * not. Embeddings and reranking both go to Cohere, which is the one hard
   * shared dependency in this system, and a fan-out multiplies the load on it by
   * the width of the fan-out — each branch running query planning, a dense
   * probe per variant and a rerank over the fused pool. Left unbounded, three
   * concurrent researchers exhausted all eight Cohere keys in turn and the
   * rotation spent its time failing over: the fan-out became *slower* than
   * running the branches one at a time, which is the opposite of the point.
   *
   * Six, and the number moved because the cost of a search did. This limit was
   * first set to three while a branch search still ran query expansion — three
   * embedding calls and a rerank apiece, so three concurrent searches meant
   * nine embeddings in flight. Branches no longer expand, so a search is one
   * embedding and one rerank, and six concurrent searches place *less* load on
   * Cohere than the old three did. Leaving it at three would have throttled the
   * fan-out against a cost that no longer exists — half of a six-branch round
   * queueing for no reason.
   *
   * The headroom is measured rather than assumed: a full delegating turn with
   * eight branches drew only two Cohere rate-limit responses, against eight
   * rotating keys.
   */
  RETRIEVAL_CONCURRENCY: 6,

  /**
   * Run an adversarial check automatically once per delegating turn, on the
   * first substantial round of findings.
   *
   * Verification was a step in the supervisor's workflow — "delegate the
   * load-bearing claims to `verifier`" — and it never happened. A measured turn
   * started thirteen researchers and every one of them was a
   * `document-researcher`: the verifier and the web researcher were configured,
   * bound and dead. That is the same failure as the supervisor that would not
   * delegate, in a new place. An instruction the model may skip is not a
   * guarantee, and the more optional a step looks the more reliably it is
   * dropped when the answer already seems complete.
   *
   * So the check moved out of the prompt and into the tool, where it is not a
   * decision. It costs one extra branch per turn, and unlike the researchers it
   * cannot run concurrently with them — it has to read what they found — so it
   * is deliberately once per turn rather than once per round.
   */
  AUTO_VERIFY: true,

  /**
   * Findings a round needs before it is worth attacking.
   *
   * A single finding is usually a gap-filling lookup rather than a claim the
   * answer rests on, and verifying it spends a branch to little effect.
   */
  AUTO_VERIFY_MIN_FINDINGS: 2,

  /**
   * Graph steps the automatic verifier may take — tighter than a researcher's.
   *
   * Verification is the one branch that cannot overlap with anything: it reads
   * what the researchers found, so it runs after them and its cost lands
   * directly on the turn. Measured at the full branch limit it added roughly
   * fifty seconds to a twenty-second fan-out, spending six searches.
   *
   * It does not need many. A researcher is answering an open question and has
   * to find the right vocabulary; the verifier already has the findings and
   * their wording, and is looking for a specific, narrow class of thing —
   * exceptions, thresholds, effective dates, contradictions. If the documents
   * hold no caveat, more searching was never going to find one.
   *
   * Measured at 30 it still took 72.2 seconds against a fan-out whose branches
   * took 14 and 16 — close to half the turn, spent entirely after the
   * researchers had finished. Twenty steps is roughly three targeted searches
   * and a report, which is what the job actually is.
   */
  VERIFY_RECURSION_LIMIT: 20,

  /** Recommended fan-out width, stated in the prompt so delegations batch. */
  TARGET_PARALLEL_WIDTH: 4,
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
