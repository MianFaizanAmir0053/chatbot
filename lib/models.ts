import { ChatDeepSeek } from "@langchain/deepseek";
import { ChatOpenAI } from "@langchain/openai";
import { ChatCohere, CohereEmbeddings } from "@langchain/cohere";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Embeddings } from "@langchain/core/embeddings";
import { AlibabaEmbeddings } from "./embeddings/alibaba";
import {
  COHERE_API_KEYS,
  COHERE_MODELS,
  EMBEDDING_CONFIG,
  FALLBACK_MODELS,
  LLM_PROVIDERS,
  MODEL_TIERS,
  SUBAGENT_PROVIDERS,
  splitKeys,
  type LlmProvider,
  env,
  features,
} from "./config";

import { credentialId, isRefused, noteFailure, usable } from "./credential-health";

/**
 * Endpoint identity for Cohere credentials.
 *
 * The health registry keys on endpoint plus key, and the Cohere client does not
 * expose a base URL, so this names the one it uses.
 */
const COHERE_ENDPOINT = "https://api.cohere.com";

export type ModelTier = "pro" | "fast";

/**
 * Which credential built a given model.
 *
 * A model object does not carry its key anywhere a caller can read it, and the
 * error it throws does not either — so without this, a refusal cannot be
 * attributed to the credential that earned it and the rotation keeps handing
 * that credential out. Weak so a model is still collected normally.
 */
const credentials = new WeakMap<BaseChatModel, string>();

function stamp(model: BaseChatModel, baseURL: string, apiKey: string): BaseChatModel {
  credentials.set(model, credentialId(baseURL, apiKey));
  return model;
}

/**
 * Report a failed call so a refused credential leaves the rotation.
 *
 * Called from wherever a chain walks candidates itself, which is the one place
 * a failure can still be tied to the model that caused it. Errors that might
 * succeed later are ignored — see `isPermanentRefusal`.
 */
export function noteModelFailure(model: BaseChatModel, error: unknown): void {
  const id = credentials.get(model);
  if (id) noteFailure(id, error);
}

export interface ModelOptions {
  temperature?: number;
  maxTokens?: number;
  /**
   * Permit Cohere as a last-resort provider.
   *
   * Off by default because Cohere cannot drive the agent loop: `createAgent`
   * emits structured content blocks, and `@langchain/cohere` rejects them with
   * "ChatCohere does not support non text message content" — reproducibly, even
   * with no middleware attached. It also lacks `tool_choice` and has a broken
   * `withStructuredOutput`.
   *
   * It is perfectly good for the auxiliary single-shot calls (query planning,
   * guardrail judges), which use plain messages, so those opt in explicitly.
   */
  allowCohere?: boolean;
}

function cohereModel(tier: ModelTier, temperature: number): BaseChatModel {
  return new ChatCohere({
    model: tier === "pro" ? COHERE_MODELS.PRO : COHERE_MODELS.FAST,
    // First key only: a comma-separated list is not a credential.
    apiKey: COHERE_API_KEYS[0],
    temperature,
  }) as unknown as BaseChatModel;
}

function openAIModel(tier: ModelTier, temperature: number, maxTokens?: number): BaseChatModel {
  return new ChatOpenAI({
    model: tier === "pro" ? FALLBACK_MODELS.PRO : FALLBACK_MODELS.FAST,
    apiKey: env.OPENAI_API_KEY,
    temperature,
    maxTokens,
    ...(env.OPENAI_BASE_URL ? { configuration: { baseURL: env.OPENAI_BASE_URL } } : {}),
  }) as unknown as BaseChatModel;
}

/**
 * Build a chat model for a tier.
 *
 * DeepSeek V4 is the primary provider; OpenAI is the drop-in fallback. Tier
 * semantics are identical across providers, so callers never branch on which
 * one is active.
 */
/**
 * Round-robin cursor over the leading provider's keys.
 *
 * Deliberately rotates keys and not models. Free tiers meter per key, so
 * spreading calls across eight keys multiplies the daily ceiling eightfold and
 * leaves each key's short-term rate limit largely untouched. Because the
 * endpoint and model never change, every request is served by exactly the same
 * model — rotation costs nothing in answer quality.
 *
 * Rotating across *models* was tried and reverted: models differ in whether
 * they can drive the agent loop at all, so it made quality a coin flip per
 * request. Model choice stays fixed and ordered by measured competence; only
 * the credential moves.
 */
let keyCursor = 0;
let lastPrimaryKey = "";

/**
 * Every key configured against the endpoint serving the primary model.
 *
 * Refused keys are dropped. This is the pool that matters most: it feeds the
 * agent's own model, and a rotation that keeps dealing out refused credentials
 * puts one at the front of most requests — where each costs a full retry
 * sequence before any fallback is reached.
 */
function primaryKeyPool(): string[] {
  const pool = LLM_PROVIDERS.filter((p) => p.baseURL === env.DEEPSEEK_BASE_URL).map(
    (p) => p.apiKey,
  );
  const configured = pool.length > 0 ? pool : splitKeys(env.DEEPSEEK_API_KEY);
  return usable(configured, (key) => credentialId(env.DEEPSEEK_BASE_URL, key));
}

/** How many primary keys are configured, refused or not. Reported by /api/health. */
export function primaryKeysConfigured(): number {
  const pool = LLM_PROVIDERS.filter((p) => p.baseURL === env.DEEPSEEK_BASE_URL);
  return pool.length > 0 ? pool.length : splitKeys(env.DEEPSEEK_API_KEY).length;
}

/**
 * The key for this call, advancing the rotation.
 *
 * Advances per model call rather than per request, so the several calls one
 * agent turn makes are themselves spread across keys instead of concentrating
 * a whole run's burst on one.
 */
function nextPrimaryKey(): string {
  const pool = primaryKeyPool();
  if (pool.length === 0) return env.DEEPSEEK_API_KEY ?? "";
  lastPrimaryKey = pool[keyCursor++ % pool.length];
  return lastPrimaryKey;
}

/** How many keys currently back the primary provider. Reported by /api/health. */
export function primaryKeyCount(): number {
  return primaryKeyPool().length;
}

/**
 * Extra body fields for a primary-provider call, or nothing.
 *
 * Spread rather than passed as an always-present `modelKwargs`, so an endpoint
 * that has never heard of `enable_thinking` receives no such field at all. Only
 * the FAST tier is touched — see `DEEPSEEK_FAST_THINKING` for the measurement
 * behind that, including the grader that returned empty content.
 */
function primaryKwargs(tier: ModelTier): { modelKwargs?: Record<string, unknown> } {
  if (tier !== "fast" || env.DEEPSEEK_FAST_THINKING === undefined) return {};
  return { modelKwargs: { enable_thinking: env.DEEPSEEK_FAST_THINKING === "true" } };
}

export function getModel(tier: ModelTier = "pro", options: ModelOptions = {}): BaseChatModel {
  const temperature = options.temperature ?? (tier === "fast" ? 0 : 0.1);

  if (features.deepseek) {
    const apiKey = nextPrimaryKey();
    const model = new ChatDeepSeek({
      model: tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST,
      apiKey,
      temperature,
      maxTokens: options.maxTokens,
      ...primaryKwargs(tier),
      configuration: { baseURL: env.DEEPSEEK_BASE_URL },
    }) as unknown as BaseChatModel;
    return stamp(model, env.DEEPSEEK_BASE_URL, apiKey);
  }

  if (features.openai) {
    return openAIModel(tier, temperature, options.maxTokens);
  }

  if (options.allowCohere && features.cohere) {
    return cohereModel(tier, temperature);
  }

  throw new Error(
    "No agent-capable reasoning provider configured. Set DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY. " +
      "COHERE_API_KEY alone cannot drive the agent loop.",
  );
}

/** Model for auxiliary single-shot calls, where Cohere is a viable last resort. */
export function getAuxModel(tier: ModelTier = "fast", options: ModelOptions = {}): BaseChatModel {
  return getModel(tier, { ...options, allowCohere: true });
}

/* ------------------------------------------------------------------ *
 * Delegated research (Deep Agents)
 * ------------------------------------------------------------------ */

/**
 * Rotation cursor for subagent credentials, separate from the primary one.
 *
 * Separate because the two pools differ in size and in what advancing them
 * means. Sharing a cursor across pools of different lengths makes the stride
 * uneven and can hand the same key to two subagents in the same fan-out, which
 * is the one thing this rotation exists to prevent: concurrent branches must
 * land on different credentials or they collide on a single key's per-minute
 * limit and the fan-out is slower than running them one at a time.
 */
let subagentCursor = 0;

/**
 * A model for one research subagent, advancing the rotation.
 *
 * Called once per subagent construction rather than once per run, so a fan-out
 * of four draws four different keys and the branches genuinely run in parallel
 * instead of queueing behind one key's rate limit.
 */
/**
 * The subagent pool, minus refused credentials.
 *
 * Matters more here than elsewhere: a fan-out draws one credential per branch,
 * so refused entries left in the pool land on specific branches and strand
 * them. The supervisor then reports a gap in the evidence that was really an
 * account problem.
 */
function liveSubagentProviders(): LlmProvider[] {
  return usable(SUBAGENT_PROVIDERS, (p) => credentialId(p.baseURL, p.apiKey));
}

export function getSubagentModel(tier: ModelTier = "pro", temperature = 0.1): BaseChatModel {
  const pool = liveSubagentProviders();
  if (pool.length === 0) return getModel(tier, { temperature });

  const provider = pool[subagentCursor++ % pool.length];
  return providerModel(provider, tier, temperature);
}

/**
 * Fallback chain for a subagent, drawn only from the reliable pool.
 *
 * Deliberately not `getFallbackModels`: that chain ends at the gateways kept
 * around to keep a single request alive when everything better is exhausted,
 * and inheriting it here would let one branch fail over onto a provider that
 * cannot sustain a research loop — stranding that branch while the supervisor
 * waits for it and reports a gap that was an infrastructure failure, not an
 * absence in the documents.
 */
export function getSubagentFallbackModels(tier: ModelTier = "pro"): BaseChatModel[] {
  const pool = liveSubagentProviders();
  if (pool.length <= 1) return [];

  // Start after the entry the rotation just handed out, so the first fallback
  // is a different credential rather than the one that has just failed.
  const start = subagentCursor % pool.length;
  return pool
    .map((_, i) => pool[(start + i) % pool.length])
    .slice(1)
    .map((p) => providerModel(p, tier, 0));
}

/** Distinct provider names backing delegated research. Reported by /api/health. */
export function subagentProviderNames(): string[] {
  return [...new Set(SUBAGENT_PROVIDERS.map((p) => p.name))];
}

/** How many credentials the subagent rotation can draw on. */
export function subagentKeyCount(): number {
  return SUBAGENT_PROVIDERS.length;
}

/**
 * Every provider that can serve auxiliary calls, best first.
 *
 * Auxiliary work (query planning, guardrail judges) runs on whichever provider
 * answers, because a configured-but-dead key is otherwise worse than no key at
 * all: it wins provider selection and then fails every call, silently disabling
 * the guardrails behind it. Callers walk this list until one succeeds.
 */
export function getAuxModels(tier: ModelTier = "fast", temperature = 0): BaseChatModel[] {
  const chain: BaseChatModel[] = [];

  const primaryId = credentialId(env.DEEPSEEK_BASE_URL, env.DEEPSEEK_API_KEY ?? "");
  if (features.deepseek && !isRefused(primaryId)) {
    chain.push(
      stamp(
        new ChatDeepSeek({
          model: tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST,
          apiKey: env.DEEPSEEK_API_KEY,
          temperature,
          ...primaryKwargs(tier),
          configuration: { baseURL: env.DEEPSEEK_BASE_URL },
        }) as unknown as BaseChatModel,
        env.DEEPSEEK_BASE_URL,
        env.DEEPSEEK_API_KEY ?? "",
      ),
    );
  }
  // Every remaining gateway, so a rate-limited primary does not silently
  // disable the guardrail or planner that sits behind it. Refused credentials
  // are dropped: this chain is walked in order on every auxiliary call, so each
  // dead entry ahead of a live one is a failed round trip paid every time.
  const primary = credentialId(env.DEEPSEEK_BASE_URL, env.DEEPSEEK_API_KEY ?? "");
  const gateways = usable(
    LLM_PROVIDERS.filter((p) => credentialId(p.baseURL, p.apiKey) !== primary),
    (p) => credentialId(p.baseURL, p.apiKey),
  );
  for (const p of gateways) {
    chain.push(providerModel(p, tier, temperature));
  }

  if (features.cohere) chain.push(cohereModel(tier, temperature));

  return chain;
}

/**
 * Instantiate one registry entry. Every provider speaks the OpenAI wire format.
 *
 * Exported so a connectivity check can exercise the same construction the agent
 * uses — a check that rebuilt the client itself could pass while the real base
 * URL or headers were wrong.
 */
export function providerModel(
  provider: LlmProvider,
  tier: ModelTier,
  temperature: number,
): BaseChatModel {
  const model = new ChatOpenAI({
    model: tier === "pro" ? provider.pro : provider.fast,
    apiKey: provider.apiKey,
    temperature,
    configuration: {
      baseURL: provider.baseURL,
      ...(provider.headers ? { defaultHeaders: provider.headers } : {}),
    },
  }) as unknown as BaseChatModel;
  return stamp(model, provider.baseURL, provider.apiKey);
}

/**
 * Cross-provider fallback chain handed to `modelFallbackMiddleware`.
 *
 * Tried in order only when the primary call fails. Free and trial keys fail in
 * two ways that look identical to the agent — a 429 burst limit and a 402
 * exhausted balance — and with a single provider either one ends the run: the
 * retry middleware exhausts its attempts and returns a non-message object,
 * which surfaces to the user as an empty answer. Chaining every configured
 * gateway means an exhausted key costs one retry instead of the request.
 *
 * The provider already serving as primary is skipped: retrying an outage
 * against the same endpoint just spends the budget twice. Cohere is
 * deliberately excluded — see ModelOptions.allowCohere.
 */
export function getFallbackModels(tier: ModelTier = "pro"): BaseChatModel[] {
  // Exclude the key the rotation just handed out, not a fixed one. With keys
  // rotating, a hardcoded exclusion would drop a healthy key from the chain
  // while leaving in the very one that is about to fail.
  const inUse = credentialId(env.DEEPSEEK_BASE_URL, lastPrimaryKey || (env.DEEPSEEK_API_KEY ?? ""));

  // Refused credentials are dropped for the same reason they are dropped from
  // the rotation: this chain is walked in order after a failure, so a dead entry
  // ahead of a live one costs the request a full retry sequence before the
  // fallback that could have answered is ever reached.
  return usable(
    LLM_PROVIDERS.filter((p) => credentialId(p.baseURL, p.apiKey) !== inUse),
    (p) => credentialId(p.baseURL, p.apiKey),
  ).map((p) => providerModel(p, tier, 0));
}

/**
 * Name of the provider actually serving the primary model.
 *
 * The primary is configured through the DEEPSEEK_* slot, but that slot can
 * point at any OpenAI-compatible gateway, so reporting a literal "deepseek"
 * mislabels the dashboard whenever it does. Resolve it against the registry and
 * fall back to the endpoint host, which is true by construction.
 */
export function activeProviderName(): string {
  const primary = `${env.DEEPSEEK_BASE_URL}|${env.DEEPSEEK_API_KEY ?? ""}`;
  const match = LLM_PROVIDERS.find((p) => `${p.baseURL}|${p.apiKey}` === primary);
  if (match) return match.name;
  try {
    return new URL(env.DEEPSEEK_BASE_URL).host.replace(/^api\./, "");
  } catch {
    return "deepseek";
  }
}

/** Model string identifying the active provider, for logs and telemetry. */
export function activeModelId(tier: ModelTier = "pro"): string {
  if (features.deepseek) {
    return `${activeProviderName()}:${tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST}`;
  }
  if (features.openai) {
    return `openai:${tier === "pro" ? FALLBACK_MODELS.PRO : FALLBACK_MODELS.FAST}`;
  }
  if (features.cohere) {
    return `cohere:${tier === "pro" ? COHERE_MODELS.PRO : COHERE_MODELS.FAST} (auxiliary only)`;
  }
  return "none";
}

/**
 * True for the failures another key could survive.
 *
 * A rate limit or an exhausted/invalid key is worth retrying elsewhere; a
 * malformed request is not, and would only waste every remaining key before
 * reporting the same error.
 */
function isKeyExhausted(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /429|rate.?limit|too many requests|401|403|quota|invalid api key/i.test(message);
}

/**
 * Embeddings across every configured Cohere key.
 *
 * Presented as one object because the vector stores construct their embedder
 * once and hold it for the process; swapping keys has to happen inside that
 * object rather than by handing callers a different one. Rotation starts from
 * the last key that worked, so a exhausted key is skipped on subsequent calls
 * instead of being retried first every time.
 */
class RotatingCohereEmbeddings extends Embeddings {
  private readonly clients: CohereEmbeddings[];
  private cursor = 0;

  /**
   * Query embeddings already computed, keyed by the exact text.
   *
   * Cohere is the one dependency this system cannot spread across keys —
   * embeddings and reranking both go to it, and it is what rate-limits a
   * fan-out. Query text repeats constantly: researchers on neighbouring
   * sub-questions reach for the same obvious phrasing, a second round re-asks
   * what the first already asked, and a follow-up question in the same
   * conversation repeats the one before it. Every one of those was a fresh
   * embedding call against the binding constraint.
   *
   * Safe because the model is fixed and embedding is deterministic: the same
   * text always yields the same vector, so a hit is indistinguishable from a
   * call except in latency and quota.
   *
   * Only queries are cached. Document embedding happens once per chunk at
   * ingest and never repeats, so caching it would spend memory to no purpose.
   */
  private readonly queryCache = new Map<string, number[]>();

  /** Kept alongside the clients so a failure can be attributed to a credential. */
  private readonly keys: string[];

  constructor(keys: string[]) {
    super({});
    this.keys = keys;
    this.clients = keys.map(
      (apiKey) => new CohereEmbeddings({ apiKey, model: EMBEDDING_CONFIG.model }),
    );
  }

  /**
   * Indices worth trying, in order, starting from the last key that worked.
   *
   * Starting from the last success is not enough on its own. Retrieval issues
   * its query variants concurrently, so several calls enter this loop before
   * any of them has updated the cursor — and each one independently pays the
   * dead key at the head. A live search with a trial key exhausted for the
   * month logged "key 1/7 exhausted" three times for a single question, once
   * per variant, every time.
   *
   * Consulting the shared health registry fixes that, because a refusal
   * recorded by the first call is visible to the others immediately.
   */
  private order(): number[] {
    const all = this.clients.map((_, i) => i);
    const live = all.filter((i) => !isRefused(credentialId(COHERE_ENDPOINT, this.keys[i])));
    // Never return nothing: if every key has been refused, trying them all is
    // still better than failing without having asked.
    const pool = live.length > 0 ? live : all;
    const start = pool.indexOf(this.cursor);
    const from = start >= 0 ? start : 0;
    return [...pool.slice(from), ...pool.slice(0, from)];
  }

  private async withFailover<T>(run: (client: CohereEmbeddings) => Promise<T>): Promise<T> {
    let lastError: unknown;

    for (const index of this.order()) {
      try {
        const result = await run(this.clients[index]);
        this.cursor = index;
        return result;
      } catch (error) {
        lastError = error;
        if (!isKeyExhausted(error)) throw error;
        // A quota measured in days or months will not clear inside this
        // process's usual lifetime, so the key leaves the pool rather than
        // being re-tried by the next caller. A short rate limit is left alone:
        // the rotation exists to ride those out.
        noteFailure(credentialId(COHERE_ENDPOINT, this.keys[index]), error);
        console.warn(
          `[cohere] embeddings key ${index + 1}/${this.clients.length} exhausted, trying the next`,
        );
      }
    }
    throw lastError;
  }

  embedDocuments(texts: string[]): Promise<number[][]> {
    return this.withFailover((client) => client.embedDocuments(texts));
  }

  async embedQuery(text: string): Promise<number[]> {
    const cached = this.queryCache.get(text);
    if (cached) return cached;

    const vector = await this.withFailover((client) => client.embedQuery(text));

    // Bounded so a long-lived process cannot grow without limit. Oldest-first
    // eviction, which for this access pattern is close to least-recently-used:
    // repeats cluster inside a turn and a query from an hour ago is not coming
    // back.
    if (this.queryCache.size >= EMBEDDING_CACHE_LIMIT) {
      const oldest = this.queryCache.keys().next().value;
      if (oldest !== undefined) this.queryCache.delete(oldest);
    }
    this.queryCache.set(text, vector);
    return vector;
  }

  /** Cache occupancy, for /api/health. */
  cacheSize(): number {
    return this.queryCache.size;
  }
}

/** Query vectors held at once. Each is ~1536 floats, so this is a few megabytes. */
const EMBEDDING_CACHE_LIMIT = 500;

let embeddingsSingleton: Embeddings | null = null;

/**
 * Embeddings are a process-wide singleton: the client is stateless but holds a
 * connection pool, and re-creating it per request wastes sockets under load.
 */
export function getEmbeddings(): Embeddings {
  if (embeddingsSingleton) return embeddingsSingleton;

  if (env.EMBEDDING_PROVIDER === "alibaba") {
    const keys = splitKeys(env.ALIBABA_API_KEY ?? env.DEEPSEEK_API_KEY);
    if (keys.length === 0) {
      throw new Error(
        "EMBEDDING_PROVIDER=alibaba needs ALIBABA_API_KEY (or DEEPSEEK_API_KEY pointing at Model Studio).",
      );
    }
    embeddingsSingleton = new AlibabaEmbeddings({
      keys,
      baseURL: env.ALIBABA_BASE_URL ?? env.DEEPSEEK_BASE_URL,
      model: env.ALIBABA_EMBEDDING_MODEL,
      // Same width as the Cohere default, so the two are at least
      // *storable* in the same shape. They are still not interchangeable —
      // see embeddingModelId and the vector store's guard.
      dimensions: EMBEDDING_CONFIG.dimensions,
    });
    return embeddingsSingleton;
  }

  if (COHERE_API_KEYS.length === 0) {
    throw new Error("COHERE_API_KEY is required for embeddings.");
  }
  embeddingsSingleton = new RotatingCohereEmbeddings(COHERE_API_KEYS);
  return embeddingsSingleton;
}

/**
 * Which model produced the vectors this process writes and queries.
 *
 * Recorded on the collection so a provider switch cannot silently corrupt
 * retrieval. Vectors from two embedding models are not comparable even at
 * identical width — measured here at 0.019 similarity against a matching
 * passage and 0.022 against an unrelated one, which is to say noise — and
 * nothing in Qdrant, LangChain or this codebase would raise an error about it.
 * Only an explicit stamp can catch it.
 */
export function embeddingModelId(): string {
  const embedder = getEmbeddings() as { embeddingModelId?: string };
  return embedder.embeddingModelId ?? `cohere:${EMBEDDING_CONFIG.model}:${EMBEDDING_CONFIG.dimensions}`;
}
