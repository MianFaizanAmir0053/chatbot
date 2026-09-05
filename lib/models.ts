import { ChatDeepSeek } from "@langchain/deepseek";
import { ChatOpenAI } from "@langchain/openai";
import { ChatCohere, CohereEmbeddings } from "@langchain/cohere";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  COHERE_MODELS,
  EMBEDDING_CONFIG,
  FALLBACK_MODELS,
  LLM_PROVIDERS,
  MODEL_TIERS,
  type LlmProvider,
  env,
  features,
} from "./config";

export type ModelTier = "pro" | "fast";

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
    apiKey: env.COHERE_API_KEY,
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
export function getModel(tier: ModelTier = "pro", options: ModelOptions = {}): BaseChatModel {
  const temperature = options.temperature ?? (tier === "fast" ? 0 : 0.1);

  if (features.deepseek) {
    return new ChatDeepSeek({
      model: tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST,
      apiKey: env.DEEPSEEK_API_KEY,
      temperature,
      maxTokens: options.maxTokens,
      configuration: { baseURL: env.DEEPSEEK_BASE_URL },
    }) as unknown as BaseChatModel;
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

  if (features.deepseek) {
    chain.push(
      new ChatDeepSeek({
        model: tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST,
        apiKey: env.DEEPSEEK_API_KEY,
        temperature,
        configuration: { baseURL: env.DEEPSEEK_BASE_URL },
      }) as unknown as BaseChatModel,
    );
  }
  // Every remaining gateway, so a rate-limited primary does not silently
  // disable the guardrail or planner that sits behind it.
  const primary = `${env.DEEPSEEK_BASE_URL}|${env.DEEPSEEK_API_KEY ?? ""}`;
  for (const p of LLM_PROVIDERS) {
    if (`${p.baseURL}|${p.apiKey}` === primary) continue;
    chain.push(providerModel(p, tier, temperature));
  }

  if (features.cohere) chain.push(cohereModel(tier, temperature));

  return chain;
}

/** Instantiate one registry entry. Every provider speaks the OpenAI wire format. */
function providerModel(provider: LlmProvider, tier: ModelTier, temperature: number): BaseChatModel {
  return new ChatOpenAI({
    model: tier === "pro" ? provider.pro : provider.fast,
    apiKey: provider.apiKey,
    temperature,
    configuration: { baseURL: provider.baseURL },
  }) as unknown as BaseChatModel;
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
  const primary = `${env.DEEPSEEK_BASE_URL}|${env.DEEPSEEK_API_KEY ?? ""}`;

  return LLM_PROVIDERS.filter((p) => `${p.baseURL}|${p.apiKey}` !== primary).map((p) =>
    providerModel(p, tier, 0),
  );
}

/** Model string identifying the active provider, for logs and telemetry. */
export function activeModelId(tier: ModelTier = "pro"): string {
  if (features.deepseek) {
    return `deepseek:${tier === "pro" ? MODEL_TIERS.PRO : MODEL_TIERS.FAST}`;
  }
  if (features.openai) {
    return `openai:${tier === "pro" ? FALLBACK_MODELS.PRO : FALLBACK_MODELS.FAST}`;
  }
  if (features.cohere) {
    return `cohere:${tier === "pro" ? COHERE_MODELS.PRO : COHERE_MODELS.FAST} (auxiliary only)`;
  }
  return "none";
}

let embeddingsSingleton: CohereEmbeddings | null = null;

/**
 * Embeddings are a process-wide singleton: the client is stateless but holds a
 * connection pool, and re-creating it per request wastes sockets under load.
 */
export function getEmbeddings(): CohereEmbeddings {
  if (!features.cohere) {
    throw new Error("COHERE_API_KEY is required for embeddings.");
  }
  if (!embeddingsSingleton) {
    embeddingsSingleton = new CohereEmbeddings({
      apiKey: env.COHERE_API_KEY,
      model: EMBEDDING_CONFIG.model,
    });
  }
  return embeddingsSingleton;
}
