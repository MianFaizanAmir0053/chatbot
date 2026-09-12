import { NextResponse } from "next/server";
import { activeSearchProvider } from "@/lib/agents/websearch";
import { conversationStoreInfo } from "@/lib/conversations/store";
import { credentialId, isRefused, refusedCredentials } from "@/lib/credential-health";
import {
  COHERE_API_KEYS,
  LLM_PROVIDERS,
  MODEL_TIERS,
  RERANK_CONFIG,
  SUBAGENT_CONFIG,
  env,
  features,
} from "@/lib/config";
import {
  activeModelId,
  activeProviderName,
  primaryKeyCount,
  primaryKeysConfigured,
  subagentKeyCount,
  subagentProviderNames,
} from "@/lib/models";
import { sparseIndexStats } from "@/lib/retrieval/hybrid";
import { snapshotsAvailable } from "@/lib/snapshots/store";
import { s3Healthy } from "@/lib/s3";
import { getVectorStore } from "@/lib/vectorstore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Operational readiness.
 *
 * Reports what is actually reachable rather than what is merely configured, so
 * a silent degradation — Qdrant down and the store quietly running in memory —
 * is visible instead of being discovered through bad answers.
 */
export async function GET() {
  const checks: Record<string, unknown> = {};

  const store = await getVectorStore().catch(() => null);
  checks.vectorStore = store
    ? {
        driver: store.name,
        persistent: store.persistent,
        reachable: await store.healthy().catch(() => false),
        chunks: await store.count().catch(() => 0),
        configured: features.qdrant ? env.QDRANT_URL : "not configured",
      }
    : { error: "unavailable" };

  checks.storage = {
    configured: features.s3,
    bucket: env.AWS_S3_BUCKET_NAME || null,
    region: env.AWS_REGION,
    ...(await s3Healthy()),
  };

  const agentCapable = features.deepseek || features.openai;
  checks.models = {
    provider: features.deepseek
      ? activeProviderName()
      : features.openai
        ? "openai (fallback)"
        : "none",
    /**
     * The failover chain, primary first, collapsed to one entry per gateway.
     *
     * A provider holding several keys contributes one chain entry per key, so
     * listing them raw would repeat the same name eight times and read as a
     * misconfiguration. The key count is the useful part: it is how many
     * per-key rate limits that gateway can absorb before the chain moves on.
     */
    chain: [...new Set(LLM_PROVIDERS.map((p) => p.name))].map((name) => {
      const entries = LLM_PROVIDERS.filter((p) => p.name === name);
      // Live count against configured, because they diverge silently. A gateway
      // reported as "8 keys" while seven of them are refused describes failover
      // depth this deployment does not have.
      const live = entries.filter((p) => !isRefused(credentialId(p.baseURL, p.apiKey))).length;
      if (entries.length === 1) return live === 1 ? name : `${name} (refused)`;
      return live === entries.length
        ? `${name} (${entries.length} keys)`
        : `${name} (${live} of ${entries.length} keys live)`;
    }),
    /** Total entries actually tried, which is what bounds failover depth. */
    chainDepth: LLM_PROVIDERS.length,
    /**
     * Keys the primary provider rotates through, one per model call.
     * Free tiers meter per key, so this multiplies the daily ceiling.
     */
    keyRotation: primaryKeyCount(),
    /**
     * Keys configured against the primary provider, refused ones included.
     *
     * Reported next to `keyRotation` so a shrinking pool is visible. The two
     * diverging is the signal that credentials are being refused — otherwise a
     * rotation quietly narrowing to one key looks exactly like a healthy one.
     */
    keysConfigured: primaryKeysConfigured(),
    /**
     * Credentials taken out of the rotation after a refusal that retrying
     * cannot fix. Endpoint and reason only — never the key.
     */
    refusedCredentials: refusedCredentials(),
    pro: activeModelId("pro"),
    fast: activeModelId("fast"),
    configuredTiers: MODEL_TIERS,
    fallbackAvailable: features.deepseek && features.openai,
    agentCapable,
    ...(agentCapable
      ? {}
      : {
          note:
            "No agent-capable provider. Set DEEPSEEK_API_KEY or OPENAI_API_KEY — Cohere can serve " +
            "auxiliary calls but cannot drive the agent loop.",
        }),
  };

  checks.retrieval = {
    embeddings: features.cohere ? "cohere:embed-v4.0" : "unavailable (COHERE_API_KEY missing)",
    /**
     * How many Cohere keys back embeddings and reranking.
     *
     * Worth surfacing on its own: Cohere is the only source of both, so this
     * number is how many rate limits retrieval can absorb before it degrades
     * to fusion order.
     */
    cohereKeys: COHERE_API_KEYS.length,
    reranker: features.cohere ? `cohere:${RERANK_CONFIG.model}` : "disabled",
    sparseIndex: sparseIndexStats(),
  };

  checks.tools = {
    webSearch: activeSearchProvider(),
    tavilyConfigured: features.tavily,
    /**
     * Whether earlier captures of fetched pages are kept.
     *
     * Reported because its absence is silent and changes what the system can
     * answer: with no archive, "what changed on this page?" has no evidence
     * behind it, and the honest reply is that the page has not been read
     * before.
     */
    pageArchive: (await snapshotsAvailable()) ? "available" : "unavailable (no MONGODB_URI)",
  };

  checks.guardrails = {
    moderation: features.moderation
      ? "openai"
      : env.OPENAI_BASE_URL
        ? "disabled (OPENAI_BASE_URL set — gateways do not serve omni-moderation)"
        : "disabled (OPENAI_API_KEY missing)",
    piiRedaction: true,
    injectionDetection: true,
    groundednessCheck: true,
  };

  // Reported because the fallback engages silently: a database that is
  // unreachable at boot costs durability, and on a read-only filesystem the
  // fallback keeps conversations in memory only, losing them on every restart.
  checks.conversations = await conversationStoreInfo();

  checks.observability = {
    langsmith: features.tracing ? env.LANGSMITH_PROJECT : "disabled",
  };

  // Delegated research runs on a deliberately narrower provider set than the
  // main chain, so reporting the main chain here would misdescribe it — and the
  // per-branch ceilings are the only thing bounding spend inside a subagent,
  // which makes them worth being able to read without the source.
  checks.delegation = {
    providers: subagentProviderNames(),
    keys: subagentKeyCount(),
    branchesPerRound: SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND,
    maxRounds: SUBAGENT_CONFIG.MAX_DELEGATION_ROUNDS,
    maxBranches:
      SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND * SUBAGENT_CONFIG.MAX_DELEGATION_ROUNDS,
    // One budget for the turn, supervisor and branches together — the limiter's
    // tally lives in shared state, so it counts both.
    // Reported because it is the difference between a verification step that
    // exists and one that runs: as a prompt instruction it was skipped on every
    // measured turn.
    autoVerify: SUBAGENT_CONFIG.AUTO_VERIFY,
    retrievalConcurrency: SUBAGENT_CONFIG.RETRIEVAL_CONCURRENCY,
    turnModelCalls: SUBAGENT_CONFIG.TURN_MODEL_CALLS,
    turnToolCalls: SUBAGENT_CONFIG.TURN_TOOL_CALLS,
  };

  // Everything needed to answer a question about an already-indexed document.
  // Losing any of these means the core path is broken.
  const core = [
    agentCapable,
    features.cohere,
    Boolean((checks.vectorStore as { reachable?: boolean })?.reachable),
  ];
  // Uploading needs S3. Its absence degrades the service rather than breaking
  // queries, but reporting "healthy" while uploads fail would be misleading.
  const uploadsWorking = Boolean((checks.storage as { reachable?: boolean })?.reachable);

  const status = !core.some(Boolean)
    ? "unhealthy"
    : core.every(Boolean) && uploadsWorking
      ? "healthy"
      : "degraded";

  return NextResponse.json(
    { status, timestamp: new Date().toISOString(), checks },
    { status: status === "unhealthy" ? 503 : 200 },
  );
}
