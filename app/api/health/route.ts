import { NextResponse } from "next/server";
import { activeSearchProvider } from "@/lib/agents/websearch";
import { MODEL_TIERS, RERANK_CONFIG, env, features } from "@/lib/config";
import { activeModelId } from "@/lib/models";
import { sparseIndexStats } from "@/lib/retrieval/hybrid";
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
    provider: features.deepseek ? "deepseek" : features.openai ? "openai (fallback)" : "none",
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
    reranker: features.cohere ? `cohere:${RERANK_CONFIG.model}` : "disabled",
    sparseIndex: sparseIndexStats(),
  };

  checks.tools = {
    webSearch: activeSearchProvider(),
    tavilyConfigured: features.tavily,
  };

  checks.guardrails = {
    moderation: features.moderation ? "openai" : "disabled (OPENAI_API_KEY missing)",
    piiRedaction: true,
    injectionDetection: true,
    groundednessCheck: true,
  };

  checks.observability = {
    langsmith: features.tracing ? env.LANGSMITH_PROJECT : "disabled",
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
