"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ACCEPTED_UPLOADS, useKnowledge } from "../components/knowledge-context";
import {
  Button,
  Card,
  CardHeader,
  EmptyState,
  Meter,
  Pill,
  Row,
  SectionTitle,
  Skeleton,
  Stat,
  compactNumber,
  relativeTime,
  toneFor,
  type Tone,
} from "../components/ui";
import {
  ActivityIcon,
  AlertIcon,
  CloudIcon,
  CpuIcon,
  DatabaseIcon,
  DocumentIcon,
  GlobeIcon,
  LayersIcon,
  NetworkIcon,
  RefreshIcon,
  SearchIcon,
  ShieldIcon,
  TrashIcon,
  UploadIcon,
} from "../components/icons";

/* ------------------------------------------------------------------ *
 * Types — the shape /api/health reports.
 *
 * Every field is optional: the route degrades a subsystem to an error
 * object rather than failing the whole response, and the dashboard has
 * to render that honestly instead of crashing.
 * ------------------------------------------------------------------ */

type HealthStatus = "healthy" | "degraded" | "unhealthy";

type Health = {
  status: HealthStatus;
  timestamp: string;
  checks: {
    vectorStore?: {
      driver?: string;
      persistent?: boolean;
      reachable?: boolean;
      chunks?: number;
      configured?: string;
      error?: string;
    };
    storage?: {
      configured?: boolean;
      bucket?: string | null;
      region?: string;
      reachable?: boolean;
      error?: string;
      hint?: string;
    };
    models?: {
      provider?: string;
      pro?: string;
      fast?: string;
      fallbackAvailable?: boolean;
      agentCapable?: boolean;
      note?: string;
      /** Failover chain, primary first. */
      chain?: string[];
    };
    retrieval?: {
      embeddings?: string;
      reranker?: string;
      sparseIndex?: { built?: boolean; documents?: number };
    };
    tools?: { webSearch?: string; tavilyConfigured?: boolean };
    guardrails?: Record<string, string | boolean>;
    observability?: { langsmith?: string };
    delegation?: {
      providers?: string[];
      keys?: number;
      branchesPerRound?: number;
      maxRounds?: number;
      maxBranches?: number;
      turnModelCalls?: number;
      turnToolCalls?: number;
    };
  };
};

const STATUS_COPY: Record<HealthStatus, { tone: Tone; label: string; detail: string }> = {
  healthy: {
    tone: "success",
    label: "All systems operational",
    detail: "Retrieval, generation and ingestion are all reachable.",
  },
  degraded: {
    tone: "warn",
    label: "Running degraded",
    detail: "The core query path works, but at least one subsystem is unavailable.",
  },
  unhealthy: {
    tone: "danger",
    label: "Service unavailable",
    detail: "The core query path cannot serve grounded answers right now.",
  },
};

const REFRESH_MS = 30_000;

export default function DashboardPage() {
  const kb = useKnowledge();
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, forceTick] = useState(0);

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      // 503 still carries a full report — it is the unhealthy case, not a failure.
      const data = (await res.json()) as Health;
      setHealth(data);
      setFetchedAt(new Date().toISOString());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach /api/health");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const refreshKnowledge = kb.refresh;

  useEffect(() => {
    void load();
    void refreshKnowledge();
    const poll = window.setInterval(() => {
      void load();
      void refreshKnowledge();
    }, REFRESH_MS);
    // Keep the "updated Ns ago" label truthful between polls.
    const tick = window.setInterval(() => forceTick((n) => n + 1), 10_000);
    return () => {
      window.clearInterval(poll);
      window.clearInterval(tick);
    };
  }, [load, refreshKnowledge]);

  const checks = health?.checks ?? {};
  const status = health?.status ?? "degraded";
  const copy = STATUS_COPY[status];

  const largest = useMemo(
    () => kb.documents.reduce((max, d) => Math.max(max, d.chunks), 0),
    [kb.documents],
  );

  const avgChunks = kb.documents.length
    ? Math.round(kb.totalChunks / kb.documents.length)
    : 0;

  return (
    <>
      <header className="glass sticky top-0 z-10 flex h-16 shrink-0 items-center justify-between gap-4 border-b border-line px-4 pl-16 lg:px-6">
        <div className="min-w-0">
          <h1 className="truncate text-[15px] font-semibold tracking-tight text-ink">Dashboard</h1>
          <p className="truncate text-xs text-ink-3">
            Live operational status · updated {relativeTime(fetchedAt)}
          </p>
        </div>
        <Button size="sm" variant="secondary" onClick={load} disabled={refreshing}>
          <RefreshIcon className={`w-3.5 h-3.5 ${refreshing ? "animate-spin-slow" : ""}`} />
          <span className="hidden sm:inline">Refresh</span>
        </Button>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6 lg:px-6">
        <div className="mx-auto w-full max-w-6xl space-y-8">
          {/* ---------- Status banner ---------- */}
          {loading ? (
            <Skeleton className="h-24 w-full rounded-xl" />
          ) : error ? (
            <Card className="border-[var(--danger)]">
              <div className="flex items-start gap-3">
                <AlertIcon className="mt-0.5 w-5 h-5 shrink-0" style={{ color: "var(--danger)" }} />
                <div>
                  <p className="text-sm font-semibold text-ink">Health check unreachable</p>
                  <p className="mt-1 text-xs text-ink-3">{error}</p>
                </div>
              </div>
            </Card>
          ) : (
            <Card className="animate-fade-up">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <span
                    className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl"
                    style={{
                      background: `var(--${copy.tone}-soft)`,
                      color: `var(--${copy.tone})`,
                    }}
                  >
                    <ActivityIcon className="w-5 h-5" />
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-base font-semibold tracking-tight text-ink">
                        {copy.label}
                      </h2>
                      <Pill tone={copy.tone} dot pulse={status !== "healthy"}>
                        {status}
                      </Pill>
                    </div>
                    <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-3">
                      {copy.detail}
                    </p>
                  </div>
                </div>

                <dl className="flex gap-6">
                  <MiniStat
                    label="Query path"
                    ok={Boolean(checks.models?.agentCapable && checks.vectorStore?.reachable)}
                  />
                  <MiniStat label="Ingestion" ok={Boolean(checks.storage?.reachable)} />
                  <MiniStat label="Reranking" ok={!checks.retrieval?.reranker?.includes("disabled")} />
                </dl>
              </div>
            </Card>
          )}

          {/* ---------- Key metrics ---------- */}
          <section>
            <SectionTitle>Corpus</SectionTitle>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Stat
                label="Documents"
                value={kb.loading ? "—" : kb.documents.length}
                icon={<DocumentIcon className="w-4 h-4" />}
                tone="accent"
                hint={
                  kb.documents.length
                    ? `${avgChunks} chunks per document on average`
                    : "Nothing indexed yet"
                }
              />
              <Stat
                label="Indexed chunks"
                value={kb.loading ? "—" : compactNumber(kb.totalChunks)}
                icon={<LayersIcon className="w-4 h-4" />}
                tone="accent"
                hint={`Vector store reports ${compactNumber(checks.vectorStore?.chunks ?? 0)}`}
              />
              <Stat
                label="Sparse index"
                value={compactNumber(checks.retrieval?.sparseIndex?.documents ?? 0)}
                icon={<SearchIcon className="w-4 h-4" />}
                tone={checks.retrieval?.sparseIndex?.built ? "success" : "warn"}
                hint={
                  checks.retrieval?.sparseIndex?.built
                    ? "BM25 index built"
                    : "Builds on the first query"
                }
              />
              <Stat
                label="Vector store"
                value={
                  <span className="text-lg">{checks.vectorStore?.driver ?? kb.driver ?? "—"}</span>
                }
                icon={<DatabaseIcon className="w-4 h-4" />}
                tone={toneFor(checks.vectorStore?.reachable)}
                trailing={
                  <Pill tone={checks.vectorStore?.persistent ? "success" : "warn"}>
                    {checks.vectorStore?.persistent ? "persistent" : "in-memory"}
                  </Pill>
                }
                hint={
                  checks.vectorStore?.persistent
                    ? "Survives a restart"
                    : "Set QDRANT_URL to persist documents"
                }
              />
            </div>
          </section>

          {/* ---------- Subsystems ---------- */}
          <section>
            <SectionTitle>Subsystems</SectionTitle>
            {loading ? (
              <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-44 w-full rounded-xl" />
                ))}
              </div>
            ) : (
              <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
                <Card>
                  <CardHeader
                    icon={<DatabaseIcon className="w-4 h-4" />}
                    title="Vector store"
                    subtitle="Dense retrieval backend"
                    action={<StatusPill ok={checks.vectorStore?.reachable} />}
                  />
                  <Row label="Driver">{checks.vectorStore?.driver ?? "—"}</Row>
                  <Row label="Persistence">
                    {checks.vectorStore?.persistent ? "Durable" : "Process memory"}
                  </Row>
                  <Row label="Chunks stored" mono>
                    {compactNumber(checks.vectorStore?.chunks ?? 0)}
                  </Row>
                  <Row label="Endpoint" mono>
                    {checks.vectorStore?.configured ?? "not configured"}
                  </Row>
                </Card>

                <Card>
                  <CardHeader
                    icon={<CpuIcon className="w-4 h-4" />}
                    title="Models"
                    subtitle="Planning and synthesis"
                    action={<StatusPill ok={checks.models?.agentCapable} />}
                  />
                  <Row label="Provider">{checks.models?.provider ?? "none"}</Row>
                  <Row label="Pro tier" mono>
                    {checks.models?.pro ?? "—"}
                  </Row>
                  <Row label="Fast tier" mono>
                    {checks.models?.fast ?? "—"}
                  </Row>
                  <Row label="Fallback">
                    {checks.models?.fallbackAvailable ? "Configured" : "None"}
                  </Row>
                  {(checks.models?.chain?.length ?? 0) > 0 && (
                    <div className="flex items-center justify-between gap-3 py-2">
                      <span className="shrink-0 text-xs text-ink-3">Failover chain</span>
                      {/* Order is the order tried, so keep it visually sequential. */}
                      <div className="flex flex-wrap items-center justify-end gap-1">
                        {checks.models?.chain?.map((name, i) => (
                          <span key={name} className="flex items-center gap-1">
                            {i > 0 && <span className="text-[10px] text-ink-3">→</span>}
                            <Pill tone={i === 0 ? "accent" : "neutral"}>{name}</Pill>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {checks.models?.note && (
                    <Note tone="warn">{checks.models.note}</Note>
                  )}
                </Card>

                <Card>
                  <CardHeader
                    icon={<NetworkIcon className="w-4 h-4" />}
                    title="Delegated research"
                    subtitle="Specialist subagents, isolated contexts"
                    action={
                      <StatusPill ok={(checks.delegation?.providers?.length ?? 0) > 0} />
                    }
                  />
                  <Row label="Researchers">
                    Documents · Web · Verifier
                  </Row>
                  <Row label="Parallel branches per round">
                    {checks.delegation?.branchesPerRound ?? "—"}
                  </Row>
                  <Row label="Delegation rounds per turn">
                    {checks.delegation?.maxRounds ?? "—"}
                  </Row>
                  {/* Ceilings for the whole turn, supervisor and branches
                      together: the limiters keep their tally in shared agent
                      state, so they count a researcher's calls as well. */}
                  <Row label="Model calls per turn">
                    {checks.delegation?.turnModelCalls ?? "—"}
                  </Row>
                  <Row label="Tool calls per turn">
                    {checks.delegation?.turnToolCalls ?? "—"}
                  </Row>
                  {(checks.delegation?.providers?.length ?? 0) > 0 && (
                    <div className="flex items-center justify-between gap-3 py-2">
                      <span className="shrink-0 text-xs text-ink-3">Branch providers</span>
                      <div className="flex flex-wrap items-center justify-end gap-1">
                        {checks.delegation?.providers?.map((name) => (
                          <Pill key={name} tone="neutral">
                            {name}
                          </Pill>
                        ))}
                      </div>
                    </div>
                  )}
                  <Note tone="neutral">
                    Branches run on a narrower provider set than the main chain — a gateway kept
                    as a last resort for single requests would set the latency of an entire
                    fan-out.
                  </Note>
                </Card>

                <Card>
                  <CardHeader
                    icon={<SearchIcon className="w-4 h-4" />}
                    title="Retrieval"
                    subtitle="Embeddings, reranking, hybrid search"
                    action={
                      <StatusPill ok={!checks.retrieval?.embeddings?.includes("unavailable")} />
                    }
                  />
                  <Row label="Embeddings" mono>
                    {checks.retrieval?.embeddings ?? "—"}
                  </Row>
                  <Row label="Reranker" mono>
                    {checks.retrieval?.reranker ?? "—"}
                  </Row>
                  <Row label="Sparse index">
                    {checks.retrieval?.sparseIndex?.built
                      ? `Built · ${compactNumber(checks.retrieval.sparseIndex.documents ?? 0)} chunks`
                      : "Not built yet"}
                  </Row>
                </Card>

                <Card>
                  <CardHeader
                    icon={<CloudIcon className="w-4 h-4" />}
                    title="Object storage"
                    subtitle="Upload and ingestion source"
                    action={<StatusPill ok={checks.storage?.reachable} />}
                  />
                  <Row label="Bucket" mono>
                    {checks.storage?.bucket ?? "not configured"}
                  </Row>
                  <Row label="Region" mono>
                    {checks.storage?.region ?? "—"}
                  </Row>
                  <Row label="Uploads">
                    {checks.storage?.reachable ? "Accepting files" : "Unavailable"}
                  </Row>
                  {(checks.storage?.error || checks.storage?.hint) && (
                    <Note tone="danger">{checks.storage.error ?? checks.storage.hint}</Note>
                  )}
                </Card>

                <Card>
                  <CardHeader
                    icon={<GlobeIcon className="w-4 h-4" />}
                    title="Tools"
                    subtitle="What the agent can reach"
                    action={<StatusPill ok={Boolean(checks.tools?.webSearch)} />}
                  />
                  <Row label="Web search">{checks.tools?.webSearch ?? "—"}</Row>
                  <Row label="Tavily key">
                    {checks.tools?.tavilyConfigured ? "Configured" : "Not set"}
                  </Row>
                  <Row label="Tracing" mono>
                    {checks.observability?.langsmith ?? "disabled"}
                  </Row>
                </Card>

                <Card>
                  <CardHeader
                    icon={<ShieldIcon className="w-4 h-4" />}
                    title="Guardrails"
                    subtitle="Input and output safety"
                    action={<StatusPill ok label="active" />}
                  />
                  <div className="flex flex-wrap gap-1.5 pt-0.5">
                    {Object.entries(checks.guardrails ?? {}).map(([key, value]) => {
                      const off = value === false || String(value).startsWith("disabled");
                      return (
                        <Pill key={key} tone={off ? "neutral" : "success"} dot title={String(value)}>
                          {humanise(key)}
                        </Pill>
                      );
                    })}
                  </div>
                </Card>
              </div>
            )}
          </section>

          {/* ---------- Documents ---------- */}
          <section>
            <SectionTitle
              action={
                <span className="tnum text-[11px] text-ink-3">
                  {kb.documents.length} indexed · {compactNumber(kb.totalChunks)} chunks
                </span>
              }
            >
              Indexed documents
            </SectionTitle>

            <Card padded={false}>
              {kb.loading ? (
                <div className="space-y-2 p-5">
                  {[0, 1, 2].map((i) => (
                    <Skeleton key={i} className="h-10 w-full" />
                  ))}
                </div>
              ) : kb.documents.length === 0 ? (
                <EmptyState
                  icon={<DocumentIcon className="w-5 h-5" />}
                  title="No documents indexed"
                  body="Upload a PDF, DOCX, Markdown or CSV file and it becomes queryable as soon as ingestion finishes."
                  action={<UploadButton />}
                />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[560px] text-left">
                    <thead>
                      <tr className="border-b border-line">
                        <th className="px-5 py-2.5 text-[11px] font-semibold uppercase tracking-[0.07em] text-ink-3">
                          Document
                        </th>
                        <th className="px-4 py-2.5 text-right text-[11px] font-semibold uppercase tracking-[0.07em] text-ink-3">
                          Chunks
                        </th>
                        <th className="w-[36%] px-4 py-2.5 text-[11px] font-semibold uppercase tracking-[0.07em] text-ink-3">
                          Share of index
                        </th>
                        <th className="px-5 py-2.5" />
                      </tr>
                    </thead>
                    <tbody>
                      {[...kb.documents]
                        .sort((a, b) => b.chunks - a.chunks)
                        .map((doc) => (
                          <tr
                            key={doc.source}
                            className="group border-b border-line last:border-0 transition-colors hover:bg-surface-muted"
                          >
                            <td className="px-5 py-3">
                              <div className="flex items-center gap-2.5">
                                <DocumentIcon className="w-4 h-4 shrink-0 text-ink-3" />
                                <span
                                  className="max-w-[28ch] truncate text-[13px] font-medium text-ink"
                                  title={doc.source}
                                >
                                  {doc.source}
                                </span>
                              </div>
                            </td>
                            <td className="tnum px-4 py-3 text-right text-[13px] text-ink-2">
                              {doc.chunks}
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2.5">
                                <Meter value={largest ? doc.chunks / largest : 0} />
                                <span className="tnum w-10 shrink-0 text-right text-[11px] text-ink-3">
                                  {kb.totalChunks
                                    ? `${Math.round((doc.chunks / kb.totalChunks) * 100)}%`
                                    : "—"}
                                </span>
                              </div>
                            </td>
                            <td className="px-5 py-3 text-right">
                              <button
                                type="button"
                                onClick={() => kb.remove(doc.source)}
                                title={`Remove ${doc.source}`}
                                aria-label={`Remove ${doc.source}`}
                                className="grid h-7 w-7 place-items-center rounded-md text-ink-3 opacity-0 transition-opacity hover:text-[var(--danger)] group-hover:opacity-100 focus-visible:opacity-100"
                              >
                                <TrashIcon className="w-3.5 h-3.5" />
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </section>

          <p className="pb-2 text-center text-[11px] text-ink-3">
            Auto-refreshing every {REFRESH_MS / 1000}s
            {health?.timestamp && ` · report generated ${new Date(health.timestamp).toLocaleTimeString()}`}
          </p>
        </div>
      </main>
    </>
  );
}

/* ------------------------------------------------------------------ *
 * Local pieces
 * ------------------------------------------------------------------ */

function MiniStat({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="min-w-[84px]">
      <dt className="text-[11px] text-ink-3">{label}</dt>
      <dd className="mt-1 flex items-center gap-1.5 text-[13px] font-medium text-ink">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: ok ? "var(--success)" : "var(--danger)" }}
        />
        {ok ? "Ready" : "Down"}
      </dd>
    </div>
  );
}

function StatusPill({ ok, label }: { ok?: boolean; label?: string }) {
  const tone = toneFor(ok);
  return (
    <Pill tone={tone} dot>
      {label ?? (ok === undefined ? "unknown" : ok ? "operational" : "unavailable")}
    </Pill>
  );
}

function Note({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <p
      className="mt-3 rounded-lg px-2.5 py-2 text-[11px] leading-relaxed"
      style={{ background: `var(--${tone}-soft)`, color: `var(--${tone})` }}
    >
      {children}
    </p>
  );
}

function UploadButton() {
  const kb = useKnowledge();
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_UPLOADS}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void kb.upload(file);
        }}
      />
      <Button
        type="button"
        variant="primary"
        size="sm"
        disabled={kb.uploading}
        onClick={() => inputRef.current?.click()}
      >
        <UploadIcon className="w-3.5 h-3.5" />
        {kb.uploading ? "Indexing…" : "Upload a document"}
      </Button>
    </>
  );
}

/** `piiRedaction` → `Pii redaction`. */
function humanise(key: string): string {
  const spaced = key.replace(/([a-z])([A-Z])/g, "$1 $2");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
