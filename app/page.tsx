"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ACCEPTED_UPLOADS, useKnowledge } from "./components/knowledge-context";
import { Button, Pill, type Tone } from "./components/ui";
import {
  BrandMark,
  CheckIcon,
  ChevronIcon,
  CopyIcon,
  DocumentIcon,
  GlobeIcon,
  LinkIcon,
  ListIcon,
  PaperclipIcon,
  PlusIcon,
  SearchIcon,
  SendIcon,
  ShieldIcon,
  SparkIcon,
  StopIcon,
  UploadIcon,
} from "./components/icons";

/* ------------------------------------------------------------------ *
 * Types — mirror the SSE event protocol emitted by /api/chat
 * ------------------------------------------------------------------ */

type Todo = { content: string; status: "pending" | "in_progress" | "completed" };

type SourceDoc = {
  index: number;
  source: string;
  section?: string;
  page?: number;
  score: number;
  excerpt: string;
};

type TraceEntry = { kind: "status" | "tool" | "warning"; label: string; detail?: string };

type Groundedness = {
  score: number;
  verdict: "grounded" | "partially_grounded" | "unsupported";
  passed: boolean;
  unsupportedClaims: string[];
};

type Message = {
  role: "user" | "assistant";
  content: string;
  todos?: Todo[];
  trace?: TraceEntry[];
  sources?: SourceDoc[];
  web?: Array<{ title: string; url: string }>;
  groundedness?: Groundedness;
  blocked?: boolean;
};

/* ------------------------------------------------------------------ *
 * Presentation helpers
 * ------------------------------------------------------------------ */

const TOOL_LABELS: Record<string, string> = {
  search_documents: "Searching documents",
  list_documents: "Listing documents",
  web_search: "Searching the web",
  fetch_url: "Reading a web page",
  calculator: "Calculating",
  write_todos: "Planning",
};

const SUGGESTIONS = [
  { icon: ListIcon, text: "Summarise the key findings across my documents" },
  { icon: SearchIcon, text: "What does the documentation say about rate limits?" },
  { icon: SparkIcon, text: "Compare the conclusions of the two most recent uploads" },
  { icon: GlobeIcon, text: "Check my documents against current public guidance" },
];

function SimpleMarkdown({ content }: { content: string }) {
  const rendered = useMemo(() => {
    let html = content.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, _lang, code) => {
      return `<pre><code>${String(code).trim()}</code></pre>`;
    });
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Citation markers get a visual anchor so provenance is scannable.
    html = html.replace(/\[(\d+)\]/g, '<sup class="citation-ref">$1</sup>');
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    html = html.replace(/^[-*] (.*)$/gm, '<li class="ml-4 list-disc">$1</li>');
    html = html.replace(/\n/g, "<br />");
    return html;
  }, [content]);

  return <div className="prose-content" dangerouslySetInnerHTML={{ __html: rendered }} />;
}

function TypingIndicator({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 py-1">
      <span className="flex items-center gap-1">
        {[0, 160, 320].map((delay) => (
          <span
            key={delay}
            className="w-1.5 h-1.5 rounded-full animate-pulse-dot"
            style={{ background: "var(--accent)", animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
      {label && <span className="text-xs text-ink-3">{label}</span>}
    </div>
  );
}

/** Copy-to-clipboard with a transient confirmation. */
function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      window.setTimeout(() => setDone(false), 1600);
    } catch {
      /* clipboard is unavailable over plain http; nothing useful to say */
    }
  }

  return (
    <button
      type="button"
      onClick={copy}
      title={done ? "Copied" : "Copy answer"}
      aria-label={done ? "Copied" : "Copy answer"}
      className="grid h-7 w-7 place-items-center rounded-md text-ink-3 hover:bg-surface-hover hover:text-ink transition-colors"
    >
      {done ? (
        <CheckIcon className="w-3.5 h-3.5" style={{ color: "var(--success)" }} />
      ) : (
        <CopyIcon className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

/** The agent's live plan. This is `todos` straight out of agent state. */
function PlanPanel({ todos }: { todos: Todo[] }) {
  if (todos.length === 0) return null;
  const done = todos.filter((t) => t.status === "completed").length;
  const pct = Math.round((done / todos.length) * 100);

  return (
    <div className="mb-3 rounded-lg border border-line bg-surface-inset p-3">
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-3">
          Plan
        </span>
        <span className="tnum text-[11px] text-ink-3">
          {done}/{todos.length}
        </span>
      </div>
      <div
        className="mb-3 h-1 w-full overflow-hidden rounded-full"
        style={{ background: "var(--surface-hover)" }}
      >
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: "var(--accent)" }}
        />
      </div>
      <ul className="space-y-1.5">
        {todos.map((todo, i) => (
          <li key={i} className="flex items-start gap-2 text-[13px]">
            <span className="mt-[3px] shrink-0">
              {todo.status === "completed" ? (
                <CheckIcon className="w-3.5 h-3.5" style={{ color: "var(--success)" }} />
              ) : todo.status === "in_progress" ? (
                <span
                  className="block h-3 w-3 rounded-full border-2 border-current border-t-transparent animate-spin-slow"
                  style={{ color: "var(--accent)" }}
                />
              ) : (
                <span
                  className="block h-3 w-3 rounded-full border-2"
                  style={{ borderColor: "var(--line-strong)" }}
                />
              )}
            </span>
            <span className={todo.status === "completed" ? "text-ink-3 line-through" : "text-ink-2"}>
              {todo.content}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** What the agent actually did, collapsed by default. */
function TracePanel({ trace }: { trace: TraceEntry[] }) {
  const [open, setOpen] = useState(false);
  if (trace.length === 0) return null;

  const warnings = trace.filter((t) => t.kind === "warning").length;

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 -ml-1.5 text-[11px] font-medium text-ink-3 hover:bg-surface-hover hover:text-ink transition-colors"
      >
        <ChevronIcon
          className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        {trace.length} step{trace.length === 1 ? "" : "s"}
        {warnings > 0 && (
          <span style={{ color: "var(--warn)" }}>
            · {warnings} warning{warnings === 1 ? "" : "s"}
          </span>
        )}
      </button>

      {open && (
        <ol className="mt-2 space-y-1.5 border-l border-line pl-3.5 ml-1 animate-fade-in">
          {trace.map((entry, i) => (
            <li key={i} className="relative text-[11px] leading-relaxed text-ink-3">
              <span
                className="absolute -left-[18px] top-[5px] h-1.5 w-1.5 rounded-full"
                style={{
                  background:
                    entry.kind === "warning"
                      ? "var(--warn)"
                      : entry.kind === "tool"
                        ? "var(--accent)"
                        : "var(--line-strong)",
                }}
              />
              <span className="text-ink-2">{entry.label}</span>
              {entry.detail && <span className="text-ink-3"> — {entry.detail}</span>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function GroundednessBadge({ g }: { g: Groundedness }) {
  // -1 signals the judge itself was unavailable; don't imply a verified result.
  if (g.score < 0) return null;

  const tone: Tone = g.passed ? (g.score >= 0.8 ? "success" : "warn") : "danger";
  const label = g.passed
    ? g.score >= 0.8
      ? "Grounded in sources"
      : "Partially grounded"
    : "Weakly supported";

  return (
    <Pill
      tone={tone}
      className="mt-3"
      title={
        g.unsupportedClaims.length > 0
          ? `Unsupported: ${g.unsupportedClaims.join("; ")}`
          : undefined
      }
    >
      <ShieldIcon className="w-3 h-3" />
      {label}
      <span className="tnum opacity-70">{Math.round(g.score * 100)}%</span>
    </Pill>
  );
}

function SourcesPanel({
  sources,
  web,
}: {
  sources: SourceDoc[];
  web: Array<{ title: string; url: string }>;
}) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0 && web.length === 0) return null;

  return (
    <div className="mt-3 border-t border-line pt-2.5">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 -ml-1.5 text-[11px] font-semibold transition-colors hover:bg-surface-hover"
        style={{ color: "var(--accent)" }}
      >
        <ChevronIcon
          className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        {sources.length} document source{sources.length === 1 ? "" : "s"}
        {web.length > 0 && ` · ${web.length} web result${web.length === 1 ? "" : "s"}`}
      </button>

      {open && (
        <div className="mt-2.5 grid gap-2 animate-fade-in">
          {sources.map((s) => (
            <article
              key={s.index}
              className="rounded-lg border border-line bg-surface-inset p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className="tnum grid h-5 min-w-5 shrink-0 place-items-center rounded px-1 font-mono text-[10px] font-semibold"
                    style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                  >
                    {s.index}
                  </span>
                  <span className="truncate text-[12px] font-medium text-ink" title={s.source}>
                    {s.source}
                  </span>
                </div>
                <span
                  className="tnum shrink-0 text-[11px] font-medium"
                  title="Relevance score"
                  style={{ color: "var(--ink-3)" }}
                >
                  {Math.round(s.score * 100)}%
                </span>
              </div>

              {(s.section || s.page) && (
                <div className="mt-1 text-[11px] text-ink-3">
                  {s.section}
                  {s.section && s.page ? " · " : ""}
                  {s.page ? `page ${s.page}` : ""}
                </div>
              )}

              <p className="mt-2 line-clamp-3 text-[12px] leading-relaxed text-ink-3">
                {s.excerpt}
              </p>
            </article>
          ))}

          {web.map((w, i) => (
            <a
              key={i}
              href={w.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group rounded-lg border border-line bg-surface-inset p-3 transition-colors hover:bg-surface-hover"
            >
              <div className="flex items-center gap-2">
                <GlobeIcon className="w-3.5 h-3.5 shrink-0 text-ink-3" />
                <span
                  className="truncate text-[12px] font-medium"
                  style={{ color: "var(--accent)" }}
                >
                  {w.title || w.url}
                </span>
                <LinkIcon className="w-3 h-3 shrink-0 text-ink-3 opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
              <div className="mt-1 truncate text-[11px] text-ink-3">{w.url}</div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default function ChatPage() {
  const kb = useKnowledge();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const endRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  // Grow the composer with its content instead of scrolling a one-line box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  /* --- Upload --- */

  const ingest = useCallback(
    async (file: File) => {
      const result = await kb.upload(file);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            result.status === "ingested"
              ? `Indexed **${result.file}** — ${result.chunks} chunks${
                  result.pages ? ` from ${result.pages} pages` : ""
                }. Ask me anything about it.`
              : result.status === "skipped"
                ? `**${result.file}** is already indexed and unchanged.`
                : `Could not index **${result.file}**: ${result.error ?? "unknown error"}`,
        },
      ]);
    },
    [kb],
  );

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) await ingest(file);
  }

  /* --- Drag and drop over the whole conversation area --- */

  function onDragEnter(e: React.DragEvent) {
    if (!e.dataTransfer.types.includes("Files")) return;
    dragDepth.current += 1;
    setDragging(true);
  }

  function onDragLeave() {
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await ingest(file);
  }

  /* --- Chat --- */

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setStatus("");
  }

  function newConversation() {
    stop();
    setMessages([]);
    setThreadId(null);
    setInput("");
    textareaRef.current?.focus();
  }

  async function send(e?: React.FormEvent, override?: string) {
    e?.preventDefault();
    const text = (override ?? input).trim();
    if (!text || loading) return;

    const history = messages
      .filter((m) => !m.blocked)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);
    setStatus("Starting");

    const controller = new AbortController();
    abortRef.current = controller;

    // The assistant message is created up-front and mutated as events arrive.
    setMessages((prev) => [...prev, { role: "assistant", content: "", trace: [], todos: [] }]);

    const patch = (fn: (m: Message) => Message) =>
      setMessages((prev) => {
        const next = [...prev];
        const i = next.length - 1;
        if (i >= 0 && next[i].role === "assistant") next[i] = fn(next[i]);
        return next;
      });

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Omit threadId entirely on the first message rather than sending null:
        // "absent" is what the server means by a new conversation.
        body: JSON.stringify({
          message: text,
          history,
          ...(threadId ? { threadId } : {}),
          mode: "agentic",
        }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;

          let data: Record<string, unknown>;
          try {
            data = JSON.parse(trimmed.slice(6));
          } catch {
            continue;
          }

          switch (data.event) {
            case "thread":
              setThreadId(String(data.threadId));
              break;

            case "status":
              setStatus(String(data.stage ?? ""));
              patch((m) => ({
                ...m,
                trace: [
                  ...(m.trace ?? []),
                  { kind: "status", label: String(data.stage), detail: data.detail as string },
                ],
              }));
              break;

            case "todos":
              patch((m) => ({ ...m, todos: data.todos as Todo[] }));
              break;

            case "tool_call": {
              const name = String(data.name);
              const args = data.args as Record<string, unknown> | undefined;
              setStatus(TOOL_LABELS[name] ?? name);
              patch((m) => ({
                ...m,
                trace: [
                  ...(m.trace ?? []),
                  {
                    kind: "tool",
                    label: TOOL_LABELS[name] ?? name,
                    detail: typeof args?.query === "string" ? `"${args.query}"` : undefined,
                  },
                ],
              }));
              break;
            }

            case "token":
              patch((m) => ({ ...m, content: m.content + String(data.text ?? "") }));
              break;

            case "revised_answer":
              patch((m) => ({ ...m, content: String(data.text ?? "") }));
              break;

            case "sources":
              patch((m) => ({
                ...m,
                sources: data.documents as SourceDoc[],
                web: data.web as Array<{ title: string; url: string }>,
              }));
              break;

            case "groundedness":
              patch((m) => ({ ...m, groundedness: data as unknown as Groundedness }));
              break;

            case "ingested":
              void kb.refresh();
              break;

            case "warning":
              patch((m) => ({
                ...m,
                trace: [...(m.trace ?? []), { kind: "warning", label: String(data.message) }],
              }));
              break;

            case "blocked":
              patch((m) => ({ ...m, blocked: true }));
              break;

            case "error":
              patch((m) => ({
                ...m,
                content: m.content || `Error: ${String(data.message)}`,
              }));
              break;
          }
        }
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        const msg = error instanceof Error ? error.message : "unknown error";
        patch((m) => ({ ...m, content: m.content || `Error: ${msg}` }));
      }
    } finally {
      setLoading(false);
      setStatus("");
      abortRef.current = null;
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div
      className="relative flex min-h-0 flex-1 flex-col"
      onDragEnter={onDragEnter}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {/* Header */}
      <header className="glass sticky top-0 z-10 flex h-16 shrink-0 items-center justify-between gap-4 border-b border-line px-4 pl-16 lg:px-6">
        <div className="min-w-0">
          <h1 className="truncate text-[15px] font-semibold tracking-tight text-ink">
            Conversation
          </h1>
          <p className="truncate text-xs text-ink-3">
            {kb.documents.length === 0
              ? "No documents indexed — answers will rely on web search"
              : `Grounded in ${kb.documents.length} document${kb.documents.length === 1 ? "" : "s"}`}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {loading && (
            <Pill tone="accent" dot pulse className="hidden sm:inline-flex">
              {status || "Working"}
            </Pill>
          )}
          <Button size="sm" variant="secondary" onClick={newConversation} disabled={isEmpty && !loading}>
            <PlusIcon className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">New chat</span>
          </Button>
        </div>
      </header>

      {/* Conversation */}
      <main className="min-h-0 flex-1 overflow-y-auto px-4 lg:px-6">
        <div className="mx-auto w-full max-w-3xl py-6">
          {isEmpty ? (
            <EmptyConversation
              onPick={(text) => send(undefined, text)}
              onUpload={() => fileInputRef.current?.click()}
              hasDocuments={kb.documents.length > 0}
            />
          ) : (
            <div className="space-y-6">
              {messages.map((m, i) => (
                <div key={i} className="animate-fade-up">
                  {m.role === "user" ? (
                    <div className="flex justify-end">
                      <div
                        className="max-w-[85%] rounded-2xl rounded-br-md px-4 py-2.5 text-[14px] leading-relaxed whitespace-pre-wrap"
                        style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
                      >
                        {m.content}
                      </div>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <span
                        className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg"
                        style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                      >
                        <SparkIcon className="w-4 h-4" />
                      </span>

                      <div className="min-w-0 flex-1">
                        <div className="rounded-2xl rounded-tl-md border border-line bg-surface px-4 py-3.5 shadow-xs">
                          {m.todos && m.todos.length > 0 && <PlanPanel todos={m.todos} />}
                          {m.trace && m.trace.length > 0 && <TracePanel trace={m.trace} />}

                          {m.content ? (
                            <SimpleMarkdown content={m.content} />
                          ) : (
                            loading &&
                            i === messages.length - 1 && <TypingIndicator label={status} />
                          )}

                          {m.blocked && (
                            <Pill tone="danger" className="mt-3">
                              <ShieldIcon className="w-3 h-3" />
                              Blocked by input guardrails
                            </Pill>
                          )}

                          {m.groundedness && <GroundednessBadge g={m.groundedness} />}
                          {(m.sources || m.web) && (
                            <SourcesPanel sources={m.sources ?? []} web={m.web ?? []} />
                          )}
                        </div>

                        {m.content && !loading && (
                          <div className="mt-1 flex items-center gap-0.5 pl-1">
                            <CopyButton text={m.content} />
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          <div ref={endRef} className="h-2" />
        </div>
      </main>

      {/* Composer */}
      <footer className="glass shrink-0 border-t border-line px-4 py-3 lg:px-6">
        <form onSubmit={send} className="mx-auto w-full max-w-3xl">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_UPLOADS}
            onChange={handleUpload}
            className="hidden"
          />

          <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-1.5 shadow-sm transition-colors focus-within:border-[var(--accent)]">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={kb.uploading || loading}
              title="Attach a document"
              aria-label="Attach a document"
              className="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink disabled:opacity-40"
            >
              {kb.uploading ? (
                <span
                  className="h-3.5 w-3.5 rounded-full border-2 border-current border-t-transparent animate-spin-slow"
                  aria-hidden="true"
                />
              ) : (
                <PaperclipIcon className="w-4 h-4" />
              )}
            </button>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={1}
              placeholder="Ask a question about your documents…"
              aria-label="Message"
              className="min-h-9 flex-1 resize-none bg-transparent px-1 py-2 text-[14px] leading-relaxed text-ink outline-none placeholder:text-ink-3"
            />

            {loading ? (
              <Button type="button" variant="secondary" onClick={stop} className="shrink-0">
                <StopIcon className="w-3.5 h-3.5" />
                Stop
              </Button>
            ) : (
              <Button type="submit" variant="primary" disabled={!input.trim()} className="shrink-0">
                <SendIcon className="w-4 h-4" />
                <span className="hidden sm:inline">Send</span>
              </Button>
            )}
          </div>

          <p className="mt-2 px-1 text-center text-[11px] text-ink-3">
            <kbd className="font-mono">Enter</kbd> to send ·{" "}
            <kbd className="font-mono">Shift + Enter</kbd> for a new line · drop a file anywhere to
            index it
          </p>
        </form>
      </footer>

      {/* Drop overlay */}
      {dragging && (
        <div className="pointer-events-none absolute inset-0 z-20 grid place-items-center bg-bg/85 animate-fade-in">
          <div
            className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-10 py-8"
            style={{ borderColor: "var(--accent)", background: "var(--accent-soft)" }}
          >
            <UploadIcon className="w-7 h-7" style={{ color: "var(--accent)" }} />
            <p className="text-sm font-medium text-ink">Drop to index</p>
            <p className="text-xs text-ink-3">PDF, DOCX, TXT, Markdown or CSV</p>
          </div>
        </div>
      )}
    </div>
  );
}

function EmptyConversation({
  onPick,
  onUpload,
  hasDocuments,
}: {
  onPick: (text: string) => void;
  onUpload: () => void;
  hasDocuments: boolean;
}) {
  return (
    <div className="py-10 text-center animate-fade-up">
      <BrandMark className="mx-auto mb-5 h-12 w-12" />
      <h2 className="text-xl font-semibold tracking-tight text-ink">
        Ask anything about your documents
      </h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-ink-3">
        The agent plans multi-step questions, searches your corpus and the web, verifies the
        answer against what it retrieved, and cites every claim.
      </p>

      {!hasDocuments && (
        <div className="mt-6 flex justify-center">
          <Button variant="primary" onClick={onUpload}>
            <UploadIcon className="w-4 h-4" />
            Upload your first document
          </Button>
        </div>
      )}

      <div className="mx-auto mt-8 grid max-w-2xl gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, text }) => (
          <button
            key={text}
            type="button"
            onClick={() => onPick(text)}
            className="group flex items-start gap-2.5 rounded-xl border border-line bg-surface p-3 text-left transition-all hover:border-[var(--line-strong)] hover:shadow-sm"
          >
            <span
              className="mt-px grid h-6 w-6 shrink-0 place-items-center rounded-md transition-colors"
              style={{ background: "var(--surface-muted)", color: "var(--ink-3)" }}
            >
              <Icon className="w-3.5 h-3.5" />
            </span>
            <span className="text-[13px] leading-snug text-ink-2 group-hover:text-ink">{text}</span>
          </button>
        ))}
      </div>

      <div className="mt-8 flex items-center justify-center gap-2 text-[11px] text-ink-3">
        <DocumentIcon className="w-3.5 h-3.5" />
        Hybrid retrieval · reranking · groundedness check
      </div>
    </div>
  );
}
