"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useConversations } from "./components/conversations-context";
import { DocumentsDialog, DocumentsTrigger } from "./components/documents-dialog";
import { ACCEPTED_UPLOADS, useKnowledge } from "./components/knowledge-context";
import { ThemeToggle } from "./components/theme-toggle";
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
  PlusIcon,
  SearchIcon,
  SendIcon,
  ShieldIcon,
  NetworkIcon,
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

/**
 * One delegated research branch.
 *
 * Tracked separately from the trace rather than as another trace line because a
 * delegation is the only step that is still running while later steps appear.
 * Several start at once and finish out of order, so they need a live per-branch
 * state; a trace is an append-only log and cannot show that.
 */
type Delegation = {
  id: string;
  agent: string;
  task: string;
  done: boolean;
};

const AGENT_LABELS: Record<string, string> = {
  "document-researcher": "Documents",
  "web-researcher": "Web",
  verifier: "Verifying",
};

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
  delegations?: Delegation[];
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

/** Opening prompts. The hint says which sources each one will actually reach. */
const SUGGESTIONS = [
  { icon: ListIcon, text: "Summarise the key findings across my documents", hint: "Whole corpus" },
  {
    icon: SearchIcon,
    text: "What does the documentation say about rate limits?",
    hint: "Targeted retrieval",
  },
  {
    icon: SparkIcon,
    text: "Compare the conclusions of the two most recent uploads",
    hint: "Two most recent documents",
  },
  {
    icon: GlobeIcon,
    text: "Check my documents against current public guidance",
    hint: "Corpus · web search",
  },
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
      <span className="flex items-end gap-1">
        {[0, 140, 280].map((delay) => (
          <span
            key={delay}
            className="w-1.5 h-1.5 rounded-full animate-wave"
            style={{ background: "var(--accent-color)", animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
      {label && (
        // Keyed on the label so each new phase of the run announces itself
        // with a small movement instead of silently swapping words.
        <span key={label} className="animate-slide-in-left text-xs text-muted-foreground">
          {label}
        </span>
      )}
    </div>
  );
}

/** A keycap. Small enough to sit inside a hint line without shouting. */
function Key({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-border bg-secondary px-1 py-px font-mono text-[11px] font-medium text-foreground">
      {children}
    </kbd>
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
      className="press inline-flex h-7 items-center gap-1.5 rounded-md px-1.5 text-[11px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
    >
      {done ? (
        <>
          <CheckIcon
            key="done"
            className="w-3.5 h-3.5 animate-pop"
            style={{ color: "var(--success)" }}
          />
          <span className="animate-fade-in" style={{ color: "var(--success)" }}>
            Copied
          </span>
        </>
      ) : (
        <>
          <CopyIcon className="w-3.5 h-3.5" />
          <span className="sr-only">Copy answer</span>
        </>
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
    <div className="mb-3 rounded-md border border-border bg-secondary p-3">
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <span className="eyebrow font-semibold">
          Plan
        </span>
        <span className="tnum text-[11px] text-muted-foreground">
          {done}/{todos.length}
        </span>
      </div>
      <div
        className="mb-3 h-1 w-full overflow-hidden rounded-full"
        style={{ background: "var(--accent)" }}
      >
        <div
          className="h-full rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.32,0.72,0,1)]"
          style={{ width: `${pct}%`, background: "var(--accent-color)" }}
        />
      </div>
      <ul className="space-y-1.5">
        {todos.map((todo, i) => (
          <li key={i} className="flex items-start gap-2 text-[13px]">
            <span className="mt-[3px] shrink-0">
              {todo.status === "completed" ? (
                <CheckIcon className="w-3.5 h-3.5 animate-pop" style={{ color: "var(--success)" }} />
              ) : todo.status === "in_progress" ? (
                <span
                  className="block h-3 w-3 rounded-full border-2 border-current border-t-transparent animate-spin-slow"
                  style={{ color: "var(--accent-color)" }}
                />
              ) : (
                <span
                  className="block h-3 w-3 rounded-full border-2"
                  style={{ borderColor: "var(--input)" }}
                />
              )}
            </span>
            <span className={`transition-colors duration-300 ${todo.status === "completed" ? "text-muted-foreground line-through" : "text-foreground"}`}>
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
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 -ml-1.5 text-[11px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
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
        <ol className="mt-2 space-y-1.5 border-l border-border pl-3.5 ml-1 stagger animate-fade-in">
          {trace.map((entry, i) => (
            <li key={i} className="relative text-[11px] leading-relaxed text-muted-foreground">
              <span
                className="absolute -left-[18px] top-[5px] h-1.5 w-1.5 rounded-full"
                style={{
                  background:
                    entry.kind === "warning"
                      ? "var(--warn)"
                      : entry.kind === "tool"
                        ? "var(--accent-color)"
                        : "var(--line-strong)",
                }}
              />
              <span className="text-foreground">{entry.label}</span>
              {entry.detail && <span className="text-muted-foreground"> — {entry.detail}</span>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * Live view of the delegated research branches.
 *
 * Open by default while anything is still running and collapsed once every
 * branch has reported, because its value is entirely in the waiting: delegated
 * mode is the slowest setting, and a user who cannot see four researchers
 * working in parallel just experiences a long unexplained pause. Afterwards the
 * same information is only provenance, and the answer should have the space.
 *
 * Each row states the sub-question rather than the researcher's name alone —
 * "Documents" three times says nothing, while the three sub-questions show
 * exactly how the agent decomposed the problem, which is the part worth
 * reading and the part worth catching when it is wrong.
 */
function DelegationPanel({ delegations }: { delegations: Delegation[] }) {
  const running = delegations.filter((d) => !d.done).length;
  const [open, setOpen] = useState(true);

  if (delegations.length === 0) return null;

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 -ml-1.5 text-[11px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
      >
        <ChevronIcon
          className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        <NetworkIcon className="w-3.5 h-3.5" />
        {delegations.length} researcher{delegations.length === 1 ? "" : "s"}
        {running > 0 ? (
          <span style={{ color: "var(--accent-color)" }}>· {running} running</span>
        ) : (
          <span className="text-muted-foreground">· all reported</span>
        )}
      </button>

      {open && (
        <ul className="mt-2 space-y-1.5 border-l border-border pl-3.5 ml-1 stagger animate-fade-in">
          {delegations.map((d) => (
            <li key={d.id} className="relative text-[11px] leading-relaxed">
              <span
                className={`absolute -left-[18px] top-[5px] h-1.5 w-1.5 rounded-full ${
                  d.done ? "" : "animate-pulse"
                }`}
                style={{ background: d.done ? "var(--ok, var(--line-strong))" : "var(--accent-color)" }}
              />
              <span className="text-foreground font-medium">
                {AGENT_LABELS[d.agent] ?? d.agent}
              </span>
              {d.task && <span className="text-muted-foreground"> — {d.task}</span>}
            </li>
          ))}
        </ul>
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
    <div className="mt-3 border-t border-border pt-2.5">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 -ml-1.5 text-[11px] font-semibold transition-colors hover:bg-accent"
        style={{ color: "var(--accent-color)" }}
      >
        <ChevronIcon
          className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        {sources.length} document source{sources.length === 1 ? "" : "s"}
        {web.length > 0 && ` · ${web.length} web result${web.length === 1 ? "" : "s"}`}
      </button>

      {open && (
        <div className="mt-2.5 grid gap-2 stagger animate-fade-in">
          {sources.map((s) => (
            <article
              key={s.index}
              className="lift rounded-md border border-border bg-secondary p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className="tnum grid h-5 min-w-5 shrink-0 place-items-center rounded px-1 font-mono text-[11px] font-semibold"
                    style={{ background: "var(--accent-soft)", color: "var(--accent-color)" }}
                  >
                    {s.index}
                  </span>
                  <span className="truncate text-[12px] font-medium text-foreground" title={s.source}>
                    {s.source}
                  </span>
                </div>
                <span
                  className="tnum shrink-0 text-[11px] font-medium"
                  title="Relevance score"
                  style={{ color: "var(--muted-foreground)" }}
                >
                  {Math.round(s.score * 100)}%
                </span>
              </div>

              {(s.section || s.page) && (
                <div className="mt-1 text-[11px] text-muted-foreground">
                  {s.section}
                  {s.section && s.page ? " · " : ""}
                  {s.page ? `page ${s.page}` : ""}
                </div>
              )}

              <p className="mt-2 line-clamp-3 text-[12px] leading-relaxed text-muted-foreground">
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
              className="lift group rounded-md border border-border bg-secondary p-3 hover:bg-accent"
            >
              <div className="flex items-center gap-2">
                <GlobeIcon className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
                <span
                  className="truncate text-[12px] font-medium"
                  style={{ color: "var(--accent-color)" }}
                >
                  {w.title || w.url}
                </span>
                <LinkIcon className="w-3 h-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
              <div className="mt-1 truncate text-[11px] text-muted-foreground">{w.url}</div>
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

/** Keeps the address in step with the open thread, without a navigation. */
function syncAddress(id: string | null) {
  const url = id ? `/?c=${id}` : "/";
  if (window.location.pathname + window.location.search !== url) {
    window.history.replaceState(null, "", url);
  }
}

export default function ChatPage() {
  const kb = useKnowledge();
  const {
    activeId,
    setActiveId,
    isUnsaved,
    markSaved,
    load: loadConversation,
    refresh: refreshConversations,
  } = useConversations();
  /** The thread the view is currently showing, for change detection. */
  const threadRef = useRef<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [webSearch, setWebSearch] = useState(true);
  const [deepThinking, setDeepThinking] = useState(false);
  const [deepAgents, setDeepAgents] = useState(false);

  const [restoring, setRestoring] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);

  const endRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);

  // Whether the view is following the tail of the conversation. A long answer
  // streams for tens of seconds; scrolling up to re-read an earlier passage has
  // to survive the next token, so following stops the moment the user leaves
  // the bottom and resumes when they return.
  const [pinned, setPinned] = useState(true);

  function onScroll(e: React.UIEvent<HTMLElement>) {
    const el = e.currentTarget;
    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }

  const jumpToLatest = useCallback(() => {
    setPinned(true);
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (!pinned) return;
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status, pinned]);

  /**
   * Adopt the conversation named in the address on first paint.
   *
   * Read from `window.location` rather than `useSearchParams`, which would
   * force this prerendered page behind a Suspense boundary for a value only
   * needed once the page is interactive. `popstate` keeps Back working.
   */
  useEffect(() => {
    const adopt = () => {
      const id = new URLSearchParams(window.location.search).get("c");
      setActiveId(id);
    };
    adopt();
    window.addEventListener("popstate", adopt);
    return () => window.removeEventListener("popstate", adopt);
  }, [setActiveId]);

  /**
   * Load — or clear — the transcript whenever the selected conversation
   * changes, wherever the change came from: the address bar, the sidebar or
   * the dashboard's list.
   */
  useEffect(() => {
    let cancelled = false;

    if (!activeId) {
      // Only a switch away from a thread clears the view; the very first render
      // of a fresh chat has nothing to clear and must not wipe a run in flight.
      if (threadRef.current !== null) {
        abortRef.current?.abort();
        abortRef.current = null;
        setMessages([]);
        setThreadId(null);
        setLoading(false);
        setStatus("");
      }
      threadRef.current = null;
      syncAddress(null);
      return;
    }

    if (activeId === threadRef.current) return;

    abortRef.current?.abort();
    abortRef.current = null;
    threadRef.current = activeId;
    setThreadId(activeId);
    setLoading(false);
    setStatus("");
    syncAddress(activeId);

    // A conversation this client just minted — by attaching a document before
    // sending anything — has no stored transcript yet. Fetching it would 404 and
    // flash "restoring" over an empty chat, so the load is skipped rather than
    // its failure handled.
    if (isUnsaved(activeId)) {
      setMessages([]);
      setRestoring(false);
      return;
    }

    setRestoring(true);

    void loadConversation(activeId).then((stored) => {
      if (cancelled) return;
      setRestoring(false);
      if (!stored) {
        setMessages([]);
        return;
      }
      setMessages(
        stored.map((turn) => ({
          role: turn.role,
          content: turn.content,
          sources: turn.sources,
          web: turn.web,
          groundedness: turn.groundedness as Message["groundedness"],
          blocked: turn.blocked,
        })),
      );
      setPinned(true);
    });

    return () => {
      cancelled = true;
    };
  }, [activeId, loadConversation, isUnsaved]);

  /**
   * Two shortcuts, both for the hands already on the keyboard: "/" jumps to
   * the composer from anywhere in the page, and Escape stops a run in flight
   * without reaching for the button that replaced Send.
   */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement | null;
      const typing =
        el?.tagName === "INPUT" || el?.tagName === "TEXTAREA" || el?.isContentEditable;

      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        textareaRef.current?.focus();
        return;
      }

      if (e.key === "Escape" && abortRef.current) {
        e.preventDefault();
        stop();
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
    threadRef.current = null;
    setActiveId(null);
    syncAddress(null);
    setInput("");
    textareaRef.current?.focus();
  }

  async function send(e?: React.FormEvent, override?: string) {
    e?.preventDefault();
    const text = (override ?? input).trim();
    if (!text || loading) return;

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
        // The transcript is deliberately not sent. The server holds the
        // conversation itself — in the agent's checkpointer, and in the
        // conversation store for when that is gone — so replaying it from here
        // uploaded the whole chat on every turn to be validated and discarded.
        body: JSON.stringify({
          message: text,
          ...(threadId ? { threadId } : {}),
          mode: "agentic",
          webSearch,
          thinking: deepThinking ? "deep" : "standard",
          deepAgents,
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
            case "thread": {
              // The server names the thread on the first turn. Adopting it here
              // is what makes the conversation resumable: it goes into the
              // address, the sidebar's selection and the saved list at once.
              const id = String(data.threadId);
              // Stored server-side from this point, so reopening it later should
              // fetch the transcript rather than assume an empty chat.
              markSaved(id);
              setThreadId(id);
              if (threadRef.current !== id) {
                threadRef.current = id;
                setActiveId(id);
                syncAddress(id);
              }
              break;
            }

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

            case "delegation": {
              const agent = String(data.agent ?? "researcher");
              setStatus(`Researching in parallel · ${AGENT_LABELS[agent] ?? agent}`);
              patch((m) => ({
                ...m,
                delegations: [
                  ...(m.delegations ?? []),
                  {
                    id: String(data.id),
                    agent,
                    task: String(data.task ?? ""),
                    done: false,
                  },
                ],
              }));
              break;
            }

            case "delegation_result":
              // One batch resolves as a single tool result, so every branch it
              // started finishes at the same moment. Branch ids are prefixed
              // with the batch's own id, which is what pairs them up.
              patch((m) => ({
                ...m,
                delegations: (m.delegations ?? []).map((d) =>
                  d.id.startsWith(`${String(data.batchId)}:`) ? { ...d, done: true } : d,
                ),
              }));
              break;

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
      // The turn changed the thread's title, message count and status; the
      // sidebar and dashboard both read that list.
      void refreshConversations();
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
      <header className="glass sticky top-0 z-10 flex h-16 shrink-0 items-center justify-between gap-4 border-b border-border px-4 pl-16 lg:px-6">
        <div className="min-w-0">
          <div className="eyebrow">
            {isEmpty ? "New conversation" : `${messages.length} messages`}
          </div>
          <h1 className="truncate font-display text-[15px] font-semibold text-foreground">
            {kb.documents.length === 0
              ? "Grounded research — no corpus yet"
              : "Grounded research, cited to source"}
          </h1>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <div className="hidden items-center gap-3 eyebrow xl:flex">
            <span>
              Corpus{" "}
              <b className="font-semibold text-foreground">
                {kb.documents.length} doc{kb.documents.length === 1 ? "" : "s"}
              </b>
            </span>
            <span className="w-3 border-t border-border" />
            <span>
              Retrieval <b className="font-semibold text-foreground">Hybrid</b>
            </span>
            <span className="w-3 border-t border-border" />
            <span>
              Web <b className="font-semibold text-foreground">{webSearch ? "On" : "Off"}</b>
            </span>
          </div>
          {loading && (
            <Pill tone="accent" dot pulse className="hidden sm:inline-flex">
              {status || "Working"}
            </Pill>
          )}
          <Button
            size="sm"
            variant="secondary"
            onClick={newConversation}
            disabled={isEmpty && !loading}
          >
            <PlusIcon className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">New chat</span>
          </Button>
          <ThemeToggle />
        </div>
      </header>

      {/* Conversation */}
      <main
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto px-4 lg:px-6"
      >
        <div className="mx-auto w-full max-w-[72ch] py-8">
          {restoring ? (
            <div className="space-y-6" aria-busy="true" aria-label="Loading conversation">
              {[0, 1, 2].map((i) => (
                <div key={i} className="space-y-2">
                  <div className="shimmer h-3 w-24 rounded" />
                  <div className="shimmer h-4 w-full rounded" />
                  <div className="shimmer h-4 w-4/5 rounded" />
                </div>
              ))}
            </div>
          ) : isEmpty ? (
            <EmptyConversation
              onPick={(text) => send(undefined, text)}
              onUpload={() => fileInputRef.current?.click()}
              hasDocuments={kb.documents.length > 0}
            />
          ) : (
            <div className="space-y-8">
              {messages.map((m, i) => (
                <article key={i} className="animate-rise">
                  {m.role === "user" ? (
                    <div className="flex flex-col items-end">
                      <div className="eyebrow mb-1">You</div>
                      <div className="max-w-[85%] rounded-md bg-primary px-4 py-2.5 text-[14px] leading-relaxed text-primary-foreground whitespace-pre-wrap">
                        {m.content}
                      </div>
                    </div>
                  ) : (
                    <div
                      className="min-w-0"
                      // Only the answer still being written announces itself;
                      // marking every message live would replay the whole
                      // transcript on each token.
                      aria-live={loading && i === messages.length - 1 ? "polite" : undefined}
                      aria-busy={loading && i === messages.length - 1}
                    >
                      <div className="eyebrow mb-2 flex items-center gap-2">
                        <SparkIcon className="w-3 h-3" />
                        Agentic RAG · grounded response
                      </div>

                      {m.todos && m.todos.length > 0 && <PlanPanel todos={m.todos} />}
                      {m.delegations && m.delegations.length > 0 && (
                        <DelegationPanel delegations={m.delegations} />
                      )}
                      {m.trace && m.trace.length > 0 && <TracePanel trace={m.trace} />}

                      {m.content ? (
                        <SimpleMarkdown content={m.content} />
                      ) : (
                        loading && i === messages.length - 1 && <TypingIndicator label={status} />
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

                      {m.content && !loading && (
                        <div className="mt-2 flex items-center gap-0.5 -ml-1.5">
                          <CopyButton text={m.content} />
                        </div>
                      )}
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}
          <div ref={endRef} className="h-2" />
        </div>
      </main>

      {/* Follow-the-tail escape hatch, shown only when the view has drifted. */}
      {!isEmpty && !pinned && (
        <div className="relative z-10 h-0">
          <button
            type="button"
            onClick={jumpToLatest}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-[11px] font-medium text-foreground shadow-md transition-colors hover:bg-accent animate-slide-up"
          >
            <ChevronIcon className="w-3 h-3 rotate-90" />
            {loading ? "Answer still streaming" : "Jump to latest"}
          </button>
        </div>
      )}

      {/* Composer */}
      <footer className="glass shrink-0 border-t border-border px-4 py-3 lg:px-6">
        <form onSubmit={send} className="mx-auto w-full max-w-[72ch]">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_UPLOADS}
            onChange={handleUpload}
            className="hidden"
          />

          <div className="rounded-lg border border-border bg-secondary/60 transition-[border-color] duration-200 focus-within:border-ring">
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
              rows={2}
              placeholder="Ask a grounded question about your documents…"
              aria-label="Message"
              className="min-h-18 w-full resize-none bg-transparent px-3.5 pt-3 pb-1 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
            />

            <div className="flex items-center justify-between gap-3 px-2 pb-2">
              <div className="flex min-w-0 items-center gap-1">
                <DocumentsTrigger
                  onClick={() => setDocsOpen(true)}
                  count={kb.documents.length}
                  busy={kb.uploading}
                />
                {kb.documents.length === 0 && !kb.uploading && (
                  <span className="hidden text-[12px] text-muted-foreground sm:inline">
                    none attached
                  </span>
                )}
              </div>

              <div className="flex shrink-0 items-center gap-2.5">
                <span className="hidden eyebrow sm:block">
                  {deepAgents ? "Deep agents" : deepThinking ? "Deep research" : "Hybrid"}
                  {webSearch ? " · Web" : ""}
                </span>
                {loading ? (
                  <button
                    type="button"
                    onClick={stop}
                    title="Stop generating (Esc)"
                    aria-label="Stop generating"
                    className="press grid size-8 place-items-center rounded-md border border-border text-foreground hover:bg-accent"
                  >
                    <StopIcon className="w-3.5 h-3.5" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    title="Send (Enter)"
                    aria-label="Send message"
                    className="press grid size-8 place-items-center rounded-md bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40"
                  >
                    <SendIcon className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="mt-2.5 flex flex-wrap items-center gap-1.5 px-1">
            <ModeToggle
              active={webSearch}
              onClick={() => setWebSearch(!webSearch)}
              icon={<GlobeIcon className="w-3.5 h-3.5" />}
              label="Web search"
              title={
                webSearch
                  ? "The agent may search the web when your documents don't cover the question"
                  : "Documents only — the web tools are withheld entirely"
              }
            />
            <ModeToggle
              active={deepThinking}
              onClick={() => setDeepThinking(!deepThinking)}
              icon={<SparkIcon className="w-3.5 h-3.5" />}
              label="Deep research"
              title={
                deepThinking
                  ? "Decomposes the question, probes each part separately, searches for contradicting evidence, and reports coverage gaps. Slower and more thorough."
                  : "Standard depth — fastest, and enough for most single-fact questions"
              }
            />
            <ModeToggle
              active={deepAgents}
              onClick={() => setDeepAgents(!deepAgents)}
              icon={<NetworkIcon className="w-3.5 h-3.5" />}
              label="Deep agents"
              title={
                deepAgents
                  ? "Splits the question and hands each part to a specialist researcher with its own context, running them in parallel, then verifies the load-bearing claims before answering. The most thorough setting."
                  : "Single researcher — one context does all the work"
              }
            />
            <span className="ml-auto hidden items-center gap-1.5 text-[11px] text-muted-foreground sm:inline-flex">
              <Key>{loading ? "Esc" : "/"}</Key>
              {loading ? "to stop" : "to focus"}
              <span className="mx-0.5 h-3 w-px bg-border" />
              <Key>Enter</Key>
              to send
            </span>
          </div>

          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Every answer is checked against your indexed sources.
          </p>

          <p className="sr-only">
            Press slash to focus the composer, Enter to send, Shift + Enter for a new line, Escape
            to stop a running answer, or drop a file anywhere to index it.
          </p>
        </form>
      </footer>

      <DocumentsDialog open={docsOpen} onClose={() => setDocsOpen(false)} disabled={loading} />

      {/* Drop overlay */}
      {dragging && (
        <div className="pointer-events-none absolute inset-0 z-20 grid place-items-center bg-background/85 animate-fade-in">
          <div
            className="flex animate-zoom-in flex-col items-center gap-3 rounded-md border-2 border-dashed px-10 py-8"
            style={{ borderColor: "var(--accent-color)", background: "var(--accent-soft)" }}
          >
            <UploadIcon className="w-7 h-7" style={{ color: "var(--accent-color)" }} />
            <p className="text-sm font-medium text-foreground">Drop to index</p>
            <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, Markdown or CSV</p>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * A composer toggle.
 *
 * Rendered as a pressable pill rather than a checkbox because these change what
 * the next message costs and how long it takes, so the current state has to be
 * readable at a glance from the composer itself.
 */
function ModeToggle({
  active,
  onClick,
  icon,
  label,
  title,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  title: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
        active
          ? "border-transparent bg-primary text-primary-foreground"
          : "border-border text-muted-foreground hover:bg-accent hover:text-foreground"
      }`}
    >
      {icon}
      {label}
      <span className="sr-only">{active ? " (on)" : " (off)"}</span>
    </button>
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
    <div className="py-8">
      <div className="flex items-center gap-3 eyebrow">
        <span>Research session</span>
        <span className="flex-1 border-t border-border" />
        <span>{hasDocuments ? "Corpus ready" : "Corpus empty"}</span>
      </div>

      <div className="animate-rise">
        <BrandMark className="mt-10 h-8 w-8" />
        <h2 className="mt-5 text-balance font-display text-[clamp(2rem,4vw,2.5rem)] font-semibold leading-[1.08] text-foreground">
          What have the indexed papers concluded?
        </h2>
        <p className="mt-4 max-w-[56ch] text-[15px] leading-relaxed text-muted-foreground">
          Ask anything across your corpus. Every answer is sourced to the exact document and page
          that supports it — citations included.
        </p>
      </div>

      {!hasDocuments && (
        <div className="mt-6">
          <Button variant="primary" onClick={onUpload}>
            <UploadIcon className="w-4 h-4" />
            Upload your first document
          </Button>
        </div>
      )}

      <div className="stagger mt-9 grid grid-cols-1 gap-x-6 sm:grid-cols-2">
        {SUGGESTIONS.map(({ text, hint }, index) => (
          <button
            key={text}
            type="button"
            onClick={() => onPick(text)}
            className="group border-t border-border py-4 text-left transition-colors hover:border-foreground"
          >
            <span className="eyebrow block">0{index + 1}</span>
            <span className="mt-1.5 block text-[14px] font-medium leading-snug text-foreground">
              {text}
            </span>
            <span className="mt-1 block text-[12px] text-muted-foreground">{hint}</span>
          </button>
        ))}
      </div>

      <div className="mt-9 flex items-center gap-2 border-t border-border pt-4 text-[11px] text-muted-foreground">
        <DocumentIcon className="w-3.5 h-3.5" />
        Hybrid retrieval · reranking · groundedness check
      </div>
    </div>
  );
}
