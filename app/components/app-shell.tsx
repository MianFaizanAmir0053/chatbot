"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ConversationRow } from "./conversation-row";
import { useConversations } from "./conversations-context";
import { ACCEPTED_UPLOADS, useKnowledge } from "./knowledge-context";
import { compactNumber, relativeTime, type Tone } from "./ui";
import {
  BrandMark,
  ChatIcon,
  CloseIcon,
  DashboardIcon,
  MenuIcon,
  PlusIcon,
  TrashIcon,
  UploadIcon,
} from "./icons";

const NAV = [
  { href: "/", label: "Chat", icon: ChatIcon },
  { href: "/dashboard", label: "Dashboard", icon: DashboardIcon },
];

/**
 * Application chrome: a sidebar that collapses on desktop and slides away as a
 * drawer below `lg`. One control drives both, so the header always has the
 * same affordance in the same place.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(true);

  // Start collapsed on small screens; the drawer overlays content there.
  useEffect(() => {
    const media = window.matchMedia("(max-width: 1023px)");
    const sync = (e: MediaQueryList | MediaQueryListEvent) => setOpen(!e.matches);
    sync(media);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="flex h-dvh overflow-hidden bg-background text-foreground">
      {open && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
          className="fixed inset-0 z-30 animate-fade-in bg-foreground/25 backdrop-blur-[1px] lg:hidden"
        />
      )}

      <aside
        className={`${
          open ? "w-66 max-lg:translate-x-0" : "w-0 max-lg:-translate-x-full"
        } fixed inset-y-0 left-0 z-40 flex shrink-0 flex-col overflow-hidden border-r border-border bg-sidebar transition-[width,transform] duration-200 ease-out max-lg:w-66 lg:static`}
      >
        <SidebarContent onNavigate={() => window.innerWidth < 1024 && setOpen(false)} />
      </aside>

      {/* Positioned relative to the content column, so the toggle sits at the
          left edge of the header whether the sidebar is open or collapsed. */}
      <div className="relative flex min-w-0 flex-1 flex-col">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
          aria-expanded={open}
          className="press absolute left-3 top-3.5 z-20 grid h-9 w-9 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          {open ? <CloseIcon className="w-4 h-4" /> : <MenuIcon className="w-4.5 h-4.5" />}
        </button>
        {children}
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const kb = useKnowledge();
  const conversations = useConversations();
  const fileRef = useRef<HTMLInputElement>(null);
  // A failed upload and a successful one used to read identically. The tone
  // travels with the text so the difference is visible before it is read.
  const [notice, setNotice] = useState<{ text: string; tone: Tone } | null>(null);
  const [query, setQuery] = useState("");

  /**
   * Opens a saved thread — or a blank one — in the chat view.
   *
   * The selection is published through context rather than carried in the
   * link, because a link to `/?c=<id>` from the chat view is a navigation to
   * the page that is already mounted: it fires no popstate and the view would
   * keep showing the previous thread.
   */
  const openChat = (id: string | null) => {
    conversations.setActiveId(id);
    if (pathname !== "/") router.push("/");
    onNavigate();
  };

  /**
   * Fork a conversation and open the copy.
   *
   * Opening it is the point — a fork the user has to go and find reads as though
   * nothing happened. The new thread inherits the original's documents, so it is
   * immediately answerable rather than an empty chat.
   */
  async function forkChat(id: string) {
    const forkId = await conversations.fork(id);
    if (!forkId) {
      setNotice({ text: "Could not fork that conversation", tone: "danger" });
      window.setTimeout(() => setNotice(null), 6000);
      return;
    }
    openChat(forkId);
    setNotice({ text: "Forked — it shares the original's documents", tone: "success" });
    window.setTimeout(() => setNotice(null), 4000);
  }

  /**
   * Delete a conversation, its transcript and the documents it owns.
   *
   * Confirmed rather than undoable, and the prompt names the consequences the
   * user cannot see: the embeddings go too, and any fork of this conversation
   * loses the documents it was reading through it. Both are irreversible, and
   * neither is guessable from a bare "delete?".
   */
  async function deleteChat(chat: { id: string; title: string; forks: number }) {
    const consequences = [
      `Delete “${chat.title}”?`,
      "",
      "This removes the transcript and the documents uploaded to it, including their embeddings.",
      chat.forks > 0
        ? `${chat.forks} conversation${chat.forks === 1 ? "" : "s"} forked from this one will lose access to those documents.`
        : "",
      "This cannot be undone.",
    ]
      .filter(Boolean)
      .join("\n");

    if (!window.confirm(consequences)) return;

    const { chunksRemoved } = await conversations.remove(chat.id);
    // Refreshed because the document list is scoped to the open conversation,
    // and deleting the open one changes what that list should show.
    void kb.refresh();
    setNotice({
      text:
        chunksRemoved > 0
          ? `Deleted “${chat.title}” and ${chunksRemoved} indexed chunk${chunksRemoved === 1 ? "" : "s"}`
          : `Deleted “${chat.title}”`,
      tone: "neutral",
    });
    window.setTimeout(() => setNotice(null), 4000);
  }

  // The filter only earns its space once scanning the list stops being
  // instant; below that it is a control that never gets used.
  const searchable = kb.documents.length > 5;
  const visible = searchable
    ? kb.documents.filter((doc) => doc.source.toLowerCase().includes(query.trim().toLowerCase()))
    : kb.documents;

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const result = await kb.upload(file);
    setNotice(
      result.status === "ingested"
        ? { text: `Indexed ${result.file} — ${result.chunks} chunks`, tone: "success" }
        : result.status === "skipped"
          ? { text: `${result.file} is already indexed`, tone: "neutral" }
          : { text: result.error ?? "Upload failed", tone: "danger" },
    );
    // An error stays long enough to be read and acted on; a confirmation does not.
    window.setTimeout(() => setNotice(null), result.status === "failed" ? 9000 : 5000);
  }

  return (
    <div className="flex h-full min-w-66 flex-col">
      {/* Brand */}
      <div className="flex h-16 shrink-0 items-center gap-2.5 px-4">
        <BrandMark className="w-7 h-7 shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="truncate font-display text-[13px] font-semibold leading-tight text-foreground">
            Agentic RAG
          </div>
          <div className="mt-0.5 truncate text-[11px] leading-tight text-muted-foreground">
            Retrieval workspace
          </div>
        </div>
      </div>

      {/* Primary navigation */}
      <nav className="shrink-0 px-2" aria-label="Primary">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={`press relative flex h-9 items-center gap-2.5 rounded-md px-2.5 text-[13px] ${
                active
                  ? "bg-accent font-medium text-foreground"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
              }`}
            >
              {active && <ActiveBar />}
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Conversations */}
      <SideSection
        label="Conversations"
        action={
          <button
            type="button"
            onClick={() => openChat(null)}
            title="Start a new conversation"
            aria-label="Start a new conversation"
            className="press grid h-6 w-6 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <PlusIcon className="w-3.5 h-3.5" />
          </button>
        }
      />

      <div className="max-h-52 min-h-0 shrink-0 overflow-y-auto px-2">
        {conversations.loading ? (
          <ListSkeleton rows={2} />
        ) : conversations.conversations.length === 0 ? (
          <EmptyLine>No conversations yet.</EmptyLine>
        ) : (
          conversations.conversations.slice(0, 20).map((chat) => (
            <ConversationRow
              key={chat.id}
              chat={chat}
              active={chat.id === conversations.activeId}
              relativeTime={relativeTime}
              onOpen={() => openChat(chat.id)}
              onRename={(title) => void conversations.rename(chat.id, title)}
              onFork={() => void forkChat(chat.id)}
              onDelete={() => void deleteChat(chat)}
            />
          ))
        )}
      </div>

      {/* Knowledge base */}
      <SideSection
        label="Documents"
        action={
          <span className="tnum text-[11px] text-muted-foreground">{kb.documents.length}</span>
        }
      />

      {searchable && (
        <div className="shrink-0 px-2 pb-1">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter…"
            aria-label="Filter indexed documents"
            className="h-7 w-full rounded-md bg-foreground/5 px-2.5 text-[12px] text-foreground outline-none transition-colors placeholder:text-muted-foreground focus:bg-foreground/10"
          />
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-2">
        {kb.loading ? (
          <ListSkeleton rows={3} />
        ) : kb.documents.length === 0 ? (
          <EmptyLine>Nothing indexed yet — add a PDF, DOCX, Markdown or CSV file.</EmptyLine>
        ) : visible.length === 0 ? (
          <EmptyLine>No match for “{query}”.</EmptyLine>
        ) : (
          visible.map((doc) => (
            <SideRow
              key={doc.source}
              label={doc.source}
              meta={String(doc.chunks)}
              metaTitle={`${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`}
              onRemove={() => kb.remove(doc.source)}
              removeLabel={`Remove ${doc.source}`}
            />
          ))
        )}
      </div>

      {/* Upload */}
      <div className="shrink-0 px-2 pt-2">
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPTED_UPLOADS}
          onChange={onPick}
          className="hidden"
        />
        <button
          type="button"
          disabled={kb.uploading}
          onClick={() => fileRef.current?.click()}
          className="press flex h-9 w-full items-center gap-2.5 rounded-md px-2.5 text-[13px] text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
        >
          {kb.uploading ? (
            <>
              <span
                className="h-3.5 w-3.5 shrink-0 animate-spin-slow rounded-full border-2 border-current border-t-transparent"
                aria-hidden="true"
              />
              <span className="truncate">Indexing {kb.uploadingName}</span>
            </>
          ) : (
            <>
              <UploadIcon className="w-4 h-4 shrink-0" />
              Add document
            </>
          )}
        </button>

        {notice && (
          <p
            role="status"
            className="mt-1.5 animate-slide-up rounded-md px-2.5 py-1.5 text-[12px] leading-snug"
            style={{
              background:
                notice.tone === "neutral" ? "var(--secondary)" : `var(--${notice.tone}-soft)`,
              color: notice.tone === "neutral" ? "var(--muted-foreground)" : `var(--${notice.tone})`,
            }}
          >
            {notice.text}
          </p>
        )}
      </div>

      {/* Store status */}
      <div className="mt-2 flex shrink-0 items-center gap-2 border-t border-border px-4 py-3">
        <span
          className="size-1.5 shrink-0 rounded-full"
          title={kb.persistent ? "Persistent store" : "Volatile store"}
          style={{ background: kb.persistent ? "var(--status)" : "var(--warning)" }}
        />
        <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">
          <span className="font-medium text-foreground">{kb.driver ?? "connecting…"}</span>
          {kb.driver ? (kb.persistent ? " · persistent" : " · in-memory") : null}
        </span>
        <span
          className="tnum shrink-0 text-[11px] text-muted-foreground"
          title={`${kb.totalChunks} indexed chunks`}
        >
          {compactNumber(kb.totalChunks)}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Sidebar pieces
 *
 * Rows are flat. A bordered card per document turned a five-item list
 * into five stacked panels — a lot of ink for something you scan rather
 * than read. Weight and a hover wash carry the structure instead, and
 * the borders are spent on the one divider that separates the chrome
 * from the workspace.
 * ------------------------------------------------------------------ */

/** Marks the selected row without adding a box around it. */
function ActiveBar() {
  return <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-foreground" />;
}

function SideSection({ label, action }: { label: string; action?: React.ReactNode }) {
  return (
    <div className="mt-5 mb-1 flex h-6 shrink-0 items-center justify-between gap-2 px-4">
      <span className="eyebrow">{label}</span>
      {action}
    </div>
  );
}

/**
 * One list row.
 *
 * The trailing meta and the remove control share a slot: the count or
 * timestamp is what you read, the delete is what you reach for, and
 * swapping them on hover keeps every row a single line either way.
 */
function SideRow({
  label,
  meta,
  metaTitle,
  active = false,
  onOpen,
  onRemove,
  removeLabel,
}: {
  label: string;
  meta: string;
  metaTitle?: string;
  active?: boolean;
  onOpen?: () => void;
  onRemove: () => void;
  removeLabel: string;
}) {
  const text = (
    <span
      className={`min-w-0 flex-1 truncate text-left text-[13px] ${
        active ? "font-medium text-foreground" : "text-foreground/85"
      }`}
      title={label}
    >
      {label}
    </span>
  );

  return (
    <div
      className={`press group relative flex h-9 items-center gap-2 rounded-md pl-2.5 pr-1.5 ${
        active ? "bg-accent" : "hover:bg-accent/60"
      }`}
    >
      {active && <ActiveBar />}
      {onOpen ? (
        <button
          type="button"
          onClick={onOpen}
          className="flex min-w-0 flex-1 items-center rounded-md focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {text}
        </button>
      ) : (
        <span className="flex min-w-0 flex-1 items-center">{text}</span>
      )}

      <span
        className="tnum shrink-0 text-[11px] text-muted-foreground group-hover:hidden"
        title={metaTitle}
      >
        {meta}
      </span>
      <button
        type="button"
        onClick={onRemove}
        title={removeLabel}
        aria-label={removeLabel}
        className="press hidden h-6 w-6 shrink-0 place-items-center rounded-md text-muted-foreground hover:text-[var(--danger)] focus-visible:grid group-hover:grid"
      >
        <TrashIcon className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function ListSkeleton({ rows }: { rows: number }) {
  return (
    <div className="space-y-1 py-1">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="shimmer h-7 rounded-md" />
      ))}
    </div>
  );
}

function EmptyLine({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-2.5 py-1.5 text-[12px] leading-relaxed text-muted-foreground">{children}</p>
  );
}
