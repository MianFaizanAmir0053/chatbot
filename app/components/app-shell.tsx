"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ACCEPTED_UPLOADS, useKnowledge } from "./knowledge-context";
import { ThemeToggle } from "./theme-toggle";
import { Button, Pill, compactNumber } from "./ui";
import {
  BrandMark,
  ChatIcon,
  CloseIcon,
  DashboardIcon,
  DatabaseIcon,
  DocumentIcon,
  MenuIcon,
  TrashIcon,
  UploadIcon,
} from "./icons";

const NAV = [
  { href: "/", label: "Chat", icon: ChatIcon, description: "Ask your corpus" },
  { href: "/dashboard", label: "Dashboard", icon: DashboardIcon, description: "System health" },
];

/**
 * Application chrome: a persistent sidebar on desktop, a dismissible drawer
 * below `lg`. The knowledge library lives here so it is reachable from every
 * view instead of being hidden behind a header popover.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      {/* Mobile scrim */}
      {drawerOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/45 lg:hidden animate-fade-in"
          onClick={() => setDrawerOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[268px] flex-col border-r border-line bg-surface
                    transition-transform duration-200 ease-out lg:static lg:translate-x-0
                    ${drawerOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <SidebarContent onNavigate={() => setDrawerOpen(false)} />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          aria-label="Open navigation"
          className="lg:hidden fixed left-3 top-3 z-20 grid h-9 w-9 place-items-center rounded-lg border border-line bg-surface text-ink-2 shadow-sm"
        >
          <MenuIcon className="w-4.5 h-4.5" />
        </button>
        {children}
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate: () => void }) {
  const pathname = usePathname();
  const kb = useKnowledge();
  const fileRef = useRef<HTMLInputElement>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const result = await kb.upload(file);
    setNotice(
      result.status === "ingested"
        ? `Indexed ${result.file} — ${result.chunks} chunks`
        : result.status === "skipped"
          ? `${result.file} is already indexed`
          : (result.error ?? "Upload failed"),
    );
    window.setTimeout(() => setNotice(null), 5000);
  }

  return (
    <>
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 h-16 border-b border-line shrink-0">
        <BrandMark className="w-9 h-9 shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold tracking-tight text-ink leading-tight">
            Agentic RAG
          </div>
          <div className="text-[11px] text-ink-3 leading-tight truncate">
            Retrieval workspace
          </div>
        </div>
        <button
          type="button"
          onClick={onNavigate}
          aria-label="Close navigation"
          className="lg:hidden grid h-8 w-8 place-items-center rounded-lg text-ink-3 hover:bg-surface-hover"
        >
          <CloseIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Primary navigation */}
      <nav className="p-3 space-y-1 shrink-0">
        {NAV.map(({ href, label, icon: Icon, description }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={`group flex items-center gap-3 rounded-lg px-2.5 py-2 transition-colors ${
                active ? "text-ink" : "text-ink-2 hover:bg-surface-hover hover:text-ink"
              }`}
              style={active ? { background: "var(--accent-soft)" } : undefined}
            >
              <Icon
                className="w-4 h-4 shrink-0"
                {...(active ? { strokeWidth: 2 } : {})}
              />
              <span className="min-w-0 flex-1">
                <span
                  className="block text-[13px] font-medium leading-tight"
                  style={active ? { color: "var(--accent)" } : undefined}
                >
                  {label}
                </span>
                <span className="block text-[11px] text-ink-3 leading-tight truncate">
                  {description}
                </span>
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Knowledge base */}
      <div className="flex min-h-0 flex-1 flex-col border-t border-line">
        <div className="flex items-center justify-between gap-2 px-4 pt-3.5 pb-2 shrink-0">
          <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-3">
            Knowledge base
          </span>
          <span className="tnum text-[11px] text-ink-3">
            {kb.documents.length} · {compactNumber(kb.totalChunks)} chunks
          </span>
        </div>

        <div className="px-3 pb-2 shrink-0">
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED_UPLOADS}
            onChange={onPick}
            className="hidden"
          />
          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            disabled={kb.uploading}
            onClick={() => fileRef.current?.click()}
          >
            {kb.uploading ? (
              <>
                <span
                  className="w-3.5 h-3.5 rounded-full border-2 border-current border-t-transparent animate-spin-slow"
                  aria-hidden="true"
                />
                <span className="truncate">Indexing {kb.uploadingName}</span>
              </>
            ) : (
              <>
                <UploadIcon className="w-3.5 h-3.5" />
                Add document
              </>
            )}
          </Button>
          {notice && (
            <p className="mt-2 text-[11px] leading-snug text-ink-3 animate-fade-in">{notice}</p>
          )}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {kb.loading ? (
            <div className="space-y-1.5 px-2 pt-1">
              {[0, 1, 2].map((i) => (
                <div key={i} className="shimmer h-9 rounded-lg" />
              ))}
            </div>
          ) : kb.documents.length === 0 ? (
            <p className="px-2 py-3 text-[11px] leading-relaxed text-ink-3">
              Nothing indexed yet. Add a PDF, DOCX, Markdown or CSV file to start asking
              grounded questions.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {kb.documents.map((doc) => (
                <li key={doc.source}>
                  <div className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-surface-hover transition-colors">
                    <DocumentIcon className="w-3.5 h-3.5 shrink-0 text-ink-3" />
                    <span className="min-w-0 flex-1">
                      <span
                        className="block truncate text-[12px] text-ink-2 leading-tight"
                        title={doc.source}
                      >
                        {doc.source}
                      </span>
                      <span className="tnum block text-[10px] text-ink-3 leading-tight">
                        {doc.chunks} chunk{doc.chunks === 1 ? "" : "s"}
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={() => kb.remove(doc.source)}
                      title={`Remove ${doc.source}`}
                      aria-label={`Remove ${doc.source}`}
                      className="shrink-0 grid h-6 w-6 place-items-center rounded-md text-ink-3 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 hover:text-[var(--danger)]"
                    >
                      <TrashIcon className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Store status */}
      <div className="border-t border-line p-3 shrink-0">
        <div className="flex items-center justify-between gap-2 rounded-lg bg-surface-muted px-2.5 py-2">
          <span className="flex min-w-0 items-center gap-2">
            <DatabaseIcon className="w-3.5 h-3.5 shrink-0 text-ink-3" />
            <span className="min-w-0">
              <span className="block text-[11px] font-medium text-ink-2 leading-tight truncate">
                {kb.driver ?? "connecting…"}
              </span>
              <span className="block text-[10px] text-ink-3 leading-tight">
                {kb.persistent ? "Persistent store" : "In-memory — resets on restart"}
              </span>
            </span>
          </span>
          <Pill tone={kb.persistent ? "success" : "warn"} dot>
            {kb.persistent ? "Live" : "Volatile"}
          </Pill>
        </div>
        <div className="mt-2 flex items-center justify-between px-0.5">
          <span className="text-[10px] text-ink-3">v0.1.0</span>
          <ThemeToggle />
        </div>
      </div>
    </>
  );
}
