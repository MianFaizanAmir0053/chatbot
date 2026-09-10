"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ACCEPTED_UPLOADS, useKnowledge } from "./knowledge-context";
import { CloseIcon, DocumentIcon, PaperclipIcon, TrashIcon, UploadIcon } from "./icons";

/**
 * The documents attached to the open conversation.
 *
 * Lives in the composer rather than the sidebar because attaching a document is
 * part of asking a question, and because the list has to answer one question
 * precisely: what will this chat actually read? A corpus-wide list beside a chat
 * implies the agent will consult all of it, and since documents are scoped to a
 * conversation, it will not.
 *
 * A dialog rather than an always-visible panel. Documents matter at two moments
 * — adding one, and checking what is there — and neither is most of the time;
 * a permanent list spends vertical space the transcript needs on information
 * that is usually settled.
 */
export function DocumentsDialog({
  open,
  onClose,
  disabled,
}: {
  open: boolean;
  onClose: () => void;
  disabled?: boolean;
}) {
  const kb = useKnowledge();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const ingest = useCallback(
    async (files: FileList | File[] | null) => {
      const list = Array.from(files ?? []);
      if (list.length === 0) return;
      setNotice(null);

      // Sequential, because ingest embeds and reranks: several at once would
      // multiply the load on the one dependency a fan-out already contends for.
      for (const file of list) {
        const result = await kb.upload(file);
        if (!result.ok) {
          setNotice(result.error ?? `${result.file} could not be indexed`);
          return;
        }
        if (result.status === "skipped") {
          setNotice(`${result.file} is already attached to this chat`);
        }
      }
    },
    [kb],
  );

  // Escape closes, and focus moves into the dialog on open so the keyboard is
  // not left behind in the composer.
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const empty = kb.documents.length === 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-0 animate-fade-in sm:items-center sm:p-6"
      role="presentation"
      // Only a click on the backdrop itself closes; one that started inside the
      // panel and drifted out must not, or selecting text dismisses the dialog.
      onMouseDown={(e) => {
        if (!panelRef.current?.contains(e.target as Node)) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Documents in this chat"
        className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-t-xl border border-border bg-background shadow-2xl sm:rounded-xl"
      >
        <header className="flex shrink-0 items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 className="text-[14px] font-semibold text-foreground">Documents in this chat</h2>
            <p className="mt-0.5 text-[12px] leading-relaxed text-muted-foreground">
              {empty
                ? "Nothing attached yet. Only what you add here is searched."
                : `${kb.documents.length} document${kb.documents.length === 1 ? "" : "s"} · ${kb.totalChunks} passage${kb.totalChunks === 1 ? "" : "s"} searchable`}
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="press grid size-7 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </header>

        <div
          className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void ingest(e.dataTransfer?.files ?? null);
          }}
        >
          {empty ? (
            <div
              className={`grid place-items-center gap-2 rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors ${
                dragging ? "border-ring bg-accent" : "border-border"
              }`}
            >
              <DocumentIcon className="w-6 h-6 text-muted-foreground" />
              <p className="text-[13px] font-medium text-foreground">No documents yet</p>
              <p className="max-w-[36ch] text-[12px] leading-relaxed text-muted-foreground">
                Drop a file here, or use Add below. Documents you attach are searchable from this
                chat only.
              </p>
            </div>
          ) : (
            <ul className="space-y-1.5">
              {kb.documents.map((doc) => (
                <li
                  key={doc.source}
                  className="group flex items-center gap-2.5 rounded-md border border-border bg-secondary/50 px-2.5 py-2"
                >
                  <DocumentIcon className="w-4 h-4 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate text-[13px] text-foreground" title={doc.source}>
                    {doc.source}
                  </span>
                  <span className="tnum shrink-0 text-[11px] text-muted-foreground">
                    {doc.chunks} passage{doc.chunks === 1 ? "" : "s"}
                  </span>
                  <button
                    type="button"
                    onClick={() => void kb.remove(doc.source)}
                    title={`Remove ${doc.source} from this chat`}
                    aria-label={`Remove ${doc.source} from this chat`}
                    className="press grid size-6 shrink-0 place-items-center rounded text-muted-foreground opacity-0 transition-opacity hover:bg-accent focus-visible:opacity-100 group-hover:opacity-100"
                    style={{ color: "var(--danger, currentColor)" }}
                  >
                    <TrashIcon className="w-3.5 h-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {notice && (
            <p
              className="mt-3 rounded-md px-2.5 py-2 text-[12px] leading-relaxed"
              style={{ background: "var(--warn-soft)", color: "var(--warn)" }}
            >
              {notice}
            </p>
          )}
        </div>

        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border px-4 py-3">
          <p className="text-[11px] text-muted-foreground">PDF, DOCX, TXT, MD or CSV · 25MB</p>
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED_UPLOADS}
              onChange={(e) => {
                void ingest(e.target.files);
                // Cleared so re-picking the same file fires change again.
                e.target.value = "";
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={kb.uploading || disabled}
              className="press inline-flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-[12px] font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
            >
              {kb.uploading ? (
                <>
                  <span
                    className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-current border-t-transparent"
                    aria-hidden="true"
                  />
                  Indexing{kb.uploadingName ? ` ${kb.uploadingName}` : ""}
                </>
              ) : (
                <>
                  <UploadIcon className="w-3.5 h-3.5" />
                  Add documents
                </>
              )}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

/**
 * The composer's entry point to the dialog.
 *
 * Shows the count on the button so the common question — is anything attached?
 * — is answered without opening anything.
 */
export function DocumentsTrigger({
  onClick,
  count,
  busy,
  disabled,
}: {
  onClick: () => void;
  count: number;
  busy?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-haspopup="dialog"
      title={
        count === 0
          ? "Attach documents to this chat"
          : `${count} document${count === 1 ? "" : "s"} in this chat`
      }
      className="press inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-[12px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-40"
    >
      {busy ? (
        <span
          className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-current border-t-transparent"
          aria-hidden="true"
        />
      ) : (
        <PaperclipIcon className="w-4 h-4" />
      )}
      <span className="hidden sm:inline">{busy ? "Indexing" : "Documents"}</span>
      {count > 0 && !busy && (
        <span
          className="tnum grid h-4 min-w-4 place-items-center rounded-full px-1 text-[10px] font-semibold"
          style={{ background: "var(--accent-soft)", color: "var(--accent-color)" }}
        >
          {count}
        </span>
      )}
    </button>
  );
}
