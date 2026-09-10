"use client";

import React, { useEffect, useRef, useState } from "react";
import type { ConversationSummary } from "./conversations-context";
import { CheckIcon, ForkIcon, PencilIcon, TrashIcon } from "./icons";

/**
 * One conversation in the sidebar, with rename, fork and delete.
 *
 * Renaming happens inline rather than in a dialog. The title is the only handle
 * on a conversation and it is read in the list, so editing it in place keeps the
 * thing being renamed visible next to its neighbours — a modal hides exactly the
 * context that makes a good name obvious.
 *
 * Actions stay hidden until the row is hovered or focused, because the list is
 * long and three buttons per row would compete with the titles for attention.
 * They remain reachable by keyboard: `focus-within` reveals them, so tabbing
 * through the list is not a dead end.
 */
export function ConversationRow({
  chat,
  active,
  onOpen,
  onRename,
  onFork,
  onDelete,
  relativeTime,
}: {
  chat: ConversationSummary;
  active: boolean;
  onOpen: () => void;
  onRename: (title: string) => void;
  onFork: () => void;
  onDelete: () => void;
  relativeTime: (iso: string) => string;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(chat.title);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!editing) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [editing]);

  function commit() {
    const next = draft.trim();
    setEditing(false);
    // An unchanged or emptied title is a cancel, not a rename: sending it would
    // spend a request to set the value that is already there, and an empty
    // string would leave the row with no handle at all.
    if (next && next !== chat.title) onRename(next);
    else setDraft(chat.title);
  }

  if (editing) {
    return (
      <div className="flex h-9 items-center gap-1.5 rounded-md bg-accent pl-2.5 pr-1.5">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commit();
            }
            if (e.key === "Escape") {
              e.preventDefault();
              setDraft(chat.title);
              setEditing(false);
            }
          }}
          // Committing on blur as well as Enter, because clicking away is what
          // people do with an inline field and losing the edit would be worse
          // than saving something they can edit again.
          onBlur={commit}
          maxLength={200}
          aria-label={`Rename ${chat.title}`}
          className="min-w-0 flex-1 bg-transparent text-[13px] text-foreground outline-none"
        />
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={commit}
          title="Save name"
          aria-label="Save name"
          className="press grid h-6 w-6 shrink-0 place-items-center rounded-md text-muted-foreground hover:text-foreground"
        >
          <CheckIcon className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  return (
    <div
      className={`press group relative flex h-9 items-center gap-1 rounded-md pl-2.5 pr-1 focus-within:bg-accent ${
        active ? "bg-accent" : "hover:bg-accent/60"
      }`}
    >
      {active && (
        <span
          aria-hidden="true"
          className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full"
          style={{ background: "var(--accent-color)" }}
        />
      )}

      <button
        type="button"
        onClick={onOpen}
        onDoubleClick={() => setEditing(true)}
        className="flex min-w-0 flex-1 items-center gap-1.5 rounded-md focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        title={chat.forkedFrom ? `${chat.title} (forked)` : chat.title}
      >
        <span
          className={`min-w-0 flex-1 truncate text-left text-[13px] ${
            active ? "font-medium text-foreground" : "text-foreground/85"
          }`}
        >
          {chat.title}
        </span>
        {/* A fork is not obviously a fork from its title alone, and knowing it
            inherits another conversation's documents explains why it can answer
            from files that were never uploaded to it. */}
        {chat.forkedFrom && (
          <span
            className="shrink-0 rounded px-1 text-[10px] font-medium text-muted-foreground"
            style={{ background: "var(--accent)" }}
            title="Forked — inherits the parent's documents"
          >
            fork
          </span>
        )}
      </button>

      <span
        className="tnum shrink-0 pr-1 text-[11px] text-muted-foreground group-hover:hidden group-focus-within:hidden"
        title={`${chat.messages} message${chat.messages === 1 ? "" : "s"}${
          chat.forks > 0 ? ` · ${chat.forks} fork${chat.forks === 1 ? "" : "s"}` : ""
        }`}
      >
        {relativeTime(chat.updatedAt)}
      </span>

      <div className="hidden shrink-0 items-center gap-0.5 group-hover:flex group-focus-within:flex">
        <RowAction label={`Rename ${chat.title}`} onClick={() => setEditing(true)}>
          <PencilIcon className="w-3.5 h-3.5" />
        </RowAction>
        <RowAction label={`Fork ${chat.title}`} onClick={onFork}>
          <ForkIcon className="w-3.5 h-3.5" />
        </RowAction>
        <RowAction label={`Delete ${chat.title}`} onClick={onDelete} danger>
          <TrashIcon className="w-3.5 h-3.5" />
        </RowAction>
      </div>
    </div>
  );
}

function RowAction({
  label,
  onClick,
  danger = false,
  children,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`press grid h-6 w-6 place-items-center rounded-md text-muted-foreground ${
        danger ? "hover:text-[var(--danger)]" : "hover:text-foreground"
      }`}
    >
      {children}
    </button>
  );
}
