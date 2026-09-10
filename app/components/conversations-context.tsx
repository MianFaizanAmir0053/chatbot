"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/**
 * The list of saved conversations, shared by the sidebar and the dashboard.
 *
 * Held in one place because both surfaces open, delete and re-title the same
 * threads: a conversation started in the chat view has to appear in the
 * dashboard's list without a reload, and deleting it from either has to remove
 * it from both.
 */

export type ConversationSummary = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: number;
  /** Set when this conversation was forked from another. */
  forkedFrom?: string;
  /** How many conversations were forked from this one. */
  forks: number;
  status: "completed" | "flagged" | "in_progress" | "blocked";
};

export type StoredSource = {
  index: number;
  source: string;
  section?: string;
  page?: number;
  score: number;
  excerpt: string;
};

export type StoredMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: StoredSource[];
  web?: Array<{ title: string; url: string }>;
  groundedness?: {
    score: number;
    verdict: string;
    passed: boolean;
    unsupportedClaims: string[];
  };
  blocked?: boolean;
  createdAt: string;
};

type ConversationsState = {
  conversations: ConversationSummary[];
  loading: boolean;
  error: string | null;
  /**
   * The thread the chat view currently has open.
   *
   * Held here rather than read from the URL by each surface: `useSearchParams`
   * forces a Suspense boundary around every prerendered page that calls it,
   * and the chat view is the only component that needs to parse the address —
   * it publishes the result for the sidebar to highlight.
   */
  activeId: string | null;
  setActiveId: (id: string | null) => void;
  /**
   * The active conversation's id, creating one if the chat is still fresh.
   *
   * A conversation normally gets its id from the first message, but a document
   * can be attached before anything is said — and that upload has to be filed
   * against a conversation or it lands in the shared corpus, visible from every
   * other chat. Minting the id here means the upload and the message that
   * follows agree on which conversation they belong to.
   *
   * Returns the id synchronously so a caller can pass it straight into the
   * request it is about to make; waiting for the state update would send the
   * upload unscoped.
   */
  ensureActiveId: () => string;
  /**
   * True for a conversation this client minted that the server has not stored.
   *
   * Attaching a document before saying anything creates the conversation id
   * locally, so the chat view would try to fetch a transcript that does not
   * exist — a 404 and a "restoring" flash on what is plainly a new, empty chat.
   * Asking first avoids the request rather than handling its failure.
   */
  isUnsaved: (id: string) => boolean;
  /** Called once the server has stored the conversation, so it can be loaded. */
  markSaved: (id: string) => void;
  refresh: () => Promise<void>;
  load: (id: string) => Promise<StoredMessage[] | null>;
  /**
   * Delete a conversation and the documents it owns.
   *
   * Resolves to how many chunks were purged, so the caller can say what
   * happened rather than leaving the user to guess whether their files went
   * with it.
   */
  remove: (id: string) => Promise<{ chunksRemoved: number }>;
  rename: (id: string, title: string) => Promise<boolean>;
  /**
   * Copy a conversation into a new one that inherits its documents.
   *
   * `upTo` truncates to the first N messages, so a user can branch from a point
   * mid-thread. Returns the new id so the caller can open it.
   */
  fork: (id: string, options?: { upTo?: number; title?: string }) => Promise<string | null>;
};

const ConversationsContext = createContext<ConversationsState | null>(null);

export function ConversationsProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  /**
   * Mirrors activeId for synchronous reads.
   *
   * ensureActiveId has to return an id in the same tick so the caller can put
   * it in the request it is making; React state is not readable that soon after
   * a set, so the ref is the value of record and the state exists to re-render.
   */
  const activeRef = useRef<string | null>(null);
  /**
   * Ids minted here that the server has not seen yet.
   *
   * A ref rather than state: it is read during the same tick an id is created
   * and never affects rendering on its own.
   */
  const unsavedRef = useRef<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/conversations", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setConversations(data.conversations ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to read conversations");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const load = useCallback(async (id: string): Promise<StoredMessage[] | null> => {
    try {
      const res = await fetch(`/api/conversations/${id}`, { cache: "no-store" });
      if (!res.ok) return null;
      const data = await res.json();
      return (data.conversation?.messages ?? []) as StoredMessage[];
    } catch {
      return null;
    }
  }, []);

  const ensureActiveId = useCallback((): string => {
    const current = activeRef.current;
    if (current) return current;

    const created =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    activeRef.current = created;
    unsavedRef.current.add(created);
    setActiveId(created);
    return created;
  }, []);

  const isUnsaved = useCallback((id: string) => unsavedRef.current.has(id), []);

  const markSaved = useCallback((id: string) => {
    unsavedRef.current.delete(id);
  }, []);

  const remove = useCallback(async (id: string): Promise<{ chunksRemoved: number }> => {
    // Removed locally first: the list is the only feedback that the delete
    // happened, and waiting a round trip to redraw it reads as a dead click.
    setConversations((current) => current.filter((c) => c.id !== id));
    if (activeRef.current === id) activeRef.current = null;
    setActiveId((current) => (current === id ? null : current));
    try {
      const res = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      return { chunksRemoved: Number(data.chunksRemoved ?? 0) };
    } catch {
      // The optimistic removal is corrected by the next refresh.
      return { chunksRemoved: 0 };
    }
  }, []);

  const rename = useCallback(async (id: string, title: string): Promise<boolean> => {
    const trimmed = title.trim();
    if (!trimmed) return false;

    // Shown immediately, then reconciled. A rename that redraws only after the
    // round trip reads as though the edit was rejected.
    setConversations((current) =>
      current.map((c) => (c.id === id ? { ...c, title: trimmed } : c)),
    );
    try {
      const res = await fetch(`/api/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: trimmed }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return true;
    } catch {
      // Put the server's version back rather than leaving a title that only
      // exists on this screen.
      void refresh();
      return false;
    }
  }, [refresh]);

  const fork = useCallback(
    async (id: string, options: { upTo?: number; title?: string } = {}) => {
      try {
        const res = await fetch(`/api/conversations/${id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(options),
        });
        if (!res.ok) return null;
        const data = await res.json();
        const forkId = data.conversation?.id as string | undefined;
        if (!forkId) return null;

        // Refreshed rather than inserted locally: the fork's summary carries
        // counts the server derives, and the parent's fork count has changed too.
        await refresh();
        return forkId;
      } catch {
        return null;
      }
    },
    [refresh],
  );

  // Kept in step with every path that changes the active conversation — the
  // address bar, the sidebar and the dashboard all call setActiveId, and a stale
  // ref would file an upload against the previous conversation.
  const selectActive = useCallback((id: string | null) => {
    activeRef.current = id;
    setActiveId(id);
  }, []);

  const value = useMemo(
    () => ({
      conversations,
      loading,
      error,
      activeId,
      setActiveId: selectActive,
      ensureActiveId,
      isUnsaved,
      markSaved,
      refresh,
      load,
      remove,
      rename,
      fork,
    }),
    [
      conversations,
      loading,
      error,
      activeId,
      selectActive,
      ensureActiveId,
      isUnsaved,
      markSaved,
      refresh,
      load,
      remove,
      rename,
      fork,
    ],
  );

  return <ConversationsContext.Provider value={value}>{children}</ConversationsContext.Provider>;
}

export function useConversations(): ConversationsState {
  const context = useContext(ConversationsContext);
  if (!context) throw new Error("useConversations must be used inside ConversationsProvider");
  return context;
}
