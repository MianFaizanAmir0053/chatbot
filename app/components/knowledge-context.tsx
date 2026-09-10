"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useConversations } from "./conversations-context";

/**
 * Knowledge-base state shared by the sidebar, the chat view and the dashboard.
 *
 * Upload lives here rather than in the chat page so a document can be added
 * from the sidebar, the composer or the dashboard and every surface reflects
 * it from one fetch.
 *
 * Documents are scoped to the open conversation. `documents` is what this
 * conversation can actually see and cite, which is what every chat surface
 * should show — listing the whole corpus beside a chat implies the agent will
 * read all of it, and it will not. `corpusDocuments` is the unscoped view,
 * kept separately for the dashboard, whose job is the deployment rather than
 * any one conversation.
 */

export type KnowledgeDoc = { source: string; chunks: number };

export type UploadOutcome = {
  ok: boolean;
  file: string;
  status: "ingested" | "skipped" | "failed";
  chunks: number;
  pages?: number;
  error?: string;
};

type KnowledgeState = {
  /** Visible to the open conversation: its own documents plus anything unscoped. */
  documents: KnowledgeDoc[];
  totalChunks: number;
  /** Everything in the store, regardless of conversation. For the dashboard. */
  corpusDocuments: KnowledgeDoc[];
  corpusTotalChunks: number;
  /** The conversation `documents` is scoped to, or null when none is open. */
  scopedTo: string | null;
  driver: string | null;
  persistent: boolean;
  loading: boolean;
  uploading: boolean;
  /** Name of the file currently being ingested, for progress affordances. */
  uploadingName: string | null;
  error: string | null;
  refresh: () => Promise<void>;
  upload: (file: File) => Promise<UploadOutcome>;
  remove: (source: string) => Promise<void>;
};

const KnowledgeContext = createContext<KnowledgeState | null>(null);

export function KnowledgeProvider({ children }: { children: React.ReactNode }) {
  const { activeId, ensureActiveId } = useConversations();
  const [documents, setDocuments] = useState<KnowledgeDoc[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [corpusDocuments, setCorpusDocuments] = useState<KnowledgeDoc[]>([]);
  const [corpusTotalChunks, setCorpusTotalChunks] = useState(0);
  const [driver, setDriver] = useState<string | null>(null);
  const [persistent, setPersistent] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      // Two views of the same store. The corpus request is skipped when no
      // conversation is open, because the scoped answer is already the corpus
      // and a second identical round trip would buy nothing.
      const [scopedRes, corpusRes] = await Promise.all([
        fetch(
          activeId
            ? `/api/documents?threadId=${encodeURIComponent(activeId)}`
            : "/api/documents",
          { cache: "no-store" },
        ),
        activeId ? fetch("/api/documents", { cache: "no-store" }) : Promise.resolve(null),
      ]);

      if (!scopedRes.ok) throw new Error(`HTTP ${scopedRes.status}`);
      const data = await scopedRes.json();
      setDocuments(data.documents ?? []);
      setTotalChunks(data.totalChunks ?? 0);
      setDriver(data.driver ?? null);
      setPersistent(Boolean(data.persistent));

      if (corpusRes?.ok) {
        const corpus = await corpusRes.json();
        setCorpusDocuments(corpus.documents ?? []);
        setCorpusTotalChunks(corpus.totalChunks ?? 0);
      } else {
        setCorpusDocuments(data.documents ?? []);
        setCorpusTotalChunks(data.totalChunks ?? 0);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to read the knowledge base");
    } finally {
      setLoading(false);
    }
  }, [activeId]);

  // Re-scopes on every conversation switch. Without this the previous chat's
  // documents stay on screen next to the new one, which is exactly the
  // confusion scoping exists to remove.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file: File): Promise<UploadOutcome> => {
      setUploading(true);
      setUploadingName(file.name);
      try {
        const form = new FormData();
        form.append("file", file);
        // Files the document to the open conversation, creating one if the chat
        // has not been sent yet — an upload with no conversation would land in
        // the shared corpus and be visible from every other chat.
        form.append("threadId", ensureActiveId());
        const res = await fetch("/api/upload", { method: "POST", body: form });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          return {
            ok: false,
            file: file.name,
            status: "failed",
            chunks: 0,
            error: data.error ?? `Upload failed (HTTP ${res.status})`,
          };
        }

        const ingest = data.ingest ?? {};
        await refresh();
        return {
          ok: ingest.status !== "failed",
          file: file.name,
          status: ingest.status ?? "failed",
          chunks: ingest.chunks ?? 0,
          pages: ingest.pages,
          error: ingest.error,
        };
      } catch (e) {
        return {
          ok: false,
          file: file.name,
          status: "failed",
          chunks: 0,
          error: e instanceof Error ? e.message : "Upload failed",
        };
      } finally {
        setUploading(false);
        setUploadingName(null);
      }
    },
    [refresh, ensureActiveId],
  );

  const remove = useCallback(
    async (source: string) => {
      // Drop it locally first so the list does not sit there during the
      // round-trip; refresh reconciles against the server either way.
      setDocuments((prev) => prev.filter((d) => d.source !== source));
      // Scoped, so removing a document here cannot empty an identically-named
      // one from another conversation.
      const query = new URLSearchParams({ source });
      if (activeId) query.set("threadId", activeId);
      await fetch(`/api/documents?${query.toString()}`, { method: "DELETE" });
      await refresh();
    },
    [refresh, activeId],
  );

  const value = useMemo<KnowledgeState>(
    () => ({
      documents,
      totalChunks,
      corpusDocuments,
      corpusTotalChunks,
      scopedTo: activeId,
      driver,
      persistent,
      loading,
      uploading,
      uploadingName,
      error,
      refresh,
      upload,
      remove,
    }),
    [
      documents,
      totalChunks,
      corpusDocuments,
      corpusTotalChunks,
      activeId,
      driver,
      persistent,
      loading,
      uploading,
      uploadingName,
      error,
      refresh,
      upload,
      remove,
    ],
  );

  return <KnowledgeContext.Provider value={value}>{children}</KnowledgeContext.Provider>;
}

export function useKnowledge(): KnowledgeState {
  const ctx = useContext(KnowledgeContext);
  if (!ctx) throw new Error("useKnowledge must be used inside <KnowledgeProvider>");
  return ctx;
}

/** File types the upload route accepts. Shared by every upload affordance. */
export const ACCEPTED_UPLOADS = ".pdf,.docx,.txt,.md,.markdown,.csv";
