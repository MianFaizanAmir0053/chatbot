"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

/**
 * Knowledge-base state shared by the sidebar, the chat view and the dashboard.
 *
 * Upload lives here rather than in the chat page so a document can be added
 * from the sidebar, the composer or the dashboard and every surface reflects
 * it from one fetch.
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
  documents: KnowledgeDoc[];
  totalChunks: number;
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
  const [documents, setDocuments] = useState<KnowledgeDoc[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [driver, setDriver] = useState<string | null>(null);
  const [persistent, setPersistent] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/documents", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocuments(data.documents ?? []);
      setTotalChunks(data.totalChunks ?? 0);
      setDriver(data.driver ?? null);
      setPersistent(Boolean(data.persistent));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to read the knowledge base");
    } finally {
      setLoading(false);
    }
  }, []);

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
    [refresh],
  );

  const remove = useCallback(
    async (source: string) => {
      // Drop it locally first so the list does not sit there during the
      // round-trip; refresh reconciles against the server either way.
      setDocuments((prev) => prev.filter((d) => d.source !== source));
      await fetch(`/api/documents?source=${encodeURIComponent(source)}`, { method: "DELETE" });
      await refresh();
    },
    [refresh],
  );

  const value = useMemo<KnowledgeState>(
    () => ({
      documents,
      totalChunks,
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
