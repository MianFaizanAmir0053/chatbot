import type { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";

/**
 * A thin driver contract over the vector database.
 *
 * The retrieval pipeline is written against this interface, so swapping Qdrant
 * for pgvector/Upstash later means adding one file — not touching retrieval.
 */
/**
 * Which conversation's documents an operation may see.
 *
 * `undefined` means the whole corpus — the behaviour before conversations
 * existed, and still what the dashboard and health checks want. A string
 * narrows to that conversation *plus* any unscoped document, so a corpus
 * uploaded before scoping existed stays reachable rather than becoming
 * invisible the moment this shipped.
 *
 * An array is a conversation and its ancestors, which is how a fork reads the
 * documents of the conversation it came from. Inheriting the scope rather than
 * copying the documents is the whole reason forking is cheap: duplicating them
 * would mean re-embedding every chunk, so a fork would cost as much as the
 * original upload and take as long.
 *
 * Passed explicitly at every call site rather than held as ambient state.
 * Retrieval runs concurrently across several research branches in one request,
 * so a mutable "current thread" would be read by a branch belonging to a
 * different one — and a leak here shows another user's documents.
 */
export type ThreadScope = string | string[] | undefined;

/** Normalise a scope to the thread ids it covers. Empty means unscoped. */
export function scopeIds(scope: ThreadScope): string[] {
  if (!scope) return [];
  return (Array.isArray(scope) ? scope : [scope]).filter(Boolean);
}

export interface DocumentSummary {
  source: string;
  chunks: number;
  /** The conversation that uploaded it, or undefined for a corpus-wide document. */
  threadId?: string;
}

/**
 * Group chunks into one entry per document per owner.
 *
 * Shared by both drivers so the grouping rule cannot drift between them — the
 * in-memory store is the fallback for an unreachable Qdrant, and a difference
 * here would show up only in that degraded state, where it is hardest to spot.
 *
 * Keyed on owner as well as source, because the same filename in two
 * conversations is two documents. Merging them would report one entry with a
 * combined chunk count and no single owner, and a delete against that entry
 * would silently reach the wrong conversation's copy.
 */
export function summariseSources(docs: Document[]): DocumentSummary[] {
  const groups = new Map<string, DocumentSummary>();

  for (const doc of docs) {
    const source = String(doc.metadata?.source ?? "unknown");
    const owner = doc.metadata?.threadId ? String(doc.metadata.threadId) : undefined;
    const key = `${owner ?? ""}|${source}`;

    const existing = groups.get(key);
    if (existing) existing.chunks += 1;
    else groups.set(key, { source, chunks: 1, threadId: owner });
  }

  return [...groups.values()];
}

export interface VectorStoreDriver {
  readonly name: string;
  /** True when the backing store survives a process restart. */
  readonly persistent: boolean;
  getStore(): Promise<VectorStore>;
  addDocuments(docs: Document[]): Promise<number>;
  similaritySearchWithScore(
    query: string,
    k: number,
    threadId?: ThreadScope,
  ): Promise<[Document, number][]>;
  /** Every chunk in scope — required to build the BM25 sparse index. */
  getAllDocuments(threadId?: ThreadScope): Promise<Document[]>;
  /**
   * Remove a document.
   *
   * Scoped, because the same filename in two conversations is two different
   * documents: deleting "policy.pdf" from one chat must not empty it from
   * another. Without a scope it removes every copy, which is what the
   * dashboard's corpus-wide delete means.
   */
  deleteBySource(source: string, threadId?: ThreadScope): Promise<void>;
  /**
   * Remove every document belonging to one conversation.
   *
   * Exists so deleting a conversation does not strand its embeddings in the
   * store, paying for vectors nothing can reach. Matches that conversation
   * exactly — never its ancestors, and never unscoped documents, both of which
   * are shared with conversations that still exist.
   */
  deleteByThread(threadId: string): Promise<number>;
  /**
   * Documents in scope, with the conversation each belongs to.
   *
   * The owner is returned because the caller cannot otherwise tell an own
   * document from an inherited one, and the two behave differently: deleting an
   * inherited document from a fork matches nothing, so the UI has to know not
   * to offer it. Grouped by source *and* owner, so the same filename uploaded
   * to two conversations is two entries rather than one with a merged count.
   */
  listSources(threadId?: ThreadScope): Promise<DocumentSummary[]>;
  count(threadId?: ThreadScope): Promise<number>;
  healthy(): Promise<boolean>;
}

export { getVectorStore, resetVectorStore } from "./factory";
