import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import { getEmbeddings } from "../models";
import {
  scopeIds,
  summariseSources,
  type DocumentSummary,
  type ThreadScope,
  type VectorStoreDriver,
} from "./index";

/**
 * Non-persistent fallback used only when Qdrant is unreachable.
 *
 * This exists so a partially-configured environment still boots and serves
 * requests. It is explicitly reported as non-persistent by /api/health so the
 * degraded state is visible rather than silent.
 */
export class MemoryDriver implements VectorStoreDriver {
  readonly name = "memory";
  readonly persistent = false;

  private store: MemoryVectorStore | null = null;
  private docs: Document[] = [];

  async getStore(): Promise<VectorStore> {
    if (!this.store) {
      this.store = new MemoryVectorStore(getEmbeddings());
    }
    return this.store as unknown as VectorStore;
  }

  async addDocuments(docs: Document[]): Promise<number> {
    if (docs.length === 0) return 0;
    const store = await this.getStore();
    await store.addDocuments(docs);
    this.docs.push(...docs);
    return docs.length;
  }

  /** In scope when it belongs to this conversation, or to none. */
  private inScope(doc: Document, threadId?: ThreadScope): boolean {
    const ids = scopeIds(threadId);
    if (ids.length === 0) return true;
    const owner = doc.metadata?.threadId;
    return !owner || ids.includes(String(owner));
  }

  async deleteByThread(threadId: string): Promise<number> {
    const before = this.docs.length;
    this.docs = this.docs.filter((d) => d.metadata?.threadId !== threadId);
    const removed = before - this.docs.length;
    if (removed > 0) {
      // MemoryVectorStore has no delete primitive; rebuild from the survivors.
      this.store = new MemoryVectorStore(getEmbeddings());
      if (this.docs.length > 0) await this.store.addDocuments(this.docs);
    }
    return removed;
  }

  async similaritySearchWithScore(
    query: string,
    k: number,
    threadId?: ThreadScope,
  ): Promise<[Document, number][]> {
    const store = await this.getStore();
    if (!threadId) return store.similaritySearchWithScore(query, k);

    // MemoryVectorStore cannot filter, so over-fetch and then narrow. The
    // multiplier matters: asking for exactly k and discarding the out-of-scope
    // ones returns fewer passages than requested, and most severely for a
    // conversation whose documents are a small slice of the corpus.
    const hits = await store.similaritySearchWithScore(query, k * 5);
    return hits.filter(([doc]) => this.inScope(doc, threadId)).slice(0, k);
  }

  async getAllDocuments(threadId?: ThreadScope): Promise<Document[]> {
    return threadId ? this.docs.filter((d) => this.inScope(d, threadId)) : this.docs;
  }

  async deleteBySource(source: string, threadId?: ThreadScope): Promise<void> {
    // Exact thread match, not "thread or unscoped": removing a document from
    // one conversation must not touch the corpus every conversation shares.
    this.docs = this.docs.filter(
      (d) =>
        d.metadata?.source !== source ||
        (threadId ? d.metadata?.threadId !== threadId : false),
    );
    // MemoryVectorStore has no delete primitive; rebuild from the survivors.
    this.store = new MemoryVectorStore(getEmbeddings());
    if (this.docs.length > 0) await this.store.addDocuments(this.docs);
  }

  async listSources(threadId?: ThreadScope): Promise<DocumentSummary[]> {
    return summariseSources(this.docs.filter((d) => this.inScope(d, threadId)));
  }

  async count(threadId?: ThreadScope): Promise<number> {
    return threadId ? this.docs.filter((d) => this.inScope(d, threadId)).length : this.docs.length;
  }

  async healthy(): Promise<boolean> {
    return true;
  }
}
