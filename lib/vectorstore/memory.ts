import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import { getEmbeddings } from "../models";
import type { VectorStoreDriver } from "./index";

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

  async similaritySearchWithScore(query: string, k: number): Promise<[Document, number][]> {
    const store = await this.getStore();
    return store.similaritySearchWithScore(query, k);
  }

  async getAllDocuments(): Promise<Document[]> {
    return this.docs;
  }

  async deleteBySource(source: string): Promise<void> {
    this.docs = this.docs.filter((d) => d.metadata?.source !== source);
    // MemoryVectorStore has no delete primitive; rebuild from the survivors.
    this.store = new MemoryVectorStore(getEmbeddings());
    if (this.docs.length > 0) await this.store.addDocuments(this.docs);
  }

  async listSources(): Promise<Array<{ source: string; chunks: number }>> {
    const counts = new Map<string, number>();
    for (const d of this.docs) {
      const src = String(d.metadata?.source ?? "unknown");
      counts.set(src, (counts.get(src) ?? 0) + 1);
    }
    return [...counts.entries()].map(([source, chunks]) => ({ source, chunks }));
  }

  async count(): Promise<number> {
    return this.docs.length;
  }

  async healthy(): Promise<boolean> {
    return true;
  }
}
