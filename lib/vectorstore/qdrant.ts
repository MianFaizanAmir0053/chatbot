import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import { EMBEDDING_CONFIG, env } from "../config";
import { getEmbeddings } from "../models";
import type { VectorStoreDriver } from "./index";

/**
 * Qdrant-backed persistent vector store.
 *
 * Documents are stored with their metadata in the Qdrant payload, which lets us
 * filter and delete by source without a second bookkeeping database.
 */
export class QdrantDriver implements VectorStoreDriver {
  readonly name = "qdrant";
  readonly persistent = true;

  private store: QdrantVectorStore | null = null;
  private client: QdrantClient;
  private collection: string;

  constructor() {
    this.collection = env.QDRANT_COLLECTION;
    this.client = new QdrantClient({
      url: env.QDRANT_URL,
      apiKey: env.QDRANT_API_KEY,
      checkCompatibility: false,
    });
  }

  /**
   * Determine the embedding width to size the collection with.
   *
   * Measured from the embedder itself rather than trusted from config: a
   * mismatch between the collection's declared size and the real vector width
   * makes every upsert fail with an opaque Qdrant error. Probing once at
   * creation time means changing the embedding model just works.
   */
  private async resolveDimensions(): Promise<number> {
    try {
      const [probe] = await getEmbeddings().embedDocuments(["dimension probe"]);
      if (probe?.length) return probe.length;
    } catch (error) {
      console.warn("[qdrant] dimension probe failed, using configured value:", error);
    }
    return EMBEDDING_CONFIG.dimensions;
  }

  async getStore(): Promise<VectorStore> {
    if (this.store) return this.store as unknown as VectorStore;

    // Only probe when the collection doesn't exist yet — an existing collection
    // already has its size fixed, and probing would just cost an API call.
    const { collections } = await this.client.getCollections();
    const exists = collections.some((c) => c.name === this.collection);
    const size = exists ? EMBEDDING_CONFIG.dimensions : await this.resolveDimensions();

    this.store = new QdrantVectorStore(getEmbeddings(), {
      client: this.client,
      collectionName: this.collection,
      collectionConfig: {
        vectors: {
          size,
          // Cohere embeddings are normalised, so cosine is the correct metric.
          distance: "Cosine",
        },
        // Keep the HNSW graph in memory but let payloads spill to disk: payloads
        // are large (full chunk text) and only read for the final top-k.
        on_disk_payload: true,
      },
    });

    await this.store.ensureCollection();
    await this.ensurePayloadIndex();
    return this.store as unknown as VectorStore;
  }

  /**
   * Index the source field so delete-by-source and source filters use an index
   * rather than a full payload scan.
   */
  private async ensurePayloadIndex(): Promise<void> {
    try {
      await this.client.createPayloadIndex(this.collection, {
        field_name: "metadata.source",
        field_schema: "keyword",
      });
    } catch {
      // Already exists — Qdrant returns 4xx rather than being idempotent here.
    }
  }

  async addDocuments(docs: Document[]): Promise<number> {
    if (docs.length === 0) return 0;
    const store = (await this.getStore()) as QdrantVectorStore;
    // Batch to stay under Qdrant's request size ceiling and to bound the size of
    // each embedding call.
    const BATCH = 64;
    for (let i = 0; i < docs.length; i += BATCH) {
      await store.addDocuments(docs.slice(i, i + BATCH));
    }
    return docs.length;
  }

  async similaritySearchWithScore(query: string, k: number): Promise<[Document, number][]> {
    const store = await this.getStore();
    return store.similaritySearchWithScore(query, k);
  }

  async getAllDocuments(): Promise<Document[]> {
    await this.getStore();
    const docs: Document[] = [];
    let offset: string | number | undefined | null = undefined;

    // Scroll the whole collection; BM25 needs the full corpus in memory.
    for (;;) {
      const res = await this.client.scroll(this.collection, {
        limit: 256,
        offset: offset ?? undefined,
        with_payload: true,
        with_vector: false,
      });
      for (const point of res.points) {
        const payload = (point.payload ?? {}) as Record<string, unknown>;
        docs.push(
          new Document({
            pageContent: String(payload.content ?? ""),
            metadata: (payload.metadata ?? {}) as Record<string, unknown>,
          }),
        );
      }
      if (!res.next_page_offset) break;
      offset = res.next_page_offset as string | number;
    }
    return docs;
  }

  async deleteBySource(source: string): Promise<void> {
    await this.getStore();
    await this.client.delete(this.collection, {
      filter: { must: [{ key: "metadata.source", match: { value: source } }] },
      wait: true,
    });
  }

  async listSources(): Promise<Array<{ source: string; chunks: number }>> {
    const docs = await this.getAllDocuments();
    const counts = new Map<string, number>();
    for (const d of docs) {
      const src = String(d.metadata?.source ?? "unknown");
      counts.set(src, (counts.get(src) ?? 0) + 1);
    }
    return [...counts.entries()].map(([source, chunks]) => ({ source, chunks }));
  }

  async count(): Promise<number> {
    try {
      await this.getStore();
      const res = await this.client.count(this.collection, { exact: true });
      return res.count;
    } catch {
      return 0;
    }
  }

  async healthy(): Promise<boolean> {
    try {
      await this.client.getCollections();
      return true;
    } catch {
      return false;
    }
  }
}
