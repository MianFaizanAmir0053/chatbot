import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import { EMBEDDING_CONFIG, env } from "../config";
import { getEmbeddings } from "../models";
import {
  scopeIds,
  summariseSources,
  type DocumentSummary,
  type ThreadScope,
  type VectorStoreDriver,
} from "./index";

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
    try {
      // Scoped retrieval filters on this on every search, so it must be
      // indexed; without it Qdrant falls back to scanning payloads and the
      // filter costs more than the search.
      await this.client.createPayloadIndex(this.collection, {
        field_name: "metadata.threadId",
        field_schema: "keyword",
      });
    } catch {
      /* already exists */
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

  async similaritySearchWithScore(
    query: string,
    k: number,
    threadId?: ThreadScope,
  ): Promise<[Document, number][]> {
    const store = await this.getStore();
    const filter = this.scopeFilter(threadId);
    // Filtering inside Qdrant rather than after the fact: a post-hoc filter
    // asks for k results and then throws some away, so a conversation with few
    // documents silently gets fewer passages than it asked for — worst exactly
    // when it has least to work with.
    return store.similaritySearchWithScore(query, k, filter);
  }

  /**
   * Qdrant filter for a scope, or undefined for the whole corpus.
   *
   * "This conversation OR unscoped" rather than a plain equality, so documents
   * ingested before scoping existed stay visible instead of disappearing.
   * Qdrant expresses absence as IsEmpty, which has no shorthand in a match.
   */
  private scopeFilter(threadId?: ThreadScope) {
    const ids = scopeIds(threadId);
    if (ids.length === 0) return undefined;
    return {
      should: [
        // `match: { any }` rather than one clause per id: the ancestor chain of
        // a deep fork would otherwise grow the filter without bound.
        { key: "metadata.threadId", match: { any: ids } },
        { is_empty: { key: "metadata.threadId" } },
      ],
    };
  }

  async deleteByThread(threadId: string): Promise<number> {
    await this.getStore();
    // Counted before deleting: Qdrant's delete reports an operation status
    // rather than how many points it matched, and the caller wants to tell the
    // user what was removed.
    const filter = { must: [{ key: "metadata.threadId", match: { value: threadId } }] };
    let removed = 0;
    try {
      const res = await this.client.count(this.collection, { exact: true, filter });
      removed = res.count;
    } catch {
      /* count is advisory; the delete still runs */
    }
    await this.client.delete(this.collection, { filter, wait: true });
    return removed;
  }

  async getAllDocuments(threadId?: ThreadScope): Promise<Document[]> {
    await this.getStore();
    const docs: Document[] = [];
    let offset: string | number | undefined | null = undefined;
    const filter = this.scopeFilter(threadId);

    // Scroll the whole collection; BM25 needs the full corpus in memory.
    for (;;) {
      const res = await this.client.scroll(this.collection, {
        limit: 256,
        offset: offset ?? undefined,
        with_payload: true,
        with_vector: false,
        ...(filter ? { filter } : {}),
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

  async deleteBySource(source: string, threadId?: ThreadScope): Promise<void> {
    await this.getStore();
    // Scoped deletes match the thread exactly — not "thread or unscoped" —
    // because removing a document from one conversation must never reach the
    // shared corpus that every other conversation can also see.
    const must: Array<Record<string, unknown>> = [
      { key: "metadata.source", match: { value: source } },
    ];
    if (threadId) must.push({ key: "metadata.threadId", match: { value: threadId } });

    await this.client.delete(this.collection, { filter: { must }, wait: true });
  }

  async listSources(threadId?: ThreadScope): Promise<DocumentSummary[]> {
    return summariseSources(await this.getAllDocuments(threadId));
  }

  async count(threadId?: ThreadScope): Promise<number> {
    try {
      await this.getStore();
      const filter = this.scopeFilter(threadId);
      const res = await this.client.count(this.collection, {
        exact: true,
        ...(filter ? { filter } : {}),
      });
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
