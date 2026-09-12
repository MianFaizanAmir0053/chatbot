import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import { EMBEDDING_CONFIG, env } from "../config";
import { embeddingModelId, getEmbeddings } from "../models";
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
    await this.assertEmbeddingModelMatches(exists);
    return this.store as unknown as VectorStore;
  }

  /**
   * Refuse to query a collection that was written by a different embedding model.
   *
   * Vectors from two embedding models are not comparable, even at identical
   * width. Measured on this deployment: a question embedded by Alibaba scored
   * 0.019 against the passage that answers it and 0.022 against an unrelated
   * one, while the same pair within one provider scored 0.60 and 0.12. Across
   * providers the signal is negative — noise.
   *
   * Nothing would report that. Both providers emit 1536 floats, so upserts
   * succeed, searches succeed, and every answer is quietly built from whatever
   * sat closest to nothing in particular. A stored stamp is the only thing that
   * can catch it, so the model that wrote the collection is recorded on it and
   * checked here.
   *
   * Failing loudly is the point: the fix is to re-index, and a corpus embedded
   * by two models cannot be repaired by anything else.
   */
  private async assertEmbeddingModelMatches(collectionExisted: boolean): Promise<void> {
    const current = embeddingModelId();
    try {
      const info = (await this.client.getCollection(this.collection)) as {
        payload_schema?: unknown;
      };
      void info;

      const stored = await this.readEmbeddingStamp();

      if (!stored) {
        // Either a fresh collection, or one written before stamping existed.
        // Adopt the current model rather than guessing: an unstamped collection
        // is assumed to match, which is true for every deployment that has not
        // changed provider, and the stamp makes any later change detectable.
        await this.writeEmbeddingStamp(current);
        if (collectionExisted) {
          console.warn(
            `[qdrant] collection "${this.collection}" had no embedding stamp; ` +
              `adopting "${current}". If it was indexed with a different model, re-index.`,
          );
        }
        return;
      }

      if (stored !== current) {
        throw new Error(
          `Collection "${this.collection}" was indexed with ${stored} but this process embeds ` +
            `with ${current}. Vectors from two embedding models are not comparable, so every ` +
            `search would return near-random passages without raising an error. Re-index the ` +
            `corpus, or set the embedding provider back to the one that wrote it.`,
        );
      }
    } catch (error) {
      // A genuine mismatch must surface; anything else (an old Qdrant without
      // the metadata call, a transient failure) must not take retrieval down.
      if (error instanceof Error && error.message.includes("not comparable")) throw error;
      console.warn(
        "[qdrant] could not verify the embedding stamp:",
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  /**
   * The stamp lives in a zero-vector point under a reserved id.
   *
   * Qdrant has no collection-level metadata field, and a payload index cannot
   * hold a value. A single reserved point costs one vector's storage and is
   * filtered out of results by its own `metadata.source`, which no real
   * document can take.
   */
  private static readonly STAMP_ID = "00000000-0000-4000-8000-0000000000e1";

  private async readEmbeddingStamp(): Promise<string | null> {
    const found = await this.client.retrieve(this.collection, {
      ids: [QdrantDriver.STAMP_ID],
      with_payload: true,
    });
    const payload = found?.[0]?.payload as { embeddingModel?: unknown } | undefined;
    return typeof payload?.embeddingModel === "string" ? payload.embeddingModel : null;
  }

  private async writeEmbeddingStamp(modelId: string): Promise<void> {
    const size = await this.vectorSize();
    await this.client.upsert(this.collection, {
      wait: true,
      points: [
        {
          id: QdrantDriver.STAMP_ID,
          vector: new Array(size).fill(0),
          payload: {
            embeddingModel: modelId,
            // Reserved so the point cannot match a document filter, and is
            // recognisable if anyone inspects the collection by hand.
            metadata: { source: "__embedding_stamp__" },
          },
        },
      ],
    });
  }

  private async vectorSize(): Promise<number> {
    const info = (await this.client.getCollection(this.collection)) as {
      config?: { params?: { vectors?: { size?: number } } };
    };
    return info?.config?.params?.vectors?.size ?? EMBEDDING_CONFIG.dimensions;
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
        // The embedding stamp is bookkeeping, not a document. Left in, it would
        // join the BM25 corpus as an empty entry and skew every document
        // frequency in the index.
        if (point.id === QdrantDriver.STAMP_ID) continue;
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
      // The embedding stamp occupies a point but is not a chunk, and this
      // number is shown to the user as how much they have indexed.
      const stamp = await this.readEmbeddingStamp();
      return Math.max(0, res.count - (stamp && !threadId ? 1 : 0));
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
