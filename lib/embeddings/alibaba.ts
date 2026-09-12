import { Embeddings } from "@langchain/core/embeddings";
import { credentialId, isRefused, noteFailure } from "../credential-health";

/* ------------------------------------------------------------------ *
 * Alibaba Model Studio embeddings
 *
 * A second embedding provider, so retrieval does not rest entirely on Cohere.
 * Measured on the same pair of texts, `text-embedding-v4` separates a matching
 * passage from an unrelated one slightly better than `embed-v4.0` does (0.53
 * against 0.48), and it can be asked for 1536 dimensions, matching the width
 * already configured.
 *
 * What it is NOT is a rotation partner for Cohere. That was measured too, and
 * the result is unambiguous: embedding the same question with both and
 * comparing across providers gives 0.019 against a matching passage and 0.022
 * against an unrelated one — a *negative* signal, noise. Both produce 1536
 * floats, so mixing them raises no error anywhere; every search would simply
 * compare a query in one space against documents in another and return
 * whatever sat closest to nothing in particular.
 *
 * So a collection belongs to exactly one embedding model, and switching
 * providers means re-indexing. `embeddingModelId` exists to make that
 * enforceable rather than remembered — see the vector store's guard.
 * ------------------------------------------------------------------ */

export interface AlibabaEmbeddingsOptions {
  keys: string[];
  baseURL: string;
  model: string;
  dimensions: number;
}

/** Texts per request. The endpoint rejects larger batches. */
const MAX_BATCH = 10;

export class AlibabaEmbeddings extends Embeddings {
  private readonly keys: string[];
  private readonly baseURL: string;
  private readonly model: string;
  private readonly dimensions: number;
  private cursor = 0;

  /**
   * Query vectors already computed, keyed by exact text.
   *
   * The same reasoning as the Cohere embedder's cache: query text repeats
   * constantly across branches, rounds and follow-ups, and embedding is
   * deterministic for a fixed model, so a hit is indistinguishable from a call
   * except in latency and quota. Documents are embedded once at ingest and
   * never repeat, so only queries are cached.
   */
  private readonly queryCache = new Map<string, number[]>();

  constructor(options: AlibabaEmbeddingsOptions) {
    super({});
    this.keys = options.keys;
    this.baseURL = options.baseURL.replace(/\/$/, "");
    this.model = options.model;
    this.dimensions = options.dimensions;
  }

  /** Which model produced these vectors, for the store's compatibility guard. */
  get embeddingModelId(): string {
    return `alibaba:${this.model}:${this.dimensions}`;
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    let lastError: unknown;

    // Refused keys are skipped, and a refusal recorded by one caller is visible
    // to every other immediately — which matters because retrieval issues its
    // query variants concurrently.
    const live = this.keys.filter((k) => !isRefused(credentialId(this.baseURL, k)));
    const pool = live.length > 0 ? live : this.keys;

    for (let attempt = 0; attempt < pool.length; attempt++) {
      const index = (this.cursor + attempt) % pool.length;
      const key = pool[index];
      try {
        const res = await fetch(`${this.baseURL}/embeddings`, {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({ model: this.model, input: texts, dimensions: this.dimensions }),
          signal: AbortSignal.timeout(60_000),
        });

        const json = (await res.json()) as {
          data?: Array<{ embedding: number[]; index: number }>;
          error?: { message?: string };
        };

        if (!res.ok || !json.data) {
          const message = json.error?.message ?? `HTTP ${res.status}`;
          throw Object.assign(new Error(`${res.status} ${message}`), { status: res.status });
        }

        // Returned order is not guaranteed, and a silently reordered batch
        // would attach every vector to the wrong chunk — a corruption that
        // produces no error and is invisible until answers are subtly wrong.
        const ordered = [...json.data].sort((a, b) => a.index - b.index);
        this.cursor = index;
        return ordered.map((d) => d.embedding);
      } catch (error) {
        lastError = error;
        noteFailure(credentialId(this.baseURL, key), error);
      }
    }

    throw lastError;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += MAX_BATCH) {
      out.push(...(await this.embedBatch(texts.slice(i, i + MAX_BATCH))));
    }
    return out;
  }

  async embedQuery(text: string): Promise<number[]> {
    const cached = this.queryCache.get(text);
    if (cached) return cached;

    const [vector] = await this.embedBatch([text]);

    if (this.queryCache.size >= 500) {
      const oldest = this.queryCache.keys().next().value;
      if (oldest !== undefined) this.queryCache.delete(oldest);
    }
    this.queryCache.set(text, vector);
    return vector;
  }

  /** Cache occupancy, for /api/health. */
  cacheSize(): number {
    return this.queryCache.size;
  }
}
