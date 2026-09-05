import type { Document } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";

/**
 * A thin driver contract over the vector database.
 *
 * The retrieval pipeline is written against this interface, so swapping Qdrant
 * for pgvector/Upstash later means adding one file — not touching retrieval.
 */
export interface VectorStoreDriver {
  readonly name: string;
  /** True when the backing store survives a process restart. */
  readonly persistent: boolean;
  getStore(): Promise<VectorStore>;
  addDocuments(docs: Document[]): Promise<number>;
  similaritySearchWithScore(query: string, k: number): Promise<[Document, number][]>;
  /** Every chunk in the store — required to build the BM25 sparse index. */
  getAllDocuments(): Promise<Document[]>;
  deleteBySource(source: string): Promise<void>;
  listSources(): Promise<Array<{ source: string; chunks: number }>>;
  count(): Promise<number>;
  healthy(): Promise<boolean>;
}

export { getVectorStore, resetVectorStore } from "./factory";
