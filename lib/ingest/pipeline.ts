import { createHash } from "crypto";
import { downloadFileFromS3 } from "../s3";
import { getVectorStore } from "../vectorstore";
import { invalidateSparseIndex } from "../retrieval/hybrid";
import { chunkDocument } from "./chunk";
import { extractDocument } from "./extract";

export interface IngestInput {
  name: string;
  key: string;
  type: string;
}

export interface IngestResult {
  source: string;
  status: "ingested" | "skipped" | "failed";
  chunks: number;
  pages?: number;
  error?: string;
}

/**
 * Tracks content hashes of already-ingested documents.
 *
 * Re-uploading the same file was previously re-embedding it wholesale, which
 * both cost money and polluted retrieval with duplicate chunks that crowd out
 * genuine variety in the top-k.
 */
const ingestedHashes = new Map<string, string>();

function hashBuffer(buf: Buffer): string {
  return createHash("sha256").update(buf).digest("hex").slice(0, 32);
}

/** S3 object → extracted text → chunks → embeddings → vector store. */
export async function ingestFile(file: IngestInput): Promise<IngestResult> {
  try {
    const buffer = await downloadFileFromS3(file.key);
    const hash = hashBuffer(buffer);

    if (ingestedHashes.get(file.name) === hash) {
      return { source: file.name, status: "skipped", chunks: 0 };
    }

    const extracted = await extractDocument(buffer, file.type, file.name);
    if (!extracted.text || extracted.text.trim().length < 50) {
      return {
        source: file.name,
        status: "failed",
        chunks: 0,
        error:
          "No extractable text found. The file may be a scanned image requiring OCR.",
      };
    }

    const chunks = await chunkDocument(extracted, {
      source: file.name,
      fileKey: file.key,
      fileType: file.type,
      documentTitle: file.name.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " "),
    });

    const store = await getVectorStore();
    // Replace rather than append, so re-uploading an edited file doesn't leave
    // stale chunks from the previous revision competing in retrieval.
    await store.deleteBySource(file.name);
    await store.addDocuments(chunks);

    ingestedHashes.set(file.name, hash);
    invalidateSparseIndex();

    return {
      source: file.name,
      status: "ingested",
      chunks: chunks.length,
      pages: extracted.pageCount,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[ingest] ${file.name} failed:`, message);
    return { source: file.name, status: "failed", chunks: 0, error: message };
  }
}

/** Ingest several files, isolating failures so one bad file can't sink the batch. */
export async function ingestFiles(files: IngestInput[]): Promise<IngestResult[]> {
  const results: IngestResult[] = [];
  for (const file of files) {
    results.push(await ingestFile(file));
  }
  return results;
}

export async function removeDocument(source: string): Promise<void> {
  const store = await getVectorStore();
  await store.deleteBySource(source);
  ingestedHashes.delete(source);
  invalidateSparseIndex();
}

export async function knowledgeBaseStatus() {
  const store = await getVectorStore();
  const [sources, count] = await Promise.all([store.listSources(), store.count()]);
  return {
    driver: store.name,
    persistent: store.persistent,
    totalChunks: count,
    documents: sources,
  };
}
