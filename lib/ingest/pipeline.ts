import { createHash } from "crypto";
import { downloadFileFromS3 } from "../s3";
import { getVectorStore, type ThreadScope } from "../vectorstore";
import { invalidateSparseIndex } from "../retrieval/hybrid";
import { chunkDocument } from "./chunk";
import { extractDocument } from "./extract";

export interface IngestInput {
  name: string;
  key: string;
  type: string;
  /**
   * The conversation this upload belongs to.
   *
   * Absent means corpus-wide, which is what a dashboard upload is.
   */
  threadId?: string;
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

/**
 * Dedupe identity for an upload.
 *
 * Keyed by conversation as well as filename, because the same file uploaded to
 * two conversations is two documents. Keying on the name alone made the second
 * upload look like a repeat of the first and skip ingest entirely — leaving a
 * conversation that had plainly just been given a document unable to see it.
 */
function ingestKey(file: IngestInput): string {
  return `${file.threadId ?? "*"}:${file.name}`;
}

function hashBuffer(buf: Buffer): string {
  return createHash("sha256").update(buf).digest("hex").slice(0, 32);
}

/** S3 object → extracted text → chunks → embeddings → vector store. */
export async function ingestFile(file: IngestInput): Promise<IngestResult> {
  try {
    const buffer = await downloadFileFromS3(file.key);
    const hash = hashBuffer(buffer);

    if (ingestedHashes.get(ingestKey(file)) === hash) {
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
      threadId: file.threadId,
    });

    const store = await getVectorStore();
    // Replace rather than append, so re-uploading an edited file doesn't leave
    // stale chunks from the previous revision competing in retrieval.
    await store.deleteBySource(file.name, file.threadId);
    await store.addDocuments(chunks);

    ingestedHashes.set(ingestKey(file), hash);
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

/**
 * Remove every document belonging to one conversation.
 *
 * Called when a conversation is deleted. Without it the embeddings outlive the
 * thread that owned them: unreachable through any conversation, still counted
 * in the corpus, still occupying the vector store and still costing whatever
 * the store charges for them — a leak that is invisible precisely because
 * nothing can retrieve the orphans to notice they are there.
 *
 * Only that conversation's own documents go. Ancestors are shared with the
 * conversation that created them, and unscoped documents are shared with
 * everything.
 */
export async function removeThreadDocuments(threadId: string): Promise<number> {
  if (!threadId) return 0;
  const store = await getVectorStore();
  const removed = await store.deleteByThread(threadId);

  for (const key of [...ingestedHashes.keys()]) {
    if (key.startsWith(`${threadId}:`)) ingestedHashes.delete(key);
  }
  if (removed > 0) invalidateSparseIndex();
  return removed;
}

export async function removeDocument(source: string, threadId?: string): Promise<void> {
  const store = await getVectorStore();
  await store.deleteBySource(source, threadId);
  ingestedHashes.delete(`${threadId ?? "*"}:${source}`);
  invalidateSparseIndex();
}

/**
 * What a conversation can see, or the whole corpus when unscoped.
 *
 * The dashboard and health checks want everything; the chat view wants only
 * this conversation's documents plus anything unscoped.
 */
export async function knowledgeBaseStatus(threadId?: ThreadScope) {
  const store = await getVectorStore();
  const [sources, count] = await Promise.all([
    store.listSources(threadId),
    store.count(threadId),
  ]);
  return {
    driver: store.name,
    persistent: store.persistent,
    totalChunks: count,
    documents: sources,
  };
}
