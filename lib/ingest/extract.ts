import { extractText, getDocumentProxy } from "unpdf";

export interface ExtractedDocument {
  text: string;
  pageCount?: number;
  /** Per-page text, when the format exposes page boundaries. Enables page citations. */
  pages?: string[];
}

/** Collapse the ragged whitespace PDF extractors emit, without losing paragraphs. */
function normalise(text: string): string {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/ /g, " ")
    // Join words hyphenated across a line break.
    .replace(/(\w)-\n(\w)/g, "$1$2")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Extract per-page text from a PDF.
 *
 * Page-level granularity is kept because it is the only reliable citation
 * anchor a PDF gives us, and the chunker propagates it into chunk metadata.
 */
async function extractPdf(buffer: Buffer): Promise<ExtractedDocument> {
  const data = new Uint8Array(buffer);
  const pdf = await getDocumentProxy(data);
  const { text, totalPages } = await extractText(pdf, { mergePages: false });
  const pages = (Array.isArray(text) ? text : [text]).map(normalise);

  return {
    text: normalise(pages.join("\n\n")),
    pageCount: totalPages,
    pages,
  };
}

/**
 * Extract text from a .docx.
 *
 * mammoth understands the OOXML package. The previous implementation read the
 * zip container as UTF-8, which produced binary garbage.
 */
async function extractDocx(buffer: Buffer): Promise<ExtractedDocument> {
  const mammoth = await import("mammoth");
  const result = await mammoth.extractRawText({ buffer });
  return { text: normalise(result.value) };
}

export async function extractDocument(
  buffer: Buffer,
  fileType: string,
  fileName: string,
): Promise<ExtractedDocument> {
  const type = (fileType || "").toLowerCase();
  const ext = fileName.toLowerCase().split(".").pop() ?? "";

  if (type === "application/pdf" || ext === "pdf") {
    return extractPdf(buffer);
  }

  if (
    type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    ext === "docx"
  ) {
    return extractDocx(buffer);
  }

  if (type === "application/msword" || ext === "doc") {
    // Legacy binary .doc is a different (OLE2) format that mammoth cannot read.
    throw new Error(
      "Legacy .doc files are not supported. Please convert the file to .docx or PDF and re-upload.",
    );
  }

  if (type.startsWith("text/") || ["txt", "md", "markdown", "csv"].includes(ext)) {
    return { text: normalise(buffer.toString("utf-8")) };
  }

  throw new Error(`Unsupported file type: ${fileType || ext || "unknown"}`);
}
