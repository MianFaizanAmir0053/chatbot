import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { RETRIEVAL_CONFIG } from "../config";
import type { ExtractedDocument } from "./extract";

/** Lines that look like section headings in manuals, specs and reports. */
const HEADING_PATTERNS: RegExp[] = [
  /^#{1,6}\s+\S/,                       // markdown
  /^\d+(?:\.\d+)*[.)]?\s+\S{3,}/,       // "3.2 Braking systems"
  /^[A-Z][A-Z0-9 ,.:'/&-]{5,80}$/,      // ALL CAPS HEADING
  /^(?:chapter|section|part|appendix)\s+[\dIVXLC]+/i,
];

function isHeading(line: string): boolean {
  const t = line.trim();
  if (t.length < 3 || t.length > 120) return false;
  if (/[.!?]$/.test(t) && !/^#{1,6}\s/.test(t)) return false; // full sentences aren't headings
  return HEADING_PATTERNS.some((re) => re.test(t));
}

interface Section {
  heading: string;
  text: string;
  page?: number;
}

/** Split raw text into heading-delimited sections, tracking the source page. */
function splitIntoSections(pages: string[]): Section[] {
  const sections: Section[] = [];
  let heading = "Introduction";
  let buffer: string[] = [];
  let pageOfBuffer = 1;

  const flush = () => {
    const text = buffer.join("\n").trim();
    if (text) sections.push({ heading, text, page: pageOfBuffer });
    buffer = [];
  };

  pages.forEach((pageText, idx) => {
    const pageNo = idx + 1;
    for (const line of pageText.split("\n")) {
      if (isHeading(line)) {
        flush();
        heading = line.trim().replace(/^#{1,6}\s*/, "");
        pageOfBuffer = pageNo;
      } else {
        if (buffer.length === 0) pageOfBuffer = pageNo;
        buffer.push(line);
      }
    }
  });
  flush();
  return sections;
}

export interface ChunkOptions {
  source: string;
  fileKey?: string;
  fileType?: string;
  documentTitle?: string;
}

/**
 * Turn an extracted document into embeddable chunks.
 *
 * Two things make this better than a flat recursive split:
 *
 * 1. Section-aware boundaries — we split within headings first, so a chunk
 *    rarely straddles two unrelated topics.
 * 2. Contextual prefixing — each chunk is embedded with its document title and
 *    section heading prepended. An isolated chunk saying "it must be replaced
 *    every 10,000 km" is nearly unretrievable; the same chunk prefixed with
 *    "Motorcycle Manual > Drive Chain Maintenance" is not. Because LangChain
 *    embeds `pageContent`, the prefix must live there to influence the vector
 *    at all. The unprefixed text is preserved in `metadata.originalText` so
 *    verbatim quotes shown to the user never include the synthetic header.
 */
export async function chunkDocument(
  extracted: ExtractedDocument,
  options: ChunkOptions,
): Promise<Document[]> {
  const pages = extracted.pages?.length ? extracted.pages : [extracted.text];
  const sections = splitIntoSections(pages);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: RETRIEVAL_CONFIG.CHUNK_SIZE,
    chunkOverlap: RETRIEVAL_CONFIG.CHUNK_OVERLAP,
    separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
  });

  const title = options.documentTitle ?? options.source;
  const chunks: Document[] = [];

  for (const section of sections) {
    const pieces = await splitter.splitText(section.text);

    pieces.forEach((piece, i) => {
      const contextHeader =
        section.heading && section.heading !== "Introduction"
          ? `${title} > ${section.heading}`
          : title;

      chunks.push(
        new Document({
          pageContent: `[${contextHeader}]\n\n${piece}`,
          metadata: {
            originalText: piece,
            source: options.source,
            fileKey: options.fileKey,
            fileType: options.fileType,
            section: section.heading,
            page: section.page,
            chunkIndex: chunks.length,
            sectionChunkIndex: i,
            ingestedAt: new Date().toISOString(),
          },
        }),
      );
    });
  }

  return chunks.filter((c) => String(c.metadata.originalText).trim().length > 40);
}
