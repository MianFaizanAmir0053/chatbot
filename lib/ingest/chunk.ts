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
  /** The immediate heading this text sits under. */
  heading: string;
  /** Ancestor headings, outermost first — "Expenses > Travel > Meals". */
  breadcrumb: string[];
  text: string;
  page?: number;
}

/**
 * Nesting depth of a heading, so ancestors can be tracked.
 *
 * Markdown states depth directly and numbered headings imply it — "3.2.1" is
 * one level below "3.2". Everything else is treated as a single level, which is
 * the safe assumption when the document gives no structural signal.
 */
function headingLevel(line: string): number {
  const t = line.trim();
  const md = /^(#{1,6})\s/.exec(t);
  if (md) return md[1].length;
  const numbered = /^(\d+(?:\.\d+)*)[.)]?\s/.exec(t);
  if (numbered) return numbered[1].split(".").length;
  return 1;
}

/**
 * Short trailing sections are folded into the previous one.
 *
 * A heading immediately followed by another heading — common in tables of
 * contents and in documents where a title sits on its own line above a figure —
 * otherwise produces a chunk of a few words. Those were silently discarded by
 * the minimum-length filter, taking their heading's content with them.
 */
const MIN_SECTION_CHARS = 120;

/** Split raw text into heading-delimited sections, tracking page and ancestry. */
function splitIntoSections(pages: string[]): Section[] {
  const sections: Section[] = [];
  const ancestors: Array<{ level: number; title: string }> = [];
  let heading = "Introduction";
  let buffer: string[] = [];
  let pageOfBuffer = 1;

  const flush = () => {
    const text = buffer.join("\n").trim();
    if (!text) {
      buffer = [];
      return;
    }

    const previous = sections[sections.length - 1];
    if (previous && text.length < MIN_SECTION_CHARS) {
      // Too small to retrieve on its own; keep the words with their context.
      previous.text += `\n\n${heading}\n${text}`;
    } else {
      sections.push({
        heading,
        breadcrumb: ancestors.map((a) => a.title),
        text,
        page: pageOfBuffer,
      });
    }
    buffer = [];
  };

  pages.forEach((pageText, idx) => {
    const pageNo = idx + 1;
    for (const line of pageText.split("\n")) {
      if (isHeading(line)) {
        flush();
        const title = line.trim().replace(/^#{1,6}\s*/, "");
        const level = headingLevel(line);

        // Drop any sibling or deeper heading; what remains is this one's path.
        while (ancestors.length > 0 && ancestors[ancestors.length - 1].level >= level) {
          ancestors.pop();
        }
        ancestors.push({ level, title });

        heading = title;
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
      // Full ancestry, not just the nearest heading. "Meals" alone is ambiguous
      // across a document that also has expense, per-diem and hospitality
      // sections; "Handbook > Expenses > Travel > Meals" is not, and the whole
      // path is what the embedding sees.
      const path = [title, ...section.breadcrumb];
      if (section.heading !== "Introduction") path.push(section.heading);

      // Deduplicate case-insensitively: a document's H1 usually restates its
      // title, which would otherwise open every header with the same words
      // twice and waste the most heavily weighted part of the embedded text.
      const seenPart = new Set<string>();
      const contextHeader = path
        .filter((part) => {
          const key = part.trim().toLowerCase();
          if (!key || seenPart.has(key)) return false;
          seenPart.add(key);
          return true;
        })
        .join(" > ");

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
