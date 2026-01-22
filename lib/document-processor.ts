import { Document } from "@langchain/core/documents";
import { getPresignedUrl } from "./s3";
import https from "https";
import http from "http";
import { extractText } from "unpdf";

/**
 * Download file from S3 using presigned URL
 */
async function downloadFileFromUrl(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith("https") ? https : http;

    protocol.get(url, (response) => {
      const chunks: Buffer[] = [];

      response.on("data", (chunk) => {
        chunks.push(chunk);
      });

      response.on("end", () => {
        resolve(Buffer.concat(chunks));
      });

      response.on("error", (error) => {
        reject(error);
      });
    });
  });
}

/**
 * Extract text from PDF using pdfjs-dist
 */
async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  try {
    // Convert Buffer to Uint8Array for unpdf
    const uint8Array = new Uint8Array(buffer);
    const { text } = await extractText(uint8Array);
    return text;
  } catch (error) {
    console.error("Error extracting PDF text:", error);
    throw new Error(`Failed to extract text from PDF: ${error}`);
  }
}

/**
 * Process uploaded file into documents
 */
export async function processUploadedFile(
  fileName: string,
  fileKey: string,
  fileType: string,
  s3Url: string
): Promise<Document[]> {
  try {
    let content = "";
    let documents: Document[] = [];

    if (fileType === "application/pdf") {
      // Download from S3 and extract PDF text
      const buffer = await downloadFileFromUrl(s3Url);
      content = await extractTextFromPDF(buffer);
    } else if (
      fileType === "text/plain" ||
      fileType.includes("text") ||
      fileType.includes("plain")
    ) {
      // Download and read text file
      const buffer = await downloadFileFromUrl(s3Url);
      content = buffer.toString("utf-8");
    } else if (
      fileType.includes("word") ||
      fileType.includes("document") ||
      fileType === "application/msword" ||
      fileType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) {
      // For DOCX/DOC files, we'll extract basic text
      // Note: Full .docx support would require additional libraries like docx-parser
      const buffer = await downloadFileFromUrl(s3Url);
      content = buffer.toString("utf-8", 0, Math.min(buffer.length, 10000));
    }

    if (content) {
      documents.push(
        new Document({
          pageContent: content,
          metadata: {
            source: fileName,
            fileKey: fileKey,
            fileType: fileType,
            uploadedAt: new Date().toISOString(),
          },
        })
      );
    }

    return documents;
  } catch (error) {
    console.error(`Error processing file ${fileName}:`, error);
    throw new Error(`Failed to process file ${fileName}: ${error}`);
  }
}

/**
 * Process multiple files
 */
export async function processMultipleFiles(
  files: Array<{
    name: string;
    key: string;
    type: string;
    url: string;
  }>
): Promise<Document[]> {
  const allDocuments: Document[] = [];

  for (const file of files) {
    try {
      const docs = await processUploadedFile(
        file.name,
        file.key,
        file.type,
        file.url
      );
      allDocuments.push(...docs);
    } catch (error) {
      console.error(`Failed to process file ${file.name}:`, error);
    }
  }

  return allDocuments;
}
