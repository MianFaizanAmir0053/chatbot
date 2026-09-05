import { NextRequest, NextResponse } from "next/server";
import { GUARDRAIL_CONFIG, features } from "@/lib/config";
import { ingestFile } from "@/lib/ingest/pipeline";
import { uploadFileToS3 } from "@/lib/s3";

export const runtime = "nodejs";
export const maxDuration = 300;

/**
 * Allowed upload types.
 *
 * Legacy binary .doc is deliberately excluded: it is an OLE2 container that the
 * extractor cannot read, and accepting it would only produce a file that fails
 * silently at ingest time.
 */
const ALLOWED = new Map<string, string[]>([
  ["application/pdf", ["pdf"]],
  ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", ["docx"]],
  ["text/plain", ["txt", "md", "markdown"]],
  ["text/markdown", ["md", "markdown"]],
  ["text/csv", ["csv"]],
]);

function isAllowed(type: string, name: string): boolean {
  const ext = name.toLowerCase().split(".").pop() ?? "";
  if (ALLOWED.has(type)) return true;
  // Browsers are inconsistent about MIME types, so accept on extension too.
  return [...ALLOWED.values()].some((exts) => exts.includes(ext));
}

export async function POST(req: NextRequest) {
  try {
    if (!features.s3) {
      return NextResponse.json(
        {
          error:
            "File storage is not configured. Set AWS_S3_BUCKET_NAME, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
        },
        { status: 503 },
      );
    }

    const formData = await req.formData();
    const file = formData.get("file");

    if (!file || typeof file === "string") {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    if (!isAllowed(file.type, file.name)) {
      return NextResponse.json(
        {
          error:
            "Unsupported file type. Upload a PDF, DOCX, TXT, MD or CSV file. " +
            "Legacy .doc files must be converted to .docx or PDF first.",
        },
        { status: 400 },
      );
    }

    if (file.size > GUARDRAIL_CONFIG.MAX_FILE_BYTES) {
      return NextResponse.json(
        {
          error: `File exceeds the ${Math.round(
            GUARDRAIL_CONFIG.MAX_FILE_BYTES / 1024 / 1024,
          )}MB limit.`,
        },
        { status: 400 },
      );
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const { key, url } = await uploadFileToS3(buffer, file.name, file.type);

    // Ingest immediately so the document is queryable the moment upload returns,
    // rather than on the user's next message.
    const ingest = await ingestFile({ name: file.name, key, type: file.type });

    return NextResponse.json({
      success: ingest.status !== "failed",
      file: { name: file.name, size: file.size, type: file.type, key, url },
      ingest,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Upload failed";
    console.error("[upload] failed:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
