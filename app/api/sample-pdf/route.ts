import { NextResponse } from "next/server";
import { features } from "@/lib/config";
import { getPresignedUrl, listUploads } from "@/lib/s3";

export const runtime = "nodejs";

/**
 * Hand back a demo document from the bucket.
 *
 * The previous version hardcoded one object key, which broke as soon as that
 * object was removed. This picks the most recent upload instead.
 */
export async function GET() {
  try {
    if (!features.s3) {
      return NextResponse.json({ error: "S3 is not configured." }, { status: 503 });
    }

    const uploads = await listUploads();
    const pdfs = uploads
      .filter((o) => o.key.toLowerCase().endsWith(".pdf"))
      .sort((a, b) => (b.lastModified ?? "").localeCompare(a.lastModified ?? ""));

    if (pdfs.length === 0) {
      return NextResponse.json({ error: "No PDF documents found in the bucket." }, { status: 404 });
    }

    const target = pdfs[0];
    return NextResponse.json({
      url: await getPresignedUrl(target.key, 3600),
      key: target.key,
      fileName: target.key.split("/").pop(),
      size: target.size,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[sample-pdf] failed:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
