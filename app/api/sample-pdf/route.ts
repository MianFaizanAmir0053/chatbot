import { getPresignedUrl } from "@/lib/s3";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    // The file key for the motorcycle manual in S3
    const fileKey = "uploads/1768592884514-motorcycle-operator-manual.pdf";

    // Generate a fresh presigned URL (valid for 1 hour)
    const url = await getPresignedUrl(fileKey, 3600);

    return NextResponse.json({ url, fileName: "motorcycle-operator-manual.pdf" });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("Error generating presigned URL:", error);
    return NextResponse.json(
      { error: `Failed to generate presigned URL: ${errorMessage}` },
      { status: 500 }
    );
  }
}
