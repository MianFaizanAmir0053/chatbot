import { NextRequest, NextResponse } from "next/server"
import { uploadFileToS3 } from "@/lib/s3"

// Maximum file size (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024

// Allowed file types
const ALLOWED_TYPES = [
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
]

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const file = formData.get("file") as File | null

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      return NextResponse.json(
        { error: "Invalid file type. Only PDF, DOC, DOCX, and TXT files are allowed." },
        { status: 400 }
      )
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "File size exceeds 10MB limit" },
        { status: 400 }
      )
    }

    // Convert file to buffer
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Upload to S3
    const { key, url } = await uploadFileToS3(buffer, file.name, file.type)

    return NextResponse.json({
      success: true,
      file: {
        name: file.name,
        size: file.size,
        type: file.type,
        key,
        url,
      },
    })
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Upload failed"
    console.error("Upload error:", error)
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    )
  }
}
