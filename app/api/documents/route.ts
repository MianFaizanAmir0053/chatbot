import { NextRequest, NextResponse } from "next/server";
import { knowledgeBaseStatus, removeDocument } from "@/lib/ingest/pipeline";

export const runtime = "nodejs";

/** List what is currently in the knowledge base. */
export async function GET() {
  try {
    return NextResponse.json(await knowledgeBaseStatus());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to read knowledge base";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** Remove a document and all of its chunks from the vector store. */
export async function DELETE(req: NextRequest) {
  try {
    const source = req.nextUrl.searchParams.get("source");
    if (!source) {
      return NextResponse.json({ error: "Missing 'source' query parameter" }, { status: 400 });
    }
    await removeDocument(source);
    return NextResponse.json({ success: true, removed: source });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Delete failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
