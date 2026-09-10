import { NextRequest, NextResponse } from "next/server";
import { scopeChainFor } from "@/lib/conversations/store";
import { knowledgeBaseStatus, removeDocument } from "@/lib/ingest/pipeline";

export const runtime = "nodejs";

/**
 * List the knowledge base, optionally narrowed to one conversation.
 *
 * Without `threadId` this is the whole corpus, which is what the dashboard
 * shows. With one it is that conversation's documents plus anything unscoped.
 */
export async function GET(req: NextRequest) {
  try {
    const threadId = req.nextUrl.searchParams.get("threadId");
    // The chain, so a fork lists the documents it inherited alongside its own —
    // the dialog has to show what the agent will actually read.
    const scope = threadId ? await scopeChainFor(threadId) : undefined;
    return NextResponse.json(await knowledgeBaseStatus(scope));
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
    // Scoped when the caller says which conversation, so removing a document
    // from one chat leaves an identically-named document in another alone.
    const threadId = req.nextUrl.searchParams.get("threadId") ?? undefined;
    await removeDocument(source, threadId);
    return NextResponse.json({ success: true, removed: source, threadId });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Delete failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
