import { processMultipleFiles } from "@/lib/document-processor";
import { addDocumentsToVectorStore, hasDocuments, REFUSAL } from "@/lib/rag-agent";
import { runLangGraphHybridBeta } from "@/lib/langgraph-hybrid-beta";
import { NextRequest } from "next/server";

interface HistoryItem {
  role: "user" | "assistant";
  content: string;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const message: string = body?.message ?? "";
    const history: HistoryItem[] = body?.history ?? [];
    const files = body?.files ?? [];

    // Process uploaded files and add to vector store
    if (files && files.length > 0) {
      try {
        const processedDocs = await processMultipleFiles(files);
        if (processedDocs.length > 0) {
          await addDocumentsToVectorStore(processedDocs);
        }
      } catch (error) {
        console.error("[chat-beta] Error processing files:", error);
      }
    }

    // Ensure we have documents before running RAG
    const docsExist = await hasDocuments();
    if (!docsExist) {
      return streamResponse(REFUSAL);
    }

    // Run the beta LangGraph hybrid RAG
    const answer = await runLangGraphHybridBeta(message, history);
    return streamResponse(answer);
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    console.error("[chat-beta] Chat Error:", err);
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

function streamResponse(content: string): Response {
  const stream = new ReadableStream({
    async start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          `data: ${JSON.stringify({ chunk: content })}\n\n`
        )
      );
      controller.enqueue(
        new TextEncoder().encode(
          `data: ${JSON.stringify({ done: true })}\n\n`
        )
      );
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
