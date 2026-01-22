import { processMultipleFiles } from "@/lib/document-processor";
import {
  addDocumentsToVectorStore,
  runHybridRag,
  hasDocuments,
  REFUSAL,
} from "@/lib/rag-agent";
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
        console.error("Error processing files:", error);
      }
    }

    try {
      // Check if we have documents
      const docsExist = await hasDocuments();

      if (!docsExist) {
        return streamResponse(REFUSAL);
      }

      const answer = await runHybridRag(message, history);
      return streamResponse(answer);

    } catch (error: unknown) {
      console.error("RAG Error:", error);
      return streamResponse(REFUSAL);
    }
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    console.error("Chat Error:", err);
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}


/**
 * Stream a response as SSE
 */
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
