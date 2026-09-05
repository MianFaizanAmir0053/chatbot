import { NextRequest } from "next/server";
import { randomUUID } from "crypto";
import { HumanMessage } from "@langchain/core/messages";
import { buildAgent, runConfig } from "@/lib/agents/agent";
import { hasReasoningProvider } from "@/lib/config";
import { ChatRequestSchema, guardInput } from "@/lib/guardrails/input";
import {
  checkGroundedness,
  stripInvalidCitations,
  validateCitations,
} from "@/lib/guardrails/output";
import { ingestFiles } from "@/lib/ingest/pipeline";

export const runtime = "nodejs";
// The agent loop can run well past the default serverless budget on multi-step
// questions, so ask for the maximum the platform will grant.
export const maxDuration = 300;

/* ------------------------------------------------------------------ *
 * SSE helpers
 * ------------------------------------------------------------------ */

type Emit = (event: string, data: unknown) => void;

/**
 * Pull the user-visible text out of a streamed chunk's content.
 *
 * `content` arrives either as a plain string or as an array of content blocks.
 * `createAgent` uses the block form, so matching on `typeof content === "string"`
 * silently discarded every token: the run completed, tools fired and the
 * groundedness judge scored an empty string as perfectly grounded, while the
 * client received no answer at all.
 *
 * Only `text` blocks are collected. Reasoning blocks are deliberately excluded —
 * they are not part of the answer and must not reach the transcript.
 */
function visibleText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  let out = "";
  for (const block of content) {
    if (typeof block === "string") {
      out += block;
      continue;
    }
    const { type, text } = (block ?? {}) as { type?: string; text?: unknown };
    if (type === "text" && typeof text === "string") out += text;
  }
  return out;
}

function sseStream(run: (emit: Emit, signal: AbortSignal) => Promise<void>): Response {
  const encoder = new TextEncoder();
  const controllerRef = new AbortController();

  const stream = new ReadableStream({
    async start(controller) {
      let closed = false;
      const emit: Emit = (event, data) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ event, ...(data as object) })}\n\n`));
        } catch {
          closed = true;
        }
      };

      try {
        await run(emit, controllerRef.signal);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("[chat] stream failed:", error);
        emit("error", { message });
      } finally {
        emit("done", {});
        closed = true;
        try {
          controller.close();
        } catch {
          /* already closed by the client disconnecting */
        }
      }
    },
    cancel() {
      // Client went away — abort the agent so we stop paying for tokens.
      controllerRef.abort();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

function clientKey(req: NextRequest): string {
  return (
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "anonymous"
  );
}

/* ------------------------------------------------------------------ *
 * Route
 * ------------------------------------------------------------------ */

export async function POST(req: NextRequest) {
  const parsed = ChatRequestSchema.safeParse(await req.json().catch(() => ({})));

  if (!parsed.success) {
    return Response.json(
      { error: "Invalid request", details: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  if (!hasReasoningProvider()) {
    return Response.json(
      {
        error:
          "No reasoning provider configured. Set DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY in .env.",
      },
      { status: 503 },
    );
  }

  const { message, files, threadId, mode } = parsed.data;
  const thread = threadId || randomUUID();

  return sseStream(async (emit, signal) => {
    emit("thread", { threadId: thread });

    /* --- Input guardrails, before any spend --- */
    const verdict = await guardInput(message, clientKey(req));
    if (!verdict.allowed) {
      emit("blocked", { reason: verdict.reason, signals: verdict.signals });
      emit("token", { text: verdict.reason });
      return;
    }
    if (verdict.signals.length > 0) {
      console.warn(`[chat] input signals (allowed): ${verdict.signals.join(", ")}`);
    }

    /* --- Ingest any newly attached files into the vector store --- */
    if (files.length > 0) {
      emit("status", { stage: "ingesting", detail: `Processing ${files.length} file(s)` });
      const results = await ingestFiles(files);
      emit("ingested", { results });

      const failed = results.filter((r) => r.status === "failed");
      if (failed.length > 0) {
        emit("warning", {
          message: failed.map((f) => `${f.source}: ${f.error}`).join("; "),
        });
      }
    }

    /* --- Run the agent --- */
    emit("status", { stage: "thinking" });

    const { agent, collector } = buildAgent({ enableTodos: mode === "agentic" });

    let answer = "";
    let lastTodos: unknown[] = [];
    /**
     * Final assistant text seen in the node updates.
     *
     * Not every OpenAI-compatible endpoint streams token deltas — some return
     * the completion whole, in which case "messages" mode yields only empty
     * chunks and the answer appears solely in the update payload. Tracking it
     * here means the transcript is correct either way.
     */
    let bufferedAnswer = "";

    const stream = await agent.stream(
      { messages: [new HumanMessage(message)] },
      { ...runConfig({ threadId: thread, signal }), streamMode: ["updates", "messages"] },
    );

    for await (const chunk of stream) {
      const [streamMode, payload] = chunk as [string, unknown];


      if (streamMode === "messages") {
        // Token-level streaming of the model's visible output.
        const [token] = payload as [{ content?: unknown; getType?: () => string }];
        const type = token?.getType?.();
        if (type === "ai") {
          const text = visibleText(token.content);
          if (text) {
            answer += text;
            emit("token", { text });
          }
        }
        continue;
      }

      if (streamMode === "updates") {
        const updates = payload as Record<string, { todos?: unknown[]; messages?: unknown[] }>;
        for (const [node, update] of Object.entries(updates ?? {})) {
          // Surface the plan as the agent writes and revises it.
          if (Array.isArray(update?.todos) && update.todos !== lastTodos) {
            lastTodos = update.todos;
            emit("todos", { todos: update.todos });
          }

          const last = update?.messages?.at?.(-1) as
            | {
                tool_calls?: Array<{ name: string; args: unknown }>;
                name?: string;
                content?: unknown;
                getType?: () => string;
              }
            | undefined;

          // Classify by message type first. `name` is not a tool marker:
          // createAgent stamps the agent's own name onto its AI messages, so
          // testing `name` before type reported every assistant turn as a tool
          // result and dropped the answer on the floor.
          const type = last?.getType?.();

          if (last?.tool_calls?.length) {
            for (const call of last.tool_calls) {
              emit("tool_call", { name: call.name, args: call.args, node });
            }
          } else if (type === "tool") {
            emit("tool_result", { name: last?.name ?? "tool" });
          } else if (type === "ai") {
            const text = visibleText(last?.content);
            if (text) bufferedAnswer = text;
          }
        }
      }
    }

    // Nothing streamed, but the agent did produce an answer: send it as one
    // chunk rather than leaving the client with an empty assistant turn.
    if (!answer && bufferedAnswer) {
      answer = bufferedAnswer;
      emit("token", { text: bufferedAnswer });
    }

    // When every provider refuses, the retry middleware reports the exhaustion
    // as ordinary message content. Returning that as an answer — or returning
    // nothing at all — hides an outage behind a blank reply, so raise it.
    if (!answer.trim() || /^Model call failed after \d+ attempts/.test(answer)) {
      const detail = answer.trim()
        ? answer.replace(/\s+/g, " ").slice(0, 300)
        : "The model returned no output.";
      console.error(`[chat] no usable answer: ${detail}`);
      emit("error", {
        message:
          "Every configured model provider refused the request — most often a rate limit " +
          `or an exhausted balance. Details: ${detail}`,
      });
      return;
    }

    /* --- Output guardrails --- */
    emit("status", { stage: "verifying" });

    const citations = validateCitations(answer, collector.documents);
    if (!citations.valid) {
      console.warn(`[chat] invalid citation refs: ${citations.invalidRefs.join(", ")}`);
      const cleaned = stripInvalidCitations(answer, collector.documents);
      emit("revised_answer", { text: cleaned });
      answer = cleaned;
    }

    const grounded = await checkGroundedness(answer, collector.documents);
    emit("groundedness", {
      score: grounded.score,
      verdict: grounded.verdict,
      passed: grounded.passed,
      unsupportedClaims: grounded.unsupportedClaims,
    });

    if (!grounded.passed) {
      emit("warning", {
        message:
          "This answer may not be fully supported by the source documents. Treat it with caution.",
      });
    }

    /* --- Provenance --- */
    emit("sources", {
      documents: collector.documents.map((d, i) => ({
        index: i + 1,
        source: d.doc.metadata?.source,
        section: d.doc.metadata?.section,
        page: d.doc.metadata?.page,
        score: d.score,
        excerpt: String(d.doc.metadata?.originalText ?? d.doc.pageContent).slice(0, 400),
      })),
      web: collector.webResults,
      searches: collector.searches,
    });
  });
}
