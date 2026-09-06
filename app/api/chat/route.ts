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

/**
 * True when a streamed chunk came from a nested graph rather than the supervisor.
 *
 * LangGraph builds a nested run's checkpoint namespace by appending to its
 * parent's, separated by `|`, so depth is a structural property of the run
 * rather than anything a model or a tool controls. The root graph's own nodes
 * are at depth zero or one; a researcher invoked inside `delegate_research` is
 * deeper.
 *
 * Erring towards treating a chunk as nested is the safe direction: the worst
 * case is that a token is not streamed live, and the answer is still delivered
 * whole from the update stream. The opposite mistake puts a researcher's raw
 * working in front of the user as if it were the answer.
 */
function isNested(meta: Record<string, unknown> | undefined): boolean {
  const ns = meta?.checkpoint_ns ?? meta?.langgraph_checkpoint_ns;
  if (typeof ns !== "string" || ns.length === 0) return false;
  return ns.split("|").filter(Boolean).length > 1;
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

  const { message, files, threadId, mode, webSearch, thinking, deepAgents } = parsed.data;
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
    emit("status", {
      stage: deepAgents
        ? "delegating research"
        : thinking === "deep"
          ? "deep research"
          : "thinking",
      detail: webSearch ? undefined : "documents only",
    });

    const { agent, collector } = buildAgent({
      enableTodos: mode === "agentic",
      webSearch,
      mode: thinking,
      deepAgents,
    });

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
    /** Tool calls already reported, so a middleware re-emit is not a second step. */
    const reportedCalls = new Set<string>();
    /** Tool results already reported, for the same reason. */
    const reportedResults = new Set<string>();

    const stream = await agent.stream(
      { messages: [new HumanMessage(message)] },
      {
        ...runConfig({ threadId: thread, signal, mode: thinking, deepAgents }),
        streamMode: ["updates", "messages"],
      },
    );

    for await (const chunk of stream) {
      const [streamMode, payload] = chunk as [string, unknown];


      if (streamMode === "messages") {
        // Token-level streaming of the model's visible output.
        const [token, meta] = payload as [
          { content?: unknown; getType?: () => string },
          Record<string, unknown> | undefined,
        ];

        // Subagents run inside the `task` tool, and the callback manager is
        // inherited, so their model output arrives on this same stream. Emitting
        // it would splice every researcher's internal working into the user's
        // answer. Nesting depth is the discriminator: LangGraph namespaces a
        // nested run by appending to its parent's, so anything deeper than the
        // root graph is a delegation and belongs only in the finding it returns.
        if (isNested(meta)) continue;

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
                tool_calls?: Array<{ id?: string; name: string; args: unknown }>;
                name?: string;
                tool_call_id?: string;
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
              // Middleware nodes re-emit the model's last message as their own
              // update, so one tool call arrives from model_request and again
              // from every after_model hook. Reporting each occurrence showed
              // the same step three times in the trace and made a single search
              // look like a loop. Identity is the call, not the node that
              // mentioned it.
              const id = call.id ?? `${call.name}:${JSON.stringify(call.args ?? {})}`;
              if (reportedCalls.has(id)) continue;
              reportedCalls.add(id);

              // A delegation is not an ordinary tool call and reporting it as
              // one hides the thing the user most wants to see: several `task`
              // calls in a single message are the fan-out, and rendering them
              // as five identical "task" rows says nothing about which
              // sub-questions are being researched or by whom.
              if (call.name === "delegate_research") {
                const args = (call.args ?? {}) as {
                  tasks?: Array<{ researcher?: string; question?: string }>;
                };
                // One call starts a whole batch, so report each branch
                // separately — the fan-out is precisely what the user is
                // waiting on, and a single "delegate_research" row hides it.
                (args.tasks ?? []).forEach((t, i) => {
                  emit("delegation", {
                    id: `${id}:${i}`,
                    agent: t.researcher ?? "researcher",
                    task: t.question ?? "",
                  });
                });
                continue;
              }

              emit("tool_call", { name: call.name, args: call.args, node });
            }
          } else if (type === "tool") {
            const id =
              last?.tool_call_id ?? `${last?.name}:${String(last?.content).slice(0, 60)}`;
            if (!reportedResults.has(id)) {
              reportedResults.add(id);
              if (last?.name === "delegate_research") {
                // The batch resolves as one tool message, so every branch it
                // started completes together. Close them all.
                emit("delegation_result", {
                  batchId: last?.tool_call_id ?? id,
                  chars: String(last?.content ?? "").length,
                });
              } else {
                emit("tool_result", { name: last?.name ?? "tool" });
              }
            }
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
