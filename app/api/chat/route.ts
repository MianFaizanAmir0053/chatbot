import { NextRequest } from "next/server";
import { randomUUID } from "crypto";
import { AIMessage, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { buildAgent, runConfig } from "@/lib/agents/agent";
import {
  ANSWER_NODE,
  isAnswerChunk,
  isNested,
  visibleText,
} from "@/lib/agents/stream-filter";
import { appendMessages, getConversation, scopeChainFor } from "@/lib/conversations/store";
import { hasReasoningProvider } from "@/lib/config";
import { refusedCredentials } from "@/lib/credential-health";
import { ChatRequestSchema, guardInput } from "@/lib/guardrails/input";
import {
  checkGroundedness,
  hasToolCallMarkup,
  normaliseCitations,
  stripInvalidCitations,
  stripToolCallMarkup,
  validateCitations,
} from "@/lib/guardrails/output";
import { ingestFiles } from "@/lib/ingest/pipeline";

export const runtime = "nodejs";
// The agent loop can run well past the default serverless budget on multi-step
// questions, so ask for the maximum the platform will grant.
export const maxDuration = 300;

/**
 * How much of a stored transcript is replayed into a thread whose checkpoint
 * is gone. Every replayed turn is context the model pays for on this turn, so
 * this trades recall against cost; the most recent turns carry nearly all of
 * the referential weight ("it", "the second one") that replay exists to serve.
 */
const MAX_REPLAYED_TURNS = 12;

/**
 * One structured line per turn, for finding the failures that do not throw.
 *
 * Every defect worth fixing in this route has been silent: an answer that read
 * perfectly while citing nothing, a groundedness score of 1 awarded because the
 * judge was handed no evidence, a planner's JSON spliced into the reply, a
 * delegation that fanned out to nothing. None produced an error, and each was
 * found either by reading a trace after the fact or by a test written once the
 * damage was already understood.
 *
 * A line per turn makes that class visible without either. It is JSON on one
 * line so a log search can filter on it — `grep '"cited":0'` finds every
 * uncited answer, `'"grounded":-1'` every turn nothing could verify.
 */
function logTurn(fields: Record<string, unknown>) {
  console.log(`[turn] ${JSON.stringify(fields)}`);
}

/* ------------------------------------------------------------------ *
 * SSE helpers
 * ------------------------------------------------------------------ */

type Emit = (event: string, data: unknown) => void;

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

  const { message, files, threadId, mode, webSearch, thinking, deepAgents } = parsed.data;
  const thread = threadId || randomUUID();

  return sseStream(async (emit, signal) => {
    const startedAt = Date.now();
    emit("thread", { threadId: thread });

    /* --- Input guardrails, before any spend --- */
    const verdict = await guardInput(message, clientKey(req));
    if (!verdict.allowed) {
      emit("blocked", { reason: verdict.reason, signals: verdict.signals });
      emit("token", { text: verdict.reason });
      // Kept in the transcript so reopening the thread shows why it stopped
      // rather than an unexplained gap.
      const at = new Date().toISOString();
      await appendMessages(thread, [
        { role: "user", content: message, createdAt: at },
        { role: "assistant", content: verdict.reason ?? "Blocked", blocked: true, createdAt: at },
      ]);
      logTurn({ thread, ms: Date.now() - startedAt, outcome: "blocked", signals: verdict.signals });
      return;
    }
    if (verdict.signals.length > 0) {
      console.warn(`[chat] input signals (allowed): ${verdict.signals.join(", ")}`);
    }

    // Recorded before the run, not after it: a question whose answer is
    // abandoned, aborted or lost to a crash still has to be findable in the
    // conversation list, which is the only way back to the thread.
    await appendMessages(thread, [
      { role: "user", content: message, createdAt: new Date().toISOString() },
    ]);

    /* --- Ingest any newly attached files into the vector store --- */
    if (files.length > 0) {
      emit("status", { stage: "ingesting", detail: `Processing ${files.length} file(s)` });
      // Attached mid-conversation, so they belong to this conversation.
      const results = await ingestFiles(files.map((f) => ({ ...f, threadId: thread })));
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
      // This conversation and every ancestor it was forked from. Resolving the
      // chain here rather than passing the bare id is what makes a fork usable:
      // its documents live under the parent's id, so a fork scoped to itself
      // alone would find nothing and confidently report that the documents do
      // not cover the question.
      threadId: await scopeChainFor(thread),
      // Branch progress, forwarded as it happens.
      //
      // The delegation tool returns one value after its slowest branch, so
      // without this the interface could say only that research was under way
      // — measured at around eighty seconds of it, most spent with findings
      // already in hand and no way to show them.
      onBranch: (e) => {
        if (e.phase === "start") {
          emit("delegation", { id: e.id, agent: e.researcher, task: e.question });
        } else {
          emit("delegation_done", { id: e.id, outcome: e.outcome, ms: e.ms });
        }
      },
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
    /**
     * The model wrote a tool call as text rather than calling the tool.
     *
     * Tracked for the whole turn because the consequence outlives the chunk it
     * was spotted in: the call never ran, so whatever the model wrote alongside
     * it was written without the evidence it was reaching for.
     */
    let unexecutedToolCall = false;
    /**
     * Root-graph text that was withheld because it came from a node other than
     * the one that writes the answer.
     *
     * Expected to be non-zero — tools and middleware think out loud. It only
     * means something when the answer streamed nothing at all, which is the
     * signature of the node allowlist having gone stale rather than of a
     * provider that does not send deltas.
     */
    let suppressedRootText = 0;

    const config = runConfig({ threadId: thread, signal, mode: thinking, deepAgents });

    /**
     * Rebuild the agent's memory when a reopened thread has none.
     *
     * The checkpointer is per-process: restart the server, or reopen a
     * conversation stored yesterday, and the graph has no record of it while
     * the user is looking at the whole transcript. Replaying the stored turns
     * into this run's input restores the context the user can see, so a
     * follow-up like "and the second one?" still resolves.
     */
    const priorMessages: BaseMessage[] = [];
    if (threadId) {
      let checkpointed = 0;
      try {
        // `getState` is typed against the graph's own state shape, which is not
        // exported in a form this route can name; only the message count is
        // needed, so it is read structurally.
        const state = (await agent.getState(config)) as
          | { values?: { messages?: unknown[] } }
          | undefined;
        checkpointed = state?.values?.messages?.length ?? 0;
      } catch {
        // No checkpoint for this thread; treat it as empty and replay.
      }

      if (checkpointed === 0) {
        const stored = await getConversation(thread);
        // The turn just recorded above is the incoming question — replaying it
        // would send it twice.
        const earlier = (stored?.messages ?? []).slice(0, -1).slice(-MAX_REPLAYED_TURNS);
        for (const turn of earlier) {
          priorMessages.push(
            turn.role === "user" ? new HumanMessage(turn.content) : new AIMessage(turn.content),
          );
        }
        if (priorMessages.length > 0) {
          emit("status", {
            stage: "restoring context",
            detail: `${priorMessages.length} earlier turn(s)`,
          });
        }
      }
    }

    const stream = await agent.stream(
      { messages: [...priorMessages, new HumanMessage(message)] },
      { ...config, streamMode: ["updates", "messages"] },
    );

    for await (const chunk of stream) {
      const [streamMode, payload] = chunk as [string, unknown];

      if (streamMode === "messages") {
        // Token-level streaming of the model's visible output.
        const [token, meta] = payload as [
          { content?: unknown; getType?: () => string },
          Record<string, unknown> | undefined,
        ];

        // Every model call inside the run shares this stream, because the
        // callback manager is inherited: subagents delegated to, the query
        // planner inside a retrieval tool, the summariser, the moderation pass.
        // None of them are the answer, and emitting them splices raw internal
        // working into what the user is reading. Only the supervisor's own
        // model node writes the answer, so that is what is streamed.
        const type = token?.getType?.();
        if (type === "ai") {
          const text = visibleText(token.content);
          if (!text) continue;
          if (!isAnswerChunk(meta)) {
            // Counted so a stale node name is visible in the logs rather than
            // silently costing live streaming for every answer.
            if (!isNested(meta)) suppressedRootText++;
            continue;
          }
          answer += text;
          emit("token", { text });

          // A tool call the model wrote as text instead of calling.
          //
          // Retracted for the same reason a preamble is: it is not the answer.
          // The difference is what it implies — the call never ran, so anything
          // the model wrote around it was composed without the result it was
          // asking for. That is recorded, so a turn that never got its evidence
          // is not later presented as a researched answer.
          //
          // Tested against the accumulated answer rather than the chunk: the
          // opening tag arrives split across several tokens, so by the time the
          // shape is recognisable some of it has already been streamed. Hence a
          // retraction rather than a filter.
          if (!unexecutedToolCall && hasToolCallMarkup(answer)) {
            unexecutedToolCall = true;
            console.warn("[chat] model emitted a tool call as text; it was never executed");
            answer = stripToolCallMarkup(answer);
            emit("revised_answer", { text: answer });
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
            // Anything the model wrote before deciding to call a tool was a
            // preamble, not the answer.
            //
            // Models narrate — "I'll first check your documents", "Since your
            // documents don't cover this, I'll now search the web" — and every
            // AI turn's text is streamed, so those announcements accumulate in
            // front of the real answer. They are also pure noise here: the
            // interface shows each search as it happens, so the commentary
            // describes something the user is already watching.
            //
            // A tool call is the signal that what came before it was not the
            // answer, and it is a reliable one: the model does not call a tool
            // after it has finished answering. The prompt forbids narrating as
            // well, but only this survives a model that narrates anyway.
            if (answer.trim()) {
              answer = "";
              emit("revised_answer", { text: "" });
            }

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
                // Announced by the tool itself, as each branch starts.
                //
                // It used to be announced from these arguments, which was the
                // only channel available but reported a plan rather than what
                // happened: branches appeared the moment the model asked for
                // them and all closed together when the last one finished, so
                // a branch that failed in five seconds looked identical to one
                // that ran for ninety. The verifier was worse — it is started
                // by the tool, not requested by the model, so it had to be
                // guessed at from a predicate.
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
            // Sanitised here too, not only on the token path: a provider that
            // returns the completion whole never streams a chunk, so the update
            // is the only place its unexecuted call would be seen.
            if (text && hasToolCallMarkup(text)) {
              if (!unexecutedToolCall) {
                unexecutedToolCall = true;
                console.warn("[chat] model emitted a tool call as text; it was never executed");
              }
              bufferedAnswer = stripToolCallMarkup(text);
            } else if (text) {
              bufferedAnswer = text;
            }
          }
        }
      }
    }

    // Nothing streamed, but the agent did produce an answer: send it as one
    // chunk rather than leaving the client with an empty assistant turn.
    if (!answer && bufferedAnswer) {
      if (suppressedRootText > 0) {
        // The run streamed root-graph text and none of it was the answer node's.
        // Either `createAgent` renamed that node, or the answer arrived only in
        // the update. Worth saying out loud: the answer below is correct, but
        // it appeared all at once instead of typing out.
        console.warn(
          `[chat] no tokens passed the answer-node filter (${suppressedRootText} chunk(s) ` +
            `withheld) — check that "${ANSWER_NODE}" is still the agent's model node`,
        );
      }
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

      // Two very different failures reach this point and they must not be
      // reported as one. If retrieval gathered evidence, the providers plainly
      // answered — the agent simply spent its turn researching and never wrote
      // the answer up, which a delegating run can do by chasing one gap after
      // another. Blaming that on a rate limit sends the user to check billing
      // for a problem that is entirely on this side, and hides the evidence the
      // run did collect.
      const researched = collector.documents.length > 0 || collector.searches.length > 0;

      const failure = 
        // A third cause, and the only one where the run was never given the
        // chance to answer: the model wrote its tool call in a template the
        // gateway does not parse, so the call was never made and generation
        // stopped waiting for a result that could not arrive. Reporting this as
        // a budget overrun or a rate limit would send the user to check the
        // wrong thing entirely — the fix is a different provider, not a
        // narrower question or a topped-up balance.
        unexecutedToolCall
          ? "The model tried to search but wrote the request in a format this provider " +
            "does not execute, so nothing was retrieved and it stopped waiting for a result. " +
            "This is a quirk of one provider — asking again will usually land on another one."
          : researched
          ? "The research completed but no answer was written — the turn ran out of budget " +
            `before synthesising. It gathered ${collector.documents.length} passage(s) across ` +
            `${collector.searches.length} search(es). Asking a narrower question usually gets ` +
            "an answer from the same evidence."
          : "Every configured model provider refused the request — most often a rate limit " +
            `or an exhausted balance. Details: ${detail}`;

      emit("error", { message: failure });

      // The sources are still worth sending: they are what the run actually
      // established, and they let the user see the research rather than only
      // the failure.
      if (researched) {
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
      }

      // Record the failure as a turn, rather than leaving the question
      // unanswered in the transcript.
      //
      // The question is stored before the run so an abandoned turn stays
      // findable. When the run then fails, this path used to return without
      // writing anything, so the stored thread kept a user message with no
      // reply — and the user, seeing nothing, asked again. A measured thread
      // ended up holding the same question three times with two of them
      // unanswered, all of it replayed into the model's context on every
      // subsequent turn as questions it had apparently ignored.
      //
      // Marked `blocked`, which is the existing flag for "this turn produced no
      // answer, and here is why" — the same treatment a refused input gets.
      await appendMessages(thread, [
        {
          role: "assistant",
          content: failure,
          blocked: true,
          createdAt: new Date().toISOString(),
        },
      ]);

      logTurn({
        thread,
        ms: Date.now() - startedAt,
        outcome: "no-answer",
        mode: deepAgents ? "delegated" : thinking,
        passages: collector.documents.length,
        webSources: collector.webResults.length,
        searches: collector.searches.length,
        unexecutedToolCall,
        refusedCredentials: refusedCredentials().length,
      });
      return;
    }

    /* --- Output guardrails --- */
    emit("status", { stage: "verifying" });

    // Repair near-miss marker formats before validating. A marker the validator
    // does not recognise is not caught as invalid — it is not seen at all, so
    // the answer passes with a broken citation still in the text.
    const normalised = normaliseCitations(answer);
    if (normalised !== answer) {
      console.warn("[chat] repaired non-standard citation markers");
      answer = normalised;
      emit("revised_answer", { text: answer });
    }

    const citations = validateCitations(answer, collector.documents, collector.webResults.length);
    if (!citations.valid) {
      console.warn(`[chat] invalid citation refs: ${citations.invalidRefs.join(", ")}`);
      const cleaned = stripInvalidCitations(answer, collector.documents, collector.webResults.length);
      emit("revised_answer", { text: cleaned });
      answer = cleaned;
    }

    // A substantive answer that cites nothing, when there was evidence to cite.
    //
    // Citation validation cannot catch this: it looks for markers and checks
    // that each resolves, so an answer with no markers has nothing invalid in
    // it and passes. The gap is not theoretical — a delegated turn whose
    // researcher returned malformed markers had them all stripped, and what
    // reached the user was a long, confident, entirely uncited answer that
    // every check called clean.
    if (
      citations.refs.length === 0 &&
      answer.trim().length > 400 &&
      (collector.documents.length > 0 || collector.webResults.length > 0)
    ) {
      console.warn(
        `[chat] answer cites nothing despite ${collector.documents.length} passage(s) and ` +
          `${collector.webResults.length} web source(s)`,
      );
      emit("warning", {
        message:
          "This answer does not cite any of the sources the run retrieved, so individual " +
          "claims cannot be traced back. Treat it as a summary rather than as sourced.",
      });
    }

    // Web sources are judged alongside the documents. Passing only documents
    // meant a web-researched answer reached the judge with no evidence at all,
    // which it treated as nothing to dispute and scored as fully grounded.
    const grounded = await checkGroundedness(answer, collector.documents, collector.webResults);
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
    const documents = collector.documents.map((d, i) => ({
      index: i + 1,
      source: String(d.doc.metadata?.source ?? "unknown"),
      section: d.doc.metadata?.section as string | undefined,
      page: d.doc.metadata?.page as number | undefined,
      score: d.score,
      excerpt: String(d.doc.metadata?.originalText ?? d.doc.pageContent).slice(0, 400),
    }));

    emit("sources", {
      documents,
      web: collector.webResults,
      searches: collector.searches,
    });

    /* --- Transcript --- */
    await appendMessages(thread, [
      {
        role: "assistant",
        content: answer,
        sources: documents,
        web: collector.webResults,
        groundedness: {
          score: grounded.score,
          verdict: grounded.verdict,
          passed: grounded.passed,
          unsupportedClaims: grounded.unsupportedClaims,
        },
        createdAt: new Date().toISOString(),
      },
    ]);

    logTurn({
      thread,
      ms: Date.now() - startedAt,
      mode: deepAgents ? "delegated" : thinking,
      web: webSearch,
      chars: answer.length,
      passages: collector.documents.length,
      webSources: collector.webResults.length,
      searches: collector.searches.length,
      cited: citations.refs.length,
      invalidCited: citations.invalidRefs.length,
      grounded: grounded.score,
      verdict: grounded.verdict,
      unexecutedToolCall,
      refusedCredentials: refusedCredentials().length,
    });
  });
}
