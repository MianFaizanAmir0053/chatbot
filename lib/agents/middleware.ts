import {
  ClearToolUsesEdit,
  contextEditingMiddleware,
  createMiddleware,
  modelCallLimitMiddleware,
  modelFallbackMiddleware,
  modelRetryMiddleware,
  openAIModerationMiddleware,
  piiMiddleware,
  summarizationMiddleware,
  todoListMiddleware,
  toolCallLimitMiddleware,
  toolRetryMiddleware,
} from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { GUARDRAIL_CONFIG, env, features } from "../config";
import { getFallbackModels, getModel } from "../models";

/**
 * The agent harness.
 *
 * Every layer here is a first-party LangChain middleware rather than something
 * hand-rolled: planning, PII handling, moderation, retries, fallbacks, budget
 * ceilings and context management are all solved problems, and the library's
 * implementations are the ones that stay correct as the agent loop evolves.
 *
 * Order matters. Middleware wraps the model call from the outside in, so the
 * cheap deterministic guards sit outermost and the expensive ones run last.
 */
export function buildMiddleware(options: { enableTodos?: boolean } = {}) {
  const { enableTodos = true } = options;
  const middleware = [];

  /* --- Budget ceilings: hard stops on runaway loops (cheapest guard first) --- */

  middleware.push(
    modelCallLimitMiddleware({
      runLimit: GUARDRAIL_CONFIG.MAX_MODEL_CALLS_PER_RUN,
      // "end" lets the agent finish with what it has rather than throwing away
      // a partially-complete answer the user already waited for.
      exitBehavior: "end",
    }),
  );

  middleware.push(
    toolCallLimitMiddleware({
      runLimit: GUARDRAIL_CONFIG.MAX_TOOL_CALLS_PER_RUN,
      exitBehavior: "end",
    }),
  );

  /* --- Planning: multi-step decomposition into a visible todo list --- */

  if (enableTodos) {
    // Exposes a `write_todos` tool and tracks `todos` in agent state. The system
    // prompt it injects tells the model to use it only for genuinely multi-step
    // work, so simple questions don't pay the planning tax.
    middleware.push(todoListMiddleware());
  }

  /* --- Resilience: retries and cross-provider fallback --- */

  middleware.push(
    modelRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2,
      onFailure: "continue",
    }),
  );

  middleware.push(
    toolRetryMiddleware({
      maxRetries: 2,
      backoffFactor: 2,
      onFailure: "continue",
    }),
  );

  // Only meaningful when a second provider is actually configured.
  const fallbacks = getFallbackModels("pro");
  if (fallbacks.length > 0) {
    middleware.push(modelFallbackMiddleware(...fallbacks));
  }

  /* --- Context management: keep long conversations inside the window --- */

  middleware.push(
    contextEditingMiddleware({
      edits: [
        // Retrieval tool results are large. Once the conversation grows past the
        // trigger, older tool outputs are cleared while their conclusions
        // survive in the assistant messages that referenced them.
        new ClearToolUsesEdit({
          trigger: { tokens: 60000 },
          keep: { messages: 4 },
          clearToolInputs: false,
        }),
      ],
    }),
  );

  middleware.push(
    summarizationMiddleware({
      model: getModel("fast", { temperature: 0 }),
      trigger: { tokens: 100000 },
      keep: { messages: 8 },
    }),
  );

  /* --- Privacy: redact PII before it reaches a model or the transcript --- */

  // Redaction rather than blocking: a user pasting an email into a question
  // should still get an answer, just without the address leaving the process.
  for (const piiType of ["email", "credit_card", "ip"] as const) {
    middleware.push(
      piiMiddleware(piiType, {
        strategy: piiType === "credit_card" ? "block" : "redact",
        applyToInput: true,
        applyToOutput: true,
        applyToToolResults: true,
      }),
    );
  }

  /* --- Safety: content moderation on input and output --- */

  if (features.moderation) {
    middleware.push(
      // Wrapped: the moderation API is a network dependency, and an outage or
      // quota error there must not take down chat entirely.
      resilient(
        openAIModerationMiddleware({
          model: new ChatOpenAI({ model: "gpt-5.6-luna", apiKey: env.OPENAI_API_KEY }),
          checkInput: true,
          checkOutput: true,
          checkToolResults: true,
          exitBehavior: "end",
          violationMessage:
            "I can't help with that request. Ask me about your documents and I'll do my best.",
        }),
        "moderation",
      ),
    );
  }

  /* --- Telemetry: structured visibility into what the agent actually did --- */

  middleware.push(agentTelemetryMiddleware());

  return middleware;
}

/** Hook names on an AgentMiddleware that may perform I/O and therefore fail. */
const HOOKS = [
  "beforeAgent",
  "beforeModel",
  "afterModel",
  "afterAgent",
  "wrapModelCall",
  "wrapToolCall",
] as const;

type AnyMiddleware = Record<string, unknown>;

/**
 * Make a middleware's failures non-fatal.
 *
 * Middleware that calls out to a network service — moderation being the obvious
 * case — will throw on quota exhaustion or an outage, and an unhandled throw
 * inside a hook aborts the whole agent run. Degrading one guardrail is far
 * better than returning a 500 for every message, so failures are logged and the
 * hook is treated as a no-op.
 *
 * Wrapping happens per-hook so the middleware's own control-flow returns
 * (a moderation block, for instance) still work normally.
 */
function resilient<T>(middleware: T, label: string): T {
  const source = middleware as AnyMiddleware;
  const wrapped: AnyMiddleware = Object.create(
    Object.getPrototypeOf(source) as object | null,
  );
  Object.assign(wrapped, source);

  for (const hookName of HOOKS) {
    const entry = source[hookName];
    if (!entry) continue;

    // Hooks come in two shapes: a bare function, or { hook, canJumpTo } where
    // canJumpTo declares the control-flow targets the middleware may jump to.
    const isDescriptor = typeof entry === "object" && typeof (entry as AnyMiddleware).hook === "function";
    const original = isDescriptor ? (entry as AnyMiddleware).hook : entry;
    if (typeof original !== "function") continue;

    const guarded = async (...args: unknown[]) => {
      try {
        return await (original as (...a: unknown[]) => unknown).apply(source, args);
      } catch (error) {
        console.error(`[middleware:${label}] ${hookName} failed, continuing without it:`, error);
        // wrapModelCall / wrapToolCall receive the continuation as their last
        // argument; skipping the middleware means invoking it directly.
        if (hookName.startsWith("wrap")) {
          const handler = args.at(-1);
          if (typeof handler === "function") {
            return (handler as (a: unknown) => unknown)(args[0]);
          }
        }
        return undefined;
      }
    };

    wrapped[hookName] = isDescriptor
      ? { ...(entry as AnyMiddleware), hook: guarded }
      : guarded;
  }

  return wrapped as T;
}

/**
 * Logs each model turn and the tools it chose.
 *
 * Custom middleware exists mainly so agent behaviour is debuggable in
 * production without attaching a tracer — LangSmith gives the deep view, this
 * gives the always-on one.
 */
function agentTelemetryMiddleware() {
  return createMiddleware({
    name: "AgentTelemetry",
    afterModel: (state) => {
      const last = state.messages?.at(-1);
      const toolCalls =
        (last as { tool_calls?: Array<{ name: string }> } | undefined)?.tool_calls ?? [];
      if (toolCalls.length > 0) {
        console.log(`[agent] tools: ${toolCalls.map((t) => t.name).join(", ")}`);
      }
      return undefined;
    },
  });
}
