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
import { AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { FALLBACK_MODELS, GUARDRAIL_CONFIG, SUBAGENT_CONFIG, env, features } from "../config";
import { isWorthRetrying } from "../credential-health";
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
export interface MiddlewareOptions {
  enableTodos?: boolean;
  /**
   * The run delegates research to subagents.
   *
   * Changes three things here and nothing else: the budgets become
   * whole-turn budgets rather than supervisor-only ones, the tool limit gains a
   * per-delegation ceiling, and delegated findings are protected from context
   * clearing. The delegation tool itself is built in buildAgent, alongside the
   * researchers' own tools, because it needs the run's evidence collector.
   */
  deepAgents?: boolean;
  /**
   * Names of the tools this agent was built with.
   *
   * Supplied by the caller because middleware cannot read the agent's tool
   * list, and only exact names make repairing a mangled one safe.
   */
  toolNames?: string[];
}

export function buildMiddleware(options: MiddlewareOptions = {}) {
  const { enableTodos = true, deepAgents: delegating = false, toolNames = [] } = options;
  const middleware = [];

  /* --- Budget ceilings: hard stops on runaway loops (cheapest guard first) --- */

  middleware.push(
    modelCallLimitMiddleware({
      // When delegating this is the budget for the entire turn, not the
      // supervisor's share of it. The counter lives in agent state and subagents
      // inherit that state, so this limiter sees the researchers' calls as well
      // as the supervisor's — which makes it a real ceiling on the whole
      // fan-out, and makes a separate per-branch ceiling both redundant and
      // actively harmful (a branch would start counting from the supervisor's
      // total and refuse itself before running).
      runLimit: delegating
        ? SUBAGENT_CONFIG.TURN_MODEL_CALLS
        : GUARDRAIL_CONFIG.MAX_MODEL_CALLS_PER_RUN,
      // "end" lets the agent finish with what it has rather than throwing away
      // a partially-complete answer the user already waited for.
      exitBehavior: "end",
    }),
  );

  // `end` stops the run on the call that exceeded the limit, and is documented
  // to raise NotImplementedError when a message carries several tool calls.
  // `continue` is the safer ceiling for a delegating turn: the calls over the
  // limit come back as errors and the model decides how to finish, so a fan-out
  // that overruns keeps the branches it did complete instead of losing them.
  const toolExit = delegating ? "continue" : "end";

  middleware.push(
    toolCallLimitMiddleware({
      runLimit: delegating
        ? SUBAGENT_CONFIG.TURN_TOOL_CALLS
        : GUARDRAIL_CONFIG.MAX_TOOL_CALLS_PER_RUN,
      exitBehavior: toolExit,
    }),
  );

  // A per-tool ceiling on delegation itself. The run-wide tool limit above
  // counts searches and delegations alike, so without this a supervisor could
  // spend its whole tool budget fanning out — and one such call starts a whole
  // batch of subagents rather than performing a single retrieval.
  if (delegating) {
    middleware.push(
      toolCallLimitMiddleware({
        toolName: "delegate_research",
        // One above the ceiling, because the ceiling itself is enforced inside
        // the tool now. This middleware blocks with a generic error and no
        // instruction, and a supervisor that hit it ended the turn with no
        // answer at all — twice, with findings already gathered. The tool
        // refuses the same call with "answer now from what you have", which is
        // the outcome worth having, so it must be the one the model sees.
        // Kept installed one round higher as a backstop for a model that
        // ignores that instruction and calls again regardless.
        runLimit: SUBAGENT_CONFIG.MAX_DELEGATION_ROUNDS + 1,
        exitBehavior: "continue",
      }),
    );
  }

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
      maxRetries: 4,
      backoffFactor: 2,
      // Rate limits, not transport blips, are what actually fails these calls:
      // free and trial tiers meter per minute, so the default sub-second
      // backoff retries inside the same closed window and exhausts every
      // attempt without ever waiting long enough to be let back in. Starting
      // at two seconds spans the window; jitter stops parallel calls in one
      // agent turn from retrying in lockstep.
      initialDelayMs: 2_000,
      maxDelayMs: 30_000,
      jitter: true,
      onFailure: "continue",
      // Do not spend the backoff on an answer that will never change.
      //
      // The default retries anything not explicitly stamped non-retryable, and
      // `@langchain/openai` stamps only 401, 404 and 400s about tool calls. A
      // 400 saying the *account* is refused — "Organization has been
      // restricted" — is left unstamped, so it was retried four times over
      // roughly thirty seconds before the fallback chain was reached at all.
      // With most keys in that state, that was the dominant cost of a request.
      //
      // Only refusals of the credential itself are excluded here. Everything
      // ambiguous still retries, because the backoff exists for rate limits and
      // those are precisely the errors worth waiting out.
      retryOn: (error: Error) => isWorthRetrying(error),
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
          // Never clear delegated findings. A `delegate_research` result is not
          // a retrieval that can be re-run — it is the distilled output of a
          // whole batch of subagents, and it is what the supervisor synthesises
          // from. Clearing it discards every branch that produced it while
          // leaving the run looking healthy: the supervisor writes a confident
          // answer with the evidence for it deleted, which is the exact failure
          // delegation was introduced to prevent. `write_todos` is excluded for
          // the same reason at lower stakes — losing the plan mid-run makes the
          // agent re-plan and repeat work it has already done.
          excludeTools: ["delegate_research", "write_todos"],
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
          model: new ChatOpenAI({
            model: FALLBACK_MODELS.FAST,
            apiKey: env.OPENAI_API_KEY,
            ...(env.OPENAI_BASE_URL ? { configuration: { baseURL: env.OPENAI_BASE_URL } } : {}),
          }),
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

  /* --- Repair: recover calls a model wrapped in its own template --- */

  // Before telemetry, so the log records the tool that will actually run rather
  // than the mangled name the model wrote.
  if (toolNames.length > 0) {
    middleware.push(toolCallRepairMiddleware(toolNames));
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
/**
 * Recover a tool call whose name arrived wrapped in the model's own template.
 *
 * Some models emit calls in a template their gateway only half-translates, and
 * what reaches the agent is a real call with valid arguments under a name like
 * `uncensored_tool_call>list_documents`. No such tool exists, so the call fails,
 * the model is told so, and the turn spends another round trip re-asking for
 * something it already asked for correctly.
 *
 * The arguments are intact and the intended tool is named right there, so the
 * call is repairable. Matching requires the real name to sit at a boundary in
 * the mangled one — a separator or the end — rather than merely appearing
 * somewhere inside it, because a loose match between tools with overlapping
 * names would route a call to the wrong one, which is far worse than failing it.
 * Ambiguous names are left alone for the same reason.
 */
/** A tool name is matched literally, so any regex metacharacter in it is inert. */
function escapeRegExp(literal: string): string {
  return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * The real tool a mangled name refers to, or null when it cannot be known.
 *
 * The real name must sit at a boundary of the mangled one and run to its end —
 * `uncensored_tool_call>list_documents` resolves, `list_documents_v2` does not.
 * Merely appearing somewhere inside is not enough: a loose match between tools
 * with overlapping names would route a call to the wrong tool, which is far
 * worse than failing it, since the model is told about a failure and can
 * correct itself but is told nothing when the wrong tool answers.
 *
 * Ambiguity returns null for the same reason. Exported for the test that pins
 * these boundaries down.
 */
export function repairToolName(mangled: string, known: string[]): string | null {
  if (known.includes(mangled)) return null;
  // Longest first, so `search_documents` wins over `search` when a name ends
  // with both.
  const matches = [...known]
    .sort((a, b) => b.length - a.length)
    .filter((name) => new RegExp(`(?:^|[^A-Za-z0-9_])${escapeRegExp(name)}$`).test(mangled));
  return matches.length === 1 ? matches[0] : null;
}

export function toolCallRepairMiddleware(toolNames: string[]) {
  const known = toolNames;

  return createMiddleware({
    name: "ToolCallRepair",
    afterModel: (state) => {
      const last = state.messages?.at(-1);
      if (!AIMessage.isInstance(last) || !last.id) return undefined;
      const calls = last.tool_calls;
      if (!calls?.length) return undefined;

      let repaired = false;
      const fixed = calls.map((call) => {
        const real = repairToolName(call.name, known);
        if (!real) return call;
        console.warn(`[agent] repaired mangled tool name: ${call.name} -> ${real}`);
        repaired = true;
        return { ...call, name: real };
      });

      if (!repaired) return undefined;

      // A new message carrying the same id, so the reducer replaces the model's
      // turn rather than appending a second copy of it. Cloned rather than
      // mutated because the original may already have been checkpointed.
      return {
        messages: [
          new AIMessage({
            id: last.id,
            content: last.content,
            tool_calls: fixed,
            invalid_tool_calls: last.invalid_tool_calls,
            additional_kwargs: last.additional_kwargs,
            response_metadata: last.response_metadata,
            usage_metadata: last.usage_metadata,
            name: last.name,
          }),
        ],
      };
    },
  });
}

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
