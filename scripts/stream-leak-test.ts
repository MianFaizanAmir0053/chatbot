/**
 * Can an auxiliary model call reach the user's answer? — no quota required.
 *
 * A run's token stream carries every model call made anywhere inside it: the
 * supervisor writing its reply, delegated researchers, and — the case this
 * exists for — helpers invoked from inside a tool. The query planner is one of
 * those, and `withStructuredOutput` returns its result as message *content* on
 * every provider configured here rather than as a tool call, so its raw
 * `{"variants":[...],"hypotheticalAnswer":"..."}` object was streamed to the
 * user, appended to the transcript and then scored for groundedness. The HyDE
 * probe inside it is an invented answer by design, so what the user saw read as
 * the system fabricating facts.
 *
 * Two independent defences now stop that, and both are checked here:
 *
 *   1. `structuredInvoke` tags its calls `langsmith:nostream`, which makes
 *      LangGraph drop the run from the stream entirely.
 *   2. The SSE route only forwards chunks from the node that writes the answer,
 *      so an untagged helper added later still cannot splice text into it.
 *
 * Every check has a counterpart that would fail if the check were vacuous: the
 * leak is proved *reachable* before it is proved blocked, and the real answer is
 * proved to still get through, because a filter that blocks everything would
 * otherwise look like a pass.
 *
 * Usage: npx tsx --env-file=.env scripts/stream-leak-test.ts
 */

import { createAgent } from "langchain";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  ANSWER_NODE,
  isAnswerChunk,
  isNested,
  visibleText,
} from "../lib/agents/stream-filter";
import { structuredInvoke } from "../lib/structured";

/** Text that must never reach the user. Distinctive so a match is unambiguous. */
const PROBE = "__LEAK_PROBE__";
/** Shaped like the real query plan, which is what actually leaked. */
const AUX_CONTENT = `{"variants":["${PROBE}"],"hypotheticalAnswer":"${PROBE}"}`;
const ANSWER = "The oil capacity is 3.4 litres.";
const NOSTREAM_TAG = "langsmith:nostream";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(70)}\n${s}\n${"=".repeat(70)}`);
}

/* ------------------------------------------------------------------ *
 * Fakes
 *
 * Purpose-built rather than borrowed: the behaviour under test is our stream
 * handling, and `FakeToolCallingModel` echoes its input as its answer, which
 * would put the probe into the answer by way of message history and prove
 * nothing about where a chunk came from.
 * ------------------------------------------------------------------ */

/** A model that returns exactly what it is told to, one scripted turn at a time. */
class ScriptedModel extends BaseChatModel {
  private turn = 0;
  constructor(private readonly script: Array<Partial<AIMessage> & { content: string }>) {
    super({});
  }
  _llmType() {
    return "scripted";
  }
  // createAgent requires a tool-binding model; the script decides the calls, so
  // the tools themselves are irrelevant here.
  bindTools() {
    return this;
  }
  async _generate(): Promise<ChatResult> {
    const step = this.script[Math.min(this.turn, this.script.length - 1)];
    this.turn++;
    const message = new AIMessage(step);
    return { generations: [{ text: step.content, message }], llmOutput: {} };
  }
}

/** Stands in for the query planner: an ordinary model call made inside a tool. */
class AuxModel extends BaseChatModel {
  constructor() {
    super({});
  }
  _llmType() {
    return "aux";
  }
  bindTools() {
    return this;
  }
  async _generate(): Promise<ChatResult> {
    return {
      generations: [{ text: AUX_CONTENT, message: new AIMessage(AUX_CONTENT) }],
      llmOutput: {},
    };
  }
}

/* ------------------------------------------------------------------ *
 * 1. The predicates
 * ------------------------------------------------------------------ */

function predicates() {
  banner("1. Which chunks count as the answer");

  const rootAnswer = { langgraph_node: ANSWER_NODE, langgraph_checkpoint_ns: "model_request:abc" };
  const rootTool = { langgraph_node: "tools", langgraph_checkpoint_ns: "tools:abc" };
  const rootMiddleware = {
    langgraph_node: "summarization.beforeModel",
    langgraph_checkpoint_ns: "summarization.beforeModel:abc",
  };
  const nestedAnswer = {
    langgraph_node: ANSWER_NODE,
    langgraph_checkpoint_ns: "tools:abc|model_request:def",
  };

  check("the supervisor's model node is the answer", isAnswerChunk(rootAnswer));
  // The regression: a helper inside a tool runs at the root graph's own depth,
  // so nesting alone cannot tell it apart from the supervisor.
  check("a model call inside a tool is not", !isAnswerChunk(rootTool), "the leaked case");
  check("nor is a middleware thinking out loud", !isAnswerChunk(rootMiddleware));
  check("nor is a delegated researcher", !isAnswerChunk(nestedAnswer));
  check("nor is a chunk with no metadata at all", !isAnswerChunk(undefined));

  // Depth still has to work on its own, since it is what keeps subagents out.
  check("depth alone still sees a tool as un-nested", !isNested(rootTool), "hence the node check");
  check("and sees a researcher as nested", isNested(nestedAnswer));

  check("text blocks are read", visibleText([{ type: "text", text: "hello" }]) === "hello");
  check(
    "reasoning is not part of the answer",
    visibleText([{ type: "reasoning", text: "hmm" }, { type: "text", text: "hi" }]) === "hi",
  );
}

/* ------------------------------------------------------------------ *
 * 2, 3 & 4. A real graph, with the hazard wired in
 *
 * Everything is asserted through LangGraph's own stream rather than through a
 * handler attached to the model, because that is how the route sees it. A
 * handler on the model observes different plumbing and can disagree with what
 * the stream actually carries — which is a fact about the harness, not about
 * whether a user would see the text.
 * ------------------------------------------------------------------ */

/**
 * Run the agent with a tool that makes `helper` call inside it, and report every
 * chunk the stream produced — both what the route would forward and what it
 * would not.
 */
async function runWithHelperInTool<T>(helper: () => Promise<T>) {
  // Returned to the caller so it can assert the helper actually ran. A helper
  // that silently failed would produce a clean stream and look like a pass.
  let helperResult: T | undefined;
  const probe = tool(
    async () => {
      helperResult = await helper();
      // Deliberately clean: the probe must only ever be able to reach the user
      // through the stream, never through the tool result and the message
      // history that follows it.
      return "searched";
    },
    {
      name: "probe_search",
      description: "Search documents",
      schema: z.object({ query: z.string() }),
    },
  );

  const agent = createAgent({
    model: new ScriptedModel([
      { content: "", tool_calls: [{ name: "probe_search", args: { query: "oil" }, id: "c1" }] },
      { content: ANSWER },
    ]),
    tools: [probe],
    systemPrompt: "Answer from the documents.",
  });

  const stream = await agent.stream(
    { messages: [new HumanMessage("What is the oil capacity?")] },
    { streamMode: ["updates", "messages"] },
  );

  let forwarded = "";
  let everything = "";
  for await (const chunk of stream) {
    const [mode, payload] = chunk as [string, unknown];
    if (mode !== "messages") continue;
    const [token, meta] = payload as [BaseMessage, Record<string, unknown> | undefined];
    if (token?.getType?.() !== "ai") continue;
    const text = visibleText(token.content);
    if (!text) continue;
    everything += text;
    if (isAnswerChunk(meta)) forwarded += text;
  }
  return { forwarded, everything, helperResult };
}

async function routeFilter() {
  banner("2. The route forwards only the answer node");

  const aux = new AuxModel();
  const { forwarded, everything } = await runWithHelperInTool(() =>
    aux.invoke([new HumanMessage("plan this")]),
  );

  // Non-vacuity, and the anchor for every check below: an untagged helper's
  // output really is on the stream, at the root graph's own depth. If this
  // failed, the rest would be testing a hazard that no longer exists.
  check(
    "an untagged helper's output does reach the stream",
    everything.includes(PROBE),
    "the leak is reachable",
  );
  check("but the route does not forward it", !forwarded.includes(PROBE), "filtered by node");
  // The other way this could pass without meaning anything: a filter that
  // rejects every chunk blocks the leak and the answer alike.
  check("while the answer still gets through", forwarded.includes(ANSWER), forwarded || "(nothing)");
}

async function nostreamTag() {
  banner("3. The nostream tag keeps a helper off the stream entirely");

  const aux = new AuxModel();
  const { forwarded, everything } = await runWithHelperInTool(() =>
    aux.invoke([new HumanMessage("plan this")], { tags: [NOSTREAM_TAG] }),
  );

  check(
    "a tagged helper produces no chunk at all",
    !everything.includes(PROBE),
    "suppressed before the route sees it",
  );
  check("and the answer is unaffected", forwarded.includes(ANSWER), forwarded || "(nothing)");
}

/* ------------------------------------------------------------------ *
 * 4. The production call path
 * ------------------------------------------------------------------ */

const Schema = z.object({ answer: z.string() });

/** Answers by tool call, which every provider here supports. */
class ToolCallingFake extends BaseChatModel {
  constructor(fields?: Record<string, unknown>) {
    super(fields ?? {});
  }
  _llmType() {
    return "tool-calling-fake";
  }
  bindTools() {
    return this;
  }
  async _generate(): Promise<ChatResult> {
    const message = new AIMessage({
      content: AUX_CONTENT,
      tool_calls: [{ name: "probe_plan", args: { answer: PROBE }, id: "t1" }],
    });
    return { generations: [{ text: "", message }], llmOutput: {} };
  }
}

/** The same model, refusing the native path the way ChatCohere does. */
class NoNativeStructuredFake extends ToolCallingFake {
  withStructuredOutput(): never {
    throw new Error("Input is not an AIMessageChunk");
  }
}

/**
 * The real thing: `structuredInvoke` called from inside a tool, which is exactly
 * where the query planner sits.
 *
 * Both of its routes are exercised. It tries `withStructuredOutput` first and
 * falls back to binding a single tool when a provider rejects that — Cohere does.
 * Tagging one route and not the other would leave the leak in place for whichever
 * providers took the untagged one, which is the kind of gap that only shows up on
 * the provider nobody tested.
 */
async function structuredHelperInTool() {
  banner("4. structuredInvoke inside a tool, as the query planner does it");

  for (const [label, Model] of [
    ["native structured output", ToolCallingFake],
    ["the tool-call fallback", NoNativeStructuredFake],
  ] as const) {
    const { forwarded, everything, helperResult } = await runWithHelperInTool(() =>
      structuredInvoke(new Model(), Schema, [new HumanMessage("plan")], { name: "probe_plan" }),
    );

    // Non-vacuity: a planner that never ran could not have leaked, so a clean
    // stream would prove nothing.
    check(`${label}: the planner ran`, helperResult?.answer === PROBE, JSON.stringify(helperResult));
    check(
      `${label}: nothing of it reaches the stream`,
      !everything.includes(PROBE),
      "no chunk emitted at all",
    );
    check(`${label}: the answer is unaffected`, forwarded.includes(ANSWER), forwarded || "(nothing)");
  }
}

async function main() {
  predicates();
  await routeFilter();
  await nostreamTag();
  await structuredHelperInTool();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
