/**
 * Agent wiring test — no model quota required.
 *
 * Drives the real agent graph, middleware stack and tools with a scripted fake
 * model, so it verifies *our* wiring rather than a provider's behaviour:
 * middleware composes, tools execute and collect evidence, the todo list lands
 * in state, and streaming emits the events the SSE route depends on.
 *
 *   npx tsx --env-file=.env scripts/agent-wiring-test.ts
 */
import { createAgent, FakeToolCallingModel } from "langchain";
import { HumanMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { buildMiddleware } from "../lib/agents/middleware";
import { buildTools, createEvidenceCollector } from "../lib/agents/tools";
import { SUPERVISOR_PROMPT, getCheckpointer } from "../lib/agents/agent";
import { chunkDocument } from "../lib/ingest/chunk";
import { invalidateSparseIndex } from "../lib/retrieval/hybrid";
import { getVectorStore } from "../lib/vectorstore";
import { GUARDRAIL_CONFIG } from "../lib/config";

const MANUAL = `# Engine Oil

Change engine oil every 6,000 km. Use SAE 10W-40 grade oil. Engine oil capacity
is 3.4 litres including the filter.

# Tyre Pressure

Front tyre cold pressure is 225 kPa. The rear tyre requires 250 kPa with a
passenger.`;

let failures = 0;
function assert(ok: boolean, msg: string) {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${msg}`);
  if (!ok) failures += 1;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(66)}\n${s}\n${"=".repeat(66)}`);
}

async function main() {
  banner("Setup");
  const chunks = await chunkDocument({ text: MANUAL, pages: [MANUAL] }, {
    source: "manual.txt",
    fileType: "text/plain",
    documentTitle: "Manual",
  });
  const store = await getVectorStore();
  await store.deleteBySource("manual.txt");
  await store.addDocuments(chunks);
  invalidateSparseIndex();
  console.log(`  indexed ${chunks.length} chunks into ${store.name}`);

  /* --- 1. The middleware stack composes --- */
  banner("1. Middleware harness");
  const middleware = buildMiddleware({ enableTodos: true });
  console.log(`  ${middleware.length} middleware active`);
  const names = middleware.map((m) => (m as { name?: string }).name ?? "(unnamed)");
  console.log(`  ${names.join(", ")}`);
  assert(middleware.length >= 8, "full middleware stack assembles");
  assert(names.some((n) => /todo/i.test(n)), "todo/planning middleware present");
  assert(names.some((n) => /pii/i.test(n)), "PII middleware present");
  assert(names.some((n) => /limit/i.test(n)), "budget-limit middleware present");

  /* --- 2. Tools are registered and executable --- */
  banner("2. Tools");
  const collector = createEvidenceCollector();
  const tools = buildTools(collector);
  const toolNames: string[] = tools.map((t) => t.name);
  console.log(`  ${toolNames.join(", ")}`);
  assert(toolNames.includes("search_documents"), "search_documents registered");
  assert(toolNames.includes("web_search"), "web_search registered");
  assert(toolNames.includes("write_todos") === false, "write_todos comes from middleware, not tools");

  const calc = tools.find((t) => t.name === "calculator")!;
  const calcOut = await calc.invoke({ expression: "(1250 * 0.08) + 340" });
  assert(String(calcOut).includes("440"), `calculator computes correctly (${calcOut})`);

  const badCalc = await calc.invoke({ expression: "process.exit(1)" });
  assert(
    String(badCalc).includes("Only arithmetic"),
    "calculator rejects non-arithmetic input",
  );

  const fetchTool = tools.find((t) => t.name === "fetch_url")!;
  const ssrf = await fetchTool.invoke({ url: "http://127.0.0.1:8080/admin" });
  assert(String(ssrf).includes("Refusing"), "fetch_url blocks loopback (SSRF guard)");
  const scheme = await fetchTool.invoke({ url: "file:///etc/passwd" });
  assert(String(scheme).includes("Only http"), "fetch_url blocks non-HTTP schemes");

  /* --- 3. Real retrieval through the tool interface --- */
  banner("3. search_documents through the tool interface");
  const search = tools.find((t) => t.name === "search_documents")!;
  const searchOut = String(await search.invoke({ query: "engine oil capacity" }));
  assert(searchOut.includes("3.4"), "retrieves the correct passage");
  assert(/\[1\]/.test(searchOut), "returns numbered, citable excerpts");
  assert(collector.documents.length > 0, "evidence collector captured provenance");

  /* --- 4. The agent graph runs, calls tools, and streams --- */
  banner("4. Agent graph with a scripted model");
  // Scripted with retrieval only. `write_todos` returns a Command that rewrites
  // the message list, which trips FakeToolCallingModel's "reset to index 0 when
  // the history looks fresh" heuristic and makes it loop on planning forever.
  // Planning is therefore exercised in its own run below.
  const fake = new FakeToolCallingModel({
    toolCalls: [
      [{ name: "search_documents", args: { query: "engine oil capacity" }, id: "call_search_1" }],
      [],
    ],
  });

  const agentCollector = createEvidenceCollector();
  const agent = createAgent({
    model: fake,
    tools: buildTools(agentCollector),
    systemPrompt: SUPERVISOR_PROMPT,
    middleware: buildMiddleware({ enableTodos: true }),
    checkpointer: getCheckpointer(),
    name: "wiring_test_agent",
  });

  const observed = {
    toolCalls: [] as string[],
    toolResults: [] as string[],
    todos: [] as unknown[],
    sawMessages: false,
  };

  const stream = await agent.stream(
    { messages: [new HumanMessage("What is the engine oil capacity?")] },
    {
      configurable: { thread_id: randomUUID() },
      recursionLimit: GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS,
      streamMode: ["updates", "messages"],
    },
  );

  // FakeToolCallingModel cycles its script with `index % length`, so it never
  // stops calling tools on its own. Stop once the expected sequence has been
  // observed; hitting the recursion limit here would be a harness artefact,
  // not a defect in the agent.
  const seenAll = () =>
    // Wait for the tool to have actually run, not merely been requested —
    // provenance only exists once search_documents executes.
    observed.toolResults.includes("search_documents") &&
    agentCollector.documents.length > 0 &&
    observed.sawMessages;

  try {
    for await (const chunk of stream) {
      const [mode, payload] = chunk as [string, unknown];
      if (mode === "messages") {
        observed.sawMessages = true;
        if (seenAll()) break;
        continue;
      }
      const updates = payload as Record<string, { todos?: unknown[]; messages?: unknown[] }>;
      for (const update of Object.values(updates ?? {})) {
        if (Array.isArray(update?.todos)) observed.todos = update.todos;
        const last = update?.messages?.at?.(-1) as
          | { tool_calls?: Array<{ name: string }>; name?: string; content?: unknown; getType?: () => string }
          | undefined;
        for (const c of last?.tool_calls ?? []) observed.toolCalls.push(c.name);
        if (last?.getType?.() === "tool" && last.name) observed.toolResults.push(last.name);
      }
      if (seenAll()) break;
    }
  } catch (error) {
    // The scripted model never stops calling tools, so exhausting the recursion
    // limit is expected here. Any other error is a real failure.
    if (!(error instanceof Error) || !/Recursion limit/i.test(error.message)) throw error;
    console.log("  (scripted model cycled to the recursion limit, as expected)");
  }

  console.log(`  tool calls  : ${observed.toolCalls.join(" -> ") || "(none)"}`);
  console.log(`  tool results: ${observed.toolResults.join(" -> ") || "(none)"}`);
  assert(observed.toolCalls.includes("search_documents"), "retrieval tool call reaches the stream");
  assert(observed.toolResults.includes("search_documents"), "the tool actually executes in the graph");
  assert(observed.sawMessages, "token-level message stream is emitted");
  assert(
    agentCollector.documents.length > 0,
    "agent run collected document provenance for citations",
  );

  /* --- 5. Planning state reaches the UI --- */
  banner("5. Planning (todo list in agent state)");
  const planned = await runPlanningAgent();
  console.log(`  todos in state: ${planned.length}`);
  assert(planned.length === 2, "todo list lands in agent state for the UI");

  banner("Summary");
  console.log(failures === 0 ? "  All wiring checks passed." : `  ${failures} check(s) failed.`);
  process.exit(failures === 0 ? 0 : 1);
}

/**
 * Exercise the planning middleware on its own.
 *
 * Uses invoke() rather than stream(): the scripted model loops on write_todos,
 * so we read the resulting state directly instead of waiting for termination.
 */
async function runPlanningAgent(): Promise<unknown[]> {
  const todos = [
    { content: "Find the engine oil capacity", status: "in_progress" as const },
    { content: "Report the figure with a citation", status: "pending" as const },
  ];
  const fake = new FakeToolCallingModel({
    toolCalls: [[{ name: "write_todos", args: { todos }, id: "call_todo_1" }], []],
  });

  const agent = createAgent({
    model: fake,
    tools: buildTools(createEvidenceCollector()),
    systemPrompt: SUPERVISOR_PROMPT,
    middleware: buildMiddleware({ enableTodos: true }),
    checkpointer: getCheckpointer(),
    name: "planning_test_agent",
  });

  const config = {
    configurable: { thread_id: randomUUID() },
    recursionLimit: GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS,
  };

  try {
    await agent.invoke(
      { messages: [new HumanMessage("Compare three maintenance intervals.")] },
      config,
    );
  } catch (error) {
    // Expected: the scripted model loops on write_todos and never terminates.
    if (!(error instanceof Error) || !/Recursion limit/i.test(error.message)) throw error;
  }

  // Read the persisted state rather than racing the stream for it.
  const snapshot = (await agent.getState(config)) as unknown as
    | { values?: Record<string, unknown> }
    | undefined;
  const values = snapshot?.values ?? {};
  console.log(`  state keys: ${Object.keys(values).join(", ") || "(none)"}`);
  const todosInState = values.todos;
  return Array.isArray(todosInState) ? todosInState : [];
}

main().catch((error) => {
  console.error("\nWiring test crashed:", error);
  process.exit(1);
});
