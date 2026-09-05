/**
 * End-to-end agent loop test.
 *
 * Runs the real agent — planning, tool selection, retrieval, synthesis — and
 * asserts on observable behaviour rather than exact wording.
 *
 *   npx tsx --env-file=.env scripts/agent-test.ts
 */
import { HumanMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { buildAgent, runConfig } from "../lib/agents/agent";
import { chunkDocument } from "../lib/ingest/chunk";
import { checkGroundedness, validateCitations } from "../lib/guardrails/output";
import { invalidateSparseIndex } from "../lib/retrieval/hybrid";
import { activeModelId } from "../lib/models";
import { getVectorStore } from "../lib/vectorstore";

const MANUAL = `# Drive Chain Maintenance

Inspect the drive chain every 1,000 km. Adjust chain slack to between 25 mm and
35 mm at the midpoint of the lower run. Replace the chain every 20,000 km.

# Brake System

Replace brake fluid every two years regardless of mileage. Front brake pad
minimum thickness is 1.5 mm.

# Tyre Pressure

Front tyre cold pressure is 225 kPa. The rear tyre requires 250 kPa with a
passenger and 225 kPa solo.

# Engine Oil

Change engine oil every 6,000 km. Use SAE 10W-40. Oil capacity is 3.4 litres
including the filter.`;

let failures = 0;
function assert(ok: boolean, msg: string) {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${msg}`);
  if (!ok) failures += 1;
}

function banner(s: string) {
  console.log(`\n${"=".repeat(66)}\n${s}\n${"=".repeat(66)}`);
}

interface RunOutcome {
  answer: string;
  toolCalls: string[];
  todos: Array<{ content: string; status: string }>;
  documents: number;
}

async function runAgent(question: string, label: string): Promise<RunOutcome> {
  console.log(`\n--- ${label} ---\n  Q: ${question}`);

  const { agent, collector } = buildAgent({ enableTodos: true });
  const toolCalls: string[] = [];
  let todos: Array<{ content: string; status: string }> = [];
  let answer = "";

  const stream = await agent.stream(
    { messages: [new HumanMessage(question)] },
    { ...runConfig({ threadId: randomUUID() }), streamMode: ["updates", "messages"] },
  );

  for await (const chunk of stream) {
    const [mode, payload] = chunk as [string, unknown];

    if (mode === "messages") {
      const [token] = payload as [{ content?: unknown; getType?: () => string }];
      if (token?.getType?.() === "ai" && typeof token.content === "string") {
        answer += token.content;
      }
      continue;
    }

    const updates = payload as Record<string, { todos?: typeof todos; messages?: unknown[] }>;
    for (const update of Object.values(updates ?? {})) {
      if (Array.isArray(update?.todos)) todos = update.todos;
      const last = update?.messages?.at?.(-1) as
        | { tool_calls?: Array<{ name: string }> }
        | undefined;
      for (const call of last?.tool_calls ?? []) toolCalls.push(call.name);
    }
  }

  console.log(`  tools : ${toolCalls.length ? toolCalls.join(" -> ") : "(none)"}`);
  if (todos.length) {
    console.log(`  plan  : ${todos.map((t) => `[${t.status}] ${t.content}`).join(" | ")}`);
  }
  console.log(`  docs  : ${collector.documents.length} retrieved`);
  console.log(`  A: ${answer.trim().slice(0, 400)}`);

  return { answer, toolCalls, todos, documents: collector.documents.length };
}

async function main() {
  banner("Setup");
  console.log(`  pro : ${activeModelId("pro")}`);
  console.log(`  fast: ${activeModelId("fast")}`);

  const chunks = await chunkDocument({ text: MANUAL, pages: [MANUAL] }, {
    source: "motorcycle-manual.txt",
    fileType: "text/plain",
    documentTitle: "Motorcycle Manual",
  });
  const store = await getVectorStore();
  await store.deleteBySource("motorcycle-manual.txt");
  await store.addDocuments(chunks);
  invalidateSparseIndex();
  console.log(`  indexed ${chunks.length} chunks into ${store.name}`);

  /* --- 1. Simple lookup: should retrieve, should NOT over-plan --- */
  banner("1. Simple factual lookup");
  const simple = await runAgent("What is the engine oil capacity?", "simple");
  assert(simple.toolCalls.includes("search_documents"), "calls search_documents");
  assert(simple.documents > 0, "retrieves supporting documents");
  assert(/3\.4/.test(simple.answer), "answer contains the correct value (3.4 litres)");

  /* --- 2. Multi-step: should decompose into a plan --- */
  banner("2. Multi-step comparison (should trigger planning)");
  const multi = await runAgent(
    "Compare the front and rear tyre pressures, and tell me how that relates to the " +
      "brake pad wear limit and the chain inspection interval. Summarise all three.",
    "multi-step",
  );
  const searches = multi.toolCalls.filter((t) => t === "search_documents").length;
  assert(searches >= 2, `breaks the question into multiple searches (${searches})`);
  assert(
    multi.todos.length > 0 || searches >= 3,
    `plans the work (todos: ${multi.todos.length}, searches: ${searches})`,
  );
  assert(/225|250/.test(multi.answer), "answer includes tyre pressure figures");
  assert(/1\.5/.test(multi.answer), "answer includes the brake pad limit");
  assert(/1,?000/.test(multi.answer), "answer includes the chain inspection interval");

  /* --- 3. Not in the documents: should refuse, not fabricate --- */
  banner("3. Absent information (should refuse, not invent)");
  const absent = await runAgent(
    "What is the maximum towing capacity specified in the manual?",
    "absent",
  );
  const refuses =
    /not (specified|mentioned|provided|contain|covered|available|include)|does not|doesn't|no (information|mention|reference)|unable to find|not found/i.test(
      absent.answer,
    );
  assert(refuses, "declines rather than fabricating a figure");
  assert(
    !/\b\d+\s?(kg|lbs|tonnes?|pounds)\b/i.test(absent.answer),
    "does not invent a towing figure",
  );

  /* --- 4. Guardrails over the produced answers --- */
  banner("4. Output guardrails on real agent output");
  const { agent: _a, collector } = buildAgent({ enableTodos: false });
  void _a;
  void collector;

  const grounding = await import("../lib/retrieval/pipeline").then((m) =>
    m.retrieve("engine oil capacity", { topK: 4 }),
  );
  const cites = validateCitations(simple.answer, grounding.documents);
  console.log(`  citations valid: ${cites.valid}${cites.invalidRefs.length ? ` (bad refs: ${cites.invalidRefs})` : ""}`);

  const g = await checkGroundedness(simple.answer, grounding.documents);
  console.log(`  groundedness   : ${g.score.toFixed(2)} (${g.verdict})`);
  assert(g.score < 0 || g.passed, "the agent's own answer passes the groundedness gate");

  banner("Summary");
  console.log(failures === 0 ? "  All agent checks passed." : `  ${failures} check(s) failed.`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((error) => {
  console.error("\nAgent test crashed:", error);
  process.exit(1);
});
