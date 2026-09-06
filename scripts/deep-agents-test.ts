/**
 * Does delegated research actually work, end to end?
 *
 * Structured as two tiers because they fail for different reasons and cost
 * very differently. The structural tier needs no provider and no vector store,
 * so it runs in under a second and catches the wiring mistakes that would
 * otherwise surface as a confusing model failure ten minutes into a live run.
 * The live tier issues a real delegating turn.
 *
 * Each check exists because the corresponding failure is silent. A citation
 * that resolves to the wrong passage still renders; a subagent whose tools were
 * built against a different evidence store still answers, with sources the
 * client cannot show; a fan-out that runs sequentially still returns the right
 * answer, just slowly enough that nobody would use the feature. None of these
 * announce themselves.
 *
 * Usage:
 *   npx tsx --env-file=.env scripts/deep-agents-test.ts             # structural only
 *   npx tsx --env-file=.env scripts/deep-agents-test.ts --live      # + a real turn
 */

import { Document } from "@langchain/core/documents";
import { HumanMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { SUBAGENT_CONFIG } from "../lib/config";
import { buildAgent, composeSystemPrompt, runConfig } from "../lib/agents/agent";
import {
  createEvidenceCollector,
  registerEvidence,
  type EvidenceCollector,
} from "../lib/agents/tools";
import { subagentProviderNames } from "../lib/models";
import { formatContext } from "../lib/retrieval/pipeline";
import type { RankedDocument } from "../lib/retrieval/rerank";

let failures = 0;

function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}

function ranked(source: string, chunkIndex: number, text: string): RankedDocument {
  return {
    doc: new Document({
      pageContent: text,
      metadata: { source, chunkIndex, originalText: text },
    }),
    score: 0.9,
    retrievers: ["dense"],
    reranked: true,
  };
}

/* ------------------------------------------------------------------ *
 * Structural
 * ------------------------------------------------------------------ */

function citationOrdinals() {
  console.log("\n[citations] run-global ordinals");

  const collector: EvidenceCollector = createEvidenceCollector();

  // Two searches, as a supervisor and a subagent would each make.
  const first = [ranked("a.pdf", 0, "alpha"), ranked("a.pdf", 1, "bravo")];
  const second = [ranked("a.pdf", 1, "bravo"), ranked("b.pdf", 0, "charlie")];

  const o1 = registerEvidence(collector, first);
  const o2 = registerEvidence(collector, second);

  check("first search numbers from 1", JSON.stringify(o1) === "[1,2]", JSON.stringify(o1));
  check(
    "second search continues rather than restarting",
    JSON.stringify(o2) === "[2,3]",
    `${JSON.stringify(o2)} (a repeat keeps its number, a new passage gets the next)`,
  );
  check(
    "evidence stored once per passage",
    collector.documents.length === 3,
    `${collector.documents.length} documents`,
  );

  // The property everything else depends on: marker n resolves to documents[n-1].
  let resolves = true;
  for (const [key, ordinal] of collector.ordinals) {
    const doc = collector.documents[ordinal - 1];
    const actual = `${doc?.doc.metadata?.source}#${doc?.doc.metadata?.chunkIndex}`;
    if (actual !== key) resolves = false;
  }
  check("every marker resolves to the passage it was assigned", resolves);

  const rendered = formatContext(second, o2);
  check(
    "rendered excerpts carry the global numbers",
    rendered.includes("[2]") && rendered.includes("[3]") && !rendered.includes("[1]"),
    rendered.split("\n")[0],
  );
}

/**
 * The prompt must not tell the supervisor to do the opposite of what it is
 * configured to do.
 *
 * This check exists because the first working build failed exactly here and
 * failed invisibly. Delegation was wired correctly — the tool was bound, the
 * researchers compiled, the ceilings were installed — but the delegation
 * instructions were appended to a workflow whose third step was "Search the
 * documents first", so the model searched directly and never delegated. The
 * run answered the question correctly in 35 seconds and reported nothing
 * unusual. Only counting delegations revealed the feature had not engaged.
 */
function promptComposition() {
  console.log("\n[prompt] one workflow, never two");

  const direct = composeSystemPrompt({ deepAgents: false });
  const delegating = composeSystemPrompt({ deepAgents: true });
  const deepDelegating = composeSystemPrompt({ deepAgents: true, mode: "deep" });

  check(
    "the direct prompt tells the agent to search",
    direct.includes("Search the documents first"),
  );
  check(
    "the delegating prompt does NOT tell it to search first",
    !delegating.includes("Search the documents first") && delegating.includes("no search tools of your own"),
    delegating.includes("Search the documents first")
      ? "contradiction present — delegation will not fire"
      : "no contradiction",
  );
  check("the delegating prompt introduces the team", delegating.includes("delegate_research"));
  check(
    "the direct prompt has no delegation instructions",
    !direct.includes("`delegate_research` tool"),
  );
  check(
    "deep + delegating asks for a wider fan-out, not deeper solo search",
    deepDelegating.includes("Decompose further") &&
      !deepDelegating.includes("Search each sub-question separately"),
  );
  check(
    "grounding rules survive in both",
    direct.includes("Never invent a citation number") &&
      delegating.includes("Never invent a citation number"),
  );
}

async function delegationWiring() {
  console.log("\n[wiring] the task tool and its researchers");

  const withAgents = buildAgent({ deepAgents: true, webSearch: true, enableTodos: true });
  const without = buildAgent({ deepAgents: false, webSearch: true, enableTodos: true });

  // The compiled graph keeps its node map on the builder, and node names carry
  // the middleware that contributed them — which makes the graph itself the
  // evidence that a middleware is installed, rather than the config that was
  // meant to install it.
  const nodes = (agent: ReturnType<typeof buildAgent>) =>
    Object.keys(
      (agent.agent as unknown as { builder?: { nodes?: Record<string, unknown> } }).builder
        ?.nodes ?? {},
    );

  const delegating = nodes(withAgents);
  const plain = nodes(without);

  check("delegating agent compiles", delegating.length > 0, `${delegating.length} graph nodes`);
  check("non-delegating agent still compiles", plain.length > 0, `${plain.length} graph nodes`);

  const hasDelegationCeiling = (list: string[]) => list.some((n) => n.includes("[delegate_research]"));
  check(
    "the per-tool delegation ceiling is installed",
    hasDelegationCeiling(delegating),
    delegating.filter((n) => n.includes("[delegate_research]")).join(", ") || "absent",
  );
  check(
    "and is absent when delegation is off",
    !hasDelegationCeiling(plain),
    hasDelegationCeiling(plain) ? "leaked into the non-delegating agent" : "correctly absent",
  );

  // What the model can actually call, read off the compiled tool node rather
  // than off the config that was meant to produce it.
  const boundTools = (agent: ReturnType<typeof buildAgent>) =>
    (
      (
        (agent.agent as unknown as {
          builder?: { nodes?: { tools?: { runnable?: { tools?: Array<{ name?: string }> } } } };
        }).builder?.nodes?.tools?.runnable?.tools ?? []
      ).map((t) => t?.name ?? "?")
    ).sort();

  const delegatingTools = boundTools(withAgents);
  const plainTools = boundTools(without);

  check(
    "the delegation tool is bound to the supervisor",
    delegatingTools.includes("delegate_research"),
    delegatingTools.join(", "),
  );
  check(
    "and is absent without delegation",
    !plainTools.includes("delegate_research"),
    plainTools.join(", "),
  );

  // The decisive one. A supervisor holding both `task` and `search_documents`
  // takes the direct route and never delegates, however the prompt is worded —
  // measured, twice, before the tools were separated.
  check(
    "the delegating supervisor has no retrieval tools of its own",
    !delegatingTools.includes("search_documents") &&
      !delegatingTools.includes("web_search") &&
      !delegatingTools.includes("fetch_url"),
    delegatingTools.includes("search_documents")
      ? "search_documents still bound — the model will bypass delegation"
      : "retrieval is delegation-only",
  );
  check(
    "but keeps the planning tools it needs",
    delegatingTools.includes("list_documents") && delegatingTools.includes("write_todos"),
    delegatingTools.join(", "),
  );
  check(
    "the non-delegating agent keeps its retrieval",
    plainTools.includes("search_documents"),
    plainTools.join(", "),
  );

  check(
    "subagents are pinned to reliable providers",
    subagentProviderNames().length > 0,
    subagentProviderNames().join(", "),
  );

  check(
    "the turn has a bounded model-call budget",
    SUBAGENT_CONFIG.TURN_MODEL_CALLS > 0 && SUBAGENT_CONFIG.TURN_TOOL_CALLS > 0,
    `${SUBAGENT_CONFIG.TURN_MODEL_CALLS} model calls, ` +
      `${SUBAGENT_CONFIG.TURN_TOOL_CALLS} tool calls, ` +
      `max ${SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND} branches x ` +
      `${SUBAGENT_CONFIG.MAX_DELEGATION_ROUNDS} rounds`,
  );

  // Branch-level ceilings must stay absent. They read the supervisor's tally
  // through shared state and refuse the delegation before the researcher runs.
  const subMiddleware = (
    withAgents.agent as unknown as {
      builder?: { nodes?: Record<string, unknown> };
    }
  ).builder?.nodes ?? {};
  check(
    "no duplicate model-call limiter on the supervisor",
    Object.keys(subMiddleware).filter((n) => n.startsWith("ModelCallLimitMiddleware.before_model"))
      .length <= 1,
    "one limiter owns the turn budget",
  );
}

/**
 * The zod 3/zod 4 boundary, exercised rather than assumed.
 *
 * `deepagents` depends on zod 4 while this project is on zod 3, so npm nests a
 * second copy and the two are different objects at runtime. Every tool here is
 * built with the project's zod 3 and handed to a library that builds its own
 * `task` schema with zod 4; if the interop were broken, schema conversion would
 * throw or silently produce an empty parameter set — a tool the model can see
 * but never call correctly. Compiling the middleware and converting its schema
 * is what actually proves this, and a typecheck does not.
 */
async function zodBoundary() {
  console.log("\n[interop] zod 3 tools inside a zod 4 library");

  const { buildDelegationTool } = await import("../lib/agents/subagents");
  const collector = createEvidenceCollector();
  const delegate = buildDelegationTool(collector, { webSearch: false }) as unknown as {
    name?: string;
    schema?: unknown;
  };

  check("delegation tool builds", delegate?.name === "delegate_research", delegate?.name ?? "none");

  const { toJsonSchema } = await import("@langchain/core/utils/json_schema");
  let converted: Record<string, unknown> | null = null;
  try {
    converted = toJsonSchema(delegate.schema as never) as Record<string, unknown>;
  } catch (error) {
    check("schema converts to JSON schema", false, (error as Error).message.slice(0, 90));
  }

  if (converted) {
    const props = (converted.properties ?? {}) as Record<string, { items?: unknown }>;
    check(
      "the batch schema survives conversion with its array intact",
      "tasks" in props && Boolean(props.tasks?.items),
      Object.keys(props).join(", ") || "no properties",
    );
  }
}

/**
 * The branches must actually run at the same time.
 *
 * Measured by invoking the tool directly with three sub-questions and comparing
 * elapsed time against the slowest branch. Delegation that runs sequentially
 * produces exactly the same answer as delegation that runs concurrently, so
 * nothing about the output reveals the difference — the earlier `task`-based
 * build fanned out to five researchers strictly one after another and took 220
 * seconds, and every correctness check passed.
 */
async function concurrency() {
  console.log("\n[concurrency] branches run at the same time");

  const { buildDelegationTool } = await import("../lib/agents/subagents");
  const collector = createEvidenceCollector();
  const delegate = buildDelegationTool(collector, { webSearch: false });

  const questions = [
    "What do the documents say about receipt itemisation?",
    "What expense approval limits do the documents specify?",
    "What do the documents say about travel booking?",
  ];

  const started = Date.now();
  const out = await (delegate as unknown as { invoke: (a: unknown) => Promise<string> }).invoke({
    tasks: questions.map((q) => ({ researcher: "document-researcher", question: q })),
  });
  const elapsed = Date.now() - started;

  check(
    "all branches reported",
    questions.every((_, i) => out.includes(`Finding ${i + 1} of ${questions.length}`)),
    `${out.length} chars returned in ${(elapsed / 1000).toFixed(1)}s`,
  );

  // Branch labels must not look like citation markers, or the supervisor has
  // two numbering schemes in one payload and can cite the wrong thing.
  check(
    "branch labels do not collide with citation markers",
    !/^s*[d+]s+(document|web)-researcher/m.test(out),
    "labels are headed, not bracketed",
  );

  // Three sequential branches against this stack take well over a minute; three
  // concurrent ones cost about one branch. The threshold sits between those,
  // generously, so the check fails on a regression to sequential rather than on
  // an unlucky slow provider.
  check(
    "three branches cost roughly one, not three",
    elapsed < 90_000,
    `${(elapsed / 1000).toFixed(1)}s for ${questions.length} researchers`,
  );

  check(
    "branch evidence landed in the shared collector",
    collector.documents.length > 0,
    `${collector.documents.length} passages, ${collector.searches.length} searches`,
  );
}

/* ------------------------------------------------------------------ *
 * Live
 * ------------------------------------------------------------------ */

async function liveTurn() {
  console.log("\n[live] one delegating turn");

  const question =
    process.env.TEST_QUESTION ??
    "Compare what the documents say about receipt itemisation and about expense approval " +
      "limits, and note anything that conflicts between them.";

  const { agent, collector } = buildAgent({
    deepAgents: true,
    webSearch: false,
    enableTodos: true,
    mode: "standard",
  });

  const started = Date.now();
  const delegations: Array<{ id: string; agent: string; at: number }> = [];
  const results: Array<{ id: string; at: number }> = [];
  let answer = "";

  const stream = await agent.stream(
    { messages: [new HumanMessage(question)] },
    {
      ...runConfig({ threadId: randomUUID(), deepAgents: true }),
      streamMode: ["updates"],
    },
  );

  for await (const chunk of stream) {
    const [, payload] = chunk as [string, Record<string, { messages?: unknown[] }>];
    for (const update of Object.values(payload ?? {})) {
      const last = update?.messages?.at?.(-1) as
        | {
            tool_calls?: Array<{ id?: string; name: string; args?: Record<string, unknown> }>;
            name?: string;
            tool_call_id?: string;
            content?: unknown;
            getType?: () => string;
          }
        | undefined;

      for (const call of last?.tool_calls ?? []) {
        if (call.name !== "delegate_research") continue;
        const id = call.id ?? randomUUID();
        if (delegations.some((d) => d.id === id)) continue;
        delegations.push({
          id,
          agent: String(call.args?.subagent_type ?? "?"),
          at: Date.now() - started,
        });
      }

      if (last?.getType?.() === "tool" && last.name === "delegate_research") {
        const id = last.tool_call_id ?? randomUUID();
        if (!results.some((r) => r.id === id)) results.push({ id, at: Date.now() - started });
      }
      if (last?.getType?.() === "ai") {
        const text =
          typeof last.content === "string"
            ? last.content
            : (last.content as Array<{ type?: string; text?: string }> | undefined)
                ?.filter((b) => b?.type === "text")
                .map((b) => b.text)
                .join("") ?? "";
        if (text.trim()) answer = text;
      }
    }
  }

  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  console.log(`  turn completed in ${seconds}s`);

  check("the supervisor delegated", delegations.length > 0, `${delegations.length} task calls`);
  check("delegations reported back", results.length > 0, `${results.length} findings`);
  check("an answer was produced", answer.trim().length > 0, `${answer.length} chars`);

  // Parallelism: delegations issued in one message start together. Comparing
  // start times is the only way to tell a genuine fan-out from a sequence, and
  // the difference is the whole latency argument for delegating.
  console.log(
    `    dispatch: ${delegations.map((d) => `${d.agent}@${d.at}ms`).join(", ")}`,
  );
  check(
    "the fan-out was batched into few calls",
    delegations.length <= 2,
    `${delegations.length} delegate_research call(s) — each one is a whole parallel batch`,
  );

  // Delegated evidence must land in the shared collector, or the client renders
  // an answer whose citations point at nothing.
  check(
    "delegated retrievals reached the shared evidence store",
    collector.documents.length > 0,
    `${collector.documents.length} passages, ${collector.searches.length} searches`,
  );

  const refs = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const dangling = refs.filter((n) => n < 1 || n > collector.documents.length);
  check(
    "every citation in the answer resolves",
    dangling.length === 0,
    refs.length === 0
      ? "no citations in the answer"
      : `${refs.length} refs, ${dangling.length} dangling`,
  );

  console.log(`\n  answer: ${answer.replace(/\s+/g, " ").slice(0, 300)}`);
}

async function main() {
  citationOrdinals();
  promptComposition();
  await delegationWiring();
  await zodBoundary();
  if (process.argv.includes("--live")) await concurrency();

  if (process.argv.includes("--live")) {
    await liveTurn();
  } else {
    console.log("\n[live] skipped — pass --live to issue a real delegating turn");
  }

  console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
