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
import { normaliseCitations } from "../lib/guardrails/output";
import { formatContext } from "../lib/retrieval/pipeline";
import type { RankedDocument } from "../lib/retrieval/rerank";

let failures = 0;
let inconclusive = 0;

function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}

/**
 * A check that cannot mean anything when the provider quota is spent.
 *
 * Every assertion about what an agent *found* depends on retrieval having
 * worked. When Cohere or the model provider is rate limited, researchers return
 * "I encountered rate limit errors" as their finding and the run then shows no
 * passages, no searches and no citations — reporting three failures that say
 * nothing about the code. A whole validation run failed exactly this way after
 * testing exhausted a daily token allowance.
 *
 * Reporting those as failures is worse than not running them: a suite that
 * cries wolf for environmental reasons is a suite people learn to ignore, and
 * the next real regression goes unread. SKIP is counted separately so a run
 * with skips is never mistaken for a clean one.
 */
function checkUnlessThrottled(name: string, ok: boolean, throttled: boolean, detail = "") {
  if (!ok && throttled) {
    console.log(`  SKIP  ${name}  — provider quota exhausted, cannot judge`);
    inconclusive++;
    return;
  }
  check(name, ok, detail);
}

/** Did this text come back because a provider refused rather than because the documents are silent? */
function looksThrottled(text: string): boolean {
  return /rate limit|429|quota|tokens per day|insufficient balance|exhausted/i.test(text);
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
 * Malformed citation markers must be repaired, not ignored.
 *
 * The validator recognises `[n]` and nothing else, so a marker in a different
 * shape is not caught as invalid — it is not seen. A measured delegating turn
 * answered with `[1†L1-L3]`, a citation style from another vendor's tooling,
 * and the pipeline reported "no citations in the answer" while showing that
 * text to the user. A fabricated number gets stripped; a malformed one used to
 * sail through every check.
 */
function citationRepair() {
  console.log("\n[citations] malformed markers are repaired");

  const cases: Array<[string, string]> = [
    ["Receipts under GBP 10 [1†L1-L3] need no itemisation.", "[1]"],
    ["Per the handbook 【2†source】 travel is capped.", "[2]"],
    ["Both documents agree [1, 3].", "[1][3]"],
    ["Stated plainly [ 4 ].", "[4]"],
    ["A limit applies [2:page 7].", "[2]"],
  ];

  for (const [input, expected] of cases) {
    const out = normaliseCitations(input);
    check(
      `repairs ${input.match(/[[【][^\]】]*[\]】]/)?.[0] ?? input}`,
      out.includes(expected),
      out.replace(/\s+/g, " "),
    );
  }

  // A well-formed marker must survive untouched.
  const clean = "Receipts under GBP 10 [1] need no itemisation [2].";
  check("leaves well-formed markers alone", normaliseCitations(clean) === clean, clean);

  // Mixed formats in one string, as a real finding produces.
  const mixed = "Evidence: 【2†L1-L2】 and [1†L3] and [3].";
  check(
    "repairs several formats in one finding",
    normaliseCitations(mixed) === "Evidence: [2] and [1] and [3].",
    normaliseCitations(mixed),
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

  // Process narration reaches the user as part of the answer. Asked "what is
  // Islam", the agent replied "To provide a comprehensive answer I'll first
  // check if there are any relevant documents... Since your documents don't
  // appear to contain information about Islam, I'll now search the web" — all
  // of it describing searches the interface was already displaying.
  check(
    "both prompts forbid narrating the process",
    direct.includes("Never narrate what you are about to do") &&
      delegating.includes("Never narrate what you are about to do"),
  );
  check(
    "but still require saying where facts came from",
    direct.includes("Saying where a fact came from is not narration"),
    "attribution is not narration",
  );

  // Asked "what is Islam" with the web tool available, the agent searched the
  // documents, found nothing, and replied "Would you like me to perform a web
  // search?" — handing the work back to the user, who asked precisely so they
  // would not have to do it.
  check(
    "the prompt forbids asking permission to use its own tools",
    direct.includes("Never ask permission to use a tool you already have") &&
      direct.includes("Never end by offering to do something you could have just done"),
  );

  // Breadth and few rounds have to be asked for together. Capping the rounds
  // stopped a supervisor that delegated four times and answered nothing, but on
  // its own it pushed the opposite way: a later turn delegated a single
  // sub-question, gathered three passages and answered in 477 characters. The
  // first round is the only one wide enough to matter, so the prompt has to
  // push for breadth there and restraint afterwards.
  check(
    "the delegating prompt asks for breadth in the first round",
    delegating.includes("one chance at breadth"),
    "wide first round, few rounds after",
  );
  check(
    "and still discourages extra rounds",
    delegating.includes("One round is normally enough"),
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

  // Concurrent branches must hold different credentials. Sharing one compiled
  // agent per specialist put every branch of a fan-out on a single key, where
  // they queued behind its per-minute limit: one branch took 12.4s while three
  // concurrent branches took 128.1s — worse than running them sequentially.
  // The failure is invisible in the output, which is why it needs a check.
  const keyOf = (agent: unknown) =>
    (agent as { options?: { model?: { lc_kwargs?: { apiKey?: string } } } })?.options?.model
      ?.lc_kwargs?.apiKey ?? "";

  const { buildDelegationTool: build } = await import("../lib/agents/subagents");
  build(createEvidenceCollector(), { webSearch: false });

  // Reach the pool through a fresh compile: two slots of the same specialist
  // must not resolve to the same credential.
  const { getSubagentModel } = await import("../lib/models");
  const keys = [keyOf({ options: { model: getSubagentModel("pro") } }), keyOf({ options: { model: getSubagentModel("pro") } })];
  check(
    "consecutive subagent models draw different credentials",
    keys[0] !== "" && keys[0] !== keys[1],
    keys[0] === keys[1] ? "rotation returned the same key twice" : "rotation advances per build",
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
/**
 * The web researcher must actually work, not merely be configured.
 *
 * Every other live check runs with `webSearch: false`, so this specialist was
 * wired, bound and never once executed — exactly the state the verifier was in
 * when it turned out to be dead. "Configured" has already proven not to mean
 * "runs" in this system, so it gets its own exercise.
 *
 * Deliberately asks something the uploaded documents cannot answer, so a
 * document search cannot accidentally satisfy it and hide a broken web path.
 */
async function webResearcher() {
  console.log("\n[web] the web researcher runs end to end");

  const { buildDelegationTool } = await import("../lib/agents/subagents");
  const collector = createEvidenceCollector();
  const delegate = buildDelegationTool(collector, { webSearch: true });

  const out = await (delegate as unknown as { invoke: (a: unknown) => Promise<string> }).invoke({
    tasks: [
      {
        researcher: "web-researcher",
        question:
          "What is LangChain's Deep Agents library, and what does it provide? Answer from " +
          "public web sources.",
      },
    ],
  });

  check("the web researcher returned a finding", out.length > 200, `${out.length} chars`);
  check(
    "it did not fail outright",
    !/FAILED —/.test(out),
    /FAILED —/.test(out) ? out.match(/FAILED — [^\n]*/)?.[0] ?? "failed" : "no failure",
  );
  check(
    "it reached the web rather than the documents",
    collector.webResults.length > 0,
    `${collector.webResults.length} web result(s), ${collector.documents.length} passage(s)`,
  );
  check(
    "web findings are attributed as external",
    /web|http|according to|source/i.test(out),
    "attribution present",
  );

  console.log(`    ${out.replace(/\s+/g, " ").slice(0, 200)}`);
}

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

  const invoke = (tasks: Array<{ researcher: string; question: string }>) =>
    (delegate as unknown as { invoke: (a: unknown) => Promise<string> }).invoke({ tasks });

  // A single branch first, as the unit to compare against. One task is below
  // AUTO_VERIFY_MIN_FINDINGS, so this times a researcher alone with no
  // verification attached.
  const soloStarted = Date.now();
  await invoke([{ researcher: "document-researcher", question: questions[0] }]);
  const solo = Date.now() - soloStarted;

  const started = Date.now();
  const out = await invoke(
    questions.map((q) => ({ researcher: "document-researcher", question: q })),
  );
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
    !/^\s*\[\d+\]\s+(document|web)-researcher/m.test(out),
    "labels are headed, not bracketed",
  );

  /**
   * Concurrency as a ratio against one branch, not as a wall-clock threshold.
   *
   * An absolute limit measures the provider, not the code. The same code path
   * timed 19.3s, 70.8s and 139.2s on three consecutive runs, because Cohere's
   * per-minute quota was progressively more depleted by the testing itself —
   * so a fixed threshold turns into a coin flip that fails for reasons the
   * change under test has nothing to do with.
   *
   * A ratio moves with those conditions. Sequential execution would cost about
   * four branch-times here: three researchers one after another, then the
   * serial verification pass. Concurrent execution costs about two — the
   * slowest researcher, then verification. Three sits between them with room
   * on both sides.
   */
  const ratio = solo > 0 ? elapsed / solo : Infinity;
  check(
    "three branches plus verification cost far less than four sequential",
    ratio < 3,
    `${(elapsed / 1000).toFixed(1)}s vs ${(solo / 1000).toFixed(1)}s for one branch ` +
      `(${ratio.toFixed(1)}x; sequential would be ~4x)`,
  );

  check(
    "branch evidence landed in the shared collector",
    collector.documents.length > 0,
    `${collector.documents.length} passages, ${collector.searches.length} searches`,
  );

  // The verifier must run without being asked. As a prompt step it was skipped
  // every time: a measured turn started thirteen researchers, all of them
  // document-researchers, with the verifier configured, bound and never used.
  check(
    "an adversarial check ran without being requested",
    /adversarial check/i.test(out),
    /adversarial check/i.test(out) ? "verifier fired automatically" : "verifier did not run",
  );

  // Findings must reach the supervisor with markers it can reuse.
  //
  // This is where citations were actually being lost. Researchers cite in an
  // annotated format of their own — measured output used `【2†L1-L2】` — and the
  // supervisor is told to reuse a finding's markers verbatim, so it copied a
  // broken one or cited nothing. The answer then reached validation with no
  // recognisable citations and was pronounced valid, because an unknown marker
  // shape is not judged invalid, it is not seen. Two consecutive delegating
  // turns answered with no citations before this was found.
  const findingMarkers = [...out.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const throttled = looksThrottled(out);
  checkUnlessThrottled(
    "findings carry valid citation markers",
    findingMarkers.length > 0,
    throttled,
    findingMarkers.length > 0
      ? `${findingMarkers.length} marker(s): ${[...new Set(findingMarkers)].join(", ")}`
      : "none — the supervisor has nothing to cite",
  );
  check(
    "and no annotated marker survives into a finding",
    !/【\d+†|\[\d+†/.test(out),
    /【\d+†|\[\d+†/.test(out) ? "annotated markers still present" : "markers are plain [n]",
  );
  checkUnlessThrottled(
    "every finding marker resolves to gathered evidence",
    findingMarkers.length > 0 &&
      findingMarkers.every((n) => n >= 1 && n <= collector.documents.length),
    throttled,
    `${collector.documents.length} passages gathered`,
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
  checkUnlessThrottled(
    "delegated retrievals reached the shared evidence store",
    collector.documents.length > 0,
    looksThrottled(answer),
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
  citationRepair();
  promptComposition();
  await delegationWiring();
  await zodBoundary();
  if (process.argv.includes("--live")) {
    await concurrency();
    await webResearcher();
  }

  if (process.argv.includes("--live")) {
    await liveTurn();
  } else {
    console.log("\n[live] skipped — pass --live to issue a real delegating turn");
  }

  const skipped = inconclusive > 0 ? ` (${inconclusive} skipped)` : "";
  console.log(
    `\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}${skipped}`,
  );

  // A run with skips must never read as a clean run. The skipped checks are the
  // ones that matter most — whether research actually found anything — and they
  // were skipped precisely because the answer to that was unobtainable.
  if (inconclusive > 0) {
    console.log(
      "  Skipped checks could not be judged: retrieval was rate limited, so this run says\n" +
        "  nothing about them either way. Re-run once the provider quota resets.",
    );
  }
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
