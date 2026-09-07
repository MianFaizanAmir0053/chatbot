/**
 * End-to-end answer quality across every mode the product offers.
 *
 * The existing suite proves the machinery works — tools bind, branches run
 * concurrently, citations resolve. This asks the question a user would: given a
 * real question, is the answer good? It covers the three shapes that behave
 * differently, because each has broken independently at some point: a document
 * question, a question the documents cannot answer, and a comparison worth
 * delegating.
 *
 * Usage: npx tsx --env-file=.env scripts/e2e-quality.ts
 */

import { Document } from "@langchain/core/documents";
import { HumanMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { buildAgent, runConfig } from "../lib/agents/agent";
import { normaliseCitations } from "../lib/guardrails/output";
import { invalidateSparseIndex } from "../lib/retrieval/hybrid";
import { getVectorStore } from "../lib/vectorstore";
import type { EvidenceCollector } from "../lib/agents/tools";

/**
 * The corpus this run asserts against, seeded rather than assumed.
 *
 * Earlier versions of these checks read whatever happened to be in the vector
 * store, which made them a test of the environment as much as of the code: when
 * Docker stopped, Qdrant became unreachable, the store silently fell back to an
 * empty in-memory driver, and every document assertion failed for a reason that
 * had nothing to do with the agent. Seeding makes the run reproducible on a
 * cold machine and gives the assertions known right answers to check against.
 */
const FIXTURE = [
  {
    source: "expense-policy.txt",
    section: "Receipts",
    text:
      "Receipts under 10 GBP do not require itemisation. Any receipt of 10 GBP or more must " +
      "include an itemised breakdown showing each line item and its price.",
  },
  {
    source: "expense-policy.txt",
    section: "Approval limits",
    text:
      "A line manager may approve expense claims up to 500 GBP. Claims above 500 GBP require " +
      "director approval. Late submissions require director approval regardless of amount.",
  },
  {
    source: "travel-handbook.txt",
    section: "Meals",
    text:
      "Employees may expense meals up to 45 GBP per day when travelling. Alcohol is not " +
      "reimbursable under any circumstances.",
  },
  {
    source: "travel-handbook.txt",
    section: "Receipts",
    text:
      "All travel receipts must be itemised irrespective of value. This supersedes the general " +
      "10 GBP itemisation threshold for travel expenses only.",
  },
];

async function seedCorpus(): Promise<void> {
  const store = await getVectorStore();
  const docs = FIXTURE.map(
    (f, i) =>
      new Document({
        // Retrieval embeds a contextualised form and quotes `originalText`, so
        // both are set exactly as the ingest pipeline would leave them.
        pageContent: `${f.source} > ${f.section}\n\n${f.text}`,
        metadata: {
          source: f.source,
          section: f.section,
          chunkIndex: i,
          originalText: f.text,
        },
      }),
  );
  await store.addDocuments(docs);
  invalidateSparseIndex();
  console.log(`[fixture] seeded ${docs.length} chunks into the ${store.name} store\n`);
}

let failures = 0;
let skipped = 0;

function check(name: string, ok: boolean, detail = "") {
  console.log(`    ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}

function throttled(text: string): boolean {
  return /rate limit|429|quota|tokens per day|daily token/i.test(text);
}

interface Outcome {
  answer: string;
  tools: string[];
  ms: number;
  collector: EvidenceCollector;
}

async function ask(
  question: string,
  opts: { deepAgents?: boolean; webSearch?: boolean } = {},
): Promise<Outcome> {
  const { agent, collector } = buildAgent({
    enableTodos: true,
    webSearch: opts.webSearch ?? true,
    mode: "standard",
    deepAgents: opts.deepAgents ?? false,
  });

  const started = Date.now();
  const tools: string[] = [];
  let answer = "";

  const stream = await agent.stream(
    { messages: [new HumanMessage(question)] },
    {
      ...runConfig({ threadId: randomUUID(), deepAgents: opts.deepAgents }),
      streamMode: ["updates"],
    },
  );

  for await (const chunk of stream) {
    const [, payload] = chunk as [string, Record<string, { messages?: unknown[] }>];
    for (const upd of Object.values(payload ?? {})) {
      const last = upd?.messages?.at?.(-1) as
        | { tool_calls?: Array<{ name: string }>; getType?: () => string; content?: unknown }
        | undefined;
      for (const c of last?.tool_calls ?? []) tools.push(c.name);
      if (last?.getType?.() === "ai" && !last.tool_calls?.length) {
        const c = last.content;
        const t =
          typeof c === "string"
            ? c
            : (c as Array<{ type?: string; text?: string }> | undefined)
                ?.filter((b) => b?.type === "text")
                .map((b) => b.text)
                .join("") ?? "";
        if (t.trim()) answer = t;
      }
    }
  }

  return { answer: normaliseCitations(answer), tools, ms: Date.now() - started, collector };
}

/** Narration and permission-asking are answer defects, so every case checks them. */
function sharedAnswerRules(o: Outcome) {
  check("produced an answer", o.answer.trim().length > 40, `${o.answer.length} chars`);
  check(
    "does not narrate its process",
    !/\b(I'll (first|then|now)|Let me (search|check)|I will (search|check)|I'm going to (search|check))\b/i.test(
      o.answer,
    ),
    "no preamble",
  );
  check(
    "does not ask permission to use its own tools",
    !/would you like me to (search|perform|check|look)/i.test(o.answer),
    "no hand-back",
  );
}

async function documentQuestion() {
  console.log("\n[1] a question the documents answer  (standard mode)");
  const o = await ask("What is the receipt itemisation threshold?", { deepAgents: false });
  console.log(`    ${o.ms}ms · tools: ${[...new Set(o.tools)].join(", ") || "none"}`);

  sharedAnswerRules(o);
  const isThrottled = throttled(o.answer);
  if (isThrottled) {
    console.log("    SKIP  evidence checks — provider quota exhausted");
    skipped += 3;
  } else {
    check("searched the documents", o.tools.includes("search_documents"));
    check("gathered evidence", o.collector.documents.length > 0, `${o.collector.documents.length} passages`);
    const refs = [...o.answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
    check(
      "cited the documents, and every citation resolves",
      refs.length > 0 && refs.every((n) => n >= 1 && n <= o.collector.documents.length),
      refs.length === 0 ? "no citations" : `${refs.length} refs`,
    );
    // Accuracy, not just mechanics: the fixture says 10 GBP.
    check(
      "states the correct threshold",
      /\b10\b/.test(o.answer),
      /\b10\b/.test(o.answer) ? "found 10 GBP" : "threshold missing or wrong",
    );
  }
  console.log(`    > ${o.answer.replace(/\s+/g, " ").slice(0, 180)}`);
}

async function webQuestion() {
  console.log("\n[2] a question the documents cannot answer  (standard mode)");
  const o = await ask("What is the capital of Japan and roughly what is its population?", {
    deepAgents: false,
  });
  console.log(`    ${o.ms}ms · tools: ${[...new Set(o.tools)].join(", ") || "none"}`);

  sharedAnswerRules(o);
  if (throttled(o.answer)) {
    console.log("    SKIP  web checks — provider quota exhausted");
    skipped += 2;
  } else {
    check("went to the web itself", o.tools.includes("web_search"), o.tools.join(", "));
    check(
      "actually answered rather than deferring",
      /tokyo/i.test(o.answer),
      /tokyo/i.test(o.answer) ? "answer present" : "did not answer",
    );
  }
  console.log(`    > ${o.answer.replace(/\s+/g, " ").slice(0, 180)}`);
}

async function delegatedQuestion() {
  console.log("\n[3] a comparison worth delegating  (deep agents)");
  const o = await ask(
    "Compare what the documents say about receipt itemisation and about expense approval " +
      "limits, and note anything that conflicts.",
    { deepAgents: true, webSearch: false },
  );
  console.log(`    ${o.ms}ms · tools: ${[...new Set(o.tools)].join(", ") || "none"}`);

  sharedAnswerRules(o);
  check("delegated rather than searching itself", o.tools.includes("delegate_research"));

  if (throttled(o.answer)) {
    console.log("    SKIP  evidence checks — provider quota exhausted");
    skipped += 2;
  } else {
    check("branch evidence reached the shared store", o.collector.documents.length > 0, `${o.collector.documents.length} passages`);
    const refs = [...o.answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
    check(
      "the synthesised answer carries resolving citations",
      refs.length > 0 && refs.every((n) => n >= 1 && n <= o.collector.documents.length),
      refs.length === 0 ? "no citations in a delegated answer" : `${refs.length} refs`,
    );
    // The fixture plants a genuine conflict: the expense policy exempts
    // receipts under 10 GBP from itemisation, while the travel handbook
    // requires every travel receipt to be itemised and says so supersedes it.
    // Surfacing that is the entire argument for the verifier, so it is the
    // sharpest available test of whether delegated research is worth its cost.
    check(
      "surfaces the planted conflict between the two documents",
      /supersed|conflict|however|but .*travel|irrespective|overrid|except/i.test(o.answer),
      "contradiction detected",
    );
    check(
      "covers both sides of the comparison",
      /itemis/i.test(o.answer) && /(approval|500|director)/i.test(o.answer),
      "both sub-questions answered",
    );
  }
  console.log(`    > ${o.answer.replace(/\s+/g, " ").slice(0, 180)}`);
}

async function main() {
  await seedCorpus();
  await documentQuestion();
  await webQuestion();
  await delegatedQuestion();

  const note = skipped > 0 ? ` (${skipped} skipped — provider quota)` : "";
  console.log(`\n${failures === 0 ? "ALL QUALITY CHECKS PASSED" : `${failures} FAILED`}${note}`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
