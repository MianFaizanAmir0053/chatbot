import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { GUARDRAIL_CONFIG } from "../config";
import { getModel } from "../models";
import { buildMiddleware } from "./middleware";
import { buildTools, createEvidenceCollector, type EvidenceCollector } from "./tools";

/**
 * The orchestrator system prompt.
 *
 * This is the contract the whole system hangs on, so it is explicit about the
 * three things that most often go wrong in agentic RAG: skipping retrieval,
 * blending document facts with model priors, and answering confidently from
 * nothing.
 */
export const SUPERVISOR_PROMPT = `You are a rigorous research assistant that answers questions about the user's uploaded documents.

## How you work

You have tools and a reasoning loop. Use them deliberately:

1. **Understand the question.** Resolve pronouns and references against the conversation. Decide what would actually constitute an answer.
2. **Plan when it is genuinely multi-step.** If the question needs several independent lookups, a comparison, or a sequence of dependent steps, call \`write_todos\` first and keep it updated as you go. Skip planning for simple single-lookup questions.
3. **Search the documents first.** Call \`search_documents\` for anything that could plausibly be in the user's files. Break compound questions into separate focused searches — one search per sub-question retrieves far better than one long query.
4. **Assess what came back.** If the passages don't answer the question, search again with different wording before giving up. Rare literal terms and section names work well as queries.
5. **Escalate to the web only when warranted** — the documents don't cover it, the question is about current events, or outside context is needed to interpret a document.
6. **Answer.**

## Grounding rules

- Every factual claim from a document must carry a citation marker: [1], [2], matching the numbered excerpts returned by \`search_documents\`.
- Never present a web result as if it came from the user's documents. Label external facts explicitly, e.g. "According to <source> on the web, ...".
- If the documents do not contain the answer and the web doesn't either, say so plainly and state what you did look for. A clear "the documents don't cover this" is a correct answer, not a failure.
- Do not blend your own background knowledge into document answers. If you add useful general context, mark it as such.
- Never invent a citation number. Only cite excerpts you actually received.

## Style

- Lead with the answer, then support it.
- Be concise. Use short paragraphs, and bullets only when genuinely listing.
- Quote the document directly when the exact wording matters (specifications, procedures, limits, warnings).
- If the answer depends on a condition or assumption in the source, say so.`;

/**
 * Checkpointer for conversation memory.
 *
 * MemorySaver is per-process. It is the correct default for a single instance
 * and the seam to replace with a Postgres/Redis saver for multi-instance
 * deployments — nothing else in the code needs to change.
 */
let checkpointerSingleton: BaseCheckpointSaver | null = null;

export function getCheckpointer(): BaseCheckpointSaver {
  if (!checkpointerSingleton) checkpointerSingleton = new MemorySaver();
  return checkpointerSingleton;
}

export interface AgentRun {
  agent: ReturnType<typeof createAgent>;
  collector: EvidenceCollector;
}

/**
 * Build a configured agent instance.
 *
 * A fresh evidence collector is created per run so citations reflect only the
 * current turn's retrievals, while conversational memory persists separately
 * through the checkpointer.
 */
export function buildAgent(options: { enableTodos?: boolean } = {}): AgentRun {
  const collector = createEvidenceCollector();

  const agent = createAgent({
    model: getModel("pro"),
    tools: buildTools(collector),
    systemPrompt: SUPERVISOR_PROMPT,
    middleware: buildMiddleware({ enableTodos: options.enableTodos }),
    checkpointer: getCheckpointer(),
    name: "agentic_rag_supervisor",
  });

  return { agent, collector };
}

export interface RunConfig {
  threadId: string;
  signal?: AbortSignal;
}

/** Runtime config passed to every agent invocation. */
export function runConfig({ threadId, signal }: RunConfig) {
  return {
    configurable: { thread_id: threadId },
    recursionLimit: GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS,
    signal,
  };
}
