import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { GUARDRAIL_CONFIG, SUBAGENT_CONFIG, type ThinkingMode } from "../config";
import { getModel } from "../models";
import { buildMiddleware } from "./middleware";
import { buildDelegationTool } from "./subagents";
import { buildTools, createEvidenceCollector, type EvidenceCollector } from "./tools";

/**
 * The orchestrator system prompt.
 *
 * This is the contract the whole system hangs on, so it is explicit about the
 * three things that most often go wrong in agentic RAG: skipping retrieval,
 * blending document facts with model priors, and answering confidently from
 * nothing.
 */
/**
 * The prompt is assembled from parts rather than written as one block, and the
 * workflow section is *chosen* rather than appended to.
 *
 * Appending was tried first and does not work. The direct workflow's third step
 * is "Search the documents first. Call `search_documents`…", so bolting a
 * delegation section onto the end produced a prompt that instructed the model
 * both to search immediately and not to search at all. It resolved the conflict
 * the way models generally do — by following the earlier, more concrete
 * instruction — and delegated research silently degraded into ordinary
 * research: a measured run made three direct searches and zero delegations,
 * answered correctly, and looked entirely healthy while the feature it was
 * exercising never engaged.
 *
 * Only one workflow is ever present. The grounding rules and style are shared,
 * because they are true regardless of who does the retrieving.
 */
const PROMPT_ROLE = `You are a rigorous research assistant that answers questions about the user's uploaded documents.`;

const DIRECT_WORKFLOW = `

## How you work

You have tools and a reasoning loop. Use them deliberately:

1. **Understand the question.** Resolve pronouns and references against the conversation. Decide what would actually constitute an answer.
2. **Plan when it is genuinely multi-step.** If the question needs several independent lookups, a comparison, or a sequence of dependent steps, call \`write_todos\` first and keep it updated as you go. Skip planning for simple single-lookup questions.
3. **Search the documents first.** Call \`search_documents\` for anything that could plausibly be in the user's files. Break compound questions into separate focused searches — one search per sub-question retrieves far better than one long query.
4. **Assess what came back.** If the passages don't answer the question, search again with different wording before giving up. Rare literal terms and section names work well as queries.
5. **Escalate to the web only when warranted** — the documents don't cover it, the question is about current events, or outside context is needed to interpret a document.
6. **Answer.**`;

const SHARED_RULES = `

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
 * The ordinary supervisor prompt: this agent does its own retrieval.
 *
 * Exported because the wiring tests build agents directly against it.
 */
export const SUPERVISOR_PROMPT = PROMPT_ROLE + DIRECT_WORKFLOW + SHARED_RULES;

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

/**
 * Assemble the supervisor's system prompt for a given configuration.
 *
 * Exported so the composition can be asserted directly. The failure this
 * guards against is a prompt that contradicts itself, which no type or
 * compile-time check can see and which does not raise an error at runtime
 * either — the agent simply follows whichever instruction it finds more
 * concrete and quietly stops using the feature that was configured.
 */
export function composeSystemPrompt({
  mode = "standard",
  deepAgents = false,
}: {
  mode?: ThinkingMode;
  deepAgents?: boolean;
}): string {
  // Exactly one workflow, and a depth addendum matched to it. Depth and
  // delegation are orthogonal settings but they are not independent *prompts*:
  // the deep instructions tell whoever is retrieving how hard to dig, and when
  // that is a researcher rather than the supervisor, the supervisor must be
  // told to spend the team harder instead of to search harder itself.
  const workflow = deepAgents ? DELEGATION_WORKFLOW : DIRECT_WORKFLOW;
  const depth =
    mode === "deep" ? (deepAgents ? DEEP_DELEGATION_PROMPT : DEEP_THINKING_PROMPT) : "";

  return PROMPT_ROLE + workflow + SHARED_RULES + depth;
}

export interface AgentRun {
  agent: ReturnType<typeof createAgent>;
  collector: EvidenceCollector;
}

/**
 * Extra contract for deep mode.
 *
 * Deep mode is not "the same run with a larger budget" — a bigger candidate
 * pool only helps if the agent actually interrogates it. This makes the extra
 * spend buy coverage and self-checking: decompose before searching, probe each
 * sub-question separately, seek disconfirming passages, and say where the
 * documents are silent rather than smoothing over the gap.
 */
const DEEP_THINKING_PROMPT = `

## Deep research mode

This question warrants thorough investigation. Work it properly:

1. **Decompose first.** Write a todo list breaking the question into every distinct sub-question, including the ones only implied by it. Keep it updated as you learn.
2. **Search each sub-question separately.** One search per sub-question retrieves far better than one broad search. Vary your wording between attempts — synonyms, the document's likely phrasing, rare literal terms, section names.
3. **Search again after reading.** What comes back usually reveals better vocabulary than the question used. Use it. Two or three refined passes are expected, not excessive.
4. **Look for what would contradict you.** Before concluding, search for exceptions, conditions, limits and superseding rules. A confident answer that missed a caveat is worse than a hedged one.
5. **Cross-check across sources.** Where several documents or sections bear on the same point, compare them and say so if they disagree.
6. **Be explicit about coverage.** State what the documents establish, what they only imply, and what they do not address at all. Never fill a gap with plausible-sounding general knowledge.

Structure the answer so the reasoning is inspectable: lead with the conclusion, then the evidence for each sub-question with its citations, then the limits and any conflicts you found.`;

/**
 * Contract for delegated research.
 *
 * Replaces the supervisor's own investigation rather than adding to it. The
 * instructions are explicit that a delegation is self-contained and that
 * several belong in one message, because both are things the model gets wrong
 * by default and both are expensive: a task description that says "the
 * threshold mentioned above" reaches a subagent that has never seen the
 * conversation and produces a confidently empty finding, and delegations issued
 * one per turn run strictly sequentially, which forfeits the entire latency
 * argument for delegating at all.
 */
const DELEGATION_WORKFLOW = `

## How you work

You are a research **supervisor**. You do not investigate questions yourself —
you have a team, and your job is to direct it.

Your \`delegate_research\` tool sends a list of sub-questions to specialist
researchers, each with its own clean context window. **They all run at the same
time**, so one call with five sub-questions costs about what one sub-question
costs. Each researcher does its own searching and returns a finding.

You have no search tools of your own. \`delegate_research\` is the only way to
reach a document, so if you need to know something, delegate it — including when
the gap is a single fact.

1. **Understand the question.** Resolve pronouns and references against the
   conversation. Decide what would actually constitute an answer.
2. **Decompose.** Call \`write_todos\` to split the question into distinct
   sub-questions, including the ones only implied by it. Even an apparently
   simple question usually has two or three.
3. **Fan out — this is your first real action, and it is ONE call.** Put every
   sub-question into a single \`delegate_research\` call, about
   ${SUBAGENT_CONFIG.TARGET_PARALLEL_WIDTH} at a time:

       delegate_research(tasks=[
         {researcher: "document-researcher", question: "<sub-question 1, in full>"},
         {researcher: "document-researcher", question: "<sub-question 2, in full>"},
         {researcher: "document-researcher", question: "<sub-question 3, in full>"},
       ])

   Those three researchers run **concurrently**, so all three findings arrive in
   roughly the time one takes. Calling \`delegate_research\` three separate times
   with one question each produces the same findings but takes three times as
   long, and it is the single biggest thing you control for how long the user
   waits. Batch everything that does not depend on an earlier finding — which,
   because you decomposed into independent sub-questions, is nearly all of it.

   **Be generous here — this call is your one chance at breadth.** Because the
   researchers run in parallel, a fourth sub-question costs almost nothing in
   time, while a sub-question you leave out is one nobody investigates: you get
   very few rounds, so anything omitted now is simply missing from the answer.
   Send one researcher per distinct thing the question asks about, plus any
   related point the answer will need. Delegating one sub-question when the
   question had three is the most common way this goes wrong.
4. **Write self-contained task descriptions.** The researcher sees only the text
   you write — not this conversation, not the user's question, not your other
   delegations. Never write "the threshold mentioned above" or "this document".
   State the sub-question in full and name every term, document and constraint
   it depends on.
5. **Pick the right specialist.** \`document-researcher\` for anything that could
   be in the user's files — this is the default and most delegations are this.
   \`web-researcher\` only when the documents cannot cover it. \`verifier\` for a
   claim you are about to assert but have not tested.
6. **Read the adversarial check.** Your first substantial batch of findings
   comes back with one attached automatically — a verifier has already hunted
   for exceptions, conditions, limits, superseding rules and disagreements
   between documents. You do not need to request it. Treat what it found as
   binding: if it narrows or qualifies a claim, the answer must say so. An
   answer that missed a caveat is worse than a hedged one. Delegate to
   \`verifier\` yourself only for a specific claim the automatic check did not
   cover.
7. **Then answer. One round is normally enough.** Your researchers have already
   searched, re-searched and been checked adversarially, so the usual shape of
   a turn is: decompose, one \`delegate_research\` call, answer. A second call
   is for a *specific, named* fact you still lack — put every such gap into
   that one call. Do not open a third. Chasing progressively smaller gaps is
   how a turn runs out of budget with nothing written, and a clearly-stated
   gap is a better answer than no answer at all.
8. **Synthesise.** Combine the findings into one answer. If a gap remains, say
   what is missing and answer everything else — never return empty-handed.

Citation markers inside a finding are already correct and shared across every
researcher. Reuse them exactly as returned. Never renumber them, and never cite
a number no finding gave you.

Structure the answer so the reasoning is inspectable: lead with the conclusion,
then the evidence per sub-question with its citations, then the limits, the
conflicts you found, and what the documents do not address.`;

/**
 * Deep mode's addendum when the work is delegated.
 *
 * The direct deep prompt instructs the agent to search each sub-question
 * itself, re-search on what it learns, and hunt for contradicting passages —
 * all of which are now a researcher's job, and repeating them here would
 * recreate the same contradiction that stopped delegation firing at all. What
 * survives is the part that is genuinely the supervisor's: wider coverage,
 * harder verification, and honesty about what the documents leave unsettled.
 */
const DEEP_DELEGATION_PROMPT = `

## Deep research mode

This question warrants thorough investigation, so spend the team harder:

- **Decompose further than feels necessary.** Split into every distinct
  sub-question, including implied ones, and delegate each. Breadth is the point.
- **Run a second round.** When the first findings return, they usually reveal
  sub-questions the original decomposition missed — the document's own
  vocabulary, a referenced policy, an exception worth chasing. Delegate those
  too rather than settling for the first pass.
- **Verify every load-bearing claim,** not only the ones you doubt. Delegate
  them to \`verifier\` in a batch.
- **Cross-check the findings against each other.** Where two researchers bear on
  the same point, compare them and say so explicitly if they disagree.
- **Be explicit about coverage.** State what the documents establish, what they
  only imply, and what they do not address at all. Never fill a gap with
  plausible-sounding general knowledge.`;

export interface AgentOptions {
  enableTodos?: boolean;
  /** Give the agent the web tools. Off restricts it to the uploaded documents. */
  webSearch?: boolean;
  /** Reasoning depth: retrieval breadth, iteration budget and the prompt contract. */
  mode?: ThinkingMode;
  /**
   * Delegate research to specialist subagents with isolated context windows.
   *
   * Independent of `mode`: depth controls how hard a single context works,
   * delegation controls how many contexts the work is spread across.
   */
  deepAgents?: boolean;
}

/**
 * Build a configured agent instance.
 *
 * A fresh evidence collector is created per run so citations reflect only the
 * current turn's retrievals, while conversational memory persists separately
 * through the checkpointer.
 */
export function buildAgent(options: AgentOptions = {}): AgentRun {
  const { webSearch = true, mode = "standard", deepAgents = false } = options;
  const collector = createEvidenceCollector();

  /**
   * One result cache for the whole run, supervisor and branches alike, so a
   * query one researcher has already run is not paid for again by another.
   * Built here because this is the only scope that spans both.
   */
  const resultCache = new Map<string, Promise<string>>();

  // Exactly one workflow, and a depth addendum matched to it. Depth and
  // delegation are orthogonal settings but they are not independent *prompts*:
  // the deep instructions tell whoever is retrieving how hard to dig, and when
  // that is a researcher rather than the supervisor, the supervisor must be
  // told to spend the team harder instead of to search harder itself.
  const systemPrompt = composeSystemPrompt({ mode, deepAgents });

  const agent = createAgent({
    model: getModel("pro"),
    tools: [
      ...buildTools(collector, {
        webSearch,
        mode,
        resultCache,
        label: "supervisor",
        delegating: deepAgents,
      }),
      // Added as an ordinary tool rather than through Deep Agents' subagent
      // middleware, because that middleware's `task` tool delegates one
      // sub-question per call and depends on the model batching several calls
      // into one message to fan out — which the providers here do not do. See
      // buildDelegationTool for the measurement.
      ...(deepAgents ? [buildDelegationTool(collector, { webSearch, resultCache })] : []),
    ],
    systemPrompt,
    middleware: buildMiddleware({ enableTodos: options.enableTodos, deepAgents }),
    checkpointer: getCheckpointer(),
    name: "agentic_rag_supervisor",
  });

  return { agent, collector };
}

export interface RunConfig {
  threadId: string;
  signal?: AbortSignal;
  mode?: ThinkingMode;
  /** Delegating runs take more graph steps: fan out, collect, verify, synthesise. */
  deepAgents?: boolean;
}

/** Runtime config passed to every agent invocation. */
export function runConfig({
  threadId,
  signal,
  mode = "standard",
  deepAgents = false,
}: RunConfig) {
  return {
    configurable: { thread_id: threadId },
    // Deep mode plans, probes each sub-question and re-searches; delegation adds
    // a fan-out and a verification round on top. Both need more graph steps than
    // a plain turn, and steps are not spend — the model-call ceilings, on the
    // supervisor and inside every subagent, are what bound cost. Raising this
    // only stops the step counter from ending a run the budget could still
    // afford.
    recursionLimit: deepAgents
      ? // A delegating turn is budgeted for SUBAGENT_CONFIG.TURN_MODEL_CALLS
        // model calls, and this graph spends roughly a dozen steps per call
        // before any tool runs — eighteen nodes, most of them middleware hooks.
        // At the deep multiplier the step counter would run out first and end a
        // run the budget could still afford, which is the exact failure the
        // limit was raised to stop in the first place.
        GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS * 4
      : mode === "deep"
        ? GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS * 2
        : GUARDRAIL_CONFIG.MAX_AGENT_ITERATIONS,
    signal,
  };
}

/** Subagent budgets, surfaced for /api/health and the dashboard. */
export const DELEGATION_BUDGET = SUBAGENT_CONFIG;
