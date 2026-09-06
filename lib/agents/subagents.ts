import {
  modelFallbackMiddleware,
  modelRetryMiddleware,
  toolRetryMiddleware,
  type AnyAgentMiddleware,
} from "langchain";
import { createSubAgent, type SubAgent } from "deepagents";
import { tool } from "@langchain/core/tools";
import { HumanMessage } from "@langchain/core/messages";
import { setMaxListeners } from "events";
import { z } from "zod";
import { SUBAGENT_CONFIG } from "../config";
import { getSubagentFallbackModels, getSubagentModel } from "../models";
import { buildTools, type EvidenceCollector } from "./tools";

/**
 * Delegated research: the supervisor splits a question and hands each part to a
 * subagent with its own context window.
 *
 * The problem this solves is specific. Deep mode asks one agent to decompose a
 * question, probe every sub-question separately, re-search on what it learns,
 * hunt for contradicting passages and cross-check sources — all in a single
 * transcript. At the deep retrieval profile that transcript accumulates up to
 * fourteen passages per search across as many as nine searches, and long before
 * the end `contextEditingMiddleware` begins clearing older tool results to stay
 * inside the window. So the instruction to cross-check evidence and the
 * mechanism keeping the run affordable are working against each other: the
 * passages needed for the cross-check are exactly the ones being deleted, and
 * the deeper the question the more certainly they are gone.
 *
 * Delegation removes the conflict rather than trading one side off against the
 * other. Each subagent reads its own passages in its own window and returns a
 * finding — a few sentences and the citation numbers behind them — so the
 * supervisor accumulates conclusions instead of raw retrieval. Evidence for the
 * whole question stays intact because no single context ever had to hold it
 * all, and the clearing threshold is never approached.
 *
 * Two design choices are worth stating because the obvious alternatives are
 * wrong here:
 *
 * Subagents retrieve at the `focused` profile, not `deep`. Deep runs five query
 * variants because a lone agent has no other way to widen coverage; once the
 * question is already split, paying for five variants per branch re-broadens
 * work that decomposition has broadened and multiplies embedding and rerank
 * spend by the fan-out width.
 *
 * Findings are shaped by the prompt rather than by a `responseFormat` schema.
 * Structured output would be tidier, but it costs an extra model call per
 * delegation and leans on provider behaviour that this deployment's gateways
 * support unevenly — and a branch that fails to satisfy a schema returns
 * nothing at all, which is a far worse failure than slightly untidy prose.
 */

/**
 * Every concurrent branch attaches its own abort listener to the run's signal,
 * so a fan-out legitimately exceeds Node's default ceiling of ten and the
 * runtime warns about a possible leak on every delegating turn. The listeners
 * are real and intended, and they are released when the branches settle — but
 * left alone the warning trains the reader to ignore exactly the message that
 * would report an actual leak. Raised to cover the widest fan-out with room for
 * the retry middleware's own listeners.
 */
setMaxListeners(SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND * 4);

/** Shared contract: what a research branch must return, and how it must cite. */
const FINDING_CONTRACT = `
## What you must return

Return a finding, not a transcript. The supervisor cannot see your searches —
only the text you return — so it has to stand alone.

Structure it exactly like this:

**Finding.** Two to five sentences answering the task you were given. Lead with
the answer.
**Evidence.** The specific passages supporting it, each with its citation
marker.
**Gaps.** What the sources do not settle. Write "None" if they settle it fully.

## Citation rules

- Excerpt numbers from \`search_documents\` are stable for the entire
  conversation turn and shared with every other researcher working on this
  question. Reuse them exactly as given. Never renumber, never invent one.
- Never state as fact anything you did not retrieve. If the passages do not
  support a claim, leave it out or put it under Gaps.
- Report an absence plainly. "The documents do not address X" is a correct and
  useful finding, and far more valuable to the supervisor than a plausible
  guess it cannot distinguish from evidence.`;

const DOCUMENT_RESEARCHER_PROMPT = `You are a document researcher. You investigate one narrow sub-question against the user's uploaded documents and report what you find.

Work the sub-question properly, but only the sub-question you were given:

1. Search with \`search_documents\` using the wording most likely to appear in the document, not the wording of the task.
2. Read what comes back. It usually reveals better vocabulary than your first query — rare literal terms, section names, the document's own phrasing. Search again with it.
3. Two or three refined passes are expected. Stop as soon as the sub-question is answered; do not keep searching for its own sake.
4. If nothing relevant returns after genuinely different phrasings, stop and report the absence.
${FINDING_CONTRACT}`;

const WEB_RESEARCHER_PROMPT = `You are a web researcher. You investigate one narrow sub-question against public sources and report what you find.

1. Use \`web_search\` for the sub-question you were given.
2. Use \`fetch_url\` when a result looks right but its snippet is too short to settle the point.
3. Prefer primary and authoritative sources. Say which source a claim came from.

Everything you return is external context, never a document fact. Attribute it explicitly — "According to <source> on the web, ..." — so the supervisor cannot mistake it for something from the user's files.
${FINDING_CONTRACT}`;

const VERIFIER_PROMPT = `You are an adversarial verifier. You are given a claim and you look for what would undermine it.

You are not here to confirm anything. Your value is entirely in what you find that contradicts, narrows or supersedes the claim:

1. Search for exceptions, conditions, thresholds, effective dates, and superseding rules.
2. Search for the claim's own terms *negated* — where a document states when something does not apply, or applies differently.
3. Check whether different documents or sections disagree. If they do, say which says what.
4. Search wording that would appear in a caveat, not wording that would appear in a confirmation.

Report honestly. If you looked and found nothing that undermines the claim, say so plainly — a clean verification is a real result. Do not manufacture doubt.
${FINDING_CONTRACT}`;

/**
 * Middleware a subagent runs with — resilience only, deliberately no ceilings.
 *
 * A subagent is separately compiled, so the parent's retries and fallbacks stop
 * at the delegation boundary: a branch that meets a rate limit with no retry of
 * its own is simply lost, and the supervisor waits for it and then reports a
 * gap that was an infrastructure failure rather than an absence in the
 * documents. Those have to be installed here.
 *
 * Budget ceilings must *not* be. `modelCallLimitMiddleware` and
 * `toolCallLimitMiddleware` keep their tallies in agent state, and a subagent
 * inherits the parent's state, so a branch starts counting from what the
 * supervisor has already spent rather than from zero. Installing them here
 * measured the supervisor against the branch's allowance and refused every
 * delegation before a researcher ran at all — every `task` call came back
 * "run level call limit reached with 6 model calls" and the supervisor reported
 * that it could not find the documents. A branch is bounded instead by the
 * recursion limit passed at invocation, which is per-call and cannot be
 * inherited.
 *
 * Jitter on the retry matters more here than anywhere else: a fan-out that
 * meets a rate limit fails in lockstep, every branch retrying at the same
 * instant against the same limiter.
 */
function subagentMiddleware(): AnyAgentMiddleware[] {
  // Annotated rather than inferred: each middleware factory returns its own
  // state-shape generic, so an inferred array literal narrows to the first
  // entry's type and rejects every later push.
  const middleware: AnyAgentMiddleware[] = [
    modelRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2,
      // Same reasoning as the supervisor's retry: free tiers meter per minute,
      // so a sub-second backoff retries inside a window that is still closed.
      initialDelayMs: 2_000,
      maxDelayMs: 20_000,
      jitter: true,
      onFailure: "continue",
    }),
    toolRetryMiddleware({ maxRetries: 2, backoffFactor: 2, onFailure: "continue" }),
  ];

  const fallbacks = getSubagentFallbackModels("pro");
  if (fallbacks.length > 0) middleware.push(modelFallbackMiddleware(...fallbacks));

  return middleware;
}

export interface SubagentOptions {
  /** Give the researchers web access. Off keeps every branch on the documents. */
  webSearch?: boolean;
  /** Shared result cache, so branches do not re-run each other's searches. */
  resultCache?: Map<string, Promise<string>>;
}

/** The researcher a delegation is addressed to. */
const RESEARCHERS = ["document-researcher", "web-researcher", "verifier"] as const;
type ResearcherName = (typeof RESEARCHERS)[number];

/**
 * Compile the specialist researchers for this run.
 *
 * Every subagent's tools are built against the *same* evidence collector as the
 * supervisor's. That is what makes delegated citations work: a passage found
 * inside a branch is registered in the run's shared evidence, gets a run-global
 * ordinal, and resolves correctly when the finished answer is validated and
 * when the client renders its sources. The branch's context is isolated; its
 * evidence is not, and must not be.
 *
 * Each researcher type is constructed with its own call to `getSubagentModel`,
 * which advances the credential rotation, and this runs per request — so the
 * specialists hold different keys from each other and from the supervisor, and
 * a later request draws different keys again. The rotation is per researcher
 * *type* rather than per delegation: a subagent is compiled once, so two
 * concurrent `document-researcher` branches share a credential. Spreading those
 * too would need a chat model that rotates internally per invocation, which is
 * a disproportionate amount of delicate wrapping for a burst of three or four
 * calls; `modelFallbackMiddleware` covers the case where a key does give out.
 */
function compileResearchers(collector: EvidenceCollector, options: SubagentOptions) {
  const { webSearch = true, resultCache } = options;

  const researchTools = (label: string, allowWeb: boolean) =>
    buildTools(collector, { webSearch: allowWeb, mode: "focused", resultCache, label });

  const specs: Record<ResearcherName, SubAgent> = {
    "document-researcher": {
      name: "document-researcher",
      description: "Investigates one sub-question against the uploaded documents.",
      systemPrompt: DOCUMENT_RESEARCHER_PROMPT,
      mode: "isolated",
      tools: researchTools("document-researcher", false),
      model: getSubagentModel("pro"),
      middleware: subagentMiddleware(),
    },
    verifier: {
      name: "verifier",
      description: "Adversarially checks one claim for exceptions and contradictions.",
      systemPrompt: VERIFIER_PROMPT,
      mode: "isolated",
      tools: researchTools("verifier", false),
      model: getSubagentModel("pro"),
      middleware: subagentMiddleware(),
    },
    "web-researcher": {
      name: "web-researcher",
      description: "Investigates one sub-question against public web sources.",
      systemPrompt: WEB_RESEARCHER_PROMPT,
      mode: "isolated",
      // Falls back to the document tools when the web is disabled, so a
      // misaddressed delegation still does useful work rather than failing.
      tools: researchTools("web-researcher", webSearch),
      model: getSubagentModel("pro"),
      middleware: subagentMiddleware(),
    },
  };

  // Compiled lazily and reused: compiling is not free, and a fan-out commonly
  // addresses the same specialist several times.
  const compiled = new Map<ResearcherName, ReturnType<typeof createSubAgent>>();
  return (name: ResearcherName) => {
    let agent = compiled.get(name);
    if (!agent) {
      agent = createSubAgent(specs[name]);
      compiled.set(name, agent);
    }
    return agent;
  };
}

/**
 * The delegation tool: one call, many researchers, genuinely concurrent.
 *
 * Shaped as a batch rather than one-task-per-call because the alternative does
 * not work on the models this system runs on. Deep Agents' own `task` tool takes
 * a single delegation and depends on the model emitting several tool calls in
 * one message to produce a fan-out. Measured against the primary provider, that
 * never happens: asked explicitly to call a tool once for each of three topics
 * in a single message, with `parallel_tool_calls: true` set, groq's
 * `openai/gpt-oss-120b` returned exactly one tool call — both with the flag and
 * without it. A supervisor built on `task` therefore delegates strictly
 * sequentially; a measured fan-out of three took 28 seconds just to dispatch,
 * each branch waiting for the one before, which forfeits the entire latency
 * argument for delegating.
 *
 * Taking an array moves concurrency out of the model's tool-calling behaviour
 * and into code, where it is guaranteed rather than hoped for. One tool call
 * starts every researcher at once.
 *
 * Failures are contained per branch. `Promise.all` over raw invocations would
 * lose a whole fan-out to one researcher's rate limit, so each branch resolves
 * to either a finding or a short note about what went wrong and the supervisor
 * synthesises from what returned — degrading coverage on one sub-question
 * rather than failing the turn.
 */
export function buildDelegationTool(
  collector: EvidenceCollector,
  options: SubagentOptions = {},
) {
  const researcherFor = compileResearchers(collector, options);

  return tool(
    async ({ tasks }: { tasks: Array<{ researcher: ResearcherName; question: string }> }) => {
      const batch = tasks.slice(0, SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND);
      if (batch.length === 0) return "No tasks supplied. Provide at least one sub-question.";

      console.log(
        `[delegate] ${batch.length} researcher(s) in parallel: ` +
          batch.map((t) => t.researcher).join(", "),
      );

      const findings = await Promise.all(
        batch.map(async (t, i) => {
          // Never bracketed. The findings themselves are full of `[1]`, `[2]`
          // citation markers, and labelling the branches the same way would put
          // two different numbering schemes in one payload for the supervisor
          // to confuse — with the failure landing on citations, which are the
          // one thing here that has to be exactly right.
          const label = `### Finding ${i + 1} of ${batch.length} · ${t.researcher}\nSub-question: ${t.question}`;
          try {
            const result = await researcherFor(t.researcher).invoke(
              { messages: [new HumanMessage(t.question)] },
              // Bounded here rather than by middleware inside the branch: a
              // recursion limit passed at invocation is per-call by
              // construction and cannot be inherited from the supervisor's
              // state the way the limit middlewares' counters are.
              { recursionLimit: SUBAGENT_CONFIG.BRANCH_RECURSION_LIMIT },
            );

            const last = result.messages.at(-1);
            const content = last?.content;
            const text =
              typeof content === "string"
                ? content
                : (content as Array<{ type?: string; text?: string }> | undefined)
                    ?.filter((b) => b?.type === "text")
                    .map((b) => b.text)
                    .join("") ?? "";

            return `${label}\n${text.trim() || "(the researcher returned nothing)"}`;
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            console.warn(`[delegate] ${t.researcher} failed: ${message.slice(0, 120)}`);
            return `${label}\nFAILED — ${message.slice(0, 200)}\nTreat this sub-question as unresearched.`;
          }
        }),
      );

      return `${findings.length} finding(s):\n\n${findings.join("\n\n---\n\n")}`;
    },
    {
      name: "delegate_research",
      description:
        "Delegate several sub-questions at once to specialist researchers, each with its own " +
        "clean context. They run CONCURRENTLY, so put every independent sub-question into a " +
        "single call rather than calling this repeatedly — that is what makes it fast. " +
        "Each researcher sees ONLY the question you write for it: not this conversation, not " +
        "the user's question, not the other tasks. Write each question in full, naming every " +
        "term, document and constraint it depends on. You get back findings whose citation " +
        "markers are already correct and can be reused verbatim.",
      schema: z.object({
        tasks: z
          .array(
            z.object({
              researcher: z
                .enum(RESEARCHERS)
                .describe(
                  "document-researcher for the user's files (the default); " +
                    "web-researcher for public sources; " +
                    "verifier to attack a claim you have not yet tested",
                ),
              question: z
                .string()
                .describe("The complete, self-contained sub-question for this researcher"),
            }),
          )
          .min(1)
          .max(SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND)
          .describe("Every sub-question to research at the same time"),
      }),
    },
  );
}
