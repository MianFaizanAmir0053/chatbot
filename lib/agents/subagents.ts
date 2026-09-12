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
import { isPermanentRefusal } from "../credential-health";
import { normaliseCitations } from "../guardrails/output";
import { getSubagentFallbackModels, getSubagentModel } from "../models";
import type { ThreadScope } from "../vectorstore";
import { toolCallRepairMiddleware } from "./middleware";
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

- Write a citation as **\`[7]\`** — a number in square brackets, nothing else
  inside them. Not \`【7†L1-L2】\`, not \`[7†source]\`, not \`[7, lines 1-2]\`.
  Do not add line ranges, file names or any other annotation inside the
  brackets; put that in your prose if it matters.
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
3. **Make at most four searches, and stop as soon as the sub-question is answered.** Two or three is usual.
4. **If two searches with genuinely different wording both return nothing relevant, stop and report the absence.** Do not keep rephrasing. The documents either cover this or they do not, and a prompt "the documents do not address this" is worth far more than a slow one — you are one of several researchers working in parallel, and the supervisor waits for the slowest.
${FINDING_CONTRACT}`;

const WEB_RESEARCHER_PROMPT = `You are a web researcher. You investigate one narrow sub-question against public sources and report what you find.

1. Use \`web_search\` for the sub-question you were given.
2. Use \`fetch_url\` when a result looks right but its snippet is too short to settle the point.
3. **If the task asks what changed, what a page used to say, or to compare versions, call \`page_history\` and read the earlier capture.** Never describe a previous version you have not read. If no capture exists the tool says so, and reporting that plainly is the correct finding — a comparison you invent is worse than one you cannot make.
4. Prefer primary and authoritative sources. Say which source a claim came from.

Everything you return is external context, never a document fact. Attribute it explicitly — "According to <source> on the web, ..." — so the supervisor cannot mistake it for something from the user's files.
${FINDING_CONTRACT}`;

const VERIFIER_PROMPT = `You are an adversarial verifier. You are given a claim and you look for what would undermine it.

You are not here to confirm anything. Your value is entirely in what you find that contradicts, narrows or supersedes the claim:

**Make at most three searches.** You already have the findings and their exact wording, so you are not hunting for vocabulary — you are checking a narrow, specific class of thing. Choose your three well:

1. Exceptions, conditions, thresholds, effective dates and superseding rules.
2. The claim's own terms *negated* — where a document states when something does not apply, or applies differently.
3. Wording that would appear in a caveat, not wording that would appear in a confirmation.

Also compare the findings against each other as you read them: where two bear on the same point and disagree, say which says what. That costs no searches at all.

Report honestly and quickly. If nothing you found undermines the claims, say so plainly — a clean verification is a real result, and reaching it fast matters because every other researcher has already finished and the answer is waiting on you. Do not manufacture doubt, and do not keep searching for a caveat that is not there.
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
function subagentMiddleware(toolNames: string[] = []): AnyAgentMiddleware[] {
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
      // As on the supervisor: a refused credential cannot be retried into
      // working, and a branch that spends its backoff discovering that is a
      // branch the whole fan-out waits on for nothing.
      retryOn: (error: Error) => !isPermanentRefusal(error),
    }),
    toolRetryMiddleware({ maxRetries: 2, backoffFactor: 2, onFailure: "continue" }),
  ];

  // Researchers draw from the gateways most likely to mangle a tool call, since
  // the pool is rotated per branch precisely to spread across providers.
  if (toolNames.length > 0) middleware.push(toolCallRepairMiddleware(toolNames));

  const fallbacks = getSubagentFallbackModels("pro");
  if (fallbacks.length > 0) middleware.push(modelFallbackMiddleware(...fallbacks));

  return middleware;
}

/**
 * A research branch opening or settling.
 *
 * Reported as it happens rather than with the batch, because a fan-out returns
 * one value after its slowest member and the wait is long — measured at around
 * eighty seconds, during which the interface could say only that research was
 * happening. Branches finish at very different times (5s to 99s in one
 * measurement), so most of that silence was spent with findings already in hand
 * and nothing to show for them.
 */
export interface BranchEvent {
  id: string;
  researcher: string;
  question: string;
  phase: "start" | "done";
  /** On settle: how the branch ended, which the interface distinguishes. */
  outcome?: "ok" | "partial" | "failed";
  ms?: number;
}

export interface SubagentOptions {
  /** Give the researchers web access. Off keeps every branch on the documents. */
  webSearch?: boolean;
  /** Shared result cache, so branches do not re-run each other's searches. */
  resultCache?: Map<string, Promise<string>>;
  /** Restrict every researcher to one conversation's documents. */
  threadId?: ThreadScope;
  /**
   * Called as each branch starts and settles.
   *
   * The tool has no route to the event stream — it returns a string to the
   * model — so progress has to leave through a callback supplied by whoever
   * built the agent. Optional, and a run with no listener behaves exactly as
   * before.
   */
  onBranch?: (event: BranchEvent) => void;
}

/** The researcher a delegation is addressed to. */
const RESEARCHERS = ["document-researcher", "web-researcher", "verifier"] as const;
type ResearcherName = (typeof RESEARCHERS)[number];

/**
 * Will this batch trigger the automatic adversarial check?
 *
 * Exported so the API layer can show the verifier branch in the UI without
 * restating the rule. The check runs inside the delegation tool, which has no
 * route to the event stream, so the alternative was for the route to re-derive
 * the condition — and a duplicated predicate drifts, leaving the interface
 * either showing a researcher that never ran or hiding one that did.
 *
 * Necessarily an approximation of one part: whether a check has already run
 * this turn is state inside the tool, so a caller sees "would qualify" rather
 * than "will run". The route uses it for the first batch, which is the one that
 * triggers it.
 */
export function batchQualifiesForVerification(
  tasks: Array<{ researcher?: string }>,
): boolean {
  if (!SUBAGENT_CONFIG.AUTO_VERIFY) return false;
  // Verifying a verifier is circular.
  if (tasks.some((t) => t.researcher === "verifier")) return false;
  if (tasks.length >= SUBAGENT_CONFIG.AUTO_VERIFY_MIN_FINDINGS) return true;

  // A lone web branch is the exception to the "one finding is a gap-filling
  // lookup" rule, and it earns the extra branch.
  //
  // A passage came from a file the user uploaded and can be shown to them; a
  // web finding is whatever a model chose to read, and when it is the turn's
  // only evidence the whole answer rests on it with nothing to cross-check it
  // against. A measured turn asked for an audit comparing two versions of a
  // page, retrieved one of them, and wrote a scored comparison of both — on a
  // single web branch that nothing attacked.
  return tasks.some((t) => t.researcher === "web-researcher");
}

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
 * Each specialist is compiled once *per concurrency slot*, not once per type,
 * and every compilation calls `getSubagentModel` again — which advances the
 * credential rotation. Branch `i` of a fan-out therefore runs on a different
 * key from branch `j`.
 *
 * That pooling is the whole point, and it was added on evidence. Compiling one
 * agent per type meant every concurrent `document-researcher` shared a single
 * model instance and so a single key: the branches dispatched together and then
 * queued behind one credential's per-minute limit, retrying with backoff. Once
 * the embedding cache removed Cohere as the dominant cost this became the
 * binding constraint — a lone branch took 12.4s while three concurrent branches
 * took 128.1s, which is worse than running them one at a time.
 *
 * A pool is the cheap version of the obvious fix. Rotating credentials *inside*
 * a chat model would mean wrapping streaming, tool binding and structured
 * output; compiling a handful of agents costs nothing but memory, because
 * compilation is local work with no network in it.
 */
function compileResearchers(collector: EvidenceCollector, options: SubagentOptions) {
  const { webSearch = true, resultCache, threadId } = options;

  const researchTools = (label: string, allowWeb: boolean) =>
    buildTools(collector, {
      webSearch: allowWeb,
      mode: "focused",
      resultCache,
      label,
      threadId,
    });

  // A factory rather than a literal, because `model` must be evaluated afresh
  // for every compilation: `getSubagentModel` advances the credential rotation,
  // so building the spec once would bake one key into every instance and
  // reintroduce exactly the contention the pool exists to remove.
  const specFor = (name: ResearcherName): SubAgent => {
    // Built for the requested specialist only. The web researcher is the one
    // that gets web access; when it is switched off it falls back to the
    // document tools, so a misaddressed delegation still does useful work
    // rather than failing.
    const tools = researchTools(name, name === "web-researcher" ? webSearch : false);
    const middleware = subagentMiddleware(tools.map((t) => t.name));

    const base: Record<ResearcherName, Omit<SubAgent, "model" | "tools" | "middleware">> = {
      "document-researcher": {
        name: "document-researcher",
        description: "Investigates one sub-question against the uploaded documents.",
        systemPrompt: DOCUMENT_RESEARCHER_PROMPT,
        mode: "isolated",
      },
      verifier: {
        name: "verifier",
        description: "Adversarially checks one claim for exceptions and contradictions.",
        systemPrompt: VERIFIER_PROMPT,
        mode: "isolated",
      },
      "web-researcher": {
        name: "web-researcher",
        description: "Investigates one sub-question against public web sources.",
        systemPrompt: WEB_RESEARCHER_PROMPT,
        mode: "isolated",
      },
    };
    return { ...base[name], tools, middleware, model: getSubagentModel("pro") };
  };

  // One compiled agent per (specialist, slot). Compiled lazily, because most
  // turns never use every slot, and reused for the rest of the run.
  const pool = new Map<string, ReturnType<typeof createSubAgent>>();

  return (name: ResearcherName, slot = 0) => {
    // Slots wrap, so a fan-out wider than the pool reuses keys rather than
    // failing — degraded, not broken.
    const key = `${name}:${slot % SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND}`;
    let agent = pool.get(key);
    if (!agent) {
      agent = createSubAgent(specFor(name));
      pool.set(key, agent);
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

  /** Delegation rounds so far, so branch ids stay unique across a turn. */
  let roundSeq = 0;

  /**
   * Whether this turn's adversarial check has already run.
   *
   * Held per tool instance, and `buildDelegationTool` is called once per
   * request, so it scopes to one turn: the check happens on the first
   * substantial round and not again on the gap-filling rounds after it.
   */
  let verified = false;

  /** Run one branch, returning its finding or a note about why it has none. */
  const runBranch = async (
    researcher: ResearcherName,
    question: string,
    label: string,
    // The concurrency slot decides which pooled agent — and so which
    // credential — this branch runs on. Branches in one fan-out must pass
    // distinct slots or they collide on a single key's per-minute limit.
    slot = 0,
    recursionLimit: number = SUBAGENT_CONFIG.BRANCH_RECURSION_LIMIT,
    // Identifies this branch to the interface. Generated by the caller so the
    // start and settle events pair up.
    branchId = `${researcher}-${slot}`,
  ): Promise<string> => {
    // Per-branch timing, logged unconditionally.
    //
    // A fan-out's cost is not visible from the outside: the tool returns one
    // value after the slowest branch, so a batch that is silently serialising,
    // or one branch that is far slower than the rest, looks identical to a
    // healthy one. Reasoning about it from the total was wrong twice — first
    // blaming Cohere, then credential contention — and each wrong guess cost a
    // full measurement cycle. Start and end times per branch settle it.
    const startedAt = Date.now();
    console.log(`[branch] ${researcher}#${slot} start +0ms`);
    options.onBranch?.({ id: branchId, researcher, question, phase: "start" });

    // One place for both the log line and the event, so a branch cannot settle
    // in the interface without settling in the log or the reverse.
    const done = (outcome: string) => {
      const ms = Date.now() - startedAt;
      console.log(`[branch] ${researcher}#${slot} ${outcome} in ${ms}ms`);
      options.onBranch?.({
        id: branchId,
        researcher,
        question,
        phase: "done",
        outcome: outcome === "ok" ? "ok" : outcome === "FAILED" ? "failed" : "partial",
        ms,
      });
    };

    /**
     * Latest assistant prose seen from this branch, kept outside the try.
     *
     * Streamed rather than awaited as one value so that a branch which runs out
     * of budget still contributes what it learned. `invoke` throws on the
     * recursion limit and the entire result is lost with it: a measured branch
     * researched for 81 seconds, hit the ceiling, and returned nothing at all,
     * while the supervisor was told the sub-question could not be researched.
     * Streaming keeps each assistant turn as it arrives, so the throw costs the
     * final tidy-up rather than the whole branch.
     */
    let partial = "";

    const readText = (content: unknown): string =>
      typeof content === "string"
        ? content
        : (content as Array<{ type?: string; text?: string }> | undefined)
            ?.filter((b) => b?.type === "text")
            .map((b) => b.text)
            .join("") ?? "";

    try {
      const stream = await researcherFor(researcher, slot).stream(
        { messages: [new HumanMessage(question)] },
        // Bounded here rather than by middleware inside the branch: a recursion
        // limit passed at invocation is per-call by construction and cannot be
        // inherited from the supervisor's state the way the limit middlewares'
        // counters are.
        { recursionLimit, streamMode: "updates" },
      );

      for await (const update of stream) {
        for (const node of Object.values(
          (update ?? {}) as Record<string, { messages?: unknown[] }>,
        )) {
          const last = node?.messages?.at?.(-1) as
            | { tool_calls?: unknown[]; getType?: () => string; content?: unknown }
            | undefined;
          // Only finished assistant prose. A message carrying tool calls is the
          // branch deciding what to do next, not reporting what it found.
          if (last?.getType?.() === "ai" && !(last.tool_calls as unknown[])?.length) {
            const t = readText(last.content);
            if (t.trim()) partial = t;
          }
        }
      }

      const text = partial;

      // A branch whose model calls were exhausted returns normally, and the
      // retry middleware's own message becomes its "finding".
      //
      // `modelRetryMiddleware` runs with onFailure "continue", so a branch that
      // never reached a model still resolves successfully, carrying text like
      // "Model call failed after 4 attempts with ... 429 Daily token limit
      // exceeded". Passed on unchanged, the supervisor reads that as research:
      // a measured turn synthesised an answer explaining the rate limits to the
      // user as though that were the finding, while every evidence check
      // reported zero passages and no citations.
      //
      // Labelling it a failure keeps the distinction the whole design rests on
      // — between the documents being silent and the infrastructure being
      // unavailable — which the supervisor cannot otherwise tell apart.
      if (/^Model call failed after \d+ attempts/.test(text.trim())) {
        done("EXHAUSTED");
        console.warn(`[delegate] ${researcher} exhausted its retries`);
        return (
          `${label}\nFAILED — the model provider refused every attempt ` +
          `(${text.replace(/\s+/g, " ").slice(0, 160)}).\n` +
          `This sub-question was NOT researched. Do not treat this as evidence, and do not ` +
          `conclude the documents are silent on it — say plainly that it could not be checked.`
        );
      }

      done("ok");

      // Repair the finding's citation markers before the supervisor ever sees
      // them.
      //
      // Researchers reliably invent an annotated marker format — measured
      // output cited passages as `【2†L1-L2】` rather than `[2]` — and the
      // damage compounds through delegation rather than staying local. The
      // supervisor is told to reuse a finding's markers verbatim, so it
      // faithfully copies a broken one, or gives up and cites nothing; either
      // way the answer reaches validation with no recognisable citations and is
      // pronounced valid, because a marker in an unknown shape is not seen as
      // invalid, it is not seen at all. Two consecutive delegating turns
      // produced answers with no citations for exactly this reason.
      //
      // Repairing here rather than only on the final answer means the
      // supervisor synthesises from clean markers, which is also what its own
      // instructions assume.
      return `${label}\n${
        normaliseCitations(text).trim() || "(the researcher returned nothing)"
      }`;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const exhausted = /[Rr]ecursion limit/.test(message);

      // A branch that ran out of steps usually has findings, just no closing
      // summary — it was still searching when the ceiling stopped it. Returning
      // what it gathered is far better than discarding the work and telling the
      // supervisor the sub-question is unresearched, which invites it either to
      // spend another round re-asking or to report a gap that is not real.
      if (partial.trim()) {
        done(exhausted ? "PARTIAL (budget)" : "PARTIAL (error)");
        console.warn(`[delegate] ${researcher} returned partial: ${message.slice(0, 100)}`);
        return (
          `${label}\n${normaliseCitations(partial).trim()}\n\n` +
          `(PARTIAL — this researcher ${
            exhausted ? "reached its search budget" : "hit an error"
          } before finishing. What is above was found and is usable; treat anything it does ` +
          `not mention as unchecked rather than absent.)`
        );
      }

      done("FAILED");
      console.warn(`[delegate] ${researcher} failed: ${message.slice(0, 120)}`);
      return `${label}\nFAILED — ${message.slice(0, 200)}\nTreat this sub-question as unresearched.`;
    }
  };

  return tool(
    async ({ tasks }: { tasks: Array<{ researcher: ResearcherName; question: string }> }) => {
      const batch = tasks.slice(0, SUBAGENT_CONFIG.MAX_BRANCHES_PER_ROUND);
      if (batch.length === 0) return "No tasks supplied. Provide at least one sub-question.";

      // Rounds are numbered so a second delegation's branches cannot be
      // mistaken for the first's still running.
      const round = ++roundSeq;

      console.log(
        `[delegate] ${batch.length} researcher(s) in parallel: ` +
          batch.map((t) => t.researcher).join(", "),
      );

      const findings = await Promise.all(
        batch.map((t, i) =>
          runBranch(
            t.researcher,
            t.question,
            // Never bracketed. The findings themselves are full of `[1]`, `[2]`
            // citation markers, and labelling the branches the same way would
            // put two numbering schemes in one payload for the supervisor to
            // confuse — with the failure landing on citations, which are the
            // one thing here that has to be exactly right.
            `### Finding ${i + 1} of ${batch.length} · ${t.researcher}\nSub-question: ${t.question}`,
            // The branch index is the concurrency slot, so each concurrent
            // branch draws a different pooled agent and therefore a different
            // credential. Passing a constant here would put the whole fan-out
            // back on one key.
            i,
            undefined,
            `r${round}-b${i}`,
          ),
        ),
      );

      const sections = [...findings];

      // The adversarial pass, run here rather than asked for in the prompt.
      // See SUBAGENT_CONFIG.AUTO_VERIFY: as a workflow step it was skipped every
      // time, and a check that only runs when the answer already looks complete
      // is the one that never runs at all.
      // `batchQualifiesForVerification` also covers the supervisor having asked
      // for a verifier itself, in which case checking twice would spend a
      // branch to reach the same conclusion.
      const substantial = !verified && batchQualifiesForVerification(batch);

      if (substantial) {
        verified = true;
        console.log("[delegate] adversarial verification of the findings");

        // Sequential by necessity: the verifier's input is what the researchers
        // returned, so it cannot overlap with them.
        sections.push(
          await runBranch(
            "verifier",
            "Below are research findings drawn from the user's documents. Identify the " +
              "load-bearing claims and attack them: search for exceptions, conditions, " +
              "thresholds, effective dates, superseding rules, and any place two documents " +
              "disagree. Report only what genuinely undermines, narrows or qualifies a " +
              "claim, and say plainly if you find nothing.\n\n" +
              findings.join("\n\n---\n\n"),
            "### Adversarial check of the findings above · verifier",
            0,
            // Tighter than a researcher's: this branch is serial, so its cost
            // lands whole on the turn.
            SUBAGENT_CONFIG.VERIFY_RECURSION_LIMIT,
            `r${round}-verify`,
          ),
        );
      }

      return (
        `${findings.length} finding(s)${substantial ? " and an adversarial check" : ""}:\n\n` +
        sections.join("\n\n---\n\n")
      );
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
