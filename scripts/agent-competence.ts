/**
 * Can this model actually drive the agent loop?
 *
 * Tool calling is necessary but not sufficient. A weaker model will call tools
 * and then emit the tool's structured output as its answer, or never reach a
 * stop condition and exhaust the recursion limit. Both look like success to a
 * tool-calling probe and like a broken product to a user, so the pro rotation
 * must be gated on a real agent turn rather than a single completion.
 *
 * Runs one full turn against the live agent and judges the answer.
 *
 * Usage: PIN_PROVIDER=groq PIN_MODEL=openai/gpt-oss-120b \
 *          npx tsx --env-file=.env scripts/agent-competence.ts
 */

const provider = process.env.PIN_PROVIDER ?? "";
const model = process.env.PIN_MODEL ?? "";

if (!provider || !model) {
  console.error("set PIN_PROVIDER and PIN_MODEL");
  process.exit(1);
}

// Point the primary slot at the pair under test, before any module reads the
// environment.
//
// Setting LLM_PROVIDER_ORDER alone is not enough and silently tests the wrong
// thing: the agent's model comes from the DEEPSEEK_* slot, so narrowing the
// registry left the primary on whatever that slot already held. Four providers
// once "failed" this check identically for exactly that reason — none of them
// had been called.
const upper = provider.toUpperCase();
const key = process.env[`${upper}_API_KEY`];
const baseURL = process.env[`${upper}_BASE_URL`];

if (!key || !baseURL) {
  console.error(`set ${upper}_API_KEY and ${upper}_BASE_URL in .env first`);
  process.exit(1);
}

process.env.LLM_PROVIDER_ORDER = provider;
process.env.DEEPSEEK_API_KEY = key;
process.env.DEEPSEEK_BASE_URL = baseURL;
process.env.DEEPSEEK_MODEL_PRO = model;
process.env.DEEPSEEK_MODEL_FAST = model;
process.env[`${upper}_MODEL_PRO`] = model;
process.env[`${upper}_MODEL_FAST`] = model;

async function main() {
  const { HumanMessage } = await import("@langchain/core/messages");
  const { buildAgent, runConfig } = await import("../lib/agents/agent");
  const { randomUUID } = await import("crypto");

  const { agent } = buildAgent({ enableTodos: true });
  const started = Date.now();

  try {
    const res = await agent.invoke(
      { messages: [new HumanMessage("What is the receipt itemisation threshold?")] },
      runConfig({ threadId: randomUUID() }),
    );

    const last = res.messages.at(-1);
    const text = String(last?.content ?? "").trim();
    const ms = Date.now() - started;

    // The failure modes that a tool-calling probe cannot see.
    const leaked = /^\s*[{[]/.test(text) && /"variants"|"query"|"queries"/.test(text);
    const empty = text.length === 0;
    const failedCall = /^Model call failed/.test(text);

    const verdict = empty
      ? "EMPTY"
      : failedCall
        ? "PROVIDER-ERROR"
        : leaked
          ? "LEAKS-TOOL-JSON"
          : "OK";

    console.log(`${verdict}\t${ms}ms\t${provider}:${model}`);
    console.log(`  ${text.replace(/\s+/g, " ").slice(0, 160)}`);
  } catch (error) {
    const msg = (error as Error).message;
    const verdict = /[Rr]ecursion limit/.test(msg) ? "RECURSION-LIMIT" : "THREW";
    console.log(`${verdict}\t${Date.now() - started}ms\t${provider}:${model}`);
    console.log(`  ${msg.replace(/\s+/g, " ").slice(0, 160)}`);
  }
}

void main();

export {};
