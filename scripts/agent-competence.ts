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

// Narrow the registry to the single pair under test before any module reads
// the environment, so the agent cannot quietly fall back to another provider
// and report its success as this model's.
process.env.LLM_PROVIDER_ORDER = provider;
process.env[`${provider.toUpperCase()}_MODEL_PRO`] = model;
process.env[`${provider.toUpperCase()}_MODEL_FAST`] = model;

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
