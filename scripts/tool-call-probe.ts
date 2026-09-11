/**
 * Which providers return a tool call as a tool call?
 *
 * A gateway that accepts a tools array and then describes the call in prose —
 * or emits it in some template syntax of its own — looks completely healthy from
 * the outside. The request succeeds, tokens come back, and the text reads like
 * an answer. What actually happened is that the tool never ran, so the reply is
 * ungrounded, and the raw call markup is shown to the user as if it were the
 * answer.
 *
 * That is what `<uncensored_tool_call>search_documents<arg_key>query</arg_key>`
 * arriving at the top of a reply means. It is a template the model emitted as
 * text, not a call anything executed.
 *
 * Usage: npx tsx --env-file=.env scripts/tool-call-probe.ts
 */
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { LLM_PROVIDERS } from "../lib/config";
import { providerModel } from "../lib/models";

const search = tool(async ({ query }: { query: string }) => `no results for ${query}`, {
  name: "search_documents",
  description: "Search the indexed documents for passages relevant to a question.",
  schema: z.object({ query: z.string().describe("The search query") }),
});

/**
 * Call-like markup that arrived as text instead of as a tool call.
 *
 * Deliberately matches the shape rather than one vendor's tag: several
 * families have their own template (`<tool_call>`, `<|python_tag|>`,
 * `<function=...>`), and an "uncensored" fine-tune renames it again. What they
 * share is a tag or key/value wrapper naming a tool, in the content.
 */
const LEAKED_CALL =
  /<\|?[a-z_]*tool_call\|?>|<arg_key>|<arg_value>|<\|python_tag\|>|<function(?:_call)?[=>]|```tool_code/i;

async function probe(name: string, model: ReturnType<typeof providerModel>) {
  try {
    const response = await model.bindTools!([search]).invoke([
      new SystemMessage(
        "You answer from indexed documents. Always call search_documents before answering.",
      ),
      new HumanMessage("What is the engine oil capacity?"),
    ]);

    const calls = (response as { tool_calls?: Array<{ name: string }> }).tool_calls ?? [];
    const content = typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    const leaked = LEAKED_CALL.test(content);

    if (leaked) {
      const sample = content.replace(/\s+/g, " ").slice(0, 100);
      return `LEAKS  emitted call markup as text: ${sample}`;
    }
    if (calls.length > 0) {
      const names = calls.map((c) => c.name);
      const bad = names.filter((n) => n !== "search_documents");
      return bad.length > 0
        ? `MANGLED tool name(s): ${bad.join(", ")}`
        : `OK     called ${names.join(", ")}`;
    }
    return `NOCALL answered without calling: ${content.replace(/\s+/g, " ").slice(0, 70)}`;
  } catch (error) {
    return `ERROR  ${(error as Error).message.replace(/\s+/g, " ").slice(0, 70)}`;
  }
}

const ROUNDS = Number(process.argv[2] ?? 5);

async function main() {
  const names = [...new Set(LLM_PROVIDERS.map((p) => p.name))];

  console.log(`Does each gateway return a tool call as a tool call? (${ROUNDS} rounds)\n`);
  for (const name of names) {
    // Walk this gateway's keys rather than always taking the first: with most
    // of one provider's credentials refused, testing only the first key
    // measures the account, not the model.
    const keys = LLM_PROVIDERS.filter((p) => p.name === name);
    const tally = new Map<string, number>();
    let sample = "";

    for (let round = 0; round < ROUNDS; round++) {
      const p = keys[round % keys.length];
      const status = await probe(name, providerModel(p, "pro", 0));
      const verdict = status.slice(0, 7).trim();
      tally.set(verdict, (tally.get(verdict) ?? 0) + 1);
      if ((verdict === "LEAKS" || verdict === "MANGLED" || verdict === "NOCALL") && !sample) {
        sample = status;
      }
    }

    const summary = [...tally.entries()].map(([v, n]) => `${v}×${n}`).join("  ");
    console.log(`  ${name.padEnd(12)} ${keys[0].pro.padEnd(38)} ${summary}`);
    if (sample) console.log(`  ${" ".repeat(12)} ${sample}`);
  }
}

void main();

export {};
