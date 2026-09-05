/**
 * Verify every provider in the configured chain answers through the real model
 * layer — not just curl — so base URL, auth and any required headers are proven
 * as the application actually builds them.
 */
import { HumanMessage } from "@langchain/core/messages";
import { LLM_PROVIDERS } from "../lib/config";
import { providerModel } from "../lib/models";

async function main() {
  console.log(`chain: ${LLM_PROVIDERS.map((p) => p.name).join(" -> ")}\n`);

  for (const provider of LLM_PROVIDERS) {
    const started = Date.now();
    try {
      const model = providerModel(provider, "fast", 0);
      const res = await model.invoke([new HumanMessage("Reply with exactly: ok")]);
      const text = String(res.content).replace(/\s+/g, " ").slice(0, 40);
      console.log(`${provider.name.padEnd(12)} OK   ${Date.now() - started}ms  "${text}"`);
    } catch (error) {
      console.log(
        `${provider.name.padEnd(12)} FAIL ${Date.now() - started}ms  ${(error as Error).message.slice(0, 90)}`,
      );
    }
  }
}

void main();
