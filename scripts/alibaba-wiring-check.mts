/**
 * Verify the two things the wiring depends on, through the real client class.
 *
 * 1. `modelKwargs: { enable_thinking: false }` actually reaches the endpoint.
 *    Without it a FAST-tier grader spends its whole `max_tokens` on reasoning
 *    and returns empty content with `finish_reason: "length"` — measured on
 *    deepseek-v4-pro at max_tokens 16. A grader that returns "" fails open.
 * 2. Streaming works, since the chat route streams every answer.
 *
 * Run: npx tsx scripts/alibaba-wiring-check.mts
 */
import { ChatDeepSeek } from "@langchain/deepseek";

const BASE_URL =
  process.env.AL_BASE ??
  "https://ws-w7d45hqpi51jfix5.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";
const API_KEY = process.env.AL_KEY ?? "";

const GRADER = [
  { role: "system" as const, content: 'You are a relevance grader. Reply with exactly one word: "yes" or "no".' },
  {
    role: "user" as const,
    content:
      'Passage: "Section 12.3: Either party may terminate upon sixty days written notice."\nQuestion: "What is the notice period?"\nIs the passage relevant?',
  },
];

function build(modelName: string, kwargs?: Record<string, unknown>) {
  return new ChatDeepSeek({
    model: modelName,
    apiKey: API_KEY,
    temperature: 0,
    maxTokens: 16,
    ...(kwargs ? { modelKwargs: kwargs } : {}),
    configuration: { baseURL: BASE_URL },
  });
}

async function graderCall(label: string, modelName: string, kwargs?: Record<string, unknown>) {
  const t0 = Date.now();
  try {
    const reply = await build(modelName, kwargs).invoke(GRADER);
    const text = typeof reply.content === "string" ? reply.content.trim() : JSON.stringify(reply.content);
    const out = reply.usage_metadata?.output_tokens ?? "-";
    console.log(
      `${label.padEnd(50)} ${String(Date.now() - t0 + "ms").padEnd(8)} out=${String(out).padEnd(5)} ` +
        `${text ? `answer="${text.slice(0, 24)}"` : "*** EMPTY ANSWER ***"}`,
    );
  } catch (error) {
    console.log(`${label.padEnd(50)} FAILED ${String((error as Error).message).slice(0, 120)}`);
  }
}

console.log("=== FAST-tier grader at max_tokens=16 ===");
await graderCall("deepseek-v4-pro (thinking default)", "deepseek-v4-pro");
await graderCall("deepseek-v4-pro + enable_thinking:false", "deepseek-v4-pro", { enable_thinking: false });
await graderCall("qwen3.5-plus (thinking default)", "qwen3.5-plus");
await graderCall("qwen3.5-plus + enable_thinking:false", "qwen3.5-plus", { enable_thinking: false });
await graderCall("qwen-flash (no thinking phase)", "qwen-flash");

console.log("\n=== streaming through the same client ===");
const streamer = new ChatDeepSeek({
  model: "qwen-plus",
  apiKey: API_KEY,
  temperature: 0,
  maxTokens: 120,
  configuration: { baseURL: BASE_URL },
});
const t0 = Date.now();
let ttft: number | null = null;
let chunks = 0;
let text = "";
for await (const chunk of await streamer.stream("Count from 1 to 12, comma separated.")) {
  chunks++;
  const piece = typeof chunk.content === "string" ? chunk.content : "";
  if (piece && ttft === null) ttft = Date.now() - t0;
  text += piece;
}
console.log(`ttft=${ttft}ms total=${Date.now() - t0}ms chunks=${chunks} text="${text.trim().slice(0, 60)}"`);
