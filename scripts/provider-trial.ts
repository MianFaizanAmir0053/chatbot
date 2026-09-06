/**
 * Trial an unfamiliar OpenAI-compatible provider before adopting it.
 *
 * Checks the two things that decide whether it can serve this agent at all:
 * that a completion comes back, and that a tools array produces a real tool
 * call. Listing models proves neither — several gateways advertise a catalogue
 * and then reject tools, or accept the array and answer without using it,
 * which strands the agent loop while looking like success.
 *
 * Candidates are read from CANDIDATES below and keys from the environment, so
 * no credential is written into the file.
 *
 * Usage: npx tsx --env-file=.env scripts/provider-trial.ts
 */

type Candidate = { name: string; baseURL: string; envKey: string; models: string[] };

const CANDIDATES: Candidate[] = [
  {
    name: "venice",
    baseURL: "https://api.venice.ai/api/v1",
    envKey: "VENICE_API_KEY",
    models: ["zai-org-glm-5-2", "z-ai-glm-5-3", "gemini-3-8-flash"],
  },
  {
    name: "requesty",
    baseURL: "https://router.requesty.ai/v1",
    envKey: "REQUESTY_API_KEY",
    models: ["google/gemma-4-31b-it", "xai/grok-4.6", "xai/grok-4-fast"],
  },
  {
    name: "siliconflow",
    baseURL: "https://api.siliconflow.com/v1",
    envKey: "SILICONFLOW_API_KEY",
    models: ["deepseek-ai/DeepSeek-V4-Pro", "zai-org/GLM-5.3", "moonshotai/Kimi-K3"],
  },
  {
    name: "llm7",
    baseURL: "https://api.llm7.io/v1",
    envKey: "LLM7_API_KEY",
    models: ["claude-opus-5", "claude-sonnet-5", "codestral-latest"],
  },
  {
    name: "aionlabs",
    baseURL: "https://api.aionlabs.ai/v1",
    envKey: "AIONLABS_API_KEY",
    models: ["aion-labs/aion-3.0", "aion-labs/aion-3.0-mini"],
  },
  {
    name: "pollinations",
    baseURL: "https://text.pollinations.ai/openai",
    envKey: "POLLINATIONS_API_KEY",
    models: ["openai-fast"],
  },
];

const TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};

async function trial(c: Candidate, model: string, key: string): Promise<string> {
  const started = Date.now();
  try {
    const res = await fetch(`${c.baseURL}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "What is the weather in Paris? Use the tool." }],
        tools: [TOOL],
        max_tokens: 600,
      }),
      signal: AbortSignal.timeout(75_000),
    });

    const ms = Date.now() - started;
    const text = await res.text();

    if (!res.ok) {
      let detail = text.slice(0, 70);
      try {
        const j = JSON.parse(text);
        detail = String(j.error?.message ?? j.message ?? j.error ?? detail).slice(0, 70);
      } catch {
        /* non-JSON error body */
      }
      return `FAIL  ${String(ms).padStart(6)}ms  HTTP ${res.status} ${detail.replace(/\s+/g, " ")}`;
    }

    const json = JSON.parse(text);
    if (json.error) {
      return `FAIL  ${String(ms).padStart(6)}ms  ${String(json.error.message ?? json.error).slice(0, 70)}`;
    }

    const msg = json.choices?.[0]?.message;
    if (Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0) {
      return `TOOLS ${String(ms).padStart(6)}ms  usable by the agent`;
    }
    const content = String(msg?.content ?? "").replace(/\s+/g, " ").slice(0, 45);
    return `CHAT  ${String(ms).padStart(6)}ms  no tool call — "${content}"`;
  } catch (error) {
    return `FAIL  ${String(Date.now() - started).padStart(6)}ms  ${(error as Error).message.slice(0, 70)}`;
  }
}

/**
 * Sustained load, which is what an agent turn actually applies.
 *
 * A single successful call proves very little on a trial account: several of
 * these answered once and then returned 402 on the next request, because the
 * balance covered exactly that one call. One agent turn issues many, so the
 * question is not "does it respond" but "does it keep responding".
 */
async function burst(c: Candidate, model: string, key: string, n = 5): Promise<string> {
  let ok = 0;
  let firstFailure = "";

  for (let i = 0; i < n; i++) {
    const result = await trial(c, model, key);
    if (result.startsWith("TOOLS") || result.startsWith("CHAT")) ok++;
    else if (!firstFailure) firstFailure = result.replace(/^FAIL\s+\d+ms\s+/, "");
  }
  return `${ok}/${n}${firstFailure ? `  first failure: ${firstFailure}` : ""}`;
}

async function main() {
  const sustained = process.argv.includes("--burst");

  for (const c of CANDIDATES) {
    const key = process.env[c.envKey];
    console.log(`\n=== ${c.name} (${c.baseURL}) ===`);
    if (!key) {
      console.log(`  ${c.envKey} not set, skipped`);
      continue;
    }
    for (const model of c.models) {
      const single = await trial(c, model, key);
      console.log(`  ${model.padEnd(38)} ${single}`);
      if (sustained && (single.startsWith("TOOLS") || single.startsWith("CHAT"))) {
        console.log(`  ${" ".repeat(38)} sustained: ${await burst(c, model, key)}`);
      }
    }
  }
}

void main();

export {};
