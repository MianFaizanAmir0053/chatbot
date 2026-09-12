/**
 * Live probe: can the Alibaba MaaS endpoint drive this project's agent loop?
 *
 * The bench in the scratchpad measured the raw HTTP surface. This measures the
 * thing that actually matters — `createAgent` from `langchain`, the same call
 * `lib/agents/agent.ts` makes, talking to Qwen through `ChatDeepSeek` (an
 * OpenAI-compatible wrapper). It is given real filesystem tools over this repo
 * and a question that cannot be answered in one lookup, so a model that cannot
 * sustain a multi-step tool loop fails visibly rather than plausibly.
 *
 * Run: npx tsx scripts/alibaba-probe.ts [model]
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { createAgent, tool } from "langchain";
import { ChatDeepSeek } from "@langchain/deepseek";
import { z } from "zod";

const BASE_URL =
  process.env.AL_BASE ??
  "https://ws-w7d45hqpi51jfix5.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";
const API_KEY = process.env.AL_KEY ?? "";
const MODEL = process.argv[2] ?? "qwen3.8-flash";

if (!API_KEY) {
  console.error("AL_KEY not set");
  process.exit(1);
}

const ROOT = resolve(import.meta.dirname, "..");

/** Keep the agent inside the repo regardless of what path it invents. */
function safe(relative: string): string {
  const full = resolve(ROOT, relative);
  if (!full.startsWith(ROOT)) throw new Error("path outside repo: " + relative);
  return full;
}

let toolCalls = 0;

const listDir = tool(
  ({ path }) => {
    toolCalls++;
    const full = safe(path);
    return readdirSync(full)
      .filter((name) => name !== "node_modules" && !name.startsWith("."))
      .map((name) => (statSync(join(full, name)).isDirectory() ? name + "/" : name))
      .join("\n");
  },
  {
    name: "list_dir",
    description: "List files and directories at a repo-relative path.",
    schema: z.object({ path: z.string().describe("Repo-relative directory, e.g. 'lib'") }),
  },
);

const readFile = tool(
  ({ path, start, end }) => {
    toolCalls++;
    const lines = readFileSync(safe(path), "utf8").split("\n");
    const from = Math.max(1, start ?? 1);
    const to = Math.min(lines.length, end ?? from + 120);
    return lines
      .slice(from - 1, to)
      .map((line, i) => `${from + i}\t${line}`)
      .join("\n");
  },
  {
    name: "read_file",
    description: "Read a repo-relative file, optionally a line range. Returns numbered lines.",
    schema: z.object({
      path: z.string(),
      start: z.number().optional(),
      end: z.number().optional(),
    }),
  },
);

const grepRepo = tool(
  ({ pattern, dir }) => {
    toolCalls++;
    const re = new RegExp(pattern);
    const hits: string[] = [];
    const walk = (d: string) => {
      for (const name of readdirSync(d)) {
        if (name === "node_modules" || name.startsWith(".")) continue;
        const full = join(d, name);
        if (statSync(full).isDirectory()) {
          walk(full);
        } else if (/\.(ts|tsx|js|mjs)$/.test(name)) {
          readFileSync(full, "utf8")
            .split("\n")
            .forEach((line, i) => {
              if (hits.length < 40 && re.test(line)) {
                hits.push(`${full.slice(ROOT.length + 1).replace(/\\/g, "/")}:${i + 1}: ${line.trim().slice(0, 160)}`);
              }
            });
        }
      }
    };
    walk(safe(dir ?? "lib"));
    return hits.length ? hits.join("\n") : "no matches";
  },
  {
    name: "grep_repo",
    description: "Regex search source files under a repo-relative directory. Returns path:line: text.",
    schema: z.object({ pattern: z.string(), dir: z.string().optional() }),
  },
);

/**
 * Retry the endpoint's throttle.
 *
 * Alibaba's free tier reports throttling as `403 Free quota exhausted`, rejected
 * in ~130ms before any inference. It is transient — the same model answers
 * seconds later — but 403 is not in any SDK's retryable set, so one throttled
 * call anywhere in a tool loop kills the whole run. Retrying here rather than
 * around `invoke` matters: a mid-loop failure would otherwise discard every
 * tool result gathered so far.
 */
let throttled = 0;
async function retryingFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(input, init);
    if (response.status !== 403 || attempt >= 6) return response;
    const body = await response.clone().text();
    if (!/quota|rate|limit/i.test(body)) return response;
    throttled++;
    await new Promise((s) => setTimeout(s, 1500 * 2 ** attempt));
  }
}

// kimi-k3 rejects the parameter outright: "Parameter 'temperature'=0.1 is not
// supported for kimi-k3 model" — a 400 on the first call, before any tool runs.
const acceptsTemperature = !/^kimi/.test(MODEL);

const model = new ChatDeepSeek({
  model: MODEL,
  apiKey: API_KEY,
  ...(acceptsTemperature ? { temperature: 0.1 } : {}),
  maxTokens: 4000,
  configuration: { baseURL: BASE_URL, fetch: retryingFetch },
});

const agent = createAgent({
  model: model as never,
  tools: [listDir, readFile, grepRepo],
  systemPrompt: `You are a code archaeologist working in a TypeScript repo.

Use the tools to find real evidence before you answer. Never guess a filename — list or grep for it.
Every claim about the code must carry a \`path:line\` citation you actually saw in tool output.
When you have enough evidence, answer directly. Do not ask the user for permission to use a tool.`,
});

const QUESTION = `How does this codebase decide which LLM provider serves the agent, and what happens to a credential that gets refused? Cover: the tier system, the key rotation, and why Cohere is excluded from the agent loop. Cite path:line for each claim. Be concise — under 300 words.`;

const t0 = Date.now();
let usageIn = 0;
let usageOut = 0;
let steps = 0;

try {
  const result = await agent.invoke(
    { messages: [{ role: "user", content: QUESTION }] },
    { recursionLimit: 40 },
  );

  for (const message of result.messages) {
    const meta = message as unknown as {
      usage_metadata?: { input_tokens?: number; output_tokens?: number };
      getType?: () => string;
    };
    if (meta.usage_metadata) {
      usageIn += meta.usage_metadata.input_tokens ?? 0;
      usageOut += meta.usage_metadata.output_tokens ?? 0;
      steps++;
    }
  }

  const last = result.messages.at(-1);
  const text =
    typeof last?.content === "string"
      ? last.content
      : JSON.stringify(last?.content).slice(0, 4000);

  console.log("=".repeat(72));
  console.log(`model=${MODEL}`);
  console.log(
    `elapsed=${((Date.now() - t0) / 1000).toFixed(1)}s model_calls=${steps} tool_calls=${toolCalls} ` +
      `throttle_retries=${throttled} tokens in=${usageIn} out=${usageOut}`,
  );
  console.log("=".repeat(72));
  console.log(text);
  console.log("=".repeat(72));

  // Cheap automatic scoring: did it cite files that actually exist and are relevant?
  const citations = [...text.matchAll(/([\w./-]+\.ts):(\d+)/g)].map((m) => m[1]);
  const unique = [...new Set(citations)];
  console.log(`citations=${citations.length} unique_files=${unique.length} -> ${unique.join(", ")}`);
  const realFiles = unique.filter((f) => {
    try {
      statSync(safe(f.replace(/^chatbot\//, "")));
      return true;
    } catch {
      return false;
    }
  });
  console.log(`citations_pointing_at_real_files=${realFiles.length}/${unique.length}`);
  const topics = {
    tiers: /pro|fast|tier/i.test(text),
    rotation: /rotat|cursor|round.?robin|pool/i.test(text),
    refusal: /refus|noteFailure|credential-health|permanent/i.test(text),
    cohere: /cohere/i.test(text),
  };
  console.log("topics_covered=" + JSON.stringify(topics));
} catch (error) {
  console.log(`FAILED after ${throttled} throttle retries, ${((Date.now() - t0) / 1000).toFixed(1)}s, tool_calls=${toolCalls}`);
  console.log(String((error as Error).message).slice(0, 1200));
  process.exitCode = 1;
}
