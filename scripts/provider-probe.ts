/**
 * Probe every configured OpenAI-compatible provider and rank them.
 *
 * Ranking drives the fallback chain order in lib/models.ts. What matters for a
 * multi-step agent is not single-call latency but how many calls in quick
 * succession a provider will accept before it rate-limits, so the probe issues
 * a short burst and records how much of it survived.
 */

type Provider = {
  name: string;
  baseURL: string;
  apiKey: string;
  pro: string;
  fast: string;
  /** Extra headers a gateway requires for attribution or routing. */
  headers?: Record<string, string>;
};

const PROVIDERS: Provider[] = [
  {
    name: "groq",
    baseURL: "https://api.groq.com/openai/v1",
    apiKey: process.env.GROQ_API_KEY ?? "",
    pro: "openai/gpt-oss-120b",
    fast: "openai/gpt-oss-20b",
  },
  {
    name: "bluesminds",
    baseURL: "https://api.bluesminds.com/v1",
    apiKey: process.env.BLUESMINDS_API_KEY ?? "",
    pro: "gpt-5.6-terra",
    fast: "gpt-5.6-luna",
  },
  {
    name: "openrouter",
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY ?? "",
    pro: "openai/gpt-4o",
    fast: "openai/gpt-4o-mini",
    headers: {
      "HTTP-Referer": process.env.OPENROUTER_SITE_URL ?? "http://localhost:3000",
      "X-Title": process.env.OPENROUTER_SITE_NAME ?? "Agentic RAG",
    },
  },
  {
    name: "mistral",
    baseURL: "https://api.mistral.ai/v1",
    apiKey: process.env.MISTRAL_API_KEY ?? "",
    // The free tier rate-limits mistral-small / magistral-small on the first
    // call; the ministral line answers reliably and still supports tools.
    pro: "ministral-14b-latest",
    fast: "ministral-8b-latest",
  },
  {
    name: "bazaarlink",
    baseURL: "https://api.bazaarlink.ai/v1",
    apiKey: process.env.BAZAARLINK_API_KEY ?? "",
    pro: "deepseek-v4-pro",
    fast: "deepseek-v4-flash",
  },
];

const TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};

async function call(
  p: Provider,
  model: string,
  withTool: boolean,
): Promise<{ ok: boolean; ms: number; detail: string; toolOk?: boolean }> {
  const started = Date.now();
  try {
    const res = await fetch(`${p.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${p.apiKey}`,
        "Content-Type": "application/json",
        ...p.headers,
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "user",
            content: withTool ? "What is the weather in Paris? Use the tool." : "Reply with: ok",
          },
        ],
        ...(withTool ? { tools: [TOOL] } : {}),
        max_tokens: 400,
      }),
      signal: AbortSignal.timeout(60_000),
    });

    const ms = Date.now() - started;
    const text = await res.text();
    if (!res.ok) return { ok: false, ms, detail: `HTTP ${res.status} ${text.slice(0, 70)}` };

    const json = JSON.parse(text);
    if (json.error) return { ok: false, ms, detail: String(json.error.message).slice(0, 70) };

    const msg = json.choices?.[0]?.message;
    return {
      ok: true,
      ms,
      detail: JSON.stringify(msg?.content ?? "").slice(0, 40),
      toolOk: Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0,
    };
  } catch (error) {
    return { ok: false, ms: Date.now() - started, detail: (error as Error).message.slice(0, 70) };
  }
}

async function main() {
  const results: Array<{ name: string; score: number; line: string }> = [];

  for (const p of PROVIDERS) {
    if (!p.apiKey) {
      console.log(`${p.name}: no key supplied, skipped`);
      continue;
    }

    const basic = await call(p, p.pro, false);
    const tool = basic.ok ? await call(p, p.pro, true) : { ok: false, toolOk: false, ms: 0, detail: "skipped" };
    const fast = await call(p, p.fast, false);

    // Burst: six sequential calls, the shape of one agent turn's fan-out.
    let survived = 0;
    let firstFailure = "";
    for (let i = 0; i < 6; i++) {
      const r = await call(p, p.fast, false);
      if (r.ok) survived++;
      else if (!firstFailure) firstFailure = r.detail;
    }

    // Availability first, then tool support, then burst tolerance, then speed.
    const score =
      (basic.ok ? 1000 : 0) +
      (tool.toolOk ? 500 : 0) +
      (fast.ok ? 200 : 0) +
      survived * 50 -
      Math.min(basic.ms, 20_000) / 1000;

    const line =
      `${p.name.padEnd(12)} pro=${basic.ok ? "OK" : "FAIL"}(${basic.ms}ms) ` +
      `tools=${tool.toolOk ? "yes" : "no"} fast=${fast.ok ? "OK" : "FAIL"} ` +
      `burst=${survived}/6${firstFailure ? ` first-fail="${firstFailure}"` : ""}` +
      `${basic.ok ? "" : ` detail="${basic.detail}"`}`;

    results.push({ name: p.name, score, line });
    console.log(line);
  }

  results.sort((a, b) => b.score - a.score);
  console.log("\n=== availability order ===");
  results.forEach((r, i) => console.log(`${i + 1}. ${r.name}  (score ${Math.round(r.score)})`));
}

void main();

// Marks the file as a module. Without it TypeScript treats these top-level
// declarations as globals, which collides with the sibling audit script.
export {};
