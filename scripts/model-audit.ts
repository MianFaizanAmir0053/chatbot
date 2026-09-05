/**
 * Audit which models a provider key can actually use.
 *
 * A provider's model list advertises what exists, not what a given key may
 * call: free tiers withdraw models, upstream providers rate-limit independently,
 * and several models accept a tools array and then ignore it — which strands
 * this agent, since its loop cannot run without tool calling. So every model is
 * probed with a real tool-calling request.
 *
 * A 429 is retried once after a pause, because reporting a rate limit as
 * "broken" would be wrong: it means try again, not give up.
 *
 * Usage: npx tsx --env-file=.env scripts/model-audit.ts [openrouter|mistral]
 */

type Verdict = {
  model: string;
  status: "tools" | "chat-only" | "rate-limited" | "failed";
  ms: number;
  note: string;
};

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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function probe(
  baseURL: string,
  apiKey: string,
  model: string,
  headers: Record<string, string>,
  attempt = 1,
): Promise<Verdict> {
  const started = Date.now();
  try {
    const res = await fetch(`${baseURL}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", ...headers },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "What is the weather in Paris? Use the tool." }],
        tools: [TOOL],
        max_tokens: 300,
      }),
      signal: AbortSignal.timeout(75_000),
    });

    const ms = Date.now() - started;
    const body = await res.text();

    if (res.status === 429) {
      if (attempt === 1) {
        await sleep(20_000);
        return probe(baseURL, apiKey, model, headers, 2);
      }
      return { model, status: "rate-limited", ms, note: "429 after retry" };
    }

    if (!res.ok) {
      let msg = body.slice(0, 80);
      try {
        const j = JSON.parse(body);
        msg = String(j.error?.message ?? j.message ?? msg).slice(0, 80);
      } catch {
        /* non-JSON error body; the raw prefix is the best available detail */
      }
      return { model, status: "failed", ms, note: `HTTP ${res.status} ${msg}` };
    }

    const json = JSON.parse(body);
    if (json.error) {
      return { model, status: "failed", ms, note: String(json.error.message).slice(0, 80) };
    }

    const msg = json.choices?.[0]?.message;
    if (Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0) {
      return { model, status: "tools", ms, note: "" };
    }
    return {
      model,
      status: "chat-only",
      ms,
      note: "ignored the tools array — unusable for the agent loop",
    };
  } catch (error) {
    return {
      model,
      status: "failed",
      ms: Date.now() - started,
      note: (error as Error).message.slice(0, 80),
    };
  }
}

async function listOpenRouter(apiKey: string): Promise<string[]> {
  const res = await fetch("https://openrouter.ai/api/v1/models", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  const json = await res.json();
  // Only the zero-cost models: the paid catalogue is 400+ entries and its
  // behaviour is a billing question, not a compatibility one.
  return json.data
    .filter(
      (m: { pricing?: { prompt?: string; completion?: string } }) =>
        Number(m.pricing?.prompt ?? 0) === 0 && Number(m.pricing?.completion ?? 0) === 0,
    )
    .map((m: { id: string }) => m.id)
    .sort();
}

async function listMistral(apiKey: string): Promise<string[]> {
  const res = await fetch("https://api.mistral.ai/v1/models", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  const json = await res.json();
  return json.data
    .filter(
      (m: { capabilities?: { completion_chat?: boolean; function_calling?: boolean } }) =>
        m.capabilities?.completion_chat && m.capabilities?.function_calling,
    )
    .map((m: { id: string }) => m.id)
    .sort();
}

function report(title: string, verdicts: Verdict[]) {
  const order = { tools: 0, "chat-only": 1, "rate-limited": 2, failed: 3 } as const;
  verdicts.sort((a, b) => order[a.status] - order[b.status] || a.ms - b.ms);

  console.log(`\n=== ${title} ===`);
  for (const v of verdicts) {
    const mark =
      v.status === "tools" ? "OK  " : v.status === "chat-only" ? "PART" : "FAIL";
    console.log(
      `  ${mark} ${v.model.padEnd(48)} ${String(v.ms).padStart(6)}ms  ${v.status}${v.note ? " — " + v.note : ""}`,
    );
  }
  const usable = verdicts.filter((v) => v.status === "tools");
  console.log(`  -> ${usable.length}/${verdicts.length} usable by the agent (tool calling works)`);
}

async function main() {
  const which = process.argv[2] ?? "all";

  if (which === "all" || which === "openrouter") {
    const key = process.env.OPENROUTER_API_KEY ?? "";
    if (!key) console.log("openrouter: no key");
    else {
      const models = await listOpenRouter(key);
      console.log(`openrouter: probing ${models.length} zero-cost models…`);
      const verdicts: Verdict[] = [];
      for (const m of models) {
        verdicts.push(
          await probe("https://openrouter.ai/api/v1", key, m, {
            "HTTP-Referer": process.env.OPENROUTER_SITE_URL ?? "http://localhost:3000",
            "X-Title": process.env.OPENROUTER_SITE_NAME ?? "Agentic RAG",
          }),
        );
        await sleep(1_200);
      }
      report("OPENROUTER — zero-cost models", verdicts);
    }
  }

  if (which === "all" || which === "mistral") {
    const key = process.env.MISTRAL_API_KEY ?? "";
    if (!key) console.log("mistral: no key");
    else {
      const models = await listMistral(key);
      console.log(`\nmistral: probing ${models.length} tool-capable models…`);
      const verdicts: Verdict[] = [];
      for (const m of models) {
        verdicts.push(await probe("https://api.mistral.ai/v1", key, m, {}));
        await sleep(1_500);
      }
      report("MISTRAL — models advertising tool support", verdicts);
    }
  }
}

void main();

// Marks the file as a module, so its top-level declarations stay file-scoped
// rather than colliding with the sibling probe script's.
export {};
