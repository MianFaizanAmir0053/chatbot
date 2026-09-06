/**
 * Verify every configured credential individually.
 *
 * A pool only helps if its members work. A dead key in a failover list is worse
 * than absent: it consumes a hop, adds latency, and its failure looks the same
 * as a rate limit — so it is retried again on the next request rather than
 * skipped. This reports each key's real status without printing the key.
 *
 * Usage: npx tsx --env-file=.env scripts/key-audit.ts
 */
import { COHERE_API_KEYS, LLM_PROVIDERS, splitKeys } from "../lib/config";

const mask = (k: string) => `${k.slice(0, 6)}…${k.slice(-4)}`;

async function checkCohere(key: string): Promise<string> {
  const started = Date.now();
  try {
    const res = await fetch("https://api.cohere.com/v2/embed", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "embed-v4.0",
        texts: ["health probe"],
        input_type: "search_document",
        embedding_types: ["float"],
        output_dimension: 1536,
      }),
      signal: AbortSignal.timeout(45_000),
    });
    const json = await res.json();
    if (!res.ok || json.message) {
      return `FAIL ${res.status} ${String(json.message ?? "").slice(0, 60)}`;
    }
    return `OK   ${Date.now() - started}ms`;
  } catch (error) {
    return `FAIL ${(error as Error).message.slice(0, 60)}`;
  }
}

/**
 * Statuses that say "try again", not "this key is bad".
 *
 * 429 is a rate limit and 503 is upstream overload; both are properties of the
 * moment rather than the credential. Reporting either as a failure would
 * condemn a working key and, worse, invite removing it from a pool that is
 * there precisely to ride out these conditions.
 */
const TRANSIENT = new Set([429, 503, 502, 504]);

async function checkChat(
  baseURL: string,
  key: string,
  model: string,
  attempt = 1,
): Promise<string> {
  const started = Date.now();
  try {
    const res = await fetch(`${baseURL}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "Reply with exactly: ok" }],
        max_tokens: 400,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    const text = await res.text();
    if (!res.ok) {
      if (TRANSIENT.has(res.status) && attempt === 1) {
        await new Promise((r) => setTimeout(r, 15_000));
        return checkChat(baseURL, key, model, 2);
      }

      let detail = text.slice(0, 60);
      try {
        const j = JSON.parse(text);
        detail = String(j.error?.message ?? j.message ?? detail).slice(0, 60);
      } catch {
        /* non-JSON body; the raw prefix is the best detail available */
      }
      // Separate "busy" from "broken": the first needs patience, the second a
      // new credential, and only the second should ever prompt removing a key.
      const verdict = TRANSIENT.has(res.status) ? "BUSY" : "FAIL";
      return `${verdict} ${res.status} ${detail.replace(/\s+/g, " ")}`;
    }
    return `OK   ${Date.now() - started}ms`;
  } catch (error) {
    return `FAIL ${(error as Error).message.slice(0, 60)}`;
  }
}

async function main() {
  console.log("=== chat providers (one entry per key) ===");
  let chatOk = 0;
  let chatBusy = 0;
  for (const [i, p] of LLM_PROVIDERS.entries()) {
    const status = await checkChat(p.baseURL, p.apiKey, p.pro);
    if (status.startsWith("OK")) chatOk++;
    else if (status.startsWith("BUSY")) chatBusy++;
    console.log(`  ${String(i + 1).padStart(2)}. ${p.name.padEnd(12)} ${mask(p.apiKey)}  ${status}`);
  }
  console.log(
    `  -> ${chatOk}/${LLM_PROVIDERS.length} usable` +
      (chatBusy > 0 ? `, ${chatBusy} temporarily busy (not a key problem)` : "") +
      "\n",
  );

  console.log("=== cohere (embeddings + rerank) ===");
  let cohereOk = 0;
  for (const [i, key] of COHERE_API_KEYS.entries()) {
    const status = await checkCohere(key);
    if (status.startsWith("OK")) cohereOk++;
    console.log(`  ${String(i + 1).padStart(2)}. ${mask(key)}  ${status}`);
  }
  console.log(`  -> ${cohereOk}/${COHERE_API_KEYS.length} usable`);

  // Keys configured but filtered out of the active chain, so a dead one there
  // is not mistaken for redundancy that does not exist.
  const active = new Set(LLM_PROVIDERS.map((p) => p.apiKey));
  const idle = splitKeys(process.env.GROQ_API_KEY).filter((k) => !active.has(k));
  if (idle.length > 0) {
    console.log(`\nnote: ${idle.length} groq key(s) configured but not in the active chain`);
  }
}

void main();

export {};
