/**
 * Prove the key rotation actually cycles, and that it changes only the key.
 *
 * The failure this guards against is subtle: a rotation that quietly returns
 * the same credential every time looks identical in behaviour to a working one
 * until a quota runs out, at which point everything stops at once. So assert
 * the distribution rather than trusting it.
 *
 * Usage: npx tsx --env-file=.env scripts/rotation-proof.ts
 */
import { LLM_PROVIDERS, env } from "../lib/config";
import { getModel, primaryKeyCount } from "../lib/models";

const mask = (k: string) => (k ? `${k.slice(0, 6)}…${k.slice(-4)}` : "(none)");

function keyOf(model: unknown): string {
  // The client stores the credential it was constructed with; reading it back
  // is the only way to see which key a given call would actually use.
  const m = model as { apiKey?: string; lc_kwargs?: { apiKey?: string } };
  return m.apiKey ?? m.lc_kwargs?.apiKey ?? "";
}

function modelOf(model: unknown): string {
  const m = model as { model?: string; modelName?: string };
  return m.model ?? m.modelName ?? "?";
}

async function main() {
  const pool = primaryKeyCount();
  console.log(`primary endpoint : ${env.DEEPSEEK_BASE_URL}`);
  console.log(`keys in rotation : ${pool}`);
  console.log(`chain entries    : ${LLM_PROVIDERS.length}\n`);

  const used: string[] = [];
  const models = new Set<string>();

  for (let i = 0; i < pool * 2; i++) {
    const m = getModel("pro");
    used.push(keyOf(m));
    models.add(modelOf(m));
  }

  console.log("consecutive calls:");
  used.forEach((k, i) => console.log(`  ${String(i + 1).padStart(2)}. ${mask(k)}`));

  const distinct = new Set(used).size;
  console.log(`\ndistinct keys used : ${distinct}/${pool}`);
  console.log(`distinct models    : ${models.size} (${[...models].join(", ")})`);

  // Two calls apart by the pool size should land on the same key again.
  const cycles = pool > 1 && used[0] === used[pool] && used[1] === used[pool + 1];
  console.log(`cycles correctly   : ${pool > 1 ? cycles : "n/a (single key)"}`);
  console.log(
    `quality invariant  : ${models.size === 1 ? "yes — only the key changes" : "NO — model varies"}`,
  );

  if (pool > 1 && (distinct !== pool || !cycles || models.size !== 1)) process.exitCode = 1;
}

void main();

export {};
