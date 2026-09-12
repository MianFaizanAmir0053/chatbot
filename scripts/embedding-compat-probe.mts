/**
 * Can two embedding providers share one vector store?
 *
 * The question behind "use Alibaba for embeddings too, in rotation". Rotation
 * is safe for chat and for reranking, because both read text and return a
 * judgement. Embeddings are different: a stored vector is only comparable with
 * a query vector from the *same model*. Two models can agree on dimensionality
 * and still place the same sentence in unrelated positions.
 *
 * If that is true here, rotating per request would silently destroy retrieval —
 * every search would compare a query in one space against documents in another
 * and return whatever happened to be closest to nothing in particular. No error,
 * no warning, just worse answers.
 *
 * So it is measured rather than assumed: embed a question and a matching
 * passage with each provider, then compare within a provider and across them.
 *
 * Usage: npx tsx --env-file=.env scripts/embedding-compat-probe.mts
 */

import { COHERE_API_KEYS, EMBEDDING_CONFIG, env } from "../lib/config";

const QUESTION = "What is the engine oil capacity of the motorcycle?";
const MATCH = "Engine oil capacity: 3.4 litres including the oil filter. Use SAE 10W-40.";
const UNRELATED = "Receipts under 10 GBP do not need to be itemised on an expense claim.";

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Walks the keys: the first is out of its monthly trial quota. */
async function cohere(texts: string[], inputType: string): Promise<number[][]> {
  let last = "no keys configured";
  for (const key of COHERE_API_KEYS) {
    const res = await fetch("https://api.cohere.com/v2/embed", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: EMBEDDING_CONFIG.model,
        texts,
        input_type: inputType,
        embedding_types: ["float"],
        output_dimension: EMBEDDING_CONFIG.dimensions,
      }),
      signal: AbortSignal.timeout(45_000),
    });
    const json = (await res.json()) as { embeddings?: { float?: number[][] }; message?: string };
    if (json.embeddings?.float) return json.embeddings.float;
    last = String(json.message ?? res.status).slice(0, 90);
  }
  throw new Error(`cohere: ${last}`);
}

async function alibaba(texts: string[], model: string, dimensions?: number): Promise<number[][]> {
  const res = await fetch(`${env.DEEPSEEK_BASE_URL.replace(/\/$/, "")}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.DEEPSEEK_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input: texts, ...(dimensions ? { dimensions } : {}) }),
    signal: AbortSignal.timeout(45_000),
  });
  const json = (await res.json()) as {
    data?: Array<{ embedding: number[] }>;
    error?: { message?: string };
  };
  if (!json.data) throw new Error(`alibaba ${model}: ${json.error?.message ?? res.status}`);
  return json.data.map((d) => d.embedding);
}

async function main() {
  console.log("Do Cohere and Alibaba embeddings live in the same space?\n");

  // What dimensionalities are even on offer.
  for (const dims of [undefined, 1024, 1536]) {
    try {
      const [v] = await alibaba(["probe"], "text-embedding-v4", dims);
      console.log(`  text-embedding-v4 dimensions=${dims ?? "(default)"} -> ${v.length}`);
    } catch (e) {
      console.log(`  text-embedding-v4 dimensions=${dims ?? "(default)"} -> ${(e as Error).message.slice(0, 80)}`);
    }
  }

  const [cq, cm, cu] = await cohere([QUESTION, MATCH, UNRELATED], "search_query");
  console.log(`\n  cohere ${EMBEDDING_CONFIG.model} -> ${cq.length} dims`);

  // Matched to Cohere's width where the provider allows it, so dimensionality
  // cannot be blamed for whatever the cross-provider numbers show.
  const width = cq.length;
  let aq: number[];
  let am: number[];
  let au: number[];
  try {
    [aq, am, au] = await alibaba([QUESTION, MATCH, UNRELATED], "text-embedding-v4", width);
  } catch {
    [aq, am, au] = await alibaba([QUESTION, MATCH, UNRELATED], "text-embedding-v4");
  }
  console.log(`  alibaba text-embedding-v4 -> ${aq.length} dims`);

  console.log("\n--- within one provider (this is what retrieval relies on) ---");
  console.log(`  cohere   question vs matching passage   ${cosine(cq, cm).toFixed(4)}`);
  console.log(`  cohere   question vs unrelated passage  ${cosine(cq, cu).toFixed(4)}`);
  console.log(`  alibaba  question vs matching passage   ${cosine(aq, am).toFixed(4)}`);
  console.log(`  alibaba  question vs unrelated passage  ${cosine(aq, au).toFixed(4)}`);

  if (aq.length === cq.length) {
    console.log("\n--- across providers (what rotation would actually do) ---");
    console.log(`  alibaba question vs cohere matching passage    ${cosine(aq, cm).toFixed(4)}`);
    console.log(`  alibaba question vs cohere unrelated passage   ${cosine(aq, cu).toFixed(4)}`);
    console.log(`  cohere  question vs alibaba matching passage   ${cosine(cq, am).toFixed(4)}`);

    const withinGap = cosine(cq, cm) - cosine(cq, cu);
    const crossGap = cosine(aq, cm) - cosine(aq, cu);
    console.log(
      `\n  signal (match minus unrelated): within cohere ${withinGap.toFixed(4)}, ` +
        `across providers ${crossGap.toFixed(4)}`,
    );
    console.log(
      crossGap < withinGap * 0.5
        ? "  -> the spaces are NOT interchangeable; rotating per request would break retrieval"
        : "  -> unexpectedly comparable; worth a closer look before concluding",
    );
  } else {
    console.log(
      `\n  Dimensions differ (${cq.length} vs ${aq.length}), so the vectors cannot even be ` +
        `compared, let alone stored in one collection.`,
    );
  }
}

void main();

export {};
