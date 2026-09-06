import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { ThinkingMode } from "../config";
import { knowledgeBaseStatus } from "../ingest/pipeline";
import { formatContext, retrieve } from "../retrieval/pipeline";
import type { RankedDocument } from "../retrieval/rerank";
import { webSearch } from "./websearch";

/**
 * Tools available to the agent.
 *
 * Each tool returns text shaped for a model to read, and records structured
 * provenance in `collector` so the API layer can surface real citations without
 * having to re-parse the model's prose.
 */

export interface EvidenceCollector {
  documents: RankedDocument[];
  webResults: Array<{ title: string; url: string }>;
  searches: string[];
}

export function createEvidenceCollector(): EvidenceCollector {
  return { documents: [], webResults: [], searches: [] };
}

const NO_RESULTS =
  "No matching passages found in the knowledge base. " +
  "Try different wording, or use web_search if the question is not about the uploaded documents.";

/** Private ranges and loopback — blocked so a tool call cannot probe internal services. */
const INTERNAL_HOST = /^(localhost|127\.|0\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\.|::1$|\[::1\])/i;

export interface ToolOptions {
  /** Include the web tools. Off means the agent answers only from documents. */
  webSearch?: boolean;
  /** Retrieval depth for search_documents. */
  mode?: ThinkingMode;
}

export function buildTools(collector: EvidenceCollector, options: ToolOptions = {}) {
  const { webSearch: allowWeb = true, mode = "standard" } = options;
  /** Per-run record of tool calls already made, keyed by name and arguments. */
  const callLog = new Map<string, number>();

  /**
   * The primary RAG tool: hybrid retrieval + cross-encoder reranking.
   *
   * Exposed to the agent as an ordinary tool so retrieval happens *inside* the
   * reasoning loop — the agent can search, read what came back, refine, and
   * search again. That loop is what separates agentic RAG from a fixed pipeline.
   */
  const searchDocuments = tool(
    withRepeatGuard("search_documents", callLog, async ({ query, topK }: { query: string; topK?: number }) => {
      const result = await retrieve(query, { topK, mode });
      collector.searches.push(query);

      if (result.documents.length === 0) return NO_RESULTS;

      // Deduplicate against evidence already gathered during this run.
      const seen = new Set(
        collector.documents.map((d) => `${d.doc.metadata?.source}#${d.doc.metadata?.chunkIndex}`),
      );
      for (const d of result.documents) {
        const key = `${d.doc.metadata?.source}#${d.doc.metadata?.chunkIndex}`;
        if (!seen.has(key)) {
          collector.documents.push(d);
          seen.add(key);
        }
      }

      return (
        `Found ${result.documents.length} passages ` +
        `(top relevance ${result.topScore.toFixed(3)}):\n\n${formatContext(result.documents)}`
      );
    }),
    {
      name: "search_documents",
      description:
        "Search the uploaded document knowledge base using hybrid semantic + keyword retrieval " +
        "with cross-encoder reranking. Use this FIRST for any question that could relate to the " +
        "user's documents. Returns numbered excerpts with source, section and page. " +
        "Call it multiple times with different phrasings to cover distinct sub-questions.",
      schema: z.object({
        query: z.string().describe("A focused, self-contained search query"),
        topK: z.number().int().min(1).max(15).optional().describe("How many passages to return"),
      }),
    },
  );

  const listDocuments = tool(
    withRepeatGuard("list_documents", callLog, async () => {
      const status = await knowledgeBaseStatus();
      if (status.documents.length === 0) {
        return "The knowledge base is empty. No documents have been uploaded yet.";
      }
      return (
        `Knowledge base contains ${status.documents.length} document(s), ` +
        `${status.totalChunks} chunks total:\n` +
        status.documents.map((d) => `- ${d.source} (${d.chunks} chunks)`).join("\n")
      );
    }),
    {
      name: "list_documents",
      description:
        "List the documents currently in the knowledge base. Use this to check what is available " +
        "before searching, or when the user asks what documents you have.",
      schema: z.object({}),
    },
  );

  const searchWeb = tool(
    withRepeatGuard("web_search", callLog, async ({ query, maxResults }: { query: string; maxResults?: number }) => {
      const outcome = await webSearch(query, maxResults ?? 5);
      collector.searches.push(`web: ${query}`);

      if (outcome.results.length === 0) {
        return `Web search returned no results${outcome.error ? ` (${outcome.error})` : ""}.`;
      }

      for (const r of outcome.results) {
        if (r.url) collector.webResults.push({ title: r.title, url: r.url });
      }

      return (
        `Web results via ${outcome.provider}:\n\n` +
        outcome.results
          .map((r, i) => `[W${i + 1}] ${r.title}\n${r.url}\n${r.content.slice(0, 1200)}`)
          .join("\n\n---\n\n")
      );
    }),
    {
      name: "web_search",
      description:
        "Search the public web for current or general information. Use this when the uploaded " +
        "documents do not contain the answer, when the question concerns recent events, or when " +
        "external context is needed to interpret a document. Always prefer search_documents for " +
        "questions about the user's own files. Clearly attribute web-sourced facts as external.",
      schema: z.object({
        query: z.string().describe("The web search query"),
        maxResults: z.number().int().min(1).max(10).optional(),
      }),
    },
  );

  const fetchUrl = tool(
    withRepeatGuard("fetch_url", callLog, async ({ url }: { url: string }) => {
      let parsed: URL;
      try {
        parsed = new URL(url);
      } catch {
        return `Invalid URL: ${url}`;
      }

      // SSRF guard: refuse non-HTTP schemes and internal network targets.
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        return "Only http and https URLs can be fetched.";
      }
      if (INTERNAL_HOST.test(parsed.hostname)) {
        return "Refusing to fetch internal or loopback addresses.";
      }

      try {
        const res = await fetch(parsed.toString(), {
          headers: { "User-Agent": "Mozilla/5.0 (compatible; AgenticRAG/1.0)" },
          signal: AbortSignal.timeout(15000),
          redirect: "follow",
        });
        if (!res.ok) return `Fetch failed: HTTP ${res.status}`;

        const html = await res.text();
        const text = html
          .replace(/<script[\s\S]*?<\/script>/gi, " ")
          .replace(/<style[\s\S]*?<\/style>/gi, " ")
          .replace(/<[^>]+>/g, " ")
          .replace(/&nbsp;/g, " ")
          .replace(/\s+/g, " ")
          .trim();

        collector.webResults.push({ title: parsed.hostname, url: parsed.toString() });
        return text.slice(0, 6000) || "Page contained no readable text.";
      } catch (error) {
        return `Fetch failed: ${error instanceof Error ? error.message : String(error)}`;
      }
    }),
    {
      name: "fetch_url",
      description:
        "Fetch and extract the readable text of a public web page. Use after web_search when a " +
        "result looks promising but its snippet is too short to answer the question.",
      schema: z.object({ url: z.string().describe("Absolute http(s) URL to fetch") }),
    },
  );

  const calculator = tool(
    async ({ expression }) => {
      // Arithmetic only. The character allowlist means no identifiers can appear,
      // so there is no scope for the expression to reference.
      if (!/^[\d\s+\-*/().,%^eE]+$/.test(expression)) {
        return "Only arithmetic expressions are supported (digits and + - * / ( ) . % ^).";
      }
      try {
        const normalised = expression.replace(/\^/g, "**").replace(/,/g, "");
        const result = Function(`"use strict"; return (${normalised});`)();
        if (typeof result !== "number" || !Number.isFinite(result)) {
          return "Expression did not evaluate to a finite number.";
        }
        return `${expression} = ${result}`;
      } catch {
        return `Could not evaluate: ${expression}`;
      }
    },
    {
      name: "calculator",
      description:
        "Evaluate an arithmetic expression. Use this for any numeric computation rather than " +
        "doing mental arithmetic, which is error-prone.",
      schema: z.object({ expression: z.string().describe("e.g. (1250 * 0.08) + 340") }),
    },
  );

  // Omitting the web tools rather than refusing inside them: a tool the model
  // cannot see is one it cannot spend a turn discovering it may not use.
  return allowWeb
    ? [searchDocuments, listDocuments, searchWeb, fetchUrl, calculator]
    : [searchDocuments, listDocuments, calculator];
}

/* ------------------------------------------------------------------ *
 * Repeat-call suppression
 * ------------------------------------------------------------------ */

/**
 * Wrap a tool implementation so an identical repeat is answered, not re-run.
 *
 * Applied to the implementation rather than to the tool's `invoke`: patching
 * `invoke` looked equivalent and never fired once, because that is not the path
 * the agent runtime takes. The function passed to `tool()` is code we own and
 * is unambiguously on the call path.
 *
 * A stalled agent does not wander, it repeats — a measured run spent its whole
 * step budget alternating between two calls it had already made. Re-running
 * those costs an embedding, a rerank against the Cohere quota, and another full
 * passage payload through the context window, which is most of the token spend
 * on a stalled run.
 *
 * The reply is a short instruction rather than the cached payload: the model
 * already holds the passages from the first call, so resending them adds tokens
 * without adding information, and identical input is what kept producing
 * identical output.
 */
function withRepeatGuard<A>(
  name: string,
  seen: Map<string, number>,
  run: (args: A) => Promise<string>,
): (args: A) => Promise<string> {
  return async (args: A) => {
    const key = `${name}:${JSON.stringify(args ?? {})}`;
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);

    if (count > 1) {
      console.warn(`[tools] ${name} repeated (call ${count}) — answered without re-running`);
      return (
        `You already called ${name} with these exact arguments ${count - 1} time(s) in this ` +
        `turn and the result has not changed. Do not repeat it. Either use genuinely different ` +
        `wording, use a different tool, or answer now from what you have already retrieved — ` +
        `including saying plainly that the documents do not cover it.`
      );
    }

    return run(args);
  };
}

/**
 * Answer a repeated identical tool call from cache, and say so.
 *
 * Agents that stall do not wander — they repeat. A measured run issued six tool
 * calls of which only two were distinct: `list_documents` three times and one
 * `search_documents` three times, each returning exactly what it had returned
 * before, until the recursion limit ended the request with no answer at all.
 *
 * Re-running those is the expensive part. A repeated search costs an embedding
 * call, a rerank against the Cohere quota, and — worst of all — another full
 * passage payload back through the context window, which is most of the token
 * spend on a stalled run. Serving the repeat from cache removes that cost
 * entirely.
 *
 * The response is deliberately a short instruction rather than the cached
 * payload: the model already has the passages from the first call, so resending
 * them adds tokens without adding information, and the loop persists precisely
 * because identical input keeps producing identical output. Telling it plainly
 * that the call was repeated is the part that breaks the cycle.
 */
