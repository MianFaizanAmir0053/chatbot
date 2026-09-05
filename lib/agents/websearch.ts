import { env, features } from "../config";

export interface WebResult {
  title: string;
  url: string;
  content: string;
}

export interface WebSearchOutcome {
  provider: "tavily" | "openai" | "duckduckgo" | "none";
  results: WebResult[];
  error?: string;
}

/**
 * Web search with provider fallback.
 *
 * Providers are tried best-first and each degrades to the next on failure, so
 * the agent keeps a working search tool regardless of which keys are present:
 *
 *   Tavily      — purpose-built for agents, returns clean extracted content
 *   OpenAI      — gpt-5-search-api, needs no extra signup beyond OPENAI_API_KEY
 *   DuckDuckGo  — no key at all; last resort, snippet-only
 */

async function searchTavily(query: string, maxResults: number): Promise<WebResult[]> {
  const { TavilySearch } = await import("@langchain/tavily");
  const tool = new TavilySearch({
    maxResults,
    topic: "general",
    tavilyApiKey: env.TAVILY_API_KEY,
  });
  const raw = await tool.invoke({ query });
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  const results = parsed?.results ?? [];
  return results.map((r: Record<string, unknown>) => ({
    title: String(r.title ?? ""),
    url: String(r.url ?? ""),
    content: String(r.content ?? r.raw_content ?? ""),
  }));
}

async function searchOpenAI(query: string, maxResults: number): Promise<WebResult[]> {
  // Must follow OPENAI_BASE_URL: when the key belongs to a gateway, posting it
  // to api.openai.com would hand that key to a third party.
  const base = (env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/+$/, "");
  const res = await fetch(`${base}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-5-search-api",
      input: query,
      tools: [{ type: "web_search" }],
    }),
  });

  if (!res.ok) throw new Error(`OpenAI search failed: ${res.status} ${await res.text()}`);
  const data = await res.json();

  // The Responses API returns content blocks with url_citation annotations;
  // those annotations are the actual sources behind the synthesised answer.
  const results: WebResult[] = [];
  let synthesis = "";
  for (const item of data.output ?? []) {
    for (const block of item.content ?? []) {
      if (block.type === "output_text") {
        synthesis += block.text ?? "";
        for (const ann of block.annotations ?? []) {
          if (ann.type === "url_citation" && ann.url) {
            results.push({
              title: String(ann.title ?? ann.url),
              url: String(ann.url),
              content: "",
            });
          }
        }
      }
    }
  }

  if (results.length === 0 && synthesis) {
    return [{ title: "Web search summary", url: "", content: synthesis }];
  }
  // Attach the synthesis to the first result so its content isn't discarded.
  if (results[0] && synthesis) results[0].content = synthesis;
  return results.slice(0, maxResults);
}

async function searchDuckDuckGo(query: string, maxResults: number): Promise<WebResult[]> {
  const ddg = await import("duck-duck-scrape");
  const res = await ddg.search(query, { safeSearch: ddg.SafeSearchType.MODERATE });
  return (res.results ?? []).slice(0, maxResults).map((r) => ({
    title: r.title,
    url: r.url,
    // DuckDuckGo returns HTML-formatted snippets.
    content: String(r.description ?? "").replace(/<[^>]*>/g, ""),
  }));
}

export async function webSearch(query: string, maxResults = 5): Promise<WebSearchOutcome> {
  const providers: Array<{
    name: WebSearchOutcome["provider"];
    enabled: boolean;
    run: () => Promise<WebResult[]>;
  }> = [
    { name: "tavily", enabled: features.tavily, run: () => searchTavily(query, maxResults) },
    { name: "openai", enabled: features.openai, run: () => searchOpenAI(query, maxResults) },
    { name: "duckduckgo", enabled: true, run: () => searchDuckDuckGo(query, maxResults) },
  ];

  const errors: string[] = [];
  for (const provider of providers) {
    if (!provider.enabled) continue;
    try {
      const results = await provider.run();
      if (results.length > 0) return { provider: provider.name, results };
      errors.push(`${provider.name}: no results`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[websearch] ${provider.name} failed:`, message);
      errors.push(`${provider.name}: ${message}`);
    }
  }

  return { provider: "none", results: [], error: errors.join("; ") };
}

/** Which provider would be used right now, for /api/health. */
export function activeSearchProvider(): string {
  if (features.tavily) return "tavily";
  if (features.openai) return "openai";
  return "duckduckgo";
}
