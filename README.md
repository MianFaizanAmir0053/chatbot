# Agentic RAG Chatbot

A document-grounded question answering system built as an **autonomous agent** rather than a fixed
pipeline. It plans multi-step questions, retrieves from your documents and the web inside its
reasoning loop, verifies its own answers against the sources, and cites every claim.

---

## Architecture

```
┌─ POST /api/chat ─────────────────────────────────────────────────────────┐
│                                                                          │
│  INPUT GUARDRAILS      zod schema · rate limit · prompt-injection        │
│                        detection (pattern → LLM adjudication)            │
│         ↓                                                                │
│  INGESTION             S3 → extract → section-aware chunking →           │
│  (if files attached)   contextual prefixing → embed → Qdrant             │
│         ↓                                                                │
│  AGENT LOOP            createAgent (LangChain v1) over DeepSeek V4        │
│    ├── plans           write_todos for genuinely multi-step questions     │
│    ├── acts            search_documents · web_search · fetch_url ·        │
│    │                   list_documents · calculator                        │
│    ├── delegates       delegate_research → document-researcher ·          │
│    │  (Deep Agents)    web-researcher · verifier, concurrently             │
│    └── iterates        re-search on weak results, escalate to web         │
│         ↓                                                                │
│  MIDDLEWARE HARNESS    budget ceilings · retries · provider fallback ·    │
│                        PII redaction · moderation · context editing ·     │
│                        summarisation · telemetry                          │
│         ↓                                                                │
│  OUTPUT GUARDRAILS     citation validation · LLM-as-judge groundedness    │
│         ↓                                                                │
│  SSE STREAM            tokens · plan · tool calls · sources · verdict     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Retrieval pipeline

Each `search_documents` call runs the full advanced-RAG path:

| Stage | Technique | Why |
|---|---|---|
| 1. Query planning | Multi-query + keyword extraction + **HyDE** | Paraphrases beat vocabulary mismatch; keywords preserve rare literal tokens; HyDE embeds a hypothetical *answer*, which sits closer to real answer passages than the question does |
| 2. Dense retrieval | Cohere `embed-v4.0` → Qdrant cosine | Semantic matching |
| 3. Sparse retrieval | BM25 over the full corpus | Exact matching on part numbers, error codes, proper nouns |
| 4. Fusion | **Reciprocal Rank Fusion** (k=60) | Combines ranked lists whose scores aren't comparable — cosine and BM25 live on different scales, so only rank position is used |
| 5. Reranking | Cohere `rerank-v4.0-pro` cross-encoder | Reads query+document together. Far more accurate than bi-encoder scoring, too slow for the whole corpus — so retrieve ~50 cheaply, rerank those |
| 6. Adaptive cutoff | Relative-to-best threshold | Rerank models don't share a score scale (v4.0-pro scores an irrelevant passage ~0.28 where v3.5 scores it ~0.03), so a fixed cutoff can't be right for both |

**Contextual chunking.** Chunks are split on detected headings, then each chunk is embedded with its
document title and section prepended. An isolated chunk reading *"it must be replaced every 10,000
km"* is nearly unretrievable; the same chunk prefixed with *"Motorcycle Manual > Drive Chain
Maintenance"* is not. The unprefixed text is kept in metadata so quotes stay verbatim.

### Guardrails

| Layer | Guardrail | Behaviour on failure |
|---|---|---|
| Input | Request schema (zod) | 400 |
| Input | Rate limit (20/min per IP) | 429-style refusal |
| Input | Prompt-injection detection | Pattern match → LLM adjudication → block only if confirmed |
| Model | PII detection (email, IP redacted; credit card blocked) | Redact or block |
| Model | OpenAI moderation (input, output, tool results) | End turn with a safe message |
| Loop | Tool-call and model-call ceilings | End gracefully with partial work |
| Loop | Model + tool retries with backoff, cross-provider fallback | Continue |
| Output | Citation validation | Strip unresolvable markers |
| Output | **LLM-as-judge groundedness** | Warn the user the answer may be unsupported |

Guardrails that depend on a model **fail open** — a broken judge must not block a good answer — and
the UI marks such answers unverified rather than implying they passed.

---

## Setup

### 1. Install

```bash
npm install
```

### 2. Configure

```bash
cp .env.example .env
```

**Required:**

- `DEEPSEEK_API_KEY` — the reasoning model ([platform.deepseek.com](https://platform.deepseek.com))
- `COHERE_API_KEY` — embeddings and reranking ([dashboard.cohere.com](https://dashboard.cohere.com))

**Strongly recommended:**

- `QDRANT_URL` — without it, documents live in memory and vanish on restart
- `AWS_S3_BUCKET_NAME` + credentials — document storage

**Optional:** `OPENAI_API_KEY` (fallback provider + moderation), `TAVILY_API_KEY` (best-in-class web
search), `LANGSMITH_API_KEY` (tracing).

### 3. Start Qdrant

```bash
npm run qdrant:up
```

Or use [Qdrant Cloud](https://cloud.qdrant.io) (free 1GB tier) and set `QDRANT_URL` + `QDRANT_API_KEY`.

### 4. Run

```bash
npm run dev
```

Then check [http://localhost:3000/api/health](http://localhost:3000/api/health) — it reports what is
actually *reachable*, not merely configured, so silent degradation is visible.

---

## Model tiering

The agent fans out into many cheap calls plus a few expensive ones, so they are routed to different
tiers. This is where most of the cost saving lives.

| Tier | Model | Used for |
|---|---|---|
| `PRO` | `deepseek-v4-pro` | Supervisor, planning, final synthesis |
| `FAST` | `deepseek-v4-flash` | Query planning, graders, rewriters, guardrail judges |

DeepSeek V4 gives a 1M context window, tool calling, and automatic context caching — cache hits are
roughly 30x cheaper than misses, which matters a lot when large retrieved contexts repeat across a
conversation.

If `DEEPSEEK_API_KEY` is absent or the API errors mid-run, `modelFallbackMiddleware` transparently
falls back to OpenAI.

### Provider constraints worth knowing

**Cohere cannot drive the agent loop.** `createAgent` emits structured content blocks and
`@langchain/cohere` rejects them (`ChatCohere does not support non text message content`) —
reproducibly, even with zero middleware. It also lacks `tool_choice` and has a broken
`withStructuredOutput`. So an agent-capable provider (**DeepSeek or OpenAI**) is required;
`/api/health` reports `agentCapable: false` when neither is present.

Cohere *is* used for embeddings, reranking, and the auxiliary single-shot calls (query planning,
guardrail judges), which use plain messages and work fine.

**Auxiliary calls walk a provider chain.** A configured-but-dead key is otherwise worse than no key:
it wins provider selection and then fails every call, silently disabling the guardrails behind it.
Auxiliary calls try DeepSeek → OpenAI → Cohere until one answers.

**Structured output is provider-agnostic.** `lib/structured.ts` tries `withStructuredOutput` and
falls back to a forced tool call — and then to an unforced one where `tool_choice` is unsupported.

---

## Scripts

```bash
npm run dev           # development server
npm run build         # production build
npm run typecheck     # tsc --noEmit
npm run lint          # eslint
npm run verify        # typecheck + lint + smoke + wiring
npm run smoke         # retrieval & guardrail pipeline, real API calls
npm run test:wiring   # agent graph, middleware and tools (no model quota needed)
npm run test:agent    # live agent behaviour (needs DEEPSEEK_API_KEY or OPENAI_API_KEY)
npm run test:agents   # Deep Agents wiring: citations, prompt, tools (no quota needed)
npm run test:agents:live  # + a real delegating turn and a concurrency measurement
npm run qdrant:up     # start Qdrant in Docker
npm run qdrant:down   # remove the Qdrant container
```

- **`smoke`** exercises chunking, embedding, vector storage, hybrid retrieval, RRF fusion,
  reranking, injection detection and the groundedness judge against real APIs.
- **`test:wiring`** drives the real agent graph with a scripted fake model, so it verifies *our*
  wiring — middleware composition, tool execution, SSRF and calculator guards, planning state, and
  the stream events the SSE route depends on — without consuming model quota.
- **`test:persistence`** queries in a cold process without indexing anything, proving documents
  genuinely survive in Qdrant rather than living in process memory.
- **`test:agent`** asks the live agent real questions and asserts it retrieves, decomposes
  multi-step questions, and refuses rather than fabricating when information is absent.
- **`test:agents`** covers delegated research. Its checks target failures that are silent rather
  than loud: citation markers that resolve to the wrong passage, a system prompt that instructs the
  supervisor to search *and* to delegate, retrieval tools left bound to a supervisor that is
  supposed to delegate, and — with `--live` — a fan-out that has quietly become sequential.

## Troubleshooting

`/api/health` reports what is *reachable*, not merely configured, and explains failures:

| Symptom | Meaning |
|---|---|
| `vectorStore.persistent: false` | `QDRANT_URL` unset or unreachable — documents vanish on restart |
| `storage.error: InvalidAccessKeyId` | The AWS access key doesn't exist. Generate a new one for the IAM user |
| `storage.error: SignatureDoesNotMatch` | `AWS_SECRET_ACCESS_KEY` doesn't match the key ID |
| `storage.error: PermanentRedirect` | The bucket is in a different region — check `AWS_REGION` |
| `models.agentCapable: false` | No DeepSeek or OpenAI key. Cohere alone can't run the agent |
| `status: degraded` | Queries work but something is broken — read the individual checks |

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Agentic chat. SSE stream of tokens, plan, tool calls, sources, groundedness |
| `/api/upload` | POST | Upload a document to S3 and index it immediately |
| `/api/documents` | GET / DELETE | List or remove indexed documents |
| `/api/health` | GET | Reachability of every subsystem |
| `/api/sample-pdf` | GET | Presigned URL for the most recent PDF in the bucket |

### `/api/chat` SSE events

| Event | Payload |
|---|---|
| `thread` | `{ threadId }` — pass back to continue the conversation |
| `status` | `{ stage }` — ingesting / thinking / verifying |
| `todos` | `{ todos }` — the live plan, as the agent writes and revises it |
| `tool_call` | `{ name, args }` |
| `delegation` | `{ id, agent, task }` — one research branch of a batch started |
| `delegation_result` | `{ batchId, chars }` — every branch in that batch reported |
| `token` | `{ text }` — streamed answer text |
| `sources` | `{ documents, web, searches }` |
| `groundedness` | `{ score, verdict, passed, unsupportedClaims }` |
| `warning` / `blocked` / `error` / `done` | — |

Supported uploads: **PDF, DOCX, TXT, MD, CSV** (25MB). Legacy binary `.doc` is rejected with a clear
message rather than being accepted and silently producing garbage.

---

## Project layout

```
lib/
  config.ts              validated env, model tiers, tuning constants
  models.ts              tiered model factory + provider fallback
  s3.ts                  document storage
  vectorstore/           driver interface, Qdrant, in-memory fallback
  ingest/                extract → chunk → embed → upsert
  retrieval/             hybrid search, RRF, reranking, query planning
  agents/                agent, tools, middleware harness, web search
  guardrails/            input and output guardrails
app/api/                 chat · upload · documents · health · sample-pdf
scripts/smoke.ts         end-to-end pipeline test
```

Swapping the vector database means adding one file under `lib/vectorstore/` — retrieval is written
against the driver interface, not against Qdrant.
