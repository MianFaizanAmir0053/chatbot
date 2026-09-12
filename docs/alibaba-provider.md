# Alibaba Model Studio as the primary provider

Measured 2026-09-12 against the MaaS workspace endpoint
`https://ws-w7d45hqpi51jfix5.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`,
on the free tier of a single key. Everything below was observed, not inferred.

## What works

| Capability | Result |
| --- | --- |
| Auth, `/models` | 200, 171 models listed |
| Tool calling | Valid `tool_calls` with parseable arguments on every chat model tried |
| Parallel tool calls | 2–4 calls in one assistant turn |
| Tool-result round trip | Cites `[1]`/`[2]` correctly from injected excerpts |
| `response_format: json_schema` (strict) | Honoured; parsed cleanly |
| Streaming | 391 ms TTFT through `ChatDeepSeek.stream` |
| Embeddings | `text-embedding-v4`, `text-embedding-v3`, `qwen3.7-text-embedding` — 1024 dims, ~200 ms |
| Vision | `qwen3-vl-plus`, `qwen3-vl-235b-a22b-instruct`, `qwen-vl-ocr` all described a generated PNG correctly |
| Long context | 597k input tokens accepted; needle recalled at 5%, 50% and 95% depth |

Latency at depth: ~12 s at 119k input tokens, ~27 s at 597k.

## Agent-loop results

`scripts/alibaba-probe.mts` gives a model real `list_dir`/`read_file`/`grep_repo`
tools over this repo and asks a question that needs several dependent lookups,
then checks whether the cited files exist.

| Model | Wall | Model calls | Tool calls | Citations | Depth |
| --- | --- | --- | --- | --- | --- |
| `qwen3.8-max` | 45 s | 7 | 13 | 25, all files real | 5 files, all four topics |
| `qwen3-coder-plus` | 29 s | 16 | 15 | 14, all files real | 3 files |
| `kimi-k3` | — | — | — | 24, all files real | 3 files |

`qwen3.8-max` is the PRO tier on that basis. It found things the question did
not point at, including the retry middleware's refusal check and the fallback
chain's exclusion of the in-use key.

## Traps, and what was done about them

**A 403 does not mean the key is dead.** The free tier reports a *per-model*
token budget as `403 Free quota exhausted ... please add funds or disable the
"use free tier only" mode`, rejected in ~130 ms before any inference. It is
per-model and it recovers: one heavily-used model refused `max_tokens` above
~128 while the same key served every other model at 2048. `isPermanentRefusal`
treated every 403 as a dead credential, which would have pulled the key out of
the rotation for thirty minutes over a transient bucket. It now treats a
quota-worded 403 like a 429 — permanent only if the window is stated in days.
See `lib/credential-health.ts` and the cases in
`scripts/credential-health-test.ts`.

**The 403 is also raised by a large `max_tokens`, not just by usage.** The
endpoint appears to pre-reject when the requested completion exceeds what is
left in the bucket. Omitting `max_tokens` entirely was refused too, since the
default is large. Keep `maxTokens` set and modest.

**Thinking models starve small-budget calls.** Reasoning is not counted against
`max_tokens` the way a caller expects. `deepseek-v4-pro` asked for a one-word
relevance grade at `max_tokens: 16` spent all sixteen on reasoning and returned
**empty content** with `finish_reason: "length"` — a grader that returns nothing
fails open, and every passage it was meant to filter goes through.
`qwen3.5-plus` billed 312 output tokens against a 64-token cap. Hence
`DEEPSEEK_FAST_THINKING=false`, which sends `enable_thinking: false` on the FAST
tier only (PRO keeps its reasoning, which is what that tier is for). With the
switch off, the same graders answer in one token, three to six times faster.

**`kimi-k3` rejects `temperature`.** `400 Parameter 'temperature'=0.1 is not
supported for kimi-k3 model`, on the first call. `getModel` always sends a
temperature, so kimi-k3 cannot be a tier model without a carve-out. It is
otherwise strong — it was the only model to correctly name a rejected-promise
cache-poisoning bug that the others read as a race condition.

**`ZHIPU/GLM-5.3` is genuinely out of free quota**, unlike the transient case
above: it 403s on every call including `/models` streaming. Not usable until
funded.

## What the switch replaced

Worth recording, because it changes how to read the "before" state: the
previous `DEEPSEEK_API_KEY` was Groq key `gsk_rzptmY7…`, which is one of the
seven keys returning `400 Organization has been restricted`. `primaryKeyPool()`
matches every `LLM_PROVIDERS` entry sharing the primary base URL, so the primary
rotation was dealing all eight Groq keys — seven of them banned. Roughly seven
in eight primary model calls could not succeed, each paying a retry sequence
before the fallback chain was reached.

So the provider fleet was measured key by key the same day:

| Provider | Live keys | Tool calling | Failure on the rest |
| --- | --- | --- | --- |
| aionlabs | 5/7 | OK — 2 calls, valid args | `429 Daily token limit exceeded` (clears) |
| groq | 1/8 | OK | `400 Organization has been restricted` (does not clear) |
| gemini | 0/8 | untested | `429`, empty message |
| bluesminds | 0/1 | — | `503 No available channel for model` |

Acted on in `.env`: the seven banned Groq keys removed (kept commented),
`SUBAGENT_PROVIDER_ORDER` set to `aionlabs,groq` with gemini dropped. Gemini is
removed rather than demoted because a 429 with no text is classified temporary
on purpose — so those keys were never taken out of the rotation and kept being
dealt to branches, which then exhausted their retries. Ordering cannot fix that;
only absence can.

## An empty answer at the delegation ceiling

Found while verifying the switch, and **not** caused by it — the same failure is
recorded against the previous provider in `SUBAGENT_CONFIG.MAX_DELEGATION_ROUNDS`
("hit the cap and finished with an empty answer after 344 seconds"). Lowering
the cap from three rounds to two made it rarer without removing it.

`toolCallLimitMiddleware` blocks an over-limit `delegate_research` with a
generic error and no instruction, and the supervisor then ended the turn with no
answer at all while holding findings. `qwen3.8-max` reaches it more often than
the previous supervisor because it delegates incrementally — four branches, then
two, then a third attempt — rather than in one batch.

The ceiling is now enforced inside the tool, which refuses with
`DELEGATION LIMIT REACHED`, the number of findings already gathered, and an
instruction to answer from them. The middleware stays installed one round higher
as a backstop. Measured after the change: the same turn produced a 2090-character
answer where it had produced zero, and correctly reported *"the research failed,
not the documents"* — the distinction the design rests on. Three regression
checks cover it in `scripts/deep-agents-test.ts`, at no branch cost, since the
concurrency test has already spent both rounds by then.

## Reproducing

```bash
export AL_KEY=...            # the workspace key
export AL_BASE=https://ws-w7d45hqpi51jfix5.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
npx tsx scripts/alibaba-wiring-check.mts        # thinking switch + streaming
npx tsx scripts/alibaba-probe.mts qwen3.8-max   # full agent loop over this repo
```

## Still to do

The free tier is the binding constraint, not the models. Funding the account
removes the per-model buckets, the `max_tokens` ceiling, and the 403 retries
that currently cost wall-clock inside a tool loop. Until then a long run can
stall on a drained model even though the key is healthy.
