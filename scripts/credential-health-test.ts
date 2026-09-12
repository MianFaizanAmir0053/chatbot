/**
 * Does a refused credential leave the rotation — and does a busy one stay?
 *
 * Both halves matter, and the second more than the first. Taking a key out of
 * the pool is how this earns its keep, but the pool exists precisely to ride out
 * rate limits and outages, so a classifier that condemns a key on a 429 or a
 * connection reset would dismantle the thing it is meant to protect. Every
 * "permanent" case here is therefore paired with the near-miss that must not be
 * treated the same way.
 *
 * Usage: npx tsx --env-file=.env scripts/credential-health-test.ts
 */

import {
  credentialId,
  isPermanentRefusal,
  isRefused,
  isWorthRetrying,
  noteFailure,
  refusedCredentials,
  resetCredentialHealth,
  usable,
} from "../lib/credential-health";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(70)}\n${s}\n${"=".repeat(70)}`);
}

/** An SDK error as the OpenAI client actually throws it. */
function apiError(status: number, message: string) {
  return Object.assign(new Error(`${status} ${message}`), { status });
}

/** The same failure as reported by a gateway that only puts the code in the text. */
function textOnlyError(status: number, message: string) {
  return new Error(`${status} ${message}`);
}

function classification() {
  banner("1. What counts as refused");

  // The case this exists for: Groq's account restriction. A 400, which
  // @langchain/openai leaves unstamped, so the retry middleware would otherwise
  // back off through four attempts before the fallback chain is reached.
  check(
    "an account restriction is permanent",
    isPermanentRefusal(apiError(400, "Organization has been restricted. Please reach out to support")),
    "the reported case",
  );
  check(
    "even when the status is only in the message",
    isPermanentRefusal(textOnlyError(400, "Organization has been restricted.")),
    "gateways vary in how they report it",
  );
  check("a bad key is permanent", isPermanentRefusal(apiError(401, "Invalid Authentication")));
  check("so is a forbidden account", isPermanentRefusal(apiError(403, "Forbidden")));
  // But not every 403 is the account. Alibaba's MaaS free tier reports a
  // per-model token budget this way, and it clears: measured against a live
  // key, one heavily-used model refused `max_tokens` above ~128 while the same
  // key served every other model at 2048, and the refusing model recovered.
  check(
    "a free-tier throttle reported as 403 is not",
    !isPermanentRefusal(
      apiError(
        403,
        'Free quota exhausted. To continue accessing the model on a paid basis, please add funds or disable the "use free tier only" mode in the management console.',
      ),
    ),
    "measured against a live Alibaba MaaS key",
  );
  check(
    "unless that throttle states a daily window",
    isPermanentRefusal(apiError(403, "Free quota exhausted: daily limit reached")),
  );
  check(
    "and a revoked key",
    isPermanentRefusal(apiError(400, "The api key is invalid_api_key or revoked")),
  );

  // The other half: everything the pool exists to survive.
  // A daily ceiling cannot clear inside a request, so retrying it only spends
  // the branch's budget on a limit with hours left to run.
  check(
    "a daily limit is treated as unrecoverable for now",
    isPermanentRefusal(apiError(429, "Daily token limit exceeded. Troubleshooting URL: ...")),
    "observed in a trace",
  );
  check(
    "and a per-day request cap likewise",
    isPermanentRefusal(apiError(429, "Rate limit: 200 requests per day exceeded")),
  );

  // An exhausted balance. The clearest refusal there is: the credential is
  // valid, authenticates, and lists its models — only a completion reveals it.
  check(
    "an exhausted balance is permanent",
    isPermanentRefusal(apiError(402, "Insufficient Balance")),
    "402, observed on a live DeepSeek key",
  );
  check(
    "and so is the same thing reported as a 400",
    isPermanentRefusal(apiError(400, "Insufficient Balance")),
    "gateways differ on the status",
  );
  check(
    "as is a billing message",
    isPermanentRefusal(apiError(400, "Your account has a billing problem")),
  );

  // A monthly ceiling outlasts any process. Cohere trial keys are capped at
  // 1000 calls a month and say so, and that key was being re-tried by every
  // concurrent query variant of every search.
  check(
    "a monthly quota is unrecoverable too",
    isPermanentRefusal(
      apiError(429, "You are using a Trial key, which is limited to 1000 API calls / month."),
    ),
    "live Cohere wording — note the spaces around the slash",
  );

  check("a plain rate limit is not", !isPermanentRefusal(apiError(429, "Rate limit reached")), "429");
  // The exact message this deployment sees most. It names a window of minutes,
  // which is precisely what the backoff exists to wait out — and it contains a
  // number followed by "m", so a looser pattern could read it as a month.
  check(
    "nor a per-minute limit that states its wait",
    !isPermanentRefusal(apiError(429, "Rate limit reached. Please try again in 22m16.176s")),
    "minutes, not months",
  );
  // Deliberately not matched: several gateways send this for a per-minute
  // window that clears in seconds, and the backoff is there to wait it out.
  check(
    "nor an unqualified quota message",
    !isPermanentRefusal(apiError(429, "You exceeded your current quota, please check your plan")),
    "window unstated — assume short",
  );
  check("nor an outage", !isPermanentRefusal(apiError(503, "Service Unavailable")));
  check("nor a gateway error", !isPermanentRefusal(apiError(502, "Bad Gateway")));
  check(
    "nor a connect timeout",
    !isPermanentRefusal(Object.assign(new Error("fetch failed"), { code: "UND_ERR_CONNECT_TIMEOUT" })),
    "transport, not credential",
  );
  // A 400 is the ordinary way to report a malformed request, and that says
  // nothing about the key. Condemning on the status alone would take working
  // credentials out of the pool whenever a prompt tripped a provider's schema.
  check(
    "and an ordinary bad request is not",
    !isPermanentRefusal(apiError(400, "messages: array too long")),
    "the dangerous false positive",
  );
  check("nor a model-not-found", !isPermanentRefusal(apiError(404, "model not found")), "404");
}

function retrying() {
  banner("2. Is another attempt at the same model worth the wait?");

  // A different question from "should this credential leave the rotation", and
  // the gap between them is where the time goes.
  const freeTier = apiError(
    403,
    'Free quota exhausted. To continue accessing the model on a paid basis, please add funds or disable the "use free tier only" mode.',
  );
  check(
    "a drained per-model bucket keeps its credential",
    !isPermanentRefusal(freeTier),
    "other models on the key still work",
  );
  check(
    "but is not retried against the same model",
    !isWorthRetrying(freeTier),
    "measured: 33.2s across four attempts, then failed anyway",
  );

  // Everything the backoff exists for must still be retried.
  check(
    "a per-minute rate limit is retried",
    isWorthRetrying(apiError(429, "Rate limit reached. Please try again in 22m16.176s")),
  );
  check("an outage is retried", isWorthRetrying(apiError(503, "Service Unavailable")));
  check(
    "a transport failure is retried",
    isWorthRetrying(Object.assign(new Error("fetch failed"), { code: "UND_ERR_CONNECT_TIMEOUT" })),
  );

  // And nothing already condemned should be retried as well as dropped.
  check("a restricted account is not retried", !isWorthRetrying(apiError(400, "Organization has been restricted")));
  check("nor an exhausted balance", !isWorthRetrying(apiError(402, "Insufficient Balance")));
}

function rotation() {
  banner("3. A refused credential leaves the pool");
  resetCredentialHealth();

  const keys = ["k1", "k2", "k3"];
  const endpoint = "https://api.example.com/v1";
  const idOf = (k: string) => credentialId(endpoint, k);

  check("all keys start usable", usable(keys, idOf).length === 3, `${usable(keys, idOf).length}/3`);

  const noted = noteFailure(idOf("k2"), apiError(400, "Organization has been restricted"));
  check("the refusal is recorded", noted, "k2 marked");
  check("the key is out of the pool", !usable(keys, idOf).includes("k2"), usable(keys, idOf).join(","));
  check("the others are untouched", usable(keys, idOf).length === 2, usable(keys, idOf).join(","));

  // A rate limit must leave the pool exactly as it was.
  noteFailure(idOf("k1"), apiError(429, "Rate limit reached"));
  check("a busy key stays in", usable(keys, idOf).includes("k1"), "429 is not a refusal");

  // The safety valve: a pool that has lost every member is worse than useless,
  // because the caller then gets a configuration error instead of the
  // provider's own — which is the message that says what to actually fix.
  for (const k of keys) noteFailure(idOf(k), apiError(401, "Invalid Authentication"));
  check(
    "but a fully refused pool is returned whole",
    usable(keys, idOf).length === 3,
    "never hand back an empty pool",
  );

  check(
    "health reports the refusals without the keys",
    refusedCredentials().length === 3 &&
      refusedCredentials().every((r) => r.endpoint === endpoint && !JSON.stringify(r).includes("k1")),
    JSON.stringify(refusedCredentials()[0] ?? {}),
  );

  resetCredentialHealth();
  check("and a reset clears them", !isRefused(idOf("k2")) && refusedCredentials().length === 0);
}

function main() {
  classification();
  retrying();
  rotation();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
