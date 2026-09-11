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

  check("a plain rate limit is not", !isPermanentRefusal(apiError(429, "Rate limit reached")), "429");
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

function rotation() {
  banner("2. A refused credential leaves the pool");
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
  rotation();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
