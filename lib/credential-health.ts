/* ------------------------------------------------------------------ *
 * Credential health
 *
 * Which configured credentials have been refused in a way that retrying cannot
 * fix, so the rotation can stop spending calls on them.
 *
 * A dead key in a failover pool is worse than an absent one. It still wins its
 * turn in the rotation, fails, and is tried again on the very next request —
 * and because a refusal looks much like a rate limit, nothing upstream can tell
 * the difference. With seven of eight keys refused, seven in eight calls opened
 * on a credential that could never answer.
 *
 * The cost is not one wasted round trip. `modelRetryMiddleware` retries unless
 * an error is explicitly stamped non-retryable, and `@langchain/openai` stamps
 * only 401, 404, and 400s about tool calls. A 400 saying the account itself is
 * refused is left unstamped, so it was retried four times over roughly thirty
 * seconds before the fallback chain was even reached.
 * ------------------------------------------------------------------ */

/**
 * How long a refusal is believed.
 *
 * Long enough that a refused key stops dominating the rotation, short enough
 * that fixing the account — paying a bill, lifting a restriction — recovers on
 * its own rather than needing a restart. The cost of the retry is a single
 * failed call per credential per window, which is the price of never
 * permanently condemning a key on one classification.
 */
const RECHECK_AFTER_MS = 30 * 60 * 1000;

/** What a credential's refusal looked like, for /api/health and the logs. */
interface Refusal {
  at: number;
  detail: string;
}

const refused = new Map<string, Refusal>();

/**
 * Identity of one credential at one endpoint.
 *
 * Endpoint *and* key, matching how `LLM_PROVIDERS` identifies an entry: the
 * same key at a different gateway is a different credential, and the same
 * gateway with a different key is the case this whole mechanism is about.
 */
export function credentialId(baseURL: string, apiKey: string): string {
  return `${baseURL}|${apiKey}`;
}

/** HTTP status from an SDK error, however the provider chose to report it. */
function statusOf(error: unknown): number | undefined {
  const e = error as { status?: unknown; response?: { status?: unknown }; message?: unknown };
  for (const candidate of [e?.status, e?.response?.status]) {
    if (typeof candidate === "number") return candidate;
  }
  // Several gateways surface the status only in the message text, which is
  // where "400 Organization has been restricted" arrives from.
  const match = /^\s*(\d{3})\b/.exec(typeof e?.message === "string" ? e.message : "");
  return match ? Number(match[1]) : undefined;
}

/**
 * Language that means the account, not the request.
 *
 * Matched only alongside a 400, since that status is otherwise the ordinary way
 * to report a malformed request — which says nothing about the credential and
 * must never take a working key out of the pool.
 */
const ACCOUNT_REFUSED =
  /\b(restricted|suspended|deactivated|disabled|banned|revoked|terminated|invalid api key|invalid_api_key|incorrect api key|no longer active|insufficient balance|insufficient funds|insufficient credit|payment required|billing)\b/i;

/**
 * True when retrying this error with this credential cannot succeed.
 *
 * Deliberately narrow. A false positive removes a working key from a pool that
 * exists to survive exactly this kind of trouble, so anything ambiguous —
 * every 429, every 5xx, every timeout or transport failure — is treated as
 * temporary. Those are properties of the moment; these are properties of the
 * account.
 */
/**
 * A rate limit measured in days or months rather than minutes.
 *
 * Matched narrowly and only on the word "daily" or an explicit "per day". Most
 * 429s are per-minute and are exactly what the backoff exists to ride out, so
 * the bar for treating one as unrecoverable has to be an explicit statement
 * that the window is long. "You exceeded your current quota" deliberately does
 * not match: several gateways send that for a per-minute limit that clears in
 * seconds.
 *
 * A month matters as much as a day here. Cohere trial keys are capped at 1000
 * calls per month and say so in the 429 — a ceiling that will not clear inside
 * any process lifetime, and one that was being re-tried by every concurrent
 * query variant of every search.
 */
const DAILY_LIMIT =
  /\b(daily|monthly)\b|\bper[\s-]*(day|month)\b|\/\s*(day|month)\b|\bin\s+a\s+(day|month)\b/i;

/**
 * Language that means the moment, not the account — on a status that usually
 * means the account.
 *
 * Alibaba's MaaS free tier reports a *per-model* token budget as
 * `403 Free quota exhausted. ... please add funds or disable the "use free tier
 * only" mode`, rejected in ~130ms before any inference. Measured against a live
 * key: the same key answered every other model at `max_tokens: 2048` while one
 * heavily-used model refused anything above ~128, and that model recovered on
 * its own. So the 403 describes a model's bucket at an instant, not a dead
 * credential — and condemning the key for thirty minutes over it costs every
 * other model the key can still serve.
 *
 * Only consulted alongside a 403, which is otherwise a genuine permission
 * failure and must stay permanent.
 */
const THROTTLE =
  /\b(quota|rate[ _-]?limit|too many requests|throttl\w*|concurrenc\w+|requests? per|tokens? per)\b/i;

export function isPermanentRefusal(error: unknown): boolean {
  const status = statusOf(error);
  const detail = String((error as { message?: unknown })?.message ?? "");
  if (status === 401) return true;
  // A 403 is normally the account: the key is real and not allowed to do this.
  // But a gateway that meters a free tier per model reports that as a 403 too,
  // and that one clears — so it is treated like a 429, permanent only if the
  // window is stated in days. See THROTTLE.
  if (status === 403) return THROTTLE.test(detail) ? DAILY_LIMIT.test(detail) : true;
  // 402 is the least ambiguous refusal there is: the credential is valid and
  // the account cannot pay. No retry within a request can change that, and no
  // backoff is short enough to outlast it. A live DeepSeek key returned 402
  // "Insufficient Balance" on every call while authenticating perfectly and
  // listing its models — so nothing short of the status distinguishes it from
  // a working key until the completion is attempted.
  if (status === 402) return true;
  // Some gateways report the same condition as a 400 with the reason in the
  // text rather than in the status.
  if (status === 400) return ACCOUNT_REFUSED.test(detail);
  // A daily ceiling will not clear inside a request. Retrying it four times
  // across thirty seconds of backoff was observed costing a research branch
  // its entire budget for a limit that had hours left to run — and because a
  // fan-out waits on its slowest branch, that was the whole turn.
  if (status === 429) return DAILY_LIMIT.test(detail);
  return false;
}

/**
 * Whether waiting and trying the same model again could plausibly work.
 *
 * Distinct from `isPermanentRefusal`, which asks a different question: whether
 * the *credential* should leave the rotation. Conflating the two costs real
 * time, because there is a third case that is neither.
 *
 * Alibaba's free tier meters per model. A drained bucket is rejected in ~130ms
 * with a 403 before any inference, and it does not refill within a request — so
 * the key is fine, other models on it are fine, and retrying *this* model is
 * guaranteed to fail. Measured on a live fan-out: one branch spent 33.2 seconds
 * on four attempts against a drained bucket and failed anyway, while its four
 * siblings finished in 3 to 15 seconds. The turn then reported rate limits
 * instead of an answer, because a fan-out waits on its slowest branch.
 *
 * Returning false here hands the call straight to the fallback chain, which is
 * a *different provider* and therefore a different bucket — the only thing that
 * can actually succeed.
 */
export function isWorthRetrying(error: unknown): boolean {
  if (isPermanentRefusal(error)) return false;
  // A per-model quota rejected before inference. Backoff cannot refill it; only
  // another provider can serve the call.
  if (statusOf(error) === 403 && THROTTLE.test(String((error as { message?: unknown })?.message ?? ""))) {
    return false;
  }
  return true;
}

/**
 * Record that a credential was refused, if the error says it was.
 *
 * Returns whether the credential is now considered down, so callers can log the
 * transition once instead of on every subsequent failure.
 */
export function noteFailure(id: string, error: unknown): boolean {
  if (!isPermanentRefusal(error)) return false;
  const first = !isRefused(id);
  const message = String((error as { message?: unknown })?.message ?? error);
  refused.set(id, { at: Date.now(), detail: message.replace(/\s+/g, " ").slice(0, 120) });
  if (first) {
    console.warn(
      `[credentials] taking a credential out of the rotation for ${Math.round(
        RECHECK_AFTER_MS / 60000,
      )}m: ${message.replace(/\s+/g, " ").slice(0, 100)}`,
    );
  }
  return true;
}

/** True while a credential's refusal is still believed. */
export function isRefused(id: string): boolean {
  const entry = refused.get(id);
  if (!entry) return false;
  if (Date.now() - entry.at < RECHECK_AFTER_MS) return true;
  // The window has passed: forget it, so the next call re-tests the credential
  // rather than condemning it forever on one reading.
  refused.delete(id);
  return false;
}

/**
 * Drop refused credentials from a pool, unless that would empty it.
 *
 * Never returning an empty pool is the important half. If every credential has
 * been refused, a chain of failing candidates is still better than no model at
 * all: the failure a caller sees should be the provider's own error, not a
 * configuration error this module invented.
 */
export function usable<T>(pool: T[], idOf: (entry: T) => string): T[] {
  const live = pool.filter((entry) => !isRefused(idOf(entry)));
  return live.length > 0 ? live : pool;
}

/** Refused credentials, for /api/health. Counts and reasons, never the keys. */
export function refusedCredentials(): Array<{ endpoint: string; detail: string; forMs: number }> {
  const now = Date.now();
  const out: Array<{ endpoint: string; detail: string; forMs: number }> = [];
  for (const [id, entry] of refused) {
    if (now - entry.at >= RECHECK_AFTER_MS) continue;
    out.push({
      // The key half of the identity is dropped: this is reported over HTTP.
      endpoint: id.split("|")[0],
      detail: entry.detail,
      forMs: now - entry.at,
    });
  }
  return out;
}

/** Test seam: forget every recorded refusal. */
export function resetCredentialHealth(): void {
  refused.clear();
}
