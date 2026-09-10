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
  /\b(restricted|suspended|deactivated|disabled|banned|revoked|terminated|invalid api key|invalid_api_key|incorrect api key|no longer active)\b/i;

/**
 * True when retrying this error with this credential cannot succeed.
 *
 * Deliberately narrow. A false positive removes a working key from a pool that
 * exists to survive exactly this kind of trouble, so anything ambiguous —
 * every 429, every 5xx, every timeout or transport failure — is treated as
 * temporary. Those are properties of the moment; these are properties of the
 * account.
 */
export function isPermanentRefusal(error: unknown): boolean {
  const status = statusOf(error);
  if (status === 401 || status === 403) return true;
  if (status === 400) {
    const message = String((error as { message?: unknown })?.message ?? "");
    return ACCOUNT_REFUSED.test(message);
  }
  return false;
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
