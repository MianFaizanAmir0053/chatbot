/**
 * Can the system say what a page used to say?
 *
 * It could not. Asked to compare a page with its previous version, a run
 * fetched the page, found nothing to compare against — the result cache is
 * per-request and dies with it — and wrote a confident comparison anyway,
 * inventing the earlier version from unrelated material.
 *
 * The guardrails catch that now, but catching it only turns a wrong answer into
 * no answer. These checks cover the part that makes the question answerable:
 * that a capture is kept, that an unchanged page is not stored twice, that the
 * previous version comes back as it was, and — the one that matters most — that
 * a page never read before reports exactly that rather than returning something
 * plausible.
 *
 * Usage: npx tsx --env-file=.env scripts/snapshot-test.ts
 */

import {
  forgetSnapshots,
  saveSnapshot,
  snapshotHistory,
  snapshotKey,
  snapshotsAvailable,
} from "../lib/snapshots/store";

let failures = 0;
let skipped = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

const URL_A = `https://example.invalid/test-${Date.now().toString(36)}`;

function keys() {
  banner("1. One page is one history");

  // A fragment addresses a position within a page, not a different page, and a
  // trailing slash is the same resource. Treating either as distinct gives a
  // page two unrelated histories and makes "has this changed?" unanswerable.
  check(
    "a fragment does not start a new history",
    snapshotKey("https://a.com/p#section") === snapshotKey("https://a.com/p"),
    snapshotKey("https://a.com/p#section"),
  );
  check(
    "nor a trailing slash",
    snapshotKey("https://a.com/p/") === snapshotKey("https://a.com/p"),
    snapshotKey("https://a.com/p/"),
  );
  check(
    "but a different path does",
    snapshotKey("https://a.com/p") !== snapshotKey("https://a.com/q"),
  );
  // A query string selects content, so it is part of the page's identity.
  check(
    "and so does a query string",
    snapshotKey("https://a.com/p?v=2") !== snapshotKey("https://a.com/p"),
  );
}

async function archive() {
  banner("2. Keeping what was read");

  if (!(await snapshotsAvailable())) {
    console.log("  SKIP  no archive configured (MONGODB_URI unset) — nothing to exercise");
    skipped++;
    return;
  }

  const first = await saveSnapshot(URL_A, "Example", "Senior Software Engineer. Three projects.");
  check("the first capture has no previous version", first.previous === null, "nothing before it");
  check("and reports no change", !first.changed, "there was nothing to change from");

  // The same page, unchanged. Storing it again would fill the history with
  // duplicates and make "last changed" mean "last fetched".
  const again = await saveSnapshot(URL_A, "Example", "Senior Software Engineer. Three projects.");
  check("an unchanged page is not stored twice", !again.changed, "deduplicated by content");
  check(
    "and the history still holds one capture",
    (await snapshotHistory(URL_A)).length === 1,
    `${(await snapshotHistory(URL_A)).length}`,
  );

  const changed = await saveSnapshot(URL_A, "Example", "Senior Software Engineer. Five projects.");
  check("a changed page reports the change", changed.changed, "at fetch time");
  check(
    "and hands back what it used to say",
    changed.previous?.text === "Senior Software Engineer. Three projects.",
    JSON.stringify(changed.previous?.text ?? null),
  );

  const history = await snapshotHistory(URL_A);
  check("the history holds both, newest first", history.length === 2, `${history.length}`);
  check(
    "with the current version first",
    history[0]?.text.includes("Five") === true && history[1]?.text.includes("Three") === true,
    history.map((h) => h.text.slice(-14)).join(" | "),
  );
}

async function theHonestAnswer() {
  banner("3. A page never read before");

  // The failure this exists to prevent. Asked what changed, a model with no
  // capture must be told there is none — not handed an empty string it can
  // read as "the page used to be blank", and not left to infer a previous
  // version from context.
  const unseen = await snapshotHistory(`https://example.invalid/never-${Date.now()}`);
  check("has no history at all", unseen.length === 0, "nothing to compare against");
}

async function forgetting() {
  banner("4. Forgetting a page");

  if (!(await snapshotsAvailable())) return;

  // The archive holds copies of third-party pages, so removing one is part of
  // the feature. It is also how this test avoids leaving its fixtures behind.
  const removed = await forgetSnapshots(URL_A);
  check("captures are removed on request", removed === 2, `${removed} removed`);
  check("and the history is empty after", (await snapshotHistory(URL_A)).length === 0);
}

async function main() {
  keys();
  await archive();
  await theHonestAnswer();
  await forgetting();

  banner("Summary");
  if (skipped > 0) console.log(`  ${skipped} section(s) skipped`);
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

void main();

export {};
