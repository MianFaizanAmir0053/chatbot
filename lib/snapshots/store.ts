import { createHash } from "crypto";
import { MongoClient, type Collection, type Db } from "mongodb";
import { env } from "../config";

/* ------------------------------------------------------------------ *
 * Page snapshots
 *
 * What a fetched web page said, and when.
 *
 * This exists because of a specific failure. Asked to compare a page against
 * its previous version, a run fetched the page, found nothing to compare it
 * with — the result cache is per-request and dies with it — and wrote a
 * confident comparison anyway, inventing the "before" column from unrelated
 * material. The guardrails now catch that, but catching it only converts a
 * wrong answer into no answer: the question was reasonable and the system
 * simply had no memory of ever having read the page.
 *
 * Keeping what was read makes the question answerable. It also makes a class of
 * question possible that was not before — what changed, and when — which is
 * most of the value of pointing a research agent at a page more than once.
 *
 * Storage is deliberately modest: text only, capped, deduplicated by content
 * hash so a page fetched ten times without changing is stored once. When no
 * database is configured this degrades to nothing, and the tools that use it
 * report that there is no history rather than failing.
 * ------------------------------------------------------------------ */

/**
 * Characters kept per capture.
 *
 * Matches what `fetch_url` hands the model, so a stored snapshot is exactly
 * what was read rather than a different, longer thing the answer never saw.
 */
const MAX_SNAPSHOT_CHARS = 6_000;

/** Captures kept per URL. Oldest are pruned beyond this. */
const MAX_SNAPSHOTS_PER_URL = 10;

export interface PageSnapshot {
  url: string;
  title: string;
  text: string;
  hash: string;
  fetchedAt: string;
}

interface SnapshotDoc extends PageSnapshot {
  _id: string;
}

let client: MongoClient | null = null;
let db: Db | null = null;
let ready: Promise<Collection<SnapshotDoc> | null> | null = null;

/**
 * Normalise a URL so the same page is one series of captures.
 *
 * A fragment addresses a position within a page, not a different page, and a
 * trailing slash is the same resource — treating either as distinct would give
 * a page two unrelated histories and make "has this changed?" unanswerable for
 * the reason least worth being unanswerable for.
 */
export function snapshotKey(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    u.hash = "";
    if (u.pathname.length > 1 && u.pathname.endsWith("/")) u.pathname = u.pathname.slice(0, -1);
    return u.toString();
  } catch {
    return rawUrl;
  }
}

async function collection(): Promise<Collection<SnapshotDoc> | null> {
  if (!env.MONGODB_URI) return null;
  if (!ready) {
    ready = (async () => {
      try {
        client = new MongoClient(env.MONGODB_URI!);
        await client.connect();
        db = client.db();
        const col = db.collection<SnapshotDoc>("page_snapshots");
        // Newest capture of a URL is the common read, and pruning walks the
        // same order.
        await col.createIndex({ url: 1, fetchedAt: -1 });
        return col;
      } catch (error) {
        // A missing archive costs history, not the fetch itself. Reported once,
        // loudly, because the degradation is otherwise invisible until someone
        // asks what changed and is told nothing ever did.
        console.warn(
          "[snapshots] archive unavailable, page history will not be kept:",
          error instanceof Error ? error.message : String(error),
        );
        return null;
      }
    })();
  }
  return ready;
}

/** Content identity, so an unchanged page is not stored again. */
function hashOf(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 32);
}

/**
 * Record what a page said, unless it says exactly what it said last time.
 *
 * Returns the previous *different* capture when there is one, so the caller can
 * tell the model the page has changed at the moment it matters — while it is
 * reading the page — rather than leaving it to ask.
 */
export async function saveSnapshot(
  rawUrl: string,
  title: string,
  text: string,
): Promise<{ previous: PageSnapshot | null; changed: boolean }> {
  const col = await collection();
  if (!col || !text.trim()) return { previous: null, changed: false };

  const url = snapshotKey(rawUrl);
  const body = text.slice(0, MAX_SNAPSHOT_CHARS);
  const hash = hashOf(body);

  try {
    const latest = await col.findOne({ url }, { sort: { fetchedAt: -1 } });

    // Unchanged: no new row, and nothing to report. The capture that exists
    // already says the same thing, and its date is when the content was first
    // seen — which is the more useful date of the two.
    if (latest?.hash === hash) return { previous: null, changed: false };

    await col.insertOne({
      _id: `${hashOf(url)}-${Date.now().toString(36)}`,
      url,
      title,
      text: body,
      hash,
      fetchedAt: new Date().toISOString(),
    });

    // Pruned oldest-first rather than by age: what matters is being able to
    // reach back a few versions, and a page fetched daily should not accumulate
    // indefinitely.
    const count = await col.countDocuments({ url });
    if (count > MAX_SNAPSHOTS_PER_URL) {
      const stale = await col
        .find({ url }, { sort: { fetchedAt: 1 }, limit: count - MAX_SNAPSHOTS_PER_URL })
        .toArray();
      if (stale.length > 0) {
        await col.deleteMany({ _id: { $in: stale.map((s) => s._id) } });
      }
    }

    return { previous: latest ? toSnapshot(latest) : null, changed: Boolean(latest) };
  } catch (error) {
    console.warn(
      "[snapshots] could not record a capture:",
      error instanceof Error ? error.message : String(error),
    );
    return { previous: null, changed: false };
  }
}

function toSnapshot(doc: SnapshotDoc): PageSnapshot {
  return {
    url: doc.url,
    title: doc.title,
    text: doc.text,
    hash: doc.hash,
    fetchedAt: doc.fetchedAt,
  };
}

/** Every capture of a page, newest first. */
export async function snapshotHistory(rawUrl: string): Promise<PageSnapshot[]> {
  const col = await collection();
  if (!col) return [];
  try {
    const docs = await col
      .find({ url: snapshotKey(rawUrl) }, { sort: { fetchedAt: -1 }, limit: MAX_SNAPSHOTS_PER_URL })
      .toArray();
    return docs.map(toSnapshot);
  } catch {
    return [];
  }
}

/**
 * Forget every capture of a page.
 *
 * This archive holds copies of third-party pages, so being able to remove one
 * is part of the feature rather than an afterthought — and it is what a test
 * needs to clean up after itself. Returns how many captures were removed.
 */
export async function forgetSnapshots(rawUrl: string): Promise<number> {
  const col = await collection();
  if (!col) return 0;
  try {
    const result = await col.deleteMany({ url: snapshotKey(rawUrl) });
    return result.deletedCount ?? 0;
  } catch {
    return 0;
  }
}

/** Whether the archive is available at all, for /api/health. */
export async function snapshotsAvailable(): Promise<boolean> {
  return (await collection()) !== null;
}
