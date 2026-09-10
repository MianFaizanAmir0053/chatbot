import { randomUUID } from "crypto";
import { promises as fs } from "fs";
import path from "path";

/* ------------------------------------------------------------------ *
 * Conversation store
 *
 * The agent's own memory lives in the LangGraph checkpointer, which is
 * per-process and holds graph state rather than anything renderable. A
 * conversation the user can leave, find again in a list and reopen is a
 * different thing: it needs a title, a timestamp, and the transcript as
 * it was displayed — sources and groundedness included.
 *
 * Stored as one JSON file per conversation under `.data/conversations`,
 * with an in-memory index in front. A file per conversation keeps each
 * write small and means a corrupt file costs one thread, not all of
 * them. Where the filesystem is read-only — most serverless targets —
 * every write degrades to memory-only for the life of the process, so
 * the feature still works within a session instead of failing.
 * ------------------------------------------------------------------ */

export type StoredRole = "user" | "assistant";

export interface StoredSource {
  index: number;
  source: string;
  section?: string;
  page?: number;
  score: number;
  excerpt: string;
}

export interface StoredMessage {
  role: StoredRole;
  content: string;
  sources?: StoredSource[];
  web?: Array<{ title: string; url: string }>;
  groundedness?: {
    score: number;
    verdict: string;
    passed: boolean;
    unsupportedClaims: string[];
  };
  blocked?: boolean;
  createdAt: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: StoredMessage[];
  /**
   * The conversation this one was forked from, if any.
   *
   * Recorded rather than resolved once, because it is what lets a fork read the
   * parent's documents without owning them. Copying the documents instead would
   * mean re-embedding every chunk — the expensive half of an upload — so a fork
   * would cost as much as the original and take as long.
   */
  forkedFrom?: string;
  /** True once the user has named it, so a later turn cannot overwrite it. */
  titleLocked?: boolean;
}

/** What the sidebar and dashboard list — never the whole transcript. */
export interface ConversationSummary {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: number;
  forkedFrom?: string;
  /** How many conversations were forked from this one. */
  forks: number;
  /** Derived from the last assistant turn, so a list can show what happened. */
  status: "completed" | "flagged" | "in_progress" | "blocked";
}

const DIR = path.join(process.cwd(), ".data", "conversations");
const MAX_TITLE = 72;

/** id -> conversation. Also the fallback store when the disk is unwritable. */
const cache = new Map<string, Conversation>();
let hydrated = false;
let writable = true;

function isSafeId(id: string): boolean {
  // Ids are UUIDs from the chat route. Anything else is a path traversal
  // attempt or a bug, and either way must not reach the filesystem.
  return /^[a-zA-Z0-9_-]{1,64}$/.test(id);
}

function fileFor(id: string): string {
  return path.join(DIR, `${id}.json`);
}

/** Reads every stored conversation once per process. */
async function hydrate(): Promise<void> {
  if (hydrated) return;
  hydrated = true;
  try {
    const entries = await fs.readdir(DIR);
    await Promise.all(
      entries
        .filter((name) => name.endsWith(".json"))
        .map(async (name) => {
          try {
            const raw = await fs.readFile(path.join(DIR, name), "utf8");
            const parsed = JSON.parse(raw) as Conversation;
            if (parsed?.id && Array.isArray(parsed.messages)) cache.set(parsed.id, parsed);
          } catch {
            // One unreadable file must not take the rest of the list with it.
          }
        }),
    );
  } catch {
    // No directory yet: nothing stored, which is not an error.
  }
}

async function persist(conversation: Conversation): Promise<void> {
  if (!writable) return;
  try {
    await fs.mkdir(DIR, { recursive: true });
    await fs.writeFile(fileFor(conversation.id), JSON.stringify(conversation, null, 2), "utf8");
  } catch (error) {
    writable = false;
    console.warn(
      `[conversations] filesystem is not writable, keeping conversations in memory only: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

/** First line of the opening question, trimmed to something list-sized. */
function titleFrom(text: string): string {
  const line = text.replace(/\s+/g, " ").trim();
  if (!line) return "New conversation";
  return line.length > MAX_TITLE ? `${line.slice(0, MAX_TITLE - 1)}…` : line;
}

function statusOf(conversation: Conversation): ConversationSummary["status"] {
  const last = conversation.messages.at(-1);
  if (!last) return "in_progress";
  if (last.blocked) return "blocked";
  if (last.role === "user") return "in_progress";
  if (last.groundedness && !last.groundedness.passed) return "flagged";
  return "completed";
}

function summarise(conversation: Conversation): ConversationSummary {
  return {
    id: conversation.id,
    title: conversation.title,
    createdAt: conversation.createdAt,
    updatedAt: conversation.updatedAt,
    messages: conversation.messages.length,
    forkedFrom: conversation.forkedFrom,
    // Surfaced so deleting a conversation can warn that its forks will lose
    // access to its documents, which is otherwise an invisible consequence.
    forks: [...cache.values()].filter((c) => c.forkedFrom === conversation.id).length,
    status: statusOf(conversation),
  };
}

/**
 * A conversation and every ancestor it inherits documents from.
 *
 * Ordered nearest-first, and cycle-guarded: the chain is user-influenced data
 * read on every retrieval, so a corrupted or hand-edited file that made two
 * conversations each other's parent would otherwise hang the request rather
 * than degrade it.
 */
export async function scopeChainFor(id: string): Promise<string[]> {
  if (!isSafeId(id)) return [];
  await hydrate();

  const chain: string[] = [];
  const seen = new Set<string>();
  let cursor: string | undefined = id;

  while (cursor && !seen.has(cursor) && chain.length < 32) {
    seen.add(cursor);
    chain.push(cursor);
    cursor = cache.get(cursor)?.forkedFrom;
  }
  return chain;
}

/** Rename a conversation, and stop later turns from re-deriving the title. */
export async function renameConversation(
  id: string,
  title: string,
): Promise<Conversation | null> {
  if (!isSafeId(id)) return null;
  await hydrate();

  const conversation = cache.get(id);
  if (!conversation) return null;

  const trimmed = title.replace(/\s+/g, " ").trim();
  if (!trimmed) return null;

  conversation.title =
    trimmed.length > MAX_TITLE ? `${trimmed.slice(0, MAX_TITLE - 1)}…` : trimmed;
  // Locked, or the auto-title rule in appendMessages would quietly revert a
  // deliberate name the next time the conversation is used.
  conversation.titleLocked = true;
  conversation.updatedAt = new Date().toISOString();

  cache.set(id, conversation);
  await persist(conversation);
  return conversation;
}

/**
 * Copy a conversation into a new one, optionally truncated.
 *
 * `upTo` is a message count, so a user can branch from a point mid-thread and
 * explore a different line of questioning without disturbing the original.
 * Documents are not copied — the fork inherits them through `forkedFrom`.
 */
export async function forkConversation(
  id: string,
  options: { upTo?: number; title?: string } = {},
): Promise<Conversation | null> {
  if (!isSafeId(id)) return null;
  await hydrate();

  const source = cache.get(id);
  if (!source) return null;

  const now = new Date().toISOString();
  const messages =
    typeof options.upTo === "number"
      ? source.messages.slice(0, Math.max(0, options.upTo))
      : [...source.messages];

  const fork: Conversation = {
    id: randomUUID(),
    title: options.title?.trim() || `${source.title} (fork)`,
    createdAt: now,
    updatedAt: now,
    // Deep-copied: the transcripts share no message objects, so editing or
    // appending to one cannot mutate the other through a shared reference.
    messages: messages.map((m) => ({ ...m })),
    forkedFrom: source.id,
    titleLocked: true,
  };

  cache.set(fork.id, fork);
  await persist(fork);
  return fork;
}

/** Newest first — the order both the sidebar and the dashboard want. */
export async function listConversations(limit = 50): Promise<ConversationSummary[]> {
  await hydrate();
  return [...cache.values()]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, limit)
    .map(summarise);
}

export async function getConversation(id: string): Promise<Conversation | null> {
  if (!isSafeId(id)) return null;
  await hydrate();
  return cache.get(id) ?? null;
}

/**
 * Appends turns to a conversation, creating it on the first one.
 *
 * Called once for the question and again for the answer rather than once
 * per completed exchange: a run that is abandoned halfway still leaves the
 * question in the list, which is what makes the conversation findable
 * again at all.
 */
export async function appendMessages(
  id: string,
  messages: StoredMessage[],
): Promise<Conversation | null> {
  if (!isSafeId(id) || messages.length === 0) return null;
  await hydrate();

  const now = new Date().toISOString();
  const existing = cache.get(id);
  const conversation: Conversation = existing ?? {
    id,
    title: titleFrom(messages.find((m) => m.role === "user")?.content ?? ""),
    createdAt: now,
    updatedAt: now,
    messages: [],
  };

  conversation.messages.push(...messages);
  conversation.updatedAt = now;

  // A conversation created by an assistant-only turn (an upload notice, say)
  // has no title yet; the first real question supplies it — unless the user has
  // named it, in which case their name wins.
  if (conversation.title === "New conversation" && !conversation.titleLocked) {
    const firstQuestion = conversation.messages.find((m) => m.role === "user");
    if (firstQuestion) conversation.title = titleFrom(firstQuestion.content);
  }

  cache.set(id, conversation);
  await persist(conversation);
  return conversation;
}

export async function deleteConversation(id: string): Promise<boolean> {
  if (!isSafeId(id)) return false;
  await hydrate();
  const existed = cache.delete(id);
  if (writable) {
    try {
      await fs.unlink(fileFor(id));
    } catch {
      // Already gone, or never written because the disk is read-only.
    }
  }
  return existed;
}
