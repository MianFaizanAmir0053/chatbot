/* ------------------------------------------------------------------ *
 * Conversation storage: types and the driver contract
 *
 * The agent's own memory lives in the LangGraph checkpointer, which is
 * per-process and holds graph state rather than anything renderable. A
 * conversation the user can leave, find again in a list and reopen is a
 * different thing: it needs a title, a timestamp, and the transcript as it
 * was displayed — sources and groundedness included.
 *
 * Written against a driver so the backing store can change without touching
 * the routes, the same shape the vector store already uses. MongoDB is the
 * durable option; the JSON-file driver remains for a machine with no database
 * configured, and degrades to memory where the filesystem is read-only.
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

export interface ConversationDriver {
  readonly name: string;
  /** True when conversations survive a process restart. */
  readonly durable: boolean;
  list(limit: number): Promise<ConversationSummary[]>;
  get(id: string): Promise<Conversation | null>;
  /**
   * Append turns, creating the conversation on the first one.
   *
   * Called once for the question and again for the answer rather than once per
   * completed exchange: a run that is abandoned halfway still leaves the
   * question in the list, which is what makes the conversation findable again.
   */
  append(id: string, messages: StoredMessage[]): Promise<Conversation | null>;
  rename(id: string, title: string): Promise<Conversation | null>;
  fork(id: string, options: { upTo?: number; title?: string }): Promise<Conversation | null>;
  remove(id: string): Promise<boolean>;
  /** A conversation and every ancestor it inherits documents from, nearest first. */
  chain(id: string): Promise<string[]>;
  healthy(): Promise<boolean>;
}

export const MAX_TITLE = 72;

/** Ids come from the chat route as UUIDs. Anything else is a bug or an attack. */
export function isSafeId(id: string): boolean {
  return /^[a-zA-Z0-9_-]{1,64}$/.test(id);
}

/** First line of the opening question, trimmed to something list-sized. */
export function titleFrom(text: string): string {
  const line = text.replace(/\s+/g, " ").trim();
  if (!line) return "New conversation";
  return line.length > MAX_TITLE ? `${line.slice(0, MAX_TITLE - 1)}…` : line;
}

export function clampTitle(title: string): string | null {
  const trimmed = title.replace(/\s+/g, " ").trim();
  if (!trimmed) return null;
  return trimmed.length > MAX_TITLE ? `${trimmed.slice(0, MAX_TITLE - 1)}…` : trimmed;
}

export function statusOf(conversation: {
  messages: StoredMessage[];
}): ConversationSummary["status"] {
  const last = conversation.messages.at(-1);
  if (!last) return "in_progress";
  if (last.blocked) return "blocked";
  if (last.role === "user") return "in_progress";
  if (last.groundedness && !last.groundedness.passed) return "flagged";
  return "completed";
}

/**
 * Walk a fork chain from a lookup function.
 *
 * Shared by both drivers, and cycle-guarded: the chain is read on every
 * retrieval and is influenced by stored data, so a corrupted record that made
 * two conversations each other's parent would hang the request rather than
 * degrade it. Depth is capped for the same reason.
 */
export async function walkChain(
  id: string,
  parentOf: (id: string) => Promise<string | undefined>,
): Promise<string[]> {
  if (!isSafeId(id)) return [];

  const chain: string[] = [];
  const seen = new Set<string>();
  let cursor: string | undefined = id;

  while (cursor && !seen.has(cursor) && chain.length < 32) {
    seen.add(cursor);
    chain.push(cursor);
    cursor = await parentOf(cursor);
  }
  return chain;
}
