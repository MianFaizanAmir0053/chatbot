import { env } from "../config";
import { FileConversationDriver } from "./file-driver";
import { MongoConversationDriver } from "./mongo-driver";
import type { Conversation, ConversationDriver, ConversationSummary, StoredMessage } from "./types";

export type {
  Conversation,
  ConversationSummary,
  StoredMessage,
  StoredRole,
  StoredSource,
} from "./types";

/* ------------------------------------------------------------------ *
 * Conversation store
 *
 * One module the routes talk to, over a driver chosen once per process.
 * MongoDB when MONGODB_URI is set, JSON files otherwise.
 * ------------------------------------------------------------------ */

let driver: ConversationDriver | null = null;
let selecting: Promise<ConversationDriver> | null = null;

async function selectDriver(): Promise<ConversationDriver> {
  if (env.MONGODB_URI) {
    const mongo = new MongoConversationDriver(env.MONGODB_URI);
    if (await mongo.healthy()) {
      console.log("[conversations] using MongoDB (durable, shared between instances)");
      return mongo;
    }
    // Falling back rather than failing: a database that is unreachable at boot
    // should cost durability, not the ability to hold a conversation at all.
    // Reported loudly because the degradation is otherwise invisible until a
    // restart loses everything.
    console.warn(
      "[conversations] MONGODB_URI is set but the database is unreachable — " +
        "falling back to JSON files. Conversations will not be shared between instances.",
    );
  } else {
    console.warn(
      "[conversations] MONGODB_URI not set — using JSON files. " +
        "Set it for storage that survives a restart on a read-only filesystem.",
    );
  }
  return new FileConversationDriver();
}

/**
 * Resolve the active driver.
 *
 * The health probe runs once per process and is shared between concurrent
 * callers, so a cold start under parallel load does not open a client per
 * request — the same shape the vector store's factory uses.
 */
async function getDriver(): Promise<ConversationDriver> {
  if (driver) return driver;
  if (!selecting) {
    selecting = selectDriver().then((selected) => {
      driver = selected;
      selecting = null;
      return selected;
    });
  }
  return selecting;
}

/** Which store is in use, for /api/health. */
export async function conversationStoreInfo(): Promise<{ driver: string; durable: boolean }> {
  const active = await getDriver();
  return { driver: active.name, durable: active.durable };
}

/** Newest first — the order both the sidebar and the dashboard want. */
export async function listConversations(limit = 50): Promise<ConversationSummary[]> {
  return (await getDriver()).list(limit);
}

export async function getConversation(id: string): Promise<Conversation | null> {
  return (await getDriver()).get(id);
}

export async function appendMessages(
  id: string,
  messages: StoredMessage[],
): Promise<Conversation | null> {
  return (await getDriver()).append(id, messages);
}

export async function renameConversation(
  id: string,
  title: string,
): Promise<Conversation | null> {
  return (await getDriver()).rename(id, title);
}

export async function forkConversation(
  id: string,
  options: { upTo?: number; title?: string } = {},
): Promise<Conversation | null> {
  return (await getDriver()).fork(id, options);
}

export async function deleteConversation(id: string): Promise<boolean> {
  return (await getDriver()).remove(id);
}

/**
 * A conversation and every ancestor it inherits documents from.
 *
 * Read on every retrieval, because a fork's documents live under its parent's
 * id: a fork scoped to itself alone would find nothing and report that the
 * documents do not cover the question.
 */
export async function scopeChainFor(id: string): Promise<string[]> {
  return (await getDriver()).chain(id);
}
