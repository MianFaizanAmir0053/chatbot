import { randomUUID } from "crypto";
import { MongoClient, type Collection, type Db } from "mongodb";
import {
  clampTitle,
  isSafeId,
  statusOf,
  titleFrom,
  walkChain,
  type Conversation,
  type ConversationDriver,
  type ConversationSummary,
  type StoredMessage,
} from "./types";

/**
 * MongoDB-backed conversation storage.
 *
 * Chosen over the JSON-file driver for two reasons the file store cannot
 * address. Conversations survive a restart wherever the filesystem does not —
 * most serverless targets have a read-only disk, where the file driver silently
 * degrades to memory and every conversation is lost when the instance recycles.
 * And appends are atomic: the file driver read the whole conversation, mutated
 * it and wrote it back, so two turns landing together could lose one. Here a
 * turn is a `$push`, which the database serialises.
 *
 * Deliberately holds no read cache. The point of a shared database is that
 * several instances see the same conversations, and a process cache would serve
 * each one its own stale copy — reintroducing the problem this replaces.
 */

/** Stored shape. `_id` is the conversation id, so lookups need no secondary index. */
interface ConversationDoc {
  _id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: StoredMessage[];
  forkedFrom?: string;
  titleLocked?: boolean;
}

function toConversation(doc: ConversationDoc): Conversation {
  return {
    id: doc._id,
    title: doc.title,
    createdAt: doc.createdAt,
    updatedAt: doc.updatedAt,
    messages: doc.messages ?? [],
    forkedFrom: doc.forkedFrom,
    titleLocked: doc.titleLocked,
  };
}

export class MongoConversationDriver implements ConversationDriver {
  readonly name = "mongodb";
  readonly durable = true;

  private client: MongoClient;
  private db: Db;
  private ready: Promise<Collection<ConversationDoc>> | null = null;

  constructor(uri: string) {
    // Bounded pool and short timeouts: this sits on the request path, and a
    // database that is slow to answer should fail the write rather than hold a
    // streaming response open behind it.
    this.client = new MongoClient(uri, {
      maxPoolSize: 10,
      serverSelectionTimeoutMS: 8000,
      connectTimeoutMS: 8000,
    });
    // The database is named in the URI; falling back keeps a URI without a path
    // working rather than throwing on the first query.
    this.db = this.client.db();
  }

  /**
   * Connect once per process and reuse it.
   *
   * The promise itself is cached rather than the collection, so concurrent
   * first requests share one connection attempt instead of opening a client
   * each — under Next.js's dev reloads that is the difference between one pool
   * and one per request.
   */
  private collection(): Promise<Collection<ConversationDoc>> {
    if (!this.ready) {
      this.ready = (async () => {
        await this.client.connect();
        const col = this.db.collection<ConversationDoc>("conversations");
        try {
          await Promise.all([
            // The list is always "newest first", so it is always an index scan.
            col.createIndex({ updatedAt: -1 }),
            // Fork counts are looked up per listing; without this they are a
            // collection scan for every conversation shown.
            col.createIndex({ forkedFrom: 1 }),
          ]);
        } catch (error) {
          // Index creation can be refused on a constrained cluster. Queries
          // still work, just slower, so this must not stop the driver starting.
          console.warn(
            `[conversations] could not ensure indexes: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
        return col;
      })().catch((error) => {
        // Cleared so a later request retries rather than inheriting a rejected
        // promise for the life of the process.
        this.ready = null;
        throw error;
      });
    }
    return this.ready;
  }

  async healthy(): Promise<boolean> {
    try {
      const col = await this.collection();
      await col.estimatedDocumentCount();
      return true;
    } catch {
      return false;
    }
  }

  async list(limit: number): Promise<ConversationSummary[]> {
    const col = await this.collection();

    // Only the fields a list needs. Transcripts are the bulk of a conversation
    // and pulling them to count messages would move megabytes to render a
    // sidebar.
    const docs = await col
      .find({}, { projection: { messages: { $slice: -1 }, title: 1, createdAt: 1, updatedAt: 1, forkedFrom: 1 } })
      .sort({ updatedAt: -1 })
      .limit(limit)
      .toArray();

    // Message counts and fork counts in two aggregate queries rather than one
    // per conversation, which would be a round trip per row.
    const ids = docs.map((d) => d._id);
    const [sizes, forkCounts] = await Promise.all([
      col
        .aggregate<{ _id: string; messages: number }>([
          { $match: { _id: { $in: ids } } },
          { $project: { messages: { $size: { $ifNull: ["$messages", []] } } } },
        ])
        .toArray(),
      col
        .aggregate<{ _id: string; count: number }>([
          { $match: { forkedFrom: { $in: ids } } },
          { $group: { _id: "$forkedFrom", count: { $sum: 1 } } },
        ])
        .toArray(),
    ]);

    const sizeById = new Map(sizes.map((s) => [s._id, s.messages]));
    const forksById = new Map(forkCounts.map((f) => [f._id, f.count]));

    return docs.map((doc) => ({
      id: doc._id,
      title: doc.title,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
      messages: sizeById.get(doc._id) ?? 0,
      forkedFrom: doc.forkedFrom,
      forks: forksById.get(doc._id) ?? 0,
      // The projection kept only the last message, which is all `statusOf`
      // reads — it derives the state from the final turn.
      status: statusOf({ messages: doc.messages ?? [] }),
    }));
  }

  async get(id: string): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    const col = await this.collection();
    const doc = await col.findOne({ _id: id });
    return doc ? toConversation(doc) : null;
  }

  async append(id: string, messages: StoredMessage[]): Promise<Conversation | null> {
    if (!isSafeId(id) || messages.length === 0) return null;
    const col = await this.collection();
    const now = new Date().toISOString();

    // One atomic upsert. `$push` appends without reading first, so two turns
    // arriving together cannot overwrite each other — the failure mode the
    // file driver's read-mutate-write had.
    const firstQuestion = messages.find((m) => m.role === "user");
    const result = await col.findOneAndUpdate(
      { _id: id },
      {
        $push: { messages: { $each: messages } },
        $set: { updatedAt: now },
        $setOnInsert: {
          createdAt: now,
          title: titleFrom(firstQuestion?.content ?? ""),
        },
      },
      { upsert: true, returnDocument: "after" },
    );

    if (!result) return null;

    // A conversation created by an assistant-only turn (an upload notice, say)
    // has no title yet; the first real question supplies it — unless the user
    // has named it, in which case their name wins.
    if (result.title === "New conversation" && !result.titleLocked) {
      const question = result.messages?.find((m) => m.role === "user");
      if (question) {
        const title = titleFrom(question.content);
        await col.updateOne({ _id: id }, { $set: { title } });
        result.title = title;
      }
    }

    return toConversation(result);
  }

  async rename(id: string, title: string): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    const next = clampTitle(title);
    if (!next) return null;

    const col = await this.collection();
    const result = await col.findOneAndUpdate(
      { _id: id },
      // Locked, or the auto-title rule in `append` would quietly revert a
      // deliberate name the next time the conversation is used.
      { $set: { title: next, titleLocked: true, updatedAt: new Date().toISOString() } },
      { returnDocument: "after" },
    );
    return result ? toConversation(result) : null;
  }

  async fork(
    id: string,
    options: { upTo?: number; title?: string },
  ): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    const col = await this.collection();
    const source = await col.findOne({ _id: id });
    if (!source) return null;

    const now = new Date().toISOString();
    const messages =
      typeof options.upTo === "number"
        ? (source.messages ?? []).slice(0, Math.max(0, options.upTo))
        : (source.messages ?? []);

    const fork: ConversationDoc = {
      _id: randomUUID(),
      title: clampTitle(options.title ?? `${source.title} (fork)`) ?? "Fork",
      createdAt: now,
      updatedAt: now,
      messages: messages.map((m) => ({ ...m })),
      forkedFrom: source._id,
      titleLocked: true,
    };

    await col.insertOne(fork);
    return toConversation(fork);
  }

  async remove(id: string): Promise<boolean> {
    if (!isSafeId(id)) return false;
    const col = await this.collection();
    const result = await col.deleteOne({ _id: id });
    return result.deletedCount > 0;
  }

  async chain(id: string): Promise<string[]> {
    const col = await this.collection();
    // One lookup per ancestor. Chains are short — a fork of a fork of a fork is
    // already unusual — and $graphLookup would cost more to read than it saves.
    return walkChain(id, async (cursor) => {
      const doc = await col.findOne(
        { _id: cursor },
        { projection: { forkedFrom: 1 } },
      );
      return doc?.forkedFrom;
    });
  }
}
