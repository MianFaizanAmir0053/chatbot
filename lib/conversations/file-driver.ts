import { randomUUID } from "crypto";
import { promises as fs } from "fs";
import path from "path";
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
 * Conversations as JSON files, with an in-memory index in front.
 *
 * The fallback for a deployment with no database configured. A file per
 * conversation keeps each write small and means a corrupt file costs one thread
 * rather than all of them.
 *
 * Two limitations are inherent and are why MongoDB is preferred where it is
 * available. Where the filesystem is read-only — most serverless targets — every
 * write degrades to memory-only for the life of the process, so conversations
 * are lost when the instance recycles. And an append is a read, a mutation and a
 * write, so two turns landing at once can lose one; the database driver makes
 * that a single atomic push.
 */
export class FileConversationDriver implements ConversationDriver {
  readonly name = "file";

  private readonly dir = path.join(process.cwd(), ".data", "conversations");
  /** id -> conversation. Also the whole store when the disk is unwritable. */
  private readonly cache = new Map<string, Conversation>();
  private hydrated = false;
  private writable = true;

  get durable(): boolean {
    // Honest rather than optimistic: once a write has failed, conversations
    // exist only in this process and /api/health should say so.
    return this.writable;
  }

  async healthy(): Promise<boolean> {
    return true;
  }

  private fileFor(id: string): string {
    return path.join(this.dir, `${id}.json`);
  }

  /** Reads every stored conversation once per process. */
  private async hydrate(): Promise<void> {
    if (this.hydrated) return;
    this.hydrated = true;
    try {
      const entries = await fs.readdir(this.dir);
      await Promise.all(
        entries
          .filter((name) => name.endsWith(".json"))
          .map(async (name) => {
            try {
              const raw = await fs.readFile(path.join(this.dir, name), "utf8");
              const parsed = JSON.parse(raw) as Conversation;
              if (parsed?.id && Array.isArray(parsed.messages)) {
                this.cache.set(parsed.id, parsed);
              }
            } catch {
              // One unreadable file must not take the rest of the list with it.
            }
          }),
      );
    } catch {
      // No directory yet: nothing stored, which is not an error.
    }
  }

  private async persist(conversation: Conversation): Promise<void> {
    if (!this.writable) return;
    try {
      await fs.mkdir(this.dir, { recursive: true });
      await fs.writeFile(
        this.fileFor(conversation.id),
        JSON.stringify(conversation, null, 2),
        "utf8",
      );
    } catch (error) {
      this.writable = false;
      console.warn(
        `[conversations] filesystem is not writable, keeping conversations in memory only: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }

  private summarise(conversation: Conversation): ConversationSummary {
    return {
      id: conversation.id,
      title: conversation.title,
      createdAt: conversation.createdAt,
      updatedAt: conversation.updatedAt,
      messages: conversation.messages.length,
      forkedFrom: conversation.forkedFrom,
      // Surfaced so deleting a conversation can warn that its forks will lose
      // access to its documents, which is otherwise an invisible consequence.
      forks: [...this.cache.values()].filter((c) => c.forkedFrom === conversation.id).length,
      status: statusOf(conversation),
    };
  }

  async list(limit: number): Promise<ConversationSummary[]> {
    await this.hydrate();
    return [...this.cache.values()]
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .slice(0, limit)
      .map((c) => this.summarise(c));
  }

  async get(id: string): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    await this.hydrate();
    return this.cache.get(id) ?? null;
  }

  async append(id: string, messages: StoredMessage[]): Promise<Conversation | null> {
    if (!isSafeId(id) || messages.length === 0) return null;
    await this.hydrate();

    const now = new Date().toISOString();
    const existing = this.cache.get(id);
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
    // has no title yet; the first real question supplies it — unless the user
    // has named it, in which case their name wins.
    if (conversation.title === "New conversation" && !conversation.titleLocked) {
      const firstQuestion = conversation.messages.find((m) => m.role === "user");
      if (firstQuestion) conversation.title = titleFrom(firstQuestion.content);
    }

    this.cache.set(id, conversation);
    await this.persist(conversation);
    return conversation;
  }

  async rename(id: string, title: string): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    await this.hydrate();

    const conversation = this.cache.get(id);
    if (!conversation) return null;

    const next = clampTitle(title);
    if (!next) return null;

    conversation.title = next;
    conversation.titleLocked = true;
    conversation.updatedAt = new Date().toISOString();

    this.cache.set(id, conversation);
    await this.persist(conversation);
    return conversation;
  }

  async fork(
    id: string,
    options: { upTo?: number; title?: string },
  ): Promise<Conversation | null> {
    if (!isSafeId(id)) return null;
    await this.hydrate();

    const source = this.cache.get(id);
    if (!source) return null;

    const now = new Date().toISOString();
    const messages =
      typeof options.upTo === "number"
        ? source.messages.slice(0, Math.max(0, options.upTo))
        : [...source.messages];

    const fork: Conversation = {
      id: randomUUID(),
      title: clampTitle(options.title ?? `${source.title} (fork)`) ?? "Fork",
      createdAt: now,
      updatedAt: now,
      // Deep-copied: the transcripts share no message objects, so appending to
      // one cannot mutate the other through a shared reference.
      messages: messages.map((m) => ({ ...m })),
      forkedFrom: source.id,
      titleLocked: true,
    };

    this.cache.set(fork.id, fork);
    await this.persist(fork);
    return fork;
  }

  async remove(id: string): Promise<boolean> {
    if (!isSafeId(id)) return false;
    await this.hydrate();
    const existed = this.cache.delete(id);
    if (this.writable) {
      try {
        await fs.unlink(this.fileFor(id));
      } catch {
        // Already gone, or never written because the disk is read-only.
      }
    }
    return existed;
  }

  async chain(id: string): Promise<string[]> {
    await this.hydrate();
    return walkChain(id, async (cursor) => this.cache.get(cursor)?.forkedFrom);
  }
}
