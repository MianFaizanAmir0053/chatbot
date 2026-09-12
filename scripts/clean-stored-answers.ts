/**
 * Remove leaked planner output from stored answers, permanently.
 *
 * For a while, a query planner invoked inside a retrieval tool had its JSON
 * streamed into the answer, so replies were written to the store as
 * `{"variants":[…],"hypotheticalAnswer":"…"}` followed by the real text —
 * sometimes three objects deep.
 *
 * The leak is closed and `getConversation` already strips it on read, which
 * fixes the transcript and the turns replayed into the model. That repair is
 * deliberately non-destructive, but it only helps readers that go through this
 * codebase: an export, a different client, or a query run straight against the
 * collection still sees the JSON. This rewrites the rows so the data itself is
 * correct.
 *
 * Dry by default — it reports what it would change and touches nothing. Pass
 * `--write` to apply.
 *
 * Usage:
 *   npx tsx --env-file=.env scripts/clean-stored-answers.ts
 *   npx tsx --env-file=.env scripts/clean-stored-answers.ts --write
 */

import { env } from "../lib/config";
import { stripLeakedAuxJson } from "../lib/guardrails/output";

const WRITE = process.argv.includes("--write");

interface StoredDoc {
  _id: string;
  title?: string;
  messages?: Array<{ role: string; content: string }>;
}

async function main() {
  if (!env.MONGODB_URI) {
    console.error(
      "MONGODB_URI is not set. The JSON-file store is repaired on read and needs no migration.",
    );
    process.exit(1);
  }

  // Imported here rather than at the top so the script reports the missing
  // configuration above instead of failing on a driver it will never use.
  const { MongoClient } = await import("mongodb");
  const client = new MongoClient(env.MONGODB_URI);

  try {
    await client.connect();
    const collection = client.db().collection<StoredDoc>("conversations");

    const all = await collection.find({}).toArray();
    console.log(`${all.length} conversation(s) in the store\n`);

    let affected = 0;
    let repaired = 0;

    for (const doc of all) {
      const messages = doc.messages ?? [];
      let changedHere = 0;

      const cleaned = messages.map((m) => {
        // Only an assistant turn opening with an object can carry the leak, and
        // the check is cheap enough to run over every message.
        if (m.role !== "assistant" || !m.content?.startsWith("{")) return m;
        const text = stripLeakedAuxJson(m.content);
        if (text === m.content) return m;
        changedHere++;
        return { ...m, content: text };
      });

      if (changedHere === 0) continue;
      affected++;
      repaired += changedHere;

      const sample = messages.find(
        (m) => m.role === "assistant" && stripLeakedAuxJson(m.content) !== m.content,
      );
      console.log(
        `  ${doc._id}  ${changedHere} answer(s)  ${JSON.stringify(
          (sample?.content ?? "").slice(0, 70),
        )}`,
      );

      if (WRITE) {
        await collection.updateOne({ _id: doc._id }, { $set: { messages: cleaned } });
      }
    }

    console.log(
      `\n${repaired} answer(s) across ${affected} conversation(s) ` +
        (WRITE ? "rewritten." : "would be rewritten. Re-run with --write to apply."),
    );
  } finally {
    await client.close();
  }
}

void main();

export {};
