/**
 * Show how a document chunks: the context header each chunk is embedded with,
 * its page, and its size. The header is what the embedding actually sees, so
 * this is the fastest way to tell whether section structure survived ingestion.
 *
 * Usage: npx tsx scripts/chunk-inspect.ts [file]
 */
import { readFileSync } from "fs";
import { chunkDocument } from "../lib/ingest/chunk";

async function main() {
  const path = process.argv[2];
  const text = path
    ? readFileSync(path, "utf8")
    : `# Acme Corp Handbook

## 4. Expense Policy

### 4.1 Meals
Employees may expense meals up to 45 GBP per day when travelling. This limit applies per calendar day and covers breakfast, lunch and dinner combined. Alcohol is not reimbursable under any circumstances, and receipts under 10 GBP do not require itemisation provided the total is clearly legible on the receipt.

### 4.2 Travel
Flights over 6 hours may be booked in premium economy. All other flights must be booked in economy class regardless of destination or traveller seniority. Rail travel may be booked in first class where the ticket costs less than the equivalent standard-class flexible fare.

## 5. Submission

### 5.1 Deadlines
All expenses must be submitted within 30 days of being incurred. Submissions after this window require written director approval and a documented reason for the delay, which finance reviews monthly.

### 5.2 Contacts
The finance contact for expense queries is Dana Whitfield.`;

  const chunks = await chunkDocument(
    { text, pageCount: 1, pages: [text] },
    { source: "sample.md", documentTitle: "Acme Corp Handbook" },
  );

  console.log(`${chunks.length} chunk(s)\n`);
  for (const [i, c] of chunks.entries()) {
    const header = String(c.pageContent).split("\n")[0];
    const body = String(c.metadata.originalText);
    console.log(`${String(i + 1).padStart(2)}. ${header}`);
    console.log(`    page ${c.metadata.page} · section "${c.metadata.section}" · ${body.length} chars`);
    console.log(`    ${body.replace(/\s+/g, " ").slice(0, 90)}…\n`);
  }
}

void main();

export {};
