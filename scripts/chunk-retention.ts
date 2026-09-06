/**
 * Assert that chunking loses no substantive text.
 *
 * Section merging and the minimum-length filter both discard input by design,
 * so this checks the thing that actually matters: every fact still appears in
 * some chunk. A retrieval system that silently drops a short section is worse
 * than one that chunks it badly — the answer becomes unfindable rather than
 * merely lower-ranked.
 */
import { chunkDocument } from "../lib/ingest/chunk";

const DOC = `# Acme Corp Handbook

## 4. Expense Policy

### 4.1 Meals
Employees may expense meals up to 45 GBP per day when travelling. Alcohol is not reimbursable. Receipts under 10 GBP do not require itemisation.

### 4.2 Travel
Flights over 6 hours may be booked in premium economy. All other flights must be booked in economy class regardless of destination.

## 5. Submission

### 5.1 Deadlines
All expenses must be submitted within 30 days of being incurred. Late submissions require director approval.

### 5.2 Contacts
The finance contact is Dana Whitfield.`;

const MUST_SURVIVE = [
  "45 GBP",
  "10 GBP",
  "6 hours",
  "economy class",
  "30 days",
  "director approval",
  "Dana Whitfield",
];

async function main() {
  const chunks = await chunkDocument(
    { text: DOC, pageCount: 1, pages: [DOC] },
    { source: "handbook.md", documentTitle: "Acme Corp Handbook" },
  );

  const corpus = chunks.map((c) => String(c.metadata.originalText)).join("\n");
  let lost = 0;

  console.log(`${chunks.length} chunk(s)\n`);
  for (const term of MUST_SURVIVE) {
    const kept = corpus.includes(term);
    if (!kept) lost++;
    console.log(`  ${kept ? "KEPT" : "LOST"}  ${term}`);
  }

  console.log(`\n${MUST_SURVIVE.length - lost}/${MUST_SURVIVE.length} retained`);
  if (lost > 0) process.exitCode = 1;
}

void main();

export {};
