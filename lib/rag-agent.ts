import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { BaseLanguageModel } from "@langchain/core/language_models/base";
import { tool } from "@langchain/core/tools";
import { ChatCohere, CohereEmbeddings } from "@langchain/cohere";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import * as z from "zod";

let vectorStore: MemoryVectorStore | null = null;
let embeddings: CohereEmbeddings | null = null;

/**
 * Initialize the vector store and embeddings
 */
export async function initializeVectorStore() {
  if (vectorStore && embeddings) {
    return { vectorStore, embeddings };
  }

  embeddings = new CohereEmbeddings({
    apiKey: process.env.COHERE_API_KEY,
    model: "embed-v4.0",
  });

  vectorStore = new MemoryVectorStore(embeddings);

  return { vectorStore, embeddings };
}

/**
 * Add documents to the vector store
 */
export async function addDocumentsToVectorStore(documents: Document[]) {
  const { vectorStore } = await initializeVectorStore();
  
  // Header-aware splitting: detect sections and split on headings
  function splitWithHeadings(docs: Document[]): Document[] {
    const results: Document[] = [];
    const headingRegex = /^(#{1,6}\s+.*|[A-Z][A-Z0-9 .:\-]{3,}$|\d+(?:\.\d+)*\s+.*|[A-Za-z][\w\s-]*:\s*)$/;
    const safetyKeywords = [
      "safety",
      "pre-ride",
      "inspection",
      "t-clocs",
      "helmet",
      "protective gear",
      "brakes",
      "tires",
      "controls",
      "lights",
      "oil",
      "chain",
    ];

    for (const doc of docs) {
      const lines = doc.pageContent.split(/\r?\n/);
      let currentHeading = "General";
      let buffer: string[] = [];

      const flush = () => {
        if (buffer.length === 0) return;
        const text = buffer.join("\n").trim();
        if (!text) { buffer = []; return; }
        const lower = text.toLowerCase();
        const safetyBoost = safetyKeywords.reduce((acc, k) => acc + (lower.includes(k) ? 1 : 0), 0);
        results.push(
          new Document({
            pageContent: text,
            metadata: {
              ...doc.metadata,
              section: currentHeading,
              safetyBoost,
            },
          })
        );
        buffer = [];
      };

      for (const line of lines) {
        if (headingRegex.test(line.trim())) {
          // start a new section
          flush();
          currentHeading = line.trim();
        } else {
          buffer.push(line);
        }
      }
      flush();
    }
    return results;
  }

  const preliminarySplits = splitWithHeadings(documents);

  // Secondary character-based splitting to cap chunk length while respecting overlaps
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(preliminarySplits);
  await vectorStore.addDocuments(allSplits);

  return allSplits;
}

/**
 * Generate the best possible prompt using conversation history for context.
 * This helps disambiguate short/ambiguous user queries (e.g., "How to drive" → "How to ride a sports bike safely").
 */
export async function generateBestPrompt(
  userQuery: string,
  conversationHistory: Array<{ role: string; content: string }>
): Promise<string> {
  try {
    const llm = getLLM();

    const historyContext = conversationHistory
      .slice(-6) // Use last few turns for richer context
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n");

    const refinementPrompt = `You are a prompt refiner. Given a user's latest question and the prior conversation, rewrite the question so it is specific, contextual, and optimized for getting the best answer.

Conversation History:
${historyContext}

User's Latest Question: ${userQuery}

Rules:
- Make the prompt concise (max ~20 words)
- Inject relevant context from the conversation (e.g., the subject/topic/domain already discussed)
- Disambiguate pronouns or vague terms based on history
- Do NOT add meta text or explanations—return only the refined prompt
- If history is empty or unhelpful, return the user's original question

Refined Prompt:`;

    const response = await llm.invoke([new HumanMessage(refinementPrompt)]);
    const refined = String(response.content).trim();

    return refined.length > 0 ? refined : userQuery;
  } catch (error) {
    console.error("Error generating best prompt:", error);
    return userQuery;
  }
}

/**
 * Create a retrieval tool for the agent
 */
export async function createRetrievalTool() {
  const { vectorStore } = await initializeVectorStore();

  const retrieveSchema = z.object({
    query: z
      .string()
      .describe("The search query to retrieve relevant documents"),
  });

  const retrieve = tool(
    async ({ query }) => {
      console.log("\n🔍 RAG Query:", query);

      if (!vectorStore) {
        console.log("❌ No vector store available");
        return "[SYSTEM: NO_DOCUMENT_CONTEXT]";
      }

      // Increase top_k for better recall while preserving semantic similarity as primary ranking
      const withScores = await vectorStore.similaritySearchWithScore(query, 10);

      // Re-rank using safetyBoost only as a secondary tie-breaker (never overriding similarity relevance)
      const EPS = 1e-6; // tie tolerance
      const boosted = withScores
        .map(([doc, score]) => ({ doc, score }))
        .sort((a, b) => {
          if (Math.abs(b.score - a.score) > EPS) {
            // Primary: higher similarity score first
            return b.score - a.score;
          }
          // Tie-breaker: safetyBoost metadata
          const sa = Number(a.doc.metadata?.safetyBoost || 0);
          const sb = Number(b.doc.metadata?.safetyBoost || 0);
          return sb - sa;
        })
        .map((x) => x.doc)
        .slice(0, 5);

      if (boosted.length === 0) {
        console.log("❌ No relevant documents found");
        return "[SYSTEM: NO_DOCUMENT_CONTEXT]";
      }

      console.log(`✅ Retrieved ${boosted.length} documents (top re-ranked):`);
      boosted.forEach((doc, idx) => {
        console.log(`\n📄 Document ${idx + 1} [${doc.metadata.source || "unknown"}]:`);
        console.log(doc.pageContent.substring(0, 300) + (doc.pageContent.length > 300 ? "..." : ""));
      });

      const serialized = boosted
        .map(
          (doc) =>
            `Source: ${doc.metadata.source || "document"}\n${doc.pageContent}`
        )
        .join("\n\n---\n\n");

      return serialized;
    },
    {
      name: "retrieve",
      description:
        "Retrieve information from uploaded documents or indexed content based on a query",
      schema: retrieveSchema,
    }
  );

  return retrieve;
}

/**
 * Direct retrieval function (no agent) - returns retrieved context
 */
export async function directRetrieve(query: string): Promise<string> {
  const { vectorStore } = await initializeVectorStore();

  console.log("\n🔍 Direct RAG Query:", query);

  if (!vectorStore) {
    console.log("❌ No vector store available");
    return "";
  }

  const withScores = await vectorStore.similaritySearchWithScore(query, 5);

  if (withScores.length === 0) {
    console.log("❌ No relevant documents found");
    return "";
  }

  // Sort by similarity score
  const sorted = withScores
    .map(([doc, score]) => ({ doc, score }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.doc);

  console.log(`✅ Retrieved ${sorted.length} chunks:`);
  sorted.forEach((doc, idx) => {
    console.log(`\n📄 Chunk ${idx + 1} [${doc.metadata.source || "unknown"}]:`);
    console.log(doc.pageContent.substring(0, 300) + (doc.pageContent.length > 300 ? "..." : ""));
  });

  const serialized = sorted
    .map((doc) => `[Source: ${doc.metadata.source || "document"}]\n${doc.pageContent}`)
    .join("\n\n---\n\n");

  return serialized;
}

/**
 * Get the LLM model
 */
export function getLLM(): BaseLanguageModel {
  return new ChatCohere({
    model: "command-r7b-12-2024",
    apiKey: process.env.COHERE_API_KEY,
    temperature: 0.0, // Zero temperature to prevent any creative generation or hallucination
  });
}

/**
 * Check if vector store has documents
 */
export async function hasDocuments(): Promise<boolean> {
  const { vectorStore } = await initializeVectorStore();
  // Since MemoryVectorStore doesn't expose document count directly,
  // we'll do a simple search to verify it has data
  try {
    const results = await vectorStore.similaritySearch("test", 1);
    return results.length > 0;
  } catch {
    return false;
  }
}

// ===== Hybrid RAG with LangGraph =====

export const REFUSAL = "The manual does not specify this information.";

async function decideRetrieveOrRespond(question: string, history: Array<{ role: "user" | "assistant"; content: string }>): Promise<"retrieve" | "respond"> {
  const llm = getLLM();
  const hist = history.map((h) => `${h.role}: ${h.content}`).join("\n");
  const prompt = `You are deciding whether a question requires searching through uploaded documents.

RESPOND (without document search) if the question is:
- General knowledge (e.g., "What is the capital of France?", "How does photosynthesis work?")
- Conversational (e.g., "Hello", "How are you?", "Thank you")
- About common facts, definitions, or concepts
- Mathematical calculations or logic problems
- Programming or technical questions NOT about specific uploaded documents
- Personal opinions or creative tasks

RETRIEVE (search documents) if the question is:
- About specific document content, manuals, or uploaded materials
- References "the document", "the manual", "the text", "the file"
- About procedures, specifications, or details likely in technical documentation
- Continuation of a document-based conversation

Reply ONLY with either: RETRIEVE or RESPOND
No explanations.

History:
${hist}

Question:
${question}`;
  const res = await llm.invoke([new HumanMessage(prompt)]);
  const decision = String(res.content).trim().toUpperCase();
  return decision.includes("RESPOND") ? "respond" : "retrieve";
}

async function rewriteSearchQuery(question: string, retrievedPreview?: string): Promise<string> {
  const llm = getLLM();
  const prompt = `Rewrite the user's question into the best possible semantic search query for retrieval.\nReturn ONLY the query text.\n\nQuestion:\n${question}\n\nOptional retrieved preview (may be low-relevance):\n${retrievedPreview ?? ""}`;
  const res = await llm.invoke([new HumanMessage(prompt)]);
  return String(res.content).trim() || question;
}

export async function runHybridRag(question: string, history: Array<{ role: "user" | "assistant"; content: string }>): Promise<string> {
  const retrieveTool = await createRetrievalTool();

  // Decide whether to retrieve first
  const firstDecision = await decideRetrieveOrRespond(question, history);
  if (firstDecision === "respond") {
    // Direct answer without retrieval - use general knowledge
    const llm = getLLM();
    const historyContext = history
      .slice(-4)
      .map((h) => `${h.role}: ${h.content}`)
      .join("\n");
    
    const generalPrompt = `You are a helpful AI assistant. Answer the user's question naturally using your general knowledge.
Be concise, accurate, and friendly.

${historyContext ? `Recent conversation:\n${historyContext}\n\n` : ""}User question: ${question}`;
    
    const res = await llm.invoke([new HumanMessage(generalPrompt)]);
    const direct = String(res.content).trim();
    return direct || "I'm not sure how to answer that question.";
  }

  // Try retrieval → grade → optional rewrite → final generate
  let query = question;
  let ctx = String(await retrieveTool.invoke({ query })) || "";

  // Grade documents (simple heuristic). If too short, rewrite and retry once.
  const useful = ctx.trim().length > 200;
  if (!useful) {
    query = await rewriteSearchQuery(question, ctx.slice(0, 300));
    ctx = String(await retrieveTool.invoke({ query })) || ctx;
  }

  if (!ctx || ctx.trim() === "" || ctx.includes("[SYSTEM: NO_DOCUMENT_CONTEXT]")) {
    return REFUSAL;
  }

  // Strict answering with retrieved context
  const llm = getLLM();
  const systemPrompt = `You answer questions using ONLY the provided document excerpts.\n\nRULES:\n1. Read the DOCUMENT EXCERPTS below.\n2. If the EXACT answer to the question is written verbatim in the excerpts, copy it.\n3. If the exact answer is NOT written in the excerpts, respond ONLY with: "${REFUSAL}"\n\nFORBIDDEN:\n- Do NOT combine information from different excerpts.\n- Do NOT paraphrase or summarize.\n- Do NOT add any information not explicitly written.\n- Do NOT use your own knowledge.\n- Do NOT explain what the document does or doesn't say.\n- Do NOT say "the document mentions..." or "according to...".\n- Do NOT provide partial answers or related information.\n\nFOR YES/NO QUESTIONS:\n- If the document explicitly states the claim is true → answer "Yes." and quote the sentence.\n- If the document explicitly states the claim is false → answer "No." and quote the sentence.\n- If the document does NOT explicitly state whether the claim is true or false → answer ONLY: "${REFUSAL}"\n\nFOR COMPARISON QUESTIONS (more than, less than, better, worse):\n- The EXACT comparison must appear as ONE sentence in the document.\n- If you cannot find that exact comparison sentence → answer ONLY: "${REFUSAL}"\n\nRESPONSE FORMAT:\n- Maximum 1-2 sentences.\n- If answering, quote the relevant text.\n- If not found, respond ONLY with: "${REFUSAL}"\n\nDOCUMENT EXCERPTS:\n${ctx}\n\nEND OF EXCERPTS.`;

  const userPrompt = `Question: ${question}\n\nRemember: If the exact answer is not written in the document excerpts, respond ONLY with: "${REFUSAL}"`;
  const response = await llm.invoke([new SystemMessage(systemPrompt), new HumanMessage(userPrompt)]);
  const answer = String(response.content).trim();
  return answer || REFUSAL;
}
