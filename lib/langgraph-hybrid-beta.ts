/* eslint-disable @typescript-eslint/no-explicit-any */
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  BaseMessage,
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { createRetrievalTool, getLLM, REFUSAL } from "./rag-agent";

// beta: fully working LangGraph hybrid RAG graph (not wired to live route)

// Define typed state schema via Annotation
const LGState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    default: () => [],
    reducer: (prev, next) => prev.concat(next),
  }),
  refinedQuery: Annotation<string | undefined>(),
  retrieved: Annotation<string | undefined>(),
});

// Build the graph
export async function buildLangGraphHybridBeta() {
  const retrieveTool = await createRetrievalTool();
  const tools = [retrieveTool];
  const toolNode = new ToolNode(tools);

  const shouldRetrieve = (state: any) => {
    const lastMessage = state.messages.at(-1);
    if (
      lastMessage &&
      AIMessage.isInstance(lastMessage) &&
      (lastMessage as any).tool_calls?.length
    ) {
      return "retrieve";
    }
    return END;
  };

  async function generateQueryOrRespond(state: any) {
    const lastHuman = [...state.messages]
      .reverse()
      .find((m: BaseMessage) => m instanceof HumanMessage);
    const query = state.refinedQuery ?? lastHuman?.content ?? "";
    const ai = new AIMessage({
      content: "",
      tool_calls: [
        {
          name: "retrieve",
          args: { query },
          id: `toolcall-retrieve-${Date.now()}`,
        },
      ],
    });
    return { messages: [ai] };
  }

  async function gradeDocuments(state: any) {
    const last = state.messages.at(-1);
    let retrieved = state.retrieved;
    if (last && ToolMessage.isInstance(last)) {
      retrieved = String(last.content ?? "");
    }
    const decision =
      retrieved && retrieved.trim().length > 200 ? "generate" : "rewrite";
    return { messages: [new AIMessage(decision)], retrieved };
  }

  async function rewrite(state: any) {
    const question =
      [...state.messages]
        .reverse()
        .find((m: BaseMessage) => m instanceof HumanMessage)?.content ?? "";
    const llm = getLLM();
    const res = await llm.invoke([
      new SystemMessage(
        "Rewrite the user's question into the best possible semantic search query. Return ONLY the query.",
      ),
      new HumanMessage(String(question)),
    ]);
    const refined = String(res.content).trim();
    return { refinedQuery: refined };
  }

  async function generate(state: any) {
    const llm = getLLM();
    const question =
      [...state.messages]
        .reverse()
        .find((m: BaseMessage) => m instanceof HumanMessage)?.content ?? "";
    const ctx = state.retrieved ?? "";
    const systemPrompt = `You answer questions using ONLY the provided document excerpts.\n\nRULES:\n1. Read the DOCUMENT EXCERPTS below.\n2. If the EXACT answer to the question is written verbatim in the excerpts, copy it.\n3. If the exact answer is NOT written in the excerpts, respond ONLY with: "${REFUSAL}"\n\nFORBIDDEN:\n- Do NOT combine information from different excerpts.\n- Do NOT paraphrase or summarize.\n- Do NOT add any information not explicitly written.\n- Do NOT use your own knowledge.\n- Do NOT explain what the document does or doesn't say.\n- Do NOT say "the document mentions..." or "according to...".\n- Do NOT provide partial answers or related information.\n\nFOR YES/NO QUESTIONS:\n- If the document explicitly states the claim is true → answer "Yes." and quote the sentence.\n- If the document explicitly states the claim is false → answer "No." and quote the sentence.\n- If the document does NOT explicitly state whether the claim is true or false → answer ONLY: "${REFUSAL}"\n\nFOR COMPARISON QUESTIONS (more than, less than, better, worse):\n- The EXACT comparison must appear as ONE sentence in the document.\n- If you cannot find that exact comparison sentence → answer ONLY: "${REFUSAL}"\n\nRESPONSE FORMAT:\n- Maximum 1-2 sentences.\n- If answering, quote the relevant text.\n- If not found, respond ONLY with: "${REFUSAL}"\n\nDOCUMENT EXCERPTS:\n${ctx}\n\nEND OF EXCERPTS.`;

    const userPrompt = `Question: ${question}\n\nRemember: If the exact answer is not written in the document excerpts, respond ONLY with: "${REFUSAL}"`;
    const response = await llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ]);
    const ai = new AIMessage(String(response.content).trim());
    return { messages: [ai] };
  }

  const builder = new StateGraph(LGState)
    .addNode("generateQueryOrRespond", generateQueryOrRespond)
    .addNode("retrieve", toolNode)
    .addNode("gradeDocuments", gradeDocuments)
    .addNode("rewrite", rewrite)
    .addNode("generate", generate)

    .addEdge(START, "generateQueryOrRespond")
    .addConditionalEdges("generateQueryOrRespond", shouldRetrieve as any)
    .addEdge("retrieve", "gradeDocuments")
    .addConditionalEdges("gradeDocuments", (state: any) => {
      const lastMessage = state.messages.at(-1);
      return lastMessage && String(lastMessage.content).trim() === "generate"
        ? "generate"
        : "rewrite";
    })
    .addEdge("generate", END)
    .addEdge("rewrite", "generateQueryOrRespond");

  return builder.compile();
}

export async function runLangGraphHybridBeta(
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
): Promise<string> {
  const graph = await buildLangGraphHybridBeta();
  const initialMessages: BaseMessage[] = [];
  for (const h of history ?? []) {
    if (h.role === "user") initialMessages.push(new HumanMessage(h.content));
    else initialMessages.push(new AIMessage(h.content));
  }
  initialMessages.push(new HumanMessage(question));
  const finalState = await graph.invoke({ messages: initialMessages });
  const last = finalState.messages?.at(-1);
  const answer = last ? String(last.content).trim() : REFUSAL;
  return answer || REFUSAL;
}
