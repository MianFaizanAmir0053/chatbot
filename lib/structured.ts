import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import type { z } from "zod";

/**
 * Provider-agnostic structured output.
 *
 * `withStructuredOutput` is the idiomatic path and works on DeepSeek and
 * OpenAI, but it throws "Input is not an AIMessageChunk" on ChatCohere — the
 * base implementation streams, and Cohere's response isn't a chunk. Since
 * Cohere is a viable provider here, structured extraction can't depend on it.
 *
 * So: try the native path, and on failure fall back to binding a single tool
 * and reading its validated arguments. Tool calling works on every provider we
 * support, which makes this the more portable mechanism of the two.
 */
/**
 * Keeps an auxiliary call's output out of LangGraph's token stream.
 *
 * Structured extraction is never something a user should read, but it is not
 * reliably invisible either. `withStructuredOutput` only produces a tool call on
 * legacy GPT models; for everything else — every provider in this registry — it
 * defaults to a `json_schema` response format, so the result arrives as ordinary
 * message *content*. A planner or judge invoked from inside a tool runs at the
 * root graph's own depth, which makes that content indistinguishable from the
 * supervisor's answer by nesting alone: the query planner's raw
 * `{"variants":[...]}` object was streamed to the user, written to the
 * transcript, and then scored for groundedness as if it were part of the reply.
 *
 * LangGraph's stream handler drops any model run carrying this tag, so applying
 * it here covers every call site — including ones added later, which is the
 * point of putting it in the shared helper rather than at each call.
 */
const NOSTREAM = { tags: ["langsmith:nostream"] };

export async function structuredInvoke<T extends z.ZodTypeAny>(
  model: BaseChatModel | BaseChatModel[],
  schema: T,
  messages: BaseMessage[],
  options: { name: string; description?: string },
): Promise<z.infer<T>> {
  const { name, description = `Return the result as ${name}` } = options;
  const candidates = Array.isArray(model) ? model : [model];

  if (candidates.length === 0) {
    throw new Error(`No model available for structured output (${name}).`);
  }

  let lastError: unknown;
  for (const candidate of candidates) {
    try {
      return await invokeOne(candidate, schema, messages, name, description);
    } catch (error) {
      lastError = error;
      // Try the next provider: a configured-but-dead key shouldn't disable the
      // guardrail when another provider is available.
      if (candidates.length > 1) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[structured] ${name} failed on a provider, trying next: ${message.slice(0, 140)}`);
      }
    }
  }
  throw lastError;
}

async function invokeOne<T extends z.ZodTypeAny>(
  model: BaseChatModel,
  schema: T,
  messages: BaseMessage[],
  name: string,
  description: string,
): Promise<z.infer<T>> {
  try {
    const structured = model.withStructuredOutput(schema, { name });
    return (await structured.invoke(messages, NOSTREAM)) as z.infer<T>;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // Anything else is a genuine failure (bad key, rate limit) and should surface.
    if (!/AIMessageChunk|only supports "functionCalling"/i.test(message)) throw error;

    return extractViaToolCall(model, schema, messages, name, description);
  }
}

/** Force a single tool call and validate its arguments against the schema. */
async function extractViaToolCall<T extends z.ZodTypeAny>(
  model: BaseChatModel,
  schema: T,
  messages: BaseMessage[],
  name: string,
  description: string,
): Promise<z.infer<T>> {
  const extractor = tool(async (input) => JSON.stringify(input), {
    name,
    description,
    schema,
  });

  if (typeof model.bindTools !== "function") {
    throw new Error(`Model does not support tool calling, required for structured output (${name}).`);
  }

  // Forcing the call is preferable, but ChatCohere rejects `tool_choice`
  // outright. Fall back to an unforced bind — with a single tool bound and an
  // explicit instruction, models reliably call it anyway.
  let response;
  try {
    response = await model
      .bindTools([extractor], { tool_choice: name })
      .invoke(messages, NOSTREAM);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!/tool_choice/i.test(message)) throw error;
    response = await model.bindTools([extractor]).invoke(messages, NOSTREAM);
  }

  const calls =
    (response as { tool_calls?: Array<{ name: string; args: unknown }> }).tool_calls ?? [];
  const match = calls.find((c) => c.name === name) ?? calls[0];

  if (!match) {
    throw new Error(`Model returned no tool call for structured output (${name}).`);
  }

  // Parse rather than cast: the model can still emit arguments that don't match.
  return schema.parse(match.args) as z.infer<T>;
}
