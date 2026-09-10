/* ------------------------------------------------------------------ *
 * Which parts of a LangGraph stream are the answer
 *
 * A run's token stream carries every model call made anywhere inside it,
 * because the callback manager is inherited: the supervisor writing its reply,
 * delegated researchers, the query planner inside a retrieval tool, the
 * summariser, the moderation pass. Only the first of those is the answer.
 * Everything here exists to tell them apart.
 *
 * Extracted from the SSE route so the checks can be tested directly. A test
 * that reimplemented this logic would prove only that the copy agrees with
 * itself.
 * ------------------------------------------------------------------ */

/**
 * The graph node whose model call produces the answer.
 *
 * `createAgent` names it `model_request`. That constant is not exported, so it
 * is repeated here — deliberately as an allowlist rather than a list of nodes to
 * exclude, because the two mistakes are not equally bad. If this name ever goes
 * stale the answer simply stops streaming live and arrives whole from the update
 * stream; excluding known-bad nodes instead would let any newly added one splice
 * its working into the answer.
 */
export const ANSWER_NODE = "model_request";

/**
 * Pull the user-visible text out of a streamed chunk's content.
 *
 * `content` arrives either as a plain string or as an array of content blocks.
 * `createAgent` uses the block form, so matching on `typeof content === "string"`
 * silently discarded every token: the run completed, tools fired and the
 * groundedness judge scored an empty string as perfectly grounded, while the
 * client received no answer at all.
 *
 * Only `text` blocks are collected. Reasoning blocks are deliberately excluded —
 * they are not part of the answer and must not reach the transcript.
 */
export function visibleText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  let out = "";
  for (const block of content) {
    if (typeof block === "string") {
      out += block;
      continue;
    }
    const { type, text } = (block ?? {}) as { type?: string; text?: unknown };
    if (type === "text" && typeof text === "string") out += text;
  }
  return out;
}

/**
 * True when a streamed chunk came from a nested graph rather than the supervisor.
 *
 * LangGraph builds a nested run's checkpoint namespace by appending to its
 * parent's, separated by `|`, so depth is a structural property of the run
 * rather than anything a model or a tool controls. The root graph's own nodes
 * are at depth zero or one; a researcher invoked inside `delegate_research` is
 * deeper.
 *
 * Erring towards treating a chunk as nested is the safe direction: the worst
 * case is that a token is not streamed live, and the answer is still delivered
 * whole from the update stream. The opposite mistake puts a researcher's raw
 * working in front of the user as if it were the answer.
 */
export function isNested(meta: Record<string, unknown> | undefined): boolean {
  const ns = meta?.checkpoint_ns ?? meta?.langgraph_checkpoint_ns;
  if (typeof ns !== "string" || ns.length === 0) return false;
  return ns.split("|").filter(Boolean).length > 1;
}

/**
 * True when a streamed chunk is part of the answer the user is reading.
 *
 * Depth alone is not enough. Every model call made *inside a tool* — the query
 * planner, a summariser, a moderation pass — runs at the root graph's own depth,
 * so nesting cannot tell it apart from the supervisor writing its reply. That
 * gap put a query planner's raw `{"variants":[...],"hypotheticalAnswer":"..."}`
 * object in front of the user, then stored it in the transcript and scored it for
 * groundedness, because `withStructuredOutput` returns JSON as message *content*
 * on every provider configured here rather than as a tool call. The HyDE probe
 * inside it is a deliberately invented answer, so what the user saw looked
 * exactly like the system making facts up.
 *
 * Both tests therefore have to hold: the chunk comes from the root graph (not a
 * delegated researcher) *and* from the node that writes the answer (not a tool
 * or a middleware doing its own thinking).
 */
export function isAnswerChunk(meta: Record<string, unknown> | undefined): boolean {
  if (isNested(meta)) return false;
  return meta?.langgraph_node === ANSWER_NODE;
}
