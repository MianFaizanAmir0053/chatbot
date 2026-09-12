/* ------------------------------------------------------------------ *
 * Transcript replay
 *
 * How much of a stored conversation is fed back into a thread whose checkpoint
 * is gone — after a restart, or when a conversation stored yesterday is
 * reopened. The graph has no record of it while the user is looking at the
 * whole transcript, so a follow-up like "and the second one?" has nothing to
 * resolve against unless the recent turns are replayed.
 * ------------------------------------------------------------------ */

/**
 * Turns replayed at most, regardless of size.
 *
 * The most recent turns carry nearly all of the referential weight that replay
 * exists to serve, so this is a recall/cost trade rather than a correctness one.
 */
export const MAX_REPLAYED_TURNS = 12;

/**
 * Total characters of transcript replayed.
 *
 * A turn count alone is the wrong budget, because turns are not the same size.
 * This system writes long answers — tables, audits, multi-section comparisons
 * running to several thousand characters each — so twelve of them can fill most
 * of a context window with history before the current question is read. The
 * cost lands on every turn of a resumed thread, not once.
 */
export const MAX_REPLAYED_CHARS = 12_000;

/**
 * Characters kept from any single replayed turn.
 *
 * Without a per-message cap, one long answer consumes the whole budget and
 * crowds out every turn after it — and the later turns are the ones carrying
 * the references. Truncation is marked, so the model treats what it sees as an
 * extract rather than as the whole of what it previously said.
 */
export const MAX_REPLAYED_MESSAGE_CHARS = 2_000;

export interface ReplayTurn {
  role: string;
  content: string;
}

/**
 * The tail of a transcript that fits the replay budget, oldest first.
 *
 * Walked newest-first because recency is what matters here, then reversed. A
 * turn that does not fit is dropped whole rather than cut down to the remaining
 * budget: half a sentence at the start of a transcript reads as corruption, and
 * whatever came before it is already gone.
 */
export function replayable(stored: ReplayTurn[]): ReplayTurn[] {
  const candidates = stored.slice(-MAX_REPLAYED_TURNS);
  const kept: ReplayTurn[] = [];
  let budget = MAX_REPLAYED_CHARS;

  for (let i = candidates.length - 1; i >= 0; i--) {
    const turn = candidates[i];
    const content =
      turn.content.length > MAX_REPLAYED_MESSAGE_CHARS
        ? `${turn.content.slice(0, MAX_REPLAYED_MESSAGE_CHARS)}\n\n[…truncated]`
        : turn.content;
    if (content.length > budget) break;
    budget -= content.length;
    kept.push({ role: turn.role, content });
  }

  return kept.reverse();
}
