import { NextRequest } from "next/server";
import { z } from "zod";
import {
  deleteConversation,
  forkConversation,
  getConversation,
  renameConversation,
} from "@/lib/conversations/store";
import { removeThreadDocuments } from "@/lib/ingest/pipeline";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ id: string }> };

export async function GET(_req: NextRequest, { params }: Context) {
  const { id } = await params;
  const conversation = await getConversation(id);
  if (!conversation) return Response.json({ error: "Not found" }, { status: 404 });
  return Response.json({ conversation });
}

const RenameSchema = z.object({
  // Bounded rather than free text: the title is rendered in the sidebar, the
  // dashboard and the document dialog, and an unbounded string breaks all three.
  title: z.string().min(1).max(200),
});

/** Rename a conversation. */
export async function PATCH(req: NextRequest, { params }: Context) {
  const { id } = await params;
  const parsed = RenameSchema.safeParse(await req.json().catch(() => ({})));

  if (!parsed.success) {
    return Response.json(
      { error: "Invalid request", details: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  const conversation = await renameConversation(id, parsed.data.title);
  if (!conversation) return Response.json({ error: "Not found" }, { status: 404 });
  return Response.json({ conversation: { id: conversation.id, title: conversation.title } });
}

const ForkSchema = z.object({
  /**
   * Fork the first N messages rather than the whole transcript, so a user can
   * branch from a point mid-thread and take a different line from there.
   */
  upTo: z.number().int().min(0).optional(),
  title: z.string().min(1).max(200).optional(),
});

/** Fork a conversation into a new one that inherits its documents. */
export async function POST(req: NextRequest, { params }: Context) {
  const { id } = await params;
  const parsed = ForkSchema.safeParse(await req.json().catch(() => ({})));

  if (!parsed.success) {
    return Response.json(
      { error: "Invalid request", details: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  const fork = await forkConversation(id, parsed.data);
  if (!fork) return Response.json({ error: "Not found" }, { status: 404 });

  // No documents are copied. The fork reads the parent's through `forkedFrom`,
  // which is what keeps forking instant instead of re-embedding every chunk.
  return Response.json({
    conversation: {
      id: fork.id,
      title: fork.title,
      forkedFrom: fork.forkedFrom,
      messages: fork.messages.length,
    },
  });
}

/**
 * Delete a conversation and the documents it owns.
 *
 * The embeddings go with it deliberately. Left behind they are unreachable
 * through any conversation while still counted in the corpus and still stored —
 * a leak that hides itself, since nothing can retrieve the orphans to reveal
 * they exist.
 *
 * Documents inherited from an ancestor are untouched: they belong to the
 * conversation that uploaded them, which may still exist.
 */
export async function DELETE(_req: NextRequest, { params }: Context) {
  const { id } = await params;

  let chunksRemoved = 0;
  try {
    chunksRemoved = await removeThreadDocuments(id);
  } catch (error) {
    // A vector store that is unreachable must not strand the conversation in
    // the list. The transcript is deleted either way and the orphaned chunks
    // are reported rather than silently assumed gone.
    console.error(`[conversations] could not purge documents for ${id}:`, error);
  }

  const removed = await deleteConversation(id);
  return Response.json({ removed, chunksRemoved });
}
