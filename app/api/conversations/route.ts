import { listConversations } from "@/lib/conversations/store";

export const runtime = "nodejs";
// The list changes on every turn, so a cached response is always the wrong one.
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const conversations = await listConversations();
    return Response.json({ conversations });
  } catch (error) {
    console.error("[conversations] list failed:", error);
    return Response.json({ error: "Could not read conversations" }, { status: 500 });
  }
}
