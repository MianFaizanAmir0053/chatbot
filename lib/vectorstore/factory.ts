import { features } from "../config";
import { MemoryDriver } from "./memory";
import { QdrantDriver } from "./qdrant";
import type { VectorStoreDriver } from "./index";

let driver: VectorStoreDriver | null = null;
let probing: Promise<VectorStoreDriver> | null = null;

async function selectDriver(): Promise<VectorStoreDriver> {
  if (features.qdrant) {
    const qdrant = new QdrantDriver();
    if (await qdrant.healthy()) {
      console.log("[vectorstore] using Qdrant (persistent)");
      return qdrant;
    }
    console.warn(
      "[vectorstore] QDRANT_URL is set but unreachable — falling back to in-memory. " +
        "Documents will not survive a restart.",
    );
  } else {
    console.warn(
      "[vectorstore] QDRANT_URL not set — using in-memory store. " +
        "Set QDRANT_URL for persistent storage.",
    );
  }
  return new MemoryDriver();
}

/**
 * Resolve the active vector store driver.
 *
 * The health probe runs once per process and is shared between concurrent
 * callers, so a cold start under parallel load does not fan out into N probes.
 */
export async function getVectorStore(): Promise<VectorStoreDriver> {
  if (driver) return driver;
  if (!probing) {
    probing = selectDriver().then((d) => {
      driver = d;
      probing = null;
      return d;
    });
  }
  return probing;
}

/** Drop the cached driver so the next call re-probes. Used by tests and /api/health. */
export function resetVectorStore(): void {
  driver = null;
  probing = null;
}
