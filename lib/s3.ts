import {
  DeleteObjectCommand,
  GetObjectCommand,
  HeadBucketCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env, features } from "./config";

let clientSingleton: S3Client | null = null;

export function getS3Client(): S3Client {
  if (!clientSingleton) {
    clientSingleton = new S3Client({
      region: env.AWS_REGION,
      credentials:
        env.AWS_ACCESS_KEY_ID && env.AWS_SECRET_ACCESS_KEY
          ? {
              accessKeyId: env.AWS_ACCESS_KEY_ID,
              secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
            }
          : undefined,
    });
  }
  return clientSingleton;
}

export const BUCKET_NAME = env.AWS_S3_BUCKET_NAME ?? "";

function assertConfigured() {
  if (!features.s3) {
    throw new Error(
      "S3 is not configured. Set AWS_S3_BUCKET_NAME, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
    );
  }
}

/** Strip path separators and control characters so a filename can't escape the prefix. */
function sanitiseFileName(name: string): string {
  return name
    .replace(/[/\\]/g, "_")
    .replace(/[\x00-\x1f\x7f]/g, "")
    .replace(/\s+/g, "_")
    .slice(0, 200);
}

export interface UploadResult {
  key: string;
  url: string;
  bucket: string;
}

export async function uploadFileToS3(
  file: Buffer,
  fileName: string,
  contentType: string,
): Promise<UploadResult> {
  assertConfigured();
  const key = `uploads/${Date.now()}-${sanitiseFileName(fileName)}`;

  await getS3Client().send(
    new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: file,
      ContentType: contentType,
      ServerSideEncryption: "AES256",
      Metadata: { originalname: encodeURIComponent(fileName) },
    }),
  );

  return { key, url: await getPresignedUrl(key), bucket: BUCKET_NAME };
}

/**
 * Read an object straight into memory via the SDK.
 *
 * Ingestion uses this rather than fetching a presigned URL over HTTP: it avoids
 * a network round-trip through the public internet, cannot be affected by URL
 * expiry mid-ingest, and removes a whole class of redirect handling.
 */
export async function downloadFileFromS3(key: string): Promise<Buffer> {
  assertConfigured();
  const res = await getS3Client().send(
    new GetObjectCommand({ Bucket: BUCKET_NAME, Key: key }),
  );
  if (!res.Body) throw new Error(`S3 object has no body: ${key}`);
  const bytes = await res.Body.transformToByteArray();
  return Buffer.from(bytes);
}

export async function getPresignedUrl(key: string, expiresIn = 3600): Promise<string> {
  assertConfigured();
  return getSignedUrl(
    getS3Client(),
    new GetObjectCommand({ Bucket: BUCKET_NAME, Key: key }),
    { expiresIn },
  );
}

export async function deleteFileFromS3(key: string): Promise<void> {
  assertConfigured();
  await getS3Client().send(new DeleteObjectCommand({ Bucket: BUCKET_NAME, Key: key }));
}

export async function listUploads(prefix = "uploads/") {
  assertConfigured();
  const res = await getS3Client().send(
    new ListObjectsV2Command({ Bucket: BUCKET_NAME, Prefix: prefix, MaxKeys: 100 }),
  );
  return (res.Contents ?? []).map((o) => ({
    key: o.Key ?? "",
    size: o.Size ?? 0,
    lastModified: o.LastModified?.toISOString(),
  }));
}

export interface S3Health {
  reachable: boolean;
  /** Why the check failed, so /api/health is actionable rather than just false. */
  error?: string;
  hint?: string;
}

/** Map the common S3 failures to something a human can act on. */
function explain(error: unknown): { error: string; hint: string } {
  const err = error as { name?: string; message?: string; $metadata?: { httpStatusCode?: number } };
  const name = err.name ?? "UnknownError";
  const status = err.$metadata?.httpStatusCode;

  if (/InvalidAccessKeyId/i.test(name)) {
    return {
      error: `${name}: the access key does not exist in AWS`,
      hint: "AWS_ACCESS_KEY_ID is wrong or the key has been deleted. Generate a new access key for the IAM user.",
    };
  }
  if (/SignatureDoesNotMatch/i.test(name)) {
    return {
      error: name,
      hint: "AWS_SECRET_ACCESS_KEY does not match the access key ID.",
    };
  }
  if (/NoSuchBucket/i.test(name)) {
    return { error: name, hint: `Bucket "${BUCKET_NAME}" does not exist.` };
  }
  if (/PermanentRedirect|301/i.test(name) || status === 301) {
    return { error: name, hint: "The bucket lives in a different region — check AWS_REGION." };
  }
  if (status === 403) {
    return {
      error: `${name} (403)`,
      hint: "Credentials are rejected or lack s3:ListBucket on this bucket.",
    };
  }
  return { error: `${name}${status ? ` (${status})` : ""}`, hint: err.message?.slice(0, 160) ?? "" };
}

/** Verify credentials and bucket reachability, for /api/health. */
export async function s3Healthy(): Promise<S3Health> {
  if (!features.s3) {
    return {
      reachable: false,
      error: "not configured",
      hint: "Set AWS_S3_BUCKET_NAME, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
    };
  }
  try {
    await getS3Client().send(new HeadBucketCommand({ Bucket: BUCKET_NAME }));
    return { reachable: true };
  } catch {
    // HeadBucket flattens errors to a bare 403; ListObjectsV2 reports the real
    // cause, which is what makes the difference between "fix your key" and
    // "fix your bucket policy".
    try {
      await getS3Client().send(
        new ListObjectsV2Command({ Bucket: BUCKET_NAME, MaxKeys: 1 }),
      );
      return { reachable: true };
    } catch (detailed) {
      return { reachable: false, ...explain(detailed) };
    }
  }
}
