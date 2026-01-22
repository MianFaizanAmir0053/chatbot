import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"

// Initialize S3 client
export const s3Client = new S3Client({
  region: process.env.AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
  },
})

export const BUCKET_NAME = process.env.AWS_S3_BUCKET_NAME || ""

/**
 * Upload a file to S3
 */
export async function uploadFileToS3(
  file: Buffer,
  fileName: string,
  contentType: string
): Promise<{ key: string; url: string }> {
  const key = `uploads/${Date.now()}-${fileName}`

  const command = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
    Body: file,
    ContentType: contentType,
  })

  await s3Client.send(command)

  // Generate a presigned URL (valid for 1 hour)
  const getCommand = new GetObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
  })
  
  const url = await getSignedUrl(s3Client, getCommand, { expiresIn: 3600 })

  return { key, url }
}

/**
 * Get a presigned URL for a file in S3
 */
export async function getPresignedUrl(key: string, expiresIn = 3600): Promise<string> {
  const command = new GetObjectCommand({
    Bucket: BUCKET_NAME,
    Key: key,
  })

  return await getSignedUrl(s3Client, command, { expiresIn })
}
