import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  ListObjectsV2Command,
} from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import type { Readable } from 'stream'

export const S3_BUCKET =
  process.env.S3_BUCKET || 'tencount-runs-fyp-011190986707'
export const S3_REGION = process.env.S3_REGION || 'eu-north-1'

export const s3 = new S3Client({ region: S3_REGION })

export const SIGNED_URL_TTL = 60 * 60 * 6 // 6 hours

export function runKey(runId: string, file: string): string {
  return `runs/${runId}/${file}`
}

export async function s3PutBuffer(
  key: string,
  body: Buffer | Uint8Array | string,
  contentType: string,
): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
    }),
  )
}

export async function s3GetJSON<T>(key: string): Promise<T | null> {
  try {
    const res = await s3.send(
      new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }),
    )
    const stream = res.Body as Readable
    const chunks: Buffer[] = []
    for await (const c of stream) {
      chunks.push(typeof c === 'string' ? Buffer.from(c) : Buffer.from(c))
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf-8')) as T
  } catch (err: unknown) {
    const name = (err as { name?: string })?.name
    if (name === 'NoSuchKey' || name === 'NotFound') return null
    throw err
  }
}

export async function s3SignedUrl(key: string): Promise<string> {
  return getSignedUrl(
    s3,
    new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }),
    { expiresIn: SIGNED_URL_TTL },
  )
}

export async function s3ListRunIds(): Promise<string[]> {
  const out = await s3.send(
    new ListObjectsV2Command({
      Bucket: S3_BUCKET,
      Prefix: 'runs/',
      Delimiter: '/',
    }),
  )
  const prefixes = out.CommonPrefixes ?? []
  return prefixes
    .map((p) => p.Prefix ?? '')
    .map((p) => p.replace(/^runs\//, '').replace(/\/$/, ''))
    .filter(Boolean)
}
