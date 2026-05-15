import { NextResponse } from 'next/server'
import { s3GetJSON, s3SignedUrl, runKey } from '@/lib/s3'
import type { JobResult } from '@/lib/types'

export const dynamic = 'force-dynamic'

export async function GET(
  _request: Request,
  { params }: { params: { runId: string } },
) {
  try {
    const full = await s3GetJSON<JobResult>(runKey(params.runId, 'full.json'))
    if (!full) {
      return NextResponse.json({ error: 'Run not found' }, { status: 404 })
    }

    // Replace S3 keys with presigned URLs the browser can fetch directly.
    const videoUrl = full.videoUrl ? await s3SignedUrl(full.videoUrl) : undefined
    const heatmap = full.heatmap
      ? {
          ...full.heatmap,
          bgUrl: full.heatmap.bgUrl
            ? await s3SignedUrl(full.heatmap.bgUrl)
            : undefined,
        }
      : undefined

    return NextResponse.json({ ...full, videoUrl, heatmap })
  } catch (err) {
    console.error('[api/runs/:id] error:', err)
    return NextResponse.json(
      { error: 'Failed to load run', detail: (err as Error).message },
      { status: 500 },
    )
  }
}
