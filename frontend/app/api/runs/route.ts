import { NextResponse } from 'next/server'
import { s3GetJSON, s3ListRunIds, runKey } from '@/lib/s3'
import type { RunSummary } from '@/lib/types'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const runIds = await s3ListRunIds()
    const summaries = await Promise.all(
      runIds.map((id) => s3GetJSON<RunSummary>(runKey(id, 'summary.json'))),
    )
    const runs = summaries
      .filter((s): s is RunSummary => s !== null)
      .sort((a, b) => b.completedAt - a.completedAt)
    return NextResponse.json({ runs })
  } catch (err) {
    console.error('[api/runs] error:', err)
    return NextResponse.json(
      { error: 'Failed to list runs', detail: (err as Error).message },
      { status: 500 },
    )
  }
}
