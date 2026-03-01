import { NextResponse } from 'next/server'
import { jobStore } from '@/lib/job-store'

export async function GET(
  _request: Request,
  { params }: { params: { jobId: string } }
) {
  const job = jobStore.get(params.jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }
  return NextResponse.json(job)
}
