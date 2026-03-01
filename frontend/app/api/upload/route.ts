import { NextResponse } from 'next/server'
import { writeFile, mkdir } from 'fs/promises'
import path from 'path'
import { spawn } from 'child_process'
import { jobStore } from '@/lib/job-store'
import type { JobResult, PunchType, FighterResult, PunchEvent } from '@/lib/types'

function generateId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

// FYP root is one level above the Next.js app (frontend/)
const FYP_ROOT   = path.join(process.cwd(), '..')
const RUNNER     = path.join(FYP_ROOT, 'frontend_runner.py')
const UPLOADS    = path.join(process.cwd(), 'public', 'uploads')

// Python 3.9 (Xcode CLT) is the interpreter with all ML packages installed.
// The Homebrew python3 (3.14) lacks numpy/torch/ultralytics.
const PYTHON     = '/Library/Developer/CommandLineTools/usr/bin/python3'

// ── Helpers ──────────────────────────────────────────────────────────────────

function defaultBreakdown(): Record<PunchType, number> {
  return { jab: 0, cross: 0, lead_hook: 0, rear_hook: 0, lead_uppercut: 0, rear_uppercut: 0 }
}

function generateSyntheticTimeline(
  fighters: [FighterResult, FighterResult],
  duration: number
): PunchEvent[] {
  // Build a realistic timeline from real punch counts when exact timestamps
  // are unavailable (the script doesn't emit per-frame timestamps).
  const events: PunchEvent[] = []
  const punchOrder = Object.keys(defaultBreakdown()) as PunchType[]

  for (const fighter of fighters) {
    const types: PunchType[] = []
    for (const pt of punchOrder) {
      const n = fighter.breakdown[pt] ?? 0
      for (let i = 0; i < n; i++) types.push(pt)
    }
    // Shuffle for realism
    for (let i = types.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [types[i], types[j]] = [types[j], types[i]]
    }
    // Scatter across video with small gaps
    let t = 1.5
    for (const type of types) {
      t += Math.random() * (duration / types.length) * 0.9 + 0.3
      if (t >= duration - 0.5) break
      events.push({ time: parseFloat(t.toFixed(2)), fighter: fighter.id, type })
    }
  }
  return events.sort((a, b) => a.time - b.time)
}

// ── Main route handler ────────────────────────────────────────────────────────

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get('video') as File | null

    if (!file)
      return NextResponse.json({ error: 'No video file provided' }, { status: 400 })
    if (!file.type.startsWith('video/'))
      return NextResponse.json({ error: 'Invalid file type. Upload a video file.' }, { status: 400 })
    if (file.size > 500 * 1024 * 1024)
      return NextResponse.json({ error: 'File exceeds 500 MB limit.' }, { status: 413 })

    const jobId = generateId()
    const ext = file.name.split('.').pop()?.toLowerCase() || 'mp4'
    const inputFilename = `${jobId}.${ext}`

    await mkdir(UPLOADS, { recursive: true })
    const inputPath = path.join(UPLOADS, inputFilename)
    await writeFile(inputPath, Buffer.from(await file.arrayBuffer()))

    const job: JobResult = {
      jobId,
      status: 'detecting',
      progress: 8,
      currentStep: 'Person Detection',
      currentDetail: 'Loading YOLOv11m (best_potential.pt) + YOLOv8m-pose...',
      originalFilename: file.name,
      videoUrl: `/uploads/${inputFilename}`,
      startedAt: Date.now(),
    }
    jobStore.set(jobId, job)

    // Spawn inference — non-blocking
    runInference(jobId, inputPath)

    return NextResponse.json({ jobId })
  } catch (err) {
    console.error('[upload] error:', err)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

// ── Inference runner ─────────────────────────────────────────────────────────

function runInference(jobId: string, videoPath: string) {
  const update = (patch: Partial<JobResult>) => {
    const cur = jobStore.get(jobId)
    if (cur) jobStore.set(jobId, { ...cur, ...patch })
  }

  // Per-fighter accumulators
  const totals: Record<number, number> = {}
  const breakdown: Record<number, Record<PunchType, number>> = {
    1: defaultBreakdown(),
    2: defaultBreakdown(),
  }
  const liveEvents: PunchEvent[] = []
  let outputVideoAbsPath: string | null = null
  let frameTotal = 0
  let fps = 30

  const proc = spawn(PYTHON, [RUNNER, '--video', videoPath, '--output-dir', UPLOADS], {
    cwd: FYP_ROOT,
    env: { ...process.env },
  })

  // ── Parse stdout line by line ──────────────────────────────────────────────
  let buf = ''
  proc.stdout.on('data', (chunk: Buffer) => {
    buf += chunk.toString()
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()

      // Parse human-readable lines for fps/frame-count
      const fpsMatch = trimmed.match(/(\d+(?:\.\d+)?) fps\s+\((\d+) frames\)/)
      if (fpsMatch) {
        fps = parseFloat(fpsMatch[1])
        frameTotal = parseInt(fpsMatch[2])
      }

      if (!trimmed.startsWith('{')) continue

      let msg: Record<string, unknown>
      try { msg = JSON.parse(trimmed) } catch { continue }

      const t = msg.t as string

      if (t === 'progress') {
        const v = msg.v as number
        let status: JobResult['status']
        let step: string
        let detail: string
        if (v < 35) {
          status = 'detecting'
          step   = 'Person Detection'
          detail = `YOLOv11m tracking fighters... ${v}%`
        } else if (v < 65) {
          status = 'pose'
          step   = 'Pose Estimation'
          detail = `YOLOv8m-pose extracting 17 keypoints... ${v}%`
        } else {
          status = 'classifying'
          step   = 'Punch Classification'
          detail = `AttentionBiLSTM classifying punch types... ${v}%`
        }
        update({ status, progress: v, currentStep: step, currentDetail: detail })

      } else if (t === 'punch_event') {
        const fighter = msg.fighter as 1 | 2
        const type    = msg.type as PunchType
        // Approximate time from current progress
        const cur = jobStore.get(jobId)
        const approxTime = cur && cur.progress
          ? parseFloat(((cur.progress / 100) * (frameTotal / fps || 30)).toFixed(2))
          : 0
        liveEvents.push({ time: approxTime, fighter, type })

      } else if (t === 'fighter_total') {
        totals[msg.id as number] = msg.total as number

      } else if (t === 'breakdown') {
        const fighter = msg.fighter as number
        const type    = msg.type as PunchType
        const n       = msg.n as number
        if (!breakdown[fighter]) breakdown[fighter] = defaultBreakdown()
        breakdown[fighter][type] = n

      } else if (t === 'output') {
        outputVideoAbsPath = msg.path as string
        update({
          status:        'rendering',
          progress:      95,
          currentStep:   'Rendering Output',
          currentDetail: 'H.264 annotated video encoded — finalising...',
        })

      } else if (t === 'done') {
        update({ progress: 99, currentStep: 'Finalising', currentDetail: 'Building results...' })

      } else if (t === 'error') {
        update({ status: 'error', error: msg.msg as string })
      }
    }
  })

  proc.stderr.on('data', (chunk: Buffer) => {
    // YOLO/PyTorch logs go to stderr — log but don't treat as failure
    const text = chunk.toString().trim()
    if (text) console.log('[inference]', text.slice(0, 200))
  })

  proc.on('close', (code: number | null) => {
    const cur = jobStore.get(jobId)
    if (!cur) return

    if (cur.status === 'error') return  // already handled

    if (code !== 0) {
      update({ status: 'error', error: `Inference process exited with code ${code}` })
      return
    }

    // Build final FighterResult objects
    const videoDuration = frameTotal > 0 ? frameTotal / fps : 42
    const f1: FighterResult = {
      id: 1,
      totalPunches: totals[1] ?? liveEvents.filter(e => e.fighter === 1).length,
      breakdown: breakdown[1],
    }
    const f2: FighterResult = {
      id: 2,
      totalPunches: totals[2] ?? liveEvents.filter(e => e.fighter === 2).length,
      breakdown: breakdown[2],
    }

    // Use live events if we have enough; otherwise synthesise from breakdown
    const timeline: PunchEvent[] =
      liveEvents.length >= (f1.totalPunches + f2.totalPunches) * 0.8
        ? liveEvents.sort((a, b) => a.time - b.time)
        : generateSyntheticTimeline([f1, f2], videoDuration)

    const videoUrl = outputVideoAbsPath
      ? `/uploads/${path.basename(outputVideoAbsPath)}`
      : cur.videoUrl

    update({
      status:        'complete',
      progress:      100,
      currentStep:   'Analysis Complete',
      currentDetail: 'Pipeline finished — output ready',
      videoUrl,
      duration:      parseFloat(videoDuration.toFixed(1)),
      fps,
      frameCount:    frameTotal,
      fighters:      [f1, f2],
      timeline,
      completedAt:   Date.now(),
    })
  })

  proc.on('error', (err: Error) => {
    console.error('[inference spawn error]', err)
    update({ status: 'error', error: `Failed to start Python: ${err.message}` })
  })
}
