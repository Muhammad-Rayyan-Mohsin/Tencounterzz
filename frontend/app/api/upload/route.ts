import { NextResponse } from 'next/server'
import { writeFile, mkdir, readFile } from 'fs/promises'
import path from 'path'
import { spawn } from 'child_process'
import { jobStore } from '@/lib/job-store'
import type {
  JobResult,
  PunchType,
  FighterResult,
  PunchEvent,
  HeatmapData,
  FighterZone,
  RunSummary,
} from '@/lib/types'
import { runKey, s3PutBuffer } from '@/lib/s3'

function generateId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

// FYP root is one level above the Next.js app (frontend/)
const FYP_ROOT   = path.join(process.cwd(), '..')
const RUNNER     = path.join(FYP_ROOT, 'frontend_runner.py')
const UPLOADS    = path.join(process.cwd(), 'public', 'uploads')

// Use PYTHON_BIN env var on deployed servers; fall back to macOS Xcode CLT locally.
const PYTHON     = process.env.PYTHON_BIN || '/Library/Developer/CommandLineTools/usr/bin/python3'

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

  // Per-fighter accumulators — keyed by RAW track ID from the Python pipeline.
  // YOLO can recycle track IDs (e.g. fighter 2 dropping out and reappearing
  // as fighter 7) so we collect everything raw and remap to canonical slots
  // 1/2 once we have enough data to decide which two tracks are the real fighters.
  const rawTotals: Record<number, number> = {}
  const rawBreakdown: Record<number, Record<PunchType, number>> = {}
  const rawLiveEvents: Array<{ time: number; fighter: number; type: PunchType }> = []
  let heatmapData: HeatmapData | null = null
  /** Authoritative slot-to-raw-id map from the Python heatmap pass, if any. */
  let pythonSlotMap: { 1: number[]; 2: number[] } | null = null
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

      if (!trimmed.startsWith('{')) {
        // Mirror non-JSON inference output (e.g. referee diagnostics, ffmpeg
        // warnings) to the Node console so the dev server log is the single
        // source of truth for debugging a run.
        if (trimmed) console.log('[inference]', trimmed.slice(0, 240))
        continue
      }

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
        const fighter = msg.fighter as number
        const type    = msg.type as PunchType
        // Approximate time from current progress
        const cur = jobStore.get(jobId)
        const approxTime = cur && cur.progress
          ? parseFloat(((cur.progress / 100) * (frameTotal / fps || 30)).toFixed(2))
          : 0
        rawLiveEvents.push({ time: approxTime, fighter, type })

      } else if (t === 'fighter_total') {
        rawTotals[msg.id as number] = msg.total as number

      } else if (t === 'breakdown') {
        const fighter = msg.fighter as number
        const type    = msg.type as PunchType
        const n       = msg.n as number
        if (!rawBreakdown[fighter]) rawBreakdown[fighter] = defaultBreakdown()
        rawBreakdown[fighter][type] = n

      } else if (t === 'output') {
        outputVideoAbsPath = msg.path as string
        update({
          status:        'rendering',
          progress:      95,
          currentStep:   'Rendering Output',
          currentDetail: 'H.264 annotated video encoded — finalising...',
        })

      } else if (t === 'heatmap') {
        const fighters = msg.fighters as Record<string, { grid: string; frames: number }>
        const dom = (msg.dominance ?? {}) as Record<string, unknown>
        const cc = (dom.center_control ?? {}) as Record<string, number>
        const rawIdMap = msg.slot_raw_ids as
          | { '1'?: number[]; '2'?: number[] }
          | undefined
        if (rawIdMap) {
          pythonSlotMap = {
            1: rawIdMap['1'] ?? [],
            2: rawIdMap['2'] ?? [],
          }
        }

        // Map absolute Python bg path to a public /uploads URL
        let bgUrl: string | undefined
        const rawBg = msg.bg_path as string | undefined
        if (rawBg) {
          bgUrl = `/uploads/${path.basename(rawBg)}`
        }

        heatmapData = {
          frameSize: msg.frame_size as [number, number],
          gridSize: msg.grid_size as [number, number],
          fighters,
          dominance: {
            fighter1Pct: dom.fighter1_pct as number | undefined,
            fighter2Pct: dom.fighter2_pct as number | undefined,
            contestedPct: dom.contested_pct as number | undefined,
            centerControl: cc,
            fighter1Zone: dom.fighter1_zone as FighterZone | undefined,
            fighter2Zone: dom.fighter2_zone as FighterZone | undefined,
            fighter1Centroid: dom.fighter1_centroid as [number, number] | undefined,
            fighter2Centroid: dom.fighter2_centroid as [number, number] | undefined,
          },
          bgUrl,
        }

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

    // ── Canonical fighter mapping ────────────────────────────────────────────
    // The Python tracker assigns raw integer IDs that can drift mid-fight (one
    // logical fighter can hop between IDs 2→3→5→7 as occlusions break tracking).
    // We collapse everything to two slots. The Python heatmap pass publishes
    // its slot→raw-ID map (`slot_raw_ids`); we honor it so punch counts and
    // heatmap data agree on which raw ID is "Fighter 1" vs "Fighter 2". When
    // Python's map is unavailable (no heatmap event), we fall back to ranking
    // raw IDs by punch count.
    type PunchCounts = Record<PunchType, number>
    const slotByRawId = new Map<number, 1 | 2>()

    if (pythonSlotMap) {
      for (const id of pythonSlotMap[1]) slotByRawId.set(id, 1)
      for (const id of pythonSlotMap[2]) slotByRawId.set(id, 2)
    } else {
      const allRawIds = new Set<number>([
        ...Object.keys(rawTotals).map(Number),
        ...Object.keys(rawBreakdown).map(Number),
        ...rawLiveEvents.map(e => e.fighter),
      ])
      const rankedIds = Array.from(allRawIds).sort(
        (a, b) => (rawTotals[b] ?? 0) - (rawTotals[a] ?? 0),
      )
      const anchorIds = rankedIds.slice(0, 2)
      anchorIds.forEach((id, i) => slotByRawId.set(id, (i + 1) as 1 | 2))
      for (const id of rankedIds.slice(2)) {
        if (anchorIds.length < 2) break
        const own = rawTotals[id] ?? 0
        const distA = Math.abs((rawTotals[anchorIds[0]] ?? 0) - own)
        const distB = Math.abs((rawTotals[anchorIds[1]] ?? 0) - own)
        slotByRawId.set(id, distA <= distB ? 1 : 2)
      }
    }

    const mergedTotals: Record<1 | 2, number> = { 1: 0, 2: 0 }
    const mergedBreakdown: Record<1 | 2, PunchCounts> = {
      1: defaultBreakdown(),
      2: defaultBreakdown(),
    }
    for (const [rawIdStr, count] of Object.entries(rawTotals)) {
      const slot = slotByRawId.get(Number(rawIdStr))
      if (slot) mergedTotals[slot] += count
    }
    for (const [rawIdStr, br] of Object.entries(rawBreakdown)) {
      const slot = slotByRawId.get(Number(rawIdStr))
      if (!slot) continue
      for (const pt of Object.keys(br) as PunchType[]) {
        mergedBreakdown[slot][pt] += br[pt] ?? 0
      }
    }

    const liveEvents: PunchEvent[] = rawLiveEvents
      .map((e) => {
        const slot = slotByRawId.get(e.fighter)
        return slot ? { time: e.time, fighter: slot, type: e.type } : null
      })
      .filter((e): e is PunchEvent => e !== null)

    const f1: FighterResult = {
      id: 1,
      totalPunches:
        mergedTotals[1] > 0
          ? mergedTotals[1]
          : liveEvents.filter((e) => e.fighter === 1).length,
      breakdown: mergedBreakdown[1],
    }
    const f2: FighterResult = {
      id: 2,
      totalPunches:
        mergedTotals[2] > 0
          ? mergedTotals[2]
          : liveEvents.filter((e) => e.fighter === 2).length,
      breakdown: mergedBreakdown[2],
    }

    // Use live events if we have enough; otherwise synthesise from breakdown
    const timeline: PunchEvent[] =
      liveEvents.length >= (f1.totalPunches + f2.totalPunches) * 0.8
        ? liveEvents.sort((a, b) => a.time - b.time)
        : generateSyntheticTimeline([f1, f2], videoDuration)

    const videoUrl = outputVideoAbsPath
      ? `/uploads/${path.basename(outputVideoAbsPath)}`
      : cur.videoUrl

    const completedAt = Date.now()
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
      heatmap:       heatmapData ?? undefined,
      completedAt,
    })

    // Archive to S3 — fire-and-forget so the response isn't blocked
    const finalJob = jobStore.get(jobId)
    if (finalJob) {
      archiveRunToS3(finalJob, outputVideoAbsPath, heatmapData).catch((err) => {
        console.error('[s3 archive] failed:', err)
      })
    }
  })

  proc.on('error', (err: Error) => {
    console.error('[inference spawn error]', err)
    update({ status: 'error', error: `Failed to start Python: ${err.message}` })
  })
}

// ── S3 archive ───────────────────────────────────────────────────────────────

async function archiveRunToS3(
  job: JobResult,
  videoAbsPath: string | null,
  heatmap: HeatmapData | null,
) {
  const runId = job.jobId

  // 1) Upload the annotated video
  let videoKey: string | undefined
  if (videoAbsPath) {
    const data = await readFile(videoAbsPath)
    videoKey = runKey(runId, 'output.mp4')
    await s3PutBuffer(videoKey, data, 'video/mp4')
  } else if (job.videoUrl) {
    // Fallback: the original uploaded video, in case the pipeline didn't emit a new one
    const local = path.join(UPLOADS, path.basename(job.videoUrl))
    try {
      const data = await readFile(local)
      videoKey = runKey(runId, 'output.mp4')
      await s3PutBuffer(videoKey, data, 'video/mp4')
    } catch {
      /* nothing to archive */
    }
  }

  // 2) Upload heatmap background, if present
  let bgKey: string | undefined
  if (heatmap?.bgUrl) {
    const bgLocal = path.join(UPLOADS, path.basename(heatmap.bgUrl))
    try {
      const data = await readFile(bgLocal)
      const ext = path.extname(bgLocal).toLowerCase()
      bgKey = runKey(runId, `heatmap_bg${ext || '.jpg'}`)
      const contentType =
        ext === '.png' ? 'image/png' :
        ext === '.webp' ? 'image/webp' :
        'image/jpeg'
      await s3PutBuffer(bgKey, data, contentType)
    } catch {
      /* heatmap bg optional */
    }
  }

  // 3) Build the archived JobResult — strip local URLs, store S3 keys instead
  const archivedHeatmap: HeatmapData | undefined = heatmap
    ? { ...heatmap, bgUrl: bgKey }
    : undefined

  const archived: JobResult = {
    ...job,
    videoUrl: videoKey,
    heatmap: archivedHeatmap,
  }

  // 4) Write the full result + a lightweight summary
  await s3PutBuffer(
    runKey(runId, 'full.json'),
    JSON.stringify(archived),
    'application/json',
  )

  const [f1, f2] = job.fighters ?? []
  const summary: RunSummary = {
    runId,
    originalFilename: job.originalFilename,
    completedAt: job.completedAt ?? Date.now(),
    duration: job.duration,
    fps: job.fps,
    totalPunches: (f1?.totalPunches ?? 0) + (f2?.totalPunches ?? 0),
    f1Punches: f1?.totalPunches ?? 0,
    f2Punches: f2?.totalPunches ?? 0,
  }
  await s3PutBuffer(
    runKey(runId, 'summary.json'),
    JSON.stringify(summary),
    'application/json',
  )
}
