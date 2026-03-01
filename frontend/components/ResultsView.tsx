'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Download,
  ArrowLeft,
  Play,
  Pause,
  SpeakerHigh,
  SpeakerSlash,
  Warning,
  CircleNotch,
  Trophy,
} from '@phosphor-icons/react'
import Link from 'next/link'
import { JobResult, PunchType, FighterResult } from '@/lib/types'

const PUNCH_LABELS: Record<PunchType, string> = {
  jab: 'Jab',
  cross: 'Cross',
  lead_hook: 'Lead Hook',
  rear_hook: 'Rear Hook',
  lead_uppercut: 'Lead Uppercut',
  rear_uppercut: 'Rear Uppercut',
}

const PUNCH_COLORS: Record<PunchType, string> = {
  jab: '#e11d48',
  cross: '#f97316',
  lead_hook: '#eab308',
  rear_hook: '#22c55e',
  lead_uppercut: '#3b82f6',
  rear_uppercut: '#a855f7',
}

// Isolated video player to avoid re-renders bubbling up
function VideoPlayer({ src, filename }: { src: string; filename: string }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentTime, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)

  const toggle = () => {
    const v = videoRef.current
    if (!v) return
    playing ? v.pause() : v.play()
  }

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    const v = videoRef.current
    if (!v) return
    const rect = e.currentTarget.getBoundingClientRect()
    v.currentTime = ((e.clientX - rect.left) / rect.width) * v.duration
  }

  return (
    <div className="rounded-2xl overflow-hidden border border-white/[0.06] bg-zinc-900">
      <div className="relative aspect-video bg-zinc-950">
        <video
          ref={videoRef}
          src={src}
          className="w-full h-full object-contain"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={() => {
            const v = videoRef.current
            if (!v) return
            setCurrent(v.currentTime)
            setProgress(v.currentTime / v.duration)
          }}
          onLoadedMetadata={() => {
            if (videoRef.current) setDuration(videoRef.current.duration)
          }}
          onClick={toggle}
        />
        {!playing && (
          <div
            className="absolute inset-0 flex items-center justify-center cursor-pointer bg-black/30"
            onClick={toggle}
          >
            <div className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center">
              <Play className="w-6 h-6 text-white ml-1" weight="fill" />
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="px-4 py-3 bg-zinc-900/80 border-t border-white/[0.05]">
        <div
          className="h-1 bg-zinc-700 rounded-full overflow-hidden cursor-pointer mb-3"
          onClick={seek}
        >
          <motion.div
            className="h-full bg-rose-500 rounded-full"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="w-8 h-8 rounded-lg bg-white/[0.05] hover:bg-white/[0.08] flex items-center justify-center transition-colors"
            >
              {playing ? (
                <Pause className="w-3.5 h-3.5" weight="fill" />
              ) : (
                <Play className="w-3.5 h-3.5 ml-px" weight="fill" />
              )}
            </button>
            <button
              onClick={() => {
                if (videoRef.current) videoRef.current.muted = !muted
                setMuted(!muted)
              }}
              className="w-8 h-8 rounded-lg bg-white/[0.05] hover:bg-white/[0.08] flex items-center justify-center transition-colors"
            >
              {muted ? (
                <SpeakerSlash className="w-3.5 h-3.5 text-zinc-400" />
              ) : (
                <SpeakerHigh className="w-3.5 h-3.5" />
              )}
            </button>
            <span className="text-xs font-mono text-zinc-400">
              {fmt(currentTime)} / {fmt(duration)}
            </span>
          </div>
          <a
            href={src}
            download={filename}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download
          </a>
        </div>
      </div>
    </div>
  )
}

// Animated punch breakdown bar
function PunchBar({
  label,
  value,
  max,
  color,
  delay,
}: {
  label: string
  value: number
  max: number
  color: string
  delay: number
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-xs font-mono text-zinc-300">{value}</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${max > 0 ? (value / max) * 100 : 0}%` }}
          transition={{ delay, type: 'spring', stiffness: 80, damping: 20 }}
        />
      </div>
    </div>
  )
}

// Fighter stats card
function FighterCard({
  fighter,
  label,
  isWinner,
  delay,
}: {
  fighter: FighterResult
  label: string
  isWinner: boolean
  delay: number
}) {
  const maxBreakdown = Math.max(...Object.values(fighter.breakdown))
  const punchTypes = Object.entries(fighter.breakdown) as [PunchType, number][]

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 200, damping: 28 }}
      className="relative p-6 rounded-2xl border bg-white/[0.02] overflow-hidden"
      style={{
        borderColor: isWinner ? 'rgba(225,29,72,0.25)' : 'rgba(255,255,255,0.06)',
      }}
    >
      {isWinner && (
        <div className="absolute top-4 right-4 flex items-center gap-1 text-[10px] font-mono text-rose-400 border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 rounded-full">
          <Trophy className="w-2.5 h-2.5" weight="fill" />
          more punches
        </div>
      )}

      <div className="mb-5">
        <p className="text-xs font-mono uppercase tracking-widest text-zinc-500 mb-1">{label}</p>
        <p className="text-5xl font-semibold font-mono tracking-tighter text-zinc-50">
          {fighter.totalPunches}
        </p>
        <p className="text-sm text-zinc-500 mt-1">total punches detected</p>
      </div>

      <div className="space-y-3">
        {punchTypes.map(([type, count], i) => (
          <PunchBar
            key={type}
            label={PUNCH_LABELS[type]}
            value={count}
            max={maxBreakdown}
            color={PUNCH_COLORS[type]}
            delay={delay + i * 0.05}
          />
        ))}
      </div>
    </motion.div>
  )
}

// Punch timeline SVG visualisation
function PunchTimeline({
  timeline,
  duration,
}: {
  timeline: NonNullable<JobResult['timeline']>
  duration: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, type: 'spring', stiffness: 200, damping: 28 }}
      className="p-6 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
    >
      <div className="flex items-center justify-between mb-5">
        <div>
          <p className="text-sm font-medium text-zinc-200">Punch Timeline</p>
          <p className="text-xs text-zinc-500 mt-0.5">{timeline.length} events over {duration.toFixed(1)}s</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span className="text-xs text-zinc-400">Fighter 1</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="text-xs text-zinc-400">Fighter 2</span>
          </div>
        </div>
      </div>

      <div className="relative">
        {/* Axis line */}
        <div className="absolute left-0 right-0 top-1/2 -translate-y-px h-px bg-zinc-800" />

        {/* Center separator */}
        <div className="h-16 relative overflow-hidden">
          {timeline.map((event, i) => {
            const x = (event.time / duration) * 100
            const isF1 = event.fighter === 1
            return (
              <motion.div
                key={i}
                className="absolute w-[3px] rounded-full"
                style={{
                  left: `${x}%`,
                  top: isF1 ? '8px' : '50%',
                  height: isF1 ? 'calc(50% - 10px)' : 'calc(50% - 10px)',
                  background: PUNCH_COLORS[event.type],
                  opacity: 0.75,
                }}
                initial={{ scaleY: 0, originY: isF1 ? 1 : 0 }}
                animate={{ scaleY: 1 }}
                transition={{
                  delay: 0.6 + i * 0.01,
                  type: 'spring',
                  stiffness: 200,
                  damping: 20,
                }}
                title={`${PUNCH_LABELS[event.type]} @ ${event.time.toFixed(2)}s`}
              />
            )
          })}
        </div>

        {/* Time labels */}
        <div className="flex justify-between mt-2">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <span key={t} className="text-[10px] font-mono text-zinc-600">
              {(t * duration).toFixed(0)}s
            </span>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-5 pt-4 border-t border-white/[0.05] flex flex-wrap gap-3">
        {(Object.keys(PUNCH_LABELS) as PunchType[]).map((type) => (
          <div key={type} className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: PUNCH_COLORS[type] }}
            />
            <span className="text-[11px] text-zinc-500">{PUNCH_LABELS[type]}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default function ResultsView({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`/api/jobs/${jobId}`, { cache: 'no-store' })
      .then((r) => {
        if (r.status === 404) throw new Error('Job not found')
        if (!r.ok) throw new Error('Failed to load results')
        return r.json()
      })
      .then((data: JobResult) => {
        if (data.status !== 'complete') {
          setError('Analysis is still in progress.')
        } else {
          setJob(data)
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [jobId])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <CircleNotch className="w-8 h-8 text-zinc-600 animate-spin mx-auto mb-3" />
          <p className="text-zinc-400 text-sm">Loading results...</p>
        </div>
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Warning className="w-8 h-8 text-amber-500 mx-auto mb-3" weight="fill" />
          <p className="text-zinc-200 font-medium mb-1">Could not load results</p>
          <p className="text-zinc-500 text-sm mb-5">{error}</p>
          <Link href="/" className="text-sm text-rose-400 hover:text-rose-300 transition-colors underline underline-offset-2">
            Start a new analysis
          </Link>
        </div>
      </div>
    )
  }

  const [f1, f2] = job.fighters!
  const winner = f1.totalPunches >= f2.totalPunches ? 1 : 2

  return (
    <div className="max-w-[1400px] mx-auto px-6 py-10">
      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="flex items-start justify-between mb-8 gap-4"
      >
        <div className="min-w-0">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors mb-3"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            New analysis
          </Link>
          <h1 className="text-2xl font-semibold tracking-tighter text-zinc-100 truncate">
            {job.originalFilename}
          </h1>
          <div className="flex items-center gap-4 mt-1.5">
            <span className="text-xs font-mono text-zinc-500">{job.fps}fps</span>
            <span className="text-xs font-mono text-zinc-500">{job.duration?.toFixed(1)}s</span>
            <span className="text-xs font-mono text-zinc-500">{job.frameCount?.toLocaleString()} frames</span>
            <span className="flex items-center gap-1 text-xs font-mono text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              complete
            </span>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.4fr_1fr] gap-6">
        {/* Left column */}
        <div className="space-y-6">
          {/* Video player */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 28 }}
          >
            <VideoPlayer
              src={job.videoUrl!}
              filename={`tencount_${jobId}_output.mp4`}
            />
            <p className="text-xs text-zinc-600 mt-2 font-mono">
              * Showing original upload. Annotated output available when connected to Python backend.
            </p>
          </motion.div>

          {/* Timeline */}
          {job.timeline && job.duration && (
            <PunchTimeline timeline={job.timeline} duration={job.duration} />
          )}
        </div>

        {/* Right column — fighter stats */}
        <div className="space-y-4">
          {/* Summary bar */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, type: 'spring', stiffness: 200, damping: 28 }}
            className="p-4 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
                Total punches
              </span>
              <span className="text-xs font-mono text-zinc-500">
                {f1.totalPunches + f2.totalPunches} combined
              </span>
            </div>
            <div className="flex h-3 rounded-full overflow-hidden gap-px">
              <motion.div
                className="bg-rose-500 rounded-l-full"
                initial={{ flex: 0 }}
                animate={{ flex: f1.totalPunches }}
                transition={{ delay: 0.2, type: 'spring', stiffness: 80, damping: 20 }}
              />
              <motion.div
                className="bg-blue-500 rounded-r-full"
                initial={{ flex: 0 }}
                animate={{ flex: f2.totalPunches }}
                transition={{ delay: 0.2, type: 'spring', stiffness: 80, damping: 20 }}
              />
            </div>
            <div className="flex justify-between mt-2">
              <span className="text-xs font-mono text-rose-400">
                F1 {Math.round((f1.totalPunches / (f1.totalPunches + f2.totalPunches)) * 100)}%
              </span>
              <span className="text-xs font-mono text-blue-400">
                F2 {Math.round((f2.totalPunches / (f1.totalPunches + f2.totalPunches)) * 100)}%
              </span>
            </div>
          </motion.div>

          <FighterCard
            fighter={f1}
            label="Fighter 1 — Red Corner"
            isWinner={winner === 1}
            delay={0.15}
          />
          <FighterCard
            fighter={f2}
            label="Fighter 2 — Blue Corner"
            isWinner={winner === 2}
            delay={0.25}
          />

          {/* Model info */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="p-4 rounded-xl border border-white/[0.04] bg-white/[0.01]"
          >
            <p className="text-xs font-mono uppercase tracking-widest text-zinc-600 mb-3">
              Pipeline
            </p>
            <div className="space-y-2">
              {[
                ['Detection', 'YOLOv11m — person_11s_50epochs'],
                ['Pose', 'YOLOv8m-pose — COCO 17kp'],
                ['Classifier', 'AttentionBiLSTM — 70.9% acc'],
                ['Input dim', '41-dim (34 coords + 7 angles)'],
                ['Cooldown', '20 frames — 0.67s'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs">
                  <span className="text-zinc-600">{k}</span>
                  <span className="text-zinc-400 font-mono text-right">{v}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
