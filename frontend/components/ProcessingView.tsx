'use client'

import { useEffect, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CheckCircle,
  Circle,
  Warning,
  UploadSimple,
  UserFocus,
  Crosshair,
  Lightning,
  VideoCamera,
  type Icon as PhosphorIcon,
} from '@phosphor-icons/react'
import { JobResult, JobStatus } from '@/lib/types'

interface StepDef {
  id: string
  label: string
  description: string
  Icon: PhosphorIcon
  activeStatuses: JobStatus[]
  completeStatuses: JobStatus[]
}

const STEPS: StepDef[] = [
  {
    id: 'upload',
    label: 'Upload',
    description: 'Video received and stored',
    Icon: UploadSimple,
    activeStatuses: ['uploading'],
    completeStatuses: ['detecting', 'pose', 'classifying', 'rendering', 'complete'],
  },
  {
    id: 'detection',
    label: 'Person Detection',
    description: 'YOLOv11m tracks two fighters',
    Icon: UserFocus,
    activeStatuses: ['detecting'],
    completeStatuses: ['pose', 'classifying', 'rendering', 'complete'],
  },
  {
    id: 'pose',
    label: 'Pose Estimation',
    description: '17 COCO keypoints per fighter',
    Icon: Crosshair,
    activeStatuses: ['pose'],
    completeStatuses: ['classifying', 'rendering', 'complete'],
  },
  {
    id: 'classify',
    label: 'Punch Classification',
    description: 'AttentionBiLSTM — 6 punch types',
    Icon: Lightning,
    activeStatuses: ['classifying'],
    completeStatuses: ['rendering', 'complete'],
  },
  {
    id: 'render',
    label: 'Render Output',
    description: 'Annotated video at 30fps',
    Icon: VideoCamera,
    activeStatuses: ['rendering'],
    completeStatuses: ['complete'],
  },
]

function getStepState(step: StepDef, status: JobStatus): 'pending' | 'active' | 'complete' | 'error' {
  if (status === 'error') return 'error'
  if (step.completeStatuses.includes(status)) return 'complete'
  if (step.activeStatuses.includes(status)) return 'active'
  return 'pending'
}

function ElapsedTimer({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    return () => clearInterval(interval)
  }, [startedAt])
  const m = Math.floor(elapsed / 60)
  const s = elapsed % 60
  return (
    <span className="font-mono text-xs text-zinc-500">
      {m > 0 ? `${m}m ` : ''}{s}s elapsed
    </span>
  )
}

export default function ProcessingView({ jobId }: { jobId: string }) {
  const router = useRouter()
  const [job, setJob] = useState<JobResult | null>(null)
  const [notFound, setNotFound] = useState(false)

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' })
      if (res.status === 404) { setNotFound(true); return }
      if (!res.ok) return
      const data: JobResult = await res.json()
      setJob(data)
      if (data.status === 'complete') {
        setTimeout(() => router.push(`/results/${jobId}`), 900)
      }
    } catch {}
  }, [jobId, router])

  useEffect(() => {
    poll()
    const interval = setInterval(poll, 1000)
    return () => clearInterval(interval)
  }, [poll])

  if (notFound) {
    return (
      <div className="text-center py-24">
        <Warning className="w-10 h-10 text-amber-500 mx-auto mb-4" weight="fill" />
        <p className="text-zinc-200 font-semibold mb-1">Job not found</p>
        <p className="text-zinc-500 text-sm mb-6">This job ID doesn't exist or has expired.</p>
        <a href="/" className="text-sm text-rose-400 hover:text-rose-300 transition-colors underline underline-offset-2">
          Start a new analysis
        </a>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-3 bg-zinc-800 rounded-full w-1/3" />
        <div className="h-2 bg-zinc-800 rounded-full" />
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-14 bg-zinc-900 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 200, damping: 28 }}
    >
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs font-mono uppercase tracking-widest text-zinc-500">
            Analysing fight footage
          </p>
          {job.startedAt && <ElapsedTimer startedAt={job.startedAt} />}
        </div>
        <h1 className="text-3xl font-semibold tracking-tighter text-zinc-100 mb-1 truncate">
          {job.originalFilename}
        </h1>

        {/* Progress bar */}
        <div className="mt-5">
          <div className="flex items-center justify-between text-xs mb-2">
            <span className="text-zinc-400">{job.currentStep}</span>
            <span className="font-mono text-zinc-500">{job.progress}%</span>
          </div>
          <div className="h-[3px] bg-zinc-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-rose-500 rounded-full origin-left"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: job.progress / 100 }}
              transition={{ type: 'spring', stiffness: 60, damping: 20 }}
              style={{ transformOrigin: 'left' }}
            />
          </div>
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-2">
        {STEPS.map((step, idx) => {
          const stepState = getStepState(step, job.status)
          const isActive = stepState === 'active'
          const isDone = stepState === 'complete'
          const isPending = stepState === 'pending'

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.06, type: 'spring', stiffness: 300, damping: 30 }}
              className={`relative flex items-center gap-4 p-4 rounded-xl border transition-all duration-300 ${
                isActive
                  ? 'border-rose-500/25 bg-rose-500/[0.06]'
                  : isDone
                  ? 'border-white/[0.04] bg-white/[0.02]'
                  : 'border-transparent bg-transparent'
              }`}
            >
              {/* Status icon */}
              <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-colors"
                style={{
                  background: isActive
                    ? 'rgba(225,29,72,0.12)'
                    : isDone
                    ? 'rgba(16,185,129,0.1)'
                    : 'rgba(255,255,255,0.03)',
                  border: isActive
                    ? '1px solid rgba(225,29,72,0.3)'
                    : isDone
                    ? '1px solid rgba(16,185,129,0.2)'
                    : '1px solid rgba(255,255,255,0.06)',
                }}
              >
                {isDone ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400" weight="fill" />
                ) : isActive ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  >
                    <Circle className="w-4 h-4 text-rose-400" weight="fill" />
                  </motion.div>
                ) : (
                  <step.Icon className="w-4 h-4 text-zinc-600" />
                )}
              </div>

              {/* Text */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className={`text-sm font-medium transition-colors ${
                    isActive ? 'text-zinc-100' : isDone ? 'text-zinc-300' : 'text-zinc-600'
                  }`}>
                    {step.label}
                  </p>
                  {isActive && (
                    <span className="text-[10px] font-mono text-rose-400 border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 rounded-full">
                      active
                    </span>
                  )}
                </div>
                <AnimatePresence mode="wait">
                  {isActive ? (
                    <motion.p
                      key="detail"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="text-xs text-zinc-400 mt-0.5 font-mono truncate"
                    >
                      {job.currentDetail}
                    </motion.p>
                  ) : (
                    <motion.p
                      key="default"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-xs text-zinc-600 mt-0.5"
                    >
                      {step.description}
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Complete state */}
      <AnimatePresence>
        {job.status === 'complete' && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 25 }}
            className="mt-6 p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] flex items-center gap-3"
          >
            <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" weight="fill" />
            <div>
              <p className="text-sm font-medium text-emerald-300">Analysis complete</p>
              <p className="text-xs text-zinc-400 mt-0.5">Redirecting to results...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      <AnimatePresence>
        {job.status === 'error' && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 rounded-xl border border-amber-500/20 bg-amber-500/[0.06] flex items-center gap-3"
          >
            <Warning className="w-5 h-5 text-amber-400 flex-shrink-0" weight="fill" />
            <div>
              <p className="text-sm font-medium text-amber-300">Pipeline failed</p>
              <p className="text-xs text-zinc-400 mt-0.5">{job.error || 'An unexpected error occurred.'}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
