'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  CircleNotch,
  Warning,
  FilmStrip,
  ArrowRight,
  Clock,
  Trophy,
} from '@phosphor-icons/react'
import type { RunSummary } from '@/lib/types'

function relativeTime(ts: number): string {
  const diff = Date.now() - ts
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  if (d < 30) return `${d}d ago`
  const mo = Math.floor(d / 30)
  return `${mo}mo ago`
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function HistoryList() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/runs', { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load history')
        return r.json()
      })
      .then((data: { runs: RunSummary[] }) => setRuns(data.runs))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <CircleNotch className="w-8 h-8 text-zinc-600 animate-spin mx-auto mb-3" />
          <p className="text-zinc-400 text-sm">Loading history...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Warning className="w-8 h-8 text-amber-500 mx-auto mb-3" weight="fill" />
          <p className="text-zinc-200 font-medium mb-1">Could not load history</p>
          <p className="text-zinc-500 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-[1400px] mx-auto px-6 py-10">
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
          <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
            Run History
          </span>
        </div>
        <h1 className="text-3xl font-semibold tracking-tighter text-zinc-100">
          Past analyses
        </h1>
        <p className="text-sm text-zinc-500 mt-1.5">
          {runs.length === 0
            ? 'No past runs yet — upload a video to get started.'
            : `${runs.length} archived ${runs.length === 1 ? 'run' : 'runs'} — stored in S3.`}
        </p>
      </motion.div>

      {runs.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl border border-dashed border-white/[0.08] bg-white/[0.01] p-16 flex flex-col items-center text-center"
        >
          <FilmStrip className="w-10 h-10 text-zinc-700 mb-4" />
          <p className="text-zinc-300 font-medium mb-1">Nothing here yet</p>
          <p className="text-zinc-500 text-sm mb-6 max-w-sm">
            Completed runs are stored automatically. Run an analysis and it will
            show up here.
          </p>
          <Link
            href="/analyze"
            className="inline-flex items-center gap-1.5 text-sm bg-rose-600 hover:bg-rose-500 text-white px-4 py-2 rounded-full transition-colors font-medium"
          >
            Start an analysis
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {runs.map((run, i) => {
            const winner =
              run.f1Punches === run.f2Punches
                ? 0
                : run.f1Punches > run.f2Punches
                  ? 1
                  : 2
            const pct1 =
              run.totalPunches > 0
                ? Math.round((run.f1Punches / run.totalPunches) * 100)
                : 50
            const pct2 = 100 - pct1
            return (
              <motion.div
                key={run.runId}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  delay: i * 0.04,
                  type: 'spring',
                  stiffness: 200,
                  damping: 28,
                }}
              >
                <Link
                  href={`/history/${run.runId}`}
                  className="block p-5 rounded-2xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/[0.10] transition-colors group"
                >
                  <div className="flex items-start justify-between gap-3 mb-4">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-zinc-100 truncate group-hover:text-white">
                        {run.originalFilename}
                      </p>
                      <div className="flex items-center gap-2 mt-1 text-xs font-mono text-zinc-500">
                        <Clock className="w-3 h-3" />
                        <span>{relativeTime(run.completedAt)}</span>
                        <span className="text-zinc-700">·</span>
                        <span>{formatDate(run.completedAt)}</span>
                      </div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-zinc-300 group-hover:translate-x-0.5 transition-all flex-shrink-0 mt-1" />
                  </div>

                  <div className="mb-3">
                    <div className="flex items-baseline justify-between mb-1.5">
                      <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
                        Total punches
                      </span>
                      <span className="text-2xl font-semibold font-mono tracking-tight text-zinc-100">
                        {run.totalPunches}
                      </span>
                    </div>
                    <div className="flex h-2 rounded-full overflow-hidden gap-px">
                      <div
                        className="bg-rose-500"
                        style={{ flex: run.f1Punches || 1 }}
                      />
                      <div
                        className="bg-blue-500"
                        style={{ flex: run.f2Punches || 1 }}
                      />
                    </div>
                    <div className="flex justify-between mt-1.5">
                      <span className="text-[11px] font-mono text-rose-400 flex items-center gap-1">
                        {winner === 1 && (
                          <Trophy className="w-2.5 h-2.5" weight="fill" />
                        )}
                        F1 · {run.f1Punches} ({pct1}%)
                      </span>
                      <span className="text-[11px] font-mono text-blue-400 flex items-center gap-1">
                        F2 · {run.f2Punches} ({pct2}%)
                        {winner === 2 && (
                          <Trophy className="w-2.5 h-2.5" weight="fill" />
                        )}
                      </span>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-white/[0.04] flex items-center gap-4 text-[11px] font-mono text-zinc-600">
                    {run.duration !== undefined && (
                      <span>{run.duration.toFixed(1)}s</span>
                    )}
                    {run.fps !== undefined && <span>{run.fps}fps</span>}
                    <span className="ml-auto text-zinc-700">
                      #{run.runId.slice(0, 8)}
                    </span>
                  </div>
                </Link>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}
