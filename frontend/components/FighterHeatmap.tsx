'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Crosshair, MapTrifold, Scan, Target } from '@phosphor-icons/react'
import type { HeatmapData, FighterResult, FighterZone } from '@/lib/types'

interface Props {
  heatmap: HeatmapData
  fighter1: FighterResult
  fighter2: FighterResult
}

type TabKey = 'f1' | 'f2' | 'combined' | 'dominance'

interface TabDef {
  key: TabKey
  label: string
  short: string
  icon: typeof Crosshair
  /** Tailwind-equivalent CSS color for the active pill background */
  accent: string
  glow: string
}

const SPRING = { type: 'spring' as const, stiffness: 200, damping: 28 }
const SPRING_SOFT = { type: 'spring' as const, stiffness: 80, damping: 20 }

const ROSE_500 = '#f43f5e'
const BLUE_500 = '#3b82f6'

// ---------------------------------------------------------------------------
// Grid decoding & color ramps
// ---------------------------------------------------------------------------

/** Decode base64 → Uint8Array of length w*h (row-major). */
function decodeGrid(b64: string, expectedLen: number): Uint8Array | null {
  if (typeof window === 'undefined' || !b64) return null
  try {
    const bin = window.atob(b64)
    const len = Math.min(bin.length, expectedLen)
    const arr = new Uint8Array(expectedLen)
    for (let i = 0; i < len; i++) arr[i] = bin.charCodeAt(i) & 0xff
    return arr
  } catch {
    return null
  }
}

/** Clamp helper. */
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** Linear interpolate between two RGB tuples. */
function lerpRGB(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/**
 * Multi-stop ramp evaluation. Stops are [position 0..1, [r,g,b]].
 * Smooth interpolation between consecutive stops.
 */
function evalRamp(
  stops: Array<[number, [number, number, number]]>,
  t: number,
): [number, number, number] {
  if (t <= stops[0][0]) return stops[0][1]
  if (t >= stops[stops.length - 1][0]) return stops[stops.length - 1][1]
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i]
    const [p1, c1] = stops[i + 1]
    if (t >= p0 && t <= p1) {
      const local = (t - p0) / (p1 - p0)
      return lerpRGB(c0, c1, local)
    }
  }
  return stops[stops.length - 1][1]
}

// Rose ramp: deep ember → rose-700 → rose-500 → rose-300 → near-white tip
const ROSE_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [40, 6, 22]],
  [0.18, [136, 19, 55]], // rose-800
  [0.42, [225, 29, 72]], // rose-600
  [0.7, [251, 113, 133]], // rose-400
  [0.88, [254, 205, 211]], // rose-200
  [1.0, [255, 245, 247]],
]

// Blue ramp: deep ink → blue-700 → blue-500 → blue-300 → near-white tip
const BLUE_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [6, 18, 40]],
  [0.18, [29, 78, 216]], // blue-700
  [0.42, [37, 99, 235]], // blue-600
  [0.7, [96, 165, 250]], // blue-400
  [0.88, [191, 219, 254]], // blue-200
  [1.0, [240, 249, 255]],
]

function rampRose(t: number): [number, number, number, number] {
  const tc = clamp(t, 0, 1)
  const [r, g, b] = evalRamp(ROSE_STOPS, tc)
  const a = Math.round(Math.pow(tc, 0.55) * 255)
  return [Math.round(r), Math.round(g), Math.round(b), a]
}

function rampBlue(t: number): [number, number, number, number] {
  const tc = clamp(t, 0, 1)
  const [r, g, b] = evalRamp(BLUE_STOPS, tc)
  const a = Math.round(Math.pow(tc, 0.55) * 255)
  return [Math.round(r), Math.round(g), Math.round(b), a]
}

/**
 * Dominance ramp: signed t in [-1, 1].
 * t=+1 → fully F1 (rose), t=-1 → fully F2 (blue), t=0 → desaturated contested.
 */
function rampDominance(t: number, magnitude: number): [number, number, number, number] {
  const a = Math.round(clamp(Math.pow(magnitude, 0.5), 0, 1) * 220)
  if (Math.abs(t) < 0.001) {
    return [200, 200, 210, Math.max(40, Math.round(a * 0.4))]
  }
  if (t > 0) {
    const [r, g, b] = evalRamp(ROSE_STOPS, 0.3 + Math.abs(t) * 0.65)
    return [Math.round(r), Math.round(g), Math.round(b), a]
  }
  const [r, g, b] = evalRamp(BLUE_STOPS, 0.3 + Math.abs(t) * 0.65)
  return [Math.round(r), Math.round(g), Math.round(b), a]
}

// ---------------------------------------------------------------------------
// Offscreen heatmap rendering (writes to an ImageData at grid resolution)
// ---------------------------------------------------------------------------

type RampFn = (t: number) => [number, number, number, number]

function paintFighterImageData(
  imageData: ImageData,
  grid: Uint8Array | null,
  ramp: RampFn,
) {
  const data = imageData.data
  if (!grid) {
    data.fill(0)
    return
  }
  // Find max for normalization, with a floor so tiny grids don't blow up
  let max = 1
  for (let i = 0; i < grid.length; i++) if (grid[i] > max) max = grid[i]
  const invMax = 1 / max
  for (let i = 0; i < grid.length; i++) {
    const t = grid[i] * invMax
    const [r, g, b, a] = ramp(t)
    const o = i * 4
    data[o] = r
    data[o + 1] = g
    data[o + 2] = b
    data[o + 3] = a
  }
}

function gridMax(g: Uint8Array | null): number {
  if (!g) return 1
  let m = 1
  for (let i = 0; i < g.length; i++) if (g[i] > m) m = g[i]
  return m
}

function paintDominanceImageData(
  imageData: ImageData,
  g1: Uint8Array | null,
  g2: Uint8Array | null,
) {
  const data = imageData.data
  if (!g1 && !g2) {
    data.fill(0)
    return
  }
  // Normalize each grid independently so each fighter's "presence" is on the same scale
  const max1 = gridMax(g1)
  const max2 = gridMax(g2)
  const len = Math.max(g1?.length ?? 0, g2?.length ?? 0)
  for (let i = 0; i < len; i++) {
    const v1 = g1 ? g1[i] / max1 : 0
    const v2 = g2 ? g2[i] / max2 : 0
    const diff = v1 - v2 // [-1, 1] signed
    const mag = Math.max(v1, v2) // overall presence
    const [r, g, b, a] = rampDominance(diff, mag)
    const o = i * 4
    data[o] = r
    data[o + 1] = g
    data[o + 2] = b
    data[o + 3] = a
  }
}

// ---------------------------------------------------------------------------
// Main canvas — draws backdrop, heatmap layers, overlays
// ---------------------------------------------------------------------------

interface DrawArgs {
  ctx: CanvasRenderingContext2D
  width: number
  height: number
  gridW: number
  gridH: number
  tab: TabKey
  grid1: Uint8Array | null
  grid2: Uint8Array | null
  bgImg: HTMLImageElement | null
  centroid1?: [number, number]
  centroid2?: [number, number]
  zone1?: FighterZone
  zone2?: FighterZone
  /** Animation progress 0→1 for heatmap reveal */
  reveal: number
}

/** Build a dark gradient backdrop when no image is available. */
function drawFallbackBackdrop(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const g = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, Math.hypot(w, h) * 0.6)
  g.addColorStop(0, '#0c0c10')
  g.addColorStop(1, '#050507')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, w, h)
  // Subtle grid lines to evoke a tactical overlay
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.025)'
  ctx.lineWidth = 1
  const step = Math.max(40, Math.floor(w / 16))
  for (let x = step; x < w; x += step) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, h)
    ctx.stroke()
  }
  for (let y = step; y < h; y += step) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
    ctx.stroke()
  }
  ctx.restore()
}

/** Draw zone separators (LEFT / CENTER / RIGHT thirds). */
function drawZoneGuides(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 6])
  ctx.beginPath()
  ctx.moveTo(w * 0.35, 0)
  ctx.lineTo(w * 0.35, h)
  ctx.moveTo(w * 0.65, 0)
  ctx.lineTo(w * 0.65, h)
  ctx.stroke()
  ctx.setLineDash([])
  // Zone labels (top edge)
  ctx.fillStyle = 'rgba(255,255,255,0.25)'
  ctx.font = '10px ui-monospace, "SF Mono", Menlo, monospace'
  ctx.textBaseline = 'top'
  ctx.textAlign = 'center'
  ctx.fillText('LEFT', w * 0.175, 8)
  ctx.fillText('CENTER', w * 0.5, 8)
  ctx.fillText('RIGHT', w * 0.825, 8)
  ctx.restore()
}

/** Draw a centroid marker — outer ring, inner dot, glow. */
function drawCentroid(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  color: string,
  label: string,
  reveal: number,
) {
  const scale = clamp(reveal * 1.2, 0, 1)
  if (scale <= 0) return
  ctx.save()
  // Soft glow
  ctx.shadowColor = color
  ctx.shadowBlur = 18 * scale
  // Outer ring
  ctx.strokeStyle = 'rgba(255,255,255,0.9)'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.arc(cx, cy, 9 * scale, 0, Math.PI * 2)
  ctx.stroke()
  // Inner filled dot
  ctx.shadowBlur = 24 * scale
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(cx, cy, 5.5 * scale, 0, Math.PI * 2)
  ctx.fill()
  // Crosshair tick
  ctx.shadowBlur = 0
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'
  ctx.lineWidth = 1
  const r = 14 * scale
  ctx.beginPath()
  ctx.moveTo(cx - r, cy)
  ctx.lineTo(cx - 11 * scale, cy)
  ctx.moveTo(cx + 11 * scale, cy)
  ctx.lineTo(cx + r, cy)
  ctx.moveTo(cx, cy - r)
  ctx.lineTo(cx, cy - 11 * scale)
  ctx.moveTo(cx, cy + 11 * scale)
  ctx.lineTo(cx, cy + r)
  ctx.stroke()
  // Label pill
  if (reveal > 0.8) {
    ctx.font = '10px ui-monospace, "SF Mono", Menlo, monospace'
    const w = ctx.measureText(label).width + 12
    const lx = cx + 14 * scale
    const ly = cy - 9 * scale
    ctx.fillStyle = 'rgba(8,8,10,0.85)'
    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.beginPath()
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(lx, ly, w, 18, 6)
    } else {
      ctx.rect(lx, ly, w, 18)
    }
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, lx + 6, ly + 9)
  }
  ctx.restore()
}

/** Subtle vignette to push focus toward the center. */
function drawVignette(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.save()
  const grad = ctx.createRadialGradient(
    w * 0.5,
    h * 0.5,
    Math.min(w, h) * 0.35,
    w * 0.5,
    h * 0.5,
    Math.hypot(w, h) * 0.7,
  )
  grad.addColorStop(0, 'rgba(0,0,0,0)')
  grad.addColorStop(1, 'rgba(0,0,0,0.55)')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, w, h)
  ctx.restore()
}

/** Corner crop ticks — fight-arena framing. */
function drawCornerTicks(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth = 1.5
  const len = 14
  const pad = 10
  const corners: Array<[number, number, number, number, number, number]> = [
    [pad, pad, pad + len, pad, pad, pad + len], // top-left
    [w - pad, pad, w - pad - len, pad, w - pad, pad + len], // top-right
    [pad, h - pad, pad + len, h - pad, pad, h - pad - len], // bottom-left
    [w - pad, h - pad, w - pad - len, h - pad, w - pad, h - pad - len], // bottom-right
  ]
  corners.forEach(([sx, sy, ax, ay, bx, by]) => {
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(sx, sy)
    ctx.lineTo(bx, by)
    ctx.stroke()
  })
  ctx.restore()
}

/** Main draw routine. */
function drawScene(args: DrawArgs) {
  const {
    ctx,
    width,
    height,
    gridW,
    gridH,
    tab,
    grid1,
    grid2,
    bgImg,
    centroid1,
    centroid2,
    zone1,
    zone2,
    reveal,
  } = args

  ctx.save()
  ctx.clearRect(0, 0, width, height)

  // 1) Backdrop layer
  if (bgImg && bgImg.complete && bgImg.naturalWidth > 0) {
    // Cover-fit
    const ir = bgImg.naturalWidth / bgImg.naturalHeight
    const cr = width / height
    let dw = width
    let dh = height
    let dx = 0
    let dy = 0
    if (ir > cr) {
      dh = height
      dw = height * ir
      dx = (width - dw) / 2
    } else {
      dw = width
      dh = width / ir
      dy = (height - dh) / 2
    }
    ctx.globalAlpha = 0.45
    ctx.filter = 'grayscale(40%) brightness(0.55) contrast(1.05)'
    ctx.drawImage(bgImg, dx, dy, dw, dh)
    ctx.filter = 'none'
    ctx.globalAlpha = 1
    // Darken
    ctx.fillStyle = 'rgba(6,6,10,0.45)'
    ctx.fillRect(0, 0, width, height)
  } else {
    drawFallbackBackdrop(ctx, width, height)
  }

  // 2) Zone guides on the backdrop
  drawZoneGuides(ctx, width, height)

  // 3) Heatmap rendering — use an offscreen canvas at grid resolution for bilinear upscale
  const off = document.createElement('canvas')
  off.width = gridW
  off.height = gridH
  const offCtx = off.getContext('2d')
  if (!offCtx) {
    ctx.restore()
    return
  }
  const imageData = offCtx.createImageData(gridW, gridH)

  ctx.globalAlpha = clamp(reveal, 0, 1)

  if (tab === 'f1' && grid1) {
    paintFighterImageData(imageData, grid1, rampRose)
    offCtx.putImageData(imageData, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.globalCompositeOperation = 'lighter'
    ctx.drawImage(off, 0, 0, width, height)
  } else if (tab === 'f2' && grid2) {
    paintFighterImageData(imageData, grid2, rampBlue)
    offCtx.putImageData(imageData, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.globalCompositeOperation = 'lighter'
    ctx.drawImage(off, 0, 0, width, height)
  } else if (tab === 'combined') {
    if (grid2) {
      paintFighterImageData(imageData, grid2, rampBlue)
      offCtx.putImageData(imageData, 0, 0)
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.globalCompositeOperation = 'lighter'
      ctx.drawImage(off, 0, 0, width, height)
    }
    if (grid1) {
      const id2 = offCtx.createImageData(gridW, gridH)
      paintFighterImageData(id2, grid1, rampRose)
      offCtx.putImageData(id2, 0, 0)
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.globalCompositeOperation = 'lighter'
      ctx.drawImage(off, 0, 0, width, height)
    }
  } else if (tab === 'dominance') {
    paintDominanceImageData(imageData, grid1, grid2)
    offCtx.putImageData(imageData, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    // Normal blend feels better for diverging colors — `lighter` washes out red+blue
    ctx.globalCompositeOperation = 'screen'
    ctx.drawImage(off, 0, 0, width, height)
  }

  ctx.globalCompositeOperation = 'source-over'
  ctx.globalAlpha = 1

  // 4) Vignette to deepen edges
  drawVignette(ctx, width, height)

  // 5) Corner ticks
  drawCornerTicks(ctx, width, height)

  // 6) Centroid overlays — only for tabs where they make sense
  const drawF1 = (tab === 'f1' || tab === 'combined' || tab === 'dominance') && centroid1
  const drawF2 = (tab === 'f2' || tab === 'combined' || tab === 'dominance') && centroid2

  if (drawF1 && centroid1) {
    const cx = centroid1[0] * width
    const cy = centroid1[1] * height
    const lbl = `F1${zone1 ? ' · ' + zone1.toUpperCase() : ''}`
    drawCentroid(ctx, cx, cy, ROSE_500, lbl, reveal)
  }
  if (drawF2 && centroid2) {
    const cx = centroid2[0] * width
    const cy = centroid2[1] * height
    const lbl = `F2${zone2 ? ' · ' + zone2.toUpperCase() : ''}`
    drawCentroid(ctx, cx, cy, BLUE_500, lbl, reveal)
  }

  ctx.restore()
}

// ---------------------------------------------------------------------------
// Right-rail metric cards
// ---------------------------------------------------------------------------

function DominanceCard({
  f1Pct,
  f2Pct,
  contestedPct,
}: {
  f1Pct?: number
  f2Pct?: number
  contestedPct?: number
}) {
  const hasData =
    typeof f1Pct === 'number' || typeof f2Pct === 'number' || typeof contestedPct === 'number'

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, ...SPRING }}
      className="p-4 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
          Dominance split
        </span>
        <span className="text-xs font-mono text-zinc-600">cell-wise winner</span>
      </div>

      {hasData ? (
        <>
          <div className="flex h-3 rounded-full overflow-hidden gap-px bg-zinc-900">
            <motion.div
              className="bg-rose-500"
              initial={{ flex: 0 }}
              animate={{ flex: f1Pct ?? 0 }}
              transition={{ delay: 0.25, ...SPRING_SOFT }}
              style={{ borderTopLeftRadius: 9999, borderBottomLeftRadius: 9999 }}
            />
            <motion.div
              className="bg-zinc-600/60"
              initial={{ flex: 0 }}
              animate={{ flex: contestedPct ?? 0 }}
              transition={{ delay: 0.3, ...SPRING_SOFT }}
            />
            <motion.div
              className="bg-blue-500"
              initial={{ flex: 0 }}
              animate={{ flex: f2Pct ?? 0 }}
              transition={{ delay: 0.35, ...SPRING_SOFT }}
              style={{ borderTopRightRadius: 9999, borderBottomRightRadius: 9999 }}
            />
          </div>
          <div className="grid grid-cols-3 gap-1 mt-2.5 text-[10px] font-mono">
            <div className="flex items-center gap-1.5 text-rose-400">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              <span className="tabular-nums">F1 {Math.round(f1Pct ?? 0)}%</span>
            </div>
            <div className="flex items-center justify-center gap-1.5 text-zinc-500">
              <span className="w-1.5 h-1.5 rounded-full bg-zinc-500/70" />
              <span className="tabular-nums">contested {Math.round(contestedPct ?? 0)}%</span>
            </div>
            <div className="flex items-center justify-end gap-1.5 text-blue-400">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              <span className="tabular-nums">F2 {Math.round(f2Pct ?? 0)}%</span>
            </div>
          </div>
        </>
      ) : (
        <p className="text-xs text-zinc-500">
          Insufficient overlap — single fighter detected.
        </p>
      )}
    </motion.div>
  )
}

function CenterControlCard({ c1, c2 }: { c1?: number; c2?: number }) {
  const v1 = typeof c1 === 'number' ? clamp(c1, 0, 100) : 0
  const v2 = typeof c2 === 'number' ? clamp(c2, 0, 100) : 0
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, ...SPRING }}
      className="p-4 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
          Center control
        </span>
        <span className="text-xs font-mono text-zinc-600">% time in mid-zone</span>
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-xs text-rose-400">Fighter 1</span>
            <span className="text-xs font-mono text-zinc-300 tabular-nums">
              {v1.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-rose-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${v1}%` }}
              transition={{ delay: 0.3, ...SPRING_SOFT }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-xs text-blue-400">Fighter 2</span>
            <span className="text-xs font-mono text-zinc-300 tabular-nums">
              {v2.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${v2}%` }}
              transition={{ delay: 0.35, ...SPRING_SOFT }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function ZoneCard({
  zone1,
  zone2,
  centroid1,
  centroid2,
}: {
  zone1?: FighterZone
  zone2?: FighterZone
  centroid1?: [number, number]
  centroid2?: [number, number]
}) {
  const zones: FighterZone[] = ['left', 'center', 'right']
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25, ...SPRING }}
      className="p-4 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
          Zone distribution
        </span>
        <span className="text-xs font-mono text-zinc-600">L · C · R</span>
      </div>

      <div className="grid grid-cols-3 gap-1.5 mb-3">
        {zones.map((z) => {
          const f1Here = zone1 === z
          const f2Here = zone2 === z
          return (
            <div
              key={z}
              className="rounded-lg border border-white/[0.05] bg-zinc-900/40 px-2 py-2.5 flex flex-col items-center justify-between min-h-[56px]"
            >
              <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                {z}
              </span>
              <div className="flex items-center gap-1 mt-1.5">
                {f1Here && (
                  <span className="text-[9px] font-mono text-rose-400 border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 rounded">
                    F1
                  </span>
                )}
                {f2Here && (
                  <span className="text-[9px] font-mono text-blue-400 border border-blue-500/30 bg-blue-500/10 px-1.5 py-0.5 rounded">
                    F2
                  </span>
                )}
                {!f1Here && !f2Here && (
                  <span className="text-[9px] font-mono text-zinc-600">—</span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Mini strip showing centroid x-positions */}
      <div className="relative h-2 rounded-full bg-zinc-900 overflow-visible">
        <div className="absolute left-1/3 top-0 bottom-0 w-px bg-white/10" />
        <div className="absolute left-2/3 top-0 bottom-0 w-px bg-white/10" />
        {centroid1 && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.5, ...SPRING }}
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full"
            style={{
              left: `${clamp(centroid1[0], 0, 1) * 100}%`,
              background: ROSE_500,
              boxShadow: `0 0 8px ${ROSE_500}`,
            }}
            aria-hidden
          />
        )}
        {centroid2 && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.55, ...SPRING }}
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full"
            style={{
              left: `${clamp(centroid2[0], 0, 1) * 100}%`,
              background: BLUE_500,
              boxShadow: `0 0 8px ${BLUE_500}`,
            }}
            aria-hidden
          />
        )}
      </div>
      <div className="flex justify-between mt-1.5 text-[9px] font-mono text-zinc-600 uppercase tracking-wider">
        <span>left edge</span>
        <span>right edge</span>
      </div>
    </motion.div>
  )
}

function SampleStatsCard({ frames1, frames2 }: { frames1: number; frames2: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, ...SPRING }}
      className="p-4 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
          Sample size
        </span>
        <span className="text-xs font-mono text-zinc-600">frames analyzed</span>
      </div>
      <div className="space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
            Fighter 1
          </span>
          <span className="text-base font-mono font-semibold text-zinc-100 tabular-nums">
            {frames1.toLocaleString()}
          </span>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            Fighter 2
          </span>
          <span className="text-base font-mono font-semibold text-zinc-100 tabular-nums">
            {frames2.toLocaleString()}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Tab strip with sliding active pill
// ---------------------------------------------------------------------------

function TabStrip({
  tabs,
  active,
  onChange,
}: {
  tabs: TabDef[]
  active: TabKey
  onChange: (k: TabKey) => void
}) {
  return (
    <div
      className="inline-flex items-center gap-1 p-1 rounded-full border border-white/[0.06] bg-zinc-950/60 backdrop-blur"
      role="tablist"
      aria-label="Heatmap view"
    >
      {tabs.map((t) => {
        const isActive = t.key === active
        const Icon = t.icon
        return (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={isActive}
            aria-label={t.label}
            onClick={() => onChange(t.key)}
            className={
              'relative inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider transition-colors outline-none ' +
              'focus-visible:ring-2 focus-visible:ring-white/30 focus-visible:ring-offset-0 ' +
              (isActive ? 'text-white' : 'text-zinc-400 hover:text-zinc-200')
            }
          >
            {isActive && (
              <motion.span
                layoutId="heatmap-tab-active"
                className="absolute inset-0 rounded-full"
                style={{
                  background: t.accent,
                  boxShadow: `0 0 24px -6px ${t.glow}`,
                }}
                transition={{ type: 'spring', stiffness: 400, damping: 32 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-1.5">
              <Icon className="w-3 h-3" weight={isActive ? 'fill' : 'regular'} />
              <span className="hidden sm:inline">{t.label}</span>
              <span className="sm:hidden">{t.short}</span>
            </span>
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Backdrop image preloader (separate from canvas so React stays in control)
// ---------------------------------------------------------------------------

function useBackdropImage(url: string | undefined) {
  const [img, setImg] = useState<HTMLImageElement | null>(null)

  useEffect(() => {
    if (!url || typeof window === 'undefined') {
      setImg(null)
      return
    }
    const i = new window.Image()
    let cancelled = false
    i.onload = () => {
      if (!cancelled) setImg(i)
    }
    i.onerror = () => {
      if (!cancelled) setImg(null)
    }
    i.src = url
    return () => {
      cancelled = true
    }
  }, [url])

  return img
}

// ---------------------------------------------------------------------------
// Main section
// ---------------------------------------------------------------------------

export default function FighterHeatmapSection({ heatmap, fighter1, fighter2 }: Props) {
  const [gridW, gridH] = heatmap.gridSize
  const [frameW, frameH] = heatmap.frameSize

  const fighter1Heat = heatmap.fighters?.['1']
  const fighter2Heat = heatmap.fighters?.['2']

  // Decode grids once per heatmap payload
  const grid1 = useMemo(
    () => (fighter1Heat ? decodeGrid(fighter1Heat.grid, gridW * gridH) : null),
    [fighter1Heat, gridW, gridH],
  )
  const grid2 = useMemo(
    () => (fighter2Heat ? decodeGrid(fighter2Heat.grid, gridW * gridH) : null),
    [fighter2Heat, gridW, gridH],
  )

  // Build available tabs
  const tabs = useMemo<TabDef[]>(() => {
    const all: TabDef[] = []
    if (fighter1Heat) {
      all.push({
        key: 'f1',
        label: 'Fighter 1',
        short: 'F1',
        icon: Target,
        accent: 'linear-gradient(135deg, #e11d48 0%, #fb7185 100%)',
        glow: 'rgba(244,63,94,0.55)',
      })
    }
    if (fighter2Heat) {
      all.push({
        key: 'f2',
        label: 'Fighter 2',
        short: 'F2',
        icon: Target,
        accent: 'linear-gradient(135deg, #1d4ed8 0%, #60a5fa 100%)',
        glow: 'rgba(59,130,246,0.55)',
      })
    }
    if (fighter1Heat && fighter2Heat) {
      all.push({
        key: 'combined',
        label: 'Combined',
        short: 'Both',
        icon: MapTrifold,
        accent: 'linear-gradient(135deg, #6d28d9 0%, #db2777 50%, #2563eb 100%)',
        glow: 'rgba(168,85,247,0.5)',
      })
      all.push({
        key: 'dominance',
        label: 'Dominance',
        short: 'Dom',
        icon: Scan,
        accent: 'linear-gradient(135deg, #f43f5e 0%, #a3a3a3 50%, #3b82f6 100%)',
        glow: 'rgba(244,63,94,0.45)',
      })
    }
    return all
  }, [fighter1Heat, fighter2Heat])

  // Default tab — prefer combined, else first available
  const defaultTab: TabKey = useMemo(() => {
    if (fighter1Heat && fighter2Heat) return 'combined'
    if (fighter1Heat) return 'f1'
    if (fighter2Heat) return 'f2'
    return 'combined'
  }, [fighter1Heat, fighter2Heat])

  const [tab, setTab] = useState<TabKey>(defaultTab)

  // If the default tab changes (heatmap reload) and current tab is not available, reset.
  useEffect(() => {
    if (!tabs.some((t) => t.key === tab)) {
      setTab(defaultTab)
    }
  }, [tabs, tab, defaultTab])

  const bgImg = useBackdropImage(heatmap.bgUrl)

  // Canvas + ResizeObserver — render only when needed
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [canvasSize, setCanvasSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 })

  const aspect = frameW > 0 && frameH > 0 ? frameW / frameH : 16 / 9

  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof window === 'undefined') return
    const update = () => {
      const rect = el.getBoundingClientRect()
      const w = Math.max(0, Math.round(rect.width))
      const h = Math.max(0, Math.round(w / aspect))
      setCanvasSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }))
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [aspect])

  // Reveal animation per tab change
  const [reveal, setReveal] = useState(0)
  useEffect(() => {
    setReveal(0)
    let raf = 0
    const start = performance.now()
    const dur = 700
    const tick = (now: number) => {
      const t = clamp((now - start) / dur, 0, 1)
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3)
      setReveal(eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [tab])

  // Draw whenever inputs change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || canvasSize.w === 0 || canvasSize.h === 0) return
    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1
    canvas.width = canvasSize.w * dpr
    canvas.height = canvasSize.h * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawScene({
      ctx,
      width: canvasSize.w,
      height: canvasSize.h,
      gridW,
      gridH,
      tab,
      grid1,
      grid2,
      bgImg,
      centroid1: heatmap.dominance.fighter1Centroid,
      centroid2: heatmap.dominance.fighter2Centroid,
      zone1: heatmap.dominance.fighter1Zone,
      zone2: heatmap.dominance.fighter2Zone,
      reveal,
    })
  }, [
    canvasSize.w,
    canvasSize.h,
    gridW,
    gridH,
    tab,
    grid1,
    grid2,
    bgImg,
    heatmap.dominance.fighter1Centroid,
    heatmap.dominance.fighter2Centroid,
    heatmap.dominance.fighter1Zone,
    heatmap.dominance.fighter2Zone,
    reveal,
  ])

  // Dominance metrics — safe pulls
  const f1Pct = heatmap.dominance.fighter1Pct
  const f2Pct = heatmap.dominance.fighter2Pct
  const contestedPct = heatmap.dominance.contestedPct
  const centerControl = heatmap.dominance.centerControl ?? {}

  // Subtitle text — adapts to current tab
  const subtitle = useMemo(() => {
    switch (tab) {
      case 'f1':
        return 'Where Fighter 1 occupied the frame, weighted by time-on-screen.'
      case 'f2':
        return 'Where Fighter 2 occupied the frame, weighted by time-on-screen.'
      case 'combined':
        return 'Both fighters layered — additive blend reveals shared territory.'
      case 'dominance':
        return 'Per-cell winner: rose where F1 led, blue where F2 led, dim where contested.'
    }
  }, [tab])

  // Fail-safe: no heatmap data at all
  if (!fighter1Heat && !fighter2Heat) {
    return (
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={SPRING}
        className="p-6 rounded-2xl border border-white/[0.06] bg-white/[0.02]"
      >
        <div className="flex items-center gap-2 mb-2">
          <Crosshair className="w-4 h-4 text-zinc-500" />
          <h2 className="text-sm font-medium text-zinc-200">Spatial Dominance</h2>
        </div>
        <p className="text-xs text-zinc-500">
          Heatmap data unavailable for this fight — fighters were not tracked long enough to
          build a spatial distribution.
        </p>
      </motion.section>
    )
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING}
      className="relative rounded-2xl border border-white/[0.06] bg-white/[0.02] overflow-hidden"
    >
      {/* Section header */}
      <div className="px-6 pt-6 pb-5 flex items-start justify-between gap-4 flex-wrap border-b border-white/[0.04]">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <Crosshair className="w-3.5 h-3.5 text-rose-400" weight="bold" />
            <span className="text-[10px] font-mono uppercase tracking-[0.25em] text-zinc-500">
              Spatial · Positional control
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-semibold tracking-tighter text-zinc-50">
            Spatial Dominance
          </h2>
          <AnimatePresence mode="wait">
            <motion.p
              key={tab}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25 }}
              className="text-xs sm:text-sm text-zinc-500 mt-1 max-w-xl"
            >
              {subtitle}
            </motion.p>
          </AnimatePresence>
        </div>

        <TabStrip tabs={tabs} active={tab} onChange={setTab} />
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 xl:grid-cols-[1.6fr_1fr] gap-5 p-5 sm:p-6">
        {/* MAIN CANVAS */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1, ...SPRING }}
          className="relative rounded-2xl border border-white/[0.06] bg-zinc-950 overflow-hidden"
        >
          <div
            ref={containerRef}
            className="relative w-full"
            style={{ aspectRatio: `${frameW} / ${frameH}` }}
            role="img"
            aria-label={`Spatial heatmap visualization showing ${
              tab === 'f1'
                ? 'Fighter 1'
                : tab === 'f2'
                  ? 'Fighter 2'
                  : tab === 'combined'
                    ? 'both fighters'
                    : 'dominance per cell'
            }`}
          >
            <canvas
              ref={canvasRef}
              style={{
                display: 'block',
                width: '100%',
                height: '100%',
              }}
            />
            {/* Top-left intensity scale */}
            <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-zinc-950/70 backdrop-blur border border-white/[0.06]">
              <span className="text-[9px] font-mono uppercase tracking-wider text-zinc-500">
                presence
              </span>
              <div
                className="w-16 h-1.5 rounded-full"
                style={{
                  background:
                    tab === 'f2'
                      ? 'linear-gradient(90deg, rgba(29,78,216,0) 0%, #3b82f6 60%, #fff 100%)'
                      : tab === 'dominance'
                        ? 'linear-gradient(90deg, #3b82f6 0%, rgba(160,160,170,0.5) 50%, #f43f5e 100%)'
                        : 'linear-gradient(90deg, rgba(136,19,55,0) 0%, #f43f5e 60%, #fff 100%)',
                }}
              />
              <span className="text-[9px] font-mono uppercase tracking-wider text-zinc-500">
                high
              </span>
            </div>
            {/* Top-right grid spec badge */}
            <div className="absolute top-3 right-3 px-2.5 py-1.5 rounded-full bg-zinc-950/70 backdrop-blur border border-white/[0.06]">
              <span className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">
                {gridW}×{gridH} grid · {frameW}×{frameH} src
              </span>
            </div>
          </div>
        </motion.div>

        {/* RIGHT RAIL */}
        <div className="space-y-4">
          <DominanceCard f1Pct={f1Pct} f2Pct={f2Pct} contestedPct={contestedPct} />
          <CenterControlCard c1={centerControl['1']} c2={centerControl['2']} />
          <ZoneCard
            zone1={heatmap.dominance.fighter1Zone}
            zone2={heatmap.dominance.fighter2Zone}
            centroid1={heatmap.dominance.fighter1Centroid}
            centroid2={heatmap.dominance.fighter2Centroid}
          />
          <SampleStatsCard
            frames1={fighter1Heat?.frames ?? 0}
            frames2={fighter2Heat?.frames ?? 0}
          />
        </div>
      </div>

      {/* Bottom meta strip */}
      <div className="px-6 py-3 border-t border-white/[0.04] flex items-center justify-between flex-wrap gap-2">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-600">
          ID 01 · {fighter1.totalPunches} punches · rose
        </span>
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
          weighted spatial occupancy · normalized per-fighter
        </span>
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-600">
          ID 02 · {fighter2.totalPunches} punches · blue
        </span>
      </div>
    </motion.section>
  )
}
