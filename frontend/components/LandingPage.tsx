'use client'

import React, { useEffect, useRef, useState, memo } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { motion, AnimatePresence } from 'framer-motion'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  Barbell,
  GithubLogo,
  ArrowRight,
  Target,
  Cube,
  Brain,
  CheckCircle,
  Circle,
  Fire,
  Code,
  Camera,
  Globe,
  Package,
  Cpu,
  Shuffle,
  Play,
  Database,
  ChartLine,
  TreeStructure,
  Palette,
  Lightning,
  UserFocus,
  Crosshair,
  Graph,
  PersonSimpleRun,
  Stack,
} from '@phosphor-icons/react/dist/ssr'

gsap.registerPlugin(ScrollTrigger)

// ── Punch Shuffler ────────────────────────────────────────────────────────────

const PUNCHES = [
  { label: 'Jab', conf: 87 },
  { label: 'Cross', conf: 94 },
  { label: 'Lead Hook', conf: 78 },
  { label: 'Rear Hook', conf: 83 },
  { label: 'Lead Uppercut', conf: 71 },
  { label: 'Rear Uppercut', conf: 76 },
]

const PunchShuffler = memo(function PunchShuffler() {
  const [active, setActive] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setActive(i => (i + 1) % PUNCHES.length), 1800)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="space-y-3 mt-4">
      {PUNCHES.map((p, i) => {
        const isActive = i === active
        return (
          <div key={p.label} className="flex items-center gap-3">
            <span className={`text-[11px] font-mono w-24 text-right transition-colors duration-300 ${isActive ? 'text-zinc-100' : 'text-zinc-700'}`}>
              {p.label}
            </span>
            <div className="flex-1 h-[2px] bg-zinc-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full origin-left"
                animate={{
                  scaleX: isActive ? p.conf / 100 : 0.04,
                  backgroundColor: isActive ? '#e11d48' : '#3f3f46',
                }}
                transition={{ type: 'spring', stiffness: 70, damping: 18 }}
              />
            </div>
            <span className={`text-[11px] font-mono w-9 text-right transition-colors duration-300 ${isActive ? 'text-rose-400' : 'text-zinc-700'}`}>
              {isActive ? `${p.conf}%` : '--'}
            </span>
          </div>
        )
      })}
    </div>
  )
})

// ── Telemetry Feed ────────────────────────────────────────────────────────────

const TELEMETRY = [
  'Initialising YOLOv11m detection model...',
  'Fighter 1 bounding box locked · conf 99%',
  'Fighter 2 bounding box locked · conf 97%',
  'Extracting 17 COCO keypoints per frame...',
  'Computing wrist velocity vector: 4.2 m/s',
  'Measuring elbow flexion angle: 147°',
  'BiLSTM window complete · 30 frames read',
  'Classifying punch → Cross · conf 94%',
  'Fighter 1 total: 12 punches detected',
  'Rendering annotated H.264 output...',
]

const TelemetryFeed = memo(function TelemetryFeed() {
  const [lineIdx, setLineIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)

  useEffect(() => {
    if (charIdx < TELEMETRY[lineIdx].length) {
      const id = setTimeout(() => setCharIdx(c => c + 1), 22)
      return () => clearTimeout(id)
    }
    const id = setTimeout(() => {
      setLineIdx(i => (i + 1) % TELEMETRY.length)
      setCharIdx(0)
    }, 1600)
    return () => clearTimeout(id)
  }, [lineIdx, charIdx])

  const start = Math.max(0, lineIdx - 3)
  const history = TELEMETRY.slice(start, lineIdx)

  return (
    <div className="mt-4 font-mono text-[11px] space-y-1.5 min-h-[80px]">
      {history.map((line, i) => (
        <div key={`${lineIdx}-${i}`} className="text-zinc-700 truncate">
          <span className="text-zinc-800 mr-2">$</span>{line}
        </div>
      ))}
      <div className="text-zinc-300 flex items-center">
        <span className="text-rose-500 mr-2">›</span>
        <span>{TELEMETRY[lineIdx].slice(0, charIdx)}</span>
        <span className="inline-block w-[5px] h-3 bg-rose-500 ml-px animate-pulse" />
      </div>
    </div>
  )
})

// ── Timeline Animation ────────────────────────────────────────────────────────

type TEvent = { time: number; fighter: 1 | 2; type: string }

const SAMPLE_EVENTS: TEvent[] = [
  { time: 1.2, fighter: 1, type: 'Jab' },
  { time: 2.8, fighter: 2, type: 'Cross' },
  { time: 4.1, fighter: 1, type: 'Lead Hook' },
  { time: 5.5, fighter: 2, type: 'Jab' },
  { time: 7.0, fighter: 1, type: 'Cross' },
  { time: 8.3, fighter: 2, type: 'Rear Hook' },
  { time: 9.9, fighter: 1, type: 'Rear Uppercut' },
  { time: 11.4, fighter: 2, type: 'Lead Hook' },
  { time: 12.8, fighter: 1, type: 'Cross' },
  { time: 14.1, fighter: 2, type: 'Jab' },
]
const TOTAL_DURATION = 15

const TimelineAnim = memo(function TimelineAnim() {
  const [events, setEvents] = useState<TEvent[]>([])

  useEffect(() => {
    // Reset on every mount to avoid stale state from StrictMode double-invoke
    setEvents([])
    let i = 0
    const id = setInterval(() => {
      const event = SAMPLE_EVENTS[i]
      if (i < SAMPLE_EVENTS.length && event != null) {
        setEvents(prev => [...prev, event])
        i++
      } else {
        setEvents([])
        i = 0
      }
    }, 900)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="mt-4 space-y-4">
      {([1, 2] as const).map(fighter => (
        <div key={fighter} className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-zinc-600 w-5">F{fighter}</span>
          <div className="flex-1 h-[2px] bg-zinc-800 rounded-full relative">
            <AnimatePresence>
              {events
                .filter((e): e is TEvent => e != null && e.fighter === fighter)
                .map((e, idx) => (
                  <motion.div
                    key={`${e.time}-${idx}`}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                    className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-rose-500"
                    style={{ left: `${(e.time / TOTAL_DURATION) * 100}%` }}
                    title={e.type}
                  />
                ))}
            </AnimatePresence>
          </div>
        </div>
      ))}
      <div className="flex justify-between text-[10px] font-mono text-zinc-800 mt-1">
        <span>0s</span>
        <span>{TOTAL_DURATION}s</span>
      </div>
    </div>
  )
})

// ── Icons Marquee ─────────────────────────────────────────────────────────────

type IconItem = {
  name: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Icon: React.ComponentType<any>
  color: string
}

const ROW_1: IconItem[] = [
  { name: 'PyTorch', Icon: Fire, color: '#ee4c2c' },
  { name: 'Python', Icon: Code, color: '#3776ab' },
  { name: 'OpenCV', Icon: Camera, color: '#5c3ee8' },
  { name: 'NumPy', Icon: Stack, color: '#4dabcf' },
  { name: 'scikit-learn', Icon: Brain, color: '#f89939' },
  { name: 'SciPy', Icon: Cube, color: '#0c55a5' },
  { name: 'Matplotlib', Icon: ChartLine, color: '#3587c4' },
  { name: 'Ultralytics', Icon: Lightning, color: '#e11d48' },
]

const ROW_2: IconItem[] = [
  { name: 'YOLOv11m', Icon: Target, color: '#e11d48' },
  { name: 'YOLOv8-pose', Icon: Crosshair, color: '#e11d48' },
  { name: 'AttentionBiLSTM', Icon: Brain, color: '#8b5cf6' },
  { name: 'BoT-SORT', Icon: Shuffle, color: '#64748b' },
  { name: 'COCO Keypoints', Icon: PersonSimpleRun, color: '#10b981' },
  { name: 'Pose Estimation', Icon: UserFocus, color: '#f59e0b' },
  { name: 'ONNX', Icon: Package, color: '#9b59b6' },
  { name: 'GPU Inference', Icon: Cpu, color: '#64748b' },
]

const ROW_3: IconItem[] = [
  { name: 'Next.js 14', Icon: Globe, color: '#ffffff' },
  { name: 'React 18', Icon: Circle, color: '#61dafb' },
  { name: 'TypeScript', Icon: Code, color: '#3178c6' },
  { name: 'Framer Motion', Icon: Play, color: '#ea4c89' },
  { name: 'GSAP 3', Icon: Graph, color: '#88ce02' },
  { name: 'Tailwind CSS', Icon: Palette, color: '#38bdf8' },
  { name: 'Node.js', Icon: Database, color: '#339933' },
  { name: 'Python 3', Icon: TreeStructure, color: '#ffd43b' },
]

const MarqueeRow = memo(function MarqueeRow({
  items,
  reverse,
  duration,
}: {
  items: IconItem[]
  reverse: boolean
  duration: number
}) {
  const doubled = [...items, ...items]
  return (
    <div className="flex overflow-hidden">
      <motion.div
        className="flex gap-3 flex-none"
        animate={{ x: reverse ? ['-50%', '0%'] : ['0%', '-50%'] }}
        transition={{ duration, repeat: Infinity, ease: 'linear', repeatType: 'loop' }}
      >
        {doubled.map((item, i) => (
          <div
            key={i}
            className="flex-none flex items-center gap-2 px-4 py-2 rounded-xl border border-white/[0.05] bg-zinc-900/30 whitespace-nowrap"
          >
            <item.Icon
              className="w-4 h-4 flex-shrink-0"
              style={{ color: item.color }}
              weight="fill"
            />
            <span className="text-xs text-zinc-500 font-mono">{item.name}</span>
          </div>
        ))}
      </motion.div>
    </div>
  )
})

// ── Pipeline Stage Visuals ────────────────────────────────────────────────────

function DetectionVisual() {
  return (
    <div className="relative w-72 h-52 rounded-2xl border border-white/[0.06] bg-zinc-900/40 overflow-hidden">
      <div className="absolute inset-0 bg-grid-pattern bg-grid-sm opacity-40" />
      <div className="absolute top-6 left-7 w-[88px] h-36 border border-rose-500/60 rounded">
        <div className="absolute -top-3 left-1 bg-rose-600/90 text-white text-[9px] font-mono px-1.5 py-0.5 rounded">
          F1 · 99%
        </div>
      </div>
      <div className="absolute top-10 right-7 w-[76px] h-28 border border-zinc-500/40 rounded">
        <div className="absolute -top-3 left-1 bg-zinc-700 text-zinc-300 text-[9px] font-mono px-1.5 py-0.5 rounded">
          F2 · 97%
        </div>
      </div>
    </div>
  )
}

function PoseVisual() {
  return (
    <div className="relative w-[500px] h-[330px] rounded-2xl overflow-hidden">
      <Image
        src="/pose-boxing.png"
        alt="YOLOv8m-pose skeleton overlay — two fighters mid-exchange"
        fill
        className="object-cover object-center"
        sizes="500px"
        priority={false}
      />
      {/* Left fade — blends into dark section bg where the text lives */}
      <div className="absolute inset-0 bg-gradient-to-r from-[#0c0c0e]/70 via-transparent to-transparent" />
      {/* Amber tint to match stage-02 accent */}
      <div className="absolute inset-0 bg-amber-500/[0.04]" />
      {/* Edge refraction ring */}
      <div className="absolute inset-0 rounded-2xl shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06),inset_0_1px_0_rgba(245,158,11,0.12)]" />
    </div>
  )
}

function ClassifyVisual() {
  const bars = [
    { label: 'Cross', v: 0.94 },
    { label: 'Jab', v: 0.06 },
    { label: 'L. Hook', v: 0.0 },
  ]
  return (
    <div className="w-60 space-y-3">
      {bars.map(b => (
        <div key={b.label} className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-zinc-500 w-14 text-right">{b.label}</span>
          <div className="flex-1 h-[3px] bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-rose-500 transition-all duration-700" style={{ width: `${b.v * 100}%` }} />
          </div>
          <span className="text-[11px] font-mono text-zinc-500 w-8">{Math.round(b.v * 100)}%</span>
        </div>
      ))}
      <div className="mt-2 inline-flex items-center gap-2 bg-rose-500/10 border border-rose-500/20 rounded-full px-3 py-1">
        <CheckCircle className="w-3 h-3 text-rose-400" weight="fill" />
        <span className="text-[11px] font-mono text-rose-400">Cross · 94% confidence</span>
      </div>
    </div>
  )
}

// ── Main Landing Page ─────────────────────────────────────────────────────────

export default function LandingPage() {
  const navRef       = useRef<HTMLElement>(null)
  const heroH1Ref    = useRef<HTMLHeadingElement>(null)
  const heroSubRef   = useRef<HTMLParagraphElement>(null)
  const heroCTARef   = useRef<HTMLDivElement>(null)
  const manifestoRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero entrance stagger
      const tl = gsap.timeline({ delay: 0.15 })
      tl.from(heroH1Ref.current!.querySelectorAll('.h-line'), {
          y: 36, opacity: 0, stagger: 0.1, duration: 0.7, ease: 'power2.out',
        }, '-=0.35')
        .from(heroSubRef.current, { y: 20, opacity: 0, duration: 0.6, ease: 'power2.out' }, '-=0.45')
        .from(heroCTARef.current, { y: 16, opacity: 0, duration: 0.5, ease: 'power2.out' }, '-=0.35')

      // Navbar opaque on scroll — toggle class instead of per-frame style writes
      ScrollTrigger.create({
        start: 'top -40',
        onUpdate: self => {
          if (navRef.current) {
            navRef.current.classList.toggle('nav-scrolled', self.scroll() > 40)
          }
        },
      })

      // Manifesto text reveal
      gsap.fromTo(manifestoRef.current!.querySelectorAll('.m-line'),
        { y: 50, opacity: 0 },
        {
          y: 0, opacity: 1, stagger: 0.18, duration: 0.85, ease: 'power2.out',
          scrollTrigger: { trigger: manifestoRef.current, start: 'top 72%' },
        }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <>
      {/* ── Floating Nav ── */}
      <nav
        ref={navRef}
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300 will-change-transform"
        style={{ background: 'transparent', borderBottom: '1px solid transparent' }}
      >
        <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 bg-rose-600 rounded-[6px] flex items-center justify-center group-hover:bg-rose-500 transition-colors shadow-[0_0_0_1px_rgba(225,29,72,0.3)]">
              <Barbell weight="bold" className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold tracking-tight text-zinc-100">TenCount</span>
          </Link>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-6 text-sm text-zinc-500">
              <a href="#features" className="hover:text-zinc-200 transition-colors">Features</a>
              <a href="#pipeline" className="hover:text-zinc-200 transition-colors">Pipeline</a>
              <a
                href="https://github.com/Muhammad-Rayyan-Mohsin/TenCount"
                target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 hover:text-zinc-200 transition-colors"
              >
                <GithubLogo weight="fill" className="w-4 h-4" />Source
              </a>
            </div>
            <Link
              href="/analyze"
              className="group flex items-center gap-1.5 bg-rose-600 hover:bg-rose-500 active:scale-[0.97] text-white text-sm font-medium px-5 py-2 rounded-full transition-all duration-200"
            >
              Analyse footage
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative min-h-[100dvh] flex items-center overflow-hidden">
        {/* Unsplash: dark boxing gloves on ring canvas — photo-1549719386 */}
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1549719386-74dfcbf7dbed?auto=format&fit=crop&w=1920&q=80')`,
          }}
        />
        {/* Directional overlay — heavy bottom-left (text zone), lighter top-right (image zone) */}
        <div
          className="absolute inset-0"
          style={{
            background: `linear-gradient(
              135deg,
              rgba(12,12,14,0.92) 0%,
              rgba(12,12,14,0.78) 35%,
              rgba(12,12,14,0.55) 65%,
              rgba(12,12,14,0.38) 100%
            )`,
          }}
        />
        {/* Rose accent — upper-right glow */}
        <div
          className="absolute inset-0"
          style={{
            background: `radial-gradient(ellipse 70% 55% at 85% 15%, rgba(225,29,72,0.07) 0%, transparent 60%)`,
          }}
        />
        {/* Grid texture — very faint so image breathes through */}
        <div className="absolute inset-0 bg-grid-pattern bg-grid-sm opacity-[0.12]" />

        {/* Content — bottom-left */}
        <div className="relative z-10 pt-16 px-6 md:px-16 max-w-[1400px] mx-auto w-full">
          <div className="max-w-[700px]">
            <h1
              ref={heroH1Ref}
              className="leading-[0.88] tracking-tighter mb-8"
              style={{ fontSize: 'clamp(3.6rem, 9vw, 8rem)' }}
            >
              <span className="h-line block font-display font-bold text-zinc-50">Every punch,</span>
              <span className="h-line block font-serif italic text-zinc-400 font-normal">classified.</span>
            </h1>

            <p ref={heroSubRef} className="text-zinc-400 text-lg leading-relaxed max-w-[50ch] mb-10">
              Upload fight footage. The pipeline runs YOLOv11m detection,
              YOLOv8m-pose estimation, and AttentionBiLSTM classification
              across 6 punch types — fully automated.
            </p>

            <div ref={heroCTARef} className="flex items-center gap-5 flex-wrap">
              <Link
                href="/analyze"
                className="group inline-flex items-center gap-2 bg-rose-600 hover:bg-rose-500 active:scale-[0.98] text-white font-semibold text-sm px-7 py-3.5 rounded-full transition-all duration-200 shadow-[0_0_0_1px_rgba(225,29,72,0.3)]"
              >
                Analyse footage
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </Link>
              <a
                href="https://github.com/Muhammad-Rayyan-Mohsin/TenCount"
                target="_blank" rel="noopener noreferrer"
                className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                View on GitHub →
              </a>
            </div>
          </div>
        </div>

      </section>

      {/* ── Feature Artifact Cards ── */}
      <section id="features" className="relative py-24 md:py-32">
        {/* Background depth — full-width so orbs reach screen edges */}
        <div className="absolute -top-48 left-0 w-1/2 h-[700px] rounded-full bg-rose-600/[0.07] blur-[160px] pointer-events-none" />
        <div className="absolute -bottom-48 right-0 w-1/2 h-[700px] rounded-full bg-sky-500/[0.05] blur-[160px] pointer-events-none" />
        <div className="relative px-6 md:px-16 max-w-[1400px] mx-auto">

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true, amount: 0.4 }}
          className="relative mb-12"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="w-1 h-1 rounded-full bg-rose-500" />
            <span className="text-xs font-mono uppercase tracking-widest text-zinc-600">System modules</span>
          </div>
          <h2 className="font-display text-3xl md:text-4xl font-bold tracking-tighter text-zinc-100">
            Three AI subsystems,<br className="hidden md:block" /> one unified pipeline.
          </h2>
        </motion.div>

        <div className="relative grid grid-cols-1 md:grid-cols-[1.5fr_1.2fr_1fr] gap-4">
          {/* Card 1 */}
          <motion.div
            initial={{ opacity: 0, y: 44, scale: 0.97 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.65, delay: 0.0, ease: [0.22, 1, 0.36, 1] }}
            viewport={{ once: true, amount: 0.25 }}
            className="relative rounded-[1.75rem] border border-white/[0.07] overflow-hidden p-7 flex flex-col"
            style={{ background: 'rgba(12,10,15,0.55)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}
          >
            {/* Blur shimmer — materialises after card rises */}
            <div
              className="absolute inset-0 pointer-events-none rounded-[1.75rem] backdrop-blur-[22px] will-change-transform"
            />
            <div className="relative z-10 flex flex-col h-full">
              <div className="flex items-center gap-3 mb-1">
                <div className="w-8 h-8 bg-rose-500/10 border border-rose-500/20 rounded-xl flex items-center justify-center">
                  <Brain className="w-4 h-4 text-rose-400" />
                </div>
                <h3 className="text-sm font-semibold text-zinc-100">Classification Engine</h3>
              </div>
              <p className="text-[11px] font-mono text-zinc-600 mb-2 ml-11">AttentionBiLSTM · 6 classes</p>
              <PunchShuffler />
            </div>
          </motion.div>

          {/* Card 2 */}
          <motion.div
            initial={{ opacity: 0, y: 44, scale: 0.97 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.65, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            viewport={{ once: true, amount: 0.25 }}
            className="relative rounded-[1.75rem] border border-white/[0.07] overflow-hidden p-7 flex flex-col"
            style={{ background: 'rgba(12,10,15,0.55)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}
          >
            <div
              className="absolute inset-0 pointer-events-none rounded-[1.75rem] backdrop-blur-[22px] will-change-transform"
            />
            <div className="relative z-10 flex flex-col h-full">
              <div className="flex items-center gap-3 mb-1">
                <div className="w-8 h-8 bg-rose-500/10 border border-rose-500/20 rounded-xl flex items-center justify-center">
                  <Circle className="w-4 h-4 text-rose-400" weight="fill" />
                </div>
                <h3 className="text-sm font-semibold text-zinc-100">Neural Telemetry</h3>
              </div>
              <p className="text-[11px] font-mono text-zinc-600 mb-2 ml-11">Real-time pipeline log</p>
              <TelemetryFeed />
            </div>
          </motion.div>

          {/* Card 3 */}
          <motion.div
            initial={{ opacity: 0, y: 44, scale: 0.97 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.65, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            viewport={{ once: true, amount: 0.25 }}
            className="relative rounded-[1.75rem] border border-white/[0.07] overflow-hidden p-7 flex flex-col"
            style={{ background: 'rgba(12,10,15,0.55)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}
          >
            <div
              className="absolute inset-0 pointer-events-none rounded-[1.75rem] backdrop-blur-[22px] will-change-transform"
            />
            <div className="relative z-10 flex flex-col h-full">
              <div className="flex items-center gap-3 mb-1">
                <div className="w-8 h-8 bg-rose-500/10 border border-rose-500/20 rounded-xl flex items-center justify-center">
                  <Target className="w-4 h-4 text-rose-400" />
                </div>
                <h3 className="text-sm font-semibold text-zinc-100">Punch Timeline</h3>
              </div>
              <p className="text-[11px] font-mono text-zinc-600 mb-2 ml-11">Per-fighter event stream</p>
              <TimelineAnim />
            </div>
          </motion.div>
        </div>
        </div>
      </section>

      {/* ── Manifesto ── */}
      <section ref={manifestoRef} className="py-24 md:py-40 border-y border-white/[0.04] overflow-hidden">
        <div className="max-w-[1400px] mx-auto px-6 md:px-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 items-end">
            <div className="m-line">
              <p className="font-display font-bold tracking-tighter leading-[0.9] text-zinc-700"
                style={{ fontSize: 'clamp(2.5rem, 6vw, 5.5rem)' }}>
                Scorecards record outcomes.
              </p>
            </div>
            <div className="m-line">
              <p className="font-display font-bold tracking-tighter leading-[0.9] text-zinc-100"
                style={{ fontSize: 'clamp(3rem, 7vw, 6.5rem)' }}>
                We record the mechanics.
              </p>
            </div>
          </div>
          <div className="m-line mt-12 md:mt-16 flex flex-wrap gap-3">
            {['Wrist velocity', 'Elbow angle delta', 'Shoulder rotation', 'Temporal attention', '30-frame windows'].map(tag => (
              <span key={tag} className="text-[11px] font-mono text-zinc-700 border border-white/[0.06] rounded-full px-3 py-1 uppercase tracking-wide">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline Sticky Stack ── */}
      <section id="pipeline" className="relative">
        {([
          {
            stage: '01', title: 'Person Detection',
            model: 'YOLOv11m · best_potential.pt',
            description: 'Fine-tuned on boxing footage, the detector tracks exactly two fighters using Hungarian-algorithm assignment with a 50,000-point inertia bonus that prevents identity switching mid-match.',
            tags: ['Bounding-box crop', 'IoU assignment', '2-fighter lock', 'Inertia bonus'],
            Visual: DetectionVisual as React.ComponentType,
            tint: 'rgba(225,29,72,0.025)',
            labelColor: 'text-rose-400',
            bgImage: '/detection-boxing.png' as (string | null),
            bgFit: 'cover' as 'cover' | 'contain',
          },
          {
            stage: '02', title: 'Pose Estimation',
            model: 'YOLOv8m-pose · COCO keypoints',
            description: '17 anatomical landmarks per fighter, per frame. Skeleton integrity validation requires ≥10 visible keypoints before the sequence is passed to the classifier.',
            tags: ['17 keypoints', 'Elbow angle', 'Wrist velocity', 'Shoulder width'],
            Visual: DetectionVisual as React.ComponentType,
            tint: 'rgba(245,158,11,0.025)',
            labelColor: 'text-amber-400',
            bgImage: '/pose-boxing.png' as (string | null),
            bgFit: 'cover' as 'cover' | 'contain',
          },
          {
            stage: '03', title: 'Punch Classification',
            model: 'AttentionBiLSTM · 30-frame windows',
            description: 'A bidirectional LSTM with temporal self-attention processes 30-frame keypoint sequences (51 dimensions per frame) to classify 6 punch types with real-time confidence scoring.',
            tags: ['51-dim input', '6-class output', 'Temporal attention', 'Intensity estimate'],
            Visual: ClassifyVisual as React.ComponentType,
            tint: 'rgba(14,165,233,0.025)',
            labelColor: 'text-sky-400',
            bgImage: '/classify-boxing.png' as (string | null),
            bgFit: 'contain' as 'cover' | 'contain',
          },
        ]).map((s, i) => (
          <div
            key={s.stage}
            className="sticky top-0 min-h-[100dvh] flex items-center bg-[#0c0c0e] will-change-transform"
            style={{ zIndex: 10 + i }}
          >
            {/* Full-bleed background photo */}
            {s.bgImage && (
              <>
                <Image
                  src={s.bgImage}
                  alt=""
                  fill
                  className={s.bgFit === 'contain' ? 'object-contain object-right' : 'object-cover object-center'}
                  sizes="100vw"
                  priority={i === 0}
                />
                {/* Left sweep — keeps text zone dark and legible */}
                <div className="absolute inset-0 bg-gradient-to-r from-[#0c0c0e] from-[38%] via-[#0c0c0e]/80 via-[58%] to-[#0c0c0e]/20" />
                {/* Top + bottom bleed — merges with section above/below */}
                <div className="absolute inset-0 bg-gradient-to-b from-[#0c0c0e]/85 via-transparent to-[#0c0c0e]/85" />
                {/* Stage-specific colour wash over image zone */}
                <div className="absolute inset-0" style={{ background: s.tint.replace('0.025', '0.12') }} />
              </>
            )}

            {/* Per-stage radial accent (all stages) */}
            {!s.bgImage && (
              <div className="absolute inset-0 pointer-events-none"
                style={{ background: `radial-gradient(ellipse 60% 50% at 80% 80%, ${s.tint} 0%, transparent 60%)` }} />
            )}

            <div className="relative z-10 max-w-[1400px] mx-auto px-6 md:px-16 py-24 w-full">
              <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-12 lg:gap-24 items-center">
                <div>
                  <div className="flex items-center gap-3 mb-6">
                    <span className="text-xs font-mono text-zinc-700 uppercase tracking-widest">Stage {s.stage} / 03</span>
                    <span className="w-8 h-px bg-zinc-800" />
                    <span className={`text-xs font-mono ${s.labelColor}`}>{s.model}</span>
                  </div>
                  <h2 className="font-display font-bold tracking-tighter text-zinc-100 mb-6 leading-none"
                    style={{ fontSize: 'clamp(3rem, 8vw, 7rem)' }}>
                    {s.title}
                  </h2>
                  <p className="text-zinc-400 text-lg leading-relaxed max-w-[54ch] mb-8">{s.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {s.tags.map(tag => (
                      <span key={tag} className="text-xs font-mono text-zinc-500 border border-white/[0.07] rounded-full px-3 py-1">{tag}</span>
                    ))}
                  </div>
                </div>
                {/* Right visual — only for stages without a full-bleed image */}
                {!s.bgImage && (
                  <div className="hidden lg:flex items-center justify-center opacity-80">
                    <s.Visual />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </section>

      {/* ── CTA ── */}
      <section className="py-24 md:py-32 px-6 md:px-16 max-w-[1400px] mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }} viewport={{ once: true, amount: 0.5 }}
          className="rounded-[2.5rem] border border-white/[0.06] bg-zinc-900/40 px-10 md:px-20 py-16 md:py-24 relative overflow-hidden"
        >
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[200px] pointer-events-none"
            style={{ background: 'radial-gradient(ellipse 80% 100% at 50% 0%, rgba(225,29,72,0.06) 0%, transparent 70%)' }} />
          <div className="relative z-10 text-center">
            <div className="flex items-center justify-center gap-2 mb-6">
              <span className="w-1 h-1 rounded-full bg-rose-500 animate-pulse" />
              <span className="text-xs font-mono uppercase tracking-widest text-zinc-600">System ready</span>
            </div>
            <h2 className="font-display font-bold tracking-tighter text-zinc-100 mb-6 leading-none"
              style={{ fontSize: 'clamp(2.2rem, 6vw, 5rem)' }}>
              Ready to analyse<br className="hidden md:block" /> your footage?
            </h2>
            <p className="text-zinc-400 text-base leading-relaxed max-w-[44ch] mx-auto mb-10">
              Upload a boxing video and receive a full biomechanical breakdown —
              punch counts, classifications, and annotated output.
            </p>
            <Link href="/analyze"
              className="group inline-flex items-center gap-2 bg-rose-600 hover:bg-rose-500 active:scale-[0.98] text-white font-semibold text-base px-8 py-4 rounded-full transition-all duration-200 shadow-[0_0_0_1px_rgba(225,29,72,0.3)]"
            >
              Start analysis
              <ArrowRight className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
        </motion.div>
      </section>

      {/* ── Footer ── */}
      <footer className="rounded-t-[2.5rem] bg-zinc-900/60 border-t border-white/[0.05] px-6 md:px-16 pt-14 pb-10">
        <div className="max-w-[1400px] mx-auto">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 mb-12">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 bg-rose-600 rounded-[6px] flex items-center justify-center">
                <Barbell weight="bold" className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="font-semibold text-zinc-100 tracking-tight leading-none">TenCount</p>
                <p className="text-[11px] text-zinc-600 mt-0.5 font-mono">Boxing Analytics v2</p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm text-zinc-500">
              <Link href="/analyze" className="hover:text-zinc-200 transition-colors">Analyse</Link>
              <a
                href="https://github.com/Muhammad-Rayyan-Mohsin/TenCount"
                target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 hover:text-zinc-200 transition-colors"
              >
                <GithubLogo weight="fill" className="w-4 h-4" />GitHub
              </a>
            </div>
          </div>
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pt-8 border-t border-white/[0.05]">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-slow" />
              <span className="text-xs font-mono text-zinc-600">System Operational</span>
            </div>
            <p className="text-xs text-zinc-700 font-mono">
              TenCount · FYP Boxing Analytics · AttentionBiLSTM Pipeline
            </p>
          </div>
        </div>
      </footer>
    </>
  )
}
