import Link from 'next/link'
import { Barbell, GithubLogo, Upload } from '@phosphor-icons/react/dist/ssr'

export default function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-40 border-b border-white/[0.06] bg-[#0c0c0e]/80 backdrop-blur-xl will-change-transform">
      <div className="max-w-[1400px] mx-auto px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 bg-rose-600 rounded-[6px] flex items-center justify-center shadow-[0_0_0_1px_rgba(225,29,72,0.3)] group-hover:bg-rose-500 transition-colors">
            <Barbell weight="bold" className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold tracking-tight text-zinc-100">
            TenCount
          </span>
        </Link>

        <div className="flex items-center gap-4">
          <span className="hidden sm:flex items-center gap-1.5 text-xs font-mono text-zinc-600 border border-white/[0.06] rounded-full px-3 py-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-slow" />
            v2 pipeline
          </span>
          <a
            href="https://github.com/Muhammad-Rayyan-Mohsin/TenCount"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            <GithubLogo weight="fill" className="w-4 h-4" />
            <span className="hidden sm:inline">Source</span>
          </a>
          <Link
            href="/analyze"
            className="flex items-center gap-1.5 text-sm bg-rose-600 hover:bg-rose-500 text-white px-4 py-1.5 rounded-full transition-colors font-medium"
          >
            <Upload className="w-3.5 h-3.5" />
            Analyse
          </Link>
        </div>
      </div>
    </nav>
  )
}
