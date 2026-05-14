'use client'

import { useEffect, useState } from 'react'
import { Sun, Moon } from '@phosphor-icons/react'

type Theme = 'dark' | 'light'

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('dark')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    const saved = (localStorage.getItem('theme') as Theme | null) ?? 'dark'
    setTheme(saved)
    document.documentElement.classList.toggle('light', saved === 'light')
    setMounted(true)
  }, [])

  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.classList.toggle('light', next === 'light')
    try {
      localStorage.setItem('theme', next)
    } catch {
      /* localStorage unavailable — non-fatal */
    }
  }

  // Reserve the slot before hydration so layout doesn't shift
  if (!mounted) {
    return <div className="fixed top-4 right-4 z-[100] w-9 h-9" aria-hidden />
  }

  const isDark = theme === 'dark'
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      className="theme-toggle fixed top-4 right-4 z-[100] w-9 h-9 rounded-full border border-white/[0.08] bg-zinc-900/80 backdrop-blur flex items-center justify-center text-zinc-300 hover:text-white hover:scale-105 active:scale-95 transition-all shadow-lg shadow-black/40"
    >
      {isDark ? (
        <Sun className="w-4 h-4" weight="fill" />
      ) : (
        <Moon className="w-4 h-4" weight="fill" />
      )}
    </button>
  )
}
