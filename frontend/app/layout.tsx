import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { Plus_Jakarta_Sans, Cormorant_Garamond } from 'next/font/google'
import SmoothScrollProvider from '@/components/SmoothScrollProvider'
import ThemeToggle from '@/components/ThemeToggle'
import './globals.css'

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  variable: '--font-plus-jakarta',
  display: 'swap',
})

const cormorant = Cormorant_Garamond({
  subsets: ['latin'],
  weight: ['400', '600'],
  style: ['normal', 'italic'],
  variable: '--font-cormorant',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'TenCount — Boxing Analytics',
  description:
    'AI-powered boxing punch detection and analytics. Upload fight footage and get real-time punch counts, fighter tracking, and classification.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} ${plusJakarta.variable} ${cormorant.variable} scroll-smooth`}
    >
      <head>
        <link rel="preconnect" href="https://images.unsplash.com" />
        <link rel="dns-prefetch" href="https://images.unsplash.com" />
        {/* FOUC-prevention: apply saved theme before first paint */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('theme');if(t==='light')document.documentElement.classList.add('light')}catch(e){}",
          }}
        />
      </head>
      <body className="font-sans bg-[#0c0c0e] text-zinc-50 antialiased">
        <ThemeToggle />
        <SmoothScrollProvider>{children}</SmoothScrollProvider>
      </body>
    </html>
  )
}
