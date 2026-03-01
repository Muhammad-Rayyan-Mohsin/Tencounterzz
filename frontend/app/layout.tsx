import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { Plus_Jakarta_Sans, Cormorant_Garamond } from 'next/font/google'
import SmoothScrollProvider from '@/components/SmoothScrollProvider'
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
      <body className="font-sans bg-[#0c0c0e] text-zinc-50 antialiased">
        <SmoothScrollProvider>{children}</SmoothScrollProvider>
      </body>
    </html>
  )
}
