'use client'

import { useRef, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  UploadSimple,
  VideoCamera,
  X,
  ArrowRight,
  Warning,
  CircleNotch,
} from '@phosphor-icons/react'
import clsx from 'clsx'

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

type DropzoneState = 'idle' | 'dragging' | 'selected' | 'uploading' | 'error'

export default function VideoDropzone() {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)
  const zoneRef = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<DropzoneState>('idle')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [errorMsg, setErrorMsg] = useState('')

  const acceptFile = useCallback((f: File) => {
    if (!f.type.startsWith('video/')) {
      setErrorMsg('Please upload a video file (MP4, MOV, AVI).')
      setState('error')
      return
    }
    if (f.size > 500 * 1024 * 1024) {
      setErrorMsg('File exceeds the 500 MB limit.')
      setState('error')
      return
    }
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setState('selected')
    setErrorMsg('')
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setState('idle')
      const dropped = e.dataTransfer.files[0]
      if (dropped) acceptFile(dropped)
    },
    [acceptFile]
  )

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (!zoneRef.current?.contains(e.relatedTarget as Node)) {
      setState('idle')
    }
  }, [])

  const handleUpload = useCallback(async () => {
    if (!file) return
    setState('uploading')
    setProgress(0)

    const formData = new FormData()
    formData.append('video', file)

    const xhr = new XMLHttpRequest()
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100))
    }

    xhr.onload = () => {
      if (xhr.status === 200) {
        const { jobId } = JSON.parse(xhr.responseText)
        router.push(`/processing/${jobId}`)
      } else {
        let msg = 'Upload failed. Please try again.'
        try {
          msg = JSON.parse(xhr.responseText).error || msg
        } catch {}
        setErrorMsg(msg)
        setState('error')
      }
    }

    xhr.onerror = () => {
      setErrorMsg('Network error. Check your connection.')
      setState('error')
    }

    xhr.open('POST', '/api/upload')
    xhr.send(formData)
  }, [file, router])

  const reset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setState('idle')
    setErrorMsg('')
    setProgress(0)
  }, [previewUrl])

  const isIdle = state === 'idle' || state === 'dragging'

  return (
    <div className="flex flex-col gap-4">
      {/* Zone label */}
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-zinc-300">Fight footage</p>
        {file && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={reset}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X className="w-3 h-3" />
            Remove
          </motion.button>
        )}
      </div>

      {/* Drop zone / Preview */}
      <AnimatePresence mode="wait">
        {isIdle || state === 'error' ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            ref={zoneRef}
            className={clsx(
              'relative h-72 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-4',
              state === 'dragging'
                ? 'border-rose-500 bg-rose-500/5 shadow-[0_0_0_4px_rgba(225,29,72,0.06)]'
                : state === 'error'
                ? 'border-amber-600/60 bg-amber-600/5'
                : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]'
            )}
            onDragEnter={() => setState('dragging')}
            onDragLeave={handleDragLeave}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => state !== 'error' && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) acceptFile(f)
                e.target.value = ''
              }}
            />

            <AnimatePresence mode="wait">
              {state === 'error' ? (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-3 px-6 text-center"
                >
                  <Warning className="w-8 h-8 text-amber-500" weight="fill" />
                  <p className="text-sm text-amber-400 font-medium">{errorMsg}</p>
                  <button
                    onClick={(e) => { e.stopPropagation(); reset() }}
                    className="text-xs text-zinc-400 hover:text-zinc-200 underline underline-offset-2 transition-colors"
                  >
                    Try again
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-4 pointer-events-none select-none"
                >
                  <motion.div
                    animate={
                      state === 'dragging'
                        ? { scale: 1.15, y: -4 }
                        : { scale: 1, y: 0 }
                    }
                    transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                    className={clsx(
                      'w-14 h-14 rounded-2xl flex items-center justify-center border transition-colors',
                      state === 'dragging'
                        ? 'bg-rose-600/20 border-rose-500/40'
                        : 'bg-white/[0.04] border-white/[0.08]'
                    )}
                  >
                    <UploadSimple
                      className={clsx(
                        'w-6 h-6 transition-colors',
                        state === 'dragging' ? 'text-rose-400' : 'text-zinc-400'
                      )}
                      weight={state === 'dragging' ? 'fill' : 'regular'}
                    />
                  </motion.div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-zinc-200">
                      {state === 'dragging' ? 'Release to upload' : 'Drop your fight footage here'}
                    </p>
                    <p className="text-xs text-zinc-500 mt-1">
                      MP4 · MOV · AVI · WebM — up to 500 MB
                    </p>
                  </div>
                  <span className="text-xs text-rose-400/80 border border-rose-500/20 bg-rose-500/5 px-3 py-1 rounded-full">
                    Click to browse files
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="relative h-72 rounded-2xl overflow-hidden border border-white/[0.08] bg-zinc-900"
          >
            {previewUrl && (
              <video
                src={previewUrl}
                className="w-full h-full object-cover"
                muted
                playsInline
                onMouseOver={(e) => (e.currentTarget as HTMLVideoElement).play()}
                onMouseOut={(e) => {
                  const v = e.currentTarget as HTMLVideoElement
                  v.pause()
                  v.currentTime = 0
                }}
              />
            )}
            {/* Gradient + file info */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent pointer-events-none" />
            <div className="absolute bottom-0 inset-x-0 p-4 flex items-end justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                <VideoCamera className="w-4 h-4 text-zinc-400 flex-shrink-0" weight="fill" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-zinc-100 truncate">{file?.name}</p>
                  <p className="text-xs text-zinc-400">{file ? formatBytes(file.size) : ''}</p>
                </div>
              </div>
              <span className="text-xs text-zinc-500 font-mono flex-shrink-0">
                hover to preview
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload progress */}
      <AnimatePresence>
        {state === 'uploading' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
              <span className="flex items-center gap-1.5">
                <CircleNotch className="w-3 h-3 animate-spin" />
                Uploading video...
              </span>
              <span className="font-mono">{progress}%</span>
            </div>
            <div className="h-[3px] bg-zinc-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-rose-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ ease: 'linear' }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* CTA button */}
      <AnimatePresence>
        {state === 'selected' && (
          <motion.button
            initial={{ opacity: 0, y: 8, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            whileHover={{ scale: 1.015 }}
            whileTap={{ scale: 0.975 }}
            onClick={handleUpload}
            className="w-full flex items-center justify-center gap-2.5 py-3.5 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-sm rounded-xl transition-colors shadow-[0_4px_20px_rgba(225,29,72,0.25)]"
          >
            Begin Analysis
            <ArrowRight className="w-4 h-4" weight="bold" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Disclaimer */}
      <p className="text-xs text-zinc-600 text-center leading-relaxed">
        Processed locally — video never leaves your server.
        <br />
        Pipeline runs boxing_analytics_v2.py under the hood.
      </p>
    </div>
  )
}
