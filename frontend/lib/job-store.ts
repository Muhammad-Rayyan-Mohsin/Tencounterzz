import type { JobResult } from './types'

// In-memory store. Replace with Redis/DB in production.
declare global {
  // eslint-disable-next-line no-var
  var __jobStore: Map<string, JobResult> | undefined
}

export const jobStore: Map<string, JobResult> =
  global.__jobStore ?? new Map<string, JobResult>()

if (process.env.NODE_ENV !== 'production') {
  global.__jobStore = jobStore
}
