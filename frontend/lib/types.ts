export type JobStatus =
  | 'uploading'
  | 'detecting'
  | 'pose'
  | 'classifying'
  | 'rendering'
  | 'complete'
  | 'error'

export type PunchType =
  | 'jab'
  | 'cross'
  | 'lead_hook'
  | 'rear_hook'
  | 'lead_uppercut'
  | 'rear_uppercut'

export interface PunchEvent {
  time: number
  fighter: 1 | 2
  type: PunchType
}

export interface PunchBreakdown {
  jab: number
  cross: number
  lead_hook: number
  rear_hook: number
  lead_uppercut: number
  rear_uppercut: number
}

export interface FighterResult {
  id: 1 | 2
  totalPunches: number
  breakdown: PunchBreakdown
}

export interface JobResult {
  jobId: string
  status: JobStatus
  progress: number
  currentStep: string
  currentDetail: string
  originalFilename: string
  videoUrl?: string
  duration?: number
  fps?: number
  frameCount?: number
  fighters?: [FighterResult, FighterResult]
  timeline?: PunchEvent[]
  error?: string
  startedAt: number
  completedAt?: number
}
