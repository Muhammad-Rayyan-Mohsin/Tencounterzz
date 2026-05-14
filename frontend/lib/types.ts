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

export type FighterZone = 'left' | 'center' | 'right'

export interface FighterHeatmap {
  /** Base64-encoded uint8 row-major grid (height × width bytes) */
  grid: string
  /** Number of frames contributing to the heatmap */
  frames: number
}

export interface HeatmapDominance {
  /** % of contested cells where fighter 1 was the dominant occupant */
  fighter1Pct?: number
  /** % of contested cells where fighter 2 was the dominant occupant */
  fighter2Pct?: number
  /** % of cells where both fighters spent meaningful time */
  contestedPct?: number
  /** % of each fighter's total intensity inside the central 30% region */
  centerControl: Record<string, number>
  fighter1Zone?: FighterZone
  fighter2Zone?: FighterZone
  /** Normalized [0,1] (x, y) weighted centroid in the video frame */
  fighter1Centroid?: [number, number]
  fighter2Centroid?: [number, number]
}

export interface HeatmapData {
  /** Source video dimensions [width, height] in pixels */
  frameSize: [number, number]
  /** Heatmap grid dimensions [width, height] in cells (typically [64, 64]) */
  gridSize: [number, number]
  /** Heatmap data keyed by stringified fighter ID ('1', '2') */
  fighters: Record<string, FighterHeatmap>
  dominance: HeatmapDominance
  /** Public URL for the background image (first frame snapshot) */
  bgUrl?: string
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
  heatmap?: HeatmapData
  error?: string
  startedAt: number
  completedAt?: number
}
