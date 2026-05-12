# TenCount — Technical Reference

A production-grade technical reference for **TenCount**, a boxing analytics web application that ingests sparring/fight footage and returns per-fighter punch counts, punch-type breakdowns, intensity scores, and an annotated video.

This document is the authoritative source on the system's architecture, components, data flow, models, deployment topology, and known limitations. It is organised so that any engineer can:

1. Understand the system end-to-end in one read.
2. Locate the file responsible for any behaviour.
3. Reproduce the deployment.
4. Identify the rough edges before they bite in production.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Layout](#4-repository-layout)
5. [Frontend (Next.js)](#5-frontend-nextjs)
6. [ML / Computer-Vision Pipeline](#6-ml--computer-vision-pipeline)
7. [Frontend ↔ Python Bridge](#7-frontend--python-bridge)
8. [End-to-End Job Lifecycle](#8-end-to-end-job-lifecycle)
9. [Data Model](#9-data-model)
10. [Models & Artifacts](#10-models--artifacts)
11. [Deployment & Infrastructure](#11-deployment--infrastructure)
12. [Configuration & Environment](#12-configuration--environment)
13. [Performance Characteristics](#13-performance-characteristics)
14. [Known Limitations & Technical Debt](#14-known-limitations--technical-debt)
15. [Future Work](#15-future-work)

---

## 1. System Overview

TenCount is a single-tenant web application that performs offline analysis of pre-recorded boxing video. The user uploads an MP4/WebM file via a Next.js frontend; a Python process on the same host runs a multi-stage computer-vision pipeline (detection → tracking → pose → punch classification → intensity scoring); the frontend polls a job-status endpoint and renders a results page with per-fighter statistics and the annotated video.

The system runs as a **monolith** on a single AWS EC2 instance:

- **One Node process** (Next.js production server) serves the UI, the job API, and the static uploads directory.
- **One Python subprocess per inference job** is spawned on-demand and communicates back via line-delimited JSON on stdout.
- **In-memory job store** holds job state; no database or cache layer is currently used.

Originally a final-year project (FYP), the codebase emphasises depth of CV pipeline over operational hardening: the model stack is non-trivial (custom YOLOv11m detector + YOLOv8m pose + custom AttentionBiLSTM punch classifier), while infrastructure remains development-grade (HTTP only, ephemeral IP, no queueing, no persistence).

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                   BROWSER                                       │
│  React 18 + framer-motion + GSAP + Lenis      <───── polling every 1s ────┐    │
│                                                                            │    │
│  Drop video ──► /analyze ──► /processing/[id] ──► /results/[id]            │    │
└────────────────────────────┬───────────────────────────────────────────────┴────┘
                             │ POST /api/upload (multipart, ≤500 MB)
                             ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       Next.js 14 (App Router, port 3000)                        │
│                                                                                 │
│   /api/upload  ───────► writes /public/uploads/{jobId}.{ext}                    │
│        │                                                                        │
│        │   spawn()      ┌────────────────────────────────────────────────┐     │
│        └──────────────► │       frontend_runner.py  (per-job)            │     │
│                         │                                                │     │
│                         │  patches boxing_analytics_v2 constants, runs   │     │
│                         │  ba.main(), intercepts print(), emits          │     │
│                         │  newline-delimited JSON events on stdout       │     │
│                         └─────────────────┬──────────────────────────────┘     │
│                                           │ stdout JSON                          │
│                          ┌────────────────▼────────────────┐                    │
│   /api/jobs/[jobId] ◄────│  in-memory jobStore  (Map)      │                    │
│                          └─────────────────────────────────┘                    │
│   /api/uploads/[...path] ──► serves input video + annotated output via FS       │
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │ imports
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       boxing_analytics_v2.py  (994 lines)                       │
│                                                                                 │
│  ┌─────────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  YOLOv11m   │─►│ IoU+Hung.  │─►│ YOLOv8m  │─►│ AttentionBi  │─►│Intensity│  │
│  │  detect     │  │ tracking   │  │  pose    │  │ LSTM (6-cls) │  │ scoring │  │
│  │  (custom)   │  │ +inertia   │  │ 17 COCO  │  │ 25-frame seq │  │ 6 feats │  │
│  └─────────────┘  └────────────┘  └──────────┘  └──────────────┘  └─────────┘  │
│         │                                                              │        │
│         ▼                                                              ▼        │
│   annotated MP4 (cv2 VideoWriter, mp4v)               JSON events to stdout    │
└────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                AWS EC2 (eu-north-1, m7i-flex.large, Ubuntu 24.04)                │
│   systemd unit `tencount`  •  ports 22/80/443/3000  •  100 GB gp3 EBS            │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Version | Notes |
|---|---|---|---|
| **Frontend framework** | Next.js | 14.2.5 | App Router, RSC where applicable |
| **UI runtime** | React | 18 | |
| **Language (UI)** | TypeScript | 5.x | strict mode |
| **Styling** | Tailwind CSS | 3.4 | + CSS custom properties for tokens |
| **Animation** | framer-motion | 11.3.19 | transient state, page transitions |
| **Scroll** | GSAP + ScrollTrigger | 3.14.2 | scroll-driven hero/manifesto |
| **Smooth scroll** | Lenis | 1.3.17 | wraps the entire app |
| **Icons** | @phosphor-icons/react | 2.1.7 | tree-shaken via `optimizePackageImports` |
| **Fonts** | Geist Sans/Mono, Plus Jakarta Sans, Cormorant Garamond | — | self-hosted + Google Fonts |
| **Build tooling** | PostCSS, Autoprefixer | — | |
| **Backend runtime** | Node.js | 20.x | systemd-supervised |
| **Inference runtime** | Python | 3.10+ | venv at `~/.venv` on EC2 |
| **CV — detection** | Ultralytics YOLO (YOLOv11m, custom-trained) | ultralytics ≥ 8.0 | `best_potential.pt` |
| **CV — pose** | YOLOv8m-pose (stock) | ultralytics ≥ 8.0 | 17 COCO keypoints |
| **CV — classifier** | Custom **AttentionBiLSTM** | PyTorch ≥ 2.0 | 6-class punch typing |
| **Numerics** | NumPy ≥ 1.24, SciPy ≥ 1.11 | — | |
| **Video I/O** | OpenCV (headless) ≥ 4.8 | — | mp4v codec |
| **Tensor backend** | PyTorch ≥ 2.0 | CPU wheel on EC2; MPS/CUDA locally | |
| **Infrastructure** | AWS EC2 + EBS + Security Group | — | provisioned via bash script |
| **Region** | `eu-north-1` (Stockholm) | — | hardcoded |
| **Process supervisor** | systemd (`tencount.service`) | — | `Restart=on-failure` |
| **Storage** | Local disk (`public/uploads/`) | — | no S3, no DB |
| **State store** | In-memory `Map` in Node | — | lost on restart |
| **Transport** | HTTP (plaintext) on :3000 | — | no TLS, no reverse proxy |

---

## 4. Repository Layout

```
Tencounterzz/
├── boxing_analytics_v2.py        ← 994-line CV pipeline (detection/tracking/pose/classify)
├── frontend_runner.py            ← thin wrapper that the Node API spawns per job
├── deploy.sh                     ← AWS provisioning + lifecycle (deploy/start/stop/teardown)
├── requirements.txt              ← ultralytics, opencv-python-headless, numpy, scipy, torch
├── DEPLOYMENT.md                 ← operator-facing deploy notes
├── README.md
├── .deploy-state                 ← EC2 instance id (written by deploy.sh)
├── .eip-state                    ← Elastic IP allocation id (currently orphaned)
├── models/
│   └── punch_classifier.pt       ← AttentionBiLSTM checkpoint (~9 MB)
├── Tracking and Counting/
│   └── yolov8m-pose.pt           ← stock pose model (~51 MB)
├── runs/
│   └── person_detect/.../weights/best_potential.pt  ← custom YOLOv11m detector
└── frontend/
    ├── package.json
    ├── next.config.mjs           ← bodySizeLimit=500mb, /uploads rewrite
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── app/
    │   ├── layout.tsx            ← fonts, SmoothScrollProvider, dark theme
    │   ├── page.tsx              ← LandingPage
    │   ├── globals.css           ← design tokens (--bg, --accent, etc.)
    │   ├── analyze/page.tsx
    │   ├── processing/[jobId]/page.tsx
    │   ├── results/[jobId]/page.tsx
    │   └── api/
    │       ├── upload/route.ts           ← POST, spawns Python
    │       ├── jobs/[jobId]/route.ts     ← GET, reads jobStore
    │       └── uploads/[...path]/route.ts ← static file serve w/ range hdrs
    ├── components/
    │   ├── LandingPage.tsx
    │   ├── Nav.tsx
    │   ├── VideoDropzone.tsx
    │   ├── ProcessingView.tsx
    │   ├── ResultsView.tsx
    │   └── SmoothScrollProvider.tsx
    ├── lib/
    │   ├── job-store.ts          ← in-memory Map<string, JobResult>
    │   └── types.ts              ← JobResult / FighterResult / PunchEvent
    └── public/
        └── uploads/              ← input + annotated output videos live here
```

---

## 5. Frontend (Next.js)

### 5.1 Routing

Next.js 14 App Router. All routes are server components by default; client components are explicit (`'use client'`).

| URL | File | Purpose |
|---|---|---|
| `/` | `app/page.tsx` | Landing page wrapper for `<LandingPage />` |
| `/analyze` | `app/analyze/page.tsx` | Upload UI: left identity panel + right `<VideoDropzone />` |
| `/processing/[jobId]` | `app/processing/[jobId]/page.tsx` | 5-step pipeline progress tracker |
| `/results/[jobId]` | `app/results/[jobId]/page.tsx` | Annotated video + per-fighter stats + timeline |

The **root layout** (`app/layout.tsx`) loads Geist Sans/Mono, Plus Jakarta Sans, and Cormorant Garamond; wraps everything in `<SmoothScrollProvider>`; and applies the dark theme (`--bg: #0c0c0e`).

### 5.2 API Routes

#### `POST /api/upload`  (`app/api/upload/route.ts`)

1. Accepts multipart `FormData` with key `video`.
2. Validates MIME (`startsWith('video/')`) and size (≤ 500 MB).
3. Generates job id: `` `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}` ``.
4. Writes the file to `frontend/public/uploads/{jobId}.{ext}`.
5. Seeds `jobStore` with `status: 'detecting'`, `progress: 8`.
6. Spawns Python via `child_process.spawn()`:
   ```ts
   spawn(PYTHON, [RUNNER, '--video', videoPath, '--output-dir', UPLOADS], {
     cwd: FYP_ROOT,
     env: { ...process.env },
   })
   ```
   - `PYTHON` comes from `process.env.PYTHON_BIN` or defaults to the macOS CLT path.
   - `RUNNER` resolves to `/.../Tencounterzz/frontend_runner.py`.
   - Stdout is parsed line-by-line for JSON events.
   - Stderr is logged but does not fail the job.
7. Returns `{ jobId }` immediately. Inference continues asynchronously.

#### `GET /api/jobs/[jobId]`  (`app/api/jobs/[jobId]/route.ts`)

Single-statement handler — looks up `jobStore.get(jobId)` and returns the `JobResult` JSON, or 404 if missing. **Polled at 1 Hz by `ProcessingView`** and once on mount by `ResultsView`.

#### `GET /api/uploads/[...path]`  (`app/api/uploads/[...path]/route.ts`)

Static file server for input + annotated output videos. Rejects `..` for path traversal. Sets `Content-Type` from extension (mp4/webm/png/jpeg/octet-stream), `Content-Length`, and `Accept-Ranges: bytes` (the browser still receives a full-body response — files are read via `readFile()` not a stream, so byte-range *headers* are advertised but byte-range *responses* are not actually implemented).

A `next.config.mjs` `rewrites()` rule maps `/uploads/:path*` → `/api/uploads/:path*` so public URLs stay clean.

### 5.3 Components

| Component | State | Notable libraries / patterns |
|---|---|---|
| `LandingPage` | refs only (`navRef`, `heroH1Ref`) | GSAP timeline for hero entrance, ScrollTrigger for manifesto reveal + sticky pipeline sections, framer-motion for feature cards |
| `Nav` | static | Hardcoded links; `nav-scrolled` class toggled by GSAP at >40px scroll |
| `VideoDropzone` | `file`, `previewUrl`, `state` (idle/dragging/selected/uploading/error), `progress` | XHR with `onprogress` for upload bar; hover-to-play preview via `URL.createObjectURL` |
| `ProcessingView` | `job`, `notFound`, elapsed timer | 1 Hz polling of `/api/jobs/[jobId]`; staggered framer-motion step cards; auto-navigates to `/results/[jobId]` when `status === 'complete'` |
| `ResultsView` | `job`, `loading`, `error` | Custom `<VideoPlayer>` + `<PunchBar>` + SVG `<PunchTimeline>`; spring animations on bar widths and timeline markers |
| `SmoothScrollProvider` | Lenis instance | Syncs Lenis with GSAP ticker (`gsap.ticker.add` → `lenis.raf`); registers `ScrollTrigger` |

### 5.4 Styling

Design tokens in `globals.css`:

```css
--bg: #0c0c0e
--surface: #111113
--surface-raised: #18181b
--border: rgba(255,255,255,0.06)
--border-strong: rgba(255,255,255,0.1)
--accent: #e11d48          /* rose-600 */
--accent-dim: rgba(225,29,72,0.08)
```

Tailwind extends with font families (`font-sans`/`mono`/`display`/`serif`), an `animate-pulse-slow` keyframe (3 s), and a 40-px grid background pattern used at ~0.12 opacity in the hero and ~0.04 in cards. There is no `dark:` prefix usage — the theme is hardcoded dark.

### 5.5 Build Config

`next.config.mjs`:
- `experimental.serverActions.bodySizeLimit = '500mb'` — allows the large multipart POST.
- `experimental.optimizePackageImports = ['@phosphor-icons/react', 'framer-motion']` — tree-shakes icon and motion barrels.
- `rewrites()` → `/uploads/:path*` to `/api/uploads/:path*`.

There is **no** `output: 'standalone'` — the EC2 host runs `next start` directly against `node_modules`.

---

## 6. ML / Computer-Vision Pipeline

The pipeline lives entirely in `boxing_analytics_v2.py`. It is sequential and stateful (per-fighter state machines). It is invoked via `ba.main()` after `frontend_runner.py` patches the module-level constants.

### 6.1 Pipeline Stages (in order)

1. **Video ingest** — `cv2.VideoCapture(VIDEO_PATH)`; extracts W, H, FPS, total frames; creates `cv2.VideoWriter` with `mp4v` fourcc.
2. **Person detection** — `det_model(frame, conf=0.35, iou=0.45, imgsz=640)` using the custom YOLOv11m at `runs/person_detect/.../weights/best_potential.pt`.
3. **Tracking** — IoU-based bipartite matching with a Hungarian solver (`match_detections_to_tracks`), augmented with an `INERTIA_BONUS = 50000` cost reward for existing fighter tracks. `_recycle_fighter_id()` re-uses a recently-expired track id if a new detection appears within normalized centroid distance 0.8, preserving punch counts and keypoint buffers across brief occlusions. Tracks die after `MAX_TRACK_HISTORY = 30` missed frames.
4. **Fighter slot assignment** — top-2 tracks by score become `is_fighter=True` (`MAX_FIGHTERS = 2`); the highest-confidence non-overlapping third person becomes the (visual-only) referee.
5. **Pose estimation** — per fighter, the EMA-smoothed bbox (`BBOX_EMA_ALPHA = 0.3`) is used to crop the frame; `pose_model(crop, conf=0.25)` returns 17 COCO keypoints. Frames with fewer than `MIN_VISIBLE_KEYPOINTS = 7` confident joints are rejected.
6. **Keypoint buffering** — normalized to bbox-relative `[0,1]`, EMA-smoothed with `CLF_KPT_SMOOTH_ALPHA = 0.4`, then pushed into a per-fighter `deque` of length `CLF_SEQ_LEN = 25` (~0.83 s @ 30 fps). Frame ids are tracked in parallel; gaps of ≤ 2 frames are bridged with the last good keypoints.
7. **Punch *detection*** (rule-based, per arm) — `FighterState.check_punch()` runs a 5-frame sliding-window state machine:
   - Smooth shoulder/elbow/wrist over the window (`_smooth_kpt`).
   - Compute elbow angle (cosine law).
   - **Trigger** when *all* hold: elbow angle ≥ `PUNCH_ANGLE_THRESHOLD = 155°`, angle delta ≥ `MIN_ANGLE_DELTA = 15°`, and wrist velocity ≥ `MIN_WRIST_VELOCITY = 0.25` (normalised by shoulder width).
   - **Debounce** with `PUNCH_COOLDOWN = 20` frames per arm.
   - **Retraction gate**: after firing, the arm must drop below `RESET_ANGLE_THRESHOLD = 120°` before it can fire again.
8. **Punch *type* classification** — when a punch is detected and the buffer has ≥ `CLF_MIN_FRAMES = 15` contiguous frames, `classify_punch()` runs the AttentionBiLSTM on a `(1, 25, 75)` tensor (sequence × features). The 75-dim feature vector is hip-centred, torso-height-scaled keypoints (34) + 7 hand-engineered angles + 34 first-order velocities. Left-facing fighters are mirrored (`_mirror_buffer`) to match the training distribution.
9. **Intensity scoring** — `compute_intensity()` computes six scalars on a wider 15-frame window: cumulative wrist velocity impulse, 95th-percentile elbow angular velocity, 95th-percentile shoulder rotation speed, 95th-percentile wrist jerk, hip impulse, and post-peak wrist deceleration. After 5 punches, normalisation switches from global fallback maxima (`INTENSITY_V_MAX = 4.0`, etc.) to per-fighter 95th percentiles. Punch-type-specific weights are applied, then thresholded: Light (<0.40) / Medium / Heavy (≥0.70).
10. **Annotation** — bounding box + track id + label (colour from `PUNCH_COLORS`) with `LABEL_PERSIST_FRAMES = 45` (1.5 s) persistence; count, intensity, confidence overlay; on-screen legend and frame counter. Frame is written to the output MP4.

### 6.2 Models

| Role | File | Architecture | Size | Notes |
|---|---|---|---|---|
| Person detection | `runs/person_detect/person_11s_50epochs_20251210_022509/weights/best_potential.pt` | YOLOv11m (custom-trained) | — | conf=0.35, iou=0.45, imgsz=640 |
| Pose estimation | `Tracking and Counting/yolov8m-pose.pt` | YOLOv8m-pose (stock) | ~51 MB | 17 COCO keypoints |
| Punch typing | `models/punch_classifier.pt` | **AttentionBiLSTM**: 2-layer BiLSTM (256 hidden, dropout 0.4) + 64-unit tanh attention bottleneck → softmax over sequence → LayerNorm → 128-unit GELU dense → 6-class head | ~9 MB | Checkpoint stores `config`, `model_state`, `idx_to_class` |

The 6 punch classes: **Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut**.

### 6.3 Device Selection

Selected in priority order: `mps` (Apple Silicon) → `cuda` (NVIDIA) → `cpu`. No batching is used — detection, pose, and the LSTM all run on single examples per frame. (In production on the EC2 m7i-flex.large this is always `cpu`.)

### 6.4 Output Artefacts

- **Annotated video**: `{OUTPUT_DIR}/v2_{YYYYMMDD_HHMMSS}.mp4` (in production `OUTPUT_DIR` is `frontend/public/uploads/`).
- **Live stdout JSON events** (see [§7.2](#72-stdout-protocol)) — not persisted to disk.
- **Per-frame data** is *not* persisted; it lives only in process memory and the printed log.

---

## 7. Frontend ↔ Python Bridge

### 7.1 `frontend_runner.py`

A 155-line per-job wrapper. **Not** a daemon — it starts, runs one inference, exits.

**CLI**:
```bash
python3 frontend_runner.py --video /abs/input.mp4 --output-dir /abs/out_dir
```

Argparse runs *before* importing `boxing_analytics_v2`, so the wrapper can patch module constants in place:

```python
ba.DETECTION_MODEL  = "runs/person_detect/.../best_potential.pt"
ba.POSE_MODEL       = "Tracking and Counting/yolov8m-pose.pt"
ba.VIDEO_PATH       = args.video
ba.OUTPUT_DIR       = args.output_dir
```

It then monkey-patches `builtins.print` to intercept every line `boxing_analytics_v2` writes, parses it (regex on the `Frame N/M P%` log line for progress; pre-formatted JSON for events), and re-emits clean newline-delimited JSON on its own stdout with `flush=True`.

Finally it calls `ba.main()`. On exception it emits `{"t":"error","msg":<repr>}` and exits 1.

### 7.2 Stdout Protocol

The Node API parses these JSON objects line-by-line. All numeric fields are integers unless noted.

| Event | Shape | Meaning |
|---|---|---|
| Progress | `{"t":"progress","v":<0-100>}` | Emitted ≈ every 60 frames. Frontend maps `v < 35 → detecting`, `< 65 → pose`, `≥ 65 → classifying`. |
| Live punch | `{"t":"punch_event","fighter":1\|2,"type":"<class>"}` | Appended to `liveEvents`; used to build the timeline when ≥ 80% of expected events arrive. |
| Per-fighter total | `{"t":"fighter_total","id":1\|2,"total":<n>}` | Final count for that fighter. |
| Breakdown | `{"t":"breakdown","fighter":1\|2,"type":"<class>","n":<n>}` | Per-type counts. |
| Output path | `{"t":"output","path":"/abs/path/v2_*.mp4"}` | Triggers `progress → 95`, `status → 'rendering'`. |
| Done | `{"t":"done"}` | Final flush; `progress → 99`, status will flip to `complete` once the route finalises. |
| Error | `{"t":"error","msg":"<text>"}` | Sets `status: 'error'`, stores message. |

### 7.3 Process Plumbing

- **Spawn**: `child_process.spawn()`, `cwd = FYP_ROOT`, env inherited.
- **Backpressure**: Node accumulates raw bytes in a buffer (`Buffer.concat`) and splits on `\n`. Stderr is logged but non-fatal.
- **Concurrency**: Unbounded — every `POST /api/upload` spawns a new Python process. There is no queue, no semaphore, no GPU memory check.
- **Timeout**: None. A hung inference leaks the process indefinitely.
- **Cleanup**: On `proc.on('error')` or non-zero exit, the job is marked `error` but the input file is *not* removed.

### 7.4 Static Serving of Outputs

When the `output` event arrives, the route extracts `basename(path)` and stores `videoUrl: '/uploads/<basename>'`. Because both the input upload and the annotated output are written into `frontend/public/uploads/`, the `/api/uploads/[...path]` route serves both transparently. `Accept-Ranges: bytes` is advertised but ranges are **not** actually honoured — the route uses `readFile()`, not `createReadStream()`, so the entire file is buffered into memory per request.

---

## 8. End-to-End Job Lifecycle

```
┌────────────┐                                              t = 0 s
│  Browser   │  drop .mp4 → VideoDropzone → XHR with progress
└──────┬─────┘
       │  POST /api/upload  (multipart, ≤500 MB)
       ▼
┌──────────────────────────┐
│  /api/upload route       │  ─ validate MIME + size
│                          │  ─ generate jobId
│                          │  ─ writeFile public/uploads/{id}.ext
│                          │  ─ jobStore.set(id, {status:'detecting', progress:8})
│                          │  ─ spawn(python, runner, --video, --output-dir)
│                          │  ─ return {jobId}                  ◄── t ≈ 100 ms
└──────────────┬───────────┘
               │
               │  router.push(`/processing/${jobId}`)
               ▼
┌──────────────────────────┐
│  ProcessingView          │  polls GET /api/jobs/{id}  @ 1 Hz
└──────────────┬───────────┘
               │
               │  (meanwhile, Python is running…)
               │
               │   stdout JSON → upload/route.ts parser → jobStore.set(id, {...})
               │
               │   status: detecting → pose → classifying → rendering → complete
               │
               ▼
┌──────────────────────────┐
│  ResultsView             │  on `status === 'complete'`, navigate
│                          │  GET /api/jobs/{id} once → render video + stats
└──────────────────────────┘
```

The frontend never opens an SSE / WebSocket. All progress is pull-based via the 1 Hz poll on `/api/jobs/[jobId]`.

If Python emits fewer than 80 % of the expected `punch_event` messages (or the route otherwise has a sparse `liveEvents` array), the route **synthesises** a timeline by unpacking the per-type breakdown into a list of punch types per fighter, shuffling, and scattering across the video duration with small random gaps. This means the per-fighter totals and breakdowns are always exact, but timeline timestamps may be approximate.

---

## 9. Data Model

`frontend/lib/types.ts`:

```ts
type JobStatus =
  | 'uploading' | 'detecting' | 'pose'
  | 'classifying' | 'rendering' | 'complete' | 'error'

type PunchType =
  | 'jab' | 'cross'
  | 'lead_hook' | 'rear_hook'
  | 'lead_uppercut' | 'rear_uppercut'

interface PunchEvent     { time: number; fighter: 1 | 2; type: PunchType }
interface PunchBreakdown { jab; cross; lead_hook; rear_hook;
                           lead_uppercut; rear_uppercut: number }
interface FighterResult  { id: 1 | 2; totalPunches: number; breakdown: PunchBreakdown }

interface JobResult {
  jobId: string
  status: JobStatus
  progress: number          // 0–100
  currentStep: string       // e.g. "Person Detection"
  currentDetail: string     // e.g. "YOLOv11m tracking… 23%"
  originalFilename: string
  videoUrl?: string         // /uploads/{id}.{ext} (input) or /uploads/v2_*.mp4 (output)
  duration?: number; fps?: number; frameCount?: number
  fighters?: [FighterResult, FighterResult]
  timeline?: PunchEvent[]
  error?: string
  startedAt: number; completedAt?: number
}
```

The store is a global `Map<string, JobResult>` in `lib/job-store.ts`, preserved across HMR in dev via `globalThis`. **A server restart wipes all jobs.**

---

## 10. Models & Artifacts

| Asset | Path | Provenance |
|---|---|---|
| YOLOv11m detector | `runs/person_detect/person_11s_50epochs_20251210_022509/weights/best_potential.pt` | Custom-trained on a person-detection split. Filename encodes `person_11s_50epochs_<date>`. |
| YOLOv8m pose | `Tracking and Counting/yolov8m-pose.pt` | Stock Ultralytics release. |
| Punch classifier | `models/punch_classifier.pt` | Custom AttentionBiLSTM. Checkpoint format: `{"config": {...}, "model_state": <state_dict>, "idx_to_class": {...}}`. |

All three are loaded from disk per-process; no model warm cache.

---

## 11. Deployment & Infrastructure

### 11.1 Topology

A single AWS EC2 instance hosts the Node server and runs Python inferences as on-demand subprocesses. There is no load balancer, no container layer, and no managed database.

```
   ┌───────────────┐
   │   Internet    │
   └───────┬───────┘
           │ HTTP :3000   (plaintext)
           ▼
   ┌───────────────────────────────────────────────┐
   │  EC2  eu-north-1                              │
   │  m7i-flex.large (default) / g4dn.xlarge (GPU) │
   │  Ubuntu 24.04 (CPU AMI) / DLAMI (GPU)         │
   │  100 GB gp3 EBS                               │
   │                                               │
   │  ┌─────────────────────────────────────────┐  │
   │  │  systemd: tencount.service              │  │
   │  │   ExecStart= next start -p 3000         │  │
   │  │   Restart=on-failure                    │  │
   │  │                                         │  │
   │  │   Node 20 → spawn() → Python venv       │  │
   │  └─────────────────────────────────────────┘  │
   │                                               │
   │  Security group: 22 / 80 / 443 / 3000 open    │
   │  to 0.0.0.0/0                                 │
   └───────────────────────────────────────────────┘
```

### 11.2 Provisioning (`deploy.sh`)

Sub-commands: `deploy | start | stop | status | ssh | teardown`.

`./deploy.sh deploy` performs (idempotent where possible):

1. Create key pair `tencount-deploy-key` (saved to `~/.ssh/tencount-deploy-key.pem`, chmod 400) if missing.
2. Create security group `tencount-sg` with inbound 22, 80, 443, 3000 from `0.0.0.0/0` if missing.
3. `aws ec2 run-instances` — launch with the configured AMI + 100 GB gp3 EBS; tag `Name=TenCount-FYP`; persist instance id to `.deploy-state`.
4. `aws ec2 wait instance-running`, then resolve public IP.
5. Poll SSH for up to 5 minutes (`StrictHostKeyChecking=no`).
6. Tar the repo (excluding `.git`, `node_modules`, `.next`, `__pycache__`, `.deploy-state`, `.DS_Store` — ~80 MB) and `scp` to `/tmp/tencount-deploy.tar.gz`.
7. SSH and run the bootstrap inline:
   - `apt-get install python3-pip python3-venv libgl1 libglib2.0-0`
   - NodeSource Node 20 (if missing)
   - `python3 -m venv ~/.venv`
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   - `pip install ultralytics opencv-python-headless scipy numpy`
   - `cd frontend && npm install --production=false && npm run build`
   - Write `/etc/systemd/system/tencount.service`, `systemctl enable --now tencount`.
8. Print `http://<PUBLIC_IP>:3000`.

### 11.3 systemd Unit (excerpt)

```ini
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tencount/frontend
Environment=NODE_ENV=production
Environment=PYTHON_BIN=/home/ubuntu/tencount/.venv/bin/python3
Environment=PROJECT_ROOT=/home/ubuntu/tencount
Environment=PORT=3000
ExecStart=/usr/bin/node /home/ubuntu/tencount/frontend/node_modules/.bin/next start -p 3000
Restart=on-failure
RestartSec=5
```

Logs: `sudo journalctl -u tencount -f`.

### 11.4 Lifecycle Commands

| Command | Action |
|---|---|
| `deploy` | Full provisioning (steps above). |
| `start` | `aws ec2 start-instances` → wait → `sudo systemctl start tencount`. **A new public IP is assigned.** |
| `stop` | `aws ec2 stop-instances` — preserves EBS, pauses compute charges. |
| `status` | Print instance state + current public IP. |
| `ssh` | `ssh -i ~/.ssh/tencount-deploy-key.pem ubuntu@<ip>`. |
| `teardown` | Terminate instance, delete SG, remove `.deploy-state`. Key pair is kept. |

### 11.5 State Files

| File | Contents | Risk if deleted |
|---|---|---|
| `.deploy-state` | EC2 instance id (e.g. `i-05bdd0752c3d14b73`) | Lifecycle commands all fail; the live instance becomes unmanageable via the script. |
| `.eip-state` | Elastic IP allocation id (e.g. `eipalloc-063ee4efe218c1501`) | **Currently orphaned** — the script does not associate or use this EIP. AWS bills ~$0.005/h for an unassociated EIP. Release it or wire it up. |

### 11.6 GPU Path (gated)

The script is GPU-ready behind two environment variables:

```bash
TENCOUNT_INSTANCE_TYPE=g4dn.xlarge \
TENCOUNT_AMI_ID=ami-0248a5203d01dc336 \
./deploy.sh deploy
```

`g4dn.xlarge` requires the **"Running On-Demand G and VT instances"** quota to be raised from 0 to at least 4 (24–48 h AWS approval).

**Caveat**: the bootstrap still installs the **CPU** PyTorch wheel (`--index-url …/whl/cpu`). On the DLAMI this works (PyTorch is already present with CUDA), but the pip step will silently downgrade it to CPU. Fix is to branch on `TENCOUNT_INSTANCE_TYPE` and install the appropriate wheel.

---

## 12. Configuration & Environment

### 12.1 Environment Variables

| Var | Default | Used by |
|---|---|---|
| `PYTHON_BIN` | `/Library/Developer/CommandLineTools/usr/bin/python3` (macOS dev) | `/api/upload/route.ts` (path to Python interpreter) |
| `PROJECT_ROOT` | repo root | `frontend_runner.py` (overrides relative model paths) |
| `PORT` | `3000` | systemd `tencount.service` |
| `NODE_ENV` | `production` | systemd |
| `TENCOUNT_INSTANCE_TYPE` | `m7i-flex.large` | `deploy.sh` |
| `TENCOUNT_AMI_ID` | `ami-0dab98137e5c11cb8` | `deploy.sh` |
| AWS creds | `~/.aws/credentials` or env | `deploy.sh` (developer host only) |

### 12.2 Hardcoded CV Constants (excerpt)

All in `boxing_analytics_v2.py`. Re-tuned through experimentation; comments in the source flag the historical values (e.g. `PUNCH_COOLDOWN = 20  # was 8 — caused overcounting`).

```python
DETECTION_CONF                = 0.35
DETECTION_IOU                 = 0.45
IMAGE_SIZE                    = 640
INERTIA_BONUS                 = 50000
MAX_TRACK_HISTORY             = 30
IoU_THRESHOLD_MATCH           = 0.3
MAX_FIGHTERS                  = 2
MIN_VISIBLE_KEYPOINTS         = 7
BBOX_EMA_ALPHA                = 0.3
PUNCH_COOLDOWN                = 20
PUNCH_ANGLE_THRESHOLD         = 155
RESET_ANGLE_THRESHOLD         = 120
MIN_WRIST_VELOCITY            = 0.25
MIN_ANGLE_DELTA               = 15
CLF_SEQ_LEN                   = 25
CLF_INPUT_SIZE                = 75
CLF_MIN_FRAMES                = 15
CLF_KPT_SMOOTH_ALPHA          = 0.4
INTENSITY_HISTORY_LEN         = 15
INTENSITY_V_MAX               = 4.0
INTENSITY_OE_MAX              = 25.0
INTENSITY_MIN_PUNCHES_ADAPTIVE = 5
LABEL_PERSIST_FRAMES          = 45
```

---

## 13. Performance Characteristics

### 13.1 Inference Throughput

- **No batching** anywhere in the pipeline — detection, pose, and the LSTM all run on single examples per frame.
- **CPU path (m7i-flex.large)**: detection is the bottleneck; expect well below real-time on 1080p footage. The product surface acknowledges this — the UI is built around an asynchronous job with progress polling rather than live inference.
- **MPS/CUDA (dev only)**: roughly 15–30 fps at 640×480; closer to real-time on a discrete GPU.
- **EMA smoothing on bboxes** (α = 0.3) and **on keypoints** (α = 0.4) are not just quality features — they also reduce the per-frame work the pose model has to absorb noise from.

### 13.2 Frontend Performance Notes

- The landing page registers GSAP `ScrollTrigger`s on multiple sections; potential jank on low-end mobile is partly mitigated by `will-change: transform` hints on sticky elements.
- `ResultsView`'s `PunchTimeline` animates 50+ markers with staggered `scaleY` springs — smooth on desktop, prone to drop frames on entry-level phones. A Canvas implementation would be the obvious next step.
- Hover-to-play video previews use `URL.createObjectURL`, which keeps the full file in memory until revoked.

### 13.3 Network Performance

- Upload is a single multipart POST (no chunking, no resumable). At 500 MB cap, a network blip means restarting the full upload.
- Static video serving via `readFile()` buffers entire files into RAM per request despite advertising `Accept-Ranges: bytes`. Replacing with `fs.createReadStream()` and implementing the `Range`/`Content-Range` response is the cheapest single performance fix in the codebase.

---

## 14. Known Limitations & Technical Debt

Organised by likelihood of biting you in production.

### High

1. **In-memory job store** (`lib/job-store.ts`). A server restart, redeploy, or systemd `Restart=on-failure` wipes every job mid-flight. The file itself explicitly comments: *"Replace with Redis/DB in production."*
2. **Unbounded concurrent Python subprocesses**. Every upload spawns a new process. Three simultaneous uploads on a 2-vCPU instance will all stall; on a GPU instance they will OOM-kill each other. Needs a queue (Bull / RabbitMQ / SQS) and a worker pool.
3. **No HTTPS / no reverse proxy**. Plaintext HTTP on `:3000` exposed to `0.0.0.0/0`. Video content and any future auth tokens travel in the clear.
4. **No CI/CD**. Deployment is a developer running `./deploy.sh deploy` from a laptop with valid AWS creds. No staging, no automated tests, no rollback.
5. **No monitoring or alerting**. Only `journalctl -u tencount`. No CPU/memory/GPU metrics, no error tracking, no uptime checks.
6. **GPU wheel mismatch**. The bootstrap installs the CPU PyTorch wheel even when `TENCOUNT_INSTANCE_TYPE=g4dn.xlarge`. On the DLAMI this happens to work because PyTorch is already present, but the pip step will silently regress it.
7. **Range-request advertising without honouring**. The static handler returns `Accept-Ranges: bytes` but reads the file with `readFile()` — for large annotated videos this is both slow and a memory risk.
8. **No backups / no EBS snapshots**. The 100 GB volume holds the only copy of the trained detector and any uploaded videos.

### Medium

9. **Ephemeral public IP**. `aws ec2 stop` / `start` re-rolls the public IP, breaking any bookmarks. `.eip-state` records an EIP allocation id, but `deploy.sh` does **not** allocate or associate it — the EIP is orphaned and being billed.
10. **Synthetic timeline fallback**. If fewer than ~80 % of expected `punch_event` JSONs reach the route, timestamps are interpolated and shuffled. Counts are exact; timeline timing can drift.
11. **No request deduplication / idempotency**. Re-submitting the same video produces a second job id and a second Python process.
12. **Hardcoded model paths**. Swapping models means editing source and redeploying.
13. **Hardcoded macOS Python path fallback**. `/Library/Developer/CommandLineTools/usr/bin/python3` is the dev-mode default if `PYTHON_BIN` is unset.
14. **No timeout / no cleanup** on spawned Python processes.
15. **`StrictHostKeyChecking=no`** in `deploy.sh` SSH calls — fine for FYP, MITM-vulnerable in production.
16. **No rate limiting** on `/api/upload`. Any internet client can drain the inference capacity of the box.

### Low / cosmetic

17. **No global React error boundary** — individual views handle their own error states, but unhandled rejections crash the page.
18. **`animate-spin-slow`** Tailwind keyframe exists in the config but isn't used anywhere.
19. `originalFilename` is taken straight from the client and stored unsanitised (cosmetic only — it's never used in a filesystem path).
20. Per-type intensity breakdown is computed during inference but only printed live, not persisted to the result payload.

---

## 15. Future Work

In rough priority order, what the next engineer should do:

1. **Persist `jobStore`.** Redis is the smallest possible change; SQLite-on-disk is acceptable as an interim.
2. **Introduce a job queue.** BullMQ on Redis, with N=1 worker on CPU (or N=2–3 on a g4dn.xlarge) and explicit GPU memory accounting.
3. **Terminate TLS in front of Next.js.** Caddy or nginx with Let's Encrypt, then add a Route 53 record so users get `tencount.example.com` instead of an EC2 IP.
4. **Wire the EIP.** Either use the one in `.eip-state` or release it; the script should allocate-and-associate during `deploy` and de-associate (not release) during `teardown`.
5. **Stream static video.** `fs.createReadStream()` + real `Range` support is one route handler edit.
6. **Containerise.** A single multi-stage Dockerfile (Node + Python venv + model files) plus an ECS Fargate or simply `docker run` on the same EC2 host removes most of the bootstrap brittleness in `deploy.sh`.
7. **Fix the GPU pip step.** Branch on `TENCOUNT_INSTANCE_TYPE` to install the appropriate `torch` wheel.
8. **Emit `punch_event` reliably** with the actual frame timestamp from `boxing_analytics_v2.py` so the synthetic-timeline fallback can be deleted.
9. **CI**: GitHub Actions for `npm run build` + a smoke `python boxing_analytics_v2.py` on a tiny fixture clip.
10. **CloudWatch metrics + alarms** on CPU, memory, disk, and `tencount.service` restarts.

---

*This document was generated from a full source-level review of `boxing_analytics_v2.py`, `frontend_runner.py`, the Next.js frontend (`app/`, `components/`, `lib/`), and `deploy.sh`. Refer back to the cited file paths and constants for ground truth; any divergence between this document and the code is a bug in this document.*
