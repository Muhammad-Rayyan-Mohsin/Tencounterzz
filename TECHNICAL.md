# TenCount — Technical Reference

A production-grade technical reference for **TenCount**, a boxing analytics web application that ingests sparring/fight footage and returns per-fighter punch counts, punch-type breakdowns, intensity scores, and an annotated video.

This document is the authoritative source on the system's architecture, components, data flow, models, deployment topology, and known limitations. It is organised so that any engineer can:

1. Understand the system end-to-end in one read.
2. Locate the file responsible for any behaviour.
3. Reproduce the deployment.
4. Identify the rough edges before they bite in production.

---

## Table of Contents

0. [Elevator Pitch](#0-elevator-pitch)
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
16. [S3-Backed Run History (Persistence Layer)](#16-s3-backed-run-history-persistence-layer)
17. [Feature-by-Feature Implementation Reference](#17-feature-by-feature-implementation-reference)
18. [ELI5: How TenCount Works](#18-eli5-how-tencount-works)

---

## 0. Elevator Pitch

### 0.1 The 30-Second Version

**TenCount turns any boxing sparring video into a per-fighter scorecard.** A coach drops in an MP4; the app returns total punches, a breakdown by punch type (jab, cross, lead/rear hook, lead/rear uppercut), a Light/Medium/Heavy intensity rating for every shot, a punch-by-punch timeline, and an annotated video — without anyone wearing a sensor, tagging frames, or touching a stopwatch. The entire pipeline — detection, tracking, pose, punch classification, intensity scoring — runs as one Next.js + Python web app on a single EC2 box.

### 0.2 The 60-Second Version

Coaches today either count punches by hand (slow, biased, ends at the bell) or strap a $300+ sensor to each glove (intrusive, fragile, doesn't work for archive footage). **TenCount removes both constraints**: vision-only, retroactive analytics on any video you already have.

Under the hood it is a four-stage CV pipeline:

1. A **custom-trained YOLOv11m** detector finds people; an **IoU + Hungarian tracker with inertia and ID recycling** keeps a stable identity on each fighter through occlusions and clinches.
2. **YOLOv8m-pose** extracts 17 COCO keypoints from each fighter's crop, EMA-smoothed and buffered into 25-frame windows (~0.83 s @ 30 fps).
3. A **rule-based per-arm state machine** detects punch *events* (elbow extension + wrist velocity + angle delta, with cooldown and retraction gating).
4. A **custom AttentionBiLSTM** (2-layer BiLSTM, 256 hidden, attention bottleneck, 6-class softmax) classifies each detected punch into one of six punch types from a 75-dim feature vector (normalised keypoints + 7 engineered angles + first-order velocities).
5. A **6-feature intensity model** (wrist impulse, elbow angular velocity, shoulder rotation, jerk, hip impulse, post-peak deceleration) scores each punch as Light / Medium / Heavy, with per-fighter percentile adaptation after five punches.

The frontend is a polished **Next.js 14** app with framer-motion + GSAP + Lenis; the backend is a Node API that spawns one Python subprocess per job and parses newline-delimited JSON events off stdout. The whole stack ships as a single systemd unit on EC2 — provisioned by one bash script (`./deploy.sh deploy`) — and is reachable from any browser.

### 0.3 Why This Is Hard (and Why It's Different)

- **Identity is the foundation, not an afterthought.** Most demo boxing-analytics code treats every detection as a fresh person. TenCount carries fighter identity through occlusions with an inertia-weighted Hungarian matcher and a track-recycling pass that re-attaches a "new" detection to a recently-expired track if it appears in the same place — which is what makes per-fighter counts trustworthy.
- **Two-stage punch logic.** Rule-based detection answers *did a punch happen?* on physically-grounded features (cosine-law elbow angles, normalised wrist velocity). The neural classifier only answers *what kind?*. This split is what lets the system count reliably on a CPU box and still produce six-class typing.
- **Adaptive intensity.** The intensity scorer falls back to global maxima for the first five punches, then switches to **per-fighter 95th-percentile normalisation**, so "heavy" means heavy *for that fighter*, not relative to a hand-picked constant.
- **Production-grade UI for a research-grade pipeline.** The model stack is non-trivial (custom YOLOv11m + YOLOv8m pose + custom AttentionBiLSTM), but the user never sees any of it — they see a drag-and-drop, a five-step progress tracker, and a results page with an annotated video, animated punch bars, and an SVG timeline.

### 0.4 What Ships Today

- One-instance deploy: a **single `./deploy.sh deploy`** stands up AWS EC2 (eu-north-1), key pair, security group, 100 GB EBS, Node 20, Python venv, models, and a systemd unit (`tencount.service`) — and prints the public URL.
- **Up to 500 MB** MP4/WebM uploads, validated client- and server-side.
- **Async job lifecycle** (`detecting → pose → classifying → rendering → complete`) surfaced via 1 Hz polling on `/api/jobs/[jobId]`.
- Output: an **annotated MP4** with persistent labels (1.5 s) and an on-screen legend, plus a **typed JSON result** (totals, per-type breakdowns, timeline) that drives the results page.
- **Six punch classes**: Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut.
- **Three intensity tiers**: Light (<0.40), Medium, Heavy (≥0.70).
- Works on **two fighters + a referee** (the third detection is rendered for context but excluded from stats).
- **Persistent run history** in S3 — every completed analysis is archived to `s3://tencount-runs-fyp-011190986707/runs/<runId>/` (annotated video + heatmap background + full JSON result + summary) and surfaced on a dedicated `/history` page that re-renders past runs with the same `ResultsView` used by the live results page.

### 0.5 The One-Sentence Pitch (for slides)

> **TenCount is a single-page web app that turns any boxing video into per-fighter punch counts, six-class punch-type breakdowns, intensity ratings, and an annotated replay — powered by a custom YOLOv11m + YOLOv8m-pose + AttentionBiLSTM pipeline running on a single EC2 instance.**

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
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │ on job complete: PUT manifest + video
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│       S3 bucket  s3://tencount-runs-fyp-011190986707  (eu-north-1)              │
│                                                                                 │
│   runs/<runId>/summary.json     ← small entry for the /history index            │
│   runs/<runId>/full.json        ← full JobResult (timeline, heatmap, breakdown) │
│   runs/<runId>/output.mp4       ← annotated H.264 video                          │
│   runs/<runId>/heatmap_bg.jpg   ← first-frame snapshot used as heatmap backdrop │
│                                                                                 │
│   /history          ──► ListObjectsV2 (prefix=runs/, delim=/) + parallel GETs   │
│   /history/<runId>  ──► GET full.json + presigned-URL the video (6 h TTL)       │
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
| **Hot storage** | Local disk (`frontend/public/uploads/`) | — | input + intermediate artefacts during a live job |
| **Cold storage / history** | AWS S3 (`tencount-runs-fyp-011190986707`) | — | every completed run is archived; bucket has CORS, presigned-URL access |
| **AWS SDK (Node)** | `@aws-sdk/client-s3`, `@aws-sdk/s3-request-presigner` | v3 | reads creds from `frontend/.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) |
| **Live-job state store** | In-memory `Map` in Node | — | lost on restart, but completed runs survive in S3 |
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
    ├── .env                      ← AWS creds + S3 bucket name (gitignored)
    ├── app/
    │   ├── layout.tsx            ← fonts, SmoothScrollProvider, dark theme
    │   ├── page.tsx              ← LandingPage
    │   ├── globals.css           ← design tokens (--bg, --accent, etc.)
    │   ├── analyze/page.tsx
    │   ├── processing/[jobId]/page.tsx
    │   ├── results/[jobId]/page.tsx
    │   ├── history/page.tsx              ← list page (wraps <HistoryList/>)
    │   ├── history/[runId]/page.tsx      ← detail page (wraps <ResultsView runId=…/>)
    │   └── api/
    │       ├── upload/route.ts           ← POST, spawns Python + S3 archive on complete
    │       ├── jobs/[jobId]/route.ts     ← GET, reads in-memory jobStore (live job)
    │       ├── uploads/[...path]/route.ts ← static file serve w/ range hdrs
    │       ├── runs/route.ts             ← GET, list past runs (S3 prefix scan)
    │       └── runs/[runId]/route.ts     ← GET, fetch full.json + presigned video URL
    ├── components/
    │   ├── LandingPage.tsx
    │   ├── Nav.tsx                       ← has History link
    │   ├── VideoDropzone.tsx
    │   ├── ProcessingView.tsx
    │   ├── ResultsView.tsx               ← accepts {jobId} or {runId}; same visuals
    │   ├── HistoryList.tsx               ← grid of past-run cards
    │   ├── FighterHeatmap.tsx
    │   ├── ThemeToggle.tsx
    │   └── SmoothScrollProvider.tsx
    ├── lib/
    │   ├── job-store.ts          ← in-memory Map<string, JobResult>
    │   ├── s3.ts                 ← S3Client + put/get/list/presign helpers
    │   └── types.ts              ← JobResult / FighterResult / PunchEvent / RunSummary
    └── public/
        └── uploads/              ← input + annotated output videos live here (hot)
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
| `/results/[jobId]` | `app/results/[jobId]/page.tsx` | Annotated video + per-fighter stats + timeline (live job, in-memory) |
| `/history` | `app/history/page.tsx` | Grid of all archived runs; reads `/api/runs` |
| `/history/[runId]` | `app/history/[runId]/page.tsx` | Past-run detail; renders the **same** `<ResultsView />` with data loaded from S3 |

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

#### `GET /api/runs`  (`app/api/runs/route.ts`)

Returns the list of past runs from S3.

1. `s3ListRunIds()` calls `ListObjectsV2` with `Prefix='runs/'` and `Delimiter='/'`. Common prefixes come back as `runs/<runId>/`; the helper strips the prefix and trailing slash to return raw run ids.
2. For each run id, `s3GetJSON<RunSummary>(runs/<runId>/summary.json)` is fetched **in parallel** (`Promise.all`).
3. Nulls (missing summaries) are filtered out and the remaining records are sorted by `completedAt` descending.
4. Returns `{ runs: RunSummary[] }`. Marked `dynamic = 'force-dynamic'` so Next never tries to cache or pre-render it.

#### `GET /api/runs/[runId]`  (`app/api/runs/[runId]/route.ts`)

Returns a single archived run, in the same shape `ResultsView` expects.

1. `s3GetJSON<JobResult>(runs/<runId>/full.json)` reads the archived job result. 404 if absent.
2. `videoUrl` and `heatmap.bgUrl` are stored in S3 as relative keys; the route swaps them for **presigned URLs** with a 6 h TTL (`SIGNED_URL_TTL`) via `s3SignedUrl()`. The browser then plays the video directly from S3.
3. Returns the patched `JobResult`.

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
  videoUrl?: string         // /uploads/{id}.{ext} (live) or s3-key/presigned-url (history)
  duration?: number; fps?: number; frameCount?: number
  fighters?: [FighterResult, FighterResult]
  timeline?: PunchEvent[]
  heatmap?: HeatmapData     // spatial occupancy + dominance + bg image
  error?: string
  startedAt: number; completedAt?: number
}

interface RunSummary {
  runId: string
  originalFilename: string
  completedAt: number
  duration?: number
  fps?: number
  totalPunches: number
  f1Punches: number
  f2Punches: number
}
```

The **live** store is a global `Map<string, JobResult>` in `lib/job-store.ts`, preserved across HMR in dev via `globalThis`. A server restart wipes in-progress jobs — but **completed runs survive in S3**, fetched on demand via the `/api/runs/*` endpoints.

`RunSummary` is the lightweight projection written to `runs/<runId>/summary.json` so the `/history` page can list dozens of runs without parsing each `full.json`.

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
| `AWS_ACCESS_KEY_ID` | — | Node AWS SDK (via `frontend/.env`) — needed by `lib/s3.ts` |
| `AWS_SECRET_ACCESS_KEY` | — | Node AWS SDK (via `frontend/.env`) |
| `AWS_REGION` | `eu-north-1` | Node AWS SDK fallback |
| `S3_BUCKET` | `tencount-runs-fyp-011190986707` | `lib/s3.ts` (override target bucket) |
| `S3_REGION` | `eu-north-1` | `lib/s3.ts` (S3 client region) |
| AWS creds (CLI) | `~/.aws/credentials` or env | `deploy.sh` (developer host only) |

The `frontend/.env` file holds AWS credentials in plaintext and is **gitignored**. The deploy tarball (`deploy.sh` line ~134) does **not** exclude `.env`, so a fresh `./deploy.sh deploy` ships the credentials to the EC2 host along with the rest of the codebase. Future hardening: attach an IAM role to the EC2 instance with an inline policy granting only `s3:PutObject`, `s3:GetObject`, and `s3:ListBucket` on `tencount-runs-fyp-011190986707/*`, then remove the static keys from `.env`.

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

1. **In-memory job store for *in-flight* jobs** (`lib/job-store.ts`). A server restart, redeploy, or systemd `Restart=on-failure` wipes every job that is **currently running**. (Completed runs are safe — they're archived to S3 the moment the Python process exits cleanly.) The file itself explicitly comments: *"Replace with Redis/DB in production."*
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

## 16. S3-Backed Run History (Persistence Layer)

A dedicated persistence layer was added so the app remembers analyses across server restarts, deploys, and instance replacements. It is intentionally minimal — **no database, no auth, no user accounts** — and uses a single S3 bucket as a flat object store.

### 16.1 Bucket Topology

```
s3://tencount-runs-fyp-011190986707/         (region: eu-north-1)
└── runs/
    ├── <runId-A>/
    │   ├── summary.json     ← ~200 bytes, used by /history list
    │   ├── full.json        ← full JobResult (timeline + heatmap grid + breakdown)
    │   ├── output.mp4       ← annotated H.264 video
    │   └── heatmap_bg.jpg   ← first-frame snapshot for the heatmap backdrop
    ├── <runId-B>/
    │   └── ...
    └── ...
```

- **`runId`** = the same `jobId` minted at upload time (`<base36-timestamp>-<6-char-random>`), so a live run and its archived copy share an identifier.
- **CORS** is enabled (`GET, HEAD` from `*`) so a browser can play the annotated MP4 via a presigned URL without proxying through Next.
- **Public access is blocked**; the only way to read an object is via a presigned URL or a credentialed SDK call.
- **Object ownership: BucketOwnerEnforced** (ACLs disabled) — the default for new buckets in AWS since 2023.

### 16.2 Why Split `summary.json` and `full.json`

`full.json` includes the heatmap grid encoded as base64-bytes (64 × 64 × 2 fighters ≈ 11 KB per run after base64) plus the punch timeline and per-frame breakdown — fast for one run, slow if you load fifty to render an index page. `summary.json` is a tiny denormalised projection (`{runId, originalFilename, completedAt, duration, fps, totalPunches, f1Punches, f2Punches}`) sized in the low hundreds of bytes, which keeps the `/history` list snappy without an in-app cache.

### 16.3 The `lib/s3.ts` Helper

A single module wraps the AWS SDK so the rest of the codebase never imports `@aws-sdk/*` directly.

```ts
export const s3 = new S3Client({ region: S3_REGION })

export function runKey(runId: string, file: string): string
export async function s3PutBuffer(key: string, body: Buffer|string, contentType: string): Promise<void>
export async function s3GetJSON<T>(key: string): Promise<T | null>      // null on NoSuchKey
export async function s3SignedUrl(key: string): Promise<string>          // 6 h TTL
export async function s3ListRunIds(): Promise<string[]>                  // via CommonPrefixes
```

- `S3_BUCKET` and `S3_REGION` resolve from env vars at module load with sensible defaults.
- Credentials are picked up by the SDK's **default provider chain** — env vars (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) on local + deployed boxes, IAM role if one is later attached to the EC2 instance.
- `s3GetJSON()` catches `NoSuchKey` / `NotFound` and returns `null`; all other errors propagate so the calling route can 5xx.
- `s3ListRunIds()` uses `Delimiter: '/'` to enumerate only top-level "folders" under `runs/` — O(1) S3 calls regardless of how many objects each run holds.

### 16.4 Write Path: Archive-on-Complete

`app/api/upload/route.ts` already drains stdout from the Python pipeline into the in-memory `jobStore`. When the process exits 0 and the final `update({...})` runs, an extra step archives the job:

```ts
const finalJob = jobStore.get(jobId)
if (finalJob) {
  archiveRunToS3(finalJob, outputVideoAbsPath, heatmapData).catch(err => {
    console.error('[s3 archive] failed:', err)
  })
}
```

`archiveRunToS3()` (defined at the bottom of the same route):

1. Reads the annotated MP4 from disk (`readFile`) and `PutObject`s it to `runs/<runId>/output.mp4` with `Content-Type: video/mp4`.
2. If a heatmap exists, reads `frontend/public/uploads/<basename(bgUrl)>` and uploads it under `runs/<runId>/heatmap_bg.<ext>` with the matching MIME (jpeg/png/webp).
3. Constructs an "archived" `JobResult` where `videoUrl` and `heatmap.bgUrl` are rewritten from local paths (`/uploads/...`) to **S3 keys** (`runs/<runId>/output.mp4`). This keeps `full.json` portable — the eventual presigned URL is generated at read time, not bake time.
4. Writes `runs/<runId>/full.json` and `runs/<runId>/summary.json` (both `application/json`).

The archive call is **fire-and-forget** — failures log but do not affect the in-memory job state that the live `/results/[jobId]` page is polling. This means a live user sees their results regardless of S3 health, and only the history feature degrades if uploads fail.

### 16.5 Read Path: List + Detail

`/api/runs` (list):

```
ListObjectsV2(Prefix='runs/', Delimiter='/')
  → CommonPrefixes → run ids
  → Promise.all(s3GetJSON<RunSummary>(...summary.json))
  → filter nulls
  → sort by completedAt desc
```

`/api/runs/[runId]` (detail):

```
GetObject(runs/<runId>/full.json) → JobResult
  → presign(runs/<runId>/output.mp4)       → videoUrl
  → presign(runs/<runId>/heatmap_bg.jpg)   → heatmap.bgUrl
  → return patched JobResult
```

Presigned URLs are valid for 6 hours (`SIGNED_URL_TTL = 60 * 60 * 6`). A user opening a history detail page gets fresh URLs on every request; the URL itself is the only thing the browser uses to access the bucket.

### 16.6 UI Wiring

Two pages and one new component:

- `app/history/page.tsx` — server component, wraps `<Nav />` + `<HistoryList />`. Marked `dynamic = 'force-dynamic'` so list ordering is always fresh.
- `components/HistoryList.tsx` — client component, `fetch('/api/runs')`, renders a responsive `grid-cols-1 md:grid-cols-2 xl:grid-cols-3` of run cards. Each card shows the filename, a relative-time label (`"3h ago"`), the F1/F2 punch split, a winner badge, and a stable `#{runId.slice(0,8)}` short id. Empty state has a CTA back to `/analyze`.
- `app/history/[runId]/page.tsx` — server component, wraps `<Nav />` + `<ResultsView runId={params.runId} />`.

The cohesion trick that makes "loads exactly like our current outputs page" true: **`ResultsView` was refactored to accept either `{jobId}` or `{runId}`** via a discriminated union. The component picks `/api/jobs/${id}` or `/api/runs/${id}` based on which prop is present, and otherwise its rendering is unchanged. One component → two data sources → identical visuals.

### 16.7 Local + Deployed Configuration

Both the local dev box (`npm run dev` in `frontend/`) and the deployed EC2 instance (systemd `tencount.service`) read **the same `frontend/.env` file**:

```dotenv
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-north-1
S3_BUCKET=tencount-runs-fyp-011190986707
S3_REGION=eu-north-1
```

- The file is in `.gitignore` (matches both `.env` and `.env.local`) — credentials never reach git history.
- `deploy.sh` does **not** explicitly exclude `.env` from the upload tarball, so a fresh `./deploy.sh deploy` ships the file as-is to `/home/ubuntu/tencount/frontend/.env`. Next.js auto-loads `.env` for both dev and production builds.
- Local and deployed apps therefore write to and read from the **same bucket** — a run analysed on a developer laptop is visible on the production `/history` page within seconds, and vice versa.

### 16.8 What This Doesn't Do (and Why That's OK)

- **No auth.** Anyone who can reach `/history` can list all runs. Fine for a single-user FYP demo; for production add a session layer in front of `/api/runs/*`.
- **No retention policy.** Runs accumulate forever. Add an S3 lifecycle rule (`Expiration: 90 days`) on the `runs/` prefix when storage cost becomes a concern.
- **No deletion endpoint.** History is append-only; pruning requires `aws s3 rm`.
- **No multipart upload.** The annotated video is uploaded in a single `PutObject`. With a 500 MB upload cap on the frontend and typical annotated outputs under 100 MB, this is well within the 5 GiB single-`PutObject` limit.
- **No cross-region replication.** A region-wide outage in `eu-north-1` makes history temporarily unavailable; the live job pipeline is unaffected because it never touches S3 during inference.

---

## 17. Feature-by-Feature Implementation Reference

This section documents every user-visible feature and the exact files / functions that implement it, organised by user-journey order. Use it as the "where do I look?" map.

### 17.1 Landing Page (`/`)

**What it does** — hero with animated headline, scroll-driven manifesto, sticky pipeline showcase, feature cards.

**Implementation** — `app/page.tsx` is a thin server-component wrapper around `<LandingPage />`. The work is in `components/LandingPage.tsx`:

- **Hero entrance** — GSAP timeline on mount: stagger-in headline characters, slide-up sub-text, fade-in CTA.
- **Scroll-driven manifesto** — GSAP `ScrollTrigger` pins the manifesto block while the inner text fades through three states.
- **Sticky pipeline showcase** — each pipeline stage uses `ScrollTrigger` `pin: true, scrub: true`; the inner art swaps via opacity tweens timed to the scroll position.
- **Smooth scroll** — `components/SmoothScrollProvider.tsx` instantiates Lenis once and pipes its frame callback into the GSAP ticker (`gsap.ticker.add(t => lenis.raf(t * 1000))`), so GSAP and Lenis share one rAF.
- **Performance hint** — sticky pinned sections get `will-change: transform` so the GPU compositor doesn't repaint.

### 17.2 Upload Page (`/analyze`)

**What it does** — drag-and-drop or click-to-pick a video file, validates size/MIME, uploads with a progress bar.

**Implementation** — `app/analyze/page.tsx` splits the screen into a left "identity panel" (system overview + stats) and a right `<VideoDropzone />`.

`components/VideoDropzone.tsx`:

- State machine: `idle → dragging → selected → uploading → error` (single `state` union).
- **Drop / select** — `onDragOver` toggles `dragging`; `onChange` of a hidden `<input type=file>` handles click-to-pick.
- **Preview** — `URL.createObjectURL(file)` produces a blob URL; the dropzone shows a `<video>` thumbnail with hover-to-play. Revoked on unmount.
- **Upload** — uses `XMLHttpRequest` (not `fetch`) specifically so `onprogress` can drive a determinate progress bar. POSTs `multipart/form-data` with key `video` to `/api/upload`.
- **Success** — server returns `{jobId}`; the component calls `router.push(\`/processing/${jobId}\`)`.

### 17.3 Inference Job Lifecycle

**What it does** — kicks off the Python pipeline, streams its output back to the browser as a 5-step progress UI, hands off to results on completion.

**Implementation** — three files cooperate:

1. `app/api/upload/route.ts` — validates the multipart, writes the file to `public/uploads/<jobId>.<ext>`, seeds `jobStore` with `{status:'detecting', progress:8}`, and `child_process.spawn`s `frontend_runner.py`. Spawned with `cwd = FYP_ROOT` and inherited env (`PYTHON_BIN`, `PROJECT_ROOT`).
2. `frontend_runner.py` — patches model paths into `boxing_analytics_v2`, monkey-patches `builtins.print` to emit JSON events on stdout (progress, punch events, per-fighter totals, breakdown, output path, heatmap, done, error), then calls `ba.main()`.
3. The route's `proc.stdout.on('data')` callback splits on `\n`, parses each line as JSON, and mutates `jobStore.get(jobId)` accordingly. Status transitions: `detecting` (v<35) → `pose` (v<65) → `classifying` (v<95) → `rendering` → `complete`.

**Progress UI** — `components/ProcessingView.tsx`:

- Polls `GET /api/jobs/[jobId]` at 1 Hz with `cache: 'no-store'`.
- Renders five step cards (Person Detection, Pose Estimation, Punch Classification, Rendering, Complete) — each animated with framer-motion springs as it activates.
- Tracks elapsed time on a separate `setInterval` so the timer doesn't depend on the polling cadence.
- On `status === 'complete'`, calls `router.push(\`/results/${jobId}\`)`.

### 17.4 Fighter Identity Tracking

**What it does** — keep "fighter 1" and "fighter 2" stable across the whole video, even when they cross, clinch, or briefly leave frame.

**Implementation** — entirely in `boxing_analytics_v2.py`:

- **`match_detections_to_tracks()`** — every frame, builds an `(n_detections × n_tracks)` cost matrix from `1 - IoU`, then solves with `scipy.optimize.linear_sum_assignment` (Hungarian). Existing fighter tracks get an `INERTIA_BONUS = 50000` subtracted from their column, dominating the assignment whenever fighters are still visible.
- **`_recycle_fighter_id()`** — when a "new" detection appears within normalised centroid distance 0.8 of a recently-expired fighter track (within `MAX_TRACK_HISTORY = 30` frames), the new detection inherits the old track id, its punch counts, and its keypoint buffer. This is what survives brief occlusions and false-loss events.
- **Fighter slot assignment** — once the tracker has stable ids, the **top-2 by score** are flagged `is_fighter=True` (`MAX_FIGHTERS = 2`). The highest-confidence non-overlapping third person becomes the referee — visually annotated but excluded from stats.
- **TS-side canonicalisation** — even with all of the above, YOLO can occasionally reissue a fresh id mid-fight. The upload route therefore reads `slot_raw_ids` from the Python heatmap event (the authoritative map from canonical slot 1/2 to raw track ids) and merges all stats keyed by raw id into slots 1 and 2. If no heatmap event arrived, it falls back to ranking raw ids by punch count.

### 17.5 Pose Estimation Per Fighter

**What it does** — extract 17 COCO keypoints (head + 4 limbs) per fighter per frame, smoothed for stability.

**Implementation** — also in `boxing_analytics_v2.py`:

- **Crop selection** — each fighter's bbox is EMA-smoothed (`BBOX_EMA_ALPHA = 0.3`) before cropping; a tight crop reduces pose-model context and makes the heat map cleaner.
- **Pose model** — `pose_model(crop, conf=0.25)` calls YOLOv8m-pose; output is `(17, 3)` (x, y, conf).
- **Visibility filter** — frames with fewer than `MIN_VISIBLE_KEYPOINTS = 7` confident joints are dropped; the buffer continues with the last good frame.
- **Normalisation** — keypoints are bbox-relative `[0, 1]` so they're scale-invariant.
- **Smoothing** — per-joint EMA with `CLF_KPT_SMOOTH_ALPHA = 0.4`. Pushed into a fixed-length `deque(maxlen=25)` per fighter (one window per arm).
- **Gap repair** — frame-id bookkeeping is parallel; gaps of ≤ 2 frames are bridged by repeating the last good keypoints.

### 17.6 Punch Detection (Rule-Based)

**What it does** — decide *did a punch happen* on each arm of each fighter, independently of *what kind*.

**Implementation** — `FighterState.check_punch()` runs every frame for each arm:

1. Smooth shoulder/elbow/wrist over a 5-frame window.
2. Compute elbow angle via the cosine law.
3. Compute wrist velocity normalised by shoulder width (so a far-away fighter doesn't appear slower).
4. **Trigger** when ALL hold: elbow angle ≥ `PUNCH_ANGLE_THRESHOLD = 155°`, angle Δ ≥ `MIN_ANGLE_DELTA = 15°`, wrist velocity ≥ `MIN_WRIST_VELOCITY = 0.25`.
5. **Debounce** — `PUNCH_COOLDOWN = 20` frames (~0.67 s) per arm.
6. **Retraction gate** — after firing, the arm must drop below `RESET_ANGLE_THRESHOLD = 120°` before another punch on the same arm registers (prevents "double counting" when a fighter holds the lead arm extended).

### 17.7 Punch Classification (AttentionBiLSTM)

**What it does** — given a 25-frame keypoint window leading up to a detected punch, decide which of six types it is.

**Implementation** — `models/punch_classifier.pt` is a custom AttentionBiLSTM defined in `boxing_analytics_v2.py`:

- **Input** — `(1, 25, 75)` tensor: 25 frames × 75 features (34 hip-centred, torso-height-scaled keypoint coordinates + 7 engineered angles + 34 first-order velocities).
- **Architecture** — 2-layer BiLSTM (256 hidden, dropout 0.4) → 64-unit tanh attention bottleneck → attention-weighted sum across the sequence → LayerNorm → 128-unit GELU dense → softmax over 6 classes (jab / cross / lead_hook / rear_hook / lead_uppercut / rear_uppercut).
- **Orientation handling** — left-facing fighters have their keypoint buffer mirrored (`_mirror_buffer`) before inference so the model sees the same canonical orientation it was trained on.
- **Min-frame gate** — punches with fewer than `CLF_MIN_FRAMES = 15` contiguous frames in the buffer skip classification and contribute to the total only.

### 17.8 Intensity Scoring

**What it does** — score every punch as Light / Medium / Heavy based on physical features, then *adapt to each fighter*.

**Implementation** — `compute_intensity()` evaluates six scalars on a wider 15-frame window:

1. Cumulative wrist-velocity impulse.
2. 95th-percentile elbow angular velocity.
3. 95th-percentile shoulder rotation speed.
4. 95th-percentile wrist jerk.
5. Hip-impulse contribution.
6. Post-peak wrist deceleration (penalises whiffs).

- **Normalisation** — for the first 5 punches in a fight, each scalar is normalised by a global maximum (`INTENSITY_V_MAX = 4.0`, etc.). After that, normalisation switches to **per-fighter 95th percentiles** maintained in a running history — so "heavy" is heavy *for this specific fighter*.
- **Type weights** — different punch types get different feature weights (a hook weights shoulder rotation more than a jab).
- **Thresholds** — Light (<0.40) / Medium / Heavy (≥0.70).
- **Annotation** — the active punch shows the intensity tier in colour for `LABEL_PERSIST_FRAMES = 45` frames (1.5 s).

### 17.9 Annotated Output Video

**What it does** — produce a playable MP4 with bounding boxes, fighter ids, punch labels, an on-screen legend, and a frame counter.

**Implementation** — `boxing_analytics_v2.py` opens a `cv2.VideoWriter` with `mp4v` fourcc at the input's native fps/resolution. Each frame is drawn over (rectangles, putText) then written.

- **H.264 transcode** — the H.264 encoder fixup is done as a post-pass: after the raw `mp4v` write completes, ffmpeg transcodes to true H.264 so browser `<video>` elements actually play it (Safari rejects `mp4v`). The final path is what gets emitted via the `{"t":"output"}` event.
- **Label persistence** — punch type/intensity labels stay visible for 1.5 s after the punch fires so a viewer can read them at 30 fps.
- **Legend** — bottom-right legend lists every punch class with its colour; rendered every frame to keep encoders happy.

### 17.10 Results Page (`/results/[jobId]`)

**What it does** — show the annotated video, per-fighter totals, an animated punch-type breakdown, a timeline of every punch, and the spatial heatmap.

**Implementation** — `app/results/[jobId]/page.tsx` wraps `<ResultsView jobId={params.jobId} />`.

`components/ResultsView.tsx`:

- **Data loader** — accepts either `{jobId}` or `{runId}` (discriminated union). Computes `endpoint = jobId ? '/api/jobs/<id>' : '/api/runs/<id>'`. On mount, fetches once with `cache: 'no-store'` and stores into local state. (For live jobs it errors if `status !== 'complete'`; for runs it never checks because runs are always complete.)
- **Video player** — custom `<VideoPlayer>` (defined in the same file) with play/pause, mute, click-to-seek, click-to-toggle on the video itself, and a `<a download>` link with a generated filename `tencount_<id>_output.mp4`.
- **Animated bars** — `<PunchBar>` animates `width: 0 → value/max * 100%` with a framer-motion spring (`stiffness: 80, damping: 20`).
- **Fighter card** — `<FighterCard>` shows total punches in a large mono font + an animated bar per punch type. The "more punches" trophy badge highlights the fighter with the higher count.
- **Punch timeline** — `<PunchTimeline>` lays out every punch event as a thin vertical SVG bar at `(x = time/duration * 100%)`, colour-coded by punch type, positioned above the axis for F1 and below for F2. Each marker animates `scaleY: 0 → 1` with a tiny per-index stagger.
- **Spatial heatmap** — `<FighterHeatmapSection>` (from `components/FighterHeatmap.tsx`) decodes the base64 grids from `HeatmapData`, renders them on a canvas overlayed on the first-frame snapshot (`bgUrl`), and shows dominance / zone stats.

### 17.11 Spatial Heatmap

**What it does** — visualise where each fighter spent their time on the canvas.

**Implementation** —

- Python side: `boxing_analytics_v2.py` accumulates a `64 × 64` occupancy grid per fighter (incremented at the foot-position of every confident detection), exports the grids as base64-encoded `uint8` row-major bytes, computes a "dominance" map (which fighter occupied each cell more), centre-control percentages (% of intensity inside the central 30%), and a normalised centroid. All packaged in a single `{"t":"heatmap", ...}` JSON event emitted near the end of the run.
- Node side: the route parses the heatmap event, maps Python absolute paths to public `/uploads/...` URLs for the first-frame snapshot, and stuffs everything into `HeatmapData`.
- React side: `<FighterHeatmapSection>` renders the grids as semi-transparent canvas overlays and reads dominance/centroid into stat panels.

### 17.12 History List (`/history`)

**What it does** — list every past run with quick stats; click → detail.

**Implementation** — `components/HistoryList.tsx` calls `GET /api/runs`, renders a responsive grid (`md:2 cols, xl:3 cols`) of `<Link>` cards animated in with framer-motion staggers (`delay: i * 0.04`). Each card includes:

- Filename (truncated)
- Relative time + absolute date
- Total punches
- F1 vs F2 bar split with winner-trophy badge
- Duration, fps, short id (`#<runId[:8]>`)
- Arrow that animates `translate-x` on hover

Empty state has its own CTA back to `/analyze`.

### 17.13 History Detail (`/history/[runId]`)

**What it does** — render a past run exactly like the live results page.

**Implementation** — `app/history/[runId]/page.tsx` is one line of substance: `<ResultsView runId={params.runId} />`. Because `ResultsView` was refactored to be data-source-agnostic, the visual rendering is byte-identical to the live page; the only differences are the API endpoint it hits and the fact that `videoUrl` / `heatmap.bgUrl` are presigned S3 URLs instead of local `/uploads/...` paths.

### 17.14 Theme Toggle

**What it does** — switch between dark (default) and light themes.

**Implementation** — `components/ThemeToggle.tsx` toggles a `data-theme` attribute on `<html>`. Tokens in `globals.css` are defined twice, once for dark and once for `[data-theme="light"]`. No `dark:` Tailwind variant is used.

### 17.15 Navigation

**What it does** — fixed top nav with logo, History link, GitHub link, and a primary Analyse CTA.

**Implementation** — `components/Nav.tsx`. Static server component, no client interactivity beyond standard `<Link>` navigation.

### 17.16 Deployment

**What it does** — one-shot AWS EC2 provisioning + lifecycle management.

**Implementation** — `deploy.sh` with sub-commands `deploy | start | stop | status | ssh | teardown`. See [§11](#11-deployment--infrastructure) for the step-by-step. The S3 bucket used by the app is **not** created by this script — it was provisioned once via `aws s3api create-bucket` and CORS-configured via `aws s3api put-bucket-cors`; it lives outside the deploy script's lifecycle so a `teardown` never destroys archived runs.

---

## 18. ELI5: How TenCount Works

(For anyone who wants the whole system explained like you're five.)

### What is TenCount?

Imagine a boxing coach sitting at a laptop. They've recorded a sparring session and want to know: **who threw more punches, what kinds, how hard, and when?** Counting by hand takes forever and is biased — the coach blinks, the bell rings, they lose count.

TenCount is a website where the coach drops in their video and the computer does the counting for them. Within a few minutes they get back: a punch tally for each fighter, a breakdown of jabs vs crosses vs hooks vs uppercuts, an animated timeline showing every punch, a heatmap of where each fighter stood, and the original video with little boxes drawn around each fighter and labels above each punch.

### How does it look at a video?

The computer can't watch a video the way you and I do — to it, a video is just a fast-flipping flipbook of still images. So we look at one image (one **frame**) at a time. Boxing footage is usually 30 frames per second, so a one-minute video is 1,800 separate pictures.

For each picture, the computer goes through a little checklist:

1. **Who's in the picture?** A program called **YOLOv11m** scans the image and draws a rectangle around every person it finds. Think of it as a really fast pair of eyes that's been trained on millions of pictures of people. We trained our own version specifically to be good at spotting boxers.

2. **Which rectangle is "fighter 1" and which is "fighter 2"?** This is harder than it sounds, because in the *next* frame the rectangles move and we have to figure out which new rectangle is the same person. We use a math trick called the **Hungarian algorithm** to do this matching — it's like sorting socks: each new sock (rectangle) wants to pair with the most-similar old sock (the previous rectangle). We also give bonus points to existing fighters, so the computer doesn't accidentally relabel them whenever a referee walks through. If a fighter briefly disappears (clinch, occlusion), and a new rectangle pops up nearby a few frames later, we **recycle** the old id so the punch count doesn't reset.

3. **What's each fighter's body doing?** For each fighter's rectangle, a second program called **YOLOv8m-pose** picks out 17 points on their body — head, shoulders, elbows, wrists, hips, knees, ankles. These are the "keypoints". Now we know exactly where the elbows and wrists are.

4. **Did somebody just punch?** Punching has a very specific physical fingerprint: the elbow goes from bent to straight quickly, and the wrist moves fast away from the body. So for each arm, we watch the elbow angle. If it crosses 155° AND the change is sharp AND the wrist is moving fast — that's a punch. We then ignore that arm for 20 frames (~ 2/3 of a second) so we don't count the same punch twice.

5. **What kind of punch was it?** Once we know *a* punch happened, we hand the last 25 frames of body keypoints to a tiny brain called the **AttentionBiLSTM** (it's a fancy neural network). It looks at how the body moved during the punch and decides: jab, cross, lead hook, rear hook, lead uppercut, or rear uppercut. The "attention" part means it focuses on the moments in the 25-frame window that matter most (usually right around the elbow extending).

6. **How hard was the punch?** We measure six things — how fast the wrist moved, how fast the elbow opened, how fast the shoulders rotated, how jerky the motion was, how much the hips contributed, how cleanly the punch decelerated. We add them up with different weights for different punch types. Then we compare to *that fighter's own* recent punches: anything in the top 30 % for them is "Heavy", the next chunk is "Medium", the bottom is "Light". So "heavy" means *heavy for them* — a heavy uppercut from a featherweight isn't the same as a heavy uppercut from a heavyweight, and that's fine.

7. **Where did they fight?** Every time we see a fighter standing somewhere, we add a tiny dot to a 64×64 grid for that fighter. After the whole video, the grid is dense where they stood a lot and sparse where they didn't. Stretched and tinted, this becomes the "heatmap" — a visual answer to "did Fighter 1 own the centre? Did Fighter 2 fight off the ropes?".

8. **Draw it on the video.** Now we re-export the video with all of this info painted on: a rectangle around each fighter, a label like "Fighter 1 — Rear Hook — Heavy" above them when they punch, an on-screen legend, and a frame counter. The output is a normal MP4 file the coach can play or share.

### How does the website fit around all this?

The brain doing the analysis is written in Python. The website itself is written in TypeScript using a framework called Next.js (basically React with a built-in server). When the coach uploads a video:

1. The website grabs the file and asks the Python brain to start working on it (a separate program that runs alongside the website).
2. The Python brain prints little JSON status updates to a stream — "I'm 5 % done", "I'm 50 % done", "Fighter 1 just threw a jab", "I'm finished, here's the file path of the new video".
3. The website reads those updates and shows the coach a progress bar with 5 stages (Detecting → Pose → Classifying → Rendering → Done).
4. When it's done, the website shows the results page: video player + fighter cards + bar charts + timeline + heatmap.

### How do we remember past videos?

This is the part we added most recently. Without it, every time the website restarted (deploys, crashes, server reboots), all the analysis results would vanish. That's bad.

So now, the *moment* a run finishes successfully, the website also packs up four files and uploads them to **Amazon S3** (Amazon's giant online "folder in the sky"):

- The annotated video.
- The heatmap background image (a snapshot of the first frame of the fight).
- A big JSON file with all the numbers (timeline, breakdown, etc.).
- A tiny summary JSON file with just the most-important numbers — name, date, totals.

Everything gets organised into a per-run folder named after the run id, like `runs/abc123/`. The summary file is on purpose tiny — so when the coach later clicks the **History** tab, the website can list 50 runs in a fraction of a second by reading just the summaries.

When the coach clicks one of those past runs, the website fetches the big JSON file from S3, generates a short-lived "presigned URL" for the video (a temporary public link, valid for 6 hours), and renders the page using the *exact same* component as the live results page. So it looks identical — just sourced from S3 instead of from a fresh analysis.

### How is it all hosted?

The website lives on one small computer (a virtual server) rented from Amazon, somewhere in Sweden (the `eu-north-1` AWS region). A little script called `deploy.sh` knows how to:

- Rent a fresh computer.
- Install Python, Node, all the AI libraries.
- Upload the website and the AI models.
- Start the website running automatically, and restart it if it crashes (via something called **systemd**).
- Print the website's IP address so we can visit it.

The same script can also stop the computer (to save money — we only pay when it's running) and start it again later. The S3 bucket is separate from this computer — even if we throw the computer away, all the past runs are safe.

### Why is this hard?

A few things make it harder than a typical "AI demo":

- **Identity stays put.** Most demos treat every detection as a brand-new person. We work hard to keep "Fighter 1" being Fighter 1 the whole video, even when fighters clinch or briefly leave the frame. Without this, the punch counts are nonsense.
- **Two-step punch logic.** Instead of asking a neural network "is this a punch?" (which is unreliable on small datasets), we use physics to detect *that* a punch happened, and only use a neural network to decide *what kind* of punch. This makes the system work reliably without huge training data.
- **Intensity that means something.** "Heavy" relative to a hand-picked number is meaningless. "Heavy" relative to a fighter's own recent punches captures actual effort.
- **Looks like a real product.** The model stack is research-grade (custom detector + pose + custom classifier + intensity model), but the user never sees that — they see a drag-and-drop, a progress bar, animated cards, and a clean video.

### The TL;DR

You drop in a boxing video. A custom-trained AI finds the fighters, tracks them, watches their elbows, calls punches, classifies them, scores their intensity, and renders an annotated video. Everything is stored in S3 forever so you can come back and re-watch any past run on the History page. The whole thing runs as one Next.js + Python app on one rented Amazon computer, and it's all you need.

---

*This document was generated from a full source-level review of `boxing_analytics_v2.py`, `frontend_runner.py`, the Next.js frontend (`app/`, `components/`, `lib/`), `deploy.sh`, and the S3 persistence layer (`lib/s3.ts`, `app/api/runs/*`, `app/history/*`). Refer back to the cited file paths and constants for ground truth; any divergence between this document and the code is a bug in this document.*
