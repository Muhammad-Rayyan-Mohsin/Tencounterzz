# TenCount — Comprehensive Project Report

**Project:** Boxing Punch Detection & Analytics (FYP)
**Root:** `/Users/haider/University/FYP/Tencounterzz`
**Branch:** `main` · Latest commit: `haider/final-deployed-aws`
**Stack (one-liner):** Next.js 14 + TypeScript UI → Python bridge → YOLOv11m + YOLOv8m-pose + AttentionBiLSTM pipeline → AWS EC2 (Ubuntu 24.04)

---

## 1. What TenCount Does

A full-stack system that takes fight footage and returns a complete biomechanical breakdown: per-fighter punch counts, six-class punch type breakdown (Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut), a per-event timeline, an annotated H.264 output video, and intensity scoring (Light/Medium/Heavy) per punch.

The user journey is a three-page flow:

1. **Landing** (`/`) — cinematic marketing page explaining the pipeline.
2. **Analyze** (`/analyze`) — a drag-and-drop zone (up to 500 MB, MP4/MOV/AVI/WebM).
3. **Processing** (`/processing/[jobId]`) — live step-by-step progress polled from the backend.
4. **Results** (`/results/[jobId]`) — annotated video player + per-fighter stat cards + color-coded punch timeline.

---

## 2. Repository Layout

```
Tencounterzz/
├── boxing_analytics_v2.py          46 KB — core CV inference pipeline
├── frontend_runner.py              6.4 KB — Python↔Node JSON bridge
├── requirements.txt                ultralytics, opencv-headless, numpy, scipy, torch
├── deploy.sh                       11.7 KB — AWS EC2 deploy script
├── DEPLOYMENT.md                   deploy documentation
├── .deploy-state                   stores the current EC2 instance ID
├── .dockerignore / .gitignore      build/VCS exclusions
├── models/
│   └── punch_classifier.pt         9.4 MB — AttentionBiLSTM weights
├── Tracking and Counting/
│   └── yolov8m-pose.pt             53 MB — pose model
├── runs/person_detect/…/weights/
│   └── best_potential.pt           YOLOv11m detector weights (custom-trained)
└── frontend/                       Next.js 14 app
    ├── app/
    │   ├── layout.tsx              root layout: Geist + Plus Jakarta + Cormorant + Lenis
    │   ├── page.tsx                landing
    │   ├── analyze/page.tsx        upload page (split identity/upload panel)
    │   ├── processing/[jobId]/     live progress page
    │   ├── results/[jobId]/        analytics results page
    │   └── api/
    │       ├── upload/route.ts     POST: accepts video, spawns Python, streams stdout
    │       ├── jobs/[jobId]/route.ts  GET: job state (polled every 1s by UI)
    │       └── uploads/[...path]/route.ts  GET: streams video files with range support
    ├── components/
    │   ├── LandingPage.tsx         800 lines — hero, features, sticky-stack pipeline
    │   ├── Nav.tsx                 in-page navigation
    │   ├── VideoDropzone.tsx       325 lines — drag/drop, preview, XHR upload
    │   ├── ProcessingView.tsx      310 lines — 5-step live pipeline display
    │   ├── ResultsView.tsx         523 lines — player, cards, timeline SVG
    │   └── SmoothScrollProvider.tsx Lenis + GSAP ticker integration
    ├── lib/
    │   ├── types.ts                JobStatus, PunchType, PunchEvent, FighterResult, JobResult
    │   └── job-store.ts            in-memory Map (singleton across hot reloads)
    ├── public/                     pose-boxing.png, detection-boxing.png, classify-boxing.png, uploads/
    └── next.config.mjs             500 MB server-action body limit, uploads rewrite
```

---

## 3. Core CV Pipeline — `boxing_analytics_v2.py`

### 3.1 Models in use

| Stage | Model | Role | Checkpoint |
|---|---|---|---|
| Person Detection | **YOLOv11m** (custom-trained) | Finds fighter bounding boxes per frame | `runs/person_detect/person_11s_50epochs_20251210_022509/weights/best_potential.pt` |
| Pose Estimation | **YOLOv8m-pose** | Extracts 17 COCO keypoints per fighter | `Tracking and Counting/yolov8m-pose.pt` |
| Punch Classification | **AttentionBiLSTM** (custom) | Classifies 25-frame pose sequences into 6 punch types | `models/punch_classifier.pt` |

Device is auto-selected: MPS (Apple Silicon) → CUDA → CPU fallback.

### 3.2 Key thresholds (line refs in `boxing_analytics_v2.py`)

**Detection (L38–43):** `DETECTION_CONF=0.35`, `DETECTION_IOU=0.45`, `IMAGE_SIZE=640`.

**Tracking (L45–50):** `INERTIA_BONUS=50000.0` (cost bonus to re-match existing fighter), `MAX_TRACK_HISTORY=30` frames, `IoU_THRESHOLD_MATCH=0.3`, `MAX_FIGHTERS=2`, `MIN_VISIBLE_KEYPOINTS=7`.

**Punch detection (L52–58):**
- `PUNCH_ANGLE_THRESHOLD = 155°` — elbow angle required to count extension
- `RESET_ANGLE_THRESHOLD = 120°` — arm must retract below this before re-arming
- `PUNCH_COOLDOWN = 20` frames (≈0.67 s @ 30 fps) — per-arm refractory window
- `MIN_WRIST_VELOCITY = 0.25` (normalized to shoulder width)
- `MIN_ANGLE_DELTA = 15°` — minimum extension delta in 5-frame window

**Classifier input (L60–68):** `CLF_SEQ_LEN=25`, `CLF_INPUT_SIZE=75` (34 keypoint coords + 7 angles + 34 velocities), EMA smoothing α = 0.4.

**Intensity (L73–98):** 15-frame sliding window; 6 raw features per punch (`v_impulse, ω_elbow, ω_shoulder, jerk, hip_impulse, deceleration`). Fallback maxima (4.0, 25°, 18°, 0.5, 2.0, 0.4) used until each fighter logs ≥5 punches, then switch to per-fighter 95th-percentile adaptive normalization. Type-specific weights (L87–97) tune contribution per punch, e.g., Jab leans on wrist impulse (0.32), Lead Hook leans on shoulder rotation (0.35). Label bands: `<0.40` Light, `<0.70` Medium, `≥0.70` Heavy.

### 3.3 Classes

**`AttentionBiLSTM` (L185–205).** Architecture:
```
Input (B, 25, 75)
  → Bidirectional LSTM(75→256, 2 layers, dropout=0.4)
  → (B, 25, 512)
  → Attention MLP (512→64 tanh→1) → softmax over time → weighted sum → (B, 512)
  → LayerNorm → Dropout(0.4) → Linear(512→128) GELU → Dropout(0.2) → Linear(128→6)
```
6-class output; attention learns which frames in the 25-frame window matter (typically around punch apex).

**`FighterTrack` (L311–329).** Per-fighter bounding-box state: `box`, `smooth_box` (EMA α=0.3), `age`, `missed_frames`, `is_fighter`, `color`. Active if `missed_frames < 30`.

**`FighterState` (L332–569).** Per-fighter punch/analytics state:
- `history` (deque maxlen=5) — arm/hip keypoints for punch detection
- `intensity_history` (deque maxlen=15) — wider window for intensity features
- Per-arm `cooldown_{l,r}` and `is_punching_{l,r}` gates
- `shoulder_width` cached via EMA for velocity normalization
- `raw_feature_log` for adaptive per-fighter intensity norms

Key methods:
- **`update_keypoints`** (L359) — ingests 17-kp array, requires ≥4 arm keypoints w/ conf>0.5, updates shoulder-width EMA.
- **`check_punch`** (L507) — dual-armed state machine. For each arm: skip if cooldown>0 or already locked; require `elbow_angle > 155°`, `Δangle > 15°`, wrist displacement > 0.25·shoulder_width in the 5-frame window. On fire: increment count, arm the gate, arm the cooldown.
- **`compute_intensity`** (L385) — computes the six features, normalizes against either fallback maxima or per-fighter 95th-percentile log, applies type-specific weights, returns `{label, score, v_impulse, ω_elbow, ω_shoulder, jerk, hip_impulse, deceleration}`.

### 3.4 Tracking utilities

- **`compute_iou_matrix`** (L576) — pairwise IoU, track × detection.
- **`match_detections_to_tracks`** (L594) — Hungarian assignment with cost = `-IoU − conf − INERTIA_BONUS (if active fighter)`; IoU < 0.3 pairs are masked with 1e9.
- **`_recycle_fighter_id`** (L609) — when a new track is created near a recently-expired fighter track (<0.8 × diagonal distance), reuse the old ID so punch counts and buffers persist through brief occlusions.

### 3.5 Feature engineering for classifier

- **`_hip_normalize`** (L126) — translation-invariant (hip midpoint → origin).
- **`_torso_normalize`** (L132) — scale-invariant (divides by shoulder midpoint distance).
- **`compute_angle_features`** (L143) — returns 7 per-frame angles (radians/π): L/R elbow, shoulder rotation, L/R wrist→nose, L/R forearm direction. Forearm direction is the decisive feature distinguishing hooks (lateral) from uppercuts (upward) from straights.
- **`_fighter_facing_left`** (L228) — nose_x vs shoulder-mid x; if facing left, `_mirror_buffer` flips x and swaps anatomical L/R pairs (the classifier was trained on right-facing footage).
- **`classify_punch`** (L261) — zero-pads to 25 frames **at the start** (so real frames live at the end, matching training format), runs the full feature pipeline, forward passes through BiLSTM, softmax, returns `(type, confidence)`.

### 3.6 Main loop (L650–991)

Per-frame steps:
1. Detect → 2. Hungarian track-match with inertia bonus → 3. Select top-2 fighters + optional referee slot (non-overlapping 3rd box) → 4. Crop bbox → pose on crop → validate (≥7 keypoints) → 5. EMA-smooth classifier keypoints (α=0.4) → 6. Append to 25-frame buffer → 7. `check_punch()` per fighter → 8. If fired: check frame contiguity (max gap ≤3), mirror if needed, `classify_punch()`, `compute_intensity()` → 9. Persist punch label for 45 frames for visualization → 10. Draw bboxes, IDs, type, intensity label, count, frame counter, legend → 11. Write to output VideoWriter.

### 3.7 Output video overlays

Fighter bbox color changes to punch-type color when a punch fires (visible for 45 frames, ~1.5 s). Per punch color (BGR): Jab green, Cross red, Hooks orange, Uppercuts purple. Intensity tag: Light yellow, Medium orange, Heavy red. Referee box uses distinct corner-bracket style. Top-left frame counter, top-right legend listing all 6 punches + 3 intensity levels.

---

## 4. Python ↔ Node Bridge — `frontend_runner.py`

Runs boxing_analytics_v2 as a patched module, replacing `DETECTION_MODEL`, `POSE_MODEL`, `VIDEO_PATH`, `OUTPUT_DIR` at runtime based on CLI args (`--video`, `--output-dir`) and the `PROJECT_ROOT` env var. Intercepts `builtins.print` to translate human-readable log lines into structured JSON events on stdout. Each line is a single JSON doc consumed by the Node.js side.

**Event schema:**

| Event | Fields | Triggered by |
|---|---|---|
| `{"t":"progress","v":20-88}` | `v` is overall % (video progress 0–100 mapped into 20–88 range) | "Frame X/Y (Z%)" log |
| `{"t":"punch_event","fighter","type","total"}` | snake_case type (`lead_hook`, etc.) | "Fighter X … Type … [arm] … conf=X% … total=Y" |
| `{"t":"fighter_total","id","total"}` | Summary totals | "Fighter X — Y total" in Punch Summary |
| `{"t":"breakdown","fighter","type","n"}` | Per-type summary | Breakdown table rows |
| `{"t":"output","path"}` | Absolute path to annotated MP4 | "Output: X.mp4" |
| `{"t":"done"}` | Pipeline finished | Final "DONE" marker |
| `{"t":"error","msg"}` | Unhandled exception caught around `ba.main()` | — |

Exit code 0 on success, 1 on exception.

---

## 5. Next.js Frontend

### 5.1 Config & build

- **Next 14.2.5, React 18, TypeScript 5 (strict)**, Tailwind 3.4.
- **`next.config.mjs`:** `serverActions.bodySizeLimit = '500mb'`, `optimizePackageImports` for `@phosphor-icons/react` and `framer-motion` (tree-shaking), and rewrite `/uploads/:path*` → `/api/uploads/:path*` (videos served through the API route, not static).
- **Fonts:** Geist Sans, Geist Mono, Plus Jakarta Sans (`--font-plus-jakarta`, display), Cormorant Garamond (`--font-cormorant`, serif italic accent). All loaded via `next/font` with `display: 'swap'`.
- **Tailwind:** custom `backgroundImage.grid-pattern` (dual-axis hairlines), `animate.pulse-slow`/`spin-slow`, `font.display`/`font.serif` slots.
- **Aliases:** `@/*` → project root.

### 5.2 Routing

| URL | File | Role |
|---|---|---|
| `/` | `app/page.tsx` → `components/LandingPage.tsx` | Marketing landing |
| `/analyze` | `app/analyze/page.tsx` | Upload (split hero panel + dropzone) |
| `/processing/[jobId]` | `app/processing/[jobId]/page.tsx` → `ProcessingView` | Live pipeline |
| `/results/[jobId]` | `app/results/[jobId]/page.tsx` → `ResultsView` | Final analytics |
| `/api/upload` | POST | Accept video, spawn inference |
| `/api/jobs/[jobId]` | GET | Poll job state |
| `/api/uploads/[...path]` | GET | Stream video (Accept-Ranges, correct content-type) |

Root layout wraps all children in `SmoothScrollProvider` (Lenis smooth-scrolling integrated with GSAP ticker — `duration=1.2`, custom exponential easing).

### 5.3 Type system — `lib/types.ts`

```ts
JobStatus = 'uploading' | 'detecting' | 'pose' | 'classifying' | 'rendering' | 'complete' | 'error'
PunchType = 'jab' | 'cross' | 'lead_hook' | 'rear_hook' | 'lead_uppercut' | 'rear_uppercut'
PunchEvent  = { time: number; fighter: 1|2; type: PunchType }
PunchBreakdown = Record<PunchType, number>
FighterResult = { id: 1|2; totalPunches; breakdown }
JobResult = {
  jobId, status, progress, currentStep, currentDetail, originalFilename,
  videoUrl?, duration?, fps?, frameCount?, fighters?: [F1, F2],
  timeline?: PunchEvent[], error?, startedAt, completedAt?
}
```

### 5.4 Job store — `lib/job-store.ts`

In-memory `Map<string, JobResult>` with a `globalThis.__jobStore` reference in non-production so hot-reloads don't wipe state. **Single-instance only; not Redis/DB-backed** — suitable for the FYP demo deployment but not horizontally scalable.

### 5.5 Upload API — `app/api/upload/route.ts`

The most logic-heavy route. Flow:

1. Validates multipart/form-data has a `video` field, MIME starts with `video/`, size ≤ 500 MB. Returns `400 / 413 / 500` on failure.
2. Generates jobId: `${Date.now().toString(36)}-${random6chars}`.
3. Writes file to `frontend/public/uploads/<jobId>.<ext>`.
4. Initializes the `JobResult` with `status='detecting'`, `progress=8`, and detail "Loading YOLOv11m (best_potential.pt) + YOLOv8m-pose...".
5. Spawns Python non-blocking: `spawn(PYTHON, [RUNNER, '--video', inputPath, '--output-dir', UPLOADS], { cwd: FYP_ROOT })`. `PYTHON` = `process.env.PYTHON_BIN` or macOS Xcode CLT fallback.
6. Returns `{ jobId }` to client; client navigates to `/processing/[jobId]`.

**Streaming stdout parser (line-buffered):**
- `"XX.X fps (N frames)"` — caches fps + frameTotal for duration computation.
- Only lines starting with `{` are JSON-parsed.
- `progress` → maps to UI status buckets: `<35%` detecting, `<65%` pose, else classifying, then updates `currentStep` + `currentDetail` strings.
- `punch_event` → appends to `liveEvents` with an approximate timestamp computed from current progress × (frameTotal/fps).
- `fighter_total`, `breakdown` → aggregate into `totals` and `breakdown` maps.
- `output` → sets status to `rendering`, progress 95%, "H.264 annotated video encoded — finalising...".
- `done` → progress 99, "Building results...".
- `error` → sets `status: 'error'`, stores `msg`.

On child `close`:
- If exit code ≠ 0 and status wasn't already error: mark error.
- Otherwise compute `videoDuration`, build two `FighterResult` objects.
- **Timeline selection:** if `liveEvents.length >= 0.8 × total punches`, use real events sorted by time. Else call `generateSyntheticTimeline()` which shuffles the type order per fighter and scatters points across the duration with jittered gaps — used when the Python side doesn't emit per-frame timestamps.
- Rewrite `videoUrl` to the output MP4 basename if the pipeline produced one.
- Update job to `status='complete'`, progress 100, attach fighters, timeline, duration, fps, frameCount, completedAt.

`proc.stderr` is captured (YOLO/PyTorch log there) and truncated to 200 chars per line to avoid flooding the console.

### 5.6 Uploads streaming API — `app/api/uploads/[...path]/route.ts`

Sanitizes against path traversal (`includes('..')` check), reads file via `fs/promises`, infers content type by extension (`.mp4`→`video/mp4`, `.webm`, `.png`, `.jpg/jpeg`, fallback octet-stream), returns response with `Accept-Ranges: bytes` so the `<video>` element can range-seek.

### 5.7 Job polling API — `app/api/jobs/[jobId]/route.ts`

Trivial GET: reads `jobStore.get(jobId)` and returns it as JSON, `404` if missing.

### 5.8 Components

**`LandingPage.tsx` (798 lines).** Hero, features grid, manifesto, sticky-stack pipeline, CTA, footer. Breakdown:
- **Floating Nav** — transparent until scroll > 40 px, then `.nav-scrolled` class adds blur + border. Toggle via `ScrollTrigger.onUpdate` (classList toggle, not per-frame style writes).
- **Hero entrance:** GSAP timeline staggers `.h-line` split-heading ("Every punch," + italic serif "classified."), subhead, CTA. Background uses an Unsplash boxing-ring photo + multi-layer gradient overlays + faint grid + rose radial glow.
- **Feature cards (3):** "Classification Engine" with `PunchShuffler` — cycles the 6 punches, animates a confidence bar to the (hard-coded) confidence (Jab 87%, Cross 94%, Lead Hook 78%, Rear Hook 83%, Lead Uppercut 71%, Rear Uppercut 76%). "Neural Telemetry" with `TelemetryFeed` — typewriter-prints a rotating list of 10 pipeline log lines ("BiLSTM window complete · 30 frames read" etc.). "Punch Timeline" with `TimelineAnim` — 10 sample events scattered across a dual-fighter track, animated in every 900 ms.
- **Manifesto section:** two oversize lines ("Scorecards record outcomes." vs "We record the mechanics."), plus a tag rail (Wrist velocity, Elbow angle delta, Shoulder rotation, Temporal attention, 30-frame windows).
- **Pipeline sticky-stack:** three full-viewport sections (Stage 01/02/03 with `sticky top-0`, increasing z-index) — each with a full-bleed background image (`detection-boxing.png`, `pose-boxing.png`, `classify-boxing.png`), tint overlay, stage-specific accent (rose, amber, sky), large display headings, tag pills.
- **CTA + Footer** with GitHub source link to `github.com/Muhammad-Rayyan-Mohsin/TenCount`.
- Three marquee-row icon components (defined but not all used in final layout).

**`Nav.tsx`.** Persistent nav on non-landing pages: logo, "v2 pipeline" badge, GitHub link, "Analyse" CTA button. Fixed, `backdrop-blur-xl` over `rgba(12,12,14,0.8)`.

**`VideoDropzone.tsx` (326 lines).** State machine `'idle' | 'dragging' | 'selected' | 'uploading' | 'error'`. Features:
- Drag/drop or click-to-browse, `accept="video/*"`.
- Client-side validation (MIME + 500 MB cap) before upload.
- Live hover-preview of selected video (muted, playsInline, mouseover → `.play()`).
- `XMLHttpRequest` upload with `upload.onprogress` for real-time percent bar.
- On 200 response: `router.push('/processing/' + jobId)`.
- Error banner with inline amber warning icon and "Try again" reset.

**`ProcessingView.tsx` (310 lines).** 5-step pipeline UI: Upload → Person Detection → Pose Estimation → Punch Classification → Render Output. Each step has `activeStatuses`/`completeStatuses` arrays mapping `JobStatus` → step UI state.
- Polls `/api/jobs/[jobId]` every **1 second** (cache: 'no-store').
- `ElapsedTimer` component renders `Xm Ys elapsed`, updated per second.
- Active step: rose border + rotating Circle icon; complete step: emerald CheckCircle.
- Detail line shows `currentDetail` (e.g., "AttentionBiLSTM classifying punch types... 72%").
- On `status === 'complete'`: shows emerald "Analysis complete" toast then redirects to `/results/[jobId]` after 900 ms.
- On error: amber "Pipeline failed" toast with error message.
- Loading skeleton renders before first poll arrives.
- Not-found state if 404 returned (job expired / invalid ID).

**`ResultsView.tsx` (523 lines).** Dashboard with three major pieces:
1. **`VideoPlayer`** — custom HTML5 `<video>` wrapper with clickable play overlay, progress bar (seek by click), play/pause toggle, mute toggle, formatted time display (`m:ss / m:ss`), download button.
2. **Left column:** player + `PunchTimeline` SVG — horizontal axis line with fighter-1 strokes pointing up, fighter-2 pointing down, colored per punch type (Jab rose, Cross orange, Lead Hook amber, Rear Hook green, Lead Uppercut blue, Rear Uppercut purple). Staggered spring entrance. 5-tick time axis and a full 6-type legend below.
3. **Right column:** Summary split bar (rose F1 / blue F2 flex with computed percentages), two `FighterCard`s — each shows big 5xl-font totalPunches, then 6 `PunchBar`s per punch type (bars scale relative to that fighter's max). Winner card gets a rose trophy pill ("more punches"). Below: a pipeline info card (`Detection YOLOv11m … Input dim 41-dim (34 coords + 7 angles) … Cooldown 20 frames — 0.67s`).
- Fetches `/api/jobs/[jobId]` once on mount; errors if `status !== 'complete'`.
- Header shows filename, fps/duration/framecount in mono font, "complete" emerald pill.

**`SmoothScrollProvider.tsx`.** Lenis with `duration: 1.2` and exponential easing; integrated with GSAP ticker (`gsap.ticker.add(time * 1000 → lenis.raf)`, `lagSmoothing(0)`) and ScrollTrigger (`lenis.on('scroll', ScrollTrigger.update)`). Cleanup: remove ticker fn and `lenis.destroy()`.

### 5.9 Visual design system (`globals.css`)

CSS variables: `--bg #0c0c0e`, `--surface #111113`, `--surface-raised #18181b`, `--border rgba(255,255,255,0.06)`, `--accent #e11d48` (rose). Custom thin scrollbar (4 px, zinc-700 thumb), rose selection highlight, antialiased font smoothing, `scroll-behavior: smooth`, `-webkit-tap-highlight-color: transparent`. Text-balance utility, `.nav-scrolled` (20 px blur + background), surface utility classes.

Design language: **neutral zinc-900 base + rose-600 accent + mono-typographic telemetry**. All stat numbers rendered in Geist Mono. Prominent use of `backdrop-blur`, `border-white/[0.05]`, `rounded-[1.75rem]` / `rounded-2xl`, and subtle inner box-shadow highlights.

---

## 6. Deployment — `deploy.sh` + `DEPLOYMENT.md`

### 6.1 Platform

- **AWS EC2, region `eu-north-1` (Stockholm)**.
- **Instance:** `m7i-flex.large` default (2 vCPU, 8 GB RAM, CPU-only, burstable, ~$0.10/hr); **GPU path** via `TENCOUNT_INSTANCE_TYPE=g4dn.xlarge` + `TENCOUNT_AMI_ID=ami-0248a5203d01dc336` (Deep Learning OSS Nvidia Driver AMI, requires quota increase).
- **OS:** Ubuntu 24.04 LTS (default AMI `ami-0dab98137e5c11cb8`).
- **Storage:** 100 GB gp3 EBS root volume, persistent across stop/start.
- **No Docker, no Terraform, no CI/CD** — single bash script.

### 6.2 Subcommands (`./deploy.sh <cmd>`)

| Command | What it does |
|---|---|
| `deploy` | Key pair → SG → launch → tar + scp → remote setup → systemd service start |
| `status` | Prints instance ID, state, URL |
| `stop` | `aws ec2 stop-instances` — billing pauses, data preserved |
| `start` | Restart instance, restart systemd service (public IP is stable thanks to Elastic IP) |
| `ssh` | Interactive SSH session (`ubuntu@<ip>` with stored key) |
| `teardown` | `terminate-instances`, delete SG, delete `.deploy-state`. Key file preserved. |

### 6.3 Deploy phases (executed by `do_deploy`)

1. **Key pair** — `aws ec2 create-key-pair`, save to `~/.ssh/tencount-deploy-key.pem`, `chmod 400`. Idempotent.
2. **Security group** (`tencount-sg`): opens ports 22, 80, 443, 3000 to `0.0.0.0/0` (permissive — FYP dev mode). No egress restriction.
3. **Launch instance** with 100 GB gp3 EBS, tag `Name=TenCount-FYP`.
4. Store instance ID in `.deploy-state` (single-line text file, gitignored).
5. `aws ec2 wait instance-running` → `get_public_ip` → `wait_for_ssh` (60 iterations × 5 s SSH probe).
6. **Tar project**: excludes `.git`, `frontend/node_modules`, `frontend/.next`, `__pycache__`, `.DS_Store`, `.deploy-state`. scp to `/tmp/`.
7. **Remote setup** (heredoc script, lines 148–206):
   - `apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0`
   - NodeSource setup + `apt-get install nodejs` (20.x, conditional)
   - `python3 -m venv ~/tencount/.venv`, upgrade pip
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (CPU wheels)
   - `pip install ultralytics opencv-python-headless scipy numpy`
   - **Manual post-step:** `pip install albumentations` — required by the classifier's data-augmentation imports but not yet in the automated pip list; installed via `ssh` after first deploy.
   - `cd frontend && npm install --production=false && npm run build`
   - Write `/etc/systemd/system/tencount.service`, `daemon-reload`, `enable`, `start`
8. Print success banner with `http://<PUBLIC_IP>:3000`.

### 6.4 Systemd unit (generated)

```ini
[Unit]
Description=TenCount Boxing Analytics
After=network.target

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

[Install]
WantedBy=multi-user.target
```

Logs via `sudo journalctl -u tencount -f`. Auto-restart on crash (5 s delay).

### 6.5 Key env vars

- **`PYTHON_BIN`** — path to venv python3; consumed by `app/api/upload/route.ts` for `spawn()`.
- **`PROJECT_ROOT`** — used by `frontend_runner.py` to resolve model paths.
- **`NODE_ENV=production`** — production Next.js runtime.

### 6.6 Cost and lifecycle

- ~$0.10/hr running, $0/hr stopped (EBS storage still billed ~$10/mo for 100 GB).
- `stop/start` preserves all data; `teardown` destroys EBS volume.
- GPU upgrade: `g4dn.xlarge` ~5× cost but dramatically faster inference (MPS not used on Linux server — CUDA path).

### 6.7 Production edge (manual layer on top of `deploy.sh`)

These were configured manually on the running instance — they are **not** part of `deploy.sh` and must be re-applied after a fresh deploy:

- **Elastic IP:** `13.63.242.120` allocated via `aws ec2 allocate-address` and associated with the instance. Allocation ID tracked in `.eip-state`. Keeps DNS stable across stop/start.
- **DNS:** Cloudflare-managed `arbisoft.ai`, two A records (`@` and `www` → `13.63.242.120`), **DNS-only mode** (gray cloud, not proxied) so Let's Encrypt HTTP-01 challenges reach the origin.
- **Nginx reverse proxy:** `/etc/nginx/sites-enabled/tencount` proxies `arbisoft.ai` and `www.arbisoft.ai` → `http://localhost:3000`. Sets `X-Forwarded-*` headers, `client_max_body_size 500M`, 600 s proxy read/send timeouts for long inference jobs.
- **TLS:** `certbot --nginx -d arbisoft.ai -d www.arbisoft.ai --redirect` — Let's Encrypt cert at `/etc/letsencrypt/live/arbisoft.ai/`. Auto-renewing systemd timer. HTTP → HTTPS redirect enforced.
- **Public URL:** `https://arbisoft.ai` (the `:3000` in the deploy banner is the origin port; users should never hit it directly once nginx is live).

---

## 7. End-to-end data flow

```
User drops video in browser
  └─ VideoDropzone (XHR POST /api/upload, progress events)
       │
       ▼
  /api/upload writes file to public/uploads/<jobId>.<ext>
  spawn(python3, [frontend_runner.py, --video, …, --output-dir, …], {cwd: FYP_ROOT})
       │
       ▼
  frontend_runner.py sets ba.DETECTION_MODEL/POSE_MODEL/VIDEO_PATH/OUTPUT_DIR
  builtins.print is patched → JSON events to stdout
       │
       ▼
  boxing_analytics_v2.main() per-frame loop
     detect → track(Hungarian+inertia) → crop+pose → FighterState update →
     check_punch → classify(BiLSTM) → compute_intensity → draw+write frame →
     print "Frame N/T (X%)" + live punch events
       │  stdout JSON
       ▼
  Node side's proc.stdout 'data' handler parses per-line:
     progress/punch_event/fighter_total/breakdown/output/done/error
     jobStore.set(jobId, { …updated state… })
       │
       ▼
  ProcessingView polls /api/jobs/[jobId] every 1 s → renders step states + percent
       │
       ▼  status === 'complete'
  ResultsView fetches once → renders VideoPlayer, FighterCards, PunchTimeline
```

---

## 8. Tracked / emitted analytics (every feature)

**Per fighter:**
- Total punch count
- Breakdown: Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut
- Intensity label per punch (Light / Medium / Heavy), driven by 6 raw biomech features:
  - Wrist velocity impulse (normalized to shoulder width)
  - Elbow angular velocity (95th percentile °/frame)
  - Shoulder rotation speed (95th percentile °/frame)
  - Wrist jerk (Δvelocity/frame, 95th percentile)
  - Hip midpoint impulse
  - Post-peak wrist deceleration (signals commitment)
- Stance-aware mirroring (orthodox vs southpaw via nose-vs-shoulder x position)
- Per-arm independent cooldowns so rapid combos aren't merged

**Per match:**
- FPS, duration, frame count
- Total combined punches, F1 vs F2 share percentages
- Punch event timeline (real events if ≥80% coverage, else synthetic scatter)
- Winner heuristic (more punches)
- Annotated H.264 MP4 with bboxes, IDs, per-frame labels, referee slot, legend, frame counter

**Progress telemetry (live):**
- 5 UI phases (Upload → Detection → Pose → Classify → Render) with active/complete/pending states
- Detail string updates mapped from raw progress percent buckets (<35 detecting, <65 pose, ≥65 classifying)
- Elapsed timer (client-side)

**Model/pipeline self-reported** (hardcoded in UI):
- Classifier accuracy **70.9%**
- **6 punch classes**
- **30 fps** output video
- Pose: 17 COCO keypoints
- Classifier input: 41-dim in UI (actual code uses 75-dim — UI display is simplified)
- Cooldown 20 frames / 0.67 s

---

## 9. Notable design choices & tradeoffs

1. **Hungarian + INERTIA_BONUS (50,000) for tracking** — strongly favors re-matching existing fighters rather than spawning new tracks; combined with `_recycle_fighter_id` (0.8 × diagonal distance window) this preserves punch counts through brief occlusions.
2. **Zero-padding at the start** of the classifier buffer (not end) — matches training format; BiLSTM + attention learns to focus on the final frames where extension occurs.
3. **EMA keypoint smoothing (α=0.4)** as classifier input, while **raw keypoints** drive punch detection — the classifier benefits from noise reduction but the state machine needs responsiveness.
4. **Per-arm independent state** — two cooldowns, two gates — enables rapid L/R combinations without double-counting.
5. **Adaptive per-fighter intensity normalization** after 5 punches — accounts for camera angle, body size, video quality; falls back to fixed maxima early on.
6. **Type-specific intensity weights** — hooks weighted on shoulder rotation (0.35), jabs on wrist (0.32), uppercuts on hip impulse.
7. **In-memory job store** — simplest possible state layer; fine for single-instance demo, would need Redis for horizontal scaling.
8. **Synthetic timeline fallback** — because the Python side doesn't emit per-frame timestamps with each punch, the upload route reconstructs a realistic scatter if live events cover <80% of final counts.
9. **500 MB upload cap** enforced at three layers: Next server-action body limit, client dropzone validation, server-side POST handler — defense in depth.
10. **Permissive security group** (0.0.0.0/0 on 22/80/443/3000) — acceptable for FYP demo; production would IP-whitelist SSH and front with HTTPS.

---

## 10. Known limitations (explicit or structural)

- **`MAX_FIGHTERS=2` hardcoded** — won't generalize beyond boxing/MMA 1v1.
- **Frame-rate calibration assumes ~30 fps** — 60-fps footage would halve the effective cooldown (0.33 s).
- **Per-frame timestamps aren't emitted** with punch events — timeline is approximated from progress percent.
- **Job store is in-memory** — restart loses all in-flight jobs.
- **Nginx + Let's Encrypt HTTPS / Elastic IP / DNS / `albumentations`** are configured manually on top of `deploy.sh`; a fresh `./deploy.sh deploy` reverts to plain HTTP on `:3000` and is missing `albumentations` until re-applied.
- **No WAF, no IP restrictions** on the security group (0.0.0.0/0 on 22/80/443/3000).
- **No version pinning** for Python deps at deploy time (latest torch/ultralytics/etc.).
- **Inference is single-threaded** — per-fighter sequential processing; no batched GPU forward passes.
- **Model weights baked into tar** (≈80 MB per deploy) — no artifact registry / S3 separation.
- **No monitoring** — no CloudWatch alarms, no log retention beyond journalctl.

---

## 11. Headline numbers

| Metric | Value |
|---|---|
| Classifier accuracy (as advertised) | 70.9% |
| Punch classes | 6 |
| Pose keypoints | 17 (COCO) |
| Classifier input dim | 75 (34 coords + 7 angles + 34 velocities) |
| Classifier seq length | 25 frames |
| Detection conf threshold | 0.35 |
| Tracking IoU threshold | 0.3 |
| Max fighters | 2 (+ 1 referee slot) |
| Punch angle threshold | 155° |
| Retraction gate | 120° |
| Per-arm cooldown | 20 frames (~0.67 s @ 30 fps) |
| Intensity window | 15 frames |
| Output video fps | 30 |
| Max upload size | 500 MB |
| Frontend poll interval | 1 s |
| EC2 instance | m7i-flex.large (2 vCPU / 8 GB / CPU) |
| EBS volume | 100 GB gp3 |
| Region | eu-north-1 |
