# TenCount — Boxing Punch Detection & Analytics

A full-stack boxing analytics system that detects, tracks, and classifies punches in real-time from fight footage. Runs locally or deployed on AWS EC2.

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion, GSAP |
| Backend bridge | Python (`frontend_runner.py`) |
| CV Pipeline | YOLOv11m (detection) + YOLOv8m-pose (skeleton) + AttentionBiLSTM (classification) |
| Deployment | AWS EC2 (Ubuntu 24.04) via `deploy.sh` |

## Project Structure

```
Tencounterzz/
├── frontend/                   # Next.js app
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   ├── analyze/            # Upload page
│   │   ├── processing/[jobId]/ # Real-time processing view
│   │   ├── results/[jobId]/    # Analytics results
│   │   └── api/
│   │       ├── upload/         # Receives video, spawns Python process
│   │       ├── uploads/        # Serves uploaded videos back to the client
│   │       └── jobs/[jobId]/   # Job status polling endpoint
│   ├── components/
│   │   ├── LandingPage.tsx
│   │   ├── VideoDropzone.tsx
│   │   ├── ProcessingView.tsx
│   │   └── ResultsView.tsx
│   └── lib/
│       ├── job-store.ts        # In-memory job state (server-side)
│       └── types.ts
├── frontend_runner.py          # Python ↔ Node.js bridge (JSON stdout protocol)
├── boxing_analytics_v2.py      # Core CV inference pipeline
├── deploy.sh                   # AWS EC2 deployment script
└── requirements.txt            # Python dependencies
```

## How It Works

1. User drops a video on the frontend → upload API saves it and spawns `frontend_runner.py`
2. `frontend_runner.py` runs `boxing_analytics_v2.py` and intercepts stdout, emitting structured JSON progress events
3. Frontend polls `/api/jobs/[jobId]` and streams progress in real-time
4. On completion, results page shows punch counts, type breakdown, and intensity per fighter

## Inference Pipeline Features

- Per-arm cooldown for combination detection (jab–cross, hook–uppercut)
- Bbox EMA smoothing to reduce keypoint jitter
- Classifier keypoint EMA smoothing before buffering
- Frame contiguity tracking with hold-last-value on pose failure
- Impulse-based intensity assessment (6 features, adaptive)
- Referee detection to exclude non-fighters

## Setup

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Python backend
```bash
pip install -r requirements.txt
```

Place model weights in the FYP root:
- `runs/person_detect/.../weights/best_potential.pt` — YOLOv11m detection
- `Tracking and Counting/yolov8m-pose.pt` — YOLOv8m pose
- `models/punch_classifier.pt` — AttentionBiLSTM classifier

## Deployment (AWS EC2)

```bash
./deploy.sh deploy      # Launches EC2, uploads code, installs deps, starts app
./deploy.sh status      # Show instance state and URL
./deploy.sh stop        # Stop (no charges, data preserved)
./deploy.sh ssh         # SSH into the server
./deploy.sh teardown    # Terminate instance
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide, GPU upgrade path, and troubleshooting.

## Punch Classes

`Jab` · `Cross` · `Lead Hook` · `Rear Hook` · `Lead Uppercut` · `Rear Uppercut`

## Documentation

- [TECHNICAL.md](TECHNICAL.md) — Full technical reference (architecture, models, pipeline internals)
- [DEPLOYMENT.md](DEPLOYMENT.md) — AWS EC2 deployment guide
- [report.md](report.md) — Project report
