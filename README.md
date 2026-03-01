# TenCount — Boxing Punch Detection & Analytics

A full-stack boxing analytics system that detects, tracks, and classifies punches in real-time from fight footage.

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion, GSAP |
| Backend bridge | Python (`frontend_runner.py`) |
| CV Pipeline | YOLOv11m (detection) + YOLOv8m-pose (skeleton) + AttentionBiLSTM (classification) |

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
└── boxing_analytics_v2.py      # Core CV inference pipeline
```

## How It Works

1. User drops a video on the frontend → upload API saves it and spawns `frontend_runner.py`
2. `frontend_runner.py` runs `boxing_analytics_v2.py` and intercepts stdout, emitting structured JSON progress events
3. Frontend polls `/api/jobs/[jobId]` and streams progress in real-time
4. On completion, results page shows punch counts, type breakdown, and intensity per fighter

## Setup

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Python backend
```bash
pip install ultralytics opencv-python numpy scipy torch
```

Place model weights in the FYP root:
- `runs/person_detect/.../weights/best_potential.pt` — YOLOv11m detection
- `Tracking and Counting/yolov8m-pose.pt` — YOLOv8m pose
- `models/punch_classifier.pt` — AttentionBiLSTM classifier

## Punch Classes

`Jab` · `Cross` · `Lead Hook` · `Rear Hook` · `Lead Uppercut` · `Rear Uppercut`
