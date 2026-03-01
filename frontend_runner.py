#!/usr/bin/env python3
"""
frontend_runner.py — TenCount Next.js inference bridge.

Patches boxing_analytics_v2 with the correct model/video paths,
then intercepts stdout to emit JSON progress lines the Node.js API
can parse for real-time job status updates.

Usage (called by the Next.js upload API):
  python3 frontend_runner.py --video /abs/path/input.mp4 \
                              --output-dir /abs/path/to/output/dir
"""

import argparse
import builtins
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Parse args BEFORE importing the inference module so we can patch constants
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video",      required=True, help="Absolute path to input video")
parser.add_argument("--output-dir", required=True, help="Absolute path to output directory")
args = parser.parse_args()

# Change to FYP root so relative model paths resolve correctly
FYP_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(FYP_ROOT)

# ---------------------------------------------------------------------------
# JSON emitter — always flushed so Node.js readline sees it immediately
# ---------------------------------------------------------------------------
_orig_print = builtins.print

def emit(obj: dict):
    _orig_print(json.dumps(obj), flush=True)

# ---------------------------------------------------------------------------
# Import the module and patch the configuration constants
# ---------------------------------------------------------------------------
import boxing_analytics_v2 as ba  # noqa: E402

ba.DETECTION_MODEL = (
    "runs/person_detect/person_11s_50epochs_20251210_022509/weights/best_potential.pt"
)
ba.POSE_MODEL    = "Tracking and Counting/yolov8m-pose.pt"
ba.VIDEO_PATH    = args.video
ba.OUTPUT_DIR    = args.output_dir

# ---------------------------------------------------------------------------
# Print interceptor
# ---------------------------------------------------------------------------
_PUNCH_KEY = {
    "Jab":           "jab",
    "Cross":         "cross",
    "Lead Hook":     "lead_hook",
    "Rear Hook":     "rear_hook",
    "Lead Uppercut": "lead_uppercut",
    "Rear Uppercut": "rear_uppercut",
}

_in_summary      = False
_current_fighter = None   # type: int | None


def _intercept(*a, **kw):
    global _in_summary, _current_fighter

    text = " ".join(str(x) for x in a)
    _orig_print(*a, **kw)
    sys.stdout.flush()

    # ── Frame progress ──────────────────────────────────────────────────────
    # "  Frame 60/847  (7.1%)"
    m = re.search(r"Frame\s+(\d+)/(\d+)\s+\((\d+(?:\.\d+)?)%\)", text)
    if m:
        pct = float(m.group(3))
        # Map 0–100 % of video frames → 20–88 % of the overall progress bar
        overall = int(20 + pct * 0.68)
        emit({"t": "progress", "v": overall})
        return

    # ── Live punch event (during processing, before summary) ────────────────
    # "  Fighter 1  Jab              R arm  [Heavy]  conf=84%  total=5"
    if not _in_summary and "Fighter" in text and "arm" in text and "conf=" in text:
        m_f  = re.search(r"Fighter\s+(\d+)", text)
        m_pt = re.search(r"Fighter\s+\d+\s+([\w ]+?)\s{2,}", text)
        m_tot = re.search(r"total=(\d+)", text)
        if m_f and m_pt:
            fid   = int(m_f.group(1))
            ptype = m_pt.group(1).strip()
            emit({
                "t":       "punch_event",
                "fighter": fid,
                "type":    _PUNCH_KEY.get(ptype, ptype.lower().replace(" ", "_")),
                "total":   int(m_tot.group(1)) if m_tot else 0,
            })
        return

    # ── Punch Summary section ────────────────────────────────────────────────
    if "Punch Summary" in text:
        _in_summary = True
        return

    if _in_summary:
        # "  Fighter 1  —  31 total punches"
        m = re.search(r"Fighter\s+(\d+)\s+—\s+(\d+)\s+total", text)
        if m:
            _current_fighter = int(m.group(1))
            emit({"t": "fighter_total", "id": _current_fighter, "total": int(m.group(2))})
            return

        # "    Jab               11"
        if _current_fighter is not None:
            for ptype, key in _PUNCH_KEY.items():
                if ptype in text:
                    n_m = re.search(r"(\d+)\s*$", text.strip())
                    if n_m:
                        emit({
                            "t":       "breakdown",
                            "fighter": _current_fighter,
                            "type":    key,
                            "n":       int(n_m.group(1)),
                        })
                    return

    # ── Output path ─────────────────────────────────────────────────────────
    # "  Output: /abs/path/v2_20260301_123456.mp4"
    m = re.search(r"Output:\s*(.+\.mp4)", text)
    if m:
        emit({"t": "output", "path": m.group(1).strip()})
        return

    # ── Done ─────────────────────────────────────────────────────────────────
    if text.strip() == "DONE":
        emit({"t": "done"})


builtins.print = _intercept

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
try:
    ba.main()
except Exception as exc:
    emit({"t": "error", "msg": str(exc)})
    sys.exit(1)
