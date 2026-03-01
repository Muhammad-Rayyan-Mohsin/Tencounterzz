#!/usr/bin/env python3
"""
Boxing Analytics v2 — Punch Detection + Type Classification
=============================================================
Extends boxing_analytics_inference.py with BiLSTM punch type classification.

Pipeline:
  Video → YOLOv11m (detect fighters) → YOLOv8m-pose (keypoints)
        → FighterState.check_punch() (punch event)
        → AttentionBiLSTM (classify: Jab / Cross / Lead Hook /
                           Rear Hook / Lead Uppercut / Rear Uppercut)

Run:
    python boxing_analytics_v2.py
"""

from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================

DETECTION_MODEL    = "runs/person_detect/person_11s_50epochs_20251210_022509/weights/best_potential.pt"
POSE_MODEL         = "Tracking and Counting/yolov8m-pose.pt"
CLASSIFIER_MODEL   = "models/punch_classifier.pt"
VIDEO_PATH         = "Untitled video - Made with Clipchamp.mp4"
OUTPUT_DIR         = "runs/inference_v2"

# Detection
DETECTION_CONF     = 0.35
DETECTION_IOU      = 0.45
IMAGE_SIZE         = 640
DEVICE             = "mps" if torch.backends.mps.is_available() else \
                     ("cuda" if torch.cuda.is_available() else "cpu")

# Tracking
INERTIA_BONUS      = 50000.0
MAX_TRACK_HISTORY  = 30
IoU_THRESHOLD_MATCH = 0.3
MAX_FIGHTERS       = 2
MIN_VISIBLE_KEYPOINTS = 7

# Punch detection
PUNCH_COOLDOWN        = 20    # ~0.67s at 30fps (was 8 — caused overcounting)
PUNCH_ANGLE_THRESHOLD = 155   # elbow must reach 155° (near full extension)
RESET_ANGLE_THRESHOLD = 120   # arm must retract below this before next punch
MIN_WRIST_VELOCITY    = 0.25  # normalized wrist speed (was missing)
MIN_ANGLE_DELTA       = 15    # min extension delta across history window
DEBUG_PUNCH           = False

# Classifier
CLF_SEQ_LEN        = 25     # frames the classifier expects
CLF_KPT_FEATURES   = 34     # 17 × (x, y)
CLF_ANG_FEATURES   = 7      # 5 original + 2 forearm direction
CLF_VEL_FEATURES   = 34     # joint velocities: frame-to-frame keypoint deltas
CLF_INPUT_SIZE     = 75     # 34 + 7 + 34 (kpts + angles + velocities)

# Display: how many frames to keep showing the last punch label
LABEL_PERSIST_FRAMES = 45

# ── Intensity assessment ───────────────────────────────────────────────────────
# Empirical max values used to normalise each feature to [0, 1].
# Derived from biomechanics literature (peak wrist velocity ~8 m/s at ~30fps →
# normalized displacement ~1.5 shoulder-widths/frame at max effort).
INTENSITY_V_MAX          = 1.2    # peak wrist velocity (shoulder-widths / frame) — calibrated from biomechanics lit
INTENSITY_OE_MAX         = 30.0   # peak elbow angular velocity (degrees / frame) — 400-650 deg/s → ~13-22 deg/frame
INTENSITY_OS_MAX         = 22.0   # peak shoulder rotation speed (degrees / frame) — 500-1100 deg/s → ~17-37 deg/frame
INTENSITY_JERK_MAX       = 0.6    # wrist jerk (Δvelocity / frame)
# Weighted score: wrist velocity dominates (force ∝ v²), shoulder/elbow secondary
INTENSITY_WEIGHTS        = (0.45, 0.25, 0.18, 0.12)  # v, omega_elbow, omega_sho, jerk
INTENSITY_LIGHT_THRESH   = 0.40
INTENSITY_HEAVY_THRESH   = 0.70

# Intensity label colours (BGR)
INTENSITY_COLORS = {
    "Light":  (0,   220, 220),   # yellow
    "Medium": (0,   140, 255),   # orange
    "Heavy":  (0,   0,   220),   # red
}

# Punch type colours (BGR)
PUNCH_COLORS = {
    "Jab":           (50,  220, 50),    # green
    "Cross":         (50,  50,  220),   # red
    "Lead Hook":     (50,  165, 255),   # orange
    "Rear Hook":     (0,   100, 200),   # dark orange
    "Lead Uppercut": (220, 220, 50),    # cyan-yellow
    "Rear Uppercut": (200, 50,  200),   # purple
}

# ============================================================================
# ANGLE FEATURES  (must match train_punch_classifier.py exactly)
# ============================================================================

_SWAP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]


def _hip_normalize(seq_3d: np.ndarray) -> np.ndarray:
    """Subtract hip midpoint (kpts 11 & 12 mean) from all joints — matches training."""
    mid = (seq_3d[:, 11, :] + seq_3d[:, 12, :]) / 2   # (T, 2)
    return seq_3d - mid[:, np.newaxis, :]               # (T, 17, 2)


def _torso_normalize(seq_3d: np.ndarray) -> np.ndarray:
    """Divide by mean torso height — makes features scale-invariant. Call after _hip_normalize."""
    sh_mid = (seq_3d[:, 5, :] + seq_3d[:, 6, :]) / 2   # (T, 2)
    torso_h = np.linalg.norm(sh_mid, axis=1)             # (T,)
    valid = torso_h[torso_h > 1e-3]
    scale = float(valid.mean()) if len(valid) else 1.0
    return seq_3d / (scale + 1e-6)


def compute_angle_features(seq_3d: np.ndarray) -> np.ndarray:
    """
    seq_3d : (T, 17, 2) normalised keypoints
    returns : (T, 7)    float32  — must match train_punch_classifier.py exactly
        [0] left  elbow angle
        [1] right elbow angle
        [2] shoulder rotation
        [3] left  wrist → nose direction
        [4] right wrist → nose direction
        [5] left  forearm direction (elbow→wrist vector angle)
        [6] right forearm direction (elbow→wrist vector angle)
    """
    T   = seq_3d.shape[0]
    out = np.zeros((T, CLF_ANG_FEATURES), dtype=np.float32)
    for t in range(T):
        kp = seq_3d[t]
        nose       = kp[0]
        l_sh,r_sh  = kp[5],  kp[6]
        l_el,r_el  = kp[7],  kp[8]
        l_wr,r_wr  = kp[9],  kp[10]
        for i,(sh,el,wr) in enumerate([(l_sh,l_el,l_wr),(r_sh,r_el,r_wr)]):
            ba = sh - el;  bc = wr - el
            d  = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
            out[t, i] = np.arccos(np.clip(np.dot(ba, bc) / d, -1.0, 1.0)) / np.pi
        sv = r_sh - l_sh
        out[t, 2] = np.arctan2(sv[1], sv[0]) / np.pi
        lv = l_wr - nose
        out[t, 3] = np.arctan2(lv[1], lv[0]) / np.pi
        rv = r_wr - nose
        out[t, 4] = np.arctan2(rv[1], rv[0]) / np.pi
        # Forearm direction: elbow → wrist vector (key hook/uppercut/straight discriminator)
        l_fw = l_wr - l_el
        out[t, 5] = np.arctan2(l_fw[1], l_fw[0]) / np.pi
        r_fw = r_wr - r_el
        out[t, 6] = np.arctan2(r_fw[1], r_fw[0]) / np.pi
    return out


# ============================================================================
# PUNCH CLASSIFIER MODEL
# ============================================================================

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size=CLF_INPUT_SIZE, hidden_size=256,
                 num_layers=2, num_classes=6, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        lstm_out = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out), nn.Dropout(dropout),
            nn.Linear(lstm_out, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attention(out), dim=1)
        return self.classifier((w * out).sum(dim=1))


def load_classifier(path: str, device: str):
    ckpt        = torch.load(path, map_location=device)
    cfg         = ckpt["config"]
    idx_to_cls  = ckpt["idx_to_class"]
    model = AttentionBiLSTM(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        num_classes = cfg["num_classes"],
        dropout     = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_cls


@torch.no_grad()
def classify_punch(model, kpt_buffer: deque, device: str, idx_to_cls: dict):
    """
    Run the classifier on the current 25-frame keypoint buffer.

    kpt_buffer : deque of (17, 2) arrays already normalised to [0, 1]
                 relative to the fighter's bounding box (crop-space).
    Returns    : (punch_type: str, confidence: float)
    """
    if len(kpt_buffer) < CLF_SEQ_LEN:
        return None, 0.0

    # Buffer already contains bbox-normalised [0, 1] coords — no division needed.
    seq = np.array(kpt_buffer, dtype=np.float32)    # (25, 17, 2) in [0, 1]

    # Hip-centre + torso-height normalisation — must match training preprocessing
    seq = _hip_normalize(seq)
    seq = _torso_normalize(seq)

    flat   = seq.reshape(CLF_SEQ_LEN, CLF_KPT_FEATURES)         # (25, 34)
    angles = compute_angle_features(seq)                          # (25,  7)
    vel    = np.diff(flat, axis=0, prepend=flat[[0]])             # (25, 34) Δpos/frame
    full   = np.concatenate([flat, angles, vel], axis=1)         # (25, 75)

    t   = torch.tensor(full, dtype=torch.float32).unsqueeze(0).to(device)
    out = model(t)
    probs      = torch.softmax(out, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    return idx_to_cls[pred_idx], confidence


# ============================================================================
# TRACKING (unchanged from v1)
# ============================================================================

class FighterTrack:
    def __init__(self, track_id, box, conf):
        self.id = track_id;  self.box = box;  self.conf = conf
        self.age = 1;  self.missed_frames = 0;  self.is_fighter = False
        np.random.seed(track_id)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())

    def update(self, box, conf):
        self.box = box;  self.conf = conf
        self.age += 1;   self.missed_frames = 0

    def age_frame(self):
        self.missed_frames += 1;  self.age += 1

    def is_active(self):
        return self.missed_frames < MAX_TRACK_HISTORY


class FighterState:
    def __init__(self, fighter_id):
        self.fighter_id      = fighter_id
        self.punch_count     = 0
        self.history         = deque(maxlen=5)
        self.cooldown        = 0
        self.is_punching_l   = False   # per-arm retraction gate (left)
        self.is_punching_r   = False   # per-arm retraction gate (right)
        self.shoulder_width  = 1.0

    def _get_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b;  bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def _arm_angle(self, data, prefix):
        try:
            return self._get_angle(data[f'{prefix}_sh'],
                                   data[f'{prefix}_el'],
                                   data[f'{prefix}_wr'])
        except KeyError:
            return None

    def update_keypoints(self, keypoints, confidences):
        # Use conf > 0.5 for arm joints — 0.7 was too strict during motion blur
        # (fast punches routinely drop below 0.7, causing excessive missed frames).
        # Require 4-of-6 keypoints — one occluded arm is still useful for single-arm analysis.
        mapping = {'l_sh':5,'r_sh':6,'l_el':7,'r_el':8,'l_wr':9,'r_wr':10}
        data    = {}
        for name, idx in mapping.items():
            if confidences[idx] > 0.5:
                data[name] = keypoints[idx]
        # Store if ≥ 4 of 6 arm keypoints pass (relaxed from all-6)
        if len(data) >= 4:
            if 'l_sh' in data and 'r_sh' in data:
                sw = np.linalg.norm(data['r_sh'] - data['l_sh'])
                if sw > 10:
                    self.shoulder_width = sw
            self.history.append(data)

    def _smooth_kpt(self, key):
        """5-frame rolling mean on a named keypoint — reduces 23° RMSD pose jitter."""
        vals = [h[key] for h in self.history if key in h]
        return np.mean(vals, axis=0) if vals else None

    def compute_intensity(self, side: str) -> dict:
        """
        Compute punch intensity from the current history window.
        Called immediately after check_punch() confirms a punch.

        Features (all normalized to [0, 1]):
          v_peak        — peak per-frame wrist displacement / shoulder_width
          omega_elbow   — peak per-frame elbow angle delta (degrees/frame)
          omega_shoulder— peak per-frame shoulder rotation speed (degrees/frame)
          jerk          — peak rate of change of wrist velocity

        Weighted score → Light / Medium / Heavy label.
        """
        hist = list(self.history)
        n    = len(hist)
        p    = 'l' if side == 'left' else 'r'
        sw   = max(self.shoulder_width, 1.0)

        # Per-frame wrist velocities
        wrist_vels = []
        for i in range(1, n):
            if f'{p}_wr' in hist[i] and f'{p}_wr' in hist[i - 1]:
                d = np.linalg.norm(
                    np.array(hist[i][f'{p}_wr']) - np.array(hist[i - 1][f'{p}_wr'])
                ) / sw
                wrist_vels.append(d)

        v_peak    = max(wrist_vels) if wrist_vels else 0.0

        # Wrist jerk: rate-of-change of velocity (captures the "snap")
        jerk_peak = 0.0
        if len(wrist_vels) >= 2:
            jerk_peak = max(abs(wrist_vels[i] - wrist_vels[i - 1])
                            for i in range(1, len(wrist_vels)))

        # Per-frame elbow angular velocity
        elbow_deltas = []
        for i in range(1, n):
            a_curr = self._arm_angle(hist[i], p)
            a_prev = self._arm_angle(hist[i - 1], p)
            if a_curr is not None and a_prev is not None:
                elbow_deltas.append(abs(a_curr - a_prev))
        omega_elbow = max(elbow_deltas) if elbow_deltas else 0.0

        # Per-frame shoulder rotation speed
        sho_deltas = []
        for i in range(1, n):
            h_c, h_p = hist[i], hist[i - 1]
            if all(k in h_c and k in h_p for k in ('l_sh', 'r_sh')):
                sv_c = np.array(h_c['r_sh']) - np.array(h_c['l_sh'])
                sv_p = np.array(h_p['r_sh']) - np.array(h_p['l_sh'])
                ang_c = np.degrees(np.arctan2(sv_c[1], sv_c[0]))
                ang_p = np.degrees(np.arctan2(sv_p[1], sv_p[0]))
                d = abs(ang_c - ang_p)
                sho_deltas.append(360.0 - d if d > 180.0 else d)  # wrap-around safe
        omega_shoulder = max(sho_deltas) if sho_deltas else 0.0

        # Normalise each feature and compute weighted score
        v_norm  = min(v_peak         / INTENSITY_V_MAX,   1.0)
        oe_norm = min(omega_elbow    / INTENSITY_OE_MAX,  1.0)
        os_norm = min(omega_shoulder / INTENSITY_OS_MAX,  1.0)
        j_norm  = min(jerk_peak      / INTENSITY_JERK_MAX, 1.0)

        wv, we, ws, wj = INTENSITY_WEIGHTS
        score = wv * v_norm + we * oe_norm + ws * os_norm + wj * j_norm

        if score < INTENSITY_LIGHT_THRESH:
            label = "Light"
        elif score < INTENSITY_HEAVY_THRESH:
            label = "Medium"
        else:
            label = "Heavy"

        return {
            "label":         label,
            "score":         round(float(score), 3),
            "v_peak":        round(float(v_peak), 3),
            "omega_elbow":   round(float(omega_elbow), 3),
            "omega_shoulder": round(float(omega_shoulder), 3),
            "jerk":          round(float(jerk_peak), 3),
        }

    def check_punch(self):
        """
        Returns (fired, side, intensity) on a confirmed punch, else (False, None, None).
        Requires:
          - arm angle > PUNCH_ANGLE_THRESHOLD (near full extension)
          - angle delta across history window > MIN_ANGLE_DELTA
          - wrist velocity > MIN_WRIST_VELOCITY (normalized to shoulder width)
          - arm must retract below RESET_ANGLE_THRESHOLD before next punch
          - PUNCH_COOLDOWN frames between counts
        """
        if not self.history:
            return False, None, None

        curr = self.history[-1]

        # Always update per-arm retraction state regardless of cooldown
        for p, attr in [('l', 'is_punching_l'), ('r', 'is_punching_r')]:
            a = self._arm_angle(curr, p)
            if a is not None and a < RESET_ANGLE_THRESHOLD:
                setattr(self, attr, False)

        # During global cooldown just tick down; retraction already updated above
        if self.cooldown > 0:
            self.cooldown -= 1
            return False, None, None

        if len(self.history) < 3:
            return False, None, None

        prev = self.history[0]   # oldest frame in window

        for side, p, attr in [('left', 'l', 'is_punching_l'),
                               ('right', 'r', 'is_punching_r')]:
            # Per-arm gate: skip if this arm hasn't retracted since last punch
            if getattr(self, attr):
                continue

            # Use smoothed keypoints for angle computation to reduce jitter
            sc = {k: self._smooth_kpt(k) for k in [f'{p}_sh', f'{p}_el', f'{p}_wr']}
            if any(v is None for v in sc.values()):
                continue
            ac = self._get_angle(sc[f'{p}_sh'], sc[f'{p}_el'], sc[f'{p}_wr'])

            ap = self._arm_angle(prev, p)
            if ap is None:
                continue

            if f'{p}_wr' not in curr or f'{p}_wr' not in prev:
                continue
            wd = np.linalg.norm(
                np.array(curr[f'{p}_wr']) - np.array(prev[f'{p}_wr'])
            ) / self.shoulder_width

            delta = ac - ap

            if (ac > PUNCH_ANGLE_THRESHOLD
                    and delta > MIN_ANGLE_DELTA
                    and wd > MIN_WRIST_VELOCITY):
                self.punch_count += 1
                self.cooldown     = PUNCH_COOLDOWN
                setattr(self, attr, True)   # lock this arm until it retracts
                intensity = self.compute_intensity(side)
                return True, side, intensity

        return False, None, None


# ============================================================================
# TRACKING UTILITIES (unchanged from v1)
# ============================================================================

def compute_iou_matrix(tracks, detections):
    if not tracks or not len(detections):
        return np.zeros((len(tracks), len(detections)))
    m = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        x1t,y1t,x2t,y2t = t.box
        at = (x2t-x1t)*(y2t-y1t)
        for j, d in enumerate(detections):
            x1d,y1d,x2d,y2d = d
            ad = (x2d-x1d)*(y2d-y1d)
            xi1,yi1 = max(x1t,x1d), max(y1t,y1d)
            xi2,yi2 = min(x2t,x2d), min(y2t,y2d)
            if xi2>xi1 and yi2>yi1:
                inter = (xi2-xi1)*(yi2-yi1)
                m[i,j] = inter / (at + ad - inter)
    return m


def match_detections_to_tracks(tracks, detections, confidences):
    if not tracks:
        return [], list(range(len(detections)))
    iou   = compute_iou_matrix(tracks, detections)
    cost  = -iou - np.tile(confidences, (len(tracks), 1))
    for i in range(len(tracks)):
        if tracks[i].is_fighter and tracks[i].missed_frames == 0:
            cost[i] -= INERTIA_BONUS
    cost[iou < IoU_THRESHOLD_MATCH] = 1e9
    ti, di = linear_sum_assignment(cost)
    pairs  = [(t,d) for t,d in zip(ti,di) if cost[t,d] < 1e9]
    matched = {d for _,d in pairs}
    return pairs, [i for i in range(len(detections)) if i not in matched]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BOXING ANALYTICS v2  —  Detection + Pose + Punch Classification")
    print("=" * 70)

    # ── Validate paths ────────────────────────────────────────────────────────
    for p in [DETECTION_MODEL, POSE_MODEL, CLASSIFIER_MODEL, VIDEO_PATH]:
        if not Path(p).exists():
            print(f"  NOT FOUND: {p}"); return

    print(f"  Detection  : {DETECTION_MODEL}")
    print(f"  Pose       : {POSE_MODEL}")
    print(f"  Classifier : {CLASSIFIER_MODEL}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Device     : {DEVICE}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    det_model  = YOLO(DETECTION_MODEL)
    pose_model = YOLO(POSE_MODEL)
    clf_model, idx_to_cls = load_classifier(CLASSIFIER_MODEL, DEVICE)
    print(f"  Classes    : {list(idx_to_cls.values())}")

    # ── Video setup ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video      : {W}×{H} @ {fps:.1f} fps  ({total} frames)")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / f"v2_{ts}.mp4"
    writer     = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (W, H))

    # ── State ─────────────────────────────────────────────────────────────────
    tracks         = []
    fighter_states = {}
    kpt_buffers    = {}     # fighter_id → deque of (17, 2) full-frame px coords
    punch_labels   = {}     # fighter_id → {"type", "conf", "arm", "frames_left"}
    punch_history  = {}     # fighter_id → list of detected punch types
    next_id        = 1
    frame_n        = 0

    print("\nProcessing...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % 60 == 0:
            print(f"  Frame {frame_n}/{total}  ({100*frame_n/total:.1f}%)")

        # ── Stage 1: Detection ────────────────────────────────────────────────
        det_res = det_model(frame, conf=DETECTION_CONF, iou=DETECTION_IOU,
                            imgsz=IMAGE_SIZE, verbose=False)
        all_boxes, all_confs = [], []
        if det_res[0].boxes is not None:
            for b, c in zip(det_res[0].boxes.xyxy.cpu().numpy(),
                            det_res[0].boxes.conf.cpu().numpy()):
                all_boxes.append(b);  all_confs.append(float(c))
        all_boxes  = np.array(all_boxes)  if all_boxes  else np.empty((0,4))
        all_confs  = np.array(all_confs)  if all_confs  else np.array([])

        # ── Tracking (same logic as v1) ───────────────────────────────────────
        active = [t for t in tracks if t.is_active()]
        pairs, unmatched = match_detections_to_tracks(active, all_boxes, all_confs)

        for ti, di in pairs:
            active[ti].update(all_boxes[di], all_confs[di])
            active[ti].is_fighter = True
        for t in active:
            if not any(ti == active.index(t) for ti, _ in pairs):
                t.age_frame()

        scores = [(active[ti].conf*100 + INERTIA_BONUS if active[ti].is_fighter
                   else active[ti].conf*100, ti, di, 'matched')
                  for ti, di in pairs]
        scores += [(all_confs[di]*100, None, di, 'unmatched') for di in unmatched]
        scores.sort(reverse=True, key=lambda x: x[0])

        selected = set()
        for _, ti, di, status in scores[:MAX_FIGHTERS]:
            if status == 'matched':
                active[ti].is_fighter = True
                selected.add(active[ti].id)
            elif len([t for t in tracks if t.is_fighter and t.is_active()]) < MAX_FIGHTERS:
                nt = FighterTrack(next_id, all_boxes[di], all_confs[di])
                nt.is_fighter = True
                tracks.append(nt);  selected.add(next_id);  next_id += 1

        tracks = [t for t in tracks if t.is_active()]

        # ── Stage 2: Pose on fighter crops ────────────────────────────────────
        fighter_tracks = [t for t in tracks if t.is_fighter and t.is_active()
                          and t.missed_frames == 0]

        for track in fighter_tracks:
            x1,y1,x2,y2 = track.box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            if x2<=x1 or y2<=y1: continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            pose_res = pose_model(crop, conf=0.25, verbose=False)
            if pose_res[0].keypoints is None: continue

            kpts  = pose_res[0].keypoints.xy.cpu().numpy()
            confs = pose_res[0].keypoints.conf.cpu().numpy()
            if not len(kpts): continue

            # Translate to full frame
            kpts_full = kpts[0].copy()
            kpts_full[:, 0] += x1
            kpts_full[:, 1] += y1

            visible = np.sum(confs[0] > 0.5)
            if visible < MIN_VISIBLE_KEYPOINTS: continue

            # Init state / buffer
            if track.id not in fighter_states:
                fighter_states[track.id] = FighterState(track.id)
                kpt_buffers[track.id]    = deque(maxlen=CLF_SEQ_LEN)
                punch_labels[track.id]   = None
                punch_history[track.id]  = []

            # Update punch detector (needs full-frame pixel coords for distances)
            fighter_states[track.id].update_keypoints(kpts_full, confs[0])

            # Accumulate bbox-normalised keypoints for classifier.
            # kpts[0] is in crop-pixel coords; divide by crop dims → [0, 1].
            # This matches BoxingVI training scale (pose extracted from person crop).
            bw = float(x2 - x1) or 1.0
            bh = float(y2 - y1) or 1.0
            kpts_norm = kpts[0].copy()
            kpts_norm[:, 0] /= bw
            kpts_norm[:, 1] /= bh
            kpt_buffers[track.id].append(kpts_norm)  # (17, 2) in [0, 1] bbox-space

            # ── Classify on confirmed punch event ─────────────────────────────
            is_punch, arm_side, intensity = fighter_states[track.id].check_punch()
            if is_punch and len(kpt_buffers[track.id]) == CLF_SEQ_LEN:
                p_type, p_conf = classify_punch(
                    clf_model, kpt_buffers[track.id], DEVICE, idx_to_cls
                )
                if p_type:
                    punch_labels[track.id] = {
                        "type":        p_type,
                        "conf":        p_conf,
                        "arm":         arm_side,
                        "intensity":   intensity,
                        "frames_left": LABEL_PERSIST_FRAMES,
                    }
                    punch_history[track.id].append(p_type)
                    ilabel = intensity["label"] if intensity else "?"
                    print(f"  Fighter {track.id}  {p_type:<16}  "
                          f"{arm_side} arm  [{ilabel}]  conf={p_conf:.0%}  "
                          f"total={fighter_states[track.id].punch_count}")

        # ── Visualisation ──────────────────────────────────────────────────────
        for track in fighter_tracks:
            x1,y1,x2,y2 = track.box.astype(int)
            state = fighter_states.get(track.id)
            lbl   = punch_labels.get(track.id)

            # Determine display colour from current punch type (or default)
            box_color = track.color
            if lbl and lbl["frames_left"] > 0:
                box_color = PUNCH_COLORS.get(lbl["type"], track.color)
                lbl["frames_left"] -= 1

            # Bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 3)

            # Top label: Fighter ID + punch type + intensity
            count = state.punch_count if state else 0
            if lbl and lbl["frames_left"] >= 0:
                intensity  = lbl.get("intensity") or {}
                ilabel     = intensity.get("label", "")
                iscore     = intensity.get("score", 0.0)
                icolor     = INTENSITY_COLORS.get(ilabel, box_color)
                top_text   = f"F{track.id}: {lbl['type']}"
                intens_text = f"{ilabel}  {iscore:.2f}"
                text_col   = box_color
            else:
                top_text   = f"Fighter {track.id}"
                intens_text = ""
                text_col   = track.color
                icolor     = track.color

            cv2.putText(frame, top_text, (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2)

            # Intensity label (second line, coloured by Light/Medium/Heavy)
            if intens_text:
                cv2.putText(frame, intens_text, (x1, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, icolor, 2)

            # Bottom label: punch count
            cv2.putText(frame, f"Punches: {count}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)

        # Frame counter
        cv2.putText(frame, f"{frame_n}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # Legend (top-right): punch types + intensity scale
        lx = W - 240
        cv2.putText(frame, "Punch Types:", (lx, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        for i, (ptype, col) in enumerate(PUNCH_COLORS.items()):
            cv2.putText(frame, f"  {ptype}", (lx, 45 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        base_y = 45 + len(PUNCH_COLORS) * 18 + 10
        cv2.putText(frame, "Intensity:", (lx, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        for j, (ilabel, icol) in enumerate(INTENSITY_COLORS.items()):
            cv2.putText(frame, f"  {ilabel}", (lx, base_y + 18 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, icol, 1)

        writer.write(frame)

    cap.release()
    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Output: {out_path}")
    if out_path.exists():
        print(f"  Size  : {out_path.stat().st_size / 1e6:.1f} MB")

    print("\n  Punch Summary:")
    for fid, state in fighter_states.items():
        history = punch_history.get(fid, [])
        counts  = {}
        for pt in history:
            counts[pt] = counts.get(pt, 0) + 1
        print(f"\n  Fighter {fid}  —  {state.punch_count} total punches")
        for pt, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {pt:<18} {n:>3}")

        # Intensity breakdown from stored punch_labels history is not available here
        # (labels are overwritten each punch). Intensity was printed live per-punch above.


if __name__ == "__main__":
    main()
