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
VIDEO_PATH         = "Recording 2025-12-08 051750.mp4"
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

# Classifier keypoint smoothing (EMA applied before buffering)
CLF_KPT_SMOOTH_ALPHA = 0.4   # higher = more responsive, lower = smoother

# Display: how many frames to keep showing the last punch label
LABEL_PERSIST_FRAMES = 45

# ── Intensity assessment (impulse-based with adaptive normalization) ──────────
INTENSITY_HISTORY_LEN          = 15    # wider window for intensity (separate from 5-frame punch detection)
INTENSITY_MIN_PUNCHES_ADAPTIVE = 5     # punches before switching to per-fighter adaptive normalization

# Fallback normalization maxima (used until adaptive kicks in).
# Calibrated for impulse/percentile features at 30fps with 15-frame window.
INTENSITY_V_MAX          = 4.0     # wrist velocity impulse (sum of shoulder-widths/frame)
INTENSITY_OE_MAX         = 25.0    # 95th-pct elbow angular velocity (degrees/frame)
INTENSITY_OS_MAX         = 18.0    # 95th-pct shoulder rotation speed (degrees/frame)
INTENSITY_JERK_MAX       = 0.5     # 95th-pct wrist jerk (Δvelocity/frame)
INTENSITY_HIP_MAX        = 2.0     # hip midpoint impulse (sum of shoulder-widths/frame)
INTENSITY_DECEL_MAX      = 0.4     # max post-peak wrist deceleration per frame

# Per-punch-type weight profiles: (v_impulse, omega_elbow, omega_shoulder, jerk, hip_impulse, deceleration)
INTENSITY_WEIGHTS_DEFAULT = (0.25, 0.14, 0.16, 0.08, 0.20, 0.17)
INTENSITY_TYPE_WEIGHTS = {
    "Jab":            (0.32, 0.16, 0.10, 0.08, 0.18, 0.16),
    "Cross":          (0.32, 0.16, 0.10, 0.08, 0.18, 0.16),
    "Lead Hook":      (0.12, 0.08, 0.35, 0.08, 0.22, 0.15),
    "Rear Hook":      (0.12, 0.08, 0.35, 0.08, 0.22, 0.15),
    "Lead Uppercut":  (0.25, 0.16, 0.13, 0.08, 0.22, 0.16),
    "Rear Uppercut":  (0.25, 0.16, 0.13, 0.08, 0.22, 0.16),
}
INTENSITY_LIGHT_THRESH   = 0.40
INTENSITY_HEAVY_THRESH   = 0.70

# Intensity label colours (BGR)
INTENSITY_COLORS = {
    "Light":  (0,   220, 220),   # yellow
    "Medium": (0,   140, 255),   # orange
    "Heavy":  (0,   0,   220),   # red
}

# ── Referee module ─────────────────────────────────────────────────────────
REFEREE_CONF_THRESH = 0.30   # min detection confidence for referee slot

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
    if len(valid) == 0:
        return seq_3d          # all-zero/padding frames — return unchanged, no division
    scale = max(float(valid.mean()), 0.05)  # prevent extreme compression
    return seq_3d / scale


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


# COCO anatomical left/right pairs — must be swapped when mirroring horizontally
_COCO_LR_SWAP = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _fighter_facing_left(kpt_frame: np.ndarray) -> bool:
    """
    Returns True if the fighter is facing left in bbox-normalised [0,1] coords.
    Nose (kpt 0) x < shoulder midpoint x  ⟹  facing left.
    Falls back to False when keypoints are missing/zero.
    """
    nose_x   = float(kpt_frame[0, 0])
    sh_mid_x = float((kpt_frame[5, 0] + kpt_frame[6, 0]) / 2.0)
    if sh_mid_x < 1e-4:          # keypoints missing — default to right-facing
        return False
    return nose_x < sh_mid_x


def _mirror_buffer(buf: list) -> list:
    """
    Horizontally mirror a list of (17, 2) bbox-normalised keypoint frames.
    Steps: flip x  (x → 1-x)  then swap all anatomical L/R keypoint pairs.
    This maps a left-facing fighter into the same coordinate space as the
    right-facing fighters the classifier was trained on.
    """
    out = []
    for frame in buf:
        f = frame.copy()
        f[:, 0] = 1.0 - f[:, 0]          # mirror x in [0,1] space
        for a, b in _COCO_LR_SWAP:
            f[[a, b]] = f[[b, a]]          # swap anatomical L↔R indices
        out.append(f)
    return out


CLF_MIN_FRAMES = 15   # minimum frames for classification (zero-pad the rest)


@torch.no_grad()
def classify_punch(model, kpt_buffer, device: str, idx_to_cls: dict):
    """
    Run the classifier on a keypoint buffer.

    kpt_buffer : list or deque of (17, 2) arrays already normalised to [0, 1]
                 relative to the fighter's bounding box (crop-space).
                 May be shorter than CLF_SEQ_LEN — will be zero-padded at START
                 to match training format (zeros at start, real frames at end).
    Returns    : (punch_type: str, confidence: float)
    """
    n_real = len(kpt_buffer)
    if n_real < CLF_MIN_FRAMES:
        return None, 0.0

    # ── Reposition to training format: zeros at START, real frames at END ──
    # Training (train_punch_classifier.py:196-197) always puts real frames at
    # the end so the BiLSTM + attention learns to look at the final frames for
    # the punch.  For full 25-frame buffers this is a no-op (same layout).
    seq = np.zeros((CLF_SEQ_LEN, 17, 2), dtype=np.float32)
    real = np.array(list(kpt_buffer)[-CLF_SEQ_LEN:], dtype=np.float32)
    seq[-len(real):] = real

    # Hip-centre + torso-height normalisation — must match training preprocessing
    seq = _hip_normalize(seq)
    seq = _torso_normalize(seq)

    flat   = seq.reshape(CLF_SEQ_LEN, CLF_KPT_FEATURES)         # (25, 34)
    angles = compute_angle_features(seq)                          # (25,  7)
    vel    = np.diff(flat, axis=0, prepend=flat[[0]])             # (25, 34) Δpos/frame
    full   = np.concatenate([flat, angles, vel], axis=1)         # (25, 75)

    # NaN guard — reject if normalization or angle computation produced NaN/inf
    if not np.isfinite(full).all():
        return None, 0.0

    t   = torch.tensor(full, dtype=torch.float32).unsqueeze(0).to(device)
    out = model(t)
    probs      = torch.softmax(out, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    return idx_to_cls[pred_idx], confidence


# ============================================================================
# TRACKING (unchanged from v1)
# ============================================================================

BBOX_EMA_ALPHA = 0.3   # low alpha = more smoothing; 0.3 balances lag vs jitter

class FighterTrack:
    def __init__(self, track_id, box, conf):
        self.id = track_id;  self.box = box;  self.conf = conf
        self.smooth_box = box.astype(np.float64).copy()  # EMA-smoothed bbox
        self.age = 1;  self.missed_frames = 0;  self.is_fighter = False
        np.random.seed(track_id)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())

    def update(self, box, conf):
        self.box = box;  self.conf = conf
        self.smooth_box = (BBOX_EMA_ALPHA * box.astype(np.float64)
                           + (1 - BBOX_EMA_ALPHA) * self.smooth_box)
        self.age += 1;   self.missed_frames = 0

    def age_frame(self):
        self.missed_frames += 1;  self.age += 1

    def is_active(self):
        return self.missed_frames < MAX_TRACK_HISTORY


class FighterState:
    def __init__(self, fighter_id):
        self.fighter_id        = fighter_id
        self.punch_count       = 0
        self.history           = deque(maxlen=5)
        self.intensity_history = deque(maxlen=INTENSITY_HISTORY_LEN)
        self.cooldown_l        = 0       # per-arm cooldown (left)
        self.cooldown_r        = 0       # per-arm cooldown (right)
        self.is_punching_l     = False   # per-arm retraction gate (left)
        self.is_punching_r     = False   # per-arm retraction gate (right)
        self.shoulder_width    = 1.0
        self.raw_feature_log   = []      # raw feature tuples for adaptive normalization

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
        # Arm + hip keypoints. Hips feed the hip-impulse intensity feature.
        arm_mapping = {'l_sh':5,'r_sh':6,'l_el':7,'r_el':8,'l_wr':9,'r_wr':10}
        hip_mapping = {'l_hip':11,'r_hip':12}
        data = {}
        for name, idx in {**arm_mapping, **hip_mapping}.items():
            if confidences[idx] > 0.5:
                data[name] = keypoints[idx]
        # Require ≥ 4 ARM keypoints (hips don't count toward this threshold)
        arm_count = sum(1 for k in data if k in arm_mapping)
        if arm_count >= 4:
            if 'l_sh' in data and 'r_sh' in data:
                sw = np.linalg.norm(data['r_sh'] - data['l_sh'])
                if sw > 10:
                    if self.shoulder_width <= 1.0:   # first real measurement
                        self.shoulder_width = sw
                    else:
                        self.shoulder_width = 0.8 * self.shoulder_width + 0.2 * sw
            self.history.append(data)
            self.intensity_history.append(data)

    def _smooth_kpt(self, key):
        """5-frame rolling mean on a named keypoint — reduces 23° RMSD pose jitter."""
        vals = [h[key] for h in self.history if key in h]
        return np.mean(vals, axis=0) if vals else None

    def compute_intensity(self, side: str, punch_type: str = None) -> dict:
        """
        Compute punch intensity from the wider intensity history window.

        Uses impulse-based features (sum over window) instead of peak values,
        per-fighter adaptive normalization, and punch-type-specific weights.

        Features (6 total, all normalized to [0, 1]):
          v_impulse      — sum of per-frame wrist displacement / shoulder_width
          omega_elbow    — 95th percentile per-frame elbow angle delta
          omega_shoulder — 95th percentile per-frame shoulder rotation speed
          jerk           — 95th percentile rate of change of wrist velocity
          hip_impulse    — sum of per-frame hip midpoint displacement / shoulder_width
          deceleration   — max post-peak wrist velocity drop (commitment signal)
        """
        hist = list(self.intensity_history)
        n    = len(hist)
        p    = 'l' if side == 'left' else 'r'
        sw   = max(self.shoulder_width, 1.0)

        # ── Per-frame wrist velocities ──────────────────────────────────────
        wrist_vels = []
        for i in range(1, n):
            if f'{p}_wr' in hist[i] and f'{p}_wr' in hist[i - 1]:
                d = np.linalg.norm(
                    np.array(hist[i][f'{p}_wr']) - np.array(hist[i - 1][f'{p}_wr'])
                ) / sw
                wrist_vels.append(d)

        v_impulse = sum(wrist_vels) if wrist_vels else 0.0

        # ── Wrist jerk: 95th percentile of velocity deltas ─────────────────
        jerk_vals = []
        if len(wrist_vels) >= 2:
            jerk_vals = [abs(wrist_vels[i] - wrist_vels[i - 1])
                         for i in range(1, len(wrist_vels))]
        jerk_95 = (float(np.percentile(jerk_vals, 95)) if len(jerk_vals) >= 3
                   else (max(jerk_vals) if jerk_vals else 0.0))

        # ── Per-frame elbow angular velocity ───────────────────────────────
        elbow_deltas = []
        for i in range(1, n):
            a_curr = self._arm_angle(hist[i], p)
            a_prev = self._arm_angle(hist[i - 1], p)
            if a_curr is not None and a_prev is not None:
                elbow_deltas.append(abs(a_curr - a_prev))
        omega_elbow = (float(np.percentile(elbow_deltas, 95)) if len(elbow_deltas) >= 3
                       else (max(elbow_deltas) if elbow_deltas else 0.0))

        # ── Per-frame shoulder rotation speed ──────────────────────────────
        sho_deltas = []
        for i in range(1, n):
            h_c, h_p = hist[i], hist[i - 1]
            if all(k in h_c and k in h_p for k in ('l_sh', 'r_sh')):
                sv_c = np.array(h_c['r_sh']) - np.array(h_c['l_sh'])
                sv_p = np.array(h_p['r_sh']) - np.array(h_p['l_sh'])
                ang_c = np.degrees(np.arctan2(sv_c[1], sv_c[0]))
                ang_p = np.degrees(np.arctan2(sv_p[1], sv_p[0]))
                d = abs(ang_c - ang_p)
                sho_deltas.append(360.0 - d if d > 180.0 else d)
        omega_shoulder = (float(np.percentile(sho_deltas, 95)) if len(sho_deltas) >= 3
                          else (max(sho_deltas) if sho_deltas else 0.0))

        # ── Hip midpoint impulse (kinetic chain proxy) ─────────────────────
        hip_disps = []
        for i in range(1, n):
            h_c, h_p = hist[i], hist[i - 1]
            if all(k in h_c and k in h_p for k in ('l_hip', 'r_hip')):
                mid_c = (np.array(h_c['l_hip']) + np.array(h_c['r_hip'])) / 2
                mid_p = (np.array(h_p['l_hip']) + np.array(h_p['r_hip'])) / 2
                hip_disps.append(np.linalg.norm(mid_c - mid_p) / sw)
        hip_impulse = sum(hip_disps) if hip_disps else 0.0

        # ── Deceleration: max post-peak velocity drop (commitment signal) ──
        decel = 0.0
        if len(wrist_vels) >= 3:
            peak_idx = int(np.argmax(wrist_vels))
            post_peak = wrist_vels[peak_idx:]
            if len(post_peak) >= 2:
                decel = max((post_peak[i - 1] - post_peak[i])
                            for i in range(1, len(post_peak)))
                decel = max(decel, 0.0)

        # ── Raw feature vector ─────────────────────────────────────────────
        raw = (v_impulse, omega_elbow, omega_shoulder, jerk_95, hip_impulse, decel)

        # ── Adaptive normalization ─────────────────────────────────────────
        if len(self.raw_feature_log) >= INTENSITY_MIN_PUNCHES_ADAPTIVE:
            arr = np.array(self.raw_feature_log)
            maxima = [max(float(np.percentile(arr[:, i], 95)), 1e-6)
                      for i in range(6)]
        else:
            maxima = [INTENSITY_V_MAX, INTENSITY_OE_MAX, INTENSITY_OS_MAX,
                      INTENSITY_JERK_MAX, INTENSITY_HIP_MAX, INTENSITY_DECEL_MAX]

        normed = [min(r / m, 1.0) for r, m in zip(raw, maxima)]

        # ── Punch-type-specific weights ────────────────────────────────────
        weights = INTENSITY_TYPE_WEIGHTS.get(punch_type, INTENSITY_WEIGHTS_DEFAULT)
        score = sum(w * v for w, v in zip(weights, normed))

        # ── Store raw features for adaptive normalization ──────────────────
        self.raw_feature_log.append(raw)

        if score < INTENSITY_LIGHT_THRESH:
            label = "Light"
        elif score < INTENSITY_HEAVY_THRESH:
            label = "Medium"
        else:
            label = "Heavy"

        return {
            "label":          label,
            "score":          round(float(score), 3),
            "v_impulse":      round(float(v_impulse), 3),
            "omega_elbow":    round(float(omega_elbow), 3),
            "omega_shoulder": round(float(omega_shoulder), 3),
            "jerk":           round(float(jerk_95), 3),
            "hip_impulse":    round(float(hip_impulse), 3),
            "deceleration":   round(float(decel), 3),
        }

    def check_punch(self):
        """
        Returns (fired, side) on a confirmed punch, else (False, None).
        Intensity is computed separately in the main loop after classification.
        """
        if not self.history:
            return False, None

        curr = self.history[-1]

        # Always update per-arm retraction state regardless of cooldown
        for p, attr in [('l', 'is_punching_l'), ('r', 'is_punching_r')]:
            a = self._arm_angle(curr, p)
            if a is not None and a < RESET_ANGLE_THRESHOLD:
                setattr(self, attr, False)

        # Tick down per-arm cooldowns every frame
        if self.cooldown_l > 0: self.cooldown_l -= 1
        if self.cooldown_r > 0: self.cooldown_r -= 1

        if len(self.history) < 3:
            return False, None

        prev = self.history[0]   # oldest frame in window

        for side, p, attr in [('left', 'l', 'is_punching_l'),
                               ('right', 'r', 'is_punching_r')]:
            # Per-arm cooldown: skip if this arm is still cooling down
            cd_attr = f'cooldown_{p}'
            if getattr(self, cd_attr) > 0:
                continue

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
                setattr(self, cd_attr, PUNCH_COOLDOWN)  # per-arm cooldown
                setattr(self, attr, True)   # lock this arm until it retracts
                return True, side

        return False, None


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


def _recycle_fighter_id(all_tracks: list, box: np.ndarray):
    """
    When creating a new fighter track, check whether an *expired* fighter track
    sat near the same position.  If so, reuse its ID to keep punch counts and
    keypoint buffers continuous across brief occlusions / exits-from-frame.

    Returns the recycled track ID if a close match is found, else None.
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    bw = max(box[2] - box[0], 1.0)
    bh = max(box[3] - box[1], 1.0)
    best_id, best_dist = None, float('inf')
    for t in all_tracks:
        if not t.is_fighter or t.is_active():
            continue   # only consider expired fighter tracks
        tcx = (t.box[0] + t.box[2]) / 2
        tcy = (t.box[1] + t.box[3]) / 2
        # Normalised centroid distance (relative to bbox size)
        dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5 / max(bw, bh)
        if dist < 0.8 and dist < best_dist:   # within 80% of bbox diagonal
            best_dist, best_id = dist, t.id
    return best_id


def _box_iou(box_a, box_b):
    """IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0]);  y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]);  y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (a1 + a2 - inter + 1e-6)


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
    kpt_buffers    = {}     # fighter_id → deque of (17, 2) bbox-normalised coords
    clf_prev_kpts  = {}     # fighter_id → last smoothed (17, 2) for classifier EMA
    last_good_kpts = {}     # fighter_id → last valid (17, 2) for hold-on-fail
    kpt_frame_ids  = {}     # fighter_id → deque of frame numbers (contiguity)
    punch_labels   = {}     # fighter_id → {"type", "conf", "arm", "frames_left"}
    punch_history  = {}     # fighter_id → list of detected punch types
    next_id        = 1
    frame_n        = 0
    referee_track  = None   # single FighterTrack slot for the referee

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
                # Reuse an expired fighter ID if this detection is near where
                # that fighter was last seen — keeps punch counts continuous.
                recycled = _recycle_fighter_id(tracks, all_boxes[di])
                use_id   = recycled if recycled is not None else next_id
                if recycled is None:
                    next_id += 1
                nt = FighterTrack(use_id, all_boxes[di], all_confs[di])
                nt.is_fighter = True
                tracks.append(nt);  selected.add(use_id)

        tracks = [t for t in tracks if t.is_active()]

        # ── Referee slot (3rd detected person) ───────────────────────────────
        # Any detection that does NOT overlap an active fighter becomes a referee
        # candidate.  The highest-confidence non-overlapping detection is kept.
        active_fighter_boxes = [
            t.box for t in tracks
            if t.is_fighter and t.is_active() and t.missed_frames == 0
        ]
        ref_candidates = []
        for di in range(len(all_boxes)):
            if all_confs[di] < REFEREE_CONF_THRESH:
                continue
            overlaps = any(
                _box_iou(all_boxes[di], fb) > 0.25 for fb in active_fighter_boxes
            )
            if not overlaps:
                ref_candidates.append((all_confs[di], all_boxes[di]))

        if ref_candidates:
            ref_candidates.sort(reverse=True, key=lambda x: x[0])
            rc_conf, rc_box = ref_candidates[0]
            if referee_track is None:
                referee_track = FighterTrack(next_id, rc_box, rc_conf)
                next_id += 1
            else:
                # Update in-place — accept regardless of IoU (referee can move fast)
                referee_track.box          = rc_box
                referee_track.conf         = rc_conf
                referee_track.missed_frames = 0
                referee_track.age         += 1
        elif referee_track is not None:
            referee_track.age_frame()
            if not referee_track.is_active():
                referee_track = None

        # ── Stage 2: Pose on fighter crops ────────────────────────────────────
        fighter_tracks = [t for t in tracks if t.is_fighter and t.is_active()
                          and t.missed_frames == 0]

        for track in fighter_tracks:
            # Use EMA-smoothed bbox for pose cropping to reduce keypoint jitter
            x1,y1,x2,y2 = track.smooth_box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            if x2<=x1 or y2<=y1: continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # --- Attempt pose estimation ---
            pose_ok = False
            pose_res = pose_model(crop, conf=0.25, verbose=False)
            if (pose_res[0].keypoints is not None):
                kpts  = pose_res[0].keypoints.xy.cpu().numpy()
                confs = pose_res[0].keypoints.conf.cpu().numpy()
                if len(kpts):
                    kpts_full = kpts[0].copy()
                    kpts_full[:, 0] += x1
                    kpts_full[:, 1] += y1
                    visible = np.sum(confs[0] > 0.5)
                    if visible >= MIN_VISIBLE_KEYPOINTS:
                        pose_ok = True

            # Init state / buffer (must happen even on pose failure for hold-last)
            if track.id not in fighter_states:
                if not pose_ok:
                    continue  # can't init without a valid first frame
                fighter_states[track.id] = FighterState(track.id)
                kpt_buffers[track.id]    = deque(maxlen=CLF_SEQ_LEN)
                kpt_frame_ids[track.id]  = deque(maxlen=CLF_SEQ_LEN)
                punch_labels[track.id]   = None
                punch_history[track.id]  = []

            if pose_ok:
                # Update punch detector (needs full-frame pixel coords)
                fighter_states[track.id].update_keypoints(kpts_full, confs[0])

                # Bbox-normalise for classifier
                bw = float(x2 - x1) or 1.0
                bh = float(y2 - y1) or 1.0
                kpts_norm = kpts[0].copy()
                kpts_norm[:, 0] /= bw
                kpts_norm[:, 1] /= bh

                # EMA smoothing on classifier keypoints — reduces crop jitter
                if track.id in clf_prev_kpts:
                    kpts_smooth = (CLF_KPT_SMOOTH_ALPHA * kpts_norm
                                   + (1 - CLF_KPT_SMOOTH_ALPHA) * clf_prev_kpts[track.id])
                else:
                    kpts_smooth = kpts_norm
                clf_prev_kpts[track.id] = kpts_smooth
                last_good_kpts[track.id] = kpts_smooth

                kpt_buffers[track.id].append(kpts_smooth)
                kpt_frame_ids[track.id].append(frame_n)
            elif track.id in last_good_kpts:
                # Hold last good keypoints for 1-2 frame gaps (maintains contiguity)
                kpt_buffers[track.id].append(last_good_kpts[track.id])
                kpt_frame_ids[track.id].append(frame_n)

            # ── Classify + compute intensity on confirmed punch event ─────────
            is_punch, arm_side = fighter_states[track.id].check_punch()
            if is_punch:
                p_type = None
                p_conf = 0.0

                # Attempt classification (requires enough frames with good contiguity)
                if len(kpt_buffers[track.id]) >= CLF_MIN_FRAMES:
                    fids = list(kpt_frame_ids[track.id])
                    max_gap = max((fids[i] - fids[i-1]) for i in range(1, len(fids))) if len(fids) > 1 else 1
                    if max_gap <= 3:
                        clf_buf = list(kpt_buffers[track.id])
                        if _fighter_facing_left(clf_buf[-1]):
                            clf_buf = _mirror_buffer(clf_buf)
                        p_type, p_conf = classify_punch(
                            clf_model, clf_buf, DEVICE, idx_to_cls
                        )

                # Compute intensity AFTER classification so we can use type-specific weights
                intensity = fighter_states[track.id].compute_intensity(
                    arm_side, punch_type=p_type
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
                    ilabel = intensity["label"]
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

        # ── Referee bbox ───────────────────────────────────────────────────────
        if referee_track is not None and referee_track.missed_frames == 0:
            rx1, ry1, rx2, ry2 = referee_track.box.astype(int)
            ref_col = (160, 160, 160)
            # Corner-bracket style (distinguishes referee from fighters)
            corner = 22
            for cx, cy, dx, dy in [(rx1, ry1, 1, 1), (rx2, ry1, -1, 1),
                                    (rx1, ry2, 1, -1), (rx2, ry2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy), ref_col, 2)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner), ref_col, 2)
            cv2.putText(frame, "Referee", (rx1, max(ry1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ref_col, 2)

        # Frame counter
        cv2.putText(frame, f"{frame_n}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # Legend (top-right): punch types + intensity scale + referee status
        lx = W - 245
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
