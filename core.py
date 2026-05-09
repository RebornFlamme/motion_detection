"""Shared utilities for motion-based interface detection."""

import glob
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
INPUT_DIR = HERE / "input"
OUTPUT_DIR = HERE / "output"
PARAMS_FILE = HERE / "params.json"

VIDEO_EXTENSIONS = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV")

DEFAULT_PARAMS = {
    "delta": 4,
    "blur": 0,
    "threshold": 5,
    "nb_valid": 3,
    "skip": 1,
    "t_start": 0,
    "y_min": 0,
    "cal_y1": 0,
    "cal_y2": 100,
    "cal_mm": 100,
    "hampel_mad": 30,
}


# ── I/O ───────────────────────────────────────────────────────────────────

def find_video() -> str | None:
    """Return the first video file found in input/, or None."""
    for pat in VIDEO_EXTENSIONS:
        found = glob.glob(str(INPUT_DIR / pat))
        if found:
            return found[0]
    return None


def load_params() -> dict:
    """Load params.json merged with defaults."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return {**DEFAULT_PARAMS, **json.load(f)}
    return dict(DEFAULT_PARAMS)


def save_params(params: dict) -> None:
    """Write params to params.json."""
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved -> {PARAMS_FILE}")


def read_gray(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame idx and return it as grayscale, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None


def read_bgr(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame idx and return it as BGR, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


# ── Signal processing ─────────────────────────────────────────────────────

def compute_motion(gray_a: np.ndarray, gray_b: np.ndarray,
                   blur_k: int) -> np.ndarray:
    """Frame difference: (A + inv(B)) / 2, then optional Gaussian blur."""
    m = ((gray_a.astype(np.int16) + (255 - gray_b.astype(np.int16))) // 2)
    m = m.astype(np.uint8)
    if blur_k > 0:
        k = blur_k * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def compute_row_means(motion: np.ndarray) -> np.ndarray:
    """Mean |pixel - 128| per row (motion intensity profile)."""
    return np.mean(np.abs(motion.astype(np.float32) - 128.0), axis=1)


def find_interface(row_means: np.ndarray, threshold: float,
                   nb_valid: int) -> tuple[float, float, int, int]:
    """Detect interface as weighted centroid of the first motion zone.

    Returns:
        (y_centroid, sigma_A, zone_top, zone_bottom).
        Returns (-1, 0, -1, -1) if no interface detected.
    """
    above = row_means > threshold

    # Scan for first run of nb_valid consecutive rows above threshold
    count = 0
    start = -1
    for i, v in enumerate(above):
        if v:
            count += 1
            if count >= nb_valid:
                start = i - nb_valid + 1
                break
        else:
            count = 0

    if start < 0:
        return -1.0, 0.0, -1, -1

    # Extend zone to all contiguous rows above threshold
    ztop = start
    while ztop > 0 and above[ztop - 1]:
        ztop -= 1
    zbot = start + nb_valid - 1
    while zbot < len(above) - 1 and above[zbot + 1]:
        zbot += 1

    # Weighted centroid and sigma
    y = np.arange(ztop, zbot + 1, dtype=float)
    w = np.maximum(0.0, row_means[ztop:zbot + 1] - threshold)
    w_sum = np.sum(w)

    if w_sum < 1e-10:
        return float((ztop + zbot) / 2), 0.0, ztop, zbot

    y_centroid = np.sum(y * w) / w_sum
    sigma_A = np.sqrt(np.sum(w * (y - y_centroid) ** 2) / w_sum)
    return y_centroid, sigma_A, ztop, zbot


def hampel_filter(y: np.ndarray, window: int = 7,
                  threshold: float = 3.0) -> np.ndarray:
    """Mark outliers where |y - local_median| > threshold * MAD."""
    n = len(y)
    is_outlier = np.zeros(n, dtype=bool)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        local = y[lo:hi]
        med = np.nanmedian(local)
        mad = 1.4826 * np.nanmedian(np.abs(local - med))
        if mad > 1e-10 and np.abs(y[i] - med) > threshold * mad:
            is_outlier[i] = True
    return is_outlier


# ── Calibration ───────────────────────────────────────────────────────────

def calibration_from_params(params: dict) -> tuple[float, float]:
    """Compute px_per_cm and relative calibration uncertainty.

    Returns:
        (px_per_cm, rel_sigma_k).
    """
    px_dist = abs(params["cal_y2"] - params["cal_y1"])
    cal_cm = params["cal_mm"] / 10.0
    px_per_cm = px_dist / cal_cm if cal_cm > 0 and px_dist > 0 else 1.0
    rel_sigma_k = np.sqrt(2) / px_dist if px_dist > 0 else 0.01
    return px_per_cm, rel_sigma_k


def px_to_cm(y_px: np.ndarray, cal_y2: float,
             px_per_cm: float) -> np.ndarray:
    """Convert pixel y-coordinate to height in cm from cal_y2 (zero)."""
    return (cal_y2 - y_px) / px_per_cm


# ── Annotation ────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, row_means: np.ndarray,
                   threshold: float, y_centroid: float,
                   ztop: int, zbot: int) -> np.ndarray:
    """Draw green zone and yellow interface line on a frame."""
    out = frame.copy()
    h, nr = out.shape[0], len(row_means)
    if ztop >= 0:
        yt = int(ztop * h / nr)
        yb = int(zbot * h / nr)
        green = np.zeros_like(out[yt:yb + 1])
        green[:, :, 1] = 80
        out[yt:yb + 1] = cv2.add(out[yt:yb + 1], green)
    if y_centroid >= 0:
        iy = int(y_centroid * h / nr)
        cv2.line(out, (0, iy), (out.shape[1] - 1, iy), (0, 255, 255), 2)
    return out
