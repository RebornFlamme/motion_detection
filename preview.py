"""Interactive parameter tuning GUI.

Saves parameters to params.json on quit.
Keys: n/p = next/prev frame, s = save, q = quit (auto-saves).

Usage: uv run preview.py
"""

import sys

import cv2
import numpy as np

from core import (
    INPUT_DIR,
    annotate_frame,
    calibration_from_params,
    compute_motion,
    compute_row_means,
    find_interface,
    find_video,
    load_params,
    px_to_cm,
    read_bgr,
    read_gray,
    save_params,
)

DISPLAY_W = 360
DISPLAY_H = 640
PROFILE_W = 200
MAX_CACHE = 300


def draw_profile(row_means, h, w, threshold, iface, ztop, zbot):
    """Vertical bar chart of row-mean values with annotations."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mx = max(np.max(row_means), 1.0)
    nr = len(row_means)
    for y in range(h):
        val = row_means[int(y * nr / h)]
        bl = int(val / mx * (w - 10))
        col = (0, 255, 0) if val > threshold else (180, 180, 180)
        cv2.line(canvas, (5, y), (5 + bl, y), col, 1)
    tx = int(threshold / mx * (w - 10)) + 5
    cv2.line(canvas, (tx, 0), (tx, h - 1), (0, 0, 255), 1)
    if iface >= 0:
        cv2.line(canvas, (0, int(iface * h / nr)),
                 (w - 1, int(iface * h / nr)), (0, 255, 255), 2)
    if ztop >= 0:
        cv2.rectangle(canvas, (0, int(ztop * h / nr)),
                      (4, int(zbot * h / nr)), (0, 200, 0), -1)
    return canvas


def nothing(_):
    pass


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    video_path = find_video()
    if not video_path:
        print(f"No video found in {INPUT_DIR}/")
        sys.exit(1)

    params = load_params()
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path} ({n} frames, {fps:.0f} fps)")

    # Window + sliders
    win = "Preview — 's' save, 'q' quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_W * 2 + PROFILE_W, DISPLAY_H)

    sliders = {
        "frame":      (1, n - 1),
        "delta":      (params["delta"], 120),
        "blur":       (params["blur"], 30),
        "threshold":  (params["threshold"], 80),
        "nb_valid":   (params["nb_valid"], 50),
        "skip":       (params["skip"], 30),
        "t_start":    (params["t_start"], int(n / fps)),
        "y_min":      (params["y_min"], h),
        "cal_y1":     (params["cal_y1"], h),
        "cal_y2":     (params["cal_y2"], h),
        "cal_mm":     (params["cal_mm"], 500),
        "hampel_mad": (params["hampel_mad"], 50),
    }
    for name, (default, maximum) in sliders.items():
        cv2.createTrackbar(name, win, default, maximum, nothing)

    cache_g: dict[int, np.ndarray] = {}
    cache_c: dict[int, np.ndarray] = {}

    def cached_gray(i):
        if i not in cache_g:
            g = read_gray(cap, i)
            if g is not None:
                if len(cache_g) > MAX_CACHE:
                    del cache_g[next(iter(cache_g))]
                cache_g[i] = g
        return cache_g.get(i)

    def cached_color(i):
        if i not in cache_c:
            f = read_bgr(cap, i)
            if f is not None:
                if len(cache_c) > MAX_CACHE:
                    del cache_c[next(iter(cache_c))]
                cache_c[i] = f
        return cache_c.get(i)

    def read_sliders():
        return {
            "delta":      max(1, cv2.getTrackbarPos("delta", win)),
            "blur":       cv2.getTrackbarPos("blur", win),
            "threshold":  cv2.getTrackbarPos("threshold", win),
            "nb_valid":   max(1, cv2.getTrackbarPos("nb_valid", win)),
            "skip":       max(1, cv2.getTrackbarPos("skip", win)),
            "t_start":    cv2.getTrackbarPos("t_start", win),
            "y_min":      cv2.getTrackbarPos("y_min", win),
            "cal_y1":     cv2.getTrackbarPos("cal_y1", win),
            "cal_y2":     cv2.getTrackbarPos("cal_y2", win),
            "cal_mm":     max(1, cv2.getTrackbarPos("cal_mm", win)),
            "hampel_mad": max(1, cv2.getTrackbarPos("hampel_mad", win)),
        }

    prev_state = None

    while True:
        idx = cv2.getTrackbarPos("frame", win)
        p = read_sliders()
        delta = p["delta"]
        idx = max(delta, min(idx, n - 1))
        state = (idx, *p.values())

        # Skip redraw if nothing changed
        if state == prev_state:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                save_params(read_sliders())
                break
            elif key == ord("s"):
                save_params(read_sliders())
            elif key == ord("n"):
                cv2.setTrackbarPos("frame", win, min(idx + 1, n - 1))
            elif key == ord("p"):
                cv2.setTrackbarPos("frame", win, max(delta, idx - 1))
            continue
        prev_state = state

        # Read frames
        gc = cached_gray(idx)
        gp = cached_gray(idx - delta)
        fc = cached_color(idx)
        if gc is None or gp is None or fc is None:
            cv2.waitKey(30)
            continue

        # Compute
        motion = compute_motion(gc, gp, p["blur"])
        rm = compute_row_means(motion)
        iface, _, zt, zb = find_interface(rm, float(p["threshold"]), p["nb_valid"])

        px_per_cm, _ = calibration_from_params(p)

        # Build panels
        left = cv2.resize(
            annotate_frame(fc, rm, p["threshold"], iface, zt, zb),
            (DISPLAY_W, DISPLAY_H))
        mid = cv2.resize(
            cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR),
            (DISPLAY_W, DISPLAY_H))
        right = draw_profile(rm, DISPLAY_H, PROFILE_W, p["threshold"],
                             iface, zt, zb)

        # Overlays on left panel
        t_cur = idx / fps
        cv2.putText(left, f"#{idx} t={t_cur:.1f}s d={delta} b={p['blur']}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if iface >= 0:
            iface_cm = px_to_cm(np.array([iface]), p["cal_y2"], px_per_cm)[0]
            cv2.putText(left, f"iface={iface_cm:.1f} cm (px={iface:.0f} [{zt}-{zb}])",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(left, f"skip={p['skip']} t_start={p['t_start']}s y_min={p['y_min']}",
                    (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        px_dist = abs(p["cal_y2"] - p["cal_y1"])
        cv2.putText(left, f"CAL: {px_dist}px={p['cal_mm']}mm ({px_per_cm:.1f}px/cm)",
                    (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        # y_min line
        if p["y_min"] > 0:
            ym = int(p["y_min"] * DISPLAY_H / h)
            cv2.line(left, (0, ym), (DISPLAY_W - 1, ym), (0, 0, 255), 1)

        # Calibration lines + arrows
        cy1 = int(p["cal_y1"] * DISPLAY_H / h)
        cy2 = int(p["cal_y2"] * DISPLAY_H / h)
        cv2.line(left, (0, cy1), (DISPLAY_W - 1, cy1), (255, 0, 255), 1)
        cv2.line(left, (0, cy2), (DISPLAY_W - 1, cy2), (255, 0, 255), 1)
        mx = DISPLAY_W - 15
        cv2.arrowedLine(left, (mx, cy1), (mx, cy2), (255, 0, 255), 1)
        cv2.arrowedLine(left, (mx, cy2), (mx, cy1), (255, 0, 255), 1)

        cv2.putText(left, "'s' save  'q' quit",
                    (5, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1)

        cv2.imshow(win, np.hstack([left, mid, right]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            save_params(read_sliders())
            break
        elif key == ord("s"):
            save_params(read_sliders())
        elif key == ord("n"):
            cv2.setTrackbarPos("frame", win, min(idx + 1, n - 1))
        elif key == ord("p"):
            cv2.setTrackbarPos("frame", win, max(delta, idx - 1))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
