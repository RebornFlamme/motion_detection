"""Full analysis: process all frames, output annotated video + plot.

Reads params.json, applies motion detection with GUM uncertainties,
produces publication-ready output.

Usage: uv run analyze.py
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from core import (
    OUTPUT_DIR,
    annotate_frame,
    calibration_from_params,
    compute_motion,
    compute_row_means,
    find_interface,
    find_video,
    hampel_filter,
    load_params,
    px_to_cm,
)

PROGRESS_BAR_WIDTH = 50


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    params = load_params()

    video_path = find_video()
    if not video_path:
        print("No video found in input/")
        sys.exit(1)

    # Unpack params
    delta = params["delta"]
    blur_k = params["blur"]
    threshold = float(params["threshold"])
    nb_valid = params["nb_valid"]
    skip = params.get("skip", 1)
    t_start = params.get("t_start", 0)

    px_per_cm, rel_sigma_k = calibration_from_params(params)
    cal_y2 = params.get("cal_y2", 100)

    # Open video
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_start = int(t_start * fps)
    frame_indices = list(range(frame_start, n, skip))
    n_to_process = len(frame_indices)

    # Print summary
    print(f"Video    : {video_path}")
    print(f"           {w}x{h}, {fps:.0f} fps, {n} frames")
    print(f"Params   : delta={delta} blur={blur_k} threshold={threshold} "
          f"nb_valid={nb_valid} skip={skip}")
    print(f"           t_start={t_start}s -> {n_to_process} frames to process")
    print(f"Calib    : {px_per_cm:.1f} px/cm, sigma_k/k={rel_sigma_k:.4f}")
    print(f"Output   : {OUTPUT_DIR}/\n")

    # Video writer
    out_video = str(OUTPUT_DIR / "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps / skip, (w, h))

    # Accumulate results
    times, centroids, sigmas_A = [], [], []
    gray_buf: dict[int, np.ndarray] = {}

    for count, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_buf[idx] = gray
        gray_buf.pop(idx - delta * skip - skip, None)

        t = (idx - frame_start) / fps
        prev_idx = idx - delta * skip

        if prev_idx >= 0:
            if prev_idx not in gray_buf:
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
                ret2, f2 = cap.read()
                if ret2:
                    gray_buf[prev_idx] = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            gray_prev = gray_buf.get(prev_idx)
            if gray_prev is not None:
                motion = compute_motion(gray, gray_prev, blur_k)
                rm = compute_row_means(motion)
                yc, sA, zt, zb = find_interface(rm, threshold, nb_valid)
                frame_out = annotate_frame(frame, rm, threshold, yc, zt, zb)
                times.append(t)
                centroids.append(yc)
                sigmas_A.append(sA)
            else:
                frame_out = frame
        else:
            frame_out = frame

        writer.write(frame_out)

        if count % 50 == 0 or count == n_to_process - 1:
            pct = (count + 1) / n_to_process
            filled = int(PROGRESS_BAR_WIDTH * pct)
            bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
            print(f"\r  [{bar}] {count+1}/{n_to_process} ({pct*100:.0f}%)",
                  end="", flush=True)

    print()
    cap.release()
    writer.release()
    print(f"  Video : {out_video}")

    # Convert to arrays
    times = np.array(times)
    centroids = np.array(centroids)
    sigmas_A = np.array(sigmas_A)
    centroids[centroids < 0] = np.nan

    # Save CSV
    csv_path = OUTPUT_DIR / "interface_data.csv"
    np.savetxt(csv_path,
               np.column_stack([times, centroids, sigmas_A]),
               delimiter=";",
               header="time_s;centroid_px;sigma_A_px",
               comments="", fmt="%.4f")
    print(f"  CSV   : {csv_path}")

    # ── Filter and convert ────────────────────────────────────────────────
    y_min = params.get("y_min", 0)
    valid = ~np.isnan(centroids) & (centroids > y_min)
    t_v = times[valid]
    h_cm = px_to_cm(centroids[valid], cal_y2, px_per_cm)
    sigma_A_cm = sigmas_A[valid] / px_per_cm
    sigma_B_cm = np.abs(h_cm) * rel_sigma_k
    u_cm = np.sqrt(sigma_A_cm**2 + sigma_B_cm**2)

    # Hampel outlier rejection
    hampel_mad = params.get("hampel_mad", 30) / 10.0
    outliers = hampel_filter(h_cm, window=7, threshold=hampel_mad)
    keep = ~outliers

    n_out = int(np.sum(outliers))
    print(f"\n  Uncertainty: u_mean={np.nanmean(u_cm):.2f} cm "
          f"(sigma_A={np.nanmean(sigma_A_cm):.2f}, "
          f"sigma_B={np.nanmean(sigma_B_cm):.3f})")
    print(f"  Hampel: {n_out}/{len(h_cm)} outliers removed")

    t_k, h_k, u_k = t_v[keep], h_cm[keep], u_cm[keep]

    # ── Plot ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 13, "axes.titlesize": 14, "legend.fontsize": 9,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t_k, h_k - 2 * u_k, h_k + 2 * u_k,
                    color="#2ca02c", alpha=0.10, label="$\\pm 2u$ (95\\%)")
    ax.fill_between(t_k, h_k - u_k, h_k + u_k,
                    color="#2ca02c", alpha=0.20, label="$\\pm u$ (68\\%)")
    ax.plot(t_k, h_k, "o", color="#1f77b4", ms=2, alpha=0.7, mew=0,
            label="Interface position")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Height $h$ [cm]")
    ax.set_title("Air/water interface evolution")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.4)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.25)

    n_valid = int(np.sum(keep))
    ax.text(0.99, 0.02,
            f"$\\delta$={delta}, blur={blur_k}, thr={threshold}, "
            f"valid={nb_valid}, skip={skip}",
            transform=ax.transAxes, fontsize=7, color="0.5",
            ha="right", va="bottom")
    ax.text(0.01, 0.02,
            f"N={n_valid} | $\\bar{{u}}$={np.nanmean(u_k):.2f} cm "
            f"| {px_per_cm:.1f} px/cm",
            transform=ax.transAxes, fontsize=7, color="0.5",
            ha="left", va="bottom")

    fig.tight_layout()
    for path in (OUTPUT_DIR / "interface_height.png",
                 OUTPUT_DIR / "interface_height.pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Plot  : {path}")
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
