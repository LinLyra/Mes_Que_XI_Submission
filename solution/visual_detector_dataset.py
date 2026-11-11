"""Export a private, crop-based YOLO dataset from the tactical video.

ATD plus homography supply weak ball centres. Random crop jitter prevents the
network from learning that the ball is always at the crop centre. Temporal
blocks, rather than random frames, define validation data to prevent leakage.
All generated images and labels live below the ignored diagnostics directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.load_homography import MATRIX_COLUMNS
from visual_pseudolabels import HOM, RAW, ball_track

VIDEO = ROOT / "data/tactical/BAR-ATH_tactical.mp4"
OUT = ROOT / "evaluation/diagnostics/ball_detector"
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080


def verified_ball_center(image: np.ndarray, expected_x: float, expected_y: float,
                         radius: int = 18):
    """Return a compact bright candidate near the weak ATD projection.

    This rejects the many nominally present ATD ball tracks that visibly land
    on grass or players. It is intentionally conservative and does not try to
    recover distant candidates, because those cannot be distinguished safely
    from boots, markings and the white away kit without a trained detector.
    """
    size = 2 * radius + 1
    patch = cv2.getRectSubPix(image, (size, size),
                              (float(expected_x), float(expected_y)))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 145)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    centre = np.array([radius, radius], dtype=float)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        # Tiny 2--7 px highlights are abundant in compressed grass texture and
        # produced unacceptable false pseudo-labels in visual audits.
        if 8 <= area <= 120 and width <= 18 and height <= 18:
            distance = float(np.linalg.norm(centroids[index] - centre))
            if distance <= radius:
                candidates.append((distance, centroids[index]))
    if not candidates:
        return None
    _, local = min(candidates, key=lambda item: item[0])
    return np.array([expected_x, expected_y]) + local - centre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=int, default=160)
    parser.add_argument("--jitter", type=int, default=45)
    parser.add_argument("--stride", type=int, default=10,
                        help="sample every Nth visible ball frame")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional smoke-test limit; zero exports all")
    parser.add_argument("--keep-existing", action="store_true",
                        help="do not clear previously generated JPG/TXT files")
    parser.add_argument("--verify-radius", type=int, default=18,
                        help="maximum pixel correction for visual verification")
    args = parser.parse_args()
    if args.crop <= 2 * args.jitter + 24:
        raise SystemExit("crop must leave room for jitter and the ball box")

    # The directory contains generated, gitignored artifacts only. Clearing
    # stale samples makes repeated exports deterministic and prevents a smoke
    # test made with another stride from leaking into the real dataset.
    if not args.keep_existing:
        for split in ("train", "val"):
            for kind, suffix in (("images", "*.jpg"), ("labels", "*.txt")):
                folder = OUT / kind / split
                if folder.exists():
                    for artifact in folder.glob(suffix):
                        artifact.unlink()

    table = pd.read_csv(HOM)
    matrices = table[MATRIX_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    x, y, _ = ball_track(len(table))
    frames = np.flatnonzero(
        table.valid.to_numpy(bool) & np.isfinite(x) &
        (table.reprojection_error.to_numpy(float) < 2.0))[::args.stride]
    if args.limit:
        frames = frames[:args.limit]

    rng = np.random.default_rng(2027)
    cap = cv2.VideoCapture(str(VIDEO))
    written = {"train": 0, "val": 0}
    for frame in frames:
        point = matrices[frame] @ np.array([x[frame], y[frame], 1.0])
        px, py = point[:2] / point[2]
        dx, dy = rng.integers(-args.jitter, args.jitter + 1, size=2)
        cx, cy = float(px + dx), float(py + dy)
        half = args.crop / 2
        if not (half <= cx < FRAME_WIDTH - half and half <= cy < FRAME_HEIGHT - half):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok:
            continue
        verified = verified_ball_center(image, px, py, args.verify_radius)
        if verified is None:
            continue
        px, py = verified
        crop = cv2.getRectSubPix(image, (args.crop, args.crop), (cx, cy))
        split = "val" if (int(frame) // 750) % 5 == 0 else "train"
        stem = f"{int(frame):06d}"
        image_dir = OUT / "images" / split
        label_dir = OUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # A conservative weak box. Centre supervision is the important signal;
        # the local detector is later used as a point detector, not for IoU.
        bx = (px - (cx - half)) / args.crop
        by = (py - (cy - half)) / args.crop
        box = 12.0 / args.crop
        (label_dir / f"{stem}.txt").write_text(
            f"0 {bx:.6f} {by:.6f} {box:.6f} {box:.6f}\n")
        written[split] += 1
    cap.release()

    (OUT / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "names:\n  0: football\n")
    print("exported:", written, "total:", sum(written.values()))
    print("private dataset:", OUT)


if __name__ == "__main__":
    main()
