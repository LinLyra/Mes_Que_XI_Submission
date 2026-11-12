"""Cross-filter weak ATD labels with an independent pretrained ball detector.

Only samples where a COCO ``sports ball`` prediction agrees spatially with the
ATD/homography/image-component weak label are retained. The detector box then
replaces the fixed weak box. Outputs are private generated artifacts.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/diagnostics/ball_detector"
OUTPUT = ROOT / "evaluation/diagnostics/ball_detector_filtered"
SPORTS_BALL_CLASS = 32


def weak_center(label: Path, size: int) -> np.ndarray:
    values = list(map(float, label.read_text().split()))
    return np.asarray(values[1:3]) * size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--agreement", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    from ultralytics import YOLO

    for split in ("train", "val"):
        for kind, suffix in (("images", "*.jpg"), ("labels", "*.txt")):
            folder = args.output / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob(suffix):
                old.unlink()

    model = YOLO(args.model)
    audit = []
    totals = {}
    for split in ("train", "val"):
        images = sorted((args.source / "images" / split).glob("*.jpg"))
        kept = 0
        results = model.predict(
            [str(path) for path in images], imgsz=args.image_size,
            conf=args.confidence, classes=[SPORTS_BALL_CLASS],
            batch=args.batch, verbose=False)
        for result in results:
            image = Path(result.path)
            weak_label = args.source / "labels" / split / f"{image.stem}.txt"
            if result.boxes is None or len(result.boxes) == 0:
                audit.append((split, image.name, "no_detection", "", ""))
                continue
            target = weak_center(weak_label, args.crop_size)
            boxes = result.boxes.xywh.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            distances = np.linalg.norm(boxes[:, :2] - target, axis=1)
            index = int(np.argmin(distances))
            distance = float(distances[index])
            if distance > args.agreement:
                audit.append((split, image.name, "disagree", distance,
                              float(confidence[index])))
                continue
            x, y, width, height = boxes[index]
            width = float(np.clip(width, 4.0, 40.0))
            height = float(np.clip(height, 4.0, 40.0))
            shutil.copy2(image, args.output / "images" / split / image.name)
            label = args.output / "labels" / split / f"{image.stem}.txt"
            label.write_text(
                f"0 {x/args.crop_size:.6f} {y/args.crop_size:.6f} "
                f"{width/args.crop_size:.6f} {height/args.crop_size:.6f}\n")
            kept += 1
            audit.append((split, image.name, "kept", distance,
                          float(confidence[index])))
        totals[split] = (kept, len(images))

    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: football\n")
    with (args.output / "audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "decision", "distance_px", "confidence"])
        writer.writerows(audit)
    print("cross-filtered:", totals)
    print("private dataset:", args.output)


if __name__ == "__main__":
    main()
