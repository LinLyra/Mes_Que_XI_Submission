"""Train the local-ball detector produced by visual_detector_dataset.py.

Ultralytics is imported lazily so the CPU-only possession pipeline keeps its
small dependency set. Training outputs are private and gitignored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation/diagnostics/ball_detector_filtered/dataset.yaml"
RUNS = ROOT / "evaluation/diagnostics/ball_detector_filtered/runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install the private vision environment first: "
            "pip install -r solution/requirements-vision.txt") from exc

    model = YOLO(args.model)
    result = model.train(
        data=str(DATA), imgsz=160, epochs=args.epochs, batch=args.batch,
        device=args.device, project=str(RUNS), name="local_ball", exist_ok=True,
        seed=2027, deterministic=True, patience=20, cache="disk", workers=4,
        degrees=0.0, translate=0.08, scale=0.15, fliplr=0.5,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
    )
    print("training output:", result.save_dir)


if __name__ == "__main__":
    main()
