"""Render trajectory-valid player-search detections for visual QA."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/tactical/BAR-ATH_tactical.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.tracks)
    table = table[table.trajectory_valid].sort_values(["track_id", "frame"])
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for row in table.itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame))
        ok, image = cap.read()
        if not ok:
            continue
        centre = (float(row.pixel_x), float(row.pixel_y))
        tile = cv2.getRectSubPix(image, (320, 240), centre)
        cv2.circle(tile, (160, 120), 18, (0, 0, 255), 3)
        cv2.putText(tile, "track {} frame {} conf {:.2f}".format(
            row.track_id, row.frame, row.confidence), (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, "track {} frame {} conf {:.2f}".format(
            row.track_id, row.frame, row.confidence), (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    if not tiles:
        raise RuntimeError("no trajectory-valid detections")
    columns = 4
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)]
    rows[-1] += [blank] * (columns - len(rows[-1]))
    contact = np.vstack([np.hstack(row) for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), contact):
        raise IOError("could not write {}".format(args.out))
    print("rendered", len(tiles), "detections to", args.out)


if __name__ == "__main__":
    main()
