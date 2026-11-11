"""Recover short ATD ball gaps from tactical video without manual labels.

Only gaps bounded by two observed ball positions are considered.  The two
endpoints are projected into the image with the supplied homography, tracked
in both temporal directions, and accepted only when both tracks recover the
opposite endpoint within a strict pixel gate.  Accepted image paths are mapped
back to pitch coordinates and used to recompute player-to-ball distances.

This is deliberately a conservative experiment: long gaps are untouched and
the official/public possession labels are never read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT))

from starter.load_homography import MATRIX_COLUMNS
from sequence_v2 import LENGTH, WIDTH, META, RAW, extract, slot_teams, write, xyz
from sequence_v3 import SMART, load_events
from sequence_v7 import decode_v7
from visual_flow_validation import track_window

VIDEO = ROOT / "data/tactical/BAR-ATH_tactical.mp4"
HOM = ROOT / "data/tactical/BAR-ATH_tactical_homography.csv"
OUT = ROOT / "evaluation/results/mes_que_data_sequence_v10.csv"
DIAGNOSTICS = ROOT / "evaluation/diagnostics/visual_ball_flow.csv"


def project(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = matrix @ np.r_[point, 1.0]
    return value[:2] / value[2]


def unproject(pixel: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.linalg.solve(matrix, np.r_[pixel, 1.0])
    return value[:2] / value[2]


def read_ball(raw: Path, n: int) -> np.ndarray:
    result = np.full((n, 2), np.nan, dtype=np.float32)
    with raw.open() as handle:
        for line in handle:
            frame_s, _, ball_s = line.rstrip().split(":")
            ball = xyz(ball_s)
            if ball is not None:
                result[int(frame_s) - 1] = ball[:2]
    return result


def short_bounded_gaps(known: np.ndarray, maximum: int):
    changes = np.diff(np.r_[True, known, True].astype(np.int8))
    starts = np.flatnonzero(changes == -1)
    ends = np.flatnonzero(changes == 1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)
            if 0 < s and e < len(known) and e - s <= maximum]


def recover(video: Path, homography: np.ndarray, ball_xy: np.ndarray,
            maximum: int, gate: float):
    known = np.isfinite(ball_xy[:, 0])
    gaps = short_bounded_gaps(known, maximum)
    recovered = {}
    records = []
    cap = cv2.VideoCapture(str(video))
    for start, end in gaps:
        first, last = start - 1, end
        p0 = project(ball_xy[first], homography[first])
        p1 = project(ball_xy[last], homography[last])
        tracked = track_window(cap, first, last - first + 1, p0, p1)
        if tracked is None:
            records.append((start, end, end - start, np.nan, False, 0))
            continue
        forward, backward, blended = tracked
        endpoint_error = max(float(np.linalg.norm(forward[-1] - p1)),
                             float(np.linalg.norm(backward[0] - p0)))
        accepted = endpoint_error <= gate
        count = 0
        if accepted:
            for offset, frame in enumerate(range(start, end), start=1):
                xy = unproject(blended[offset], homography[frame])
                if np.isfinite(xy).all() and np.all(xy >= -0.03) and np.all(xy <= 1.03):
                    recovered[frame] = xy.astype(np.float32)
                    count += 1
        records.append((start, end, end - start, endpoint_error, accepted, count))
    cap.release()
    return recovered, pd.DataFrame(records, columns=[
        "start", "end", "gap_frames", "endpoint_error_px", "accepted", "recovered_frames"])


def fill_distances(raw: Path, teams: np.ndarray, recovered: dict[int, np.ndarray],
                   dh: np.ndarray, da: np.ndarray, speed: np.ndarray):
    wanted = set(recovered)
    previous = {}
    with raw.open() as handle:
        for line in handle:
            frame_s, players_s, _ = line.rstrip().split(":")
            frame = int(frame_s) - 1
            if frame not in wanted:
                continue
            bx, by = recovered[frame]
            best = [np.inf, np.inf]
            for slot, text in enumerate(players_s.split(";")):
                player = xyz(text)
                if player is None:
                    continue
                px, py, _ = player
                distance = np.hypot((px - bx) * LENGTH, (py - by) * WIDTH)
                side = int(teams[slot])
                best[side] = min(best[side], distance)
            if np.isfinite(best[0]):
                dh[frame] = best[0]
            if np.isfinite(best[1]):
                da[frame] = best[1]
            previous[frame] = (bx, by)
    # Speed is optional evidence; estimate it only inside contiguous recovered paths.
    for frame, point in previous.items():
        if frame - 1 in previous:
            prior = previous[frame - 1]
            speed[frame] = 25.0 * np.hypot(
                (point[0] - prior[0]) * LENGTH, (point[1] - prior[1]) * WIDTH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gap", type=int, default=25)
    parser.add_argument("--gate", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not VIDEO.exists():
        raise SystemExit(f"missing tactical video: {VIDEO}")

    table = pd.read_csv(HOM)
    matrices = table[MATRIX_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    ball = read_ball(RAW, len(matrices))
    recovered, diagnostics = recover(VIDEO, matrices, ball, args.max_gap, args.gate)
    DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(DIAGNOSTICS, index=False)

    teams = slot_teams(META)
    dh, da, speed, active = extract(RAW, teams)
    baseline, _, _ = decode_v7(dh.copy(), da.copy(), speed.copy(), active, load_events(SMART))
    fill_distances(RAW, teams, recovered, dh, da, speed)
    states, _, _ = decode_v7(dh, da, speed, active, load_events(SMART))
    labels = write(args.out, states)

    accepted = int(diagnostics["accepted"].sum())
    attempted = len(diagnostics)
    changed = int(np.sum(states != baseline))
    print(f"gaps attempted/accepted: {attempted}/{accepted}")
    print(f"frames recovered: {len(recovered)}")
    print(f"v10 frames changed from v7: {changed}")
    print("v10 transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)
    print("diagnostics:", DIAGNOSTICS)


if __name__ == "__main__":
    main()
