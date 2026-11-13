"""Label-free structural audit for a full-match possession submission."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import GAME, half_video_blocks
from sequence_v2 import META, RAW, extract, slot_teams
from sequence_v3 import SMART, load_events
from sequence_v4 import runs


DEFAULT = ROOT / "submission/Mes_Que_Data_XI.csv"
FPS = 25


def run_table(labels):
    starts, ends = runs(labels)
    return [(int(start), int(end), str(labels[start]))
            for start, end in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    labels = pd.read_csv(args.submission).label.to_numpy(str)
    dh, da, _, _ = extract(RAW, slot_teams(META))
    missing = ~(np.isfinite(dh) & np.isfinite(da))
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    table = run_table(labels)

    blocks = half_video_blocks(GAME)
    graded = np.zeros(len(labels), dtype=bool)
    print("half boundaries")
    for half, first, last in blocks:
        graded[first:last + 1] = True
        local = labels[first:last + 1]
        print("  half {} frames {}..{} first={} last={} transitions={}".format(
            half, first, last, local[0], local[-1],
            int(np.sum(local[1:] != local[:-1]))))
    outside = labels[~graded]
    values, counts = np.unique(outside, return_counts=True)
    print("outside-play labels:", dict(zip(values.tolist(), counts.tolist())))

    long_dead = [(s, e, (e - s) / FPS) for s, e, state in table
                 if state == "D" and np.any(graded[s:e]) and e - s >= 30 * FPS]
    print("D runs >=30s touching play:", len(long_dead))
    for row in sorted(long_dead, key=lambda item: item[2], reverse=True)[:10]:
        print("  {}..{} {:.1f}s".format(*row))

    # Dense transition bursts are a camera-cut/tracking-failure warning.
    bursts = []
    for frame in transitions:
        count = int(np.sum((transitions >= frame) & (transitions < frame + 10 * FPS)))
        if count >= 6 and (not bursts or frame > bursts[-1][1]):
            bursts.append((int(frame), int(frame + 10 * FPS), count))
    print("10s windows with >=6 transitions:", len(bursts))
    for start, end, count in bursts[:20]:
        print("  {}..{} transitions={} missing={:.1%}".format(
            start, end, count, float(np.mean(missing[start:end]))))

    missing_runs = [(s, e) for s, e, state in run_table(missing.astype(np.uint8))
                    if state == "1" and e - s >= 5 * FPS]
    print("missing-ball runs >=5s:", len(missing_runs))
    abnormal = []
    for start, end in missing_runs:
        changes = int(np.sum((transitions >= start) & (transitions < end)))
        if changes >= 3:
            abnormal.append((start, end, (end - start) / FPS, changes))
    print("  with >=3 predicted transitions:", len(abnormal))
    for row in abnormal[:20]:
        print("  {}..{} {:.1f}s transitions={}".format(*row))

    restarts = [(start, end, tags.get("Type", ""))
                for code, start, end, tags in load_events(SMART)
                if code.strip().upper() == "SET PIECES"]
    print("set-piece windows:", len(restarts))
    for start, end, kind in restarts:
        local = labels[max(0, start):min(len(labels), end)]
        dead = float(np.mean(local == "D")) if len(local) else 0.0
        changes = int(np.sum(local[1:] != local[:-1])) if len(local) > 1 else 0
        print("  {}..{} {:<24} D={:.1%} transitions={}".format(
            start, end, kind, dead, changes))


if __name__ == "__main__":
    main()
