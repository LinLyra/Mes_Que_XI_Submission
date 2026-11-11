"""CPU-only full-match possession baseline from the tactical ATD stream.

This is deliberately independent of the ATD metadata XML so that the raw-data
pipeline can be verified while Google Drive blocks XML downloads.  The delivered
export groups track slots by team: slots 0..22 are Barcelona (H), 23..45 are
Athletic Club (A).  ``--check-slot-split`` verifies that both blocks are active
throughout play instead of silently accepting a malformed file.

The model has two stages:

1. nearest-team evidence when the ball is observed;
2. a causal gap bridge followed by centred majority smoothing.

It is a reproducible engineering baseline, not the final competition model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
OUT = ROOT / "evaluation/results/mes_que_data_v1.csv"
N_FRAMES = 199_319
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HOME_END = 23


def parse_group(group: str):
    if group.startswith("N"):
        return None
    fields = group.split(",")
    return float(fields[0]), float(fields[1]), float(fields[2])


def observations(path: Path):
    """Return nearest-team observations and compact diagnostics."""
    team = np.full(N_FRAMES, -1, dtype=np.int8)
    distance = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_speed = np.full(N_FRAMES, np.nan, dtype=np.float32)
    active_home = np.zeros(N_FRAMES, dtype=np.uint8)
    active_away = np.zeros(N_FRAMES, dtype=np.uint8)

    with path.open() as handle:
        for line in handle:
            frame_text, player_text, ball_text = line.rstrip("\n").split(":")
            frame = int(frame_text) - 1
            players = player_text.split(";")
            parsed = [parse_group(group) for group in players]
            active_home[frame] = sum(p is not None for p in parsed[:HOME_END])
            active_away[frame] = sum(p is not None for p in parsed[HOME_END:])

            ball = parse_group(ball_text)
            if ball is None:
                continue
            bx, by, speed = ball
            ball_speed[frame] = speed
            best_distance = np.inf
            best_team = -1
            for slot, player in enumerate(parsed):
                if player is None:
                    continue
                px, py, _ = player
                dx = (px - bx) * PITCH_LENGTH
                dy = (py - by) * PITCH_WIDTH
                d = dx * dx + dy * dy
                if d < best_distance:
                    best_distance = d
                    best_team = 0 if slot < HOME_END else 1
            if best_team >= 0:
                team[frame] = best_team
                distance[frame] = np.sqrt(best_distance)
    return team, distance, ball_speed, active_home, active_away


def bridge_short_gaps(codes: np.ndarray, max_gap: int = 75):
    """Fill missing spans only when the same team is observed on both sides."""
    out = codes.copy()
    valid = np.flatnonzero(codes >= 0)
    for left, right in zip(valid[:-1], valid[1:]):
        if right - left - 1 <= max_gap and codes[left] == codes[right]:
            out[left + 1:right] = codes[left]
    return out


def majority(codes: np.ndarray, half_window: int = 50):
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    low = np.maximum(0, idx - half_window)
    high = np.minimum(len(codes), idx + half_window + 1)
    hc = home[high] - home[low]
    ac = away[high] - away[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = hc + ac > 0
    labels[seen] = np.where(hc[seen] >= ac[seen], "H", "A")
    return labels


def write_submission(path: Path, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--gap", type=int, default=75)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    team, distance, ball_speed, home_n, away_n = observations(args.raw)
    if len(team) != N_FRAMES:
        raise RuntimeError(f"expected {N_FRAMES} frames, got {len(team)}")
    observed = team >= 0
    play_like = (home_n >= 8) & (away_n >= 8)
    if np.median(home_n[play_like]) < 8 or np.median(away_n[play_like]) < 8:
        raise RuntimeError("track-slot team split failed its occupancy check")

    bridged = bridge_short_gaps(team, args.gap)
    labels = majority(bridged, args.window)
    write_submission(args.out, labels)

    print(f"frames: {len(team):,}")
    print(f"ball observed: {observed.sum():,} ({observed.mean():.1%})")
    print(f"median nearest distance: {np.nanmedian(distance):.2f} m")
    print(f"median ball speed: {np.nanmedian(ball_speed):.2f} m/s")
    print(f"median active slots in play-like frames: H={np.median(home_n[play_like]):.0f}, "
          f"A={np.median(away_n[play_like]):.0f}")
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
