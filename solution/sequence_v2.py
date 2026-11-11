"""Physics-aware H/A/D sequence baseline for the initial challenge.

Unlike the reference majority vote, this model keeps observation confidence and
decodes a globally coherent state path.  It is CPU-only and uses no public-GT
labels or fitted parameters.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
META = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
OUT = ROOT / "evaluation/results/mes_que_data_sequence_v2.csv"
N = 199_319
LENGTH, WIDTH = 105.0, 68.0


def slot_teams(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("ATD metadata must contain exactly two teams")
    home_id = teams[0].get("id")
    result = np.array(
        [0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
        dtype=np.int8,
    )
    if len(result) != 46:
        raise ValueError(f"expected 46 track slots, found {len(result)}")
    return result


def xyz(text: str):
    if text.startswith("N"):
        return None
    x, y, speed = text.split(",")
    return float(x), float(y), float(speed)


def extract(raw: Path, teams: np.ndarray):
    d_home = np.full(N, np.nan, dtype=np.float32)
    d_away = np.full(N, np.nan, dtype=np.float32)
    ball_speed = np.full(N, np.nan, dtype=np.float32)
    active = np.zeros(N, dtype=np.uint8)

    with raw.open() as handle:
        for line in handle:
            frame_s, players_s, ball_s = line.rstrip("\n").split(":")
            frame = int(frame_s) - 1
            players = [xyz(value) for value in players_s.split(";")]
            active[frame] = sum(value is not None for value in players)
            ball = xyz(ball_s)
            if ball is None:
                continue
            bx, by, speed = ball
            ball_speed[frame] = speed
            best = [np.inf, np.inf]
            for slot, player in enumerate(players):
                if player is None:
                    continue
                px, py, _ = player
                distance = np.hypot((px - bx) * LENGTH, (py - by) * WIDTH)
                side = int(teams[slot])
                if distance < best[side]:
                    best[side] = distance
            if np.isfinite(best[0]):
                d_home[frame] = best[0]
            if np.isfinite(best[1]):
                d_away[frame] = best[1]
    return d_home, d_away, ball_speed, active


def rolling_mean(values: np.ndarray, radius: int) -> np.ndarray:
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    lo = np.maximum(0, index - radius)
    hi = np.minimum(len(values), index + radius + 1)
    return ((total[hi] - total[lo]) / (hi - lo)).astype(np.float32)


def emissions(dh, da, speed, active):
    """Construct non-trained log evidence for H, A and D.

    Distance evidence saturates rather than letting one bad track dominate.  A
    team's evidence rises when it is both close to the ball and closer than its
    opponent.  Missing-ball evidence alone is deliberately weak: 43% of all
    frames miss the ball, many during live possession.
    """
    seen = np.isfinite(dh) & np.isfinite(da)
    n_frames = len(dh)
    h = np.zeros(n_frames, dtype=np.float32)
    a = np.zeros(n_frames, dtype=np.float32)
    d = np.zeros(n_frames, dtype=np.float32)

    h[seen] = np.exp(-np.minimum(dh[seen], 12.0) / 2.5)
    a[seen] = np.exp(-np.minimum(da[seen], 12.0) / 2.5)
    margin = np.zeros(n_frames, dtype=np.float32)
    margin[seen] = np.clip((da[seen] - dh[seen]) / 4.0, -1.0, 1.0)
    h += np.maximum(margin, 0.0)
    a += np.maximum(-margin, 0.0)

    # Aggregate roughly one second of physical evidence. This suppresses track
    # swaps while remaining much shorter than a typical possession.
    h = rolling_mean(h, 25)
    a = rolling_mean(a, 25)
    seen_rate = rolling_mean(seen.astype(np.float32), 25)

    # Strong dead-ball evidence requires a tracking-wide stoppage. A stationary
    # ball far from every player is supporting, but never sufficient by itself.
    low_activity = active < 12
    d += 3.0 * rolling_mean(low_activity.astype(np.float32), 25)
    stationary_far = seen & (np.minimum(dh, da) > 5.0) & (speed < 0.35)
    d += 0.8 * rolling_mean(stationary_far.astype(np.float32), 25)
    d += 0.35 * (1.0 - seen_rate)

    # Log-like scores with a small floor. The common scale matters less than
    # their differences because Viterbi adds transition costs below.
    return np.column_stack((np.log(h + 0.08), np.log(a + 0.08), np.log(d + 0.08)))


def viterbi(score: np.ndarray, transition_scale: float = 1.0) -> np.ndarray:
    # Switching directly between teams is expensive; entering/leaving a true
    # stoppage is slightly cheaper. Costs are per transition, not per frame.
    transition = transition_scale * np.array(
        [[0.0, -6.0, -4.0], [-6.0, 0.0, -4.0], [-4.0, -4.0, 0.0]],
        dtype=np.float32,
    )
    dp = score[0].copy()
    back = np.empty((len(score), 3), dtype=np.uint8)
    back[0] = np.arange(3)
    for frame in range(1, len(score)):
        candidates = dp[:, None] + transition
        back[frame] = np.argmax(candidates, axis=0)
        dp = candidates[back[frame], np.arange(3)] + score[frame]
        # Prevent large totals from losing float precision.
        if frame % 1000 == 0:
            dp -= dp.max()
    path = np.empty(len(score), dtype=np.uint8)
    path[-1] = np.argmax(dp)
    for frame in range(len(score) - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def write(path: Path, states: np.ndarray):
    labels = np.array(["H", "A", "D"])[states]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    teams = slot_teams(args.meta)
    dh, da, speed, active = extract(args.raw, teams)
    states = viterbi(emissions(dh, da, speed, active))
    labels = write(args.out, states)
    values, counts = np.unique(labels, return_counts=True)
    transitions = int(np.sum(labels[1:] != labels[:-1]))
    print("slot teams:", {"H": int(np.sum(teams == 0)), "A": int(np.sum(teams == 1))})
    # ATD occasionally reports a valid ball position with speed=NaN. Presence
    # must therefore be measured from position-derived distances, not speed.
    print("ball seen: {:.1%}".format(np.mean(np.isfinite(dh) | np.isfinite(da))))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("transitions:", transitions)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
