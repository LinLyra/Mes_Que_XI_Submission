"""Control anchors and duration-aware decoding on top of sequence v3.

The decoder distinguishes proximity from established control. A player must be
close to the ball *and* clearly closer than the opponent to create an anchor.
Very short decoded runs are then resolved from their accumulated evidence,
which is a lightweight semi-Markov duration constraint rather than a blind
median filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from sequence_v2 import META, RAW, emissions, extract, rolling_mean, slot_teams, viterbi, write
from sequence_v3 import SMART, add_event_priors, load_events


ROOT = Path(__file__).resolve().parents[1]
V4_OUT = ROOT / "evaluation/results/mes_que_data_sequence_v4.csv"


def add_control_anchors(
    score: np.ndarray, dh: np.ndarray, da: np.ndarray, weight: float = 0.9
):
    seen = np.isfinite(dh) & np.isfinite(da)
    nearest = np.minimum(dh, da)
    separation = np.abs(dh - da)

    # A 2.5 m capture radius is deliberately generous; the separation term
    # prevents two-player duels from becoming anchors. Persistence over eleven
    # frames rejects one-frame tracking jumps.
    quality = np.zeros(len(score), dtype=np.float32)
    quality[seen] = (
        np.clip((2.5 - nearest[seen]) / 2.0, 0.0, 1.0)
        * np.clip(separation[seen] / 2.0, 0.0, 1.0)
    )
    home = rolling_mean(quality * (dh < da), 5)
    away = rolling_mean(quality * (da < dh), 5)
    score[:, 0] += weight * home
    score[:, 1] += weight * away
    return score, home, away


def runs(states: np.ndarray):
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    ends = np.r_[starts[1:], len(states)]
    return starts, ends


def duration_cleanup(
    states: np.ndarray, score: np.ndarray, minimum: Optional[np.ndarray] = None
):
    """Resolve implausibly short runs using segment-level evidence.

    H/A require 0.4 s and D requires 0.8 s. These are definition-level
    persistence requirements, fixed without reference to public labels.
    """
    out = states.copy()
    if minimum is None:
        minimum = np.array([10, 10, 20])
    for _ in range(4):
        changed = False
        starts, ends = runs(out)
        for index, (start, end) in enumerate(zip(starts, ends)):
            state = int(out[start])
            if end - start >= minimum[state]:
                continue
            left = int(out[start - 1]) if start > 0 else None
            right = int(out[end]) if end < len(out) else None
            if left is None and right is None:
                continue
            if left == right and left is not None:
                replacement = left
            else:
                candidates = [candidate for candidate in (left, right) if candidate is not None]
                evidence = [float(score[start:end, candidate].sum()) for candidate in candidates]
                replacement = candidates[int(np.argmax(evidence))]
            out[start:end] = replacement
            changed = True
        if not changed:
            break
    return out


def decode_v4(dh, da, speed, active, events):
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(score, dh, da)
    states = duration_cleanup(viterbi(score), score)
    return states, home_anchor, away_anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--smart", type=Path, default=SMART)
    parser.add_argument("--out", type=Path, default=V4_OUT)
    args = parser.parse_args()

    dh, da, speed, active = extract(args.raw, slot_teams(args.meta))
    events = load_events(args.smart)
    states, home_anchor, away_anchor = decode_v4(dh, da, speed, active, events)
    labels = write(args.out, states)
    values, counts = np.unique(labels, return_counts=True)
    starts, _ = runs(states)
    print("control anchors: H={}, A={}".format(
        int(np.sum(home_anchor >= 0.25)), int(np.sum(away_anchor >= 0.25))))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("segments:", len(starts))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
