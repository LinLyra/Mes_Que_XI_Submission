"""Restart-aware dead-ball boundaries built without possession labels.

Smart Tagging provides a window around a restart. This model estimates the kick
inside that window from ball-motion onset, then applies asymmetric evidence:
dead immediately before the kick and the tagged team immediately afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sequence_v2 import META, N, RAW, emissions, extract, rolling_mean, slot_teams, viterbi, write
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors, duration_cleanup


ROOT = Path(__file__).resolve().parents[1]
V5_OUT = ROOT / "evaluation/results/mes_que_data_sequence_v5.csv"


def raw_ball_motion(path: Path):
    speed = np.full(N, np.nan, dtype=np.float32)
    x = np.full(N, np.nan, dtype=np.float32)
    y = np.full(N, np.nan, dtype=np.float32)
    with path.open() as handle:
        for line in handle:
            frame_s, _, ball_s = line.rstrip("\n").split(":")
            if ball_s.startswith("N"):
                continue
            bx, by, bs = ball_s.split(",")
            frame = int(frame_s) - 1
            x[frame], y[frame] = float(bx), float(by)
            try:
                speed[frame] = float(bs)
            except ValueError:
                pass
    return x, y, speed


def motion_rise(speed: np.ndarray, anchor: np.ndarray):
    motion = np.nan_to_num(speed, nan=0.0)
    before = rolling_mean(motion, 5)
    after = np.roll(before, -10)
    return after - before + 0.6 * anchor


def learn_type_offsets(events, rise):
    """Self-supervise a shared temporal anchor for each restart type."""
    grouped = {}
    for code, start, end, tags in events:
        if code.strip().upper() == "SET PIECES":
            grouped.setdefault(tags.get("Type") or "Unknown", []).append((start, end))

    offsets = {}
    for event_type, intervals in grouped.items():
        # Search only positions supported by at least half the examples. Median
        # aggregation rejects unrelated passes inside long 35-second windows.
        candidates = []
        for offset in range(-400, 401, 5):
            values = []
            for start, end in intervals:
                frame = (start + end) // 2 + offset
                if start + 10 <= frame < end - 10:
                    values.append(rise[frame])
            if len(values) >= max(2, len(intervals) // 2):
                candidates.append((float(np.median(values)), offset))
        offsets[event_type] = max(candidates)[1] if candidates else 0
    return offsets


def motion_onset(start: int, end: int, expected: int, rise: np.ndarray):
    start, end = max(0, start), min(len(rise), end)
    # Once the type-level alignment is known, refine only within +/-1.5 s.
    lo = max(start, expected - 38)
    hi = min(end - 10, expected + 39)
    if hi <= lo:
        return min(max(expected, start), max(start, end - 1))
    return lo + int(np.argmax(rise[lo:hi]))


def refined_restart_evidence(score, events, speed, anchor):
    # Temporal alignment must remain independent of player-distance tracking;
    # otherwise masking positional observations moves the inferred kick itself.
    # Raw ball speed is a separate ATD channel and gives a stable clock anchor.
    rise = motion_rise(speed, np.zeros_like(anchor))
    offsets = learn_type_offsets(events, rise)
    detected = []
    for code, start, end, tags in events:
        if code.strip().upper() != "SET PIECES":
            continue
        event_type = tags.get("Type") or "Unknown"
        expected = (start + end) // 2 + offsets.get(event_type, 0)
        kick = motion_onset(start, end, expected, rise)
        detected.append((start, end, kick, tags.get("Type")))

        # The tag begins shortly before the restart. Its pre-kick side is much
        # safer dead-ball evidence than treating a missing ball anywhere as D.
        score[max(0, start):kick, 2] += 0.75
        live_end = min(N, kick + 25)
        if tags.get("Team") == "Barcelona":
            score[kick:live_end, 0] += 0.45
        elif tags.get("Team") == "Atletic Club":
            score[kick:live_end, 1] += 0.45
    return score, detected, offsets


def decode_v5(dh, da, speed, active, events, raw_speed):
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(score, dh, da)
    anchor = np.maximum(home_anchor, away_anchor)
    score, detected, offsets = refined_restart_evidence(score, events, raw_speed, anchor)
    return duration_cleanup(viterbi(score), score), detected, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--smart", type=Path, default=SMART)
    parser.add_argument("--out", type=Path, default=V5_OUT)
    args = parser.parse_args()

    dh, da, speed, active = extract(args.raw, slot_teams(args.meta))
    _, _, raw_speed = raw_ball_motion(args.raw)
    states, detected, offsets = decode_v5(
        dh, da, speed, active, load_events(args.smart), raw_speed)
    labels = write(args.out, states)
    observed_offsets = [
        (kick - (start + end) / 2) / 25 for start, end, kick, _ in detected]
    values, counts = np.unique(labels, return_counts=True)
    print("restart detections:", len(detected))
    print("self-supervised type offsets:",
          {key: round(value / 25, 2) for key, value in sorted(offsets.items())})
    print("median offset from tag centre: {:.2f}s".format(
        float(np.median(observed_offsets))))
    print("10-90% offset: {:.2f}s .. {:.2f}s".format(
        float(np.percentile(observed_offsets, 10)),
        float(np.percentile(observed_offsets, 90))))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
