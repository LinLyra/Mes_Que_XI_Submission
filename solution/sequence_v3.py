"""Sequence v2 plus weak, reliability-aware Smart Tagging priors.

Smart Tagging is automatic and overlapping, so it is never copied into the
submission. Phase tags contribute only a small team prior; set-piece windows
contribute a restart-shaped prior (dead before the centre, live afterwards).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sequence_v2 import META, N, RAW, emissions, extract, slot_teams, viterbi, write

ROOT = Path(__file__).resolve().parents[1]
SMART = ROOT / "data/tactical/BAR-ATH_tactical_smart_tagging.xml"
V3_OUT = ROOT / "evaluation/results/mes_que_data_sequence_v3.csv"


def load_events(path: Path, fps: int = 25):
    import xml.etree.ElementTree as ET

    events = []
    for instance in ET.parse(path).iter("instance"):
        start = int(round(float(instance.findtext("start")) * fps))
        end = int(round(float(instance.findtext("end")) * fps))
        tags = {
            label.findtext("group"): label.findtext("text")
            for label in instance.findall("label")
        }
        events.append((instance.findtext("code") or "", start, end, tags))
    return events


def add_event_priors(score: np.ndarray, events):
    phase_h = np.zeros(N, dtype=np.float32)
    phase_a = np.zeros(N, dtype=np.float32)
    phase_n = np.zeros(N, dtype=np.float32)
    restart_d = np.zeros(N, dtype=np.float32)
    restart_h = np.zeros(N, dtype=np.float32)
    restart_a = np.zeros(N, dtype=np.float32)

    for code, start, end, tags in events:
        start, end = max(0, start), min(N, end)
        if end <= start:
            continue
        team = tags.get("Team")
        side = 0 if team == "Barcelona" else 1 if team == "Atletic Club" else None
        is_restart = code.strip().upper() == "SET PIECES"

        if is_restart:
            # The automatic tag is a window around the restart, not its precise
            # instant. A soft two-sided shape preserves that uncertainty.
            middle = (start + end) // 2
            restart_d[start:middle] += 1.0
            if side == 0:
                restart_h[middle:end] += 1.0
            elif side == 1:
                restart_a[middle:end] += 1.0
        elif side is not None:
            phase_n[start:end] += 1.0
            if side == 0:
                phase_h[start:end] += 1.0
            else:
                phase_a[start:end] += 1.0

    covered = phase_n > 0
    # Overlapping phase tags are averaged, preventing heavily tagged attacking
    # sequences from overpowering the physical tracking evidence.
    score[covered, 0] += 0.22 * phase_h[covered] / phase_n[covered]
    score[covered, 1] += 0.22 * phase_a[covered] / phase_n[covered]
    score[:, 2] += 0.45 * np.minimum(restart_d, 1.0)
    score[:, 0] += 0.30 * np.minimum(restart_h, 1.0)
    score[:, 1] += 0.30 * np.minimum(restart_a, 1.0)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--smart", type=Path, default=SMART)
    parser.add_argument("--out", type=Path, default=V3_OUT)
    args = parser.parse_args()

    teams = slot_teams(args.meta)
    dh, da, speed, active = extract(args.raw, teams)
    events = load_events(args.smart)
    score = add_event_priors(emissions(dh, da, speed, active), events)
    labels = write(args.out, viterbi(score))
    values, counts = np.unique(labels, return_counts=True)
    print("events:", len(events))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
