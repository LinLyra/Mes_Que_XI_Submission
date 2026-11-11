"""Externally calibrated control-anchor model.

The anchor weight (1.7) is selected on Metrica Sample Game 3, not on the
competition's public five-minute labels. It is the middle of the external
accuracy plateau, avoiding the fragmentation of the largest tested weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sequence_v2 import META, RAW, emissions, extract, slot_teams, viterbi, write
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors, duration_cleanup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/results/mes_que_data_sequence_v6.csv"
ANCHOR_WEIGHT = 1.7


def decode_v6(dh, da, speed, active, events):
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(
        score, dh, da, weight=ANCHOR_WEIGHT)
    # Selected on external Game 3: this removes 4.7% of transitions without
    # changing accuracy at 2,830 professional control-event anchors.
    states = duration_cleanup(
        viterbi(score), score, minimum=np.array([25, 25, 40]))
    return states, home_anchor, away_anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--smart", type=Path, default=SMART)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    dh, da, speed, active = extract(args.raw, slot_teams(args.meta))
    states, home_anchor, away_anchor = decode_v6(
        dh, da, speed, active, load_events(args.smart))
    labels = write(args.out, states)
    values, counts = np.unique(labels, return_counts=True)
    print("external anchor weight:", ANCHOR_WEIGHT)
    print("control anchors: H={}, A={}".format(
        int(np.sum(home_anchor >= 0.25)), int(np.sum(away_anchor >= 0.25))))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
