"""Externally calibrated sequence model with improved hand-over sensitivity.

All v6 evidence layers are retained. The Viterbi transition scale is reduced
to 0.75 because the same direction improves control-event accuracy and switch
coverage independently on Metrica Sample Games 1, 2 and 3. No competition
ground-truth labels are used for this selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sequence_v2 import META, RAW, emissions, extract, slot_teams, viterbi, write
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors, duration_cleanup
from sequence_v6 import ANCHOR_WEIGHT


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/results/mes_que_data_sequence_v7.csv"
TRANSITION_SCALE = 0.75


def decode_v7(dh, da, speed, active, events):
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(
        score, dh, da, weight=ANCHOR_WEIGHT)
    states = duration_cleanup(
        viterbi(score, transition_scale=TRANSITION_SCALE),
        score,
        minimum=np.array([25, 25, 40]),
    )
    return states, home_anchor, away_anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--smart", type=Path, default=SMART)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    dh, da, speed, active = extract(args.raw, slot_teams(args.meta))
    states, home_anchor, away_anchor = decode_v7(
        dh, da, speed, active, load_events(args.smart))
    labels = write(args.out, states)
    values, counts = np.unique(labels, return_counts=True)
    print("external transition scale:", TRANSITION_SCALE)
    print("control anchors: H={}, A={}".format(
        int(np.sum(home_anchor >= 0.25)), int(np.sum(away_anchor >= 0.25))))
    print("labels:", dict(zip(values.tolist(), counts.tolist())))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
