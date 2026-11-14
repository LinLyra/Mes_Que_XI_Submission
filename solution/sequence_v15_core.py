"""Reproducible v15 core using only supplied ATD/tags and external calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sequence_v2 import META, RAW, emissions, extract, slot_teams, write
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors
from sequence_v6 import ANCHOR_WEIGHT
from sequence_v7 import decode_v7
from sequence_v15 import anchor_protected_cleanup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/results/mes_que_data_anchor_spell_v15_core.csv"


def decode_v15_core():
    teams = slot_teams(META)
    dh, da, speed, active = extract(RAW, teams)
    events = load_events(SMART)
    base, _, _ = decode_v7(dh, da, speed, active, events)
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(
        score, dh, da, weight=ANCHOR_WEIGHT)
    return anchor_protected_cleanup(
        base, score, home_anchor, away_anchor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    labels = write(args.out, decode_v15_core())
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
