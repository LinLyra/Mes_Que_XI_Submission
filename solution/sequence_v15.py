"""Externally calibrated anchor-protected spell regularisation.

V13 fixes visually supported boundaries but still retains short team spells
created by noisy tracking.  This decoder removes a short spell only when it
lacks a persistent physical control anchor.  The three constants below were
selected on Metrica Sample Games 1 and 2, not competition ground truth: they
retain 89.78% control-event accuracy (v7: 89.99%) while reducing transitions
from 992 to 881 across the two external matches.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_v2 import META, RAW, emissions, extract, slot_teams, write
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors, runs
from sequence_v6 import ANCHOR_WEIGHT
from sequence_v7 import decode_v7
from sequence_v12 import HOM, recovered_pitch
from sequence_v13 import constrained_fusion
from starter.load_homography import MATRIX_COLUMNS
from visual_ball_flow import fill_distances


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/results/mes_que_data_anchor_spell_v15.csv"
TRACKS = ROOT / "evaluation/diagnostics/player_search_top50.csv"
MAX_SHORT_TEAM_RUN = 125
MIN_ANCHOR_PEAK = 0.20
MIN_ANCHOR_FRAMES = 5


def anchor_protected_cleanup(states: np.ndarray, score: np.ndarray,
                             home_anchor: np.ndarray, away_anchor: np.ndarray,
                             max_short: int = MAX_SHORT_TEAM_RUN,
                             min_peak: float = MIN_ANCHOR_PEAK,
                             min_frames: int = MIN_ANCHOR_FRAMES) -> np.ndarray:
    """Remove weak short H/A spells while preserving physical control.

    D spells are untouched. A short team run survives when its anchor reaches
    ``min_peak`` and remains above 0.25 for ``min_frames``. Otherwise it is
    assigned to equal flanks, or to the flank with larger accumulated model
    evidence. This can remove an island but cannot invent a new one.
    """
    out = states.copy()
    anchors = (home_anchor, away_anchor)
    for _ in range(4):
        changed = False
        starts, ends = runs(out)
        for start, end in zip(starts, ends):
            state = int(out[start])
            if state == 2 or end - start >= max_short:
                continue
            anchor = anchors[state][start:end]
            protected = (float(np.max(anchor, initial=0.0)) >= min_peak and
                         int(np.sum(anchor >= 0.25)) >= min_frames)
            if protected:
                continue
            left = int(out[start - 1]) if start else None
            right = int(out[end]) if end < len(out) else None
            if left is None and right is None:
                continue
            if left == right:
                replacement = left
            else:
                candidates = [value for value in (left, right) if value is not None]
                replacement = max(
                    candidates, key=lambda value: float(score[start:end, value].sum()))
            out[start:end] = replacement
            changed = True
        if not changed:
            break
    return out


def decode_v15(tracks_path: Path = TRACKS):
    teams = slot_teams(META)
    dh, da, speed, active = extract(RAW, teams)
    events = load_events(SMART)

    # The unmodified evidence supplies both the v7 base and the physical
    # anchors used to protect genuine short spells.
    base, _, _ = decode_v7(dh, da, speed, active, events)
    score = add_event_priors(emissions(dh, da, speed, active), events)
    score, home_anchor, away_anchor = add_control_anchors(
        score, dh, da, weight=ANCHOR_WEIGHT)

    # Decode the strictly gated visual evidence separately, then fuse it at
    # spell level so sparse trajectories cannot create possession islands.
    homography = pd.read_csv(HOM)[MATRIX_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    tracks = pd.read_csv(tracks_path)
    recovered = recovered_pitch(tracks, homography)
    visual_dh, visual_da = dh.copy(), da.copy()
    visual_speed = speed.copy()
    fill_distances(RAW, teams, recovered, visual_dh, visual_da, visual_speed)
    visual, _, _ = decode_v7(visual_dh, visual_da, visual_speed, active, events)

    alphabet = np.asarray(["H", "A", "D"])
    fused_labels, audit = constrained_fusion(
        alphabet[base], alphabet[visual], tracks)
    fused = np.asarray([{"H": 0, "A": 1, "D": 2}[label]
                        for label in fused_labels], dtype=np.uint8)
    final = anchor_protected_cleanup(
        fused, score, home_anchor, away_anchor)
    return final, audit, recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=TRACKS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    states, audit, recovered = decode_v15(args.tracks)
    labels = write(args.out, states)
    print(audit.to_string(index=False))
    print("recovered visual frames:", len(recovered))
    print("anchor rule: max_short={}, min_peak={}, min_frames={}".format(
        MAX_SHORT_TEAM_RUN, MIN_ANCHOR_PEAK, MIN_ANCHOR_FRAMES))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
