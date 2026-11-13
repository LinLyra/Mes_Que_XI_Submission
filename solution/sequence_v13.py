"""Spell-constrained fusion of v7 with validated visual ball trajectories.

v12 injected recovered ball positions before decoding.  That fixed supported
frames, but could create short possession islands and therefore hurt spellF1.
This version treats a visual trajectory as boundary/segment evidence: it may
extend an existing team spell through dead-ball uncertainty, but it may not
create an isolated team spell.  The rule is label-free and deliberately
conservative.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_v12 import HOM, OUT as V12_OUT, ROOT, TRACKS


V7_OUT = ROOT / "evaluation/results/mes_que_data_sequence_v7.csv"
OUT = ROOT / "evaluation/results/mes_que_data_visual_spell_v13.csv"
TEAM_LABELS = {"H", "A"}


def runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Return half-open constant-label runs."""
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(labels)]
    return [(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends)]


def containing_run(labels: np.ndarray, frame: int) -> tuple[int, int, str]:
    start = frame
    while start and labels[start - 1] == labels[frame]:
        start -= 1
    end = frame + 1
    while end < len(labels) and labels[end] == labels[frame]:
        end += 1
    return start, end, str(labels[frame])


def constrained_fusion(base: np.ndarray, visual: np.ndarray, tracks: pd.DataFrame,
                       max_bridge: int = 250, max_opposite_edge: int = 25):
    """Extend anchored spells to visual evidence without adding an island.

    Each validated trajectory votes using the visual decoder at its sampled
    frames.  A vote is accepted only if the same team already owns a nearby
    base spell.  Between that anchor and the evidence we permit D, plus at
    most a tiny opposite-team suffix/prefix (a boundary shift).  Processing
    in time order lets several trajectories bridge one uncertain D interval.
    """
    out = base.copy()
    audit = []
    valid = tracks[tracks.trajectory_valid].sort_values(["frame", "track_id"])
    for track_id, group in valid.groupby("track_id", sort=False):
        frames = group.frame.to_numpy(int)
        votes = visual[frames]
        team_votes = votes[np.isin(votes, list(TEAM_LABELS))]
        if not len(team_votes):
            continue
        team, count = np.unique(team_votes, return_counts=True)
        team = str(team[np.argmax(count)])
        if np.max(count) * 2 <= len(team_votes):
            continue

        # Include the complete visual team run containing each sampled point.
        evidence_start, evidence_end = int(frames.min()), int(frames.max()) + 1
        for frame in frames[visual[frames] == team]:
            s, e, label = containing_run(visual, int(frame))
            if label == team:
                evidence_start = min(evidence_start, s)
                evidence_end = max(evidence_end, e)

        # If the evidence touches an existing team spell, it is a pure boundary
        # move.  Otherwise find the closest same-team anchor on either side.
        team_idx = np.flatnonzero(out == team)
        if not len(team_idx):
            continue
        before = team_idx[team_idx < evidence_start]
        after = team_idx[team_idx >= evidence_end]
        choices = []
        if len(before):
            choices.append((evidence_start - int(before[-1]), "before", int(before[-1])))
        if len(after):
            choices.append((int(after[0]) - evidence_end + 1, "after", int(after[0])))
        if np.any(out[evidence_start:evidence_end] == team):
            choices.append((0, "touch", evidence_start))
        distance, side, anchor = min(choices)
        if distance > max_bridge:
            continue

        start = min(anchor, evidence_start)
        end = max(anchor + 1, evidence_end)
        between = out[start:end]
        opposite = ({"H", "A"} - {team}).pop()
        opposite_at = np.flatnonzero(between == opposite)
        if len(opposite_at):
            # Opposite-team evidence is allowed only as a small outer-edge
            # boundary shift, never in the middle of the bridge.
            edge_ok = (opposite_at[-1] < max_opposite_edge or
                       opposite_at[0] >= len(between) - max_opposite_edge)
            if not edge_ok or len(opposite_at) > max_opposite_edge:
                continue

        old_transitions = int(np.sum(out[start + 1:end] != out[start:end - 1]))
        out[start:end] = team
        audit.append({
            "track_id": int(track_id), "team": team, "start": start, "end": end,
            "frames_changed": int(np.sum(base[start:end] != team)),
            "local_transitions_before": old_transitions,
        })
    return out, pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=V7_OUT)
    parser.add_argument("--visual", type=Path, default=V12_OUT)
    parser.add_argument("--tracks", type=Path, default=TRACKS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    base_table = pd.read_csv(args.base)
    visual_table = pd.read_csv(args.visual)
    labels, audit = constrained_fusion(
        base_table.label.to_numpy(str), visual_table.label.to_numpy(str),
        pd.read_csv(args.tracks))
    result = base_table.copy()
    result["label"] = labels
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(audit.to_string(index=False))
    print("changed from v7:", int(np.sum(labels != base_table.label.to_numpy(str))))
    print("transitions:", int(np.sum(labels[1:] != labels[:-1])))
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
