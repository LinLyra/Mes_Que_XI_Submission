"""Rolling-window and leave-one-game-out validation on Metrica Games 1--3.

The sample games have no controlled-possession ground truth, so this script
does not claim to compute official spellF1. It uses sparse professional events
as independent team-control and boundary proxies and reports each quantity
separately instead of hiding trade-offs in a tuned composite score.
"""
from __future__ import annotations

import numpy as np

import external_csv_validation as csv_games
import external_game3_validation as game3
from sequence_v2 import emissions, viterbi
from sequence_v4 import add_control_anchors, duration_cleanup
from sequence_v15 import (MAX_SHORT_TEAM_RUN, MIN_ANCHOR_FRAMES,
                          MIN_ANCHOR_PEAK, anchor_protected_cleanup)


FPS = 25
WINDOW = 5 * 60 * FPS
STRIDE = WINDOW // 2


def bridge_short_gaps(codes, max_gap=75):
    out = codes.copy()
    valid = np.flatnonzero(codes >= 0)
    for left, right in zip(valid[:-1], valid[1:]):
        if right - left - 1 <= max_gap and codes[left] == codes[right]:
            out[left + 1:right] = codes[left]
    return out


def smooth125(dh, da):
    observed = np.full(len(dh), -1, dtype=np.int8)
    seen = np.isfinite(dh) & np.isfinite(da)
    observed[seen] = (da[seen] < dh[seen]).astype(np.int8)
    observed = bridge_short_gaps(observed)
    home = np.r_[0, np.cumsum(observed == 0)]
    away = np.r_[0, np.cumsum(observed == 1)]
    index = np.arange(len(observed))
    low = np.maximum(0, index - 125)
    high = np.minimum(len(observed), index + 126)
    return np.where(home[high] - home[low] >= away[high] - away[low], 0, 1).astype(np.uint8)


def load_all():
    games = []
    for number in (1, 2):
        game = csv_games.DATA / "Sample_Game_{}".format(number)
        frames, dh, da, speed, active = csv_games.tracking(game)
        events = csv_games.load_events(game, {int(frame): i for i, frame in enumerate(frames)})
        control = [row for row in events if row["type"] in csv_games.CONTROL_EVENTS]
        games.append((str(number), dh, da, speed, active, control))
    slots, team_map = game3.metadata()
    dh, da, speed, active = game3.tracking(slots)
    events = game3.load_events(team_map)
    control = [row for row in events if row["type"] in game3.CONTROL_EVENTS]
    games.append(("3", dh, da, speed, active, control))
    return games


def decode(dh, da, speed, active, max_short=MAX_SHORT_TEAM_RUN,
           min_peak=MIN_ANCHOR_PEAK, min_frames=MIN_ANCHOR_FRAMES):
    score, home, away = add_control_anchors(
        emissions(dh, da, speed, active), dh, da, weight=1.7)
    v7 = duration_cleanup(
        viterbi(score, transition_scale=.75), score,
        minimum=np.asarray([25, 25, 40]))
    regularised = anchor_protected_cleanup(
        v7, score, home, away, max_short=max_short,
        min_peak=min_peak, min_frames=min_frames)
    return score, home, away, v7, regularised


def event_metrics(states, control, start=0, end=None):
    end = len(states) if end is None else end
    local = [row for row in control if start <= row["frame"] < end]
    accuracy = np.mean([states[row["frame"]] == row["team"] for row in local]) if local else np.nan
    switches = [row["frame"] for previous, row in zip(local, local[1:])
                if previous["team"] != row["team"]]
    changes = np.flatnonzero(states[start + 1:end] != states[start:end - 1]) + start + 1
    coverage = np.mean([np.any(np.abs(changes - frame) <= 25)
                        for frame in switches]) if switches else np.nan
    return accuracy, coverage, len(changes), len(local), len(switches)


def rolling_report(games):
    collected = {"smooth125": [], "v7": [], "v15-core": []}
    for name, dh, da, speed, active, control in games:
        _, _, _, v7, v15 = decode(dh, da, speed, active)
        variants = {"smooth125": smooth125(dh, da), "v7": v7, "v15-core": v15}
        for start in range(0, max(1, len(v7) - WINDOW + 1), STRIDE):
            end = min(len(v7), start + WINDOW)
            if end - start < WINDOW:
                continue
            for label, states in variants.items():
                values = event_metrics(states, control, start, end)
                if values[3] >= 20:
                    collected[label].append(values)
    print("rolling five-minute windows (mean +/- std; event proxies, not spellF1)")
    for label, rows in collected.items():
        values = np.asarray(rows, dtype=float)
        print("  {:<10} n={:>3}  anchor_acc={:.2%} +/- {:.2%}  switch_cov={:.2%} +/- {:.2%}  transitions={:.1f} +/- {:.1f}".format(
            label, len(rows), np.nanmean(values[:, 0]), np.nanstd(values[:, 0]),
            np.nanmean(values[:, 1]), np.nanstd(values[:, 1]),
            np.mean(values[:, 2]), np.std(values[:, 2])))


def three_fold_report(games):
    # Predeclared selection: remain within 0.25 percentage points of the best
    # training anchor accuracy, then choose the fewest transitions.
    grid = [(m, p, c) for m in (75, 100, 125, 150, 200)
            for p in (.20, .30) for c in (3, 5)]
    prepared = []
    for name, dh, da, speed, active, control in games:
        score, home, away, v7, _ = decode(dh, da, speed, active)
        prepared.append((name, score, home, away, v7, control))
    print("leave-one-game-out (selection tolerance: 0.25 percentage points)")
    for held in range(3):
        candidates = []
        for params in grid:
            hits = total = transitions = 0
            for index, (_, score, home, away, v7, control) in enumerate(prepared):
                if index == held:
                    continue
                states = anchor_protected_cleanup(v7, score, home, away, *params)
                hits += sum(states[row["frame"]] == row["team"] for row in control)
                total += len(control)
                transitions += int(np.sum(states[1:] != states[:-1]))
            candidates.append((hits / total, transitions, params))
        best_accuracy = max(row[0] for row in candidates)
        eligible = [row for row in candidates if row[0] >= best_accuracy - .0025]
        train_accuracy, train_transitions, params = min(eligible, key=lambda row: (row[1], -row[0]))
        name, score, home, away, v7, control = prepared[held]
        states = anchor_protected_cleanup(v7, score, home, away, *params)
        test = event_metrics(states, control)
        base = event_metrics(v7, control)
        print("  holdout={} params={} train_acc={:.2%} train_trans={}  test_acc={:.2%} (v7 {:.2%}) test_cov={:.2%} (v7 {:.2%}) test_trans={} (v7 {})".format(
            name, params, train_accuracy, train_transitions,
            test[0], base[0], test[1], base[1], test[2], base[2]))


def main():
    games = load_all()
    rolling_report(games)
    three_fold_report(games)


if __name__ == "__main__":
    main()
