"""Independent validation on Metrica Sample Games 1 and 2.

The legacy CSV feeds contain home/away tracking and professional event logs.
Events are sparse control anchors rather than possession ground truth, so this
reports event-team accuracy, nearest-player accuracy and switch latency.
No competition labels or public five-minute ground truth are used.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from sequence_v2 import emissions, viterbi
from sequence_v4 import add_control_anchors, duration_cleanup


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external_data/metrica-sample-data/data"
PITCH = (105.0, 68.0)
FPS = 25
CONTROL_EVENTS = {"PASS", "SHOT", "RECOVERY", "BALL LOST"}


def _read_tracking(path: Path):
    """Read a three-row-header Metrica tracking CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        next(rows)
        next(rows)
        header = next(rows)
        ball_col = header.index("Ball")
        player_cols = list(range(3, ball_col, 2))
        values = []
        for row in rows:
            if not row or not row[1]:
                continue
            values.append(row)

    frames = np.asarray([int(row[1]) for row in values], dtype=np.int32)
    periods = np.asarray([int(row[0]) for row in values], dtype=np.int8)
    players = np.full((len(values), len(player_cols), 2), np.nan, dtype=np.float32)
    ball = np.full((len(values), 2), np.nan, dtype=np.float32)
    for i, row in enumerate(values):
        for j, col in enumerate(player_cols):
            try:
                players[i, j] = float(row[col]), float(row[col + 1])
            except (ValueError, IndexError):
                pass
        try:
            ball[i] = float(row[ball_col]), float(row[ball_col + 1])
        except (ValueError, IndexError):
            pass
    return frames, periods, players, ball


def tracking(game: Path):
    home_path = next(game.glob("*TrackingData_Home_Team.csv"))
    away_path = next(game.glob("*TrackingData_Away_Team.csv"))
    hf, hp, home, hb = _read_tracking(home_path)
    af, ap, away, ab = _read_tracking(away_path)
    if not np.array_equal(hf, af) or not np.array_equal(hp, ap):
        raise ValueError("home and away feeds are not frame-aligned")

    # Either team file may carry the ball. Prefer Home, then fill missing rows
    # from Away (Game 2 begins with missing ball coordinates in both feeds).
    ball = hb.copy()
    fill = ~np.isfinite(ball[:, 0]) & np.isfinite(ab[:, 0])
    ball[fill] = ab[fill]
    scale = np.asarray(PITCH, dtype=np.float32)
    home_distance = np.linalg.norm((home - ball[:, None, :]) * scale, axis=2)
    away_distance = np.linalg.norm((away - ball[:, None, :]) * scale, axis=2)
    dh = np.min(np.where(np.isfinite(home_distance), home_distance, np.inf), axis=1)
    da = np.min(np.where(np.isfinite(away_distance), away_distance, np.inf), axis=1)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    active = (np.isfinite(home[:, :, 0]).sum(axis=1) +
              np.isfinite(away[:, :, 0]).sum(axis=1)).astype(np.uint8)

    speed = np.full(len(ball), np.nan, dtype=np.float32)
    consecutive = ((hf[1:] == hf[:-1] + 1) & (hp[1:] == hp[:-1]) &
                   np.isfinite(ball[1:, 0]) & np.isfinite(ball[:-1, 0]))
    delta = np.linalg.norm((ball[1:] - ball[:-1]) * scale, axis=1) * FPS
    speed[1:][consecutive] = delta[consecutive]
    return hf, dh, da, speed, active


def load_events(game: Path, frame_to_index):
    path = next(game.glob("*RawEventsData.csv"))
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for event in csv.DictReader(handle):
            frame = int(event["Start Frame"])
            if frame not in frame_to_index or event["Team"] not in {"Home", "Away"}:
                continue
            rows.append({
                "frame": frame_to_index[frame],
                "team": 0 if event["Team"] == "Home" else 1,
                "type": event["Type"].upper(),
                "subtype": event["Subtype"].upper(),
            })
    return rows


def decode(dh, da, speed, active, weight=1.7, transition_scale=1.0):
    score = emissions(dh, da, speed, active)
    if weight:
        score, _, _ = add_control_anchors(score, dh, da, weight=weight)
    return duration_cleanup(
        viterbi(score, transition_scale=transition_scale),
        score,
        minimum=np.asarray([25, 25, 40]))


def evaluate(name, states, dh, da, events):
    control = [row for row in events if row["type"] in CONTROL_EVENTS]
    hits, nearest = [], []
    by_type = {}
    for row in control:
        frame = row["frame"]
        hit = int(states[frame]) == row["team"]
        hits.append(hit)
        by_type.setdefault(row["type"], []).append(hit)
        if np.isfinite(dh[frame]) and np.isfinite(da[frame]):
            nearest.append(int(dh[frame] > da[frame]) == row["team"])

    switches = []
    for previous, row in zip(control, control[1:]):
        if row["team"] == previous["team"]:
            continue
        frame = row["frame"]
        lo, hi = max(1, frame - FPS), min(len(states), frame + FPS + 1)
        changes = np.flatnonzero(states[lo:hi] != states[lo - 1:hi - 1]) + lo
        if len(changes):
            switches.append(int(changes[np.argmin(np.abs(changes - frame))] - frame))

    print("\n", name)
    print("frames:", len(states), "events:", len(events),
          "types:", dict(Counter(row["type"] for row in events).most_common()))
    print("control anchors:", len(control),
          "accuracy={:.2%}".format(np.mean(hits)),
          "nearest={:.2%} (n={})".format(np.mean(nearest), len(nearest)))
    print("transitions:", int(np.sum(states[1:] != states[:-1])))
    for event_type in sorted(by_type):
        print("  {:<10} n={:>4} accuracy={:.2%}".format(
            event_type, len(by_type[event_type]), np.mean(by_type[event_type])))
    if switches:
        print("switches matched +/-1s:", len(switches),
              "median latency={:.1f}, median |latency|={:.1f}".format(
                  np.median(switches), np.median(np.abs(switches))))
    return len(control), int(np.sum(hits))


def main():
    total_n = total_hits = 0
    for number in (1, 2):
        game = DATA / "Sample_Game_{}".format(number)
        frames, dh, da, speed, active = tracking(game)
        events = load_events(game, {int(frame): i for i, frame in enumerate(frames)})
        control = [row for row in events if row["type"] in CONTROL_EVENTS]
        variants = {
            "no anchors": decode(dh, da, speed, active, weight=0.0),
            "v4 weight=.9": decode(dh, da, speed, active, weight=0.9),
            "v6 weight=1.7": decode(dh, da, speed, active, weight=1.7),
            "v7 scale=.75": decode(
                dh, da, speed, active, weight=1.7, transition_scale=0.75),
        }
        print("\n{} ablation:".format(game.name))
        for label, variant in variants.items():
            accuracy = np.mean([int(variant[row["frame"]]) == row["team"]
                                for row in control])
            print("  {:<14} accuracy={:.2%} transitions={}".format(
                label, accuracy, int(np.sum(variant[1:] != variant[:-1]))))
        states = variants["v7 scale=.75"]
        n, hits = evaluate("Sample Game {}".format(number), states, dh, da, events)
        total_n += n
        total_hits += hits
    print("\npooled control-event accuracy: {:.2%} (n={})".format(
        total_hits / total_n, total_n))


if __name__ == "__main__":
    main()
