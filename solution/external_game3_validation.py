"""Out-of-match validation on Metrica Sports Sample Game 3.

Professional PASS/CARRY/SHOT event anchors provide independent team-control
checks. They are not possession ground truth, so this script reports anchor
accuracy and switch latency rather than competition spellF1.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from sequence_v2 import emissions, viterbi
from sequence_v4 import add_control_anchors, duration_cleanup


ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "external_data/metrica-sample-data/data/Sample_Game_3"
RAW = GAME / "Sample_Game_3_tracking.txt"
META = GAME / "Sample_Game_3_metadata.xml"
EVENTS = GAME / "Sample_Game_3_events.json"
PITCH = (105.0, 68.0)
FPS = 25
CONTROL_EVENTS = {"PASS", "CARRY", "SHOT", "RECOVERY"}


def metadata():
    root = ET.parse(META).getroot()
    teams = list(root.iter("Team"))
    home_id, away_id = teams[0].get("id"), teams[1].get("id")
    slots = np.array(
        [0 if player.get("teamId") == home_id else 1 for player in root.iter("Player")],
        dtype=np.int8,
    )
    return slots, {home_id: 0, away_id: 1}


def xy(value):
    if value.startswith("N"):
        return None
    fields = value.split(",")
    return float(fields[0]), float(fields[1])


def tracking(slots):
    n = sum(1 for _ in RAW.open())
    dh = np.full(n, np.nan, dtype=np.float32)
    da = np.full(n, np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    active = np.zeros(n, dtype=np.uint8)
    with RAW.open() as handle:
        for line in handle:
            frame_s, players_s, ball_s = line.rstrip("\n").split(":")
            frame = int(frame_s) - 1
            ball = xy(ball_s)
            players = [xy(value) for value in players_s.split(";")]
            active[frame] = sum(player is not None for player in players)
            if ball is None:
                continue
            bx[frame], by[frame] = ball
            best = [np.inf, np.inf]
            for slot, player in enumerate(players):
                if player is None:
                    continue
                distance = np.hypot(
                    (player[0] - ball[0]) * PITCH[0],
                    (player[1] - ball[1]) * PITCH[1],
                )
                side = int(slots[slot])
                best[side] = min(best[side], distance)
            if np.isfinite(best[0]):
                dh[frame] = best[0]
            if np.isfinite(best[1]):
                da[frame] = best[1]

    speed = np.full(n, np.nan, dtype=np.float32)
    consecutive = np.isfinite(bx[1:]) & np.isfinite(bx[:-1])
    speed[1:][consecutive] = FPS * np.hypot(
        (bx[1:][consecutive] - bx[:-1][consecutive]) * PITCH[0],
        (by[1:][consecutive] - by[:-1][consecutive]) * PITCH[1],
    )
    return dh, da, speed, active


def decode_variants(dh, da, speed, active):
    base_score = emissions(dh, da, speed, active)
    base = viterbi(base_score)
    anchor_score, _, _ = add_control_anchors(base_score.copy(), dh, da)
    anchored = viterbi(anchor_score)
    cleaned = duration_cleanup(anchored, anchor_score)
    return {"viterbi": base, "anchors": anchored, "duration": cleaned}


def anchor_weight_sweep(dh, da, speed, active, control):
    rows = []
    for weight in (0.5, 0.9, 1.3, 1.7, 2.1):
        score, _, _ = add_control_anchors(
            emissions(dh, da, speed, active), dh, da, weight=weight)
        states = duration_cleanup(viterbi(score), score)
        hits = []
        for row in control:
            frame = min(max(row["frame"], 0), len(states) - 1)
            hits.append(int(states[frame]) == row["team"])
        rows.append((weight, float(np.mean(hits)), int(np.sum(states[1:] != states[:-1]))))
    return rows


def duration_sweep(dh, da, speed, active, control, weight=1.7):
    score, _, _ = add_control_anchors(
        emissions(dh, da, speed, active), dh, da, weight=weight)
    raw = viterbi(score)
    rows = []
    for team_min, dead_min in ((10, 20), (15, 25), (20, 30), (25, 40)):
        states = duration_cleanup(
            raw, score, minimum=np.array([team_min, team_min, dead_min]))
        hits = []
        for row in control:
            frame = min(max(row["frame"], 0), len(states) - 1)
            hits.append(int(states[frame]) == row["team"])
        rows.append((team_min, dead_min, float(np.mean(hits)),
                     int(np.sum(states[1:] != states[:-1]))))
    return rows


def load_events(team_map):
    data = json.loads(EVENTS.read_text())["data"]
    rows = []
    for event in data:
        team = event.get("team") or {}
        team_id = team.get("id")
        if team_id not in team_map:
            continue
        subtypes = event.get("subtypes") or {}
        if isinstance(subtypes, list):
            subtype = ",".join(
                str(item.get("name") or "") for item in subtypes if isinstance(item, dict))
        else:
            subtype = str(subtypes.get("name") or "")
        rows.append({
            "frame": int(event["start"]["frame"]) - 1,
            "end": int(event["end"]["frame"]) - 1,
            "team": team_map[team_id],
            "type": event["type"]["name"].upper(),
            "subtype": subtype.upper(),
        })
    return rows


def main():
    slots, team_map = metadata()
    dh, da, speed, active = tracking(slots)
    variants = decode_variants(dh, da, speed, active)
    states = variants["duration"]
    events = load_events(team_map)
    control = [row for row in events if row["type"] in CONTROL_EVENTS]

    correct = []
    nearest_correct = []
    by_type = {}
    for row in control:
        frame = min(max(row["frame"], 0), len(states) - 1)
        hit = int(states[frame]) == row["team"]
        correct.append(hit)
        if np.isfinite(dh[frame]) and np.isfinite(da[frame]):
            nearest_correct.append(int(dh[frame] > da[frame]) == row["team"])
        by_type.setdefault(row["type"], []).append(hit)

    # Event-team changes are independent boundary anchors. Search prediction
    # changes in +/-1 second and report the closest latency.
    switches = []
    previous = None
    for row in control:
        if previous is not None and row["team"] != previous["team"]:
            frame = row["frame"]
            lo, hi = max(1, frame - FPS), min(len(states), frame + FPS + 1)
            changes = np.flatnonzero(states[lo:hi] != states[lo - 1:hi - 1]) + lo
            if len(changes):
                switches.append(int(changes[np.argmin(np.abs(changes - frame))] - frame))
        previous = row

    counts = Counter(row["type"] for row in events)
    print("frames:", len(states), "track slots:", len(slots))
    print("events:", len(events), "types:", dict(counts.most_common()))
    print("control anchors:", len(control))
    for name, variant in variants.items():
        hits = []
        for row in control:
            frame = min(max(row["frame"], 0), len(variant) - 1)
            hits.append(int(variant[frame]) == row["team"])
        print("  ablation {:<9} accuracy={:.2%}  transitions={}".format(
            name, np.mean(hits), int(np.sum(variant[1:] != variant[:-1]))))
    print("external-only anchor-weight sweep:")
    for weight, accuracy, transitions in anchor_weight_sweep(
            dh, da, speed, active, control):
        print("  weight={:.1f} accuracy={:.2%} transitions={}".format(
            weight, accuracy, transitions))
    print("external-only duration sweep at weight=1.7:")
    for team_min, dead_min, accuracy, transitions in duration_sweep(
            dh, da, speed, active, control):
        print("  min=({},{},{}) accuracy={:.2%} transitions={}".format(
            team_min, team_min, dead_min, accuracy, transitions))
    print("team accuracy at control-event start: {:.2%}".format(np.mean(correct)))
    print("nearest-player accuracy on visible anchors: {:.2%} (n={})".format(
        np.mean(nearest_correct), len(nearest_correct)))
    for event_type in sorted(by_type):
        print("  {:<10} n={:>4}  accuracy={:.2%}".format(
            event_type, len(by_type[event_type]), np.mean(by_type[event_type])))
    if switches:
        print("matched team-switch anchors within +/-1s:", len(switches))
        print("median latency: {:.1f} frames; median |latency|: {:.1f} frames".format(
            float(np.median(switches)), float(np.median(np.abs(switches)))))

    # Diagnose a possible event/tracking clock offset without changing the
    # competition model. A flat curve means model error; a sharp shifted peak
    # means the external feeds use different event timing conventions.
    offset_scores = []
    for offset in range(-25, 26, 5):
        hits = []
        for row in control:
            frame = min(max(row["frame"] + offset, 0), len(states) - 1)
            hits.append(int(states[frame]) == row["team"])
        offset_scores.append((offset, float(np.mean(hits))))
    print("event-offset diagnostic:",
          " ".join("{:+d}:{:.1%}".format(offset, score)
                   for offset, score in offset_scores))


if __name__ == "__main__":
    main()
