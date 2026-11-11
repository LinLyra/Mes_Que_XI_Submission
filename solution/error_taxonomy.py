"""Rank full-match uncertainty without consulting possession ground truth.

Intervals where independently motivated decoders disagree are the best places
to spend CV effort. The generated CSV is local-only and gitignored because it
contains frame-level derivatives of the gated match data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sequence_v2 import META, RAW, extract, slot_teams
from sequence_v3 import SMART, load_events
from sequence_v4 import decode_v4, runs
from sequence_v3 import add_event_priors
from sequence_v2 import emissions, viterbi


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/diagnostics/ambiguous_intervals.csv"
FPS = 25


def event_masks(events, n):
    restart = np.zeros(n, dtype=bool)
    tagged = np.zeros(n, dtype=bool)
    for code, start, end, _ in events:
        start, end = max(0, start), min(n, end)
        tagged[start:end] = True
        if code.strip().upper() == "SET PIECES":
            restart[start:end] = True
    return restart, tagged


def category(missing, contested, low_activity, restart):
    if restart >= 0.25:
        return "restart_boundary"
    if low_activity >= 0.25:
        return "tracking_wide_stoppage"
    if missing >= 0.60:
        return "missing_ball"
    if contested >= 0.50:
        return "contested_control"
    return "weak_or_conflicting_evidence"


def main():
    teams = slot_teams(META)
    dh, da, speed, active = extract(RAW, teams)
    events = load_events(SMART)

    v3_score = add_event_priors(emissions(dh, da, speed, active), events)
    v3 = viterbi(v3_score)
    v4, home_anchor, away_anchor = decode_v4(dh, da, speed, active, events)

    disagree = v3 != v4
    restart, tagged = event_masks(events, len(v4))
    missing = ~(np.isfinite(dh) & np.isfinite(da))
    contested = np.isfinite(dh) & np.isfinite(da) & (np.abs(dh - da) < 1.0)
    low_activity = active < 12
    anchor = np.maximum(home_anchor, away_anchor)

    starts = np.r_[0, np.flatnonzero(disagree[1:] != disagree[:-1]) + 1]
    ends = np.r_[starts[1:], len(disagree)]
    rows = []
    for start, end in zip(starts, ends):
        if not disagree[start]:
            continue
        length = end - start
        miss_rate = float(missing[start:end].mean())
        contest_rate = float(contested[start:end].mean())
        low_rate = float(low_activity[start:end].mean())
        restart_rate = float(restart[start:end].mean())
        # Longer disagreement with weak anchors is more valuable for a new
        # modality than a one-frame boundary offset with strong tracking.
        priority = length / FPS * (1.0 + miss_rate + contest_rate) * (
            1.25 - float(np.clip(anchor[start:end].mean(), 0.0, 1.0))
        )
        rows.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_s": round(length / FPS, 3),
            "v3_state": "HAD"[int(v3[start])],
            "v4_state": "HAD"[int(v4[start])],
            "missing_rate": round(miss_rate, 4),
            "contested_rate": round(contest_rate, 4),
            "low_activity_rate": round(low_rate, 4),
            "restart_overlap": round(restart_rate, 4),
            "tagged_overlap": round(float(tagged[start:end].mean()), 4),
            "mean_anchor": round(float(anchor[start:end].mean()), 4),
            "category": category(miss_rate, contest_rate, low_rate, restart_rate),
            "priority": round(priority, 4),
        })

    rows.sort(key=lambda row: row["priority"], reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    seconds = sum(row["duration_s"] for row in rows)
    counts = {}
    durations = {}
    for row in rows:
        key = row["category"]
        counts[key] = counts.get(key, 0) + 1
        durations[key] = durations.get(key, 0.0) + row["duration_s"]
    print(f"disagreement intervals: {len(rows)}")
    print(f"disagreement duration: {seconds:.1f}s ({disagree.mean():.2%} of video)")
    for key in sorted(counts, key=lambda item: durations[item], reverse=True):
        print(f"  {key:<30} {counts[key]:>4} intervals  {durations[key]:>7.1f}s")
    print("top candidates for targeted CV:")
    for row in rows[:10]:
        print("  {start_frame:>6}-{end_frame:<6} {duration_s:>6.2f}s  "
              "{category:<28} priority={priority:.2f}".format(**row))
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
