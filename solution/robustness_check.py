"""Label-free robustness checks for sequence models.

The public five-minute labels are intentionally not used. We corrupt the full
match observations in deterministic ways and measure prediction stability.
This catches models that merely exploit one fragile tracking channel.
"""

from __future__ import annotations

import numpy as np

from sequence_v2 import META, RAW, emissions, extract, slot_teams, viterbi
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import decode_v4
from sequence_v5 import decode_v5, raw_ball_motion
from sequence_v6 import decode_v6
from sequence_v7 import decode_v7


def spell_count(states: np.ndarray) -> int:
    alive = states != 2
    starts = alive & np.r_[True, (states[1:] != states[:-1]) | ~alive[:-1]]
    return int(starts.sum())


def report(name, reference, candidate, affected):
    agreement = np.mean(reference == candidate)
    local = np.mean(reference[affected] == candidate[affected]) if affected.any() else 1.0
    print(
        f"{name:<28} agreement={agreement:6.2%}  affected={local:6.2%}  "
        f"spells={spell_count(candidate):4d}"
    )


def decode(dh, da, speed, active, events=None):
    score = emissions(dh, da, speed, active)
    if events is not None:
        score = add_event_priors(score, events)
    return viterbi(score)


def main():
    rng = np.random.default_rng(2027)
    dh, da, speed, active = extract(RAW, slot_teams(META))
    events = load_events(SMART)
    base_v2 = decode(dh, da, speed, active)
    base_v3 = decode(dh, da, speed, active, events)
    base_v4, _, _ = decode_v4(dh, da, speed, active, events)
    _, _, raw_speed = raw_ball_motion(RAW)
    base_v5, _, _ = decode_v5(dh, da, speed, active, events, raw_speed)
    base_v6, _, _ = decode_v6(dh, da, speed, active, events)
    base_v7, _, _ = decode_v7(dh, da, speed, active, events)
    observed = np.isfinite(dh) & np.isfinite(da)

    print("reference spells: v2={}, v3={}, v4={}, v5={}, v6={}, v7={}".format(
        spell_count(base_v2), spell_count(base_v3), spell_count(base_v4),
        spell_count(base_v5), spell_count(base_v6), spell_count(base_v7)))
    print("v2/v3 agreement: {:.2%}".format(np.mean(base_v2 == base_v3)))
    print("v3/v4 agreement: {:.2%}\n".format(np.mean(base_v3 == base_v4)))
    print("v4/v5 agreement: {:.2%}\n".format(np.mean(base_v4 == base_v5)))
    print("v4/v6 agreement: {:.2%}\n".format(np.mean(base_v4 == base_v6)))
    print("v6/v7 agreement: {:.2%}\n".format(np.mean(base_v6 == base_v7)))

    for fraction in (0.10, 0.20, 0.30):
        mask = observed & (rng.random(len(dh)) < fraction)
        ch, ca, cs = dh.copy(), da.copy(), speed.copy()
        ch[mask] = np.nan
        ca[mask] = np.nan
        cs[mask] = np.nan
        report(f"v2 extra missing {fraction:.0%}", base_v2,
               decode(ch, ca, cs, active), mask)
        report(f"v3 extra missing {fraction:.0%}", base_v3,
               decode(ch, ca, cs, active, events), mask)
        corrupted_v4, _, _ = decode_v4(ch, ca, cs, active, events)
        report(f"v4 extra missing {fraction:.0%}", base_v4, corrupted_v4, mask)
        # Restart alignment uses the independent raw ball-speed channel, so it
        # remains available when the positional ball track is synthetically masked.
        corrupted_v5, _, _ = decode_v5(ch, ca, cs, active, events, raw_speed)
        report(f"v5 extra missing {fraction:.0%}", base_v5, corrupted_v5, mask)
        corrupted_v6, _, _ = decode_v6(ch, ca, cs, active, events)
        report(f"v6 extra missing {fraction:.0%}", base_v6, corrupted_v6, mask)
        corrupted_v7, _, _ = decode_v7(ch, ca, cs, active, events)
        report(f"v7 extra missing {fraction:.0%}", base_v7, corrupted_v7, mask)

    # Distance jitter represents homography noise and small player/ball tracking
    # errors. Clamp at zero because distances cannot be negative.
    jittered = observed.copy()
    jh, ja = dh.copy(), da.copy()
    jh[jittered] = np.maximum(0.0, jh[jittered] + rng.normal(0, 0.75, jittered.sum()))
    ja[jittered] = np.maximum(0.0, ja[jittered] + rng.normal(0, 0.75, jittered.sum()))
    report("v2 distance jitter 0.75m", base_v2,
           decode(jh, ja, speed, active), jittered)
    report("v3 distance jitter 0.75m", base_v3,
           decode(jh, ja, speed, active, events), jittered)
    jittered_v4, _, _ = decode_v4(jh, ja, speed, active, events)
    report("v4 distance jitter 0.75m", base_v4, jittered_v4, jittered)
    jittered_v5, _, _ = decode_v5(jh, ja, speed, active, events, raw_speed)
    report("v5 distance jitter 0.75m", base_v5, jittered_v5, jittered)
    jittered_v6, _, _ = decode_v6(jh, ja, speed, active, events)
    report("v6 distance jitter 0.75m", base_v6, jittered_v6, jittered)
    jittered_v7, _, _ = decode_v7(jh, ja, speed, active, events)
    report("v7 distance jitter 0.75m", base_v7, jittered_v7, jittered)


if __name__ == "__main__":
    main()
