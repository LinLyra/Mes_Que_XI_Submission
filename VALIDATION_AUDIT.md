# Validation and leakage audit

This document separates a runnable submission from evidence that it will
generalise. The public five-minute labels contain only 17 possessions; they
are a final sanity check, not a training set.

## What the public labels influenced

| Component | Source | Status |
|---|---|---|
| v7 anchor weight 1.7 | Metrica Sample Game 3 | external |
| v7 transition scale 0.75 | Metrica Games 1–3 | external |
| v7 minimum durations 25/25/40 frames | Metrica Game 3 plateau | external |
| visual detector threshold 0.75 and continuity gates | weak labels, synthetic occlusions and visual audit | label-free |
| top-25 visual search interval selection | uncertainty features only | label-free |
| v13 spell-fusion architecture | designed after observing v12's public fragmentation | **public-informed** |
| v13 `max_bridge=250`, `max_opposite_edge=25` | conservative engineering choices after public diagnosis | **public-informed** |
| v15 `max_short=125`, `min_peak=.20`, `min_frames=5` | selected on Metrica Games 1–2 before public scoring | external |
| promotion of v15 over v7 | public spellF1 plus external proxy checks | **public-informed model selection** |

Therefore v15's numerical public gain is not an unbiased estimate of its
hidden-test gain. V7 remains the clean, externally selected fallback. V15 is a
stronger candidate, not proven ground truth.

## Metric limitation

Metrica Sample Games 1–3 provide tracking and sparse professional events, but
not frame-level `H/A/D` controlled-possession labels. Computing the official
spellF1 on them would require inventing pseudo-ground-truth spells and would
make the validation circular. External validation consequently reports:

- team accuracy at PASS/CARRY/SHOT/RECOVERY/BALL LOST anchors;
- coverage of event-team changes by a predicted transition within 25 frames;
- predicted transition count as a fragmentation diagnostic.

`solution/external_rolling_validation.py` reports those quantities on rolling
five-minute windows and performs leave-one-game-out selection. The official
spellF1 and its spell bootstrap remain confined to the supplied public window.

## Public uncertainty

With 1,000 possession-level bootstrap draws, v15's public spellF1 interval is
approximately 0.334–0.562. The observed difference from smooth125 (0.4476 vs
0.4242) is therefore not statistically decisive. A 10,000-draw *paired*
bootstrap estimates the difference as +0.0240 with a 5–95% interval of
−0.1160 to +0.1628 and only a 60.7% probability that it is positive. It is
evidence of promise, not proof of superiority on the hidden match.

## External rolling result

Across 110 overlapping five-minute windows, the externally labelled event
proxies are:

| Decoder | Anchor accuracy | Switch coverage ±25f | Transitions/window |
|---|---:|---:|---:|
| smooth125 | 73.88% ± 12.01% | 16.88% ± 13.03% | 16.8 ± 7.7 |
| v7 | **82.58% ± 11.49%** | **59.63% ± 15.50%** | 27.5 ± 6.6 |
| v15 core | 82.41% ± 11.41% | 57.29% ± 15.30% | 24.7 ± 6.3 |

V15 is therefore a conservative fragmentation trade-off, not an external
accuracy win. Leave-one-game-out selection also chooses different cleanup
constants in different folds. The submission decision must keep both v7 and
v15 available; the public result alone does not justify discarding v7.

