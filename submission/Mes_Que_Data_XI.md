# Mes Que Data XI — Technical Document

## Approach

We model possession as a three-state temporal sequence: home possession (H),
away possession (A), and dead ball (D). The system is algorithmic and uses no
manual labels from the competition match.

For every frame, physical evidence is constructed from the nearest home and
away player distances to the tracked ball, ball speed, ball visibility, and
the number of active player tracks. Distance evidence is bounded so a single
tracking error cannot dominate. Approximately one second of local evidence is
aggregated before global decoding.

Proximity alone is not treated as control. A control anchor requires a player
to be within a 2.5 m capture radius, clearly closer than the opponent, and
persistent across neighbouring frames. Weak automatic Smart Tagging priors are
added without copying tags into the answer. A Viterbi decoder then finds a
globally coherent H/A/D path. A final externally calibrated cleanup removes a
short team spell only when it lacks persistent physical control evidence; it
never removes D.

## Assets used

- Tactical ATD tracking and metadata: primary physical evidence.
- Smart Tagging XML: weak phase and restart priors.
- Public Metrica Sample Games 1, 2 and 3: external validation and calibration.
- Official public five-minute labels: final sanity check and candidate
  pipeline verification, not training, tuning or model selection.

Broadcast ATD and tactical-video recovery were tested as secondary sources but
excluded from the frozen submission. The former introduced uncertain
transitions; the latter depends on a match-derived checkpoint whose public
distribution is awaiting organiser clarification.

## External validation

The anchor weight and minimum duration were selected using Metrica Sample Game
3 professional event anchors. Frozen parameters were then checked on Sample
Games 1 and 2. V7 obtains 89.99% pooled team accuracy over 2,827 independent
control-event anchors in Games 1 and 2; PASS anchors score 96.50% and 96.78%.
Reducing transition cost to 0.75 improved control-event accuracy and switch
coverage in the same direction on all three external matches. The v15 cleanup
constants were selected on Games 1 and 2. On held-out Game 3, event accuracy
changes from 67.99% to 67.92% while transitions fall from 580 to 536.

Across 110 rolling five-minute external windows, v7 and the v15 cleanup obtain
82.58% and 82.41% event-anchor accuracy respectively; average transitions fall
from 27.5 to 24.7. These are event proxies because the sample games do not
provide controlled-possession spells. A public paired bootstrap does not show
a statistically decisive difference between v15 and smooth125; v7 is retained
as the lower-leakage fallback.

Robustness tests additionally mask 30% of visible ball observations and add
tracking-distance noise. These tests are independent of the competition ground
truth.

## Implemented possession rules

- A team controls the ball when it has persistent, clearly superior proximity,
  not merely the nearest player for one frame.
- A loose or contested ball normally retains the previous coherent state until
  sufficient evidence supports a handover.
- Missing ball coordinates alone are weak dead-ball evidence because missing
  observations also occur during live play.
- Tracking-wide inactivity and a stationary ball far from players support D.
- Restarts receive weak dead-to-live evidence from automatic set-piece windows.
- Very short decoded runs are merged according to neighbouring states and
  segment-level evidence.

## Training data

The submitted model is rule-based. External validation uses the public Metrica
Sports Sample Data and professional event annotations. No human-produced ball
or possession annotations are used.

## Limitations

Long ball occlusions, crowded challenges, recoveries, shots, and exact restart
boundaries remain the main failure modes. Automatic player and ball tracking
may contain identity switches and false ball locations. The Smart Tagging feed
is automatic rather than ground truth. Missing-ball intervals remain the main
source of uncertainty because the experimental visual recovery is excluded.

## Reproduction

Use Python 3.9 or later from the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r solution/requirements.txt
.venv/bin/python solution/run_final.py
.venv/bin/python evaluation/validate_submission.py submission/Mes_Que_Data_XI.csv
```

Expected SHA-256:

`38f3b537f637ae645844d5d851ece6b68172382c2dd42741d5bc1d124a65d576`
