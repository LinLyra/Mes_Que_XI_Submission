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
globally coherent H/A/D path, followed by a duration-aware cleanup that removes
implausibly short H/A and D runs using accumulated segment evidence.

## Assets used

- Tactical ATD tracking and metadata: primary physical evidence.
- Smart Tagging XML: weak phase and restart priors.
- Public Metrica Sample Games 1, 2 and 3: external validation and calibration.
- Official public five-minute labels: pipeline sanity check only, never used to
  train, tune, or select parameters.

Broadcast ATD was aligned and tested as a secondary source. It recovered some
missing tactical ball locations, but introduced uncertain extra transitions;
therefore it was excluded from the final v7 submission. Homography and video
were not used by the final model. Experimental code is retained for audit.

## External validation

The anchor weight and minimum duration were selected using Metrica Sample Game
3 professional event anchors. Frozen parameters were then checked on Sample
Games 1 and 2. V7 obtains 89.99% pooled team accuracy over 2,827 independent
control-event anchors in Games 1 and 2; PASS anchors score 96.50% and 96.78%.
Reducing transition cost to 0.75 improved control-event accuracy and switch
coverage in the same direction on all three external matches.

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

The model is rule-based and has no learned weights fitted to the competition
match. External validation uses the public Metrica Sports Sample Data repository
and its professional event annotations. No human-produced annotations from the
provided Barcelona–Athletic Club match are used.

## Limitations

Long ball occlusions, crowded challenges, recoveries, shots, and exact restart
boundaries remain the main failure modes. Automatic player and ball tracking
may contain identity switches and false ball locations. The Smart Tagging feed
is automatic rather than ground truth. A future version would use the tactical
video and homography for algorithmic ball recovery, validated only on other
matches or synthetic corruptions.

## Reproduction

Use Python 3.9 or later from the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r solution/requirements.txt
.venv/bin/python solution/run_final.py
.venv/bin/python evaluation/validate_submission.py submission/Mes_Que_Data_XI.csv
```

Expected SHA-256:

`af5d648a37db6051a4eadf519687393d228387efcac503d0ba6e730e3b469b68`
