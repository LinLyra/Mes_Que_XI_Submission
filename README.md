# Mes Que Data XI

Our submission for **More Than A Hack 2027 — Phase 1: Ball Possession**, built
from the FC Barcelona vs Athletic Club professional match data supplied to
registered participants.

The task is to assign one state to every tactical-video frame:

- `H` — home possession while the ball is live
- `A` — away possession while the ball is live
- `D` — dead ball
- `X` — abstain

The primary metric is `spellF1`, which rewards recovering coherent possession
intervals rather than merely maximising independent frame accuracy.

## Current submission

The frozen primary candidate is **sequence v15-core**, combining:

1. player-to-ball distance, ball speed, visibility and player activity;
2. automatic Smart Tagging priors for restarts and tactical phases;
3. sustained-control anchors rather than nearest-player decisions alone;
4. global Viterbi decoding and externally calibrated, anchor-protected segment
   cleanup.

Parameters were calibrated on separate public Metrica sample matches rather
than repeatedly tuned against the competition's five-minute public labels.
External validation over Sample Games 1 and 2 reaches **89.78% team accuracy**
on 2,827 PASS/CARRY/SHOT/RECOVERY anchors.

```bash
pip install -r solution/requirements.txt
python solution/run_final.py
python evaluation/validate_submission.py submission/Mes_Que_Data_XI.csv
```

This command uses only the supplied ATD, metadata and Smart Tagging files and
reproduces the hash documented in `submission/FREEZE_MANIFEST.md`.

The implementation and validation policy are documented in
[solution/README.md](solution/README.md). The technical report is
[submission/Mes_Que_Data_XI.md](submission/Mes_Que_Data_XI.md).

## Experimental visual work

The full tactical MP4 is used locally for algorithmic ball-recovery research,
but it is separately licensed and is not committed.

- Bidirectional optical flow was validated on synthetic occlusions.
- All 43 real short gaps failed the pre-declared endpoint-consistency gate, so
  the visual module correctly abstained instead of corrupting v7.
- A pretrained sports-ball model cross-filtered the weak labels, and a local
  YOLO11n detector reached 0.930 precision / 0.881 recall on a blocked temporal
  validation set.
- The top-100 experiment examines player-centred crops in label-free
  uncertainty intervals. Six trajectories pass the fixed 0.75
  confidence and continuity gates; fusion may move an existing boundary but
  cannot create an isolated possession spell.

The visual experiment is not part of the frozen submission while publication
permission for its match-derived detector checkpoint is unresolved. Its local
CSV remains a separately hashed, non-submitted candidate.

The project promotes a visual model only if it improves independently defined
tests without introducing unstable possession transitions.

## Repository layout

```text
solution/       our sequence models, robustness tests and visual experiments
submission/     technical report; generated submission CSV remains private
evaluation/     official metric implementation and public sanity labels
starter/        official loaders and reference material
data/           local licensed assets, excluded from Git
```

## Data and licence

This repository contains code, documentation, official evaluation utilities
and the small public ground-truth window. It does **not** contain the licensed
match videos, tracking exports, full-match ground truth, generated diagnostics
or complete submission CSV.

Challenge materials originate from Metrica Sports, FC Barcelona and Mobile
World Capital Barcelona. Repository code is covered by [LICENSE](LICENSE);
competition data remains subject to its separate participant terms.
