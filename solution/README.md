# Mes Que Data XI — Phase 1 pipeline

This directory contains CPU-only, reproducible possession baselines. Match data
stays in `data/` and is excluded from Git.

## Current candidate

`sequence_v7.py` combines four evidence layers:

1. **Physical observation** — nearest home/away player distances, ball presence,
   ball speed and tracking-wide activity.
2. **Automatic tactical priors** — overlapping Smart Tagging phases are averaged
   into weak team evidence; set pieces create a soft dead-to-live restart shape.
3. **Control anchors** — proximity is not enough. A player must be close to the
   ball and clearly closer than the opponent for several frames.
4. **Duration-aware decoding** — Viterbi finds the global H/A/D path, then short
   runs are resolved from accumulated segment evidence.

Its control-anchor weight and duration floor were selected on the separate,
public Metrica Sample Game 3. That external game provides 2,830 professional
PASS/CARRY/SHOT/RECOVERY anchors and is never used as competition ground truth.
The frozen v6 parameters were then checked without further tuning on Sample
Games 1 and 2: 89.78% team accuracy over 2,827 event anchors, versus 86.17%
without control anchors. PASS anchors score 96.50% and 96.78% respectively.
V7 then reduces the transition penalty to 0.75: this direction independently
improves control-event accuracy and switch coverage on all three external
matches. On the competition public sanity window it is effectively identical
to v6, as expected; that window is not used for parameter selection.

No public ground-truth label is read by any solution script. The public five
minutes are used only by the official evaluator as a final semantic check.

## Run

```bash
python solution/sequence_v7.py
python evaluation/run_evaluation.py
python solution/robustness_check.py
python solution/external_game3_validation.py
python solution/external_csv_validation.py
python solution/test_sequence.py
python solution/visual_candidates.py
python solution/visual_pseudolabels.py
```

The submission is written to:

```text
evaluation/results/mes_que_data_sequence_v7.csv
```

## Validation policy

Do not select parameters by repeatedly scoring the public five-minute window.
Use `robustness_check.py` to test additional missing-ball rates and tracking
distance noise. Later validation should use documented external matches or
synthetic corruptions, never manually produced labels for the provided match.

## Data dependencies

Required locally, never commit these files:

```text
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The homography and videos are not required by v4. They are reserved for a later
algorithmic ball-recovery module if tracking-only error analysis justifies it.

## Visual recovery workstream

Place the licensed tactical video at
`data/tactical/BAR-ATH_tactical.mp4`. `visual_candidates.py` ranks short,
label-free review intervals using missing-ball runs, low emission margins,
decoded transitions and automatic restart windows. Its CSV is written under
the ignored `evaluation/diagnostics/` directory and must never be published.
The first 50 intervals currently cover about 13 minutes of video, allowing a
detector to focus on frames that can change spellF1 rather than processing the
entire 133-minute recording.

`visual_pseudolabels.py` uses only the supplied automatic ball track and
per-frame homography to create private pixel-coordinate pseudo-labels. At a
5-fps sampling rate it currently yields 18,463 projected ball frames with a
blocked temporal train/validation split. No human annotation or possession
ground truth is read. The derived manifest is also kept under the ignored
diagnostics directory.

`visual_flow_validation.py` measures bidirectional Lucas--Kanade tracking on
synthetic one-second occlusions whose true ATD ball path is hidden from the
tracker. With a 20-pixel endpoint-consistency gate, accepted windows have a
3.58-pixel median path error (9.07-pixel p90). `visual_ball_flow.py` then
applies that pre-declared gate only to real, short, endpoint-bounded gaps and
maps accepted paths back to pitch coordinates. On the current match all 43
candidate gaps fail the gate, so the experiment correctly abstains and v7 is
left unchanged. This is intentional: an unverified visual interpolation is
more harmful to spell boundaries than retaining missing evidence.

`visual_detector_dataset.py` turns the licensed MP4 into a private local-ball
detection dataset. It crops around projected ATD ball positions, adds
deterministic spatial jitter, writes YOLO point-like boxes, and uses the same
blocked temporal train/validation split. This is the training input for the
next GPU stage; generated frames remain ignored and must not be published.
Because the supplied automatic ball track contains false visible positions,
the exporter also requires a compact low-saturation bright component close to
the projected centre and shifts the weak box to that component. Frames that
fail this independent image check are discarded rather than learned as truth.
`visual_detector_train.py` trains a deterministic 160-pixel local YOLO model;
its heavier optional dependencies are isolated in `requirements-vision.txt`.
The visually verified export is still weak supervision, not ground truth: a
pretrained sports-ball model must cross-check it before fine-tuning, and v7 is
not replaced merely because a detector can be trained.

### Detector experiment status

`visual_detector_crossfilter.py` performs the independent cross-check with a
pretrained COCO sports-ball detector. It retained 676 training and 151 blocked
validation crops. A 30-epoch YOLO11n fine-tune reached precision 0.930, recall
0.881, mAP50 0.923 and mAP50--95 0.738 on that validation block.

This detector was then tested on every real short ATD gap. A conservative
confidence/coverage/trajectory/pitch gate accepted 7 of 43 gaps (42 frames).
Feeding those recovered coordinates into the v7 decoder changed four context
frames, no recovered frame's final state, and no possession transition. Thus
the detector is validated as a visual component but is **not** promoted as
v11. The next experiment searches player-centred crops near uncertain spell
boundaries instead of trusting discontinuous gap endpoints.

The first player-centred search exposed severe white-boot/grass false positives:
171 of 176 sampled missing-ball frames exceeded 0.5 confidence. To correct
this, `visual_detector_hard_negatives.py` automatically adds player crops whose
projected centres are at least 120 pixels from a trusted ball (1,351 train and
302 validation backgrounds). After eight training epochs the detector reached
0.898 precision and 0.816 recall. On the identical player-search audit only 14
frames exceed 0.5 confidence and four exceed 0.75. Production recovery therefore
uses the 0.75 threshold plus multi-frame trajectory consistency; medium-confidence
single-frame detections remain diagnostic only.
