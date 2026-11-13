# Phase 1 candidate freeze manifest

Generated and format-validated on 13 November 2026.

| Candidate | SHA-256 | Full-match possessions | Role |
|---|---|---:|---|
| `Mes_Que_Data_XI.csv` | `a5d71c9018b4a5755dbc81f4dee9f57ba3e11c3973d2a4ba85bf193429cc3b04` | 480 | v15 + top-100 visual, primary candidate |
| `Mes_Que_Data_XI_v7_fallback.csv` | `af5d648a37db6051a4eadf519687393d228387efcac503d0ba6e730e3b469b68` | 597 | externally calibrated, lower-leakage fallback |

Both files contain exactly 199,319 rows, frames 0–199,318, no missing values,
and only `H/A/D` labels. CSV files remain gitignored because full-match
submissions are private competition artifacts.

The primary candidate has public macroF1 0.6777 and spellF1 0.4476. The v7
fallback has 0.6377 and 0.3739. This public difference is not decisive: a
10,000-draw paired possession bootstrap gives a v15-minus-smooth125 interval
of −0.1160 to +0.1628. External rolling validation shows that v15 reduces
fragmentation with a small loss in event-anchor accuracy and switch coverage.

Top-100 visual expansion adds three independently verified trajectories beyond
top-25. They change 180 private frames (70 D-to-H and 110 D-to-A) and change no
transition. The public window is unchanged. Do not overwrite either frozen CSV
without updating its hash and rerunning `evaluation/validate_submission.py`.

