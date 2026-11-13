# Phase 1 candidate freeze manifest

Generated and format-validated on 13 November 2026.

| Candidate | SHA-256 | Full-match possessions | Role |
|---|---|---:|---|
| `Mes_Que_Data_XI.csv` | `84a7e8701a74d4e700d2d2ce9d769c0422719bed8ddb0f7d2c093e3e186b346c` | 480 | v15 + top-50 visual, primary candidate |
| `Mes_Que_Data_XI_v7_fallback.csv` | `af5d648a37db6051a4eadf519687393d228387efcac503d0ba6e730e3b469b68` | 597 | externally calibrated, lower-leakage fallback |

Both files contain exactly 199,319 rows, frames 0–199,318, no missing values,
and only `H/A/D` labels. CSV files remain gitignored because full-match
submissions are private competition artifacts.

The primary candidate has public macroF1 0.6777 and spellF1 0.4476. The v7
fallback has 0.6377 and 0.3739. This public difference is not decisive: a
10,000-draw paired possession bootstrap gives a v15-minus-smooth125 interval
of −0.1160 to +0.1628. External rolling validation shows that v15 reduces
fragmentation with a small loss in event-anchor accuracy and switch coverage.

Top-50 visual expansion adds one independently verified 11-point trajectory.
It changes 59 private frames from D to A and changes no transition. The public
window is unchanged. Do not overwrite either frozen CSV without updating its
hash and rerunning `evaluation/validate_submission.py`.

