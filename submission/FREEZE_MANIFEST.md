# Phase 1 candidate freeze manifest

Generated and format-validated on 14 November 2026.

| Candidate | SHA-256 | Full-match possessions | Role |
|---|---|---:|---|
| `Mes_Que_Data_XI.csv` | `38f3b537f637ae645844d5d851ece6b68172382c2dd42741d5bc1d124a65d576` | 480 | externally calibrated v15-core, reproducible primary |
| `Mes_Que_Data_XI_v7_fallback.csv` | `af5d648a37db6051a4eadf519687393d228387efcac503d0ba6e730e3b469b68` | 597 | externally calibrated, lower-leakage fallback |
| `Mes_Que_Data_XI_visual_pending.csv` | `a5d71c9018b4a5755dbc81f4dee9f57ba3e11c3973d2a4ba85bf193429cc3b04` | 480 | local only; not submitted pending checkpoint permission |

Both files contain exactly 199,319 rows, frames 0–199,318, no missing values,
and only `H/A/D` labels. CSV files remain gitignored because full-match
submissions are private competition artifacts.

The primary model and its constants are selected from external Metrica event
proxies. External rolling validation shows that v15-core reduces fragmentation
with a small loss in event-anchor accuracy and switch coverage. Public labels
are not used to choose between the primary and fallback.

Top-100 visual expansion adds three algorithmically accepted trajectories beyond
top-25. They change 180 private frames (70 D-to-H and 110 D-to-A) and change no
transition. A forced local rebuild reproduced its CSV byte-for-byte, but its
checkpoint is not published pending clarification of the data licence, so it
is not the formal submission. Do not overwrite a frozen CSV without updating
its hash and rerunning `evaluation/validate_submission.py`.

