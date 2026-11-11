"""Conservative experimental bridge for short ball-track occlusions.

This module is deliberately not part of the main submission yet. A missing
ball may mean either an occlusion during live play or a genuine stoppage. We
bridge only a short missing run whose two sides have the same decoded team and
strong control evidence on both sides. External validation is reported before
this is allowed into a production candidate.
"""

from __future__ import annotations

import numpy as np

from sequence_v4 import runs


def bridge_occlusions(
    states: np.ndarray,
    dh: np.ndarray,
    da: np.ndarray,
    home_anchor: np.ndarray,
    away_anchor: np.ndarray,
    max_gap: int = 75,
    flank: int = 25,
    min_anchor: float = 0.30,
):
    out = states.copy()
    missing = ~(np.isfinite(dh) & np.isfinite(da))
    starts, ends = runs(missing.astype(np.uint8))
    bridged = []
    for start, end in zip(starts, ends):
        if not missing[start] or start == 0 or end == len(out):
            continue
        if end - start > max_gap:
            continue
        side = int(out[start - 1])
        if side >= 2 or int(out[end]) != side:
            continue
        anchor = home_anchor if side == 0 else away_anchor
        left = float(np.max(anchor[max(0, start - flank):start]))
        right = float(np.max(anchor[end:min(len(anchor), end + flank)]))
        if min(left, right) < min_anchor:
            continue
        out[start:end] = side
        bridged.append((int(start), int(end), side, left, right))
    return out, bridged
