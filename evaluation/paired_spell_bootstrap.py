"""Paired possession bootstrap for the spellF1 difference between two files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import possession_metrics as pm


def bootstrap_difference(gt, first, second, draws=10000, seed=2027):
    valid = pm.valid_runs(gt)
    gt_spells = pm.possession_spells(gt, valid)
    n_gt = len(gt_spells)
    rng = np.random.RandomState(seed)

    def matched(pred):
        pred_spells = pm.possession_spells(pred, valid)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        flags = np.vstack([pm.matched_gt_flags(overlaps, n_gt, threshold)
                           for threshold in pm.SPELL_IOUS])
        return flags, len(pred_spells)

    first_flags, first_n = matched(first)
    second_flags, second_n = matched(second)

    def score(flags, n_pred, picks):
        values = []
        for row in flags:
            tp = int(row[picks].sum())
            fn = n_gt - tp
            fp = max(0, n_pred - tp)
            values.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return float(np.mean(values))

    differences = np.empty(draws)
    for draw in range(draws):
        picks = rng.randint(0, n_gt, n_gt)
        differences[draw] = (score(first_flags, first_n, picks) -
                              score(second_flags, second_n, picks))
    return {
        "mean": float(np.mean(differences)),
        "low": float(np.percentile(differences, 5)),
        "high": float(np.percentile(differences, 95)),
        "probability_positive": float(np.mean(differences > 0)),
        "draws": draws,
        "possessions": n_gt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--gt", type=Path,
                        default=Path(__file__).parent / "gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    gt = pm.label_codes(pm.read_label_csv(str(args.gt)))
    first = pm.label_codes(pm.read_label_csv(str(args.first)))
    second = pm.label_codes(pm.read_label_csv(str(args.second)))
    result = bootstrap_difference(gt, first, second, args.draws)
    print("{} minus {}".format(args.first.name, args.second.name))
    print("paired spellF1 difference mean={mean:+.4f}, 5-95%=[{low:+.4f}, {high:+.4f}]".format(**result))
    print("P(difference > 0)={:.1%}; {} draws over {} possessions".format(
        result["probability_positive"], result["draws"], result["possessions"]))


if __name__ == "__main__":
    main()
