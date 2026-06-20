"""
Ablation study — feature subset comparison for grid-stratified 7-fold
fingerprinting CV (RandomForest only).

Each window's full feature vector is 8 anchors x 4 stats (mean, std,
median, count) = 32 features.  This script re-runs the same
grid-stratified 7-fold CV with reduced feature subsets to measure how
much each statistic contributes:

  A: [mean]                       ->  8 features
  B: [mean, median]               -> 16 features
  C: [mean, std]                  -> 16 features
  D: [mean, count]                -> 16 features
  E: [mean, std, median, count]   -> 32 features (current / full)

Save: ablation_results.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localize import (
    RECORDS_DIR, N_GRIDS, N_FOLDS, WINDOW_SIZE, STATS,
    build_dataset, make_grid_stratified_folds, kfold_evaluate,
)

OUT_DIR = Path(__file__).parent

STAT_IDX = {s: i for i, s in enumerate(STATS)}  # mean=0 std=1 median=2 count=3

SUBSETS = {
    "A": ["mean"],
    "B": ["mean", "median"],
    "C": ["mean", "std"],
    "D": ["mean", "count"],
    "E": ["mean", "std", "median", "count"],
}


def _columns(stats_subset, n_anchors=8):
    return [a * len(STATS) + STAT_IDX[s] for a in range(n_anchors) for s in stats_subset]


def run_ablation():
    print(f"Parsing pcapng files (window_size = {WINDOW_SIZE})...")
    X, y, anchors, wpg = build_dataset(RECORDS_DIR, N_GRIDS, WINDOW_SIZE)
    print(f"Dataset: {X.shape[0]} windows x {X.shape[1]} features "
          f"({len(anchors)} anchors x {len(STATS)} stats)")

    folds = make_grid_stratified_folds(y, N_FOLDS)

    results = {}
    for name, stats_subset in SUBSETS.items():
        cols = _columns(stats_subset)
        Xs = X[:, cols]
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        _, _, fold_accs = kfold_evaluate(clf, Xs, y, folds)
        results[name] = {
            "n_features": len(cols),
            "stats": stats_subset,
            "mean": float(np.mean(fold_accs)),
            "std": float(np.std(fold_accs)),
            "fold_accs": fold_accs,
        }

    full_acc = results["E"]["mean"]

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  ABLATION STUDY — Feature Subset Comparison")
    print("  RandomForest, Grid-Stratified 7-Fold CV")
    print(SEP)
    hdr = f"  {'Subset':<8}{'Features':>10}{'Accuracy':>12}{'Std':>8}{'vs_full':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ["A", "B", "C", "D", "E"]:
        r = results[name]
        if name == "E":
            vs_str = "  (full)"
        else:
            vs_full = (r["mean"] - full_acc) * 100
            vs_str = f"{vs_full:+.1f}%"
        print(f"  {name:<8}{r['n_features']:>8}  {r['mean']*100:>9.1f}%"
              f"{r['std']*100:>7.1f}%{vs_str:>11}")
    print(SEP)

    chance = 100.0 / N_GRIDS
    print(f"  Chance level : {chance:.1f}%")

    best = max(results, key=lambda n: results[n]["mean"])
    print(f"  Best subset  : {best} ({results[best]['mean']*100:.1f}%)")

    _plot_ablation(results, chance)

    return results


def _plot_ablation(results, chance):
    names = ["A", "B", "C", "D", "E"]
    means = [results[n]["mean"] * 100 for n in names]
    stds  = [results[n]["std"]  * 100 for n in names]
    feats = [results[n]["n_features"] for n in names]
    labels = [f"{n}\n({'+'.join(results[n]['stats'])})\n{f}f"
              for n, f in zip(names, feats)]

    colors = ["#5DADE2"] * 4 + ["#27AE60"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="black", width=0.6)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1.5,
                f"{m:.1f}±{s:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(chance, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({chance:.1f}%)")
    ax.axhline(results["E"]["mean"] * 100, color="#27AE60", linestyle=":",
               linewidth=1.2, label=f"Full (E) = {results['E']['mean']*100:.1f}%")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Accuracy ± Std (%)")
    ax.set_title("Ablation Study — Feature Subset Comparison\n"
                  "RandomForest, Grid-Stratified 7-Fold CV")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out = OUT_DIR / "ablation_results.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    run_ablation()
