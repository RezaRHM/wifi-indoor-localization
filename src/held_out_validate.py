"""
Held-out validation for RSSI spatial interpolation.

Test grids (withheld from training entirely):
  G5  → (12.5,  2.5) m
  G15 → (30.0,  5.0) m

The IDW model is trained on the remaining 24 grids, then evaluated at
the two held-out positions by comparing predictions to actual RSSI.
Per-anchor measurement std is reported alongside the mean, and the
IDW-interpolated std is used as an uncertainty band around each
prediction.
"""

from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\n"
        "Install with:  pip install numpy matplotlib"
    ) from exc

from interpolate import (
    parse_all_grids, parse_all_grids_stats, predict_rssi, predict_uncertainty,
    GRID_POSITIONS, ANCHOR_POSITIONS,
)

HELD_OUT = [5, 15]
OUTPUT_PATH = Path(__file__).parent / "held_out_validation.png"


def held_out_validate():
    """
    Held-out validation on grids G5, G15.

    Steps
    -----
    1. Load all grids.
    2. Remove G5, G15 from the known points.
    3. Build IDW model on the remaining grids.
    4. Predict RSSI (and uncertainty) at the 2 held-out positions.
    5. Compare predicted vs measured RSSI per anchor (incl. std).
    6. Print per-grid tables and MAE summary.
    7. Save comparison bar-chart → held_out_validation.png

    Returns
    -------
    results      : {grid_num: {rloc16: (predicted_dBm, measured_dBm)}}
    per_grid_mae : {grid_num: mae_dBm}
    overall_mae  : float
    """
    print("Loading all grids...")
    all_data  = parse_all_grids()
    all_stats = parse_all_grids_stats()

    # Split
    test_data   = {g: all_data[g]  for g in HELD_OUT if g in all_data}
    train_data  = {g: m for g, m in all_data.items()  if g not in HELD_OUT}
    test_stats  = {g: all_stats[g] for g in HELD_OUT if g in all_stats}
    train_stats = {g: m for g, m in all_stats.items() if g not in HELD_OUT}

    missing = [g for g in HELD_OUT if g not in all_data]
    if missing:
        print(f"[warn] Grids not found in data: {missing}")

    print(f"\nTraining grids : {len(train_data)}  "
          f"({sorted(train_data.keys())[0]}–{sorted(train_data.keys())[-1]}, "
          f"excluding {HELD_OUT})")
    print(f"Test grids     : {sorted(test_data.keys())}")

    # ------------------------------------------------------------------
    # Predict at each held-out position
    # ------------------------------------------------------------------
    results = {}

    for grid_num in HELD_OUT:
        if grid_num not in test_data:
            continue

        x, y      = GRID_POSITIONS[grid_num]
        predicted = predict_rssi(x, y, grid_data=train_data)
        pred_unc  = predict_uncertainty(x, y, grid_stats=train_stats)
        measured  = test_data[grid_num]
        meas_stat = test_stats.get(grid_num, {})

        # Only evaluate anchors present in both sets
        common = sorted(set(predicted) & set(measured))
        results[grid_num] = {
            rloc: {
                "pred":     predicted[rloc],
                "pred_unc": pred_unc.get(rloc, float("nan")),
                "meas":     measured[rloc],
                "std":      meas_stat.get(rloc, {}).get("std", float("nan")),
            }
            for rloc in common
        }

    # ------------------------------------------------------------------
    # Print tables
    # ------------------------------------------------------------------
    per_grid_mae = {}
    all_errors   = []

    for grid_num in HELD_OUT:
        if grid_num not in results:
            continue

        anchor_data = results[grid_num]
        rlocs       = sorted(anchor_data.keys())
        errors      = [abs(d["pred"] - d["meas"]) for d in anchor_data.values()]
        mae         = sum(errors) / len(errors) if errors else float("nan")
        per_grid_mae[grid_num] = mae
        all_errors.extend(errors)

        gx, gy = GRID_POSITIONS[grid_num]
        print(f"\n{'═'*92}")
        print(f"  Grid {grid_num}  at ({gx}, {gy}) m          MAE = {mae:.2f} dBm")
        print(f"{'═'*92}")
        print(f"  {'Anchor':<12}  {'Predicted':>10}  {'± Unc':>6}  "
              f"{'Measured':>9}  {'Std':>6}  {'Error':>7}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*9}  {'─'*6}  {'─'*7}")
        for rloc in rlocs:
            d    = anchor_data[rloc]
            err  = abs(d["pred"] - d["meas"])
            pos  = ANCHOR_POSITIONS.get(rloc)
            tag  = f" ({pos[0]:.1f},{pos[1]:.1f})" if pos else ""
            print(f"  0x{rloc:04x}{tag:<8}  {d['pred']:>10.2f}  {d['pred_unc']:>6.2f}  "
                  f"{d['meas']:>9.2f}  {d['std']:>6.2f}  {err:>7.2f}")

    overall_mae = sum(all_errors) / len(all_errors) if all_errors else float("nan")

    print(f"\n{'─'*44}")
    print("MAE per test grid:")
    for g in HELD_OUT:
        if g in per_grid_mae:
            gx, gy = GRID_POSITIONS[g]
            print(f"  Grid {g:2d}  ({gx:5.1f}, {gy:4.1f}) m  →  "
                  f"{per_grid_mae[g]:.2f} dBm")
    print(f"Overall MAE : {overall_mae:.2f} dBm")
    print(f"{'─'*44}")

    # ------------------------------------------------------------------
    # Comparison plot
    # ------------------------------------------------------------------
    _save_plot(results, per_grid_mae, OUTPUT_PATH)

    return results, per_grid_mae, overall_mae


def _save_plot(results, per_grid_mae, output_path):
    """
    Grouped bar chart: predicted vs measured RSSI per anchor,
    one panel per held-out test grid. Error bars show the IDW-interpolated
    prediction uncertainty (predicted) and the measured RSSI std (measured).
    """
    fig, axes = plt.subplots(1, len(HELD_OUT), figsize=(13, 6), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "Held-Out Validation — Predicted vs Measured RSSI\n"
        f"(trained on {26 - len(HELD_OUT)} grids, "
        f"tested on {' / '.join(f'G{g}' for g in HELD_OUT)})",
        fontsize=12,
    )

    BAR_W   = 0.35
    C_PRED  = "#4C72B0"   # blue
    C_MEAS  = "#DD8452"   # orange

    for ax, grid_num in zip(axes, HELD_OUT):
        if grid_num not in results:
            ax.set_visible(False)
            continue

        anchor_data = results[grid_num]
        rlocs  = sorted(anchor_data.keys())
        preds  = [anchor_data[r]["pred"]     for r in rlocs]
        meass  = [anchor_data[r]["meas"]     for r in rlocs]
        pred_e = [anchor_data[r]["pred_unc"] for r in rlocs]
        meas_e = [anchor_data[r]["std"]      for r in rlocs]
        labels = [f"0x{r:04x}" for r in rlocs]
        xs     = np.arange(len(rlocs))

        # Grouped bars with uncertainty error bars
        ax.bar(xs - BAR_W / 2, preds, BAR_W, yerr=pred_e, capsize=3,
               label="Predicted (± IDW std)", color=C_PRED, alpha=0.87,
               edgecolor="black", linewidth=0.5)
        ax.bar(xs + BAR_W / 2, meass, BAR_W, yerr=meas_e, capsize=3,
               label="Measured (± std)", color=C_MEAS, alpha=0.87,
               edgecolor="black", linewidth=0.5)

        # Error annotations — placed below the more-negative bar of each pair
        for i, (p, m) in enumerate(zip(preds, meass)):
            err    = abs(p - m)
            anno_y = min(p, m) - 0.8
            ax.annotate(
                f"Δ{err:.1f}",
                xy=(xs[i], anno_y),
                ha="center", va="top",
                fontsize=7, color="dimgray",
            )

        # Y-axis: zoom in to show RSSI detail
        all_vals = preds + meass
        ymin = min(all_vals) - 7
        ymax = max(all_vals) + 3
        ax.set_ylim(ymin, ymax)

        mae    = per_grid_mae.get(grid_num, float("nan"))
        gx, gy = GRID_POSITIONS[grid_num]
        ax.set_title(
            f"Grid {grid_num}  at ({gx}, {gy}) m\nMAE = {mae:.2f} dBm",
            fontsize=10,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel("RSSI (dBm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Dashed reference lines at each measured value
        for m in meass:
            ax.axhline(m, color=C_MEAS, linewidth=0.4, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    held_out_validate()
