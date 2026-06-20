"""
Log-distance path-loss model fitting and comparison.

Two models are fitted on 24 training grids (G5, G15 held out) and
evaluated against those same 2 held-out grids.

  Model 1 — Shared n :
    RSSI = TxPower_i - 10·n·log10(d)
    One global path-loss exponent n; per-anchor TxPower_i.
    Fitted as a single joint least-squares system.

  Model 2 — Per-anchor n :
    RSSI = TxPower_i - 10·n_i·log10(d)
    Independent (n_i, TxPower_i) per anchor; fitted separately.

Both models are fitted on the per-anchor mean RSSI; the per-anchor RSSI
std (measurement uncertainty) is reported alongside the fitted parameters.

Results are compared against the IDW baseline from held_out_validate.py.

Note: path-loss models and IDW now both cover the same 8 known anchors
(KNOWN_ANCHORS), so the comparison is apples-to-apples.
"""

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interpolate import parse_all_grids, parse_all_grids_stats, GRID_POSITIONS, ANCHOR_POSITIONS

# ── Constants ────────────────────────────────────────────────────────────────
HELD_OUT        = [5, 15]
IDW_OVERALL_MAE = 3.87                  # held_out_validate.py (updated after rerun)
IDW_PER_GRID    = {5: 5.59, 15: 2.15}   # same source (updated after rerun)

OUT_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_split():
    """Load all grids (mean + std); return (train_data, test_data) split."""
    print("Loading all grids...")
    all_data   = parse_all_grids_stats()
    test_data  = {g: all_data[g] for g in HELD_OUT if g in all_data}
    train_data = {g: m for g, m in all_data.items() if g not in HELD_OUT}
    missing    = [g for g in HELD_OUT if g not in all_data]
    if missing:
        print(f"  [warn] held-out grids not found: {missing}")
    return train_data, test_data


def _select_anchors(train_data):
    """
    Return sorted list of RLOC16s that have:
      - a known physical position in ANCHOR_POSITIONS, and
      - at least one measurement in the training data.
    """
    seen = {r for m in train_data.values() for r in m}
    return sorted(r for r in ANCHOR_POSITIONS if r in seen)


def _build_observations(train_data, anchor_rlocs):
    """
    Collect per-anchor (distance_m, mean_rssi_dBm, std_rssi_dBm) tuples from
    training grids. Returns {rloc: [(dist, mean_rssi, std_rssi), ...]}.
    """
    obs = {r: [] for r in anchor_rlocs}
    for grid_num, stats in train_data.items():
        gx, gy = GRID_POSITIONS[grid_num]
        for rloc in anchor_rlocs:
            if rloc not in stats:
                continue
            ax, ay = ANCHOR_POSITIONS[rloc]
            d = math.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)
            if d < 0.05:          # skip coincident points (shouldn't happen)
                continue
            obs[rloc].append((d, stats[rloc]["mean"], stats[rloc]["std"]))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Shared n (joint least-squares)
# ─────────────────────────────────────────────────────────────────────────────

def fit_model1(obs, anchor_rlocs):
    """
    Solve the joint least-squares system for all anchors simultaneously:

        RSSI_ig = TxPower_i  -  10·n·log10(d_ig)

    Design matrix row for observation (anchor i, grid g):
        [ 0 … 1 … 0 | -10·log10(d_ig) ]
               ^i-th TxPower column

    Parameters returned: (n_shared, {rloc: txpower_dBm}).
    """
    n_a   = len(anchor_rlocs)
    a_idx = {r: i for i, r in enumerate(anchor_rlocs)}
    rows, y_vals = [], []

    for rloc in anchor_rlocs:
        ai = a_idx[rloc]
        for d, rssi, _std in obs[rloc]:
            row       = np.zeros(n_a + 1)
            row[ai]   = 1.0                        # TxPower_i indicator
            row[n_a]  = -10.0 * math.log10(d)     # n coefficient
            rows.append(row)
            y_vals.append(rssi)

    if not rows:
        raise ValueError("No training observations for Model 1 fit.")

    params, *_ = np.linalg.lstsq(np.array(rows), np.array(y_vals), rcond=None)
    n_shared   = float(params[n_a])
    txpower    = {r: float(params[a_idx[r]]) for r in anchor_rlocs}
    return n_shared, txpower


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Per-anchor n (independent per-anchor least-squares)
# ─────────────────────────────────────────────────────────────────────────────

def fit_model2(obs, anchor_rlocs):
    """
    Fit each anchor independently:

        RSSI_ig = TxPower_i  -  10·n_i·log10(d_ig)

    Design matrix row: [ 1 | -10·log10(d_ig) ]

    Returns {rloc: {'n': float, 'txpower': float}}.
    """
    params = {}
    for rloc in anchor_rlocs:
        pts = obs[rloc]
        if len(pts) < 2:
            print(f"  [warn] 0x{rloc:04x}: only {len(pts)} training point(s) — skipped")
            continue
        X = np.array([[1.0, -10.0 * math.log10(d)] for d, _, _ in pts])
        y = np.array([rssi for _, rssi, _ in pts])
        p, *_ = np.linalg.lstsq(X, y, rcond=None)
        params[rloc] = {"txpower": float(p[0]), "n": float(p[1])}
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_m1(qx, qy, n_shared, txpower):
    out = {}
    for rloc, tp in txpower.items():
        ax, ay = ANCHOR_POSITIONS[rloc]
        d = max(math.sqrt((qx - ax) ** 2 + (qy - ay) ** 2), 0.1)
        out[rloc] = tp - 10.0 * n_shared * math.log10(d)
    return out


def _predict_m2(qx, qy, params_m2):
    out = {}
    for rloc, p in params_m2.items():
        ax, ay = ANCHOR_POSITIONS[rloc]
        d = max(math.sqrt((qx - ax) ** 2 + (qy - ay) ** 2), 0.1)
        out[rloc] = p["txpower"] - 10.0 * p["n"] * math.log10(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output — parameter tables
# ─────────────────────────────────────────────────────────────────────────────

def _print_params(n_shared, txpower, params_m2, anchor_rlocs, obs):
    print("\n" + "─" * 62)
    print(f"  Model 1 — global n = {n_shared:.4f}")
    print(f"\n  {'Anchor':<10}  {'TxPower (dBm)':>14}  {'Position':}")
    print(f"  {'─'*10}  {'─'*14}  {'─'*18}")
    for rloc in anchor_rlocs:
        tp  = txpower.get(rloc, float("nan"))
        pos = ANCHOR_POSITIONS[rloc]
        print(f"  0x{rloc:04x}    {tp:>14.2f}  ({pos[0]:.1f}, {pos[1]:.1f})")

    print()
    print("  Model 2 — per-anchor parameters:")
    print(f"\n  {'Anchor':<10}  {'n':>8}  {'TxPower (dBm)':>14}  {'Position':}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*14}  {'─'*18}")
    for rloc in anchor_rlocs:
        p   = params_m2.get(rloc)
        pos = ANCHOR_POSITIONS[rloc]
        if p:
            print(f"  0x{rloc:04x}    {p['n']:>8.4f}  {p['txpower']:>14.2f}  ({pos[0]:.1f}, {pos[1]:.1f})")
        else:
            print(f"  0x{rloc:04x}    {'—':>8}  {'—':>14}  (insufficient data)")
    print("─" * 62)

    print("\n  Measurement uncertainty (mean RSSI std across training grids):")
    print(f"\n  {'Anchor':<10}  {'Mean Std (dBm)':>14}")
    print(f"  {'─'*10}  {'─'*14}")
    for rloc in anchor_rlocs:
        stds = [s for _, _, s in obs[rloc]]
        mean_std = sum(stds) / len(stds) if stds else float("nan")
        print(f"  0x{rloc:04x}    {mean_std:>14.2f}")
    print("─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Output — comparison table & MAE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_print(test_data, anchor_rlocs, n_shared, txpower, params_m2):
    """
    Predict at each held-out grid with both models, print a comparison table,
    and return per-grid and overall MAEs.

    Returns
    -------
    (overall_m1, overall_m2,
     per_grid_m1: {grid: mae}, per_grid_m2: {grid: mae})
    """
    W = 88
    hdr = (f"  {'Grid':<6} {'Anchor':<10} {'Measured (dBm)':>14} "
           f"{'Model1 (shared n)':>18} {'Model2 (per-anchor n)':>21} "
           f"{'ErrM1':>7} {'ErrM2':>7}")
    print("\n" + "═" * W)
    print(hdr)
    print("─" * W)

    per_grid_m1, per_grid_m2 = {}, {}
    all_e1, all_e2 = [], []

    for g in HELD_OUT:
        if g not in test_data:
            print(f"  [warn] Grid {g} missing from test data — skipped")
            continue

        gx, gy  = GRID_POSITIONS[g]
        pm1     = _predict_m1(gx, gy, n_shared, txpower)
        pm2     = _predict_m2(gx, gy, params_m2)
        meas    = test_data[g]

        # Restrict to anchors present in both model predictions and measurement
        common  = sorted(set(pm1) & set(pm2) & set(meas))
        ge1, ge2 = [], []

        for rloc in common:
            m  = meas[rloc]["mean"]
            p1 = pm1[rloc]
            p2 = pm2[rloc]
            e1, e2 = abs(p1 - m), abs(p2 - m)
            print(f"  G{g:<5} 0x{rloc:04x}   {m:>14.2f} "
                  f"{p1:>18.2f} {p2:>21.2f} {e1:>7.2f} {e2:>7.2f}")
            ge1.append(e1); all_e1.append(e1)
            ge2.append(e2); all_e2.append(e2)

        mae1 = sum(ge1) / len(ge1) if ge1 else float("nan")
        mae2 = sum(ge2) / len(ge2) if ge2 else float("nan")
        per_grid_m1[g] = mae1
        per_grid_m2[g] = mae2

        print(f"  {'':6} {'↳ MAE':>10}  {'':>14} {mae1:>18.2f} {mae2:>21.2f}")
        print()

    ov1 = sum(all_e1) / len(all_e1) if all_e1 else float("nan")
    ov2 = sum(all_e2) / len(all_e2) if all_e2 else float("nan")

    print("═" * W)
    print("\nMAE per test grid:")
    for g in HELD_OUT:
        if g in per_grid_m1:
            idw_g = IDW_PER_GRID.get(g, float("nan"))
            print(f"  Grid {g:2d}:  Model1 = {per_grid_m1[g]:.2f} dBm  |  "
                  f"Model2 = {per_grid_m2[g]:.2f} dBm  |  "
                  f"IDW = {idw_g:.2f} dBm*")

    print(f"\nOverall MAE:")
    print(f"  Model 1 (shared n)     : {ov1:.2f} dBm")
    print(f"  Model 2 (per-anchor n) : {ov2:.2f} dBm")
    print(f"  IDW* (reference)       : {IDW_OVERALL_MAE:.2f} dBm")
    print(f"\n  * IDW and path-loss models both cover the same "
          f"{len(anchor_rlocs)} known anchors.")

    return ov1, ov2, per_grid_m1, per_grid_m2


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — MAE comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mae_comparison(ov1, ov2, pg_m1, pg_m2):
    """Save path_loss_comparison.png — grouped MAE bar chart."""
    group_labels = [f"G{g}" for g in HELD_OUT] + ["Overall"]
    m1_vals  = [pg_m1.get(g, float("nan")) for g in HELD_OUT] + [ov1]
    m2_vals  = [pg_m2.get(g, float("nan")) for g in HELD_OUT] + [ov2]
    idw_vals = [IDW_PER_GRID.get(g, float("nan")) for g in HELD_OUT] + [IDW_OVERALL_MAE]

    xs = np.arange(len(group_labels))
    w  = 0.26

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    b1 = ax.bar(xs - w,   m1_vals,  w, label="Model 1 — shared n",
                color="#4C72B0", alpha=0.88, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(xs,       m2_vals,  w, label="Model 2 — per-anchor n",
                color="#55A868", alpha=0.88, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(xs + w,   idw_vals, w,
                label="IDW baseline (all 14 RLOC16s)*",
                color="#DD8452", alpha=0.88, edgecolor="black", linewidth=0.5)

    # Value labels above each bar
    for bars, vals in [(xs - w, m1_vals), (xs, m2_vals), (xs + w, idw_vals)]:
        for bx, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bx, v + 0.12, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

    # Separator before "Overall" group
    ax.axvline(len(HELD_OUT) - 0.5, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("MAE (dBm)", fontsize=10)
    ax.set_title(
        "Held-Out Validation — MAE Comparison\n"
        "Model 1 (shared n)  vs  Model 2 (per-anchor n)  vs  IDW baseline",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    valid = [v for v in m1_vals + m2_vals + idw_vals if not math.isnan(v)]
    ax.set_ylim(0, max(valid) * 1.22 if valid else 10)

    ax.text(0.01, 0.02,
            "* IDW MAE includes RLOC16s with unknown positions",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

    plt.tight_layout()
    out = OUT_DIR / "path_loss_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Per-anchor path-loss scatter + model curves
# ─────────────────────────────────────────────────────────────────────────────

def _plot_path_loss_fits(obs, anchor_rlocs, n_shared, txpower, params_m2, test_data):
    """Save path_loss_fits.png — 9-panel scatter + model curves."""
    ncols = 3
    nrows = math.ceil(len(anchor_rlocs) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 6.5, nrows * 4.8),
                             squeeze=False)
    fig.suptitle(
        "Path-Loss Fits per Anchor\n"
        "● Training  ─ Model 1 (shared n)  ── Model 2 (per-anchor n)  ★ Held-out test",
        fontsize=11,
    )
    axes_flat = [ax for row in axes for ax in row]

    for idx, rloc in enumerate(anchor_rlocs):
        ax       = axes_flat[idx]
        anc_pos  = ANCHOR_POSITIONS[rloc]

        # ── Training scatter ─────────────────────────────────────────────
        train_d    = [pt[0] for pt in obs[rloc]]
        train_rssi = [pt[1] for pt in obs[rloc]]
        train_std  = [pt[2] for pt in obs[rloc]]
        if train_d:
            ax.errorbar(train_d, train_rssi, yerr=train_std, fmt="o",
                         c="#4C72B0", ms=5, zorder=4, alpha=0.75,
                         ecolor="#4C72B0", elinewidth=0.8, capsize=2,
                         markeredgecolor="white", markeredgewidth=0.4,
                         label="Training (± std)")

        # ── Distance range for model curves ──────────────────────────────
        all_d = list(train_d)
        for g in HELD_OUT:
            if g in test_data and rloc in test_data[g]:
                anc_x, anc_y = ANCHOR_POSITIONS[rloc]
                gx, gy       = GRID_POSITIONS[g]
                all_d.append(math.sqrt((gx - anc_x) ** 2 + (gy - anc_y) ** 2))

        if all_d:
            d_lo    = max(0.3, min(all_d) * 0.75)
            d_hi    = max(all_d) * 1.25
            d_curve = np.linspace(d_lo, d_hi, 300)

            # Model 1 curve
            tp1 = txpower.get(rloc)
            if tp1 is not None:
                y_m1 = [tp1 - 10.0 * n_shared * math.log10(d) for d in d_curve]
                ax.plot(d_curve, y_m1, color="#C44E52", lw=2.0,
                        label=f"M1  n={n_shared:.2f}")

            # Model 2 curve
            p2 = params_m2.get(rloc)
            if p2:
                y_m2 = [p2["txpower"] - 10.0 * p2["n"] * math.log10(d)
                        for d in d_curve]
                ax.plot(d_curve, y_m2, color="#27AE60", lw=2.0,
                        linestyle="--",
                        label=f"M2  n={p2['n']:.2f}")

        # ── Held-out test points ──────────────────────────────────────────
        markers  = {5: "*", 15: "P"}
        marker_sz= {5: 220,  15: 140}
        for g in HELD_OUT:
            if g not in test_data or rloc not in test_data[g]:
                continue
            anc_x, anc_y = ANCHOR_POSITIONS[rloc]
            gx, gy       = GRID_POSITIONS[g]
            d_test        = math.sqrt((gx - anc_x) ** 2 + (gy - anc_y) ** 2)
            rssi_test     = test_data[g][rloc]["mean"]
            ax.scatter([d_test], [rssi_test],
                       marker=markers[g], c="#FF7F0E",
                       s=marker_sz[g], zorder=7,
                       edgecolors="black", linewidths=0.5,
                       label=f"G{g} (held-out)")

        ax.set_title(f"0x{rloc:04x}  @  ({anc_pos[0]:.1f}, {anc_pos[1]:.1f}) m",
                     fontsize=9)
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.set_ylabel("RSSI (dBm)",   fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=7)

    # Hide spare cells
    for idx in range(len(anchor_rlocs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "path_loss_fits.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_path_loss_analysis():
    """
    Full pipeline:
      load → fit Model 1 + Model 2 → print params → evaluate → print table → plot.

    Returns (overall_mae_model1, overall_mae_model2).
    """
    train_data, test_data = _load_split()
    anchor_rlocs          = _select_anchors(train_data)
    obs                   = _build_observations(train_data, anchor_rlocs)

    print(f"\n{'─'*54}")
    print(f"  Training grids  : {len(train_data)}  "
          f"(grids {HELD_OUT} held out)")
    print(f"  Anchors modelled: {len(anchor_rlocs)}  "
          f"(those with known positions)")
    print(f"  Total training observations:")
    for rloc in anchor_rlocs:
        print(f"    0x{rloc:04x}  {len(obs[rloc]):3d} pts  "
              f"d ∈ [{min(d for d,_,_ in obs[rloc]):.1f}, "
              f"{max(d for d,_,_ in obs[rloc]):.1f}] m")
    print("─" * 54)

    # ── Fit ───────────────────────────────────────────────────────────────
    print("\nFitting Model 1 (shared n, joint least-squares)...")
    n_shared, txpower = fit_model1(obs, anchor_rlocs)

    print("Fitting Model 2 (per-anchor n, independent least-squares)...")
    params_m2 = fit_model2(obs, anchor_rlocs)

    # ── Print parameters ─────────────────────────────────────────────────
    _print_params(n_shared, txpower, params_m2, anchor_rlocs, obs)

    # ── Evaluate & print table ────────────────────────────────────────────
    ov1, ov2, pg_m1, pg_m2 = evaluate_and_print(
        test_data, anchor_rlocs, n_shared, txpower, params_m2
    )

    # ── Save plots ────────────────────────────────────────────────────────
    print()
    _plot_mae_comparison(ov1, ov2, pg_m1, pg_m2)
    _plot_path_loss_fits(obs, anchor_rlocs, n_shared, txpower, params_m2, test_data)

    return ov1, ov2


if __name__ == "__main__":
    run_path_loss_analysis()
