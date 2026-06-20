"""
Generate per-anchor RSSI heatmaps over the building footprint using IDW
interpolation, then save as rssi_heatmaps.png in this directory.
"""

import math

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # non-interactive, file output only
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\n"
        "Install with:  pip install numpy matplotlib"
    ) from exc

from pathlib import Path
from interpolate import GRID_POSITIONS, ANCHOR_POSITIONS, parse_all_grids

OUTPUT_PATH = Path(__file__).parent / "rssi_heatmaps.png"

# Building extent
X_MIN, X_MAX = 0.0, 35.0
Y_MIN, Y_MAX = 0.0, 8.1


def _idw_heatmap(known_pts, xs, ys, power=2):
    """
    Vectorised IDW over a meshgrid defined by xs × ys.

    known_pts : list of (x, y, value)
    Returns   : Z array of shape (len(ys), len(xs))
    """
    if not known_pts:
        return np.full((len(ys), len(xs)), np.nan)

    px = np.array([p[0] for p in known_pts])
    py = np.array([p[1] for p in known_pts])
    pv = np.array([p[2] for p in known_pts])

    X, Y = np.meshgrid(xs, ys)
    Xf, Yf = X.ravel(), Y.ravel()

    # dists: (n_query, n_known)
    dists = np.sqrt((Xf[:, None] - px[None, :]) ** 2 +
                    (Yf[:, None] - py[None, :]) ** 2)
    # Clamp to tiny value so exact grid points get their own measurement
    dists = np.maximum(dists, 1e-9)

    weights = 1.0 / dists ** power
    Z_flat  = (weights * pv[None, :]).sum(axis=1) / weights.sum(axis=1)
    return Z_flat.reshape(len(ys), len(xs))


def generate_heatmaps(output_path=None, power=2, resolution=0.25):
    """
    Compute and save an RSSI heatmap for every anchor found in the data.

    Parameters
    ----------
    output_path : str or Path — defaults to rssi_heatmaps.png beside this file
    power       : float       — IDW exponent
    resolution  : float       — grid step in metres (smaller = finer / slower)
    """
    if output_path is None:
        output_path = OUTPUT_PATH

    print("Parsing grid data...")
    grid_data = parse_all_grids()

    all_rlocs = sorted({rloc for m in grid_data.values() for rloc in m})
    if not all_rlocs:
        print("No anchor data found — nothing to plot.")
        return

    xs = np.arange(X_MIN, X_MAX + resolution, resolution)
    ys = np.arange(Y_MIN, Y_MAX + resolution, resolution)

    ncols = 3
    nrows = math.ceil(len(all_rlocs) / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 9, nrows * 5.5),
                             squeeze=False)
    fig.suptitle("RSSI Heatmaps per Anchor — IDW Interpolation", fontsize=14, y=1.01)

    axes_flat = [ax for row in axes for ax in row]

    # Precompute all grid marker locations
    gx_all = np.array([GRID_POSITIONS[g][0] for g in sorted(GRID_POSITIONS)])
    gy_all = np.array([GRID_POSITIONS[g][1] for g in sorted(GRID_POSITIONS)])

    for idx, rloc in enumerate(all_rlocs):
        ax = axes_flat[idx]

        known_pts = [
            (GRID_POSITIONS[g][0], GRID_POSITIONS[g][1], m[rloc])
            for g, m in grid_data.items() if rloc in m
        ]

        print(f"  Anchor 0x{rloc:04x}: {len(known_pts)} grid points...")
        Z = _idw_heatmap(known_pts, xs, ys, power=power)

        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))

        # Filled contour heatmap
        cf = ax.contourf(xs, ys, Z, levels=25, cmap="RdYlGn",
                         vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("RSSI (dBm)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # All grid measurement positions
        ax.scatter(gx_all, gy_all, c="white", s=18, zorder=5,
                   edgecolors="black", linewidths=0.5)
        for gnum in sorted(GRID_POSITIONS):
            gx, gy = GRID_POSITIONS[gnum]
            ax.annotate(str(gnum), (gx, gy),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=5, ha="center", color="white", fontweight="bold",
                        zorder=6)

        # All anchor physical positions
        for a_rloc, a_pos in ANCHOR_POSITIONS.items():
            is_self = (a_rloc == rloc)
            ax.scatter(*a_pos,
                       c="red" if is_self else "orange",
                       s=220 if is_self else 55,
                       marker="*" if is_self else "D",
                       zorder=8,
                       edgecolors="black", linewidths=0.6)
            if is_self:
                ax.annotate(f"0x{a_rloc:04x}", a_pos,
                            xytext=(4, -4), textcoords="offset points",
                            fontsize=7, color="red", fontweight="bold", zorder=9)

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_title(f"Anchor  0x{rloc:04x}   "
                     f"[{vmin:.1f} … {vmax:.1f} dBm]", fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

    # Hide any unused subplot cells
    for idx in range(len(all_rlocs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color="none", label=""),   # spacer
        plt.scatter([], [], c="white", s=18, edgecolors="black", linewidths=0.5,
                    label="Grid positions"),
        plt.scatter([], [], c="red",   s=150, marker="*", edgecolors="black",
                    label="This anchor"),
        plt.scatter([], [], c="orange",s=55,  marker="D", edgecolors="black",
                    label="Other anchors"),
    ]
    fig.legend(handles=legend_handles[1:], loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    generate_heatmaps()
