"""
Predict RSSI at an arbitrary (x, y) position using IDW interpolation.

Usage
-----
    python predict_cli.py <x> <y>

Example
-------
    python predict_cli.py 12.5 4.0
"""

import sys
import math

from interpolate import (
    predict_rssi,
    parse_all_grids,
    GRID_POSITIONS,
    ANCHOR_POSITIONS,
)


def _closest_grid(x, y):
    """Return (grid_num, distance_m) for the nearest measurement grid point."""
    best_num, best_dist = min(
        ((g, math.sqrt((x - gx) ** 2 + (y - gy) ** 2))
         for g, (gx, gy) in GRID_POSITIONS.items()),
        key=lambda t: t[1],
    )
    return best_num, best_dist


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
    except ValueError:
        print("Error: x and y must be numeric values.")
        sys.exit(1)

    print("Loading grid data...")
    grid_data = parse_all_grids()

    if not grid_data:
        print("No grid data found — check that ../records/grid*.pcapng files exist.")
        sys.exit(1)

    print(f"\nQuery position : ({x:.3f}, {y:.3f}) m")

    grid_num, dist = _closest_grid(x, y)
    gx, gy = GRID_POSITIONS[grid_num]
    print(f"Closest grid   : #{grid_num}  at ({gx:.2f}, {gy:.2f}) m  "
          f"[Δ = {dist:.2f} m]")

    predictions = predict_rssi(x, y, grid_data=grid_data)
    if not predictions:
        print("No predictions available (no RSSI data in parsed files).")
        sys.exit(1)

    # Sort strongest → weakest signal
    ranked = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\n{'Rank':<5} {'RLOC16':<10} {'Anchor (x, y)':<18} {'Predicted RSSI':>16}")
    print("─" * 53)
    for rank, (rloc, rssi) in enumerate(ranked, start=1):
        pos = ANCHOR_POSITIONS.get(rloc)
        pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if pos else "unknown"
        print(f"  {rank:<3}  0x{rloc:04x}    {pos_str:<18}  {rssi:>12.2f} dBm")


if __name__ == "__main__":
    main()
