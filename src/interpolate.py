"""
Core module for RSSI spatial interpolation.

Parses IEEE 802.15.4 TAP pcapng files (link type 283) without
external tools — pure Python struct-based parsing only.
"""

import struct
import math
from pathlib import Path
from collections import defaultdict

# Number of measurement grids (G27, G28 removed — building height is 8.1 m)
N_GRIDS = 26

# Grid measurement positions (x, y in meters)
GRID_POSITIONS = {
    1:  (2.5,  2.5),  2:  (5.0,  2.5),  3:  (7.5,  2.5),  4:  (10.0, 2.5),
    5:  (12.5, 2.5),  6:  (15.0, 2.5),  7:  (17.5, 2.5),  8:  (20.0, 2.5),
    9:  (22.53,2.5),  10: (25.0, 2.5),  11: (27.5, 2.5),  12: (30.0, 2.5),
    13: (33.0, 2.5),  14: (33.0, 5.0),  15: (30.0, 5.0),  16: (27.5, 5.0),
    17: (25.0, 5.0),  18: (22.5, 5.0),  19: (20.0, 5.0),  20: (17.5, 5.0),
    21: (15.0, 5.0),  22: (12.5, 5.0),  23: (10.0, 5.0),  24: (7.5,  5.0),
    25: (5.0,  5.0),  26: (2.5,  5.0),
}

# Known anchor RLOC16 → physical position (x, y in meters)
KNOWN_ANCHORS = {0xac00, 0x4400, 0x1800, 0xa800,
                 0x5800, 0xe400, 0xb000, 0xa000}

ANCHOR_POSITIONS = {
    0xac00: (0.9,  0.0),
    0x4400: (5.4,  0.0),
    0x1800: (15.3, 0.0),
    0xa800: (10.4, 8.1),
    0x5800: (20.6, 8.1),
    0xe400: (22.6, 0.0),
    0xb000: (25.0, 0.0),
    0xa000: (10.5, 0.0),
}

# Per-anchor RSSI statistics computed by extract_anchor_stats()
STATS = ["mean", "std"]

# pcapng files are at data/records/ relative to the repo root
DATA_DIR = Path(__file__).parent.parent / "data" / "records"


# ---------------------------------------------------------------------------
# pcapng / TAP / 802.15.4 parsing
# ---------------------------------------------------------------------------

def _parse_pcapng(filepath):
    """
    Parse one pcapng file (IEEE 802.15.4 TAP, link type 283).
    Returns list of (rssi_dBm: float, rloc16: int) for every frame that
    passes the router RLOC16 filter.
    """
    with open(filepath, "rb") as fh:
        raw = fh.read()

    if len(raw) < 12:
        return []

    # Determine byte order from the SHB byte-order magic at raw[8:12].
    # SHB layout: block_type(4) + block_total_len(4) + byte_order_magic(4) + ...
    bom = struct.unpack_from("<I", raw, 8)[0]
    endian = "<" if bom == 0x1A2B3C4D else ">"

    results = []
    offset = 0

    while offset + 12 <= len(raw):
        block_type, block_len = struct.unpack_from(endian + "II", raw, offset)

        if block_len < 12 or offset + block_len > len(raw):
            break  # truncated or corrupt block

        # Enhanced Packet Block (type 6)
        if block_type == 0x00000006 and offset + 28 <= len(raw):
            # EPB body: iface_id(4)+ts_hi(4)+ts_lo(4)+cap_len(4)+orig_len(4) = 20 bytes
            cap_len = struct.unpack_from(endian + "I", raw, offset + 20)[0]
            pkt_start = offset + 28
            if pkt_start + cap_len <= len(raw):
                rssi, rloc16 = _parse_tap_and_frame(raw[pkt_start: pkt_start + cap_len])
                if rssi is not None and rloc16 is not None:
                    results.append((rssi, rloc16))

        offset += block_len

    return results


def _parse_tap_and_frame(packet):
    """
    Parse one IEEE 802.15.4 TAP packet:
      - TAP header TLVs  → extract RSSI (TLV type 1, float32 LE)
      - 802.15.4 frame   → extract source short address (RLOC16)

    Returns (rssi_dBm, rloc16) or (None, None) on failure / filter mismatch.
    """
    if len(packet) < 4:
        return None, None

    # TAP header: version(1) + reserved(1) + length(2 LE)
    tap_len = struct.unpack_from("<H", packet, 2)[0]
    if tap_len > len(packet):
        return None, None

    # Walk TLVs inside the TAP header
    rssi = None
    pos = 4
    while pos + 4 <= tap_len:
        tlv_type, tlv_len = struct.unpack_from("<HH", packet, pos)
        val_start = pos + 4

        if val_start + tlv_len > len(packet):
            break

        if tlv_type == 1 and tlv_len == 4:          # RSSI: float32 LE
            rssi = struct.unpack_from("<f", packet, val_start)[0]

        # TLV value field is padded to a 4-byte boundary
        padded_val = tlv_len + (4 - tlv_len % 4) % 4
        pos += 4 + padded_val

    if rssi is None:
        return None, None

    # 802.15.4 frame starts immediately after the TAP header
    rloc16 = _src_short_addr(packet, tap_len)

    # Keep only router RLOC16s: non-zero address, lower 10 bits all zero
    if rloc16 is not None and rloc16 != 0x0000 and (rloc16 & 0x03FF) == 0:
        return rssi, rloc16

    return rssi, None


def _src_short_addr(data, offset):
    """
    Parse an IEEE 802.15.4 frame header starting at `offset` and return
    the source short address (2-byte RLOC16), or None if unavailable.

    Supports frame versions 0/1 (802.15.4-2003/2006) and the common
    addressing patterns used by OpenThread (version 2 / 2015).
    """
    if offset + 3 > len(data):
        return None

    fc = struct.unpack_from("<H", data, offset)[0]
    pan_compress = (fc >> 6) & 1
    dst_mode     = (fc >> 10) & 3   # 0=none 2=short 3=extended
    src_mode     = (fc >> 14) & 3

    pos = offset + 3  # skip frame_control(2) + sequence_number(1)

    # Destination addressing fields
    if dst_mode != 0:
        pos += 2  # dest PAN ID
        if dst_mode == 2:
            pos += 2   # 16-bit dest addr
        elif dst_mode == 3:
            pos += 8   # 64-bit dest addr

    # Source PAN is omitted when PAN compression is active and a dest is present
    if src_mode != 0 and not (pan_compress and dst_mode != 0):
        pos += 2  # src PAN ID

    if src_mode == 2:
        if pos + 2 > len(data):
            return None
        return struct.unpack_from("<H", data, pos)[0]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_all_grids():
    """
    Parse all N_GRIDS grid pcapng files, restricted to the known anchors.

    Returns
    -------
    dict  {grid_num (int): {rloc16 (int): mean_rssi_dBm (float)}}
    """
    result = {}
    for grid_num in range(1, N_GRIDS + 1):
        path = DATA_DIR / f"grid{grid_num}.pcapng"
        if not path.exists():
            print(f"  [warn] {path.name} not found, skipping grid {grid_num}")
            continue

        frames = _parse_pcapng(str(path))

        by_rloc = defaultdict(list)
        for rssi, rloc in frames:
            if rloc in KNOWN_ANCHORS:
                by_rloc[rloc].append(rssi)

        result[grid_num] = {
            rloc: sum(vals) / len(vals)
            for rloc, vals in by_rloc.items()
        }
        print(f"  Grid {grid_num:2d}: {len(frames):5d} frames, "
              f"{len(result[grid_num])} anchors seen")

    return result


def parse_all_grids_v2():
    """
    Parse all N_GRIDS grid pcapng files into per-anchor RSSI mean, std,
    and sample count, restricted to the known anchors.

    If an anchor has only 1 sample, std = 0.0 and count = 1.

    Returns
    -------
    dict  {grid_num (int): {rloc16 (int): {"mean": float, "std": float, "count": int}}}
    """
    result = {}
    for grid_num in range(1, N_GRIDS + 1):
        path = DATA_DIR / f"grid{grid_num}.pcapng"
        if not path.exists():
            print(f"  [warn] {path.name} not found, skipping grid {grid_num}")
            continue

        frames = _parse_pcapng(str(path))

        by_rloc = defaultdict(list)
        for rssi, rloc in frames:
            if rloc in KNOWN_ANCHORS:
                by_rloc[rloc].append(rssi)

        grid_result = {}
        for rloc, vals in by_rloc.items():
            n    = len(vals)
            mean = sum(vals) / n
            if n == 1:
                std = 0.0
            else:
                var = sum((v - mean) ** 2 for v in vals) / n
                std = math.sqrt(var)
            grid_result[rloc] = {"mean": mean, "std": std, "count": n}

        result[grid_num] = grid_result

    return result


def extract_anchor_stats(path):
    """
    Parse one grid pcapng file and compute per-anchor RSSI statistics,
    restricted to the known anchors.

    Returns
    -------
    dict  {rloc16 (int): {"mean": float, "std": float}}
    """
    frames = _parse_pcapng(str(path))

    by_rloc = defaultdict(list)
    for rssi, rloc in frames:
        if rloc in KNOWN_ANCHORS:
            by_rloc[rloc].append(rssi)

    stats = {}
    for rloc, vals in by_rloc.items():
        n    = len(vals)
        mean = sum(vals) / n
        var  = sum((v - mean) ** 2 for v in vals) / n
        all_stats = {
            "mean":   mean,
            "std":    math.sqrt(var),
        }
        stats[rloc] = {s: all_stats[s] for s in STATS}
    return stats


def parse_all_grids_stats():
    """
    Parse all N_GRIDS grid pcapng files into per-anchor RSSI statistics.

    Returns
    -------
    dict  {grid_num (int): {rloc16 (int): {"mean", "std"}}}
    """
    result = {}
    for grid_num in range(1, N_GRIDS + 1):
        path = DATA_DIR / f"grid{grid_num}.pcapng"
        if not path.exists():
            print(f"  [warn] {path.name} not found, skipping grid {grid_num}")
            continue
        result[grid_num] = extract_anchor_stats(path)
    return result


def predict_uncertainty(x, y, power=2, grid_stats=None):
    """
    Predict per-anchor RSSI standard deviation at (x, y) using IDW
    interpolation over parse_all_grids_stats() output.  This gives an
    "uncertainty band" to accompany the mean RSSI from predict_rssi().

    Returns
    -------
    dict  {rloc16 (int): predicted_std_dBm (float)}
    """
    if grid_stats is None:
        grid_stats = parse_all_grids_stats()

    all_rlocs = {rloc for m in grid_stats.values() for rloc in m}
    predictions = {}

    for rloc in all_rlocs:
        weights = []
        values  = []
        exact   = False

        for grid_num, stats in grid_stats.items():
            if rloc not in stats:
                continue

            gx, gy = GRID_POSITIONS[grid_num]
            d = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

            if d == 0.0:
                predictions[rloc] = stats[rloc]["std"]
                exact = True
                break

            weights.append(1.0 / d ** power)
            values.append(stats[rloc]["std"])

        if exact or not values:
            continue

        total_w = sum(weights)
        predictions[rloc] = sum(w * v for w, v in zip(weights, values)) / total_w

    return predictions


def predict_rssi(x, y, power=2, grid_data=None):
    """
    Predict RSSI at position (x, y) using Inverse Distance Weighting
    over all N_GRIDS grid measurement points.

    Parameters
    ----------
    x, y      : float  — query position in metres
    power     : float  — IDW distance exponent (default 2)
    grid_data : dict   — pre-parsed output of parse_all_grids();
                         parsed fresh when None

    Returns
    -------
    dict  {rloc16 (int): predicted_rssi_dBm (float)}
    """
    if grid_data is None:
        grid_data = parse_all_grids()

    all_rlocs = {rloc for m in grid_data.values() for rloc in m}
    predictions = {}

    for rloc in all_rlocs:
        weights = []
        values  = []
        exact   = False

        for grid_num, measurements in grid_data.items():
            if rloc not in measurements:
                continue

            gx, gy = GRID_POSITIONS[grid_num]
            d = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

            if d == 0.0:
                # Exact grid point hit — return measured value immediately
                predictions[rloc] = measurements[rloc]
                exact = True
                break

            weights.append(1.0 / d ** power)
            values.append(measurements[rloc])

        if exact or not values:
            continue

        total_w = sum(weights)
        predictions[rloc] = sum(w * v for w, v in zip(weights, values)) / total_w

    return predictions


def evaluate(power=2):
    """
    Leave-one-out cross-validation across all N_GRIDS grids.

    For each grid g, predicts RSSI at g using the other N_GRIDS-1 grids and
    computes the Mean Absolute Error against the measured values.

    Returns
    -------
    dict  {rloc16 (int): mae_dBm (float)}
    """
    print("Parsing grid data...")
    grid_data = parse_all_grids()

    errors = defaultdict(list)

    for target in sorted(grid_data):
        tx, ty   = GRID_POSITIONS[target]
        loo_data = {g: m for g, m in grid_data.items() if g != target}
        predicted = predict_rssi(tx, ty, power=power, grid_data=loo_data)

        for rloc, pred in predicted.items():
            if rloc in grid_data[target]:
                errors[rloc].append(abs(pred - grid_data[target][rloc]))

    mae = {}
    print(f"\n{'RLOC16':<10}  {'Anchor position':<20}  {'MAE (dBm)':>10}  {'n':>5}")
    print("-" * 52)
    for rloc in sorted(errors):
        m = sum(errors[rloc]) / len(errors[rloc])
        mae[rloc] = m
        pos = ANCHOR_POSITIONS.get(rloc, ("?", "?"))
        pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if isinstance(pos[0], float) else "?"
        print(f"0x{rloc:04x}    {pos_str:<20}  {m:>10.2f}  {len(errors[rloc]):>5}")

    overall = sum(sum(v) for v in errors.values()) / max(
        1, sum(len(v) for v in errors.values())
    )
    print(f"\nOverall MAE: {overall:.2f} dBm")
    return mae


if __name__ == "__main__":
    evaluate()
