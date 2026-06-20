"""
Room-level localization using IEEE 802.15.4 (Thread) RSSI fingerprinting.
Pure Python pcapng parsing (no tshark / scapy).

Feature extraction
------------------
Each pcapng file (one grid location, G1-G26) is split into non-overlapping
WINDOW_SIZE-packet windows.  Every window → one feature vector of per-anchor
RSSI statistics (mean, std) for each of the 8 KNOWN_ANCHORS,
labelled with the grid number.  Feature dimension = 8 anchors × 2 stats = 16.

PART 1 — Evaluation: Grid-Stratified 7-Fold Cross-Validation
--------------------------------------------------------------
With only ~7-29 windows per grid, holding out 4 *entire* grids per fold
means those labels never appear in training → classifiers cannot predict
them → accuracy is 0 % by construction.

The correct approach is a *grid-stratified* split: each grid's windows
are distributed round-robin across the 7 folds so that:
  • ALL 26 grid labels appear in both train and test every fold.
  • No window appears in both train and test.
  • The 4 grids whose windows land in fold k as test still have their
    remaining ~6/7 windows in the training set for that fold.
This mirrors how fingerprinting papers implement "grid-level K-fold":
  each fold holds out one temporal slice per grid (not entire grids),
  ensuring the classifier is evaluated on unseen *samples* from known rooms.

Per-fold test set: ~26 windows drawn from all 26 grids (~1 window each).
Per-fold train set: the remaining 6/7 windows from each grid.

PART 2 — Room-Level Held-Out Test
------------------------------------
The 26 grids are grouped into 5 physical rooms (ROOMS).  One grid per room
is withheld entirely (TEST_GRIDS), so classifiers train on 21 grids and are
evaluated on 5 unseen grids — both at grid resolution and at room resolution
(i.e. did the prediction land in the correct room, even if the wrong grid).
"""

import struct
import glob
import os
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RECORDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "records")
N_GRIDS     = 26
N_FOLDS     = 7
WINDOW_SIZE = 50   # anchor-decoded packets per window

# Known anchor RLOC16s — restrict all features to these 8 anchors
KNOWN_ANCHORS = {0xac00, 0x4400, 0x1800, 0xa800,
                 0x5800, 0xe400, 0xb000, 0xa000}

# Per-anchor statistics extracted per window
STATS = ["mean", "std"]

# Room-level grouping of the 26 grids
ROOMS = {
    "Room1_EndRoom":      [1, 2, 3, 24, 25, 26],
    "Room2_MainHallW":    [4, 5, 6, 7],
    "Room3_MainHallN":    [20, 21, 22, 23],
    "Room4_MainHallE":    [8, 9, 10, 17, 18, 19],
    "Room5_Lab":          [11, 12, 13, 14, 15, 16],
}

# One grid held out per room for the room-level test (PART 2)
TEST_GRIDS = {3: "Room1", 6: "Room2", 22: "Room3", 9: "Room4", 13: "Room5"}

# grid number -> room name, derived from ROOMS
GRID_TO_ROOM = {g: room for room, grids in ROOMS.items() for g in grids}


# ── pcapng parsing ────────────────────────────────────────────────────────────

def _iter_epb(raw: bytes):
    offset = 0
    while offset + 8 <= len(raw):
        block_type = struct.unpack_from("<I", raw, offset)[0]
        block_len  = struct.unpack_from("<I", raw, offset + 4)[0]
        if block_len < 12 or offset + block_len > len(raw):
            break
        if block_type == 0x00000006:
            cap_len = struct.unpack_from("<I", raw, offset + 20)[0]
            yield raw[offset + 28 : offset + 28 + cap_len]
        offset += block_len


def _parse_tap_rssi(pkt: bytes, tap_len: int):
    tlv_off = 4
    while tlv_off + 4 <= tap_len and tlv_off + 4 <= len(pkt):
        tlv_type = struct.unpack_from("<H", pkt, tlv_off)[0]
        tlv_len  = struct.unpack_from("<H", pkt, tlv_off + 2)[0]
        if tlv_off + 4 + tlv_len > len(pkt):
            break
        if tlv_type == 1 and tlv_len == 4:
            return struct.unpack_from("<f", pkt, tlv_off + 4)[0]
        tlv_off += 4 + ((tlv_len + 3) & ~3)
    return None


def _parse_src_short(frame: bytes):
    if len(frame) < 5:
        return None
    fc           = struct.unpack_from("<H", frame, 0)[0]
    dest_mode    = (fc >> 10) & 0x3
    src_mode     = (fc >> 14) & 0x3
    pan_compress = (fc >> 6)  & 0x1
    if src_mode != 2:
        return None
    off = 3
    if dest_mode == 2:
        off += 4
    elif dest_mode == 3:
        off += 10
    if not pan_compress and dest_mode != 0:
        off += 2
    if off + 2 > len(frame):
        return None
    return struct.unpack_from("<H", frame, off)[0]


def extract_packets(path: str) -> list:
    with open(path, "rb") as fh:
        raw = fh.read()
    packets = []
    for pkt in _iter_epb(raw):
        if len(pkt) < 4:
            continue
        tap_len = struct.unpack_from("<H", pkt, 2)[0]
        if tap_len > len(pkt):
            continue
        rssi = _parse_tap_rssi(pkt, tap_len)
        if rssi is None:
            continue
        src = _parse_src_short(pkt[tap_len:])
        if src is not None:
            packets.append((src, rssi))
    return packets


# ── windowed features ─────────────────────────────────────────────────────────

# Per-statistic functions and "missing anchor" fill values, keyed by STATS name
_STAT_FUNCS = {
    "mean":   lambda arr: float(np.mean(arr)),
    "std":    lambda arr: float(np.std(arr)),
    "median": lambda arr: float(np.median(arr)),
    "count":  lambda arr: float(len(arr)),
}
_STAT_MISSING = {"mean": float("nan"), "std": float("nan"),
                 "median": float("nan"), "count": 0.0}


def windows_from_packets(packets, anchors, window_size):
    """
    Split packets into non-overlapping windows and compute, per anchor,
    the statistics in STATS (mean, std) of its RSSI values in that window.

    Returns a list of feature vectors of length len(anchors) * len(STATS),
    ordered as [anchor0_mean, anchor0_std, anchor1_mean, anchor1_std, ...].
    """
    vectors = []
    n_windows = len(packets) // window_size
    for w in range(n_windows):
        chunk = packets[w * window_size : (w + 1) * window_size]
        bucket = defaultdict(list)
        for addr, rssi in chunk:
            if addr in KNOWN_ANCHORS:
                bucket[addr].append(rssi)
        vec = []
        for a in anchors:
            vals = bucket.get(a)
            if vals:
                arr = np.array(vals, dtype=float)
                vec.extend(_STAT_FUNCS[s](arr) for s in STATS)
            else:
                vec.extend(_STAT_MISSING[s] for s in STATS)
        vectors.append(vec)
    return vectors


def build_dataset(records_dir, n_grids, window_size):
    files = sorted(
        (p for p in glob.glob(os.path.join(records_dir, "grid*.pcapng"))
         if int(os.path.basename(p).replace("grid", "").replace(".pcapng", "")) <= n_grids),
        key=lambda p: int(
            os.path.basename(p).replace("grid", "").replace(".pcapng", "")
        ),
    )
    assert len(files) == n_grids

    anchors = sorted(KNOWN_ANCHORS)

    rows, labels, windows_per_grid = [], [], []
    for grid_idx, path in enumerate(files):
        pkts = extract_packets(path)
        vecs = windows_from_packets(pkts, anchors, window_size)
        rows.extend(vecs)
        labels.extend([grid_idx + 1] * len(vecs))
        windows_per_grid.append(len(vecs))

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)

    col_means = np.nanmean(X, axis=0)
    nan_mask  = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X, y, anchors, windows_per_grid


def feature_names(anchors):
    """Human-readable names for the 16 columns of X (anchor x stat)."""
    return [f"0x{a:04X} {stat}" for a in anchors for stat in STATS]


# ── grid-stratified 7-fold split ─────────────────────────────────────────────

def make_grid_stratified_folds(y, n_folds):
    """
    For every grid, assign its windows round-robin to folds 0 … n_folds-1.
    Returns a list of n_folds arrays, each holding the *test* window indices
    for that fold.  Every grid label appears in both test and train every fold.
    """
    fold_test = [[] for _ in range(n_folds)]
    for grid_id in np.unique(y):
        idxs = np.where(y == grid_id)[0]
        for i, idx in enumerate(idxs):
            fold_test[i % n_folds].append(int(idx))
    return [np.array(f, dtype=int) for f in fold_test]


# ── classifiers ───────────────────────────────────────────────────────────────

def make_classifiers():
    return {
        "KNN-1":        KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
        "KNN-3":        KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "KNN-5":        KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM-RBF":      Pipeline([
                            ("scaler", StandardScaler()),
                            ("svm",    SVC(kernel="rbf", C=10, gamma="scale")),
                        ]),
        "SVM-Linear":   Pipeline([
                            ("scaler", StandardScaler()),
                            ("svm",    SVC(kernel="linear", C=1)),
                        ]),
    }


# ── cross-validation ─────────────────────────────────────────────────────────

def kfold_evaluate(clf, X, y, folds):
    """
    Run K-fold CV given pre-built fold test-index arrays.
    Returns (all_true, all_pred, fold_accuracies).
    """
    all_idx  = np.arange(len(y))
    all_true, all_pred = [], []
    fold_accs = []

    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], preds))
        all_true.extend(y[test_idx])
        all_pred.extend(preds)

    return np.array(all_true), np.array(all_pred), np.array(fold_accs)


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.3, linecolor="grey",
    )
    ax.set_xlabel("Predicted grid", fontsize=11)
    ax.set_ylabel("True grid", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_fold(fold_accs_dict, n_folds, out_path):
    """Line chart: per-fold accuracy for every classifier."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fold_nums = list(range(1, n_folds + 1))
    markers   = ["o", "s", "^", "D", "v", "P"]
    for (name, accs), marker in zip(fold_accs_dict.items(), markers):
        ax.plot(fold_nums, [a * 100 for a in accs],
                marker=marker, label=name, linewidth=1.6, markersize=7)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-fold accuracy — Grid-Stratified {n_folds}-Fold CV")
    ax.set_xticks(fold_nums)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(results, n_grids, out_path):
    names     = list(results.keys())
    means     = [results[n]["mean"] * 100 for n in names]
    stds      = [results[n]["std"]  * 100 for n in names]
    fig, ax   = plt.subplots(figsize=(9, 4))
    bars      = ax.bar(names, means, yerr=stds, capsize=5,
                       color="steelblue", edgecolor="white", width=0.6)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 1.5,
                f"{m:.1f}±{s:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(100 / n_grids, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({100/n_grids:.1f}%)")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Mean Accuracy ± Std (%)")
    ax.set_title(f"Room-level localization — Grid-Stratified {N_FOLDS}-Fold CV")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_grid(preds, y_true, grid_labels, title, out_path):
    per_grid = [
        np.mean(preds[y_true == g] == g) if np.any(y_true == g) else 0.0
        for g in grid_labels
    ]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(grid_labels, [a * 100 for a in per_grid], color="steelblue")
    ax.set_xlabel("Grid location")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(grid_labels)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── room-level held-out test (PART 2) ────────────────────────────────────────

def make_room_holdout_split(y, test_grids):
    """
    Split window indices: every window from a grid in `test_grids` goes to
    the test set, everything else goes to the training set.
    """
    test_grid_set = set(test_grids)
    test_idx  = np.array([i for i, g in enumerate(y) if g in test_grid_set], dtype=int)
    train_idx = np.array([i for i, g in enumerate(y) if g not in test_grid_set], dtype=int)
    return train_idx, test_idx


def room_level_holdout_evaluate(X, y):
    """
    Train each classifier on the grids not in TEST_GRIDS and evaluate on
    the held-out grids, reporting both grid-level accuracy (exact grid
    match) and room-level accuracy (predicted grid maps to the correct
    room via GRID_TO_ROOM, even if the exact grid is wrong).

    Returns
    -------
    dict {classifier_name: {"grid_acc", "room_acc", "all_true", "all_pred"}},
    train_idx, test_idx
    """
    train_idx, test_idx = make_room_holdout_split(y, TEST_GRIDS)
    classifiers = make_classifiers()

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X[train_idx], y[train_idx])
        preds  = clf.predict(X[test_idx])
        y_test = y[test_idx]

        grid_acc = accuracy_score(y_test, preds)
        true_rooms = [GRID_TO_ROOM[g] for g in y_test]
        pred_rooms = [GRID_TO_ROOM[g] for g in preds]
        room_acc = accuracy_score(true_rooms, pred_rooms)

        results[name] = {
            "grid_acc": grid_acc, "room_acc": room_acc,
            "all_true": y_test, "all_pred": preds,
        }

    return results, train_idx, test_idx


def plot_room_holdout(results, out_path):
    """Grouped bar chart: grid accuracy vs room accuracy per classifier."""
    names    = list(results.keys())
    grid_acc = [results[n]["grid_acc"] * 100 for n in names]
    room_acc = [results[n]["room_acc"] * 100 for n in names]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width / 2, grid_acc, width, label="Grid accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, room_acc, width, label="Room accuracy", color="seagreen")
    for bars, vals in ((bars1, grid_acc), (bars2, room_acc)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Room-Level Held-Out Test — Grid vs Room Accuracy\n"
        f"(trained on {N_GRIDS - len(TEST_GRIDS)} grids, "
        f"tested on {sorted(TEST_GRIDS)})"
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("PART 1: Grid-Stratified 7-Fold Cross-Validation")
    print("=" * 68)
    print(f"Parsing pcapng files  (window_size = {WINDOW_SIZE} packets) …")
    X, y, anchors, wpg = build_dataset(RECORDS_DIR, N_GRIDS, WINDOW_SIZE)

    print(f"Dataset        : {X.shape[0]} windows × {X.shape[1]} features "
          f"({len(anchors)} anchors × {len(STATS)} stats)")
    print(f"Windows/grid   : min={min(wpg)}  max={max(wpg)}  "
          f"mean={np.mean(wpg):.1f}  total={sum(wpg)}")
    print(f"Anchor RLOC16s : {[f'0x{a:04X}' for a in anchors]}")

    folds = make_grid_stratified_folds(y, N_FOLDS)
    fold_sizes = [len(f) for f in folds]
    print(f"\nFold test sizes: {fold_sizes}  (sum={sum(fold_sizes)})")
    print()

    print("=" * 68)
    print(f"Grid-Stratified {N_FOLDS}-Fold Cross-Validation")
    print(f"Each fold: test on ~{int(np.mean(fold_sizes))} windows from all {N_GRIDS} grids, "
          f"train on remaining ~{X.shape[0]-int(np.mean(fold_sizes))}")
    print("=" * 68)

    classifiers   = make_classifiers()
    grid_labels   = list(range(1, N_GRIDS + 1))
    out_dir       = os.path.dirname(os.path.abspath(__file__))

    results       = {}   # name → {mean, std, fold_accs, all_true, all_pred}
    fold_accs_all = {}   # name → [acc_fold0, ..., acc_fold6]

    # ── header row ────────────────────────────────────────────────────────────
    header = f"{'Classifier':<20}" + "".join(f"  Fold{k+1}" for k in range(N_FOLDS))
    header += "    Mean ± Std"
    print(header)
    print("-" * len(header))

    for name, clf in classifiers.items():
        all_true, all_pred, fold_accs = kfold_evaluate(clf, X, y, folds)
        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))

        results[name] = {
            "mean": mean_acc, "std": std_acc,
            "fold_accs": fold_accs,
            "all_true": all_true, "all_pred": all_pred,
        }
        fold_accs_all[name] = fold_accs

        row = f"{name:<20}"
        row += "".join(f"  {a*100:5.1f}%" for a in fold_accs)
        row += f"    {mean_acc*100:.1f}% ± {std_acc*100:.1f}%"
        print(row)

    print()

    # ── confusion matrices ────────────────────────────────────────────────────
    print("Saving confusion matrices …")
    for name, res in results.items():
        cm = confusion_matrix(res["all_true"], res["all_pred"], labels=grid_labels)
        cm_path = os.path.join(
            out_dir,
            f"cm_{name.replace('-','_').replace(' ','_')}.png"
        )
        plot_confusion(
            cm, grid_labels,
            f"{name}  (mean acc = {res['mean']*100:.1f}% ± {res['std']*100:.1f}%)",
            cm_path,
        )

    # ── per-fold line chart ───────────────────────────────────────────────────
    pf_path = os.path.join(out_dir, "per_fold_accuracy.png")
    plot_per_fold(fold_accs_all, N_FOLDS, pf_path)
    print(f"Per-fold chart     → {os.path.basename(pf_path)}")

    # ── summary bar chart ─────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "accuracy_summary.png")
    plot_summary(results, N_GRIDS, summary_path)
    print(f"Summary chart      → {os.path.basename(summary_path)}")

    # ── per-grid accuracy for best classifier ─────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["mean"])
    best      = results[best_name]
    pg_path   = os.path.join(out_dir, "per_grid_accuracy.png")
    plot_per_grid(
        best["all_pred"], best["all_true"], grid_labels,
        f"Per-grid accuracy  ({best_name},  "
        f"mean {best['mean']*100:.1f}% ± {best['std']*100:.1f}%)",
        pg_path,
    )
    print(f"Per-grid accuracy  → {os.path.basename(pg_path)}")

    # ── feature importance (full dataset) ────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    imp        = rf.feature_importances_
    names      = feature_names(anchors)
    sorted_idx = np.argsort(imp)[::-1]
    fig, ax    = plt.subplots(figsize=(14, 4))
    ax.bar([names[i] for i in sorted_idx], imp[sorted_idx],
           color="steelblue")
    ax.set_xlabel("Feature (anchor + statistic)")
    ax.set_ylabel("Feature importance")
    ax.set_title("Random Forest — feature importance (full dataset)")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    fi_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance → {os.path.basename(fi_path)}")

    # ── final summary (PART 1) ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Classifier':<20}  {'Mean acc':>10}  {'Std':>8}")
    print("-" * 44)
    for name in sorted(results, key=lambda n: -results[n]["mean"]):
        r = results[name]
        print(f"{name:<20}  {r['mean']*100:>9.1f}%  {r['std']*100:>7.1f}%")
    print("-" * 44)
    print(f"Chance level        {100/N_GRIDS:>9.1f}%")
    print("=" * 68)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("PART 2: Room-Level Held-Out Test")
    print("=" * 68)
    print(f"Rooms          : {len(ROOMS)}")
    for room, grids in ROOMS.items():
        print(f"  {room:<16} grids={grids}")
    train_grids = sorted(set(range(1, N_GRIDS + 1)) - set(TEST_GRIDS))
    print(f"Held-out grids : {TEST_GRIDS}  (one per room)")
    print(f"Train grids    : {train_grids}  ({len(train_grids)} grids)")

    room_results, train_idx, test_idx = room_level_holdout_evaluate(X, y)
    print(f"\nTrain windows  : {len(train_idx)}")
    print(f"Test windows   : {len(test_idx)}")

    print(f"\n{'Classifier':<20}  {'Grid Acc':>10}  {'Room Acc':>10}")
    print("-" * 44)
    for name in sorted(room_results, key=lambda n: -room_results[n]["room_acc"]):
        r = room_results[name]
        print(f"{name:<20}  {r['grid_acc']*100:>9.1f}%  {r['room_acc']*100:>9.1f}%")
    print("-" * 44)
    print(f"Grid chance level  {100/N_GRIDS:>9.1f}%")
    print(f"Room chance level  {100/len(ROOMS):>9.1f}%")
    print("=" * 68)

    room_holdout_path = os.path.join(out_dir, "room_holdout_accuracy.png")
    plot_room_holdout(room_results, room_holdout_path)
    print(f"\nRoom holdout chart → {os.path.basename(room_holdout_path)}")


if __name__ == "__main__":
    main()
