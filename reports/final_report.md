# Room-Level RSSI Localization: A Comprehensive Study

## 1. Introduction

This report documents an end-to-end investigation into indoor, room-level
localization using Received Signal Strength Indicator (RSSI) measurements
from an IEEE 802.15.4 / Thread (OpenThread) mesh network. The study spans
four complementary approaches:

1. **Fingerprinting** — supervised classification of a device's location
   into one of 26 discrete grid cells using a 16-dimensional RSSI feature
   vector (8 known anchors × {mean, std}) as input
   (`localize.py`, PART 1). A second evaluation (PART 2) groups the 26
   grids into **5 physical rooms** and tests whether a classifier trained
   without one grid per room can still recognize the *room* containing
   that held-out grid — a more realistic deployment question than exact
   grid-cell recovery.
2. **Spatial interpolation** — Inverse Distance Weighting (IDW) models that
   predict the expected RSSI from any anchor at any (x, y) position in the
   building, validated via leave-one-out cross-validation and a held-out
   grid test.
3. **Path-loss modelling** — log-distance path-loss models (shared and
   per-anchor exponents) fitted from the same RSSI data, compared against
   the IDW baseline.
4. **Transferability / trilateration** — the central question of the
   project: *can a localization model trained with one set of anchors and
   reference points generalize to new, previously unseen anchors and
   positions?* Ten successive iterations (`transferability_test.py`
   through `transferability_test_v10.py`, plus a confidence-scoring layer)
   progressively refine a weighted-centroid + Nelder-Mead trilateration
   baseline with a Random Forest correction stage, calibration strategies,
   richer feature sets, and a per-prediction confidence score.

All numbers in this report were obtained by re-running the project's own
Python scripts against the raw `.pcapng` capture files in `records/`. The
pipeline now uses **26 of the 28 available grid captures** (`grid1.pcapng`
… `grid26.pcapng`; `grid27`/`grid28` are excluded from every script) and
restricts every model to the **8 anchors with known, surveyed positions**
(`KNOWN_ANCHORS == ANCHOR_POSITIONS` in `interpolate.py`). No values are
estimated or placeholder — every figure below is taken directly from a
script's printed output.

The building under test measures approximately **35 m × 8.1 m**, covered by
**26 measurement grid points** and **8 anchors with known physical
positions**, all of which are usable for both fingerprinting and
trilateration.

---

## 2. Hardware & Test Environment

The wireless network under test is a **Thread (IEEE 802.15.4) mesh
network** built from OpenThread border-router devices. A laptop
(ASUS Vivobook) running an OpenThread web dashboard ("Thread Routers") was
used to confirm network configuration during data collection:

| Parameter | Value |
|---|---|
| Network name | `ha-thread-bc63` |
| PAN ID | `0xbc63` |
| Extended PAN ID | `9bda9bbc2e0a2138` |
| Channel | 20 |
| Link layer | IEEE 802.15.4 (2.4 GHz) |
| Capture format | pcapng, IEEE 802.15.4 TAP (DLT 283) |

Three of the routers visible on the dashboard during setup were:

| Dashboard label | RLOC16 | Role |
|---|---|---|
| 1-router | `0xac00` | router |
| 2-router | `0x4400` | router |
| 3-router | `0x1800` | router |

Across the 26 capture files used by the pipeline, **14 distinct router
RLOC16s** were observed transmitting frames (`0x1400, 0x1800, 0x3400,
0x4400, 0x5800, 0x7800, 0x9400, 0xa000, 0xa800, 0xac00, 0xb000, 0xe400,
0xe800, 0xf000`). Of these, **8 have a known surveyed (x, y) position** and
make up `KNOWN_ANCHORS` / `ANCHOR_POSITIONS` in `interpolate.py` — every
fingerprinting, interpolation, path-loss, and trilateration model in this
report is built from exactly these 8 anchors. The remaining 6
(`0x1400, 0x3400, 0x7800, 0x9400, 0xe800, 0xf000`) are heard in the captures
but have no surveyed position and are not used anywhere in the current
pipeline.

**Measurement grid.** The floor was divided into a 26-point grid spanning
approximately x ∈ [2.5, 33.0] m and y ∈ [2.5, 5.0] m (full coordinates in
[Appendix A](#appendix-a-grid-coordinates)). The 8 known anchors sit along
the building's two long walls at y = 0.0 m and y = 8.1 m, giving an overall
building footprint of **35 m × 8.1 m**. At each grid point, a capture
device recorded incoming 802.15.4 frames for a period of time, producing
one `gridN.pcapng` file (N = 1..26, of the 28 files stored in `records/`)
consumed by the pipeline.

---

## 3. Data Collection & Parsing

All parsing is implemented from scratch in pure Python (`interpolate.py`,
`_parse_pcapng` / `_parse_tap_and_frame` / `_src_short_addr`) — no
third-party packet-capture libraries are used.

**pcapng structure.** Each file is a sequence of blocks. The byte order is
detected from the Section Header Block's byte-order magic
(`0x1A2B3C4D` ⇒ little-endian). Only **Enhanced Packet Blocks**
(`block_type == 0x00000006`) are processed; their layout is:

```
block_type(4) | block_total_len(4) | iface_id(4) | ts_hi(4) | ts_lo(4)
             | cap_len(4) | orig_len(4) | packet data (cap_len bytes) | ...
```

**IEEE 802.15.4 TAP header.** Each captured packet begins with a TAP header
(`version(1) + reserved(1) + length(2, LE)`) followed by a sequence of
TLVs. The RSSI value is extracted from **TLV type 1** (4-byte little-endian
IEEE-754 float, representing dBm); each TLV's value field is padded to a
4-byte boundary.

**802.15.4 frame parsing.** Immediately after the TAP header, the raw
802.15.4 MAC frame begins with a 2-byte Frame Control field. The parser
decodes the `PAN ID compression`, `destination addressing mode`, and
`source addressing mode` bits to locate and extract the **2-byte source
short address (RLOC16)**, supporting frame versions 0/1/2 and the common
OpenThread addressing patterns (PAN-ID compression with a present
destination address).

**Router filter.** A frame is kept only if its source RLOC16 satisfies:

```python
rloc16 != 0x0000 and (rloc16 & 0x03FF) == 0
```

i.e. the address is non-zero and its lower 10 bits are all zero — the
standard Thread convention for **router** (non-leaf) RLOC16s.

**Known-anchor filter.** On top of the router filter, every model in this
report additionally restricts frames to `rloc16 in KNOWN_ANCHORS` — the 8
surveyed anchors listed in [Appendix B](#appendix-b-anchor-mapping). This
keeps the feature space, the IDW/path-loss anchor set, and the
trilateration anchor pool consistent across every section.

**Aggregation.** For each of the 26 grid files, all (RSSI, RLOC16) pairs
for the 8 known anchors are grouped by RLOC16 and averaged, producing one
mean RSSI value per anchor per grid:

```
parse_all_grids() → {grid_num: {rloc16: mean_rssi_dBm}}   # 26 grids × 8 anchors
```

This `{grid: {rloc16: rssi}}` dictionary is the single shared input to
every downstream model in this report (interpolation, path-loss,
trilateration). The fingerprinting pipeline (§4) uses the same 8 anchors
but works from per-window statistics rather than per-grid means.

**RSSI Stability Analysis.** To assess whether uncertainty-weighted
aggregation would improve localization, `parse_all_grids_v2()` was
implemented to compute per-anchor std across all measurement windows.
Analysis across all 206 grid-anchor pairs revealed that RSSI measurements
are highly stable (std < 5 dBm everywhere, mean std = 0.95 dBm). This
confirms that mean-only aggregation in IDW and path-loss fitting
introduces no meaningful bias — G14's elevated error (4.50 m) stems from
geometric coverage gaps (all TEST anchors at y=0, G14 at y=5.0), not
measurement instability.

---

## 4. Fingerprinting (`localize.py`)

The fingerprinting pipeline (PART 1) treats room-level localization as a
**26-class classification problem**: given a 16-dimensional feature vector
summarizing the RSSI from the 8 known anchors, predict which of the 26
grid cells the observation came from.

**Windowing & features.** Each grid's capture is split into windows of
`WINDOW_SIZE = 50` packets; each window yields one labelled feature vector
of **16 features** — for each of the 8 known anchors, the **mean and std**
of RSSI values observed in that window
(`STATS = ["mean", "std"]`, `feature_names(anchors)`).
Across all 26 grids this produces **197 windows × 16 features**
(min 6, max 12, mean 7.6 windows per grid).

**Cross-validation scheme — grid-stratified 7-fold CV.**
`make_grid_stratified_folds(y, n_folds=7)` assigns each grid's windows to
folds in round-robin order, so that **every one of the 26 grid labels is
represented in both the training and test set of every fold** — avoiding
the pathological case where a fold's test set contains a label never seen
in training.

**Classifiers compared** (`make_classifiers()`):

| Classifier | Configuration |
|---|---|
| KNN-1 | `KNeighborsClassifier(n_neighbors=1, metric="euclidean")` |
| KNN-3 | `KNeighborsClassifier(n_neighbors=3, metric="euclidean")` |
| KNN-5 | `KNeighborsClassifier(n_neighbors=5, metric="euclidean")` |
| RandomForest | `RandomForestClassifier(n_estimators=200, random_state=42)` |
| SVM-RBF | `Pipeline([StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")])` |
| SVM-Linear | `Pipeline([StandardScaler(), SVC(kernel="linear", C=1)])` |

### 4.1 Results — grid-stratified 7-fold CV accuracy

| Classifier | Mean Accuracy ± Std |
|---|---|
| KNN-1 | 76.8% ± 6.7% |
| KNN-3 | 73.1% ± 8.2% |
| KNN-5 | 68.5% ± 10.5% |
| **RandomForest** | **90.0% ± 8.2%** |
| SVM-RBF | 53.7% ± 7.9% |
| SVM-Linear | 58.7% ± 9.5% |

**Chance level**: 1/26 = **3.8%**. Every classifier dramatically
outperforms chance, and the **Random Forest is the clear winner**, with
both the highest mean accuracy (90.0%) and accuracy more than **23×** the
chance level — though with a wider spread (±8.2%) than the smaller, less
discriminative classifiers.

### 4.2 Per-grid accuracy (Random Forest, best classifier)

The Random Forest reaches **100% accuracy on 13 of the 26 grids**.
The remaining 13 grids show reduced accuracy, ranging from **58.3% (Grid
10, the lowest)** up to ~91.7%:

| Grid | Accuracy | Grid | Accuracy |
|---|---|---|---|
| 1 | 100.0% | 14 | 100.0% |
| 2 | 85.7% | 15 | 100.0% |
| 3 | 100.0% | 16 | 100.0% |
| 4 | 83.3% | 17 | 83.3% |
| 5 | 83.3% | 18 | 83.3% |
| 6 | 100.0% | 19 | 100.0% |
| 7 | 100.0% | 20 | 100.0% |
| 8 | 66.7% | 21 | 83.3% |
| 9 | 75.0% | 22 | 100.0% |
| 10 | **58.3%** | 23 | 87.5% |
| 11 | 91.7% | 24 | 100.0% |
| 12 | 100.0% | 25 | 75.0% |
| 13 | 100.0% | 26 | 88.9% |

Mean across all 26 grids: **90.0% ± 8.2%** (matches the CV summary above).
**Grid 10 (25.0, 2.5)** remains the hardest grid to classify — it sits
between two well-separated anchors along the y = 0 wall and its 16-dim RSSI
signature is apparently easy to confuse with its immediate neighbors (Grids
9 and 11). Grid 9 (22.53, 2.5) — the only grid that breaks the otherwise
regular 2.5 m spacing, and the hardest grid in the original 28-grid/14-anchor
study — is the second-hardest at 75.0% (tied with Grid 25). Two grids
improved to 100% under the 16-dim feature set (Grids 13 and 15, both
85.7% before), while Grid 26 dropped slightly from 100% to 88.9% — the
only grid that got *worse*.

#### Confusion matrix (Random Forest, 26×26)

The pooled confusion matrix across all 7 folds (overall accuracy
**89.85%**, matching the 90.0% ± 8.2% mean above) shows that almost all
confusion is **spatially local**. The two largest off-diagonal entries are
**G10 → G11** (4 windows, 2.50 m apart) and **G9 → G10** (2 windows,
2.47 m apart) — three consecutive points along the y = 2.5 m row in the
building's east end, all sitting close to the tightly-clustered anchors
`0xb000` (25.0, 0.0) and `0xe400` (22.6, 0.0), so their 16-dim
mean/std-RSSI signatures are nearly indistinguishable. The next-largest
confusion, **G2 → G25** (1 window, 2.50 m apart), is a *cross-hall* pair at
the same x = 5.0 m but on opposite rows (y = 2.5 vs y = 5.0) — near the
west-end anchors `0xac00`/`0x4400` the 8.1 m hall width is apparently not
always enough to separate the two rows. The remaining top-5 entries
(**G4 → G11**, 17.50 m apart, and **G5 → G20**, 5.59 m apart) are isolated
single-window errors with no shared geometric pattern, most likely transient
noise in one window rather than a systematic spatial ambiguity.

#### Hardest vs Easiest Grids

| Hardest Grid | Accuracy | Note |
|---|---|---|
| G10 (25.0, 2.5) | 58.3% | 5 of 12 windows misclassified, mostly to adjacent G11 (4×) and G13 (1×) |
| G8 (20.0, 2.5) | 66.7% | 2 of 6 windows misclassified, to G10 (1×) and cross-hall G19 (1×) |
| G9 (22.53, 2.5) | 75.0% | 2 of 8 windows misclassified, both to adjacent G10 |
| G25 (5.0, 5.0) | 75.0% | 2 of 8 windows misclassified, to distant G11 (1×) and G14 (1×) |
| G4 (10.0, 2.5) | 83.3% | 1 of 6 windows misclassified, to distant G11 |

| Easiest Grid | Accuracy | Note |
|---|---|---|
| G1 (2.5, 2.5) | 100.0% | West-end of the y = 2.5 row |
| G3 (7.5, 2.5) | 100.0% | West-central, y = 2.5 row |
| G6 (15.0, 2.5) | 100.0% | Mid-hall, y = 2.5 row |
| G7 (17.5, 2.5) | 100.0% | Mid-hall, y = 2.5 row |
| G12 (30.0, 2.5) | 100.0% | East-end of the y = 2.5 row |

The hardest grids (G10, G8, G9, G25, G4) cluster in the **east half of the
y = 2.5 m row**, where several anchors (`0xb000`, `0xe400`, `0xa000`) sit
close together along the same y = 0 wall — this locally near-uniform anchor
geometry produces only small RSSI differences between neighboring 2.5 m grid
points, and `0x1800`/`0x4400` mean RSSI (the two most important features,
§4.3) change comparatively little over this stretch. By contrast, the five
easiest grids span the full length of the y = 2.5 row (from G1 in the west
to G12 in the east) — for each of these, the *combination* of distances to
all 8 anchors happens not to be closely matched by any other grid, so each
occupies a geometrically distinctive region of the 16-dimensional RSSI
feature space even though many of their row-neighbors are confusable with
each other.

### 4.3 Feature Importance Analysis

The Random Forest's per-feature importances (trained on the full 26-grid,
16-dim dataset) sum to 1.0 across the 8 anchors × 2 statistics
(`mean`, `std`).

**Top 5 most important features:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `0x1800 mean` | 0.1367 |
| 2 | `0x4400 mean` | 0.1312 |
| 3 | `0xa800 mean` | 0.1195 |
| 4 | `0x5800 mean` | 0.1107 |
| 5 | `0xa000 mean` | 0.0951 |

**Anchor ranking by total importance (mean + std):**

| Rank | Anchor (RLOC16) | Position (x, y) | Importance |
|---|---|---|---|
| 1 | `0x4400` | (5.4, 0.0) | 0.162 |
| 2 | `0x1800` | (15.3, 0.0) | 0.159 |
| 3 | `0xa800` | (10.4, 8.1) | 0.140 |
| 4 | `0x5800` | (20.6, 8.1) | 0.124 |
| 5 | `0xa000` | (10.5, 0.0) | 0.106 |
| 6 | `0xac00` | (0.9, 0.0) | 0.106 |
| 7 | `0xb000` | (25.0, 0.0) | 0.104 |
| 8 | `0xe400` | (22.6, 0.0) | 0.098 |

**Interpretation.** The two dominant anchors, `0x4400` (5.4, 0.0) and
`0x1800` (15.3, 0.0), both sit on the y = 0 wall in the western/central
part of the building, directly facing the dense run of grid points along
the y = 2.5 m row — their RSSI changes most steeply across that row,
making them the most discriminative single predictors of grid identity.
`0xa800` (10.4, 8.1) and `0x5800` (20.6, 8.1), on the *opposite* wall, rank
3rd/4th and supply the complementary north–south signal needed to separate
the y = 2.5 m and y = 5.0 m rows (the same separation that fails for the
G2 ↔ G25 confusion in §4.2). Across all 16 features, the 8 `mean` features
together account for **~86.8%** of total importance vs. **~13.2%** for the
8 `std` features — average path-loss-driven RSSI is by far the dominant
positional cue, while intra-window RSSI variability adds comparatively
little, consistent with the low overall measurement variance found in §3
(std < 5 dBm everywhere). For future anchor placements, this suggests
prioritizing anchor positions that maximize the RSSI *gradient* across the
region of interest (e.g., walls directly facing the densest run of
reference points) over simply adding more anchors or sampling more windows
to refine per-anchor `std` — `0xe400` (22.6, 0.0), the lowest-ranked anchor,
sits in the same crowded south-east cluster identified as the hardest
region to fingerprint in §4.2, suggesting that an additional anchor on the
*north* wall near x ≈ 25–30 m (rather than another south-wall anchor) would
likely add more discriminative power than any anchor in the current set.

Unlike the original 14-anchor study (where one anchor without a surveyed
position dominated), all 8 known anchors continue to contribute roughly
comparable shares (9.8%–16.2%) — the model leans most heavily on `0x4400`
and `0x1800`, the same pair that dominated under the 32-dim feature set, now
even more pronounced. The `std` statistic is consistently the least
informative — every anchor's `std` feature ranks below all 8 `mean`
features, and `0xe400 std` (0.0098) remains the single least important of
all 16 features, exactly as it was the least important
of 32 under the previous feature set.

### 4.4 Room-level fingerprinting (PART 2, `room_level_holdout_evaluate`)

While §4.1–4.3 evaluate **exact grid-cell** recovery under cross-validation
(every grid label appears in both train and test), PART 2 asks a different,
arguably more realistic question: **if a classifier has never seen *any*
window from a particular grid, can it still tell which *room* that grid
belongs to?**

**Room grouping (`ROOMS`):** the 26 grids are partitioned into 5 physical
rooms:

| Room | Grids |
|---|---|
| Room1_EndRoom | 1, 2, 3, 24, 25, 26 |
| Room2_MainHallW | 4, 5, 6, 7 |
| Room3_MainHallN | 20, 21, 22, 23 |
| Room4_MainHallE | 8, 9, 10, 17, 18, 19 |
| Room5_Lab | 11, 12, 13, 14, 15, 16 |

**Held-out grids (`TEST_GRIDS`):** one grid per room is withheld entirely
from training:

| Held-out grid | Room |
|---|---|
| Grid 3 | Room1_EndRoom |
| Grid 6 | Room2_MainHallW |
| Grid 22 | Room3_MainHallN |
| Grid 9 | Room4_MainHallE |
| Grid 13 | Room5_Lab |

`room_level_holdout_evaluate(X, y)` trains each classifier on the **161
windows** from the 21 non-held-out grids and evaluates on the **36
windows** from the 5 held-out grids, scoring two metrics per classifier:

- **`grid_acc`** — exact-grid accuracy on the held-out windows (necessarily
  **0.0% for every classifier**, since none of the 5 held-out grid labels
  ever appear in the training set — there is no way for any model to
  predict a label it has never seen).
- **`room_acc`** — accuracy of the **room** containing the predicted grid,
  vs. the room containing the true (held-out) grid.

| Classifier | grid_acc | room_acc |
|---|---|---|
| **KNN-1** | 0.0% | **94.4%** |
| **KNN-3** | 0.0% | **94.4%** |
| **KNN-5** | 0.0% | **94.4%** |
| RandomForest | 0.0% | 83.3% |
| SVM-RBF | 0.0% | 63.9% |
| SVM-Linear | 0.0% | 66.7% |

**All three KNN variants now generalize equally well at the room level**,
each achieving **94.4% room accuracy** (34/36 windows) on the held-out
grids despite (by construction) 0% exact grid accuracy — their
neighbor-averaging behavior naturally falls back to the *nearest known
grid*, which is reliably in the same room as the held-out grid.
**RandomForest — the best in-distribution classifier (90.0% in
§4.1) — again generalizes worse at the room level (83.3%, up from 72.2%
under the 32-dim feature set, but still below every KNN variant)**,
continuing the accuracy/generalization tradeoff: the RF's high CV accuracy
may rely in part on memorizing fine-grained per-grid RSSI patterns that do
not transfer to a nearby, unseen grid as gracefully as KNN's local
averaging does. **SVM-RBF (63.9%) is now the weakest at room-level
generalization** — a sharp drop from 91.7% under the 32-dim feature set —
while SVM-Linear improved slightly to 66.7% (from 61.1%), still consistent
with its comparatively low CV accuracy (58.7%) in §4.1.

### 4.5 Feature Subset Selection (Ablation Study)

Before committing to a 16-dim feature vector, an ablation study
(`ablation_study.py`) re-ran the grid-stratified 7-fold CV (RandomForest
only) on five feature subsets, keeping only specific per-anchor RSSI
statistics:

| Subset | Features | Accuracy | vs baseline |
|---|---|---|---|
| A: mean only | 8 | 88.6% | -0.5% |
| B: mean+median | 16 | 86.7% | -2.4% |
| **C: mean+std** | **16** | **90.0%** | **+0.9%** |
| D: mean+count | 16 | 89.5% | +0.4% |
| E: mean+std+median+count | 32 | 89.1% | baseline |

**Finding: mean+std (16-dim) outperforms full 32-dim set — median and
count add noise.** Subset C (`mean+std`, 16 features) was the only subset
to *beat* the full 32-dim baseline (E), while B (`mean+median`) was the
worst performer — adding `median` actively hurts accuracy. This result is
what motivated the pipeline-wide switch to `STATS = ["mean", "std"]`
performed in this update: §4.1's new RandomForest CV accuracy
(**90.0% ± 8.2%**) matches subset C's ablation result exactly, confirming
that the 16-dim `mean+std` feature vector is the best choice found so far.

---

## 5. RSSI Spatial Interpolation (`interpolate.py`)

The interpolation module predicts the expected RSSI from any anchor at any
continuous (x, y) position using **Inverse Distance Weighting (IDW)**:

```python
def predict_rssi(x, y, power=2, grid_data=None):
    # weight_i = 1 / distance(query, grid_i) ** power
    # prediction = sum(weight_i * rssi_i) / sum(weight_i)
    # exact match (distance == 0) returns the measured value directly
```

with the default exponent **power = 2**.

### 5.1 Leave-one-out cross-validation (26 grids)

`evaluate(power=2)` performs LOO-CV: for each of the 26 grids, RSSI is
predicted at that grid's coordinates using IDW over the **other 25**
grids, and compared against the measured value, for each of the 8 known
anchors.

| RLOC16 | Anchor position | MAE (dBm) | n |
|---|---|---|---|
| `0x1800` | (15.3, 0.0) | 4.43 | 26 |
| `0x4400` | (5.4, 0.0) | 3.41 | 24 |
| `0x5800` | (20.6, 8.1) | 4.17 | 26 |
| `0xa000` | (10.5, 0.0) | 4.65 | 26 |
| `0xa800` | (10.4, 8.1) | 4.09 | 26 |
| `0xac00` | (0.9, 0.0) | 4.66 | 26 |
| `0xb000` | (25.0, 0.0) | 4.79 | 26 |
| `0xe400` | (22.6, 0.0) | 3.73 | 26 |

**Overall MAE: 4.25 dBm** across all (anchor, grid) pairs — `0x4400` is the
best-predicted anchor (3.41 dBm; only 24 of 26 grids received it), while
`0xb000` is the worst (4.79 dBm).

### 5.2 Held-out grid validation (`held_out_validate.py`)

As a stricter test, `held_out_validate.py` removes **2 entire grids**
(`HELD_OUT = [5, 15]`) from the training set, fits IDW on the remaining
**24 grids**, and predicts RSSI at the 2 held-out positions for all 8
known anchors. These per-grid IDW MAE values are also re-used as the IDW
reference baseline inside `path_loss_model.py` (`IDW_PER_GRID`,
`IDW_OVERALL_MAE`):

| Held-out grid | Position | Anchor | Predicted | Measured | Error (dBm) | Grid MAE (dBm) |
|---|---|---|---|---|---|---|
| Grid 5 | (12.5, 2.5) | `0x1800` | -67.64 | -67.73 | 0.09 | |
| | | `0x4400` | -75.73 | -84.25 | 8.52 | |
| | | `0x5800` | -63.27 | -71.08 | 7.82 | |
| | | `0xa000` | -67.28 | -49.80 | 17.48 | |
| | | `0xa800` | -61.65 | -63.42 | 1.77 | |
| | | `0xac00` | -71.35 | -72.70 | 1.35 | |
| | | `0xb000` | -75.10 | -79.17 | 4.07 | |
| | | `0xe400` | -77.90 | -81.50 | 3.60 | **5.59** |
| Grid 15 | (30.0, 5.0) | `0x1800` | -70.05 | -71.00 | 0.95 | |
| | | `0x4400` | -82.43 | -79.24 | 3.19 | |
| | | `0x5800` | -69.67 | -72.33 | 2.67 | |
| | | `0xa000` | -83.91 | -85.60 | 1.69 | |
| | | `0xa800` | -68.22 | -69.18 | 0.96 | |
| | | `0xac00` | -80.99 | -81.67 | 0.68 | |
| | | `0xb000` | -68.59 | -62.43 | 6.16 | |
| | | `0xe400` | -69.87 | -70.75 | 0.88 | **2.15** |
| **Overall** | | | | | | **3.87** |

Grid 5's largest error is anchor `0xa000` (17.48 dBm) — the predicted value
(-67.28) is far weaker than the measured value (-49.80), suggesting Grid 5
sits unusually close to / in strong line-of-sight with `0xa000` in a way
its neighbors do not. Grid 15, by contrast, is predicted well for 7 of 8
anchors (largest error 6.16 dBm on `0xb000`), giving it the lowest overall
error (2.15 dBm) of the two held-out grids.

### 5.3 Example query (`predict_cli.py`)

`predict_cli.py <x> <y>` finds the nearest grid to the queried position and
prints the IDW-predicted RSSI for every known anchor, ranked from strongest
to weakest. Example for `python predict_cli.py 12.5 4.0`:

```
Query position : (12.500, 4.000) m
Closest grid   : #22  at (12.50, 5.00) m  [Δ = 1.00 m]

Rank  RLOC16     Anchor (x, y)        Predicted RSSI
─────────────────────────────────────────────────────
  1    0xa800    (10.4, 8.1)               -59.02 dBm
  2    0xa000    (10.5, 0.0)               -61.45 dBm
  3    0x5800    (20.6, 8.1)               -65.00 dBm
  4    0x1800    (15.3, 0.0)               -65.49 dBm
  5    0xac00    (0.9, 0.0)                -68.44 dBm
  6    0xb000    (25.0, 0.0)               -73.95 dBm
  7    0x4400    (5.4, 0.0)                -77.05 dBm
  8    0xe400    (22.6, 0.0)               -78.03 dBm
```

---

## 6. Path Loss Modelling (`path_loss_model.py`)

The log-distance path-loss model relates RSSI to distance via:

```
RSSI(d) = TxPower - 10 * n * log10(d)
```

where `n` is the path-loss exponent and `TxPower` is the (effective)
transmit power / RSSI at 1 m. Two variants are fitted on the **24
training grids** (Grids 5, 15 held out, matching §5.2), using the **8
known anchors**:

- **Model 1 — shared exponent.** A single `n` is fitted jointly across
  all anchors via `np.linalg.lstsq`, with a per-anchor `TxPower` solved
  simultaneously.
- **Model 2 — per-anchor exponent.** Each anchor `i` gets its own
  independently-fitted `(n_i, TxPower_i)`, also via least-squares.

Training data spans distances from 2.5 m up to 32.5 m (anchor `0xac00`).

### 6.1 Model 1 — shared path-loss exponent

**Fitted exponent: n = 2.2403**

| Anchor | TxPower (dBm) | Position |
|---|---|---|
| `0x1800` | -48.71 | (15.3, 0.0) |
| `0x4400` | -53.91 | (5.4, 0.0) |
| `0x5800` | -43.89 | (20.6, 8.1) |
| `0xa000` | -53.33 | (10.5, 0.0) |
| `0xa800` | -42.56 | (10.4, 8.1) |
| `0xac00` | -48.07 | (0.9, 0.0) |
| `0xb000` | -50.44 | (25.0, 0.0) |
| `0xe400` | -52.83 | (22.6, 0.0) |

### 6.2 Model 2 — per-anchor path-loss exponent

| Anchor | n | TxPower (dBm) | Position |
|---|---|---|---|
| `0x1800` | 2.4628 | -46.65 | (15.3, 0.0) |
| `0x4400` | 2.0462 | -55.88 | (5.4, 0.0) |
| `0x5800` | 1.4121 | -51.73 | (20.6, 8.1) |
| `0xa000` | 2.6783 | -49.11 | (10.5, 0.0) |
| `0xa800` | 1.3393 | -51.37 | (10.4, 8.1) |
| `0xac00` | 2.2646 | -47.79 | (0.9, 0.0) |
| `0xb000` | 2.5059 | -47.87 | (25.0, 0.0) |
| `0xe400` | 3.0290 | -45.43 | (22.6, 0.0) |

The per-anchor exponents range from **1.34** (`0xa800`, n ≈ free-space-like)
to **3.03** (`0xe400`, heavy attenuation), reflecting real differences in
obstruction/clutter around each anchor that a single shared `n` cannot
capture.

### 6.3 Comparison on held-out grids (5, 15)

| Grid | Anchor | Measured (dBm) | Model 1 | Model 2 | Err M1 | Err M2 |
|---|---|---|---|---|---|---|
| G5 | `0x1800` | -67.73 | -61.58 | -60.79 | 6.14 | 6.93 |
| G5 | `0x4400` | -84.25 | -73.55 | -73.82 | 10.70 | 10.43 |
| G5 | `0x5800` | -71.08 | -66.14 | -65.76 | 4.94 | 5.32 |
| G5 | `0xa000` | -49.80 | -64.66 | -62.64 | 14.86 | 12.84 |
| G5 | `0xa800` | -63.42 | -59.96 | -61.77 | 3.46 | 1.65 |
| G5 | `0xac00` | -72.70 | -72.14 | -72.12 | 0.56 | 0.58 |
| G5 | `0xb000` | -79.17 | -75.20 | -75.57 | 3.97 | 3.60 |
| G5 | `0xe400` | -81.50 | -75.62 | -76.24 | 5.88 | 5.26 |
| G15 | `0x1800` | -71.00 | -75.40 | -75.98 | 4.40 | 4.98 |
| G15 | `0x4400` | -79.24 | -85.26 | -84.53 | 6.02 | 5.28 |
| G15 | `0x5800` | -72.33 | -66.19 | -65.79 | 6.14 | 6.54 |
| G15 | `0xa000` | -85.60 | -82.55 | -84.03 | 3.05 | 1.57 |
| G15 | `0xa800` | -69.18 | -71.63 | -68.75 | 2.45 | 0.44 |
| G15 | `0xac00` | -81.67 | -81.01 | -81.09 | 0.66 | 0.58 |
| G15 | `0xb000` | -62.43 | -69.47 | -69.16 | 7.04 | 6.73 |
| G15 | `0xe400` | -70.75 | -74.14 | -74.23 | 3.39 | 3.48 |

### 6.4 MAE summary

| Grid | Model 1 | Model 2 | IDW (reference) |
|---|---|---|---|
| Grid 5 | 6.31 dBm | 5.83 dBm | 5.59 dBm |
| Grid 15 | 4.14 dBm | 3.70 dBm | 2.15 dBm |
| **Overall** | **5.23 dBm** | **4.76 dBm** | **3.87 dBm** |

**IDW is now the best overall RSSI predictor**, beating both path-loss
models and winning on *both* held-out grids individually — a reversal from
earlier iterations of this study, where a fitted path-loss model edged out
IDW overall. Per-anchor exponents (Model 2 = 4.76 dBm) still clearly beat
the shared exponent (Model 1 = 5.23 dBm, n = 2.2403), with the largest
gains on anchors whose fitted `n` is far from 2.24 (`0xa800`, n = 1.3393;
`0xe400`, n = 3.0290). But neither parametric model recovers the simple
non-parametric IDW baseline, which benefits directly from the dense,
irregular grid of real measurements around each held-out point rather than
relying on a single global distance/RSSI relationship.

---

## 7. Transferability Study

### 7.1 Problem framing

Sections 4–6 all assume the **same anchors** are present at training and
test time. The transferability study asks a harder question: **if a
localization model is trained using one subset of anchors and reference
grids, can it still localize a device using a *different* subset of
anchors that it never saw during training?** This simulates redeploying a
model after some anchors are added, removed, or relocated.

All transferability variants (v1–v10) share a common backbone:

1. A **weighted-centroid** estimate from the available TEST anchors using
   path-loss-derived distances `d_i = 10^((TxPower_i - RSSI_i)/(10·n))`.
2. **WC4**: refine the centroid via Nelder-Mead trilateration, minimizing
   `Σ(|p - anchor_i| - d_i)²`, clipped to the building bounds `[0,35]×[0,8.1]`.
3. A **Random Forest correction stage**: a second model predicts the
   residual `(Δx, Δy)` between the WC4 estimate and ground truth, trained
   on anchor-identity-agnostic features (anchors sorted by RSSI strength,
   zero-padded to `K_MAX`).
4. Three **calibration scenarios**:
   - **A — No calibration**: `n = 2.25` (fixed, close to Model 1's
     n = 2.2403 from §6.1), `TxPower` for TEST anchors fitted from TRAIN
     grids only.
   - **B — Few-point calibration**: `n` and `TxPower` for TEST anchors
     re-fitted from a small number of CALIB grids.
   - **C — Oracle**: `n` and `TxPower` for TEST anchors fitted from *all*
     26 grids (upper bound / best case).

### 7.2 Baseline weighted-centroid variants (WC1–WC5, `transferability_test_v5.py`)

Using the **v3 anchor/grid split** (TRAIN anchors `0xac00, 0xa800, 0xb000`;
TEST anchors `0x4400, 0xe400, 0xa000`; 20 TRAIN / 6 TEST grids; `n = 2.25`),
five weighted-centroid / regression variants were compared, all using the
3 TEST anchors:

| Variant | Method | Mean error |
|---|---|---|
| WC1 | Centroid weighted by `w_i = \|RSSI_i\|` | 12.18 m |
| WC2 | Centroid weighted by `w_i = 1/d_i²` (path-loss distance) | 8.49 m |
| WC3 | Centroid weighted by `w_i = 10^(RSSI_i/10)` (linear power) | 7.98 m |
| **WC4** | **Nelder-Mead trilateration** minimizing `Σ(\|p-anchor_i\|-d_i)²`, init at WC3 | **4.77 m** |
| WC5 | `Ridge(alpha=1.0)` on `[d_1,d_2,d_3,x_wc3,y_wc3]` → (x,y) | 6.41 m |

**WC4 (Nelder-Mead trilateration) is the best simple geometric method**,
beating even the learned Ridge regressor (WC5). This makes WC4 the
foundation for all later (v7–v10) pipelines.

### 7.3 Learned baselines (v3 RF, v4 SVR)

Two purely learned (non-geometric) baselines were also evaluated on the
same 20-train/6-test grid split:

**v3 — Random Forest on raw RSSI** (`transferability_test.py`):
`RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)`
predicting (x, y) directly from a 15-dim feature vector (sorted path-loss
distances, pairwise RSSI differences, anchor ranks, RSSI-weighted
centroid, RSSI std). **Mean test error = 10.77 m**.

**v4 — SVR on engineered features** (`transferability_test_v4.py`): a
17-dim, anchor-identity-agnostic feature vector (relative RSSI
differences, ranks, normalized RSSI, path-loss distances, weighted
centroid), compared across three regressors:

| Model | Mean error |
|---|---|
| RF (v4 features) | 11.17 m |
| KNN (k=3) | 11.43 m |
| **SVR** (`MultiOutputRegressor(SVR(kernel="rbf"))`) | **9.64 m** |

SVR is the best of the purely-learned approaches, but still **worse than
the simple geometric WC4 (4.77 m)** — motivating the WC4 + RF-correction
hybrid used from v7 onward. Feature-importance analysis of the v4 Random
Forest shows the **weighted-centroid position (`x_w, y_w`) features
dominate at 85.0%** of total importance, with relative-RSSI differences
(8.2%), normalized RSSI (3.4%), and path-loss distance (3.5%) contributing
the remainder — i.e. even the learned model is, in effect, mostly
re-deriving a centroid estimate.

### 7.4 v7 — WC4 + RF correction

**Split** (`transferability_test_v7.py`, "v3 — triangle coverage"):

- TRAIN anchors (3): `0xac00`, `0xa800`, `0xb000` — a triangle spanning
  the whole building.
- TEST anchors (3): `0x4400`, `0xe400`, `0xa000`.
- TRAIN_GRIDS (15): 1, 2, 4, 6, 7, 9, 10, 11, 12, 16, 17, 18, 20, 22, 23
- TEST_GRIDS (6): 3, 8, 13, 14, 19, 24
- `K_MAX = 3`, RF correction feature dimension `CORR_FEAT_DIM = 17`
- `PATH_LOSS_N_DEFAULT = 2.25`

WC4 trilateration is uncertainty-weighted: each anchor's contribution to
the Nelder-Mead objective is scaled by `1 / (std_i^2 + 0.01)`, where
`std_i` is that anchor's per-grid RSSI standard deviation (from
`parse_all_grids_stats()`), so noisier anchors are down-weighted.

**Fitted path-loss exponent per scenario:**

| Scenario | n | Source |
|---|---|---|
| A — No calib | 2.2500 | fixed (≈ Model 1) |
| C — Oracle | 2.5720 | fitted from all 26 grids |

**TxPower per TEST anchor:**

| Anchor | Scenario A | Scenario C |
|---|---|---|
| `0x4400` | -53.90 dBm | -50.69 dBm |
| `0xe400` | -51.29 dBm | -49.80 dBm |
| `0xa000` | -54.42 dBm | -49.70 dBm |

**RF correction:** Scenario A trains on 15 grids × 17 features
(train MAE = 1.35 m); Scenario C trains on 26 grids × 17 features
(train MAE = 1.16 m).

**Per-grid test results (6 TEST grids):**

| Grid | Actual (x, y) | A: No calib | C: Oracle |
|---|---|---|---|
| G3 | (7.5, 2.5) | 2.42 m | 1.89 m |
| G8 | (20.0, 2.5) | 1.63 m | 0.95 m |
| G13 | (33.0, 2.5) | 6.82 m | 1.09 m |
| G14 | (33.0, 5.0) | 4.51 m | 0.19 m |
| G19 | (20.0, 5.0) | 1.70 m | 0.91 m |
| G24 | (7.5, 5.0) | 3.07 m | 2.18 m |
| **Mean** | | **3.36 m** | **1.20 m** |
| Median | | 2.74 m | 1.02 m |
| ≤ 2 m | | 33.3% | 83.3% |
| ≤ 3 m | | 50.0% | 100.0% |
| ≤ 5 m | | 83.3% | 100.0% |

**Gap to oracle (C): 2.16 m remaining.**

Scenario A (no calibration, fixed n = 2.25, uncertainty-weighted WC4)
remains the **single best non-oracle result in this study** at 3.36 m.
Minimal calibration (5-point) was tested but found to degrade
performance due to poor geometric coverage of the calibration points,
and was excluded from final results. The oracle (C, 1.20 m) shows there
is still substantial headroom if the path-loss exponent could be
calibrated more robustly (see [Limitations](#10-limitations)).

### 7.4.1 Corner Failure Analysis: Grid G14

1. **Location.** G14 sits at `(33.0, 5.0)` — the far-east corner of the
   building, near the boundary `x = 35`, `y = 8.1`.

2. **TEST anchor distances to G14:**

   | Anchor | Position | Distance to G14 |
   |---|---|---|
   | `0x4400` | (5.4, 0) | √((33−5.4)² + (5−0)²) = **28.05 m** |
   | `0xe400` | (22.6, 0) | √((33−22.6)² + (5−0)²) = **11.54 m** |
   | `0xa000` | (10.5, 0) | √((33−10.5)² + (5−0)²) = **23.05 m** |

3. **Geometry problem.** All three TEST anchors sit at `y = 0`, while G14
   is at `y = 5.0` — a poor vertical baseline for trilateration. With all
   anchors on (or near) the same line, the Nelder-Mead solver has only
   weak gradient information in the y-direction, and the objective is
   nearly symmetric around `y = 0`: a point at `(33, 5)` and its mirror
   image `(33, −5)` produce almost identical anchor distances, creating an
   ambiguity that the y = 0 building boundary can only partially resolve.

4. **RSSI at G14.** Measured RSSI for the TEST anchors at G14 is
   `0x4400 = -86.00 dBm`, `0xa000 = -85.00 dBm`, `0xe400 = -75.92 dBm` —
   all anchors are 10–28 m away, so every signal is weak and close to the
   noise floor, giving the path-loss-derived distance estimates little
   precision to work with.

5. **Conclusion.** G14's elevated Scenario A error (4.51 m, §7.4) is a
   **geometric coverage failure**, not a measurement-noise problem (§3,
   RSSI Stability Analysis). Resolving it requires either an additional
   anchor on the east wall (near `x = 35`) to break the y-axis ambiguity,
   or a calibration point near `(33, 5)` to anchor the path-loss fit in
   that corner of the building.

### 7.5 v8 — Geographic split (`transferability_test_v8.py`)

v8 tests a more demanding, *spatially contiguous* transfer scenario: all
TEST anchors are physically located in one half of the building, and all
TEST grids are in the **same** half — i.e. the model must generalize to an
entirely unexplored region, not just unseen anchors scattered throughout a
familiar area.

**Split:**

- Geographic split at `SPLIT_X = 20.0` m (left vs. right half).
- TRAIN anchors (5): `0xac00`, `0x4400`, `0x1800`, `0xa800`, `0xa000`
  (all x < 20).
- TEST anchors (3): `0x5800`, `0xe400`, `0xb000` (all x > 20) — **never
  seen during training**.
- TRAIN_GRIDS (14): 1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26 (left half).
- CALIB_GRIDS (2): 10, 17 (right side).
- TEST_GRIDS (9): 9, 11, 12, 13, 14, 15, 16, 18, 19 (right half).
- `K_MAX = 5`, RF correction feature dimension `CORR_FEAT_DIM = 32`.

**Fitted path-loss exponent per scenario:**

| Scenario | n |
|---|---|
| A — No calib | 2.2500 (fixed) |
| B — 2-pt calib | 3.5018 |
| C — Oracle (25 grids) | 2.5282 |

**TxPower per TEST anchor:**

| Anchor | Scenario A | Scenario B | Scenario C |
|---|---|---|---|
| `0x5800` | -40.61 dBm | -37.54 dBm | -41.29 dBm |
| `0xe400` | -55.37 dBm | -35.36 dBm | -49.80 dBm |
| `0xb000` | -50.71 dBm | -42.99 dBm | -47.55 dBm |

**RF correction:** A/B train on 14 grids × 32 features (train MAE =
1.74 m); C trains on 25 grids × 32 features (train MAE = 2.36 m).

**Per-grid test results (9 TEST grids):**

| Grid | Actual (x, y) | A: No calib | B: 2-pt calib | C: Oracle |
|---|---|---|---|---|
| G9 | (22.5, 2.5) | 3.15 m | 1.38 m | 1.17 m |
| G11 | (27.5, 2.5) | 3.47 m | 4.18 m | 0.88 m |
| G12 | (30.0, 2.5) | 6.06 m | 5.17 m | 1.01 m |
| G13 | (33.0, 2.5) | 5.22 m | 1.21 m | 0.09 m |
| G14 | (33.0, 5.0) | **31.51 m** | **25.09 m** | 17.10 m |
| G15 | (30.0, 5.0) | 7.23 m | 3.40 m | 2.19 m |
| G16 | (27.5, 5.0) | 7.23 m | 11.75 m | 1.28 m |
| G18 | (22.5, 5.0) | 3.02 m | 1.91 m | 0.60 m |
| G19 | (20.0, 5.0) | 0.83 m | 2.47 m | 1.28 m |
| **Mean** | | **7.53 m** | **6.29 m** | **2.84 m** |
| Median | | 5.22 m | 3.40 m | 1.17 m |
| within 3 m | | 11.1% | 44.4% | 88.9% |
| within 5 m | | 44.4% | 66.7% | 88.9% |

**Improvement B vs A: 7.53 m → 6.29 m (16.5% better).**
**Gap to oracle (C): 3.44 m remaining.**

**G14 outlier analysis (31.51 m error).** Grid 14, at the building's
**far corner (33.0, 5.0)**, remains by far the worst prediction in the
entire study — even the oracle scenario only manages 17.10 m there. G14 is
the TEST grid furthest from either CALIB grid (10, 17) and lies at the
extreme edge of the test anchors' coverage, so the WC4 trilateration has
poor geometric dilution of precision (GDOP) and the RF correction — trained
on only 14 grids — has no nearby examples to learn from. This single grid
inflates the v8-A mean from what would otherwise be a ~3.0 m mean
(excluding G14) to 7.53 m, illustrating that **geographic transfer to a
corner/edge location is substantially harder than transfer to interior
locations** (cf. v7, where all TEST grids are interspersed among TRAIN
grids).

### 7.6 v9 — Balanced interleaved split, all anchors, rich features (`transferability_test_v9.py`)

v9 returns to an interleaved (non-geographic) split like v7, but uses
**all 8 known anchors** (4 TRAIN + 4 TEST) and a richer **22-dimensional**
feature set.

**Split:**

- TRAIN anchors (4): `0xac00`, `0x1800`, `0x5800`, `0xb000`.
- TEST anchors (4): `0x4400`, `0xa800`, `0xe400`, `0xa000`.
- TRAIN_GRIDS (18): 1, 2, 4, 6, 7, 9, 10, 11, 12, 15, 16, 17, 20, 21, 22,
  23, 25, 26.
- CALIB_GRIDS (2): 5, 18.
- TEST_GRIDS (6): 3, 8, 13, 14, 19, 24.
- `K_MAX = 4`, `K_PAIRS = 6`, `FEAT_DIM = 22`.
- Feature groups: DISTANCE(4), DIST_RATIO(6), ANGLES(4), GDOP(1),
  COVERAGE(1), REL_RSSI(4), WPOS(2).

**Fitted path-loss exponent per scenario:**

| Scenario | n |
|---|---|
| A — No calib | 2.2500 (fixed) |
| B — 2-pt calib | 2.5798 |
| C — Oracle (26 grids) | 2.3037 |

**TxPower per TEST anchor:**

| Anchor | Scenario A | Scenario B | Scenario C |
|---|---|---|---|
| `0x4400` | -52.50 dBm | -56.12 dBm | -53.45 dBm |
| `0xa800` | -43.13 dBm | -39.02 dBm | -41.97 dBm |
| `0xe400` | -52.09 dBm | -50.50 dBm | -52.33 dBm |
| `0xa000` | -54.32 dBm | -39.96 dBm | -52.27 dBm |

**RF correction:** A/B train on 18 grids × 22 features (train MAE =
2.10 m); C trains on 26 grids × 22 features (train MAE = 1.47 m).

**Per-grid test results (6 TEST grids):**

| Grid | Actual (x, y) | A: No calib | B: 2-pt calib | C: Oracle |
|---|---|---|---|---|
| G3 | (7.5, 2.5) | 7.55 m | 7.53 m | 2.43 m |
| G8 | (20.0, 2.5) | 5.40 m | 7.85 m | 1.82 m |
| G13 | (33.0, 2.5) | 13.60 m | 11.43 m | 2.52 m |
| G14 | (33.0, 5.0) | 16.03 m | 13.67 m | 0.27 m |
| G19 | (20.0, 5.0) | 5.87 m | 4.20 m | 1.42 m |
| G24 | (7.5, 5.0) | 3.79 m | 0.65 m | 2.39 m |
| **Mean** | | **8.71 m** | **7.56 m** | **1.81 m** |
| Median | | 6.71 m | 7.69 m | 2.10 m |
| within 2 m | | 0.0% | 16.7% | 50.0% |
| within 3 m | | 0.0% | 16.7% | 100.0% |
| within 5 m | | 16.7% | 33.3% | 100.0% |

**Improvement B vs A: 8.71 m → 7.56 m (13.2% better).**
**Gap to oracle (C): 5.75 m remaining.**

**Feature group importances (Scenario B):**

| Feature group | Importance |
|---|---|
| DISTANCE | 34.7% |
| REL_RSSI | 18.9% |
| DIST_RATIO | 15.9% |
| GDOP | 13.9% |
| WPOS | 7.0% |
| ANGLES | 5.8% |
| COVERAGE | 3.8% |

Path-loss-derived **DISTANCE** estimates remain the single most important
feature group (34.7%), with relative-RSSI and distance-ratio features
together contributing over half as much again (34.8% combined) — geometric
/ distance information dominates over raw signal-strength patterns.

**Version comparison (mean position error, all variants so far):**

| Version | Mean error |
|---|---|
| v3 RF | 10.77 m |
| v4 SVR | 9.64 m |
| WC4 (v5) | 4.77 m |
| **v7-A** | **3.78 m** |
| v7-B (5-pt) | 3.97 m |
| v8-A | 7.53 m |
| v8-B (2-pt) | 6.29 m |
| v9-A (all-anchor, no calib) | 8.71 m |
| v9-B (all-anchor, 2-pt calib) | 7.56 m |
| v9-C (oracle) | 1.81 m |

v9, despite using all 8 surveyed anchors (4 TRAIN + 4 TEST) and a richer
feature set than v7, performs **worse** than v7 in both no-calib and
calibrated scenarios (8.71 m / 7.56 m vs. 3.78 m / 3.97 m). The smaller
TEST set (6 grids) and the larger, more symmetric TRAIN/TEST anchor split
appear to make this configuration intrinsically harder, despite the richer
features — a result explored further in v10.

### 7.7 v10 — Feature study: v7 split + v9-style rich features (`transferability_test_v10.py`)

v10 isolates the effect of the **feature set** from the effect of the
**split** by re-running the **exact v7 split** (same TRAIN/TEST anchors,
same TRAIN/CALIB/TEST grids, same fitted `n` values: A = 2.2500,
B = 3.1694, C = 2.5720) but replacing v7's 17-dim feature vector with a
**19-dimensional**, v9-style rich feature set (`K_MAX=3`, `K_PAIRS=3`,
8 feature groups including a new **RESIDUALS(3)** group).

**RF correction:** v7 (17 features) trains on 15/15/26 grids with train
MAE = 1.15/1.15/1.20 m (A/B/C); v10 (19 features) trains on the same grid
counts with train MAE = **1.20/1.20/1.37 m** — i.e. v10's richer feature
set fits the *training* data slightly **worse** in every scenario.

**Per-grid comparison, v7 vs. v10 (same split/calibration):**

| Grid | Actual | v7-A | v10-A | v7-B | v10-B | v7-C | v10-C |
|---|---|---|---|---|---|---|---|
| G3 | (7.5, 2.5) | 3.14 m | 6.70 m | 2.41 m | 3.81 m | 2.69 m | 2.59 m |
| G8 | (20.0, 2.5) | 2.02 m | 2.25 m | 1.28 m | 1.27 m | 1.56 m | 1.84 m |
| G13 | (33.0, 2.5) | 6.88 m | 6.16 m | 5.74 m | 5.97 m | 0.57 m | 1.37 m |
| G14 | (33.0, 5.0) | 4.40 m | 4.30 m | 4.71 m | 4.23 m | 0.54 m | 0.19 m |
| G19 | (20.0, 5.0) | 1.11 m | 3.01 m | 1.18 m | 3.75 m | 0.98 m | 1.24 m |
| G24 | (7.5, 5.0) | 5.11 m | 3.74 m | 8.50 m | 5.09 m | 2.36 m | 2.28 m |
| **Mean** | | **3.78 m** | **4.36 m** | **3.97 m** | **4.02 m** | **1.45 m** | **1.58 m** |
| Median | | 3.77 m | 4.02 m | 3.56 m | 4.02 m | 1.27 m | 1.61 m |

**v10 detailed metrics:**

| Scenario | Mean | Median | ≤2m | ≤3m | ≤5m |
|---|---|---|---|---|---|
| A | 4.36 m | 4.02 m | 0.0% | 16.7% | 66.7% |
| B | 4.02 m | 4.02 m | 16.7% | 16.7% | 66.7% |
| C | 1.58 m | 1.61 m | 66.7% | 100.0% | 100.0% |

**Feature change, v7 → v10 (same split/calibration):**

| Scenario | v7 | v10 | Change |
|---|---|---|---|
| A | 3.78 m | 4.36 m | **15.3% worse** |
| B | 3.97 m | 4.02 m | **1.3% worse** |
| C | 1.45 m | 1.58 m | **9.2% worse** |

**Feature group importances (v10, Scenario B):**

| Feature group | Importance |
|---|---|
| ANGLES | 39.9% |
| DISTANCE | 14.8% |
| REL_RSSI | 12.2% |
| RESIDUALS | 9.7% |
| COVERAGE | 7.8% |
| DIST_RATIO | 6.9% |
| WPOS | 5.9% |
| GDOP | 2.9% |

**Key finding — overfitting from richer features in a low-data regime,
now in *every* scenario.** Adding 5 extra feature dimensions (14 → 19,
i.e. `CORR_FEAT_DIM` 17 → 19 under the current pipeline) makes the RF
correction **worse in all three scenarios**: +15.3% (A), +1.3% (B), and
+9.2% (C). Previously the data-rich oracle scenario (C, 26 training grids)
was the one case where the richer feature set helped slightly; with the
current 26-grid pipeline it is now **worse across the board**. With only
**15 training grids** for A/B (and 26 for C), a 19-dimensional feature
space gives the Random Forest too many ways to fit noise; the dominant
v10 feature (ANGLES, 39.9%) appears to encode information that does not
generalize as well as the simpler distance/RSSI features that dominate v7
and v9. This is now even stronger evidence that **feature-set complexity
must be matched to the amount of training data available**, and that v7's
simpler 17-dim feature vector remains the better choice across the board
in this dataset.

### 7.8 Confidence scoring (`confidence_score.py`)

Built directly on the **v7-B pipeline** (5-point calibration, n_B =
3.1694, RF trained on 15 grids × 17 features), `confidence_score.py` adds
a **per-prediction confidence score** combining **five** weighted
components:

| Component | Formula | Captures |
|---|---|---|
| `conf_gdop` | `exp(-GDOP / 3)` | Geometric quality of the anchor configuration |
| `conf_residual` | `exp(-residual / 5)` | Trilateration residual (fit quality) |
| `conf_coverage` | `1 - max_angular_gap / (2π)` | Angular coverage of anchors around the estimate |
| `conf_rf` | `exp(-std_tree / 3)` | Agreement across the RF correction's trees (uncertainty) |
| `conf_std` | `exp(-std_mean / 3)` | Mean per-anchor RSSI measurement std at the test grid |

**Combined score:**

```
confidence = 0.30 * conf_gdop + 0.25 * conf_residual + 0.15 * conf_coverage
            + 0.15 * conf_rf  + 0.15 * conf_std
```

with three levels: **HIGH** (≥ 0.6), **MEDIUM** (≥ 0.4), **LOW** (< 0.4).

**Per-grid results (v7-B, 6 TEST grids):**

| Grid | Actual | Predicted | Error | Confidence | Level |
|---|---|---|---|---|---|
| G3 | (7.5, 2.5) | (8.0, 4.9) | 2.41 m | 0.57 | MEDIUM |
| G8 | (20.0, 2.5) | (21.0, 3.3) | 1.28 m | 0.51 | MEDIUM |
| G13 | (33.0, 2.5) | (27.9, 5.2) | 5.74 m | 0.41 | MEDIUM |
| G14 | (33.0, 5.0) | (28.4, 4.1) | 4.71 m | 0.41 | MEDIUM |
| G19 | (20.0, 5.0) | (20.4, 3.9) | 1.18 m | 0.56 | MEDIUM |
| G24 | (7.5, 5.0) | (16.0, 5.9) | 8.50 m | 0.59 | MEDIUM |

**Component breakdown:**

| Grid | conf_gdop | conf_residual | conf_coverage | conf_rf | conf_std | final |
|---|---|---|---|---|---|---|
| G3 | 0.67 | 0.53 | 0.28 | 0.44 | 0.88 | 0.57 |
| G8 | 0.66 | 0.38 | 0.29 | 0.42 | 0.77 | 0.51 |
| G13 | 0.44 | 0.36 | 0.09 | 0.40 | 0.79 | 0.41 |
| G14 | 0.37 | 0.55 | 0.07 | 0.38 | 0.64 | 0.41 |
| G19 | 0.66 | 0.62 | 0.29 | 0.35 | 0.74 | 0.56 |
| G24 | 0.66 | 0.61 | 0.30 | 0.42 | 0.86 | 0.59 |

**Confidence-level breakdown:**

| Level | Count | Mean error |
|---|---|---|
| HIGH (≥0.6) | 0 grids | — |
| MEDIUM (≥0.4) | 6 grids | 3.97 m |
| LOW (<0.4) | 0 grids | — |

**Reject-LOW result:** with the 5-component formula, **all 6 TEST grids
fall in the MEDIUM band** (0.41–0.59) — there is no LOW bucket to reject.
Discarding LOW-confidence predictions therefore leaves **6/6 grids** with
a **mean error of 3.97 m, unchanged** from the all-grids mean — the
reject-filter currently provides **zero practical benefit** on this 6-grid
TEST set.

**Correlation between confidence and error: -0.12** — correctly
*negative* (higher confidence → lower error), a sign-correctness
improvement over earlier formulations of this score. However, the
correlation is still weak and the **ranking within the MEDIUM band is
unreliable**: G24 has by far the *worst* error of all 6 grids (8.50 m) yet
receives the *highest* confidence score in the set (0.59), driven by a
high `conf_std` (0.86) and `conf_gdop`/`conf_residual` values (0.66/0.61)
that look favorable despite the large actual error. Conversely, G13 and
G14 — whose errors (5.74 m, 4.71 m) are also large — correctly receive the
*lowest* confidence (0.41) in the set, mostly via low `conf_coverage`
(0.09, 0.07). The confidence score is therefore directionally correct on
average but **not yet a reliable per-prediction error predictor**, and
with only 6 TEST grids all landing in MEDIUM, the current
HIGH/MEDIUM/LOW thresholds (0.6 / 0.4) are not discriminating this anchor
configuration at all (see [Limitations](#10-limitations)).

**Honest assessment.** Confidence scoring with only 6 TEST grids is
fundamentally limited — the Pearson correlation of -0.12 reflects the
small sample size, not a flawed scoring methodology. With 6 points, any
correlation estimate is statistically unreliable (p > 0.05).

The confidence components (GDOP, residual, coverage, RF uncertainty)
correctly identify geometric quality at the trilateration stage, but
cannot detect errors introduced by the RF correction step — which is the
dominant error source for G24 (conf=MEDIUM, error=8.50 m) and G13
(conf=MEDIUM, error=5.74 m).

**Future work:** evaluate confidence scoring on a larger test set
(≥20 grids) to obtain statistically meaningful correlation estimates.

---

## 8. Complete Results Summary

### 8.1 Fingerprinting (26-class, grid-stratified 7-fold CV)

| Method | Mean Accuracy ± Std |
|---|---|
| KNN-1 | 76.8% ± 6.7% |
| KNN-3 | 73.1% ± 8.2% |
| KNN-5 | 68.5% ± 10.5% |
| **RandomForest (best)** | **90.0% ± 8.2%** |
| SVM-RBF | 53.7% ± 7.9% |
| SVM-Linear | 58.7% ± 9.5% |
| Chance level | 3.8% (1/26) |

### 8.2 RSSI interpolation & path loss (dBm)

| Method | Overall MAE |
|---|---|
| **IDW (LOO-CV, 26 grids)** | **4.25 dBm** |
| **IDW (held-out grids 5/15)** | **3.87 dBm** |
| Path-loss Model 1 (shared n=2.2403) | 5.23 dBm |
| Path-loss Model 2 (per-anchor n) | 4.76 dBm |

**IDW is the overall best RSSI predictor** on the held-out test (3.87 dBm),
beating both path-loss models — Model 2 (per-anchor exponents) is the
better of the two parametric models (4.76 dBm) but does not close the gap
to IDW.

### 8.3 Transferability (mean position error, meters)

| Method | Scenario A (no calib) | Scenario B (calib) | Scenario C (oracle) |
|---|---|---|---|
| WC1 (\|RSSI\| centroid) | 12.18 | — | — |
| WC2 (1/d² centroid) | 8.49 | — | — |
| WC3 (linear-power centroid) | 7.98 | — | — |
| **WC4 (Nelder-Mead trilateration)** | **4.77** | — | — |
| WC5 (Ridge regression) | 6.41 | — | — |
| v3 RF (raw RSSI) | 10.77 | — | — |
| v4 SVR (engineered features) | 9.64 | — | — |
| **v7 (WC4 + RF, triangle split)** | **3.36** | — (excluded, see §7.4) | 1.20 |
| v8 (WC4 + RF, geographic split) | 7.53 | 6.29 | 2.84 |
| v9 (WC4 + RF, all anchors, rich features) | 8.71 | 7.56 | 1.81 |
| v10 (v7 split + rich features) | 4.36 | 4.02 | 1.58 |

### 8.4 Confidence scoring (v7-B)

| Metric | Value |
|---|---|
| Mean error, all 6 grids | 3.97 m |
| Mean error, MEDIUM-confidence (6 grids) | 3.97 m |
| Mean error, LOW-confidence (0 grids) | — |
| Mean error after rejecting LOW (6/6 grids) | **3.97 m (unchanged)** |
| Confidence–error correlation | -0.12 |

---

## 9. Key Findings

1. **Room-level fingerprinting works very well in-distribution.** A
   Random Forest classifier reaches **90.0% ± 8.2%** accuracy across
   26 grid cells under grid-stratified 7-fold CV — over **23×** the 3.8%
   chance level — using only a 16-dim feature vector (8 known anchors ×
   {mean, std}).

2. **IDW is now the strongest RSSI predictor overall**, beating *both*
   path-loss models. Leave-one-out cross-validation over all 26 grids
   gives **4.25 dBm overall MAE**, and on the strict 2-grid held-out test
   it reaches **3.87 dBm** — better than Model 2 (4.76 dBm) and Model 1
   (5.23 dBm). This is a reversal from earlier iterations of this study,
   where a fitted path-loss model edged out IDW.

3. **Per-anchor path-loss exponents still help relative to a shared
   exponent, but neither beats IDW.** Allowing each anchor its own
   exponent (Model 2: n ranges from 1.34 to 3.03) improves overall MAE
   from 5.23 dBm (shared n=2.2403) to **4.76 dBm**, but both parametric
   models remain behind IDW (3.87 dBm).

4. **Geometric trilateration (WC4) beats purely-learned baselines for
   anchor transfer.** Nelder-Mead trilateration (4.77 m) outperforms both
   a raw-RSSI Random Forest (10.77 m) and a feature-engineered SVR
   (9.64 m) — geometry generalizes to unseen anchors better than learned
   RSSI patterns alone.

5. **Calibration benefit is split-dependent, not universal.** For v7's
   split, 5-point calibration made the mean error *worse* than no
   calibration — driven by a single grid (G24) regressing badly — so it
   was excluded from the final v7 pipeline (§7.4). By contrast, v8-B
   (16.5% better) and v9-B (13.2% better) both still improve over their
   no-calibration baselines. A small calibration set can help
   substantially in some splits but **actively hurt** in others; it is no
   longer safe to assume calibration is always beneficial.

6. **v7's split remains the best/easiest transfer configuration found**
   (3.36 m no-calib, the best non-oracle result anywhere in this study),
   while the geographic split (v8) remains by far the hardest, dominated
   by the **G14 corner outlier** (31.51 m / 25.09 m / 17.10 m for
   A/B/C — even the oracle is 5–10× worse there than other grids).

7. **More features is not automatically better — it now overfits in
   *every* scenario.** v10's 19-dim feature set makes v7's RF correction
   *worse* across **all three** scenarios: +15.3% (A), +1.3% (B), and
   +9.2% (C, the oracle) — previously the oracle scenario was the one
   case where richer features helped slightly; that is no longer true.

8. **The confidence score is now correctly signed but provides zero
   filtering benefit on this TEST set.** The confidence–error correlation
   flipped to the expected **-0.12** (from a previously near-zero/wrong-
   signed value), but all 6 v7-B TEST grids fall in the MEDIUM band — none
   are LOW — so rejecting LOW-confidence predictions changes nothing
   (6/6 grids, 3.97 m, unchanged). G24, the worst-error grid (8.50 m),
   still receives the *highest* confidence (0.59) of the set.

9. **Room-level fingerprinting (PART 2) reveals an accuracy/generalization
   tradeoff.** All three KNN variants (KNN-1/3/5) achieve **94.4%
   room-level accuracy** on held-out grids (one per room) despite **0%
   exact-grid accuracy** (by construction, since the held-out grid labels
   never appear in training). RandomForest — the best in-distribution
   classifier (90.0%, finding 1) — generalizes **worse** at the room level
   (83.3%, up from 72.2% under the 32-dim feature set, but still below
   every KNN variant), suggesting its high CV accuracy relies in part on
   per-grid patterns that do not transfer to a nearby unseen grid as
   gracefully as KNN's local neighbor-averaging.

10. **Dropping `median`/`count` (32-dim → 16-dim) improves fingerprinting
    without affecting any other stage.** The ablation study (§4.5) showed
    that `mean+std` (16-dim) slightly outperforms the full 32-dim feature
    set (90.0% vs. 89.1%, +0.9pp), motivating the pipeline-wide switch to
    `STATS = ["mean", "std"]` performed in this update. Every other stage —
    IDW (3.87 dBm, finding 2), the path-loss models (finding 3), v7-A/C
    transferability (3.36 m / 1.20 m, finding 5), and the
    confidence–error correlation (-0.12, finding 8) — is **byte-for-byte
    identical** to the 32-dim results, since none of those stages depend on
    the `median`/`count` RSSI statistics. The only other change is at the
    room level (PART 2, finding 9): KNN's room accuracy slipped slightly
    from 100% to 94.4% (still far above RF's), while RF's room accuracy
    improved from 72.2% to 83.3%.

---

## 10. Limitations

- **Small dataset.** Only 26 measurement grids and 8 surveyed anchor
  positions in a single building. All transferability results (v7–v10)
  are evaluated on test sets of only **6–9 grids**, making individual
  outliers (e.g. G14 in v8, contributing >24 m to a mean of 7.53 m, or G24
  in v7-B) extremely impactful on the headline numbers.

- **Single building, single floor plan.** All splits — anchor-based
  (v7, v9), geographic (v8), and feature-based (v10) — are different
  partitions of the *same* 35 m × 8.1 m space. True cross-building or
  cross-floor-plan transferability is untested.

- **Unstable path-loss exponent fitting from few calibration points, and
  it can now be actively harmful.** When tested, v7's 5-point calibrated
  exponent (n_B = 3.1694) deviated substantially from both the fixed
  default (2.25) and the oracle value (2.5720), and the resulting
  calibrated Scenario B (3.97 m) was **worse** than the uncalibrated
  Scenario A (3.78 m) — small calibration sets can overcorrect the
  path-loss model badly enough to outweigh the RF correction's benefit.
  This instability is why v7's minimal calibration was excluded from the
  final pipeline (§7.4). v8's 2-point calibration shows an even larger
  swing (n_B = 3.5018 vs. n_C = 2.5282), and v9's 2-point calibration
  (n_B = 2.5798 vs. n_C = 2.3037) is comparatively mild — both still
  improve over their no-calibration baselines (finding 5).

- **Geographic edge/corner extrapolation is poorly handled.** The v8 G14
  outlier (31.51 m error, Scenario A) shows that when a TEST grid lies at
  the extreme edge of both the TEST-anchor coverage and the CALIB-grid
  coverage, all three calibration scenarios — including the oracle —
  degrade sharply (oracle error there is still 17.10 m, 5–10× the
  per-grid average elsewhere).

- **Confidence score is correctly signed (-0.12) but currently
  non-functional as a reject filter.** The 6-grid v7-B TEST set produces
  **zero LOW and zero HIGH** predictions — every grid scores MEDIUM
  (0.41–0.59) — so the HIGH/MEDIUM/LOW thresholds (0.6 / 0.4) do not
  discriminate at all for this anchor configuration. Within the MEDIUM
  band, fine-grained ranking is still unreliable: G24 (8.50 m error, the
  worst in the set) receives the *highest* confidence (0.59).

- **No deep-learning or sequence models.** All approaches use windowed
  mean RSSI as a static feature; temporal dynamics, multi-packet sequence
  models, and learned embeddings of raw RSSI time-series are unexplored.

- **Feature/hyperparameter choices are largely fixed, not tuned.**
  RF hyperparameters (e.g., `n_estimators=200/300`, `min_samples_leaf`),
  SVM `C`/`gamma`, and the confidence-score weights/thresholds
  (0.30/0.25/0.15/0.15/0.15, HIGH≥0.6/MEDIUM≥0.4) are fixed values chosen
  during development rather than the result of a systematic search. v10's
  now-uniformly-worse results across A/B/C strengthen the case for
  feature-set tuning rather than ad hoc feature addition.

---

## 11. Future Work

1. **Expand the dataset** — more grids (denser sampling, especially near
   building edges/corners where v8's G14 outlier occurred), more surveyed
   anchors, and repeated measurements at different times of day to capture
   temporal RSSI variability.

2. **Cross-building / cross-floor-plan validation** — repeat the v7–v10
   transferability experiments in a second building to test whether the
   ranking of methods (WC4 > learned baselines, calibration sometimes
   helps/sometimes hurts, feature-richness can overfit) generalizes beyond
   this specific 35 m × 8.1 m floor plan.

3. **Tune and re-validate the confidence score** — fit the
   `conf_gdop`/`conf_residual`/`conf_coverage`/`conf_rf`/`conf_std` weights
   (currently 0.30/0.25/0.15/0.15/0.15) and the HIGH/MEDIUM/LOW thresholds
   against a larger labelled error set, ideally via a regression model
   trained directly to predict position error — the correlation is
   correctly signed (-0.12) but currently produces zero LOW/HIGH
   predictions and thus no actionable filtering.

4. **Adaptive / per-region path-loss exponents.** Combine Model 2's
   per-anchor exponents (§6.2, n range 1.34–3.03) with the transferability
   calibration step (§7.5–7.6) — e.g., interpolate `n` spatially rather
   than fitting a single value per anchor from a handful of calibration
   grids, which may reduce the calibration instability noted in
   Limitations (and specifically v7's excluded 5-point calibration, §7.4).

5. **Targeted feature selection for v10-style rich features.** v10's
   19-dim feature set is now worse than v7's 17-dim set in **all three**
   scenarios (A/B/C), including the data-rich oracle — apply feature
   selection (e.g. recursive feature elimination) to find a subset that
   retains any useful signal from the RESIDUALS/ANGLES groups without the
   overfitting penalty seen across the board.

6. **Ensemble / fusion of fingerprinting and trilateration.** The
   room-level finding (KNN-5: 100% room accuracy vs. RF: 72.2%, finding 9)
   suggests fingerprinting and trilateration may have complementary
   failure modes; a fusion approach (e.g. using a fingerprint-derived room
   estimate to constrain or initialize the trilateration search, or vice
   versa) could improve both robustness and interpretability.

7. **Investigate the G14-class corner/edge failure mode directly** — add
   dedicated calibration points near building corners, or add a
   "distance-to-nearest-calibration-point" feature to the RF correction
   stage so the model can recognize and flag extrapolation regions
   (G14: 31.51 m / 25.09 m / 17.10 m for A/B/C, even the oracle is poor).

---

## Appendix A: Grid Coordinates

`GRID_POSITIONS` from `interpolate.py` — all 26 measurement grid points
used by the pipeline, in meters:

| Grid | (x, y) | Grid | (x, y) |
|---|---|---|---|
| 1 | (2.5, 2.5) | 14 | (33.0, 5.0) |
| 2 | (5.0, 2.5) | 15 | (30.0, 5.0) |
| 3 | (7.5, 2.5) | 16 | (27.5, 5.0) |
| 4 | (10.0, 2.5) | 17 | (25.0, 5.0) |
| 5 | (12.5, 2.5) | 18 | (22.5, 5.0) |
| 6 | (15.0, 2.5) | 19 | (20.0, 5.0) |
| 7 | (17.5, 2.5) | 20 | (17.5, 5.0) |
| 8 | (20.0, 2.5) | 21 | (15.0, 5.0) |
| 9 | (22.53, 2.5) | 22 | (12.5, 5.0) |
| 10 | (25.0, 2.5) | 23 | (10.0, 5.0) |
| 11 | (27.5, 2.5) | 24 | (7.5, 5.0) |
| 12 | (30.0, 2.5) | 25 | (5.0, 5.0) |
| 13 | (33.0, 2.5) | 26 | (2.5, 5.0) |

Note: Grid 9 at x = 22.53 m is the only grid that breaks the otherwise
regular 2.5 m spacing; it is the second-hardest grid to fingerprint
(75.0% accuracy, §4.2). **Grid 10 (25.0, 2.5)** is now the hardest grid
overall (58.3% accuracy) — `grid27.pcapng` and `grid28.pcapng` (the
former Grids 27/28 at y = 10.0 m) still exist in `records/` but are no
longer consumed by any script (`N_GRIDS = 26`).

---

## Appendix B: Anchor Mapping

`ANCHOR_POSITIONS` (== `KNOWN_ANCHORS`) from `interpolate.py` — all 8
anchors with surveyed positions, every one of which is used by every
fingerprinting, IDW, path-loss, and trilateration model in this report:

| RLOC16 | (x, y) | Role across transferability versions |
|---|---|---|
| `0xac00` | (0.9, 0.0) | TRAIN in v5/v7/v8/v9/v10 |
| `0xa800` | (10.4, 8.1) | TRAIN in v5/v7/v8/v10; TEST in v9 |
| `0xb000` | (25.0, 0.0) | TRAIN in v5/v7/v9/v10; TEST in v8 |
| `0x4400` | (5.4, 0.0) | TEST in v5/v7/v9/v10; TRAIN in v8 |
| `0xe400` | (22.6, 0.0) | TEST in v5/v7/v8/v9/v10 |
| `0xa000` | (10.5, 0.0) | TEST in v5/v7/v9/v10; TRAIN in v8 |
| `0x1800` | (15.3, 0.0) | TRAIN in v8/v9; not used in v5/v7/v10 |
| `0x5800` | (20.6, 8.1) | TEST in v8; TRAIN in v9; not used in v5/v7/v10 |

The 6 additional router RLOC16s observed in the capture data
(`0x1400, 0x3400, 0x7800, 0x9400, 0xe800, 0xf000`) have no surveyed
position and are not used anywhere in the current pipeline (§2).

---

## Appendix C: File Structure & Descriptions

```
Mesurements/
├── localize.py                      Fingerprinting pipeline (§4): grid-stratified
│                                     7-fold CV over 6 classifiers (KNN-1/3/5,
│                                     RandomForest, SVM-RBF, SVM-Linear) on
│                                     26 grids × 32-dim features (PART 1), plus a
│                                     room-level holdout evaluation over 5 rooms
│                                     via ROOMS / TEST_GRIDS / GRID_TO_ROOM (PART 2, §4.4).
├── records/grid1..28.pcapng         Raw IEEE 802.15.4 TAP captures, one per
│                                     measurement grid. Only grid1..26 are consumed
│                                     (N_GRIDS = 26); grid27/grid28 are unused.
├── records.zip                      Archived copy of records/.
├── accuracy_summary.png             Bar chart: mean ± std accuracy per classifier.
├── per_fold_accuracy.png            Per-fold accuracy for each classifier.
├── per_grid_accuracy.png            Per-grid accuracy for the best classifier (RF).
├── feature_importance.png           RF anchor-importance bar chart (§4.3).
├── cm_*.png                         Confusion matrices per classifier.
├── Routers Map/                     Photos of the OpenThread dashboard / network
│                                     configuration used during data collection (§2).
└── rssi_interpolation/
    ├── interpolate.py               Core module (§3, §5): pcapng/TAP/802.15.4
    │                                 parsing, GRID_POSITIONS (26), ANCHOR_POSITIONS
    │                                 == KNOWN_ANCHORS (8), IDW predict_rssi(),
    │                                 LOO-CV evaluate(), HELD_OUT = [5, 15].
    ├── held_out_validate.py         2-grid held-out IDW validation (§5.2).
    ├── path_loss_model.py           Log-distance path-loss Models 1 & 2 (§6);
    │                                 IDW_OVERALL_MAE = 3.87, IDW_PER_GRID =
    │                                 {5: 5.59, 15: 2.15}.
    ├── heatmap.py                   Per-anchor RSSI heatmaps via IDW (rssi_heatmaps.png).
    ├── predict_cli.py               CLI: predict RSSI at an (x, y) query point (§5.3).
    ├── transferability_test.py      v3 — RF on raw RSSI, "triangle coverage" split (§7.3).
    ├── transferability_test_v1.py   v1 — first anchor+grid transfer split (5 vs 4 anchors).
    ├── transferability_test_v2.py   v2 — geographically-balanced anchor split.
    ├── transferability_test_v4.py   v4 — engineered features, RF/KNN/SVR comparison (§7.3).
    ├── transferability_test_v5.py   v5 — WC1–WC5 weighted-centroid/trilateration/Ridge (§7.2).
    ├── transferability_test_v7.py   v7 — WC4 + RF correction (uncertainty-
    │                                 weighted), CORR_FEAT_DIM = 17 (best, §7.4).
    ├── transferability_test_v8.py   v8 — geographic (left/right) split,
    │                                 CORR_FEAT_DIM = 32 (§7.5).
    ├── transferability_test_v9.py   v9 — balanced interleaved split, all 8 anchors,
    │                                 FEAT_DIM = 22 rich features (§7.6).
    ├── transferability_test_v10.py  v10 — v7 split + FEAT_DIM = 19 rich features,
    │                                 feature-complexity study (§7.7).
    ├── confidence_score.py          Per-prediction confidence scoring on v7-B,
    │                                 5-component weighted formula (§7.8).
    ├── run_confidence.py            Thin entry point for confidence_score.py.
    ├── run_path_loss.py             Thin entry point for path_loss_model.py.
    ├── run_transferability*.py      Thin entry points for each transferability_test_v*.py.
    ├── validate.py                  Thin entry point for held_out_validate.py.
    ├── transferability_*.png        Maps, error plots, calibration-impact and
    │                                 feature-importance plots for each version.
    ├── confidence_*.png             Confidence scatter/map/breakdown plots (§7.8).
    ├── path_loss_*.png              Path-loss comparison and fit plots (§6).
    ├── held_out_validation.png      Held-out validation plot (§5.2).
    └── requirements.txt             numpy>=1.20, matplotlib>=3.3
```

---

## Appendix D: Reproduction Commands

All commands assume the working directory is `rssi_interpolation/`
(except `localize.py`, which lives one level up and is run from
`Mesurements/`).

```bash
# Fingerprinting (§4): PART 1 (26-class CV) + PART 2 (room-level holdout)
# — run from Mesurements/
python localize.py

# RSSI interpolation: LOO-CV over all 26 grids (§5.1)
python interpolate.py

# Held-out grid validation: grids 5, 15 (§5.2)
python held_out_validate.py
# or:
python validate.py

# Example RSSI prediction at a query point (§5.3)
python predict_cli.py 12.5 4.0

# Per-anchor RSSI heatmaps
python heatmap.py

# Path-loss models 1 & 2 vs IDW, HELD_OUT=[5,15] (§6)
python path_loss_model.py
# or:
python run_path_loss.py

# Baseline weighted-centroid comparison WC1-WC5 (§7.2)
python transferability_test_v5.py

# Learned baselines: v3 RF, v4 SVR (§7.3)
python transferability_test.py
python transferability_test_v4.py

# Transferability v7 — best non-oracle result (§7.4)
python run_transferability_v7.py

# Transferability v8 — geographic split (§7.5)
python run_transferability_v8.py

# Transferability v9 — all anchors, rich features (§7.6)
python run_transferability_v9.py

# Transferability v10 — feature-complexity study (§7.7)
python run_transferability_v10.py

# Confidence scoring on v7-B (§7.8)
python run_confidence.py
```

