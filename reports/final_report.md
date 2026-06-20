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

## 7. Key Findings

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

4. **Room-level fingerprinting (PART 2) reveals an accuracy/generalization
   tradeoff.** All three KNN variants (KNN-1/3/5) achieve **94.4%
   room-level accuracy** on held-out grids (one per room) despite **0%
   exact-grid accuracy** (by construction, since the held-out grid labels
   never appear in training). RandomForest — the best in-distribution
   classifier (90.0%, finding 1) — generalizes **worse** at the room level
   (83.3%, up from 72.2% under the 32-dim feature set, but still below
   every KNN variant), suggesting its high CV accuracy relies in part on
   per-grid patterns that do not transfer to a nearby unseen grid as
   gracefully as KNN's local neighbor-averaging.

---

## 8. Limitations

- **No deep-learning or sequence models.** All approaches use windowed
  mean RSSI as a static feature; temporal dynamics, multi-packet sequence
  models, and learned embeddings of raw RSSI time-series are unexplored.

---

## 9. Future Work

Future work focuses on moving toward transferable,
fingerprint-free localization using the foundations
established in Chapters 5–6:

- Invert per-anchor path-loss models to estimate
  anchor distances, then apply trilateration
- Use IDW as a fallback where path-loss fits poorly
- Validate across different buildings and anchor layouts
- Apply light per-deployment calibration to adapt
  path-loss exponents

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
wifi-indoor-localization/
├── data/
│   └── records/grid1..28.pcapng    Raw IEEE 802.15.4 TAP captures, one per
│                                    measurement grid. Only grid1..26 consumed
│                                    (N_GRIDS = 26); grid27/28 exist but unused.
├── src/
│   ├── localize.py                  Fingerprinting pipeline (§4): grid-stratified
│   │                                 7-fold CV over 6 classifiers (KNN-1/3/5,
│   │                                 RandomForest, SVM-RBF, SVM-Linear) on 26 grids
│   │                                 × 16-dim features (mean+std), plus room-level
│   │                                 holdout evaluation (PART 2, §4.4).
│   ├── ablation_study.py            Feature-subset ablation (§4.5): 5 stat combos
│   │                                 (mean-only → mean+std+median+count).
│   ├── interpolate.py               Core IDW module (§5): pcapng parsing,
│   │                                 GRID_POSITIONS (26), ANCHOR_POSITIONS (8),
│   │                                 predict_rssi(), LOO-CV evaluate().
│   ├── heatmap.py                   Per-anchor RSSI heatmaps via IDW (§5.1).
│   ├── predict_cli.py               CLI: predict RSSI at any (x, y) (§5.3).
│   ├── held_out_validate.py         2-grid held-out IDW validation (§5.2).
│   ├── validate.py                  Thin runner for held_out_validate.py.
│   ├── path_loss_model.py           Log-distance path-loss Models 1 & 2 (§6).
│   └── run_path_loss.py             Thin runner for path_loss_model.py.
├── results/
│   └── figures/                     Output PNGs from Chapters 1–6.
├── reports/
│   ├── final_report.md              Full project report (this document).
│   ├── final_report_summary.md      Executive summary.
│   ├── generate_html_report_v2.py   Builds final_report_complete.html.
│   └── make_pdf.py                  Builds PDF reports via reportlab.
├── requirements.txt                 numpy>=1.20, matplotlib>=3.3, scikit-learn, etc.
├── .gitignore
└── README.md
```


---

## Appendix D: Reproduction Commands

All commands run from the repo root (`wifi-indoor-localization/`).

```bash
# Fingerprinting — Ch 4: 26-class CV + room-level holdout
python src/localize.py

# RSSI interpolation — Ch 5: LOO-CV over all 26 grids
python src/interpolate.py

# Held-out grid validation — Ch 5: grids 5, 15
python src/held_out_validate.py
# or:
python src/validate.py

# Example RSSI prediction at a query point — Ch 5
python src/predict_cli.py 12.5 4.0

# Per-anchor RSSI heatmaps — Ch 5
python src/heatmap.py

# Path-loss models 1 & 2 vs IDW — Ch 6
python src/path_loss_model.py
# or:
python src/run_path_loss.py

# Feature-subset ablation study — Ch 4
python src/ablation_study.py
```
