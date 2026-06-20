# Room-Level RSSI Localization — Executive Summary

## Project Goal

This project investigates **room-level indoor localization** using RSSI
measurements from a 9-anchor IEEE 802.15.4 / Thread mesh network deployed
across a 35 m × 12 m building, sampled at 28 measurement grid points. Four
complementary approaches were evaluated using data parsed directly from
raw `.pcapng` captures with a custom pure-Python parser: (1) **fingerprinting**
— classifying a device's location into one of 28 grid cells from its RSSI
vector; (2) **spatial interpolation** of RSSI via Inverse Distance
Weighting (IDW); (3) **path-loss modelling** with shared and per-anchor
exponents; and (4) a ten-iteration **transferability study** asking whether
a trilateration + machine-learning correction pipeline trained with one set
of anchors and reference points can localize using a different,
previously-unseen set of anchors — the central practical question for
redeploying or extending such a system.

## Key Results

- **Fingerprinting**: Random Forest achieves **95.1% ± 4.3%** accuracy
  across 28 rooms (grid-stratified 7-fold CV), vs. a 3.6% chance level.
  KNN and SVM variants all reach 82.7–86.0%.
- **RSSI interpolation (IDW)**: **4.44 dBm** overall MAE via 28-grid
  leave-one-out CV; **5.26 dBm** on a strict 3-grid held-out test
  (Grids 5/15/27).
- **Path-loss modelling**: a per-anchor exponent model (Model 2, n ranging
  1.21–3.10) reaches **5.08 dBm** MAE on held-out grids, slightly beating
  both the shared-exponent model (5.51 dBm, n=2.2548) and IDW (5.26 dBm).
- **Transferability baselines**: weighted-centroid trilateration (WC4,
  Nelder-Mead) reaches **6.70 m**, beating learned baselines — RF on raw
  RSSI (10.43 m) and SVR on engineered features (9.69 m).
- **Best transferability result (v7)**: WC4 + Random Forest correction with
  5-point calibration reaches **4.02 m mean error** (down from 5.16 m with
  no calibration, a 22.1% improvement), vs. an oracle (full recalibration)
  bound of 1.84 m.
- **Geographic split (v8)** is far harder: 8.84 m → 7.71 m with calibration,
  dominated by a single 32.9 m outlier at the building's far corner (Grid 14).
- **All-anchor / rich-feature split (v9)**: 9.00 m → 7.25 m with
  calibration, oracle bound 1.88 m; DISTANCE-related features dominate
  (31.7% importance).
- **Feature-complexity study (v10)**: adding 5 extra feature dimensions to
  v7's pipeline (14 → 19) makes results **worse** with limited training
  data (5.16 m → 6.32 m no-calib, +22.6%), demonstrating overfitting risk.
- **Confidence scoring (v7-B)**: rejecting LOW-confidence predictions
  improves mean error from **4.02 m to 3.62 m** (10.0% reduction, retaining
  5/8 grids), though confidence-error correlation is only 0.01.

## Best Method per Scenario

| Scenario | Best method | Result |
|---|---|---|
| In-distribution room classification | Random Forest fingerprinting | 95.1% ± 4.3% accuracy |
| RSSI prediction at arbitrary positions | IDW (power=2) | 4.44 dBm MAE (LOO-CV) |
| RSSI prediction, held-out grids | Path-loss Model 2 (per-anchor n) | 5.08 dBm MAE |
| Anchor transfer, interleaved split | v7: WC4 + RF + 5-pt calibration | 4.02 m mean error |
| Anchor transfer, geographic split | v8: WC4 + RF + 3-pt calibration | 7.71 m mean error |
| Theoretical upper bound (oracle calibration) | v7-C | 1.84 m mean error |

## Main Limitations

- Small dataset: 28 grids, ≤9 surveyed anchors, single building/floor plan;
  transfer test sets are only 6–9 grids, so single outliers (G14) dominate
  headline numbers.
- Anchor `0x0000` is surveyed but never transmits in the captures, reducing
  every "4 TEST anchor" design to 3 usable anchors.
- Calibration with few points (3–5 grids) can produce unstable path-loss
  exponents (e.g. n_B=3.17 vs. n_C=2.61 in v7).
- Geographic edge/corner extrapolation (v8, Grid 14) remains poorly handled
  even under oracle calibration (14.06 m error).
- The confidence score is a useful coarse reject filter but has near-zero
  rank correlation with actual error (0.01) and was not tuned against
  ground truth.

## Main Future Direction

Expand the dataset (more grids, anchors, and a second building/floor plan)
to validate that v7's calibration + WC4 + RF approach generalizes, while
re-tuning the confidence score as a regression model trained directly on
position error — closing the gap between its current coarse reject-filter
behavior and a fully reliable per-prediction uncertainty estimate.
