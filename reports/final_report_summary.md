# Room-Level RSSI Localization — Executive Summary

## Project Goal

This project investigates **room-level indoor localization** using RSSI
measurements from an 8-anchor IEEE 802.15.4 / Thread mesh network deployed
across a 35 m × 8.1 m building, sampled at 26 measurement grid points
grouped into 5 physical rooms. Three complementary approaches were evaluated
using data parsed directly from raw `.pcapng` captures with a custom
pure-Python parser: (1) **fingerprinting** — classifying a device's location
into one of 26 grid cells from its RSSI feature vector; (2) **spatial
interpolation** of RSSI via Inverse Distance Weighting (IDW); and
(3) **path-loss modelling** with shared and per-anchor exponents.

## Key Results

- **Fingerprinting (Ch 4)**: Random Forest reaches **90.0% ± 8.2%** grid-level
  accuracy across 26 grid cells under grid-stratified 7-fold CV — over **23×**
  the 3.8% chance level — using a 16-dim feature vector (8 anchors × {mean, std}).
  KNN and SVM variants reach 58.7–89.5%. At the room level (PART 2, held-out
  one grid per room), KNN achieves **94.4%** room-level accuracy while Random
  Forest reaches **83.3%**.

- **RSSI interpolation / IDW (Ch 5)**: Leave-one-out CV over all 26 grids gives
  **4.25 dBm** overall MAE. On a strict 2-grid held-out test (G5, G15 withheld
  entirely from training), IDW achieves **3.87 dBm** MAE — the best RSSI
  prediction result in this study.

- **Path-loss modelling (Ch 6)**: A shared path-loss exponent model
  (Model 1, n = 2.24) reaches **5.23 dBm** MAE on held-out grids. Allowing
  each anchor its own exponent (Model 2, n ranging 1.34–3.03) improves this to
  **4.76 dBm**, but both parametric models remain behind IDW (3.87 dBm).

## Best Method per Task

| Task | Best method | Result |
|---|---|---|
| Grid-level classification (26 cells) | Random Forest fingerprinting | 90.0% ± 8.2% accuracy |
| Room-level classification (5 rooms) | KNN-1/3/5 fingerprinting | 94.4% room accuracy |
| RSSI prediction — all 26 grids | IDW leave-one-out CV | 4.25 dBm MAE |
| RSSI prediction — held-out grids | IDW (G5, G15 withheld) | 3.87 dBm MAE |
| Path-loss (shared exponent) | Model 1, n = 2.24 | 5.23 dBm MAE |
| Path-loss (per-anchor exponent) | Model 2, n = 1.34–3.03 | 4.76 dBm MAE |

## Main Limitations

- **Single building, single floor plan.** All results are specific to this
  35 m × 8.1 m space with 8 surveyed anchors and 26 grid points. Generalization
  to other buildings, floor plans, or anchor configurations is untested.

- **Small dataset.** With only 26 measurement grids, per-grid statistics are
  based on few samples, and individual grids have an outsized influence on
  headline numbers (e.g. Grid 10, the hardest grid at 58.3% accuracy, pulls
  down the per-grid mean significantly).

- **IDW error is still 2–5 dBm, not a position estimate.** Even the best IDW
  result (3.87 dBm MAE) is an RSSI prediction error, not a position error.
  Converting IDW-predicted RSSI to a 2D position requires a ranging model
  (e.g. path-loss inversion) and trilateration, which introduces additional
  error not yet quantified in this study.

- **No deep-learning or sequence models.** All approaches use windowed mean
  RSSI as a static feature; temporal dynamics and learned embeddings of raw
  RSSI time-series are unexplored.

## Future Direction

Future work focuses on moving toward **transferable, fingerprint-free
localization** using the foundations established in Chapters 5–6:

- Invert per-anchor path-loss models (Ch 6) to estimate anchor distances,
  then apply trilateration to produce a 2D position estimate
- Use IDW (Ch 5) as a fallback where path-loss fits are poor
- Validate the full pipeline across different buildings and anchor layouts
- Apply light per-deployment calibration to adapt path-loss exponents
  without requiring a full fingerprinting survey
