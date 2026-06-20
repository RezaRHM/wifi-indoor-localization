# Wi-Fi Indoor Localization

IEEE 802.15.4 / Thread RSSI-based indoor localization — fingerprinting, IDW spatial interpolation, and log-distance path-loss modelling over 26 measurement grids in a real building.

## Folder structure

```
wifi-indoor-localization/
├── data/
│   └── records/              # Raw pcapng captures — one file per grid position (G1–G26 (active); G27–G28 present but unused by scripts)
├── src/
│   ├── localize.py           # Ch 1–4  RSSI fingerprinting (KNN, RF, SVM), 7-fold CV, room holdout
│   ├── ablation_study.py     # Ch 4    Feature-subset ablation (mean / std / median / count)
│   ├── interpolate.py        # Ch 5    IDW interpolation core — parse + predict_rssi()
│   ├── heatmap.py            # Ch 5    Per-anchor RSSI heatmap generation
│   ├── predict_cli.py        # Ch 5    CLI: predict RSSI at any (x, y) via IDW
│   ├── held_out_validate.py  # Ch 5    IDW held-out validation (G5, G15)
│   ├── validate.py           # Ch 5    Runner for held_out_validate.py
│   ├── path_loss_model.py    # Ch 6    Log-distance path-loss fitting and comparison
│   └── run_path_loss.py      # Ch 6    Runner for path_loss_model.py
├── results/
│   └── figures/              # Output PNGs from Chapters 1–6
├── reports/
│   ├── final_report.md
│   ├── final_report_summary.md
│   ├── generate_html_report_v2.py   # Builds final_report_complete.html
│   └── make_pdf.py                  # Builds final_report.pdf / final_report_summary.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Fingerprinting (Ch 1–4)
python src/localize.py

# IDW heatmaps (Ch 5)
python src/heatmap.py

# IDW held-out validation (Ch 5)
python src/validate.py

# Path-loss modelling (Ch 6)
python src/run_path_loss.py

# Predict RSSI at a point
python src/predict_cli.py 12.5 4.0
```
