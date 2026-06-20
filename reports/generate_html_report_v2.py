"""
Build final_report_complete.html from final_report.md (v2).

Converts the markdown report to styled HTML, embeds every PNG in this
directory and the parent directory as base64 inline images, and places
each figure after its relevant section per the layout spec.

v2 updates (post v7-A/C rewrite, §7.4.1, §7.8 honest assessment):
  - P_7_4 figures now placed after §7.4.1 (Corner Failure Analysis), before §7.5
  - P_7_8 anchor updated for the new "Honest assessment" / "Future work" text
  - v7 map / calibration-impact captions updated (no calibration, A vs C)
  - Cover stats bar updated to v7-A = 3.36 m
"""

import base64
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
PARENT = HERE.parent
MD_PATH = HERE / "final_report.md"
OUT_PATH = HERE / "final_report_complete.html"

IMAGE_COUNTER = [0]


def img_b64(filename, directory=HERE):
    data = (directory / filename).read_bytes()
    IMAGE_COUNTER[0] += 1
    return base64.b64encode(data).decode("ascii")


def figure(filename, caption, directory=HERE, width=None):
    b64 = img_b64(filename, directory)
    style = f' style="max-width:{width};"' if width else ""
    return (
        f'<figure class="report-fig">'
        f'<img src="data:image/png;base64,{b64}" alt="{caption}"{style}>'
        f'<figcaption>{caption}</figcaption>'
        f'</figure>'
    )


def fig_grid(items):
    """items: list of (filename, caption, directory)"""
    figs = "".join(figure(f, c, d) for f, c, d in items)
    return f'<div class="fig-grid">{figs}</div>'


# ---------------------------------------------------------------------------
# Build placeholder -> HTML map
# ---------------------------------------------------------------------------

PLACEHOLDERS = {}

PLACEHOLDERS["P_4_1"] = figure(
    "accuracy_summary.png",
    "Mean ± std accuracy per classifier (grid-stratified 7-fold CV, 16-dim mean+std features).",
    PARENT,
)

PLACEHOLDERS["P_4_2"] = (
    figure("per_fold_accuracy.png",
           "Per-fold accuracy for each classifier across the 7 grid-stratified CV folds.",
           PARENT)
    + figure("per_grid_accuracy.png",
             "Per-grid accuracy for the Random Forest classifier (best, 90.0% mean ± 8.2%).",
             PARENT)
)

PLACEHOLDERS["P_4_2B"] = figure(
    "confusion_matrix_rf.png",
    "Random Forest confusion matrix (26×26) — grid-stratified 7-fold CV, "
    "overall accuracy 89.85% (§4.2).",
)

PLACEHOLDERS["P_4_3"] = figure(
    "feature_importance_rf.png",
    "Random Forest feature importance — 16-dim mean+std features, "
    "color-coded by anchor (§4.3).",
)

PLACEHOLDERS["P_4_4"] = figure(
    "room_holdout_accuracy.png",
    "Room-level holdout accuracy per classifier — exact-grid vs room-level (PART 2, §4.4).",
    PARENT,
)

PLACEHOLDERS["P_4_5"] = figure(
    "ablation_results.png",
    "Feature subset ablation study — accuracy by RSSI statistic combination (§4.5).",
)

PLACEHOLDERS["P_5_1"] = figure(
    "rssi_heatmaps.png",
    "Per-anchor RSSI heatmaps interpolated across the building via IDW (§5.1).",
)

PLACEHOLDERS["P_5_2"] = figure(
    "held_out_validation.png",
    "Held-out validation (Grids 5, 15) — predicted vs. measured RSSI per anchor (§5.2).",
)

PLACEHOLDERS["P_6_3"] = figure(
    "path_loss_fits.png",
    "Fitted log-distance path-loss curves (Models 1 & 2) per anchor (§6.3).",
)

PLACEHOLDERS["P_6_4"] = figure(
    "path_loss_comparison.png",
    "MAE comparison: path-loss Models 1/2 vs. IDW on held-out grids (§6.4).",
)

# §7.2 -> v2 / v5 baseline pairs
PLACEHOLDERS["P_7_2"] = (
    figure("transferability_errors_v2.png",
           "v2 — geographically-balanced anchor split: per-grid position error.")
    + figure("transferability_map_v2.png",
             "v2 — geographically-balanced anchor split: TRAIN/TEST anchor & grid map.")
    + figure("transferability_errors_v5.png",
             "v5 — WC1–WC5 weighted-centroid/trilateration variants: per-grid error (§7.2).")
    + figure("transferability_map_v5.png",
             "v5 — WC1–WC5 weighted-centroid/trilateration variants: anchor & grid map (§7.2).")
)

# v3, mid-§7.3
PLACEHOLDERS["P_7_3_V3"] = (
    figure("transferability_errors_v3.png",
           "v3 — Random Forest on raw RSSI: per-grid position error (§7.3).")
    + figure("transferability_map_v3.png",
             "v3 — Random Forest on raw RSSI: anchor & grid map (§7.3).")
)

# end of §7.3: overview map + leftover errors plot + v4
PLACEHOLDERS["P_7_3_END"] = (
    figure("transferability_map.png",
           "Transferability baseline map: TRAIN/TEST anchors and grids (§7.3).")
    + figure("transferability_errors.png",
             "Transferability baseline — per-grid position error (§7.3).")
    + figure("transferability_errors_v4.png",
             "v4 — SVR on engineered features: per-grid position error (§7.3).")
    + figure("transferability_map_v4.png",
             "v4 — SVR on engineered features: anchor & grid map (§7.3).")
)

# end of §7.4.1: G14 corner analysis, v7 map (A vs C), A-vs-C impact chart
PLACEHOLDERS["P_7_4"] = (
    figure("g14_analysis.png",
           "G14 corner-outlier analysis — the worst-predicted grid across the study (§7.4.1).")
    + figure("transferability_map_v7.png",
             "v7 transferability map — WC4 + RF correction (uncertainty-weighted): "
             "Scenario A (no calib) vs Scenario C (oracle), the best non-oracle result (§7.4).")
    + figure("transferability_calibration_impact.png",
             "v7 — WC4 + RF correction: Scenario A (no calib) vs Scenario C (oracle), "
             "per TEST grid (§7.4).")
)

# end of §7.5: v8 map, errors, summary
PLACEHOLDERS["P_7_5"] = (
    figure("transferability_map_v8.png",
           "v8 — geographic split: TRAIN/TEST anchor & grid map (§7.5).")
    + figure("transferability_errors_v8.png",
             "v8 — geographic split: per-grid position error, dominated by the G14 outlier (§7.5).")
    + figure("transferability_summary_v8.png",
             "v8 — geographic split: scenario summary (§7.5).")
)

# end of §7.6: v9 map, errors, feature importance
PLACEHOLDERS["P_7_6"] = (
    figure("transferability_map_v9.png",
           "v9 — balanced interleaved split, all anchors: TRAIN/TEST map (§7.6).")
    + figure("transferability_errors_v9.png",
             "v9 — balanced interleaved split, all anchors: per-grid position error (§7.6).")
    + figure("transferability_feature_importance_v9.png",
             "v9 — feature group importances (Scenario B) (§7.6).")
)

# end of §7.7: v10 map, errors, feature importance, v7-v10 summary
PLACEHOLDERS["P_7_7"] = (
    figure("transferability_map_v10.png",
           "v10 — feature-complexity study: TRAIN/TEST map (same split as v7) (§7.7).")
    + figure("transferability_errors_v10.png",
             "v10 — feature-complexity study: per-grid position error (§7.7).")
    + figure("transferability_feature_importance_v10.png",
             "v10 — feature group importances (Scenario B) (§7.7).")
    + figure("transferability_v7_v10_summary.png",
             "v7 vs. v10 summary — richer features make every scenario worse (§7.7).")
)

# end of §7.8: confidence map, breakdown, scatter
PLACEHOLDERS["P_7_8"] = (
    figure("confidence_map.png",
           "Confidence scoring (v7-B) — spatial map of per-prediction confidence (§7.8).")
    + figure("confidence_breakdown.png",
             "Confidence scoring (v7-B) — component breakdown per TEST grid (§7.8).")
    + figure("confidence_scatter.png",
             "Confidence scoring (v7-B) — confidence vs. position error scatter (§7.8).")
)

# end of section 8: final summary + spatial error map
PLACEHOLDERS["P_8"] = (
    figure("transferability_final_summary.png",
           "Final summary across all transferability versions (v3–v10) (§8).")
    + figure("spatial_error_map.png",
             "Spatial error map across the building (§8).")
)

# Appendix C: confusion-matrix grid
PLACEHOLDERS["P_CM"] = fig_grid([
    ("cm_RandomForest.png", "Confusion matrix — RandomForest (best classifier).", PARENT),
    ("cm_KNN_1.png", "Confusion matrix — KNN-1.", PARENT),
    ("cm_KNN_3.png", "Confusion matrix — KNN-3.", PARENT),
    ("cm_KNN_5.png", "Confusion matrix — KNN-5.", PARENT),
    ("cm_SVM_RBF.png", "Confusion matrix — SVM-RBF.", PARENT),
    ("cm_SVM_Linear.png", "Confusion matrix — SVM-Linear.", PARENT),
])


# ---------------------------------------------------------------------------
# Insert placeholder markers into the markdown source
# ---------------------------------------------------------------------------

md_text = MD_PATH.read_text()

INSERTIONS = [
    (
        "though with a wider spread (±8.2%) than the smaller, less\n"
        "discriminative classifiers.\n\n### 4.2",
        "P_4_1",
    ),
    (
        "### 4.2 Per-grid accuracy (Random Forest, best classifier)\n\n"
        "The Random Forest reaches",
        "P_4_2",
    ),
    (
        "#### Confusion matrix (Random Forest, 26×26)\n\n"
        "The pooled confusion matrix across all 7 folds (overall accuracy",
        "P_4_2B",
    ),
    (
        "exactly as it was the least important\n"
        "of 32 under the previous feature set.\n\n### 4.4",
        "P_4_3",
    ),
    (
        "still consistent\n"
        "with its comparatively low CV accuracy (58.7%) in §4.1.\n\n### 4.5",
        "P_4_4",
    ),
    (
        "confirming\n"
        "that the 16-dim `mean+std` feature vector is the best choice found so far.\n\n"
        "---\n\n## 5.",
        "P_4_5",
    ),
    (
        "`0xb000` is the worst (4.79 dBm).\n\n### 5.2",
        "P_5_1",
    ),
    (
        "giving it the lowest overall\n"
        "error (2.15 dBm) of the two held-out grids.\n\n### 5.3",
        "P_5_2",
    ),
    (
        "| G15 | `0xe400` | -70.75 | -74.14 | -74.23 | 3.39 | 3.48 |\n\n### 6.4",
        "P_6_3",
    ),
    (
        "relying on a single global distance/RSSI relationship.\n\n---\n\n## 7.",
        "P_6_4",
    ),
    (
        "foundation for all later (v7–v10) pipelines.\n\n### 7.3",
        "P_7_2",
    ),
    (
        "**Mean test error = 10.77 m**.\n\n**v4 — SVR on engineered features**",
        "P_7_3_V3",
    ),
    (
        "even the learned model is, in effect, mostly\n"
        "re-deriving a centroid estimate.\n\n### 7.4",
        "P_7_3_END",
    ),
    (
        "or a calibration point near `(33, 5)` to anchor the path-loss fit in\n"
        "   that corner of the building.\n\n### 7.5",
        "P_7_4",
    ),
    (
        "cf. v7, where all TEST grids are interspersed among TRAIN\n"
        "grids).\n\n### 7.6",
        "P_7_5",
    ),
    (
        "a result explored further in v10.\n\n### 7.7",
        "P_7_6",
    ),
    (
        "remains the better choice across the board\n"
        "in this dataset.\n\n### 7.8",
        "P_7_7",
    ),
    (
        "(≥20 grids) to obtain statistically meaningful correlation estimates.\n\n---\n\n## 8.",
        "P_7_8",
    ),
    (
        "| Confidence–error correlation | -0.12 |\n\n---\n\n## 9.",
        "P_8",
    ),
    (
        "numpy>=1.20, matplotlib>=3.3\n```\n\n---\n\n## Appendix D",
        "P_CM",
    ),
]

for anchor, key in INSERTIONS:
    count = md_text.count(anchor)
    if count != 1:
        raise SystemExit(f"Anchor for {key} found {count} times (expected 1):\n{anchor!r}")
    md_text = md_text.replace(anchor, anchor.replace("\n\n", f"\n\nIMGPLACEHOLDER_{key}\n\n", 1), 1)


# ---------------------------------------------------------------------------
# Convert markdown -> HTML
# ---------------------------------------------------------------------------

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                        extension_configs={"toc": {"permalink": False}})
body_html = md.convert(md_text)
toc_html = md.toc

# Replace placeholder paragraphs with actual figure HTML
for key, html in PLACEHOLDERS.items():
    placeholder_p = f"<p>IMGPLACEHOLDER_{key}</p>"
    if placeholder_p not in body_html:
        raise SystemExit(f"Placeholder {key} not found in generated HTML")
    body_html = body_html.replace(placeholder_p, html, 1)


# ---------------------------------------------------------------------------
# Cover page stats bar
# ---------------------------------------------------------------------------

STATS_BAR = """
<div class="stats-bar">
  <div class="stat"><span class="stat-value">90.0% ± 8.2%</span><span class="stat-label">RF fingerprinting accuracy (16-dim, 26 grids)</span></div>
  <div class="stat"><span class="stat-value">3.87 dBm</span><span class="stat-label">IDW MAE, held-out grids 5/15</span></div>
  <div class="stat"><span class="stat-value">4.77 m</span><span class="stat-label">WC4 trilateration (best geometric baseline)</span></div>
  <div class="stat"><span class="stat-value">3.36 m</span><span class="stat-label">v7-A — best non-oracle transferability result</span></div>
  <div class="stat"><span class="stat-value">26</span><span class="stat-label">measurement grids</span></div>
  <div class="stat"><span class="stat-value">8</span><span class="stat-label">surveyed anchors</span></div>
</div>
"""


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------

CSS = """
:root {
  --dark-blue: #0d2c54;
  --mid-blue: #1f4e8c;
  --accent-blue: #2e75d6;
  --light-blue: #eaf1fb;
  --row-alt: #f4f8fd;
  --border: #d7e3f4;
}

* { box-sizing: border-box; }

body {
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #1c1c1c;
  margin: 0;
  background: #fff;
}

.cover {
  background: linear-gradient(135deg, var(--dark-blue) 0%, var(--accent-blue) 100%);
  color: #fff;
  padding: 5rem 2rem 3rem 2rem;
  text-align: center;
}

.cover h1 {
  font-size: 2.4rem;
  margin: 0 0 0.5rem 0;
  color: #fff;
  border: none;
}

.cover .subtitle {
  font-size: 1.1rem;
  opacity: 0.9;
  margin-bottom: 2rem;
}

.stats-bar {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 1.5rem;
  margin-top: 2rem;
}

.stats-bar .stat {
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 10px;
  padding: 1rem 1.4rem;
  min-width: 170px;
}

.stats-bar .stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
}

.stats-bar .stat-label {
  display: block;
  font-size: 0.78rem;
  opacity: 0.85;
  margin-top: 0.3rem;
}

.container {
  max-width: 980px;
  margin: 0 auto;
  padding: 2rem;
}

.toc {
  background: var(--light-blue);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.8rem;
  margin: 2rem 0;
}

.toc h2 {
  margin-top: 0;
  border: none;
  padding: 0;
}

.toc ul {
  padding-left: 1.2rem;
}

.toc a {
  color: var(--mid-blue);
  text-decoration: none;
}

.toc a:hover {
  text-decoration: underline;
}

h1 {
  color: var(--dark-blue);
  margin-top: 2.5rem;
}

h2 {
  color: var(--dark-blue);
  border-left: 6px solid var(--accent-blue);
  padding-left: 0.7rem;
  margin-top: 3rem;
}

h3 {
  color: var(--mid-blue);
  margin-top: 2rem;
}

a { color: var(--accent-blue); }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.2rem 0;
  font-size: 0.92rem;
}

th, td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.7rem;
  text-align: left;
}

thead th {
  background: var(--dark-blue);
  color: #fff;
}

tbody tr:nth-child(even) {
  background: var(--row-alt);
}

code {
  background: #f0f3f8;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  font-size: 0.9em;
}

pre {
  background: #1e2630;
  color: #e6edf3;
  padding: 1rem 1.2rem;
  border-radius: 8px;
  overflow-x: auto;
}

pre code {
  background: none;
  color: inherit;
  padding: 0;
}

hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2.5rem 0;
}

figure.report-fig {
  margin: 1.8rem auto;
  text-align: center;
  max-width: 100%;
}

figure.report-fig img {
  max-width: 100%;
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(13, 44, 84, 0.18);
}

figure.report-fig figcaption {
  margin-top: 0.6rem;
  font-size: 0.85rem;
  color: #555;
  font-style: italic;
}

.fig-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 1.8rem 0;
}

.fig-grid figure.report-fig {
  margin: 0;
}

@media (max-width: 700px) {
  .fig-grid { grid-template-columns: 1fr; }
}

@media print {
  .cover {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  h2, h3 { page-break-after: avoid; }
  table, figure.report-fig, .fig-grid { page-break-inside: avoid; }
  .toc { page-break-after: always; }
  a { color: inherit; text-decoration: none; }
}
"""

HTML_DOC = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room-Level RSSI Localization: A Comprehensive Study</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <h1>Room-Level RSSI Localization</h1>
  <div class="subtitle">A Comprehensive Study &mdash; Fingerprinting, Spatial Interpolation,
  Path-Loss Modelling &amp; Transferability over an IEEE 802.15.4 / Thread Mesh Network</div>
  {STATS_BAR}
</div>
<div class="container">
<div class="toc">
<h2>Table of Contents</h2>
{toc_html}
</div>
{body_html}
</div>
</body>
</html>
"""

OUT_PATH.write_text(HTML_DOC)

size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"Done: {size_mb:.1f} MB, {IMAGE_COUNTER[0]} images embedded")
