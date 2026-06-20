"""
Build final_report.pdf and final_report_summary.pdf from final_report.md
and final_report_summary.md, embedding the project's plot PNGs.
"""

import os
import re
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, Preformatted, PageBreak, HRFlowable,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Colors / styles
# ---------------------------------------------------------------------------
DARKBLUE = colors.HexColor("#1a3a6b")
GRAY = colors.HexColor("#555555")
LIGHTGRAY = colors.HexColor("#f5f5f5")
CODEBG = colors.HexColor("#f0f0f0")
GRIDLINE = colors.HexColor("#cccccc")

H1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=18, leading=22,
                     textColor=DARKBLUE, spaceBefore=10, spaceAfter=8)
H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=14, leading=18,
                     textColor=DARKBLUE, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=12, leading=15,
                     textColor=GRAY, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                       spaceAfter=6, alignment=TA_LEFT)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=16, spaceAfter=4)
CAPTION = ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=8,
                          leading=10, alignment=TA_CENTER, textColor=GRAY,
                          spaceAfter=14, spaceBefore=2)
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=9, leading=11)
CELL = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=10)
CELL_HDR = ParagraphStyle("CellHdr", parent=CELL, fontName="Helvetica-Bold",
                           textColor=colors.white)

# ---------------------------------------------------------------------------
# Unicode sanitization (base-14 fonts only support WinAnsi glyphs)
# ---------------------------------------------------------------------------
SANITIZE = {
    0x2264: "<=", 0x2265: ">=", 0x2192: "->", 0x03A3: "Sum", 0x03C0: "pi",
    0x221A: "sqrt", 0x2208: "in", 0x2588: "#", 0x2500: "-", 0x2502: "|",
    0x2550: "=", 0x21B3: "->", 0x25BA: ">", 0x0394: "Delta", 0x2248: "~",
}


def sanitize(text):
    return text.translate(SANITIZE)


def inline_md(text):
    text = sanitize(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Markdown links -> plain text (internal anchors are not usable in the PDF)
    text = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8" color="#333333">\1</font>', text)
    return text


# ---------------------------------------------------------------------------
# Markdown tokenizer
# ---------------------------------------------------------------------------
def tokenize(lines):
    tokens = []
    i, n = 0, len(lines)
    buf = None

    def flush():
        nonlocal buf
        if buf is not None:
            tokens.append(buf)
            buf = None

    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            code_lines = []
            i += 1
            while i < n and not lines[i].rstrip("\n").strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            tokens.append(("code", code_lines))
            continue

        if stripped == "":
            flush()
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            tokens.append(("heading", len(m.group(1)), m.group(2)))
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            table_lines = []
            while i < n and lines[i].rstrip("\n").strip().startswith("|"):
                table_lines.append(lines[i].rstrip("\n").strip())
                i += 1
            tokens.append(("table", table_lines))
            continue

        if stripped == "---":
            flush()
            tokens.append(("hr",))
            i += 1
            continue

        m = re.match(r"^([-*])\s+(.*)$", stripped)
        if m:
            flush()
            buf = ("bullet", m.group(2))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush()
            buf = ("numbullet", m.group(1), m.group(2))
            i += 1
            continue

        # continuation or new paragraph
        if buf and buf[0] == "bullet":
            buf = ("bullet", buf[1] + " " + stripped)
        elif buf and buf[0] == "numbullet":
            buf = ("numbullet", buf[1], buf[2] + " " + stripped)
        elif buf and buf[0] == "para":
            buf = ("para", buf[1] + " " + stripped)
        else:
            flush()
            buf = ("para", stripped)
        i += 1

    flush()
    return tokens


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def make_table(lines, avail_width):
    rows = []
    for idx, line in enumerate(lines):
        escaped = line.strip().replace(r"\|", "\x00")
        cells = [c.strip().replace("\x00", "|") for c in escaped.strip("|").split("|")]
        if idx == 1 and all(re.match(r"^:?-+:?$", c) for c in cells):
            continue
        rows.append(cells)

    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append("")

    data = []
    for ri, row in enumerate(rows):
        style = CELL_HDR if ri == 0 else CELL
        data.append([Paragraph(inline_md(c), style) for c in row])

    colw = avail_width / ncols
    t = Table(data, colWidths=[colw] * ncols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARKBLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRIDLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGRAY]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def make_code_block(lines, avail_width):
    text = sanitize("\n".join(lines))
    pre = Preformatted(text, CODE)
    t = Table([[pre]], colWidths=[avail_width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def make_image_flowable(path, max_width=500, max_height=380):
    img = PILImage.open(path)
    iw, ih = img.size
    aspect = ih / float(iw)
    w, h = max_width, max_width * aspect
    if h > max_height:
        h = max_height
        w = h / aspect
    rl = RLImage(path, width=w, height=h)
    rl.hAlign = "CENTER"
    return rl


CAPTIONS = {
    "accuracy_summary.png": "Figure 4.1 -- Classifier accuracy summary (grid-stratified 7-fold CV)",
    "per_fold_accuracy.png": "Figure 4.2 -- Per-fold accuracy by classifier",
    "per_grid_accuracy.png": "Figure 4.3 -- Per-grid accuracy (Random Forest, best classifier)",
    "feature_importance.png": "Figure 4.4 -- Random Forest anchor feature importance",
    "cm_RandomForest.png": "Figure 4.5 -- Confusion matrix (Random Forest)",
    "held_out_validation.png": "Figure 5.1 -- Held-out grid validation (Grids 5, 15, 27)",
    "rssi_heatmaps.png": "Figure 5.2 -- Per-anchor RSSI heatmaps (IDW interpolation)",
    "path_loss_comparison.png": "Figure 6.1 -- Path-loss model comparison vs. IDW",
    "path_loss_fits.png": "Figure 6.2 -- Path-loss model fits per anchor",
    "transferability_map_v7.png": "Figure 7.4.1 -- v7 transferability map (WC4 + RF correction)",
    "transferability_errors_v7.png": "Figure 7.4.2 -- v7 per-grid errors",
    "transferability_calibration_impact.png": "Figure 7.4.3 -- Calibration impact (v7, Scenarios A/B/C)",
    "transferability_map_v8.png": "Figure 7.5.1 -- v8 transferability map (geographic split)",
    "transferability_errors_v8.png": "Figure 7.5.2 -- v8 per-grid errors",
    "transferability_map_v9.png": "Figure 7.6.1 -- v9 transferability map (all-anchor split)",
    "transferability_feature_importance_v9.png": "Figure 7.6.2 -- v9 feature group importances",
    "transferability_final_summary.png": "Figure 7.6.3 -- Version comparison summary",
    "transferability_v7_v10_summary.png": "Figure 7.7.1 -- v7 vs v10 comparison",
    "transferability_feature_importance_v10.png": "Figure 7.7.2 -- v10 feature group importances",
    "confidence_scatter.png": "Figure 7.8.1 -- Confidence vs. error scatter plot",
    "confidence_map.png": "Figure 7.8.2 -- Confidence map by grid",
    "confidence_breakdown.png": "Figure 7.8.3 -- Confidence component breakdown",
}

# Section number -> list of relative image paths to embed at the end of that section
IMAGE_MAP = {
    "4": ["../accuracy_summary.png", "../per_fold_accuracy.png",
          "../per_grid_accuracy.png", "../feature_importance.png",
          "../cm_RandomForest.png"],
    "5": ["held_out_validation.png", "rssi_heatmaps.png"],
    "6": ["path_loss_comparison.png", "path_loss_fits.png"],
    "7.4": ["transferability_map_v7.png", "transferability_errors_v7.png",
            "transferability_calibration_impact.png"],
    "7.5": ["transferability_map_v8.png", "transferability_errors_v8.png"],
    "7.6": ["transferability_map_v9.png",
            "transferability_feature_importance_v9.png",
            "transferability_final_summary.png"],
    "7.7": ["transferability_v7_v10_summary.png",
            "transferability_feature_importance_v10.png"],
    "7.8": ["confidence_scatter.png", "confidence_map.png",
            "confidence_breakdown.png"],
}


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------
def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(letter[0] / 2.0, 0.4 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


def draw_cover(canvas, doc):
    draw_footer(canvas, doc)


def draw_content(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(doc.leftMargin, letter[1] - 0.5 * inch,
                       "Room-Level Localization -- TechHub Assen")
    canvas.setStrokeColor(GRIDLINE)
    canvas.line(doc.leftMargin, letter[1] - 0.55 * inch,
                letter[0] - doc.rightMargin, letter[1] - 0.55 * inch)
    canvas.restoreState()
    draw_footer(canvas, doc)


# ---------------------------------------------------------------------------
# Build final_report.pdf
# ---------------------------------------------------------------------------
def build_main_report():
    doc = SimpleDocTemplate(
        os.path.join(HERE, "final_report.pdf"),
        pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.85 * inch, bottomMargin=0.7 * inch,
        title="Room-Level Localization Using IEEE 802.15.4 Ambient Signals",
    )
    avail_width = doc.width

    story = []

    # --- Cover page ---
    cover_title = ParagraphStyle("CoverTitle", fontName="Helvetica-Bold",
                                  fontSize=26, leading=32, textColor=DARKBLUE,
                                  alignment=TA_CENTER)
    cover_sub = ParagraphStyle("CoverSub", fontName="Helvetica", fontSize=15,
                                leading=20, textColor=GRAY, alignment=TA_CENTER,
                                spaceBefore=14)
    cover_date = ParagraphStyle("CoverDate", fontName="Helvetica", fontSize=12,
                                 leading=16, textColor=GRAY, alignment=TA_CENTER)

    story.append(Spacer(1, 2.6 * inch))
    story.append(Paragraph("Room-Level Localization Using IEEE 802.15.4 "
                            "Ambient Signals", cover_title))
    story.append(Paragraph("TechHub Assen -- Comprehensive Study", cover_sub))
    story.append(Spacer(1, 2.4 * inch))
    story.append(Paragraph("June 2026", cover_date))
    story.append(PageBreak())

    # --- Parse markdown body ---
    md_path = os.path.join(HERE, "final_report.md")
    with open(md_path, "r") as fh:
        lines = fh.readlines()

    tokens = tokenize(lines)

    seen_first_h2 = False
    current_section = None

    def flush_images(section_num):
        if section_num not in IMAGE_MAP:
            return
        for rel_path in IMAGE_MAP[section_num]:
            abs_path = os.path.join(HERE, rel_path)
            if not os.path.exists(abs_path):
                continue
            story.append(Spacer(1, 6))
            story.append(make_image_flowable(abs_path))
            cap = CAPTIONS.get(os.path.basename(rel_path), os.path.basename(rel_path))
            story.append(Paragraph(sanitize(cap), CAPTION))

    for tok in tokens:
        kind = tok[0]

        if kind == "heading":
            level, text = tok[1], tok[2]

            # flush images belonging to the section we are leaving
            flush_images(current_section)

            num_m = re.match(r"^(\d+(?:\.\d+)?)\.?\s+(.*)$", text)
            current_section = num_m.group(1) if num_m else None

            if level == 1:
                style = H1
            elif level == 2:
                style = H2
                if seen_first_h2:
                    story.append(PageBreak())
                seen_first_h2 = True
            else:
                style = H3

            story.append(Paragraph(inline_md(text), style))

        elif kind == "table":
            story.append(make_table(tok[1], avail_width))
            story.append(Spacer(1, 8))

        elif kind == "code":
            story.append(make_code_block(tok[1], avail_width))
            story.append(Spacer(1, 8))

        elif kind == "hr":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.75, color=GRIDLINE))
            story.append(Spacer(1, 4))

        elif kind == "bullet":
            story.append(Paragraph("• " + inline_md(tok[1]), BULLET))

        elif kind == "numbullet":
            story.append(Paragraph(f"{tok[1]}. " + inline_md(tok[2]), BULLET))

        elif kind == "para":
            story.append(Paragraph(inline_md(tok[1]), BODY))

    flush_images(current_section)

    doc.build(story, onFirstPage=draw_cover, onLaterPages=draw_content)
    return doc


# ---------------------------------------------------------------------------
# Build final_report_summary.pdf (single page)
# ---------------------------------------------------------------------------
def build_summary_report():
    doc = SimpleDocTemplate(
        os.path.join(HERE, "final_report_summary.pdf"),
        pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Room-Level Localization -- Executive Summary",
    )
    avail_width = doc.width

    title = ParagraphStyle("STitle", fontName="Helvetica-Bold", fontSize=17,
                            leading=21, textColor=DARKBLUE, alignment=TA_CENTER,
                            spaceAfter=2)
    sub = ParagraphStyle("SSub", fontName="Helvetica", fontSize=10, leading=13,
                          textColor=GRAY, alignment=TA_CENTER, spaceAfter=8)
    h2 = ParagraphStyle("SH2", fontName="Helvetica-Bold", fontSize=11.5,
                         leading=14, textColor=DARKBLUE, spaceBefore=8,
                         spaceAfter=3)
    body = ParagraphStyle("SBody", fontName="Helvetica", fontSize=8.5,
                           leading=11, spaceAfter=3)
    bullet = ParagraphStyle("SBullet", parent=body, leftIndent=12, spaceAfter=2)

    story = []
    story.append(Paragraph("Room-Level Localization Using IEEE 802.15.4 "
                            "Ambient Signals", title))
    story.append(Paragraph("TechHub Assen -- Executive Summary -- June 2026", sub))

    story.append(Paragraph("Project Goal", h2))
    story.append(Paragraph(
        "This project investigates room-level indoor localization using RSSI "
        "measurements from a 9-anchor IEEE 802.15.4 / Thread mesh network "
        "deployed across a 35&nbsp;m &times; 12&nbsp;m building, sampled at "
        "28 measurement grid points. Four complementary approaches were "
        "evaluated from raw .pcapng captures parsed with a custom pure-Python "
        "parser: (1) fingerprinting -- classifying location into one of 28 grid "
        "cells from its RSSI vector; (2) spatial RSSI interpolation via "
        "Inverse Distance Weighting; (3) log-distance path-loss modelling with "
        "shared and per-anchor exponents; and (4) a ten-iteration "
        "transferability study testing whether a trilateration + ML-correction "
        "pipeline trained with one set of anchors can localize using a "
        "different, previously unseen set of anchors.", body))

    story.append(Paragraph("Key Results", h2))
    key_results = [
        "Fingerprinting: Random Forest reaches <b>95.1% +/- 4.3%</b> accuracy across "
        "28 rooms (grid-stratified 7-fold CV) vs. a 3.6% chance level.",
        "RSSI interpolation (IDW): <b>4.44 dBm</b> overall MAE (28-grid LOO-CV); "
        "<b>5.26 dBm</b> on a strict 3-grid held-out test (Grids 5/15/27).",
        "Path-loss modelling: per-anchor exponent model (n ranging 1.21-3.10) "
        "reaches <b>5.08 dBm</b> MAE on held-out grids, beating both the "
        "shared-exponent model (5.51 dBm) and IDW (5.26 dBm).",
        "Trilateration baseline: WC4 (Nelder-Mead) reaches <b>6.70 m</b>, beating "
        "learned baselines RF-on-raw-RSSI (10.43 m) and SVR (9.69 m).",
        "Best transferability result (v7): WC4 + RF correction with 5-pt "
        "calibration reaches <b>4.02 m</b> mean error (5.16 m -> 4.02 m, 22.1% "
        "better), oracle bound 1.84 m.",
        "Geographic split (v8) is far harder: 8.84 m -> 7.71 m with calibration, "
        "dominated by a 32.9 m outlier at the building's far corner (Grid 14).",
        "All-anchor / rich-feature split (v9): 9.00 m -> 7.25 m with "
        "calibration; oracle bound 1.88 m. DISTANCE features dominate (31.7%).",
        "Feature-complexity study (v10): adding 5 extra feature dims to v7 "
        "(14 -> 19) makes results <b>worse</b> with limited data "
        "(5.16 m -> 6.32 m, +22.6%) -- overfitting risk.",
        "Confidence scoring (v7-B): rejecting LOW-confidence predictions "
        "improves mean error from <b>4.02 m to 3.62 m</b> (10.0% reduction, "
        "5/8 grids retained); confidence-error correlation only 0.01.",
    ]
    for k in key_results:
        story.append(Paragraph("• " + k, bullet))

    story.append(Paragraph("Best Method per Scenario", h2))
    table_data = [
        ["Scenario", "Best method", "Result"],
        ["In-distribution room classification", "Random Forest fingerprinting", "95.1% +/- 4.3% accuracy"],
        ["RSSI prediction at arbitrary positions", "IDW (power=2)", "4.44 dBm MAE (LOO-CV)"],
        ["RSSI prediction, held-out grids", "Path-loss Model 2 (per-anchor n)", "5.08 dBm MAE"],
        ["Anchor transfer, interleaved split", "v7: WC4 + RF + 5-pt calibration", "4.02 m mean error"],
        ["Anchor transfer, geographic split", "v8: WC4 + RF + 3-pt calibration", "7.71 m mean error"],
        ["Theoretical upper bound (oracle)", "v7-C oracle calibration", "1.84 m mean error"],
    ]
    cell_style = ParagraphStyle("SCell", fontName="Helvetica", fontSize=8, leading=10)
    cell_hdr = ParagraphStyle("SCellHdr", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white)
    data = []
    for ri, row in enumerate(table_data):
        st = cell_hdr if ri == 0 else cell_style
        data.append([Paragraph(c, st) for c in row])
    col_widths = [avail_width * 0.34, avail_width * 0.36, avail_width * 0.30]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARKBLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRIDLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGRAY]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)

    story.append(Paragraph("Main Limitations", h2))
    limitations = [
        "Small dataset: 28 grids, <=9 surveyed anchors, single building; "
        "transfer test sets are only 6-9 grids, so single outliers (G14) "
        "dominate headline numbers.",
        "Anchor 0x0000 is surveyed but never transmits in the captures, "
        "reducing every '4 TEST anchor' design to 3 usable anchors.",
        "Few-point calibration (3-5 grids) can produce unstable path-loss "
        "exponents (e.g. n_B=3.17 vs. n_C=2.61 in v7).",
        "Geographic edge/corner extrapolation (v8, Grid 14) remains poorly "
        "handled even under oracle calibration (14.06 m error).",
        "Confidence score is a useful coarse reject filter but has near-zero "
        "rank correlation with actual error (0.01) and was not tuned against "
        "ground truth.",
    ]
    for lm in limitations:
        story.append(Paragraph("• " + lm, bullet))

    story.append(Paragraph("Main Future Direction", h2))
    story.append(Paragraph(
        "Expand the dataset (more grids, anchors, and a second building/floor "
        "plan) to validate that v7's calibration + WC4 + RF approach "
        "generalizes, while re-tuning the confidence score as a regression "
        "model trained directly on position error -- closing the gap between "
        "its current coarse reject-filter behavior and a fully reliable "
        "per-prediction uncertainty estimate.", body))

    def draw_summary_footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawCentredString(letter[0] / 2.0, 0.35 * inch, "1")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_summary_footer, onLaterPages=draw_summary_footer)
    return doc


if __name__ == "__main__":
    from pypdf import PdfReader

    build_main_report()
    build_summary_report()

    n_main = len(PdfReader(os.path.join(HERE, "final_report.pdf")).pages)
    n_sum = len(PdfReader(os.path.join(HERE, "final_report_summary.pdf")).pages)

    print(f"PDFs created: final_report.pdf ({n_main} pages), "
          f"final_report_summary.pdf ({n_sum} page{'s' if n_sum != 1 else ''})")
