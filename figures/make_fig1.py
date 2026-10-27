#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.1 generator for the LMEIC JIIS submission.

Renders the five-stage LMEIC pipeline as a clean, vertically stacked
stage-bar architecture diagram, imitating the layout grammar of the prior
JIIS template figure while using the unified LMEIC colour scheme.

Each stage zone is sized from its actual content, so the vertical flow
arrows always start below all boxes (no arrow/box overlap).

Output: vector PDF (primary) plus a 300-dpi PNG backup.

Version: v3.1 (2026-10-27; revision 1: D2 thread position replaced by prior count, thread-grouped cross-fitting)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

# ---------------------------------------------------------------------------
# Fonts: register Liberation Sans (Arial/Helvetica metric-compatible).
# ---------------------------------------------------------------------------
_LIB_DIR = "/usr/share/fonts/truetype/liberation2"
for _fn in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
            "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    _p = os.path.join(_LIB_DIR, _fn)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Colour scheme (lmeic_figure_color_scheme_20260602.md).
# ---------------------------------------------------------------------------
C = {
    "white":   "#FFFFFF",
    "zone":    "#F6F9F4",
    "d1_bg":   "#F0F7EE",
    "d2_bg":   "#EFF4F7",
    "d3_bg":   "#FDF5EB",
    "neutral_bg": "#F1F3F3",
    "purple_bg":  "#F3F3F8",
    "teal_bg":    "#EFF4F4",
    "d1":      "#71A682",
    "d2":      "#7EA4B6",
    "d3":      "#D19246",
    "llm":     "#4A4F7E",
    "coreader":"#5B8C8A",
    "neutral": "#81989B",
    "ink":     "#2D3436",
    "sub":     "#636E72",
    "border":  "#B5AF8B",
    "arrow":   "#71A682",   # main flow arrow = C3 core palette green
}

# ---------------------------------------------------------------------------
# Canvas (H is generous; savefig bbox='tight' crops to content).
# ---------------------------------------------------------------------------
W = 210.0
H = 280.0
fig = plt.figure(figsize=(W * 0.082, H * 0.082))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

MX = 4.0
RB = 27.0                       # right band reserved for the variant brackets
INW = W - MX - RB               # content width (left-anchored)

# Border line style encoding information accessibility:
#   solid  = derivable from the message text
#   DASH   = requires private organisational history
DASH = (0, (4, 2))

# Font sizes (pt). Card title enlarged to fill the coloured tab (target ratio ~2.1).
FS_STAGE = 22.0
FS_CARD = 20.0
FS_SUB = 19.0
FS_CHIP = 19.0
FS_PILL = 19.0
FS_NOTE = 19.0

# Layout constants
HDR_H = 6.2         # stage header bar height
STRIP_H = 5.9       # card title strip height
LINE = 4.0          # card subtitle line spacing
PAD = 2.4           # padding between lowest content and zone bottom
GAP = 3.0           # arrow gap between stages (short arrows)
ROW_GAP = 2.2       # header-to-cards gap
PILL_GAP = 2.6      # cards-to-pill gap


def rrect(x, y, w, h, fc, ec="none", lw=0.0, rounding=1.4, ls="solid", z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       mutation_aspect=1.0, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def text(x, y, s, size, color=C["ink"], weight="normal", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z)


def arrow_down(xc, y0, y1, color=C["arrow"], lw=2.4):
    ax.add_patch(FancyArrowPatch((xc, y0), (xc, y1), arrowstyle="-|>",
                                 mutation_scale=18, color=color, lw=lw, zorder=4,
                                 shrinkA=0, shrinkB=0))


def arrow_right(yc, x0, x1, color=C["arrow"], lw=2.6):
    ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle="-|>",
                                 mutation_scale=18, color=color, lw=lw, zorder=4,
                                 shrinkA=0, shrinkB=0))


def stage_header(y_top, num, title, color):
    rrect(MX, y_top - HDR_H, INW, HDR_H, fc=color, rounding=1.7, z=3)
    bx, by = MX + 4.6, y_top - HDR_H / 2
    ax.scatter([bx], [by], s=560, c="white", zorder=4, edgecolors="none")
    # under va='center' a digit's ink sits slightly above the geometric centre,
    # so nudge it down to sit centred inside the white circle
    ax.text(bx, by - 0.10 * (FS_STAGE / 5.9), str(num), fontsize=FS_STAGE,
            color=color, fontweight="bold", ha="center", va="center", zorder=6)
    text(bx + 6.2, by, title, FS_STAGE, color="white", weight="bold", ha="left", z=5)
    return y_top - HDR_H


def card(x, y_top, w, h, title, subs, title_color, body_fc="#FFFFFF", border=None, ls="solid"):
    if border is None:
        border = "#000000"
    rrect(x, y_top - h, w, h, fc=body_fc, ec=border, lw=1.8, rounding=1.2, ls=ls, z=2)
    rrect(x, y_top - STRIP_H, w, STRIP_H, fc=title_color, rounding=1.2, z=3)
    text(x + w / 2, y_top - STRIP_H / 2, title, FS_CARD, color="white", weight="bold", z=5)
    # vertically centre the subtitle block within the white body area below the tab
    body_top = y_top - STRIP_H
    body_bot = y_top - h
    centre = (body_top + body_bot) / 2.0
    n = len(subs)
    ty = centre + (n - 1) * LINE / 2.0
    for s in subs:
        text(x + w / 2, ty, s, FS_SUB, color=C["sub"], z=5)
        ty -= LINE
    return y_top - h


def chip_w(s):
    # width scales with font size so 16-pt labels never overflow the pill
    return 0.60 * (FS_CHIP / 5.9) * len(s) + 3.4


def chip(xc, yc, s, color):
    w = chip_w(s)
    rrect(xc - w / 2, yc - 2.8, w, 5.6, fc="white", ec=color, lw=1.1, rounding=1.6, z=4)
    text(xc, yc, s, FS_CHIP, color=color, weight="bold", z=5)


def pill(xc, yc, s, color, w, fc=None, text_color="white", fs=FS_PILL):
    if fc is None:
        fc = color
    rrect(xc - w / 2, yc - 3.4, w, 6.8, fc=fc, ec=color, lw=1.5, rounding=2.6, z=4)
    text(xc, yc, s, fs, color=text_color, weight="bold", z=5)
    return yc - 3.4          # bottom y


def zone(y_top, y_bottom):
    rrect(MX, y_bottom, INW, y_top - y_bottom, fc=C["zone"], rounding=1.4, z=1)


LEG_FS = 16.0


def _tw(s, fs=LEG_FS):
    # rough advance width of a label, in figure units
    return 0.62 * (fs / 5.9) * len(s)


def lg_swatch(x, yc, color, label):
    s = 3.6
    rrect(x, yc - s / 2, s, s, fc=color, ec="none", rounding=0.8, z=7)
    text(x + s + 1.4, yc, label, LEG_FS, color=C["ink"], ha="left", z=7)
    return x + s + 1.4 + _tw(label) + 5.0


def lg_border(x, yc, ls, label):
    bw, bh = 6.0, 3.8
    rrect(x, yc - bh / 2, bw, bh, fc="white", ec="#000000", lw=1.8, rounding=0.8, ls=ls, z=7)
    text(x + bw + 1.4, yc, label, LEG_FS, color=C["ink"], ha="left", z=7)
    return x + bw + 1.4 + _tw(label) + 5.0


def legend(y_top):
    lh = 16.0
    rrect(MX, y_top - lh, INW, lh, fc=C["white"], ec=C["neutral"], lw=1.1, rounding=1.4, z=6)
    r1 = y_top - lh * 0.32
    r2 = y_top - lh * 0.72
    x = MX + 4.0
    for col, lab in [(C["d1"], "D1 content"), (C["d2"], "D2 relational"),
                     (C["d3"], "D3 role-context"), (C["coreader"], "Deep-text co-reader"),
                     (C["llm"], "LLM / decision layer")]:
        x = lg_swatch(x, r1, col, lab)
    x = MX + 4.0
    text(x, r2, "Information accessibility:", LEG_FS, color=C["ink"],
         weight="bold", ha="left", z=7)
    x += _tw("Information accessibility:", LEG_FS) + 4.0
    x = lg_border(x, r2, "solid", "from message text")
    x = lg_border(x, r2, DASH, "requires organisational history")
    return y_top - lh


def vbracket(x, y0, y1, label, color, depth=1.8, lab_dx=2.6):
    # right-side square bracket embracing [y0 (bottom), y1 (top)] with a rotated label
    ax.plot([x, x], [y0, y1], color=color, lw=2.2, zorder=7, solid_capstyle="round")
    ax.plot([x - depth, x], [y0, y0], color=color, lw=2.2, zorder=7, solid_capstyle="round")
    ax.plot([x - depth, x], [y1, y1], color=color, lw=2.2, zorder=7, solid_capstyle="round")
    ax.text(x + lab_dx, (y0 + y1) / 2, label, rotation=90, va="center", ha="center",
            fontsize=17.0, fontweight="bold", color=color, zorder=7)


# ===========================================================================
# Stack the five stages top -> bottom; each zone is sized from its content.
# ===========================================================================
cx = MX + INW / 2
cur = H - 3.0

# ---- STAGE 1 : Email-recipient instance construction ----------------------
yb = stage_header(cur, 1, "Email-recipient instance construction", C["neutral"])
row_top = yb - ROW_GAP
AG = 8.0                # inter-card gap that hosts the horizontal arrows
xs = MX + 2.5
cw = (INW - 2 * 2.5 - 2 * AG) / 3.0
ch = 19.5
card(xs, row_top, cw, ch, "Email text & metadata",
     ["Subject, latest body,", "attachments, thread ID"], C["neutral"])
card(xs + cw + AG, row_top, cw, ch, "Recipient role  $r_i$",
     ["Mgr / CS / Sch", "organisational role"], C["neutral"])
card(xs + 2 * (cw + AG), row_top, cw, ch, "Pre-timestamp history",
     ["Prior emails only", r"($t' < t_i$, no leakage)"], C["neutral"], ls=DASH)
ymid = row_top - ch / 2
arrow_right(ymid, xs + cw + 0.8, xs + cw + AG - 0.8)
arrow_right(ymid, xs + 2 * cw + AG + 0.8, xs + 2 * (cw + AG) - 0.8)
cb = pill(cx, row_top - ch - PILL_GAP - 3.4,
          r"Email-recipient instance   $x_i = (e_i,\; r_i,\; h_i^{<t_i})$", C["d1"], w=126)
s_bottom = cb - PAD
zone(cur, s_bottom)
nxt = s_bottom - GAP
arrow_down(cx, s_bottom, nxt)
cur = nxt

# ---- STAGE 2 : LLM semantic extraction (content domain D1) ----------------
y_s2_top = cur          # top of the variant brackets
yb = stage_header(cur, 2, "LLM semantic extraction \u2014 content domain (D1)", C["llm"])
row_top = yb - ROW_GAP
ch = 22.5
wL = INW * 0.46
wR = INW * 0.46
xL = MX + 2.5
xR = MX + INW - wR - 2.5
card(xL, row_top, wL, ch, "LLM semantic extractor",
     ["Structured semantic reading of", "the email text; the LLM does", "not assign importance labels"], C["llm"])
card(xR, row_top, wR, ch, "D1 content semantic features", [], C["d1"], body_fc=C["d1_bg"])
arrow_right(row_top - ch / 2, xL + wL + 0.4, xR - 0.4)
# descriptor line near the top of the body, chip row near the bottom
text(xR + wR / 2, row_top - STRIP_H - 4.0, "Objective descriptors of email content",
     FS_SUB, color=C["sub"], z=5)
chy = row_top - ch + 3.6
_chips = ["1 categorical", "5 continuous", "6 Boolean"]
_cgap = 2.4
_ws = [chip_w(s) for s in _chips]
_total = sum(_ws) + _cgap * (len(_chips) - 1)
_x = (xR + wR / 2) - _total / 2
for s, _w in zip(_chips, _ws):
    chip(_x + _w / 2, chy, s, C["d1"])
    _x += _w + _cgap
s_bottom = (row_top - ch) - PAD
zone(cur, s_bottom)
nxt = s_bottom - GAP
arrow_down(cx, s_bottom, nxt)
cur = nxt

# ---- STAGE 3 : Explicit multi-source feature channel ----------------------
yb = stage_header(cur, 3, "Explicit multi-source feature channel", C["d1"])
row_top = yb - ROW_GAP
cw = (INW - 5 * 2.4) / 4.0
xs = MX + 2.4
ch = 23.0
card(xs, row_top, cw, ch, "D2 Relational",
     ["Sender frequency,", "reply rate, thread", "depth & prior count"],
     C["d2"], body_fc=C["d2_bg"], ls=DASH)
card(xs + (cw + 2.4), row_top, cw, ch, "D3 Role-context",
     ["Role indicators,", "role-specific keywords,", "sender-to-role history"],
     C["d3"], body_fc=C["d3_bg"], ls=DASH)
card(xs + 2 * (cw + 2.4), row_top, cw, ch, "Lexical text",
     ["Word / char TF-IDF", "with truncated SVD"],
     C["neutral"], body_fc=C["neutral_bg"])
card(xs + 3 * (cw + 2.4), row_top, cw, ch, "Cross-source",
     ["D1 cues \u00d7 relational", "and role evidence,", "role-conditional terms"],
     C["llm"], body_fc=C["purple_bg"], ls=DASH)
cb = pill(cx, row_top - ch - PILL_GAP - 3.4,
          "Concatenated with D1  \u2192  explicit multi-source feature set (LMEIC-explicit)",
          C["d1"], w=150, fc="white", text_color=C["d1"], fs=FS_NOTE)
s_bottom = cb - PAD
zone(cur, s_bottom)
nxt = s_bottom - GAP
arrow_down(cx, s_bottom, nxt)
cur = nxt

# ---- STAGE 4 : Deep-text co-reader (residual text channel) ----------------
y_s3_bot = s_bottom     # bottom of the LMEIC-explicit bracket
yb = stage_header(cur, 4, "Deep-text co-reader \u2014 residual text channel", C["coreader"])
row_top = yb - ROW_GAP
ch = 19.0
wL = INW * 0.46
wR = INW * 0.46
xL = MX + 2.5
xR = MX + INW - wR - 2.5
card(xL, row_top, wL, ch, "Fine-tuned XLM-R co-reader",
     ["Text-only input;", "thread-grouped 5-fold cross-fitting"],
     C["coreader"], body_fc=C["teal_bg"])
card(xR, row_top, wR, ch, "Text-only class probabilities",
     [r"$p^{\mathrm{text}} = [\,p_H,\; p_M,\; p_L\,]$", "compact residual signal"],
     C["coreader"], body_fc=C["teal_bg"])
arrow_right(row_top - ch / 2, xL + wL + 0.4, xR - 0.4)
s_bottom = (row_top - ch) - PAD
zone(cur, s_bottom)
nxt = s_bottom - GAP
arrow_down(cx, s_bottom, nxt)
cur = nxt

# ---- STAGE 5 : Decision calibration layer ---------------------------------
y_s4_bot = s_bottom     # bottom of the LMEIC-full bracket
yb = stage_header(cur, 5, "Decision calibration layer", C["llm"])
row_top = yb - ROW_GAP
pill(cx, row_top - 3.4,
     r"Multi-source feature vector   $[\,z^{D1},\, z^{D2},\, z^{D3},\, z^{T},\, z^{C},\, z^{P}\,]$",
     C["llm"], w=144, fc=C["purple_bg"], text_color=C["llm"], fs=FS_NOTE)
row2 = row_top - 10.5
AG = 8.0                # inter-card gap that hosts the horizontal arrows
xs = MX + 2.5
cw = (INW - 2 * 2.5 - 2 * AG) / 3.0
ch = 18.0
card(xs, row2, cw, ch, "Base learners",
     ["XGBoost + LightGBM", "on the fused vector"], C["llm"], body_fc=C["purple_bg"])
card(xs + cw + AG, row2, cw, ch, "Meta-learner",
     ["Logistic regression", "(stacking)"], C["llm"], body_fc=C["purple_bg"])
card(xs + 2 * (cw + AG), row2, cw, ch, "Calibrated output",
     [r"$\hat{y}\in$ {High, Medium, Low}", r"probabilities $\hat{p}$"],
     C["llm"], body_fc=C["purple_bg"])
arrow_right(row2 - ch / 2, xs + cw + 0.8, xs + cw + AG - 0.8)
arrow_right(row2 - ch / 2, xs + 2 * cw + AG + 0.8, xs + 2 * (cw + AG) - 0.8)
s_bottom = (row2 - ch) - PAD
zone(cur, s_bottom)

# ---- Output label chips ----------------------------------------------------
nxt = s_bottom - GAP
arrow_down(cx, s_bottom, nxt)
out_bottom = nxt - 4.0 - 3.4
for i, (lab, col) in enumerate([("High", C["d3"]), ("Medium", C["d1"]), ("Low", C["neutral"])]):
    pill(cx - 30 + i * 30, nxt - 4.0, lab, col, w=26)

# ---- Legend (colour key + information-accessibility key), at the bottom -----
leg_bottom = legend(out_bottom - 4.5)

# ---- Variant brackets (LMEIC-explicit / LMEIC-full) in the right band ------
bx0 = MX + INW
vbracket(bx0 + 3.5, y_s3_bot, y_s2_top, "LMEIC-explicit", C["d1"])
vbracket(bx0 + 14.0, y_s4_bot, y_s2_top, "LMEIC-full", C["llm"])

# ---------------------------------------------------------------------------
# Crop the canvas exactly to the drawn content (no blank band) and export.
# ---------------------------------------------------------------------------
M = 1.5
y_top_extent = (H - 3.0) + M
y_bot_extent = leg_bottom - M
x_left = MX - 1.5
x_right = bx0 + 17.5
ax.set_xlim(x_left, x_right)
ax.set_ylim(y_bot_extent, y_top_extent)
fig.set_size_inches((x_right - x_left) * 0.082, (y_top_extent - y_bot_extent) * 0.082)

OUT = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(OUT, "fig1_v2.pdf"), pad_inches=0.0)
fig.savefig(os.path.join(OUT, "fig1_v2.png"), dpi=300, pad_inches=0.0)
print("written fig1_v2.pdf / fig1_v2.png")
