#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文 Fig. __ —— 试件与仪器（非数据图）。

(a) 试件与传感器布局：加筋板平面图 + 4 个 AE 探头 + 8 个导波换能器 + 加筋条脚上的
    SMARTape/DFOS + 各试件的冲击位置，并标出两个方向的声速。
(b) 两套互相独立的测量链：AE 与导波，各自从换能器走到同一个自归一化坐标轴。

数据全部来自实测文件，无示意成分：
  · 探头坐标   `l1/meta/ae_sensors.csv`（skin-side, from bottom left）
  · 试件尺寸   `l1/meta/specimens.csv`（skin_length / skin_width）
  · 各向异性   `l1/meta/specimens.csv`（vel_longitudinal / vel_lateral）
  · 冲击位置   `l1/meta/specimens.csv`（imp_x/y 或 loc_x/y）
  · 采集参数   `l1/l1_papers_fulltext.txt`（Broer et al. 2022 = 参考文献 [8]）

(a) **只画 AE 坐标系的量，不画导波换能器的位置**：正文对导波换能器的位置没有
任何论断，而 `structure_info.m` 的 `PZT_info_geo` 用的轴向与 AE 文件不同（它 x 最大
215，大于板宽 165），两套坐标系之间的确切换算无法在仓库内证实 ⇒ 不画，改在 (b)
里以文字给出换能器数与路径数。（曾试过旋转与轴对调两种换算：旋转会把两个换能器
送到离 AE 探头只有 7 mm 处，物理上不可能；对调后最小间距升到 25.5 mm，但换算
依据终究只有「物理上说得通」这一条，不足以进论文。）

加筋条在中宽是**推断**（依据：原文说两条 SMARTape 各贴在一条脚上；FBG 「两侧各 5
通道、沿加筋条排列」；AE 阵列呈平行四边形跨在两侧）。灰带宽度（16 mm）仅示意。

写：l1/figures/fig_instrumentation.png / .pdf
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as PS                                     # noqa: E402

import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402

META = HERE / "meta"
FIG = HERE / "figures"

PS.apply()

# --- 实测常量 ---------------------------------------------
W_PANEL = 165.0          # 板宽 (X, 横向穿过加筋条)  [mm]
L_PANEL = 243.0          # 板长 (Y, 沿加筋条)        [mm]
STIFF_Y = W_PANEL / 2.0  # 加筋条中心线（两侧 FBG 阵列各自沿加筋条排列 ⇒ 居中）
FOOT_HALF = 8.0          # 加筋条脚半宽（示意，仅用于画出「脚」）

AE_SENSORS = {1: (145.0, 190.0), 2: (145.0, 20.0),
              3: (20.0, 50.0), 4: (20.0, 220.0)}            # (x宽, y长) [mm]

C_AE = PS.MODALITY["AE"]
C_GW = PS.MODALITY["PZT"]
C_DFOS = PS.MODALITY["DFOS"]


def read(name: str) -> list[dict]:
    with open(META / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def hits() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in read("specimens.csv"):
        xw = r["imp_x_mm"] or r["loc_x_mm"]
        yl = r["imp_y_mm"] or r["loc_y_mm"]
        if xw and yl:
            xs.append(float(xw))
            ys.append(float(yl))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------- panel (a)
def panel_a(ax) -> None:
    ax.set_aspect("equal")

    ax.add_patch(Rectangle((0, 0), W_PANEL, L_PANEL, facecolor="white",
                           edgecolor="0.20", lw=1.1, zorder=1))

    # 加筋条与两条脚，SMARTape / DFOS 贴在脚表面
    ax.add_patch(Rectangle((0, STIFF_Y - FOOT_HALF), W_PANEL, 2 * FOOT_HALF,
                           facecolor="0.91", edgecolor="0.45", lw=0.6, zorder=2))
    for s in (-1, 1):
        y = STIFF_Y + s * FOOT_HALF
        ax.plot([0, W_PANEL], [y, y], color="0.35", lw=0.7, zorder=3)
        ax.plot([12, W_PANEL - 12], [y + s * 3.2, y + s * 3.2],
                color=C_DFOS, lw=1.6, solid_capstyle="butt", zorder=5)
    ax.text(W_PANEL - 4, STIFF_Y, "stiffener", ha="right", va="center",
            fontsize=5.5, color="0.30", zorder=6)

    # 导波阵列的位置不进本图：见文件头说明

    # AE 探头
    for ch, (x, y) in AE_SENSORS.items():
        ax.plot(x, y, marker="o", ms=4.6, mfc=C_AE, mec="white", mew=0.7, zorder=8)
        ax.text(x, y, str(ch), ha="center", va="center", fontsize=4.5,
                color="white", zorder=9)
    ax.text(133, 190, "AE sensors (4)", ha="right", va="center", fontsize=5.6,
            color=C_AE, zorder=9)

    # 冲击位置
    xs, ys = hits()
    ax.plot(xs, ys, marker="x", ms=3.4, mew=0.9, ls="none", color="0.20",
            alpha=0.70, zorder=6)
    ax.text(0, -14, "impact position of the %d specimens" % xs.size, ha="left",
            va="center", fontsize=5.6, color="0.35")

    # 两个方向的声速：左下空白处画一对小箭头
    vy = np.median([float(r["vel_longitudinal_ms"]) for r in read("specimens.csv")
                    if r["vel_longitudinal_ms"]])
    vx = np.median([float(r["vel_lateral_ms"]) for r in read("specimens.csv")
                    if r["vel_lateral_ms"]])
    ox, oy = 7.0, 7.0
    ax.annotate("", xy=(ox, oy + 26), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.15"), zorder=7)
    ax.annotate("", xy=(ox + 26, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.15"), zorder=7)
    ax.text(ox + 3.5, oy + 15, "$y$", fontsize=5.4, color="0.15", va="center")
    ax.text(ox + 15, oy + 2.5, "$x$", fontsize=5.4, color="0.15", ha="center")

    # 尺寸
    ax.annotate("", xy=(0, L_PANEL + 9), xytext=(W_PANEL, L_PANEL + 9),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color="0.55"))
    ax.text(W_PANEL / 2, L_PANEL + 12.5, "165 mm", ha="center", va="bottom",
            fontsize=5.6, color="0.35")
    ax.annotate("", xy=(W_PANEL + 12, 0), xytext=(W_PANEL + 12, L_PANEL),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color="0.55"))
    ax.text(W_PANEL + 15, L_PANEL / 2, "243 mm", ha="center", va="center",
            fontsize=5.6, color="0.35", rotation=90)

    ax.text(0, L_PANEL + 24, "$v_y$ = %.1f km/s along the stiffener,  "
            "$v_x$ = %.1f km/s across it" % (vy / 1e3, vx / 1e3),
            ha="left", va="bottom", fontsize=5.6, color="0.25")

    ax.set_xlim(-16, 196)
    ax.set_ylim(-24, 286)
    ax.set_xticks([])
    ax.set_yticks([])
    PS.thin(ax, keep=())
    ax.set_title("(a)  specimen and sensor layout", fontsize=7, loc="left", pad=3)


# ---------------------------------------------------------------- panel (b)
BOX_TOP, BOX_BOT, NBOX, GAP = 0.885, 0.145, 4, 0.028
BOX_H = (BOX_TOP - BOX_BOT - (NBOX - 1) * GAP) / NBOX
HEAD_H = 0.042
LINE0 = 0.075
DLINE = 0.034


def box(ax, x, w, y, title, lines, col) -> None:
    ax.add_patch(Rectangle((x, y), w, BOX_H, facecolor="white", edgecolor=col,
                           lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x, y + BOX_H - HEAD_H), w, HEAD_H, facecolor=col,
                           edgecolor="none", zorder=3))
    ax.text(x + 0.012, y + BOX_H - HEAD_H / 2, title, fontsize=5.9,
            color="white", ha="left", va="center", fontweight="bold", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + 0.012, y + BOX_H - LINE0 - i * DLINE, ln, fontsize=5.4,
                color="0.15", ha="left", va="center", zorder=4)


def panel_b(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    PS.thin(ax, keep=())

    CX, BW = [0.02, 0.52], 0.46
    cols = [C_AE, C_GW]
    heads = ["Acoustic emission", "Guided waves"]
    subs = ["29 specimens, C1 to C4", "17 specimens, C1, C3 and C4"]
    chains = [
        [("Transducer", ["4 $\\times$ Vallen VS900-M, 100 to 900 kHz",
                         "clamped on the skin"]),
         ("Acquisition", ["Vallen AMSY-6",
                          "pre-amplifier gain 34 dB",
                          "amplitude threshold 60 dB"]),
         ("Monitored quantity", ["hit count per 500 cycles",
                                 "redrawn on a lifetime axis"]),
         ("On the normalised axis", ["self-normalised cumulative curve",
                                     "$\\rightarrow$ SNHI at $x$ = 85"])],
        [("Transducer", ["8-element array, pitch and catch",
                         "56 paths, 6 bursts 50 to 250 kHz"]),
         ("Acquisition", ["5 cycles, 2 MHz sampling",
                          "flight time by cross-correlation",
                          "parabolic sub-sample interpolation"]),
         ("Monitored quantity", ["drift of the flight time",
                                 "Theil-Sen slope against life"]),
         ("On the normalised axis", ["self-normalised cumulative curve",
                                     "$\\rightarrow$ the same axis"])],
    ]

    for k, col in enumerate(cols):
        x = CX[k]
        ax.text(x, 0.965, heads[k], fontsize=6.6, color=col, ha="left",
                va="center", fontweight="bold")
        ax.text(x + BW, 0.965, subs[k], fontsize=5.4, color="0.40", ha="right",
                va="center")
        ax.plot([x, x + BW], [0.930, 0.930], color=col, lw=0.9)
        y = BOX_TOP - BOX_H
        for j, (t, lines) in enumerate(chains[k]):
            box(ax, x, BW, y, t, lines, col)
            if j < NBOX - 1:
                xm = x + BW / 2
                ax.annotate("", xy=(xm, y - GAP + 0.010), xytext=(xm, y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col))
            y -= (BOX_H + GAP)

    ax.text(0.02, 0.112, "Distributed strain (LUNA ODiSi-B) and fibre Bragg "
            "gratings (sm130) are also", fontsize=5.2, color="0.35", ha="left",
            va="center")
    ax.text(0.02, 0.072, "recorded on the same specimens. The two systems "
            "above share no transducer,", fontsize=5.2, color="0.35", ha="left",
            va="center")
    ax.text(0.02, 0.032, "no instrument and no monitored quantity.",
            fontsize=5.2, color="0.35", ha="left", va="center")
    ax.set_title("(b)  the two measurement systems", fontsize=7, loc="left", pad=3)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.38],
                          left=0.045, right=0.985, top=0.905, bottom=0.025,
                          wspace=0.10)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    PS.save(fig, str(FIG / "fig_instrumentation.png"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
