#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文 Fig. 2 —— Why an absolute alarm threshold cannot be transferred (four routes).

替代原来的四路线表格。四个面板，各自用一个独立角度说明「同一条绝对阈值换到别的
试件上就不成立」：

  (a) alarm time         同一阈值下，各组「达阈时刻」的离散（12 个配置 = 3 档 × 4 持续）
  (b) distributed strain 失效时测到的应变值跨试件的跨度（4 种口径）
  (c) causal anchoring   在寿命 5/10/20/30/50% 处因果重锚定后，失效值的 CV 仍不在 1 附近
  (d) guided waves       导波因果阈值下，批间方差占比 η² 与置换零分布中位（15 个配置）

读：l1/results/{l1_alarm_summary, l1_dfos_caliber_summary, l1_hi_ae_anchor,
                crosslab_variance}.csv
写：l1/figures/fig_four_routes.png / .pdf
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as PS                                  # noqa: E402

import matplotlib.pyplot as plt                           # noqa: E402

RES = HERE / "results"
FIG = HERE / "figures"

PS.apply()

LEVEL_EN = {"注意": "Attention", "警告": "Warning", "严重": "Severe"}
LEVEL_COL = {"注意": "#7c9dc4", "警告": "#b08a3e", "严重": "#a4453b"}
PERSIST = [1, 2, 3, 5]


def read(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str) -> float:
    return float(row[key])


def panel_a(ax) -> None:
    """(a) 达阈时刻的离散：每个配置一根竖棒 [t_min, t_max] + 中位点。"""
    rows = read("l1_alarm_summary.csv")
    xs, spans = [], []
    for r in rows:
        lvl = r["level"]
        gi = list(LEVEL_EN).index(lvl)
        pj = PERSIST.index(int(r["persist"]))
        x = gi * 4 + pj + gi * 0.55
        lo, mid, hi = fnum(r, "t_min"), fnum(r, "t_med"), fnum(r, "t_max")
        col = LEVEL_COL[lvl]
        ax.plot([x, x], [lo, hi], color=col, lw=2.6, solid_capstyle="butt",
                alpha=0.55, zorder=2)
        ax.plot(x, mid, marker="o", ms=2.6, color=col, zorder=4)
        xs.append(x)
        spans.append(hi - lo)
    ax.axhline(100, color="0.75", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(p) for _ in range(3) for p in PERSIST], fontsize=5.6)
    ax.set_ylim(-2, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("lifetime fraction  (%)", fontsize=6.4, labelpad=2)
    ax.set_xlabel("persistence", fontsize=6.2, labelpad=3)
    ax.set_title("(a)  alarm time", fontsize=7, loc="left", pad=4)
    # 三档报警级用颜色区分，另做图例（避免与横轴标签叠字）
    for lvl in LEVEL_EN:
        ax.plot([], [], color=LEVEL_COL[lvl], lw=2.6, alpha=0.55,
                label=LEVEL_EN[lvl])
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.01), fontsize=5.8,
              handlelength=1.2, handletextpad=0.5, borderaxespad=0.0)
    ax.grid(alpha=0.22, axis="y", lw=0.5)
    PS.thin(ax)


def panel_b(ax) -> None:
    """(b) 失效应变值跨试件的跨度（对数轴）。"""
    rows = read("l1_dfos_caliber_summary.csv")
    for i, r in enumerate(rows):
        lo, hi = fnum(r, "fail_min"), fnum(r, "fail_max")
        ax.plot([i, i], [lo, hi], color="#2f4b7c", lw=2.6,
                solid_capstyle="butt", alpha=0.55, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [lo, lo], color="#2f4b7c", lw=0.9)
        ax.plot([i - 0.22, i + 0.22], [hi, hi], color="#2f4b7c", lw=0.9)
        ax.text(i, hi * 1.18, "%.2f$\\times$" % fnum(r, "fail_span"), ha="center",
                fontsize=5.8, color="0.30")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["caliber"] for r in rows], fontsize=6)
    ax.set_ylim(150, 4200)
    ax.set_yticks([200, 500, 1000, 2000, 4000])
    ax.set_yticklabels(["200", "500", "1000", "2000", "4000"])
    ax.minorticks_off()
    ax.set_ylabel("strain at failure", fontsize=6.4, labelpad=2)
    ax.set_title("(b)  distributed strain", fontsize=7, loc="left", pad=4)
    ax.grid(alpha=0.22, axis="y", lw=0.5)
    PS.thin(ax)


def panel_c(ax) -> None:
    """(c) 因果重锚定后的失效率 CV。"""
    rows = [r for r in read("l1_hi_ae_anchor.csv") if r["variant"] == "cum_hits"]
    rows.sort(key=lambda r: int(r["anchor_pct"]))
    x = np.arange(len(rows))
    cv = np.array([fnum(r, "fail_cv") for r in rows])
    ax.axhline(1.0, color="0.55", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.plot(x, cv, marker="o", ms=3.0, lw=1.0, color="#2f4b7c", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r["anchor_pct"] for r in rows], fontsize=6)
    ax.set_ylim(0.7, 1.9)
    ax.set_xlabel("anchor position in life  (%)", fontsize=6.2, labelpad=2)
    ax.set_ylabel("CV of the failure value", fontsize=6.4, labelpad=2)
    ax.set_title("(c)  causal anchoring", fontsize=7, loc="left", pad=4)
    ax.text(0.98, 0.94, "CV = 1 means the failure\nvalues do not coincide",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="0.35")
    ax.grid(alpha=0.22, axis="y", lw=0.5)
    PS.thin(ax)


def panel_d(ax) -> None:
    """(d) 导波因果阈值下的批间方差占比与置换零分布中位。"""
    rows = [r for r in read("crosslab_variance.csv") if r["quantity"].startswith("causal")]
    order = [r for r in rows if r["quantity"].startswith("causal delta_abs")] + \
            [r for r in rows if r["quantity"].startswith("causal roll_slope")]
    n1 = sum(1 for r in order if "delta" in r["quantity"])
    xs, above = [], 0
    for i, r in enumerate(order):
        xi = float(i) if i < n1 else i + 0.5                 # 两组之间留空隙
        xs.append(xi)
        eta = float(r["eta2"])
        mu = float(r["null_median"])
        col = "#2f4b7c" if "delta" in r["quantity"] else "#a4453b"
        ax.plot(xi, eta, marker="o", ms=2.8, color=col, zorder=4)
        ax.plot([xi - 0.32, xi + 0.32], [mu, mu], color="0.25", lw=0.8, zorder=3)
        if eta > mu:
            above += 1
    ax.axvline(n1 - 0.25, color="0.8", lw=0.7)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_xticks([xs[0], xs[n1 - 1], xs[n1], xs[-1]])
    ax.set_xticklabels(["%.2f" % float(order[0]["quantity"].split("@")[1]),
                        "%.2f" % float(order[n1 - 1]["quantity"].split("@")[1]),
                        "%.2f" % float(order[n1]["quantity"].split("@")[1]),
                        "%.2f" % float(order[-1]["quantity"].split("@")[1])],
                       fontsize=6)
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("threshold  (absolute $\\Delta$, then rolling slope)", fontsize=6.2,
                  labelpad=2)
    ax.set_ylabel(r"campaign share  $\eta^2$", fontsize=6.4, labelpad=2)
    ax.set_title("(d)  guided waves", fontsize=7, loc="left", pad=4)
    ax.text(0.02, 0.94, "above its permutation\nmedian in %d of %d" % (above, len(order)),
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color="0.35")
    ax.grid(alpha=0.22, axis="y", lw=0.5)
    PS.thin(ax)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.35))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.115,
                        wspace=0.30, hspace=0.38)
    panel_a(axes[0][0])
    panel_b(axes[0][1])
    panel_c(axes[1][0])
    panel_d(axes[1][1])
    PS.save(fig, str(FIG / "fig_four_routes.png"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
