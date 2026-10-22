"""论文图统一入口：四张正文图共用一个入口，版式仍走 `paper_style`。

原来分散在 fig_protocol / fig_instrumentation / fig_four_routes / fig_two_systems
四个文件里，现在合并为一个；每个图的函数带前缀（proto_ / instr_ / routes_ / systems_）。

用法：
    python l1/figs.py                     # 四张全画
    python l1/figs.py --only protocol     # 只重画 Fig. 1
    python l1/figs.py --only routes,systems
产物：l1/figures/*.png 与同名 .pdf（300 dpi）
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import paper_style as PS                  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
FIG = HERE / 'figures'
META = HERE / 'meta'

ENTRY = {
    'protocol': ('Fig. 1 跨机构对齐协议（三段式）', 'proto'),
    'instrumentation': ('Fig. 2 试件与仪器（非数据图）', 'instr'),
    'routes': ('Fig. 4 绝对阈值不可迁移的四条路线', 'routes'),
    'systems': ('Fig. 3 两套独立系统对照', 'systems'),
}


BCOL = PS.BATCH


def proto_panel_a(ax):
    """(A) 原始记录：三个批次各在自己的墙钟上，间隔与原点都不同。"""
    rng = np.random.default_rng(7)
    rows = [('C1', 2.0, 34, BCOL['C1']), ('C2', 1.0, 26, BCOL['C2']),
            ('C3', 0.0, 42, BCOL['C3'])]
    for name, y, n, col in rows:
        x = np.sort(rng.uniform(0.0, 1.0, n))
        ax.vlines(x, y - 0.17, y + 0.17, color=col, lw=0.7)
        ax.text(1.05, y, name, fontsize=6.3, color=col, va='center', ha='left')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.62, 2.62)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '', '1'])
    ax.set_xlabel('wall-clock  (batches not concurrent)', fontsize=6.2, labelpad=2)
    ax.set_title('(A)  raw records', fontsize=7, loc='left', pad=4)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


def proto_panel_b(ax):
    """(B) 两个轴：横轴换寿命分数、纵轴自归一化，之后曲线重合。"""
    x = np.linspace(0.0, 1.0, 600)
    curves = [(0.66, 2.3, BCOL['C1']), (0.55, 1.8, BCOL['C2']),
              (0.74, 2.7, BCOL['C3'])]
    for tau, beta, col in curves:
        y = 1.0 - np.exp(-(x / tau) ** beta)
        y = y / y[-1]                       # 归一化到终值 1
        ax.plot(x, y, color=col, lw=1.05)
        j = int(np.argmin(np.abs(y - 0.5)))
        ax.plot(x[j], y[j], marker='o', ms=2.4, color=col, zorder=4)
    for yv, lab in ((0.50, r'$t_{50}$'), (0.85, r'$t_{85}$')):
        ax.axhline(yv, color='0.55', lw=0.6, ls=(0, (4, 2.2)), zorder=1)
        ax.text(0.012, yv + 0.022, lab, fontsize=6.3, color='0.32', va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('lifetime fraction\nstate index $\\times$ documented interval $/\\,n_f$',
                  fontsize=6.2, labelpad=3, linespacing=1.35)
    ax.set_ylabel('self-normalised cumulative\n$C(t)\\,/\\,C(T)$',
                  fontsize=6.0, labelpad=3.0, linespacing=1.45)
    ax.set_title('(B)  the two axes', fontsize=7, loc='left', pad=4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def proto_panel_c(ax):
    """(C) 可比：四个活动的 t85 落在同一轴上（真实数据）。"""
    df = pd.read_csv(os.path.join(RES, 'l1_hi_crosslab.csv'))
    d = df[df['variant'] == 'cum_hits'].dropna(subset=['t85_pct'])
    order = ['C1', 'C2', 'C3', 'C4']
    for i, b in enumerate(order):
        v = d.loc[d['batch'] == b, 't85_pct'].to_numpy(dtype=float)
        if v.size == 0:
            continue
        y = float(len(order) - 1 - i)
        ax.plot(v, np.full(v.shape, y), marker='o', ls='none',
                ms=2.5, color=BCOL[b], alpha=0.9)
        med = float(np.median(v))
        ax.vlines(med, y - 0.30, y + 0.30, color='0.15', lw=0.9, zorder=5)
        ax.text(-3.0, y, b, fontsize=6.3, color=BCOL[b], ha='right', va='center')
    ax.text(52.0, 3.62, 'batch medians within 4.85 pp', fontsize=5.8,
            color='0.35', ha='center', va='top')
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.75, 3.75)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(r'$t_{85}$  (% of life)', fontsize=6.2, labelpad=2)
    ax.set_title('(C)  comparable', fontsize=7, loc='left', pad=4)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


def proto_main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.35))

    # 三段的顶边与底边对齐（0.805 / 0.225），使三个标题在同一水线上
    ax_a = fig.add_axes([0.045, 0.225, 0.170, 0.580])
    ax_b = fig.add_axes([0.328, 0.225, 0.320, 0.580])
    ax_c = fig.add_axes([0.762, 0.225, 0.198, 0.580])

    proto_panel_a(ax_a)
    proto_panel_b(ax_b)
    proto_panel_c(ax_c)

    PS.save(fig, os.path.join(FIG, 'fig_protocol.png'))


W_PANEL = 165.0          # 板宽 (X, 横向穿过加筋条)  [mm]


L_PANEL = 243.0          # 板长 (Y, 沿加筋条)        [mm]


STIFF_Y = W_PANEL / 2.0  # 加筋条中心线（两侧 FBG 阵列各自沿加筋条排列 ⇒ 居中）


FOOT_HALF = 8.0          # 加筋条脚半宽（示意，仅用于画出「脚」）


AE_SENSORS = {1: (145.0, 190.0), 2: (145.0, 20.0),
              3: (20.0, 50.0), 4: (20.0, 220.0)}            # (x宽, y长) [mm]


C_AE = PS.MODALITY["AE"]


C_GW = PS.MODALITY["PZT"]


C_DFOS = PS.MODALITY["DFOS"]


def instr_read(name: str) -> list[dict]:
    with open(META / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def instr_hits() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in instr_read("specimens.csv"):
        xw = r["imp_x_mm"] or r["loc_x_mm"]
        yl = r["imp_y_mm"] or r["loc_y_mm"]
        if xw and yl:
            xs.append(float(xw))
            ys.append(float(yl))
    return np.array(xs), np.array(ys)


def instr_panel_a(ax) -> None:
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
    xs, ys = instr_hits()
    ax.plot(xs, ys, marker="x", ms=3.4, mew=0.9, ls="none", color="0.20",
            alpha=0.70, zorder=6)
    ax.text(0, -14, "impact position of the %d specimens" % xs.size, ha="left",
            va="center", fontsize=5.6, color="0.35")

    # 两个方向的声速：左下空白处画一对小箭头
    vy = np.median([float(r["vel_longitudinal_ms"]) for r in instr_read("specimens.csv")
                    if r["vel_longitudinal_ms"]])
    vx = np.median([float(r["vel_lateral_ms"]) for r in instr_read("specimens.csv")
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


BOX_TOP, BOX_BOT, NBOX, GAP = 0.885, 0.145, 4, 0.028


BOX_H = (BOX_TOP - BOX_BOT - (NBOX - 1) * GAP) / NBOX


HEAD_H = 0.042


LINE0 = 0.075


DLINE = 0.034


def instr_box(ax, x, w, y, title, lines, col) -> None:
    ax.add_patch(Rectangle((x, y), w, BOX_H, facecolor="white", edgecolor=col,
                           lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x, y + BOX_H - HEAD_H), w, HEAD_H, facecolor=col,
                           edgecolor="none", zorder=3))
    ax.text(x + 0.012, y + BOX_H - HEAD_H / 2, title, fontsize=5.9,
            color="white", ha="left", va="center", fontweight="bold", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + 0.012, y + BOX_H - LINE0 - i * DLINE, ln, fontsize=5.4,
                color="0.15", ha="left", va="center", zorder=4)


def instr_panel_b(ax) -> None:
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
            instr_box(ax, x, BW, y, t, lines, col)
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


def instr_main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.38],
                          left=0.045, right=0.985, top=0.905, bottom=0.025,
                          wspace=0.10)
    instr_panel_a(fig.add_subplot(gs[0, 0]))
    instr_panel_b(fig.add_subplot(gs[0, 1]))
    PS.save(fig, str(FIG / "fig_instrumentation.png"))


LEVEL_EN = {"注意": "Attention", "警告": "Warning", "严重": "Severe"}


LEVEL_COL = {"注意": "#7c9dc4", "警告": "#b08a3e", "严重": "#a4453b"}


PERSIST = [1, 2, 3, 5]


def routes_read(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def routes_fnum(row: dict, key: str) -> float:
    return float(row[key])


def routes_panel_a(ax) -> None:
    """(a) 达阈时刻的离散：每个配置一根竖棒 [t_min, t_max] + 中位点。"""
    rows = routes_read("l1_alarm_summary.csv")
    xs, spans = [], []
    for r in rows:
        lvl = r["level"]
        gi = list(LEVEL_EN).index(lvl)
        pj = PERSIST.index(int(r["persist"]))
        x = gi * 4 + pj + gi * 0.55
        lo, mid, hi = routes_fnum(r, "t_min"), routes_fnum(r, "t_med"), routes_fnum(r, "t_max")
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


def routes_panel_b(ax) -> None:
    """(b) 失效应变值跨试件的跨度（对数轴）。"""
    rows = routes_read("l1_dfos_caliber_summary.csv")
    for i, r in enumerate(rows):
        lo, hi = routes_fnum(r, "fail_min"), routes_fnum(r, "fail_max")
        ax.plot([i, i], [lo, hi], color="#2f4b7c", lw=2.6,
                solid_capstyle="butt", alpha=0.55, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [lo, lo], color="#2f4b7c", lw=0.9)
        ax.plot([i - 0.22, i + 0.22], [hi, hi], color="#2f4b7c", lw=0.9)
        ax.text(i, hi * 1.18, "%.2f$\\times$" % routes_fnum(r, "fail_span"), ha="center",
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


def routes_panel_c(ax) -> None:
    """(c) 因果重锚定后的失效率 CV。"""
    rows = [r for r in routes_read("l1_hi_ae_anchor.csv") if r["variant"] == "cum_hits"]
    rows.sort(key=lambda r: int(r["anchor_pct"]))
    x = np.arange(len(rows))
    cv = np.array([routes_fnum(r, "fail_cv") for r in rows])
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


def routes_panel_d(ax) -> None:
    """(d) 导波因果阈值下的批间方差占比与置换零分布中位。"""
    rows = [r for r in routes_read("crosslab_variance.csv") if r["quantity"].startswith("causal")]
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


def routes_main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.35))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.115,
                        wspace=0.30, hspace=0.38)
    routes_panel_a(axes[0][0])
    routes_panel_b(axes[0][1])
    routes_panel_c(axes[1][0])
    routes_panel_d(axes[1][1])
    PS.save(fig, str(FIG / "fig_four_routes.png"))


BATCHES = ['C1', 'C2', 'C3', 'C4']


C_CAMP, C_UNI = '#b4b4b4', '#2f4b7c'          # 批次（灰）对机构（蓝）


QUANT = ['AE cum_hits t50_pct', 'AE cum_hits t85_pct', 'PZT tof_slope_ts']


XTICK = ['AE $t_{50}$', 'AE $t_{85}$', 'PZT slope']


def systems_load():
    ae = pd.read_csv(os.path.join(RES, 'l1_hi_crosslab.csv'))
    ae = ae[(ae['variant'] == 'cum_hits')].dropna(subset=['t85_pct'])
    pz = pd.read_csv(os.path.join(RES, 'pzt_crosslab_summary.csv'))
    pz = pz.dropna(subset=['tof_slope_ts'])
    var = pd.read_csv(os.path.join(RES, 'crosslab_variance.csv'))
    return ae, pz, var


def systems_strip(ax, groups, ylab, title):
    """逐组散点 + 组内范围细线 + 中位横线。groups = [(标签, 值数组, 颜色), ...]"""
    rng = np.random.default_rng(7)
    for i, (name, v, col) in enumerate(groups):
        if v.size == 0:
            continue
        jit = (rng.random(v.size) - 0.5) * 0.26
        ax.plot(np.full(v.size, float(i)) + jit, v, marker='o', ls='none',
                ms=2.4, color=col, alpha=0.92)
        ax.vlines(float(i), float(v.min()), float(v.max()),
                  color=col, lw=0.5, alpha=0.30)
        ax.hlines(float(np.median(v)), i - 0.24, i + 0.24,
                  color='0.12', lw=1.0, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(['%s\nLab %s' % (k, PS.LAB.get(k, '?')) for k, _v, _c in groups],
                       fontsize=6, linespacing=1.45)
    ax.set_ylabel(ylab, fontsize=6.6, labelpad=2)
    ax.set_title(title, fontsize=7, loc='left', pad=4)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.grid(alpha=0.22, axis='y', lw=0.5)
    PS.thin(ax)


def systems_variance_panel(ax, var):
    """(c) 分组能解释多少方差（eta^2）；每根条上画一条短的置换中位。

    两个分组维度不是同一件事，图上必须写清楚：
      campaign   = 试验活动，C1 / C2 / C3 / C4（共 4 组）
      university = 大学，Lab A = C1+C2，Lab B = C3+C4（共 2 组）
    横线 = 打乱标签 20000 次后 eta^2 的中位，即「分组毫无意义时能有多大」。
    """
    w = 0.34
    for gi, q in enumerate(QUANT):
        for scheme, dx, col in (('batch', -w / 2, C_CAMP),
                                ('university', +w / 2, C_UNI)):
            r = var[(var['quantity'] == q) & (var['scheme'] == scheme)]
            if r.empty:
                continue
            eta = float(r['eta2'].iloc[0])
            mu = float(r['null_median'].iloc[0])
            p_le = float(r['p_le'].iloc[0])
            xc = gi + dx
            ax.bar(xc, eta, w, color=col, edgecolor='none', zorder=2)
            ax.hlines(mu, xc - w / 2, xc + w / 2, color='0.15', lw=0.9, zorder=4)
            ax.text(xc, max(eta, mu) + 0.016, '%.3f' % eta, ha='center',
                    fontsize=5.6, color='0.25')
            if scheme == 'university':
                ax.text(xc, -0.055, 'p=%.3f' % p_le, ha='center', fontsize=5.4,
                        color=C_UNI)
    ax.set_xticks(range(len(QUANT)))
    ax.set_xticklabels(XTICK, fontsize=6.2)
    ax.set_xlim(-0.55, len(QUANT) - 0.45)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel(r'variance explained,  $\eta^2$', fontsize=6.6, labelpad=2)
    ax.set_title('(c)  variance share', fontsize=7, loc='left', pad=4)
    ax.grid(alpha=0.22, axis='y', lw=0.5)
    PS.thin(ax)
    # 图例：两种分组维度写全（含组数），并把横线的含义也并进图例
    ax.bar(np.nan, np.nan, color=C_CAMP, label='by campaign (4)')
    ax.bar(np.nan, np.nan, color=C_UNI, label='by university (2)')
    ax.plot([], [], color='0.15', lw=0.9, label='permutation median')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=6,
              handlelength=1.1, handletextpad=0.5, borderaxespad=0.25)


def systems_main():
    os.makedirs(FIG, exist_ok=True)
    ae, pz, var = systems_load()
    print('AE %d specimens   PZT %d specimens   variance rows %d'
          % (len(ae), len(pz), len(var)))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.175,
                        wspace=0.42)

    systems_strip(axes[0],
          [(b, ae.loc[ae['batch'] == b, 't85_pct'].to_numpy(float), PS.BATCH[b])
           for b in BATCHES],
          r'$t_{85}$   (% of life)', '(a)  acoustic emission')

    systems_strip(axes[1],
          [(b, pz.loc[pz['batch'] == b, 'tof_slope_ts'].to_numpy(float), PS.BATCH[b])
           for b in ['C1', 'C3', 'C4']],
          'slope   ($\\mu$s / 100% of life)', '(b)  guided waves')

    systems_variance_panel(axes[2], var)

    PS.save(fig, os.path.join(FIG, 'fig_two_systems.png'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all',
                    help='all 或逗号分隔：protocol,instrumentation,routes,systems')
    a = ap.parse_args()
    todo = list(ENTRY) if a.only == 'all' else [k.strip() for k in a.only.split(',') if k.strip()]
    for k in todo:
        if k not in ENTRY:
            raise SystemExit(f'未知图名 {k}，可选 {list(ENTRY)}')
        title, mod = ENTRY[k]
        print(f'[{k}] {title}')
        globals()[mod + '_main']()


if __name__ == '__main__':
    main()
