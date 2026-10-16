#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文 Fig. 1 —— 跨机构对齐协议（英文 · 期刊版式）。

为什么重做
----------
上一版是「五个圆角矩形 + 彩色边框 + 灰底 + 中文」的幻灯片式堆叠：把公式、数字、
结论全塞进文字框，构图对称，看不出协议本身在做什么。这一版反过来 ——
**让坐标轴和曲线承担表达，文字只做标注**，三段横向：

    (A) raw records   三个批次的记录各落在自己的墙钟上（间隔不同、无共同原点）
    (B) the two axes  横轴换寿命分数、纵轴自归一化；归一化后三条曲线重合
    (C) comparable    四个试验活动的 t85 落在同一轴上（**真实数据**）

风格约束：白底、细线、衬线字（Times）、无圆角、无阴影、无渐变、低饱和配色。

数据：[C] 段读 l1/results/l1_hi_crosslab.csv（variant=cum_hits, 29 试件）；
[A][B] 是示意（[B] 的曲线用两个参数不同的 Weibull CDF 归一化得到，仅示意形状）。

输出：l1/figures/fig_protocol.png / .pdf
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')

# ---- 版式与配色统一走 paper_style（三张论文图共用）-----------------------
sys.path.insert(0, HERE)
import paper_style as PS                                  # noqa: E402

PS.apply()
BCOL = PS.BATCH


def panel_a(ax):
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


def panel_b(ax):
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


def panel_c(ax):
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


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.35))

    # 三段的顶边与底边对齐（0.805 / 0.225），使三个标题在同一水线上
    ax_a = fig.add_axes([0.045, 0.225, 0.170, 0.580])
    ax_b = fig.add_axes([0.328, 0.225, 0.320, 0.580])
    ax_c = fig.add_axes([0.762, 0.225, 0.198, 0.580])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    PS.save(fig, os.path.join(FIG, 'fig_protocol.png'))


if __name__ == '__main__':
    main()
