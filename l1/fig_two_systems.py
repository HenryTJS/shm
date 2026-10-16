#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 2 —— the same conclusion from two physically independent systems.

(a) acoustic emission: t85 of the cumulative hit curve, per campaign
(b) guided waves: Theil-Sen slope of the time-of-flight drift, per campaign
(c) share of the total variance carried by the grouping (eta^2), with its
    permutation median drawn as a small horizontal tick on each bar

为什么 (c) 不再是「批间 / 组内离差比」
------------------------------------
旧版用 4.85/14.75 = 0.33 与 0.0122/0.0425 = 0.29 作为主证据。置换检验显示该比值
**天然小于 1**（打乱批次标签 20000 次，零分布中位 0.44，99.5% 的抽样 < 1），
不具判别力。改用 eta^2 = SS_between / (SS_between + SS_within)，它在标签打乱下的
零分布是良定义的，且两套系统都给出约 0 的机构方差占比。

读：l1/results/{l1_hi_crosslab, pzt_crosslab_summary, crosslab_variance}.csv
写：l1/figures/fig_two_systems.png / .pdf
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as PS                                  # noqa: E402

import matplotlib.pyplot as plt                           # noqa: E402

RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

PS.apply()

BATCHES = ['C1', 'C2', 'C3', 'C4']
C_CAMP, C_UNI = '#b4b4b4', '#2f4b7c'          # 批次（灰）对机构（蓝）
QUANT = ['AE cum_hits t50_pct', 'AE cum_hits t85_pct', 'PZT tof_slope_ts']
XTICK = ['AE $t_{50}$', 'AE $t_{85}$', 'PZT slope']


def load():
    ae = pd.read_csv(os.path.join(RES, 'l1_hi_crosslab.csv'))
    ae = ae[(ae['variant'] == 'cum_hits')].dropna(subset=['t85_pct'])
    pz = pd.read_csv(os.path.join(RES, 'pzt_crosslab_summary.csv'))
    pz = pz.dropna(subset=['tof_slope_ts'])
    var = pd.read_csv(os.path.join(RES, 'crosslab_variance.csv'))
    return ae, pz, var


def strip(ax, groups, ylab, title):
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


def variance_panel(ax, var):
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


def main():
    os.makedirs(FIG, exist_ok=True)
    ae, pz, var = load()
    print('AE %d specimens   PZT %d specimens   variance rows %d'
          % (len(ae), len(pz), len(var)))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.175,
                        wspace=0.42)

    strip(axes[0],
          [(b, ae.loc[ae['batch'] == b, 't85_pct'].to_numpy(float), PS.BATCH[b])
           for b in BATCHES],
          r'$t_{85}$   (% of life)', '(a)  acoustic emission')

    strip(axes[1],
          [(b, pz.loc[pz['batch'] == b, 'tof_slope_ts'].to_numpy(float), PS.BATCH[b])
           for b in ['C1', 'C3', 'C4']],
          'slope   ($\\mu$s / 100% of life)', '(b)  guided waves')

    variance_panel(axes[2], var)

    PS.save(fig, os.path.join(FIG, 'fig_two_systems.png'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
