#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四模态并行对照（C1 五组）：AE / FBG / DFOS / PZT 在同一条寿命轴上的损伤演化。

背景
----
C1（L1-03/04/05/09/23）是 L1 公开集里**唯一四类传感齐全**的 5 个试件
（AE + FBG + LUNA/DFOS + PZT）；其余批次至少缺一类。

要回答的问题
------------
四种技术看的是同一次损伤演化吗？谁先动、谁最晚？模态之间的一致性能到多少？

口径（刻意同构）
----------------
把四个模态化成**同一个量**：累积自归一化曲线（cumsum / 总量），取值落在 [0,1]：

    AE   : cumsum(每块 AE 能量和  ae_energy)
    FBG  : cumsum(每块应变偏离  dev)
    DFOS : cumsum(每块局部偏离  local)
    PZT  : cumsum(相邻状态退相关 D)

横轴统一为寿命分数（cycle / n_f），再插值到共同网格。
指标：t25 / t50 / t85（达终值 x% 的寿命分数）、两两 Spearman ρ 与 Kendall τ、t50 领先滞后。

注意：累积量**构造上单调**，所以 ρ 会普遍偏高，它只用来回答「形状是否一致」；
真正有信息量的是 t50/t85 的**模态间差**（谁先谁后）。

输入（全部为已落盘产物，不重算原始数据）
    l1/results/_l1_hi_L1-*.npz        cyc / dev / ae_energy / nf
    l1/results/_l1_dfos_hi_L1-*.npz   cyc / local / nf
    l1/results/pzt_crosslab.csv       gid / life_pct / D

输出
    l1/results/four_modal.csv         每组 × 每模态的 n / t25 / t50 / t85
    l1/results/four_modal_pairs.csv   两两的 ρ / τ / t50 领先滞后
    l1/figures/fig_four_modal.png     四模态叠图（2 × 3 面板）

用法
    python l1/four_modal.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-23']
MODALITIES = ['AE', 'FBG', 'DFOS', 'PZT']
COLOR = {'AE': '#1f77b4', 'FBG': '#d62728', 'DFOS': '#2ca02c', 'PZT': '#ff7f0e'}
GRID = np.linspace(0.02, 1.00, 120)


def _cum_norm(life, val):
    """累积自归一化：y = cumsum(v) / sum(v)。

    返回 (life, y_cum, v_raw)。v_raw 是按同一排序整理过的**原生**每块量，
    用于算相关 —— 累积量构造上单调，拿它算相关恒等于 1，没有信息量。
    """
    lf = np.asarray(life, dtype=float)
    vv = np.asarray(val, dtype=float)
    ok = np.isfinite(lf) & np.isfinite(vv)
    lf, vv = lf[ok], vv[ok]
    if len(lf) < 3:
        return None, None, None
    o = np.argsort(lf)
    lf, vv = lf[o], np.clip(vv[o], 0, None)
    if vv.sum() <= 0:
        return None, None, None
    c = np.cumsum(vv)
    return lf, c / c[-1], vv


def _t_at(life, y, level):
    """累积曲线达 level 时的寿命分数（y 单调不减）。"""
    return float(np.interp(level, y, life))


def load_modalities(gid):
    """返回 {modality: (life_frac, cum_curve, raw_block)}。"""
    out = {}

    z = os.path.join(RES, '_l1_hi_%s.npz' % gid)
    if os.path.isfile(z):
        d = np.load(z, allow_pickle=True)
        nf = float(d['nf'])
        cyc = np.asarray(d['cyc'], dtype=float)
        for key, mod in (('ae_energy', 'AE'), ('dev', 'FBG')):
            lf, y, raw = _cum_norm(cyc / nf, d[key])
            if lf is not None:
                out[mod] = (lf, y, raw)

    z = os.path.join(RES, '_l1_dfos_hi_%s.npz' % gid)
    if os.path.isfile(z):
        d = np.load(z, allow_pickle=True)
        nf = float(d['nf'])
        cyc = np.asarray(d['cyc'], dtype=float)
        lf, y, raw = _cum_norm(cyc / nf, d['local'])
        if lf is not None:
            out['DFOS'] = (lf, y, raw)

    p = os.path.join(RES, 'pzt_crosslab.csv')
    if os.path.isfile(p):
        df = pd.read_csv(p)
        s = df[(df['gid'] == gid) & df['D'].notna()]
        if len(s) >= 3:
            lf, y, raw = _cum_norm(s['life_pct'].to_numpy(dtype=float) / 100.0,
                                   s['D'].to_numpy(dtype=float))
            if lf is not None:
                out['PZT'] = (lf, y, raw)

    return out


def analyse():
    rows, pairs = [], []
    curves = {}

    for gid in GROUPS:
        cur = load_modalities(gid)
        curves[gid] = cur
        if not cur:
            print('  [%s] 无任何模态产物，跳过' % gid)
            continue

        for mod in MODALITIES:
            if mod not in cur:
                continue
            lf, y, _raw = cur[mod]
            rows.append({'gid': gid, 'modality': mod, 'n': len(lf),
                         't25': _t_at(lf, y, .25),
                         't50': _t_at(lf, y, .50),
                         't85': _t_at(lf, y, .85)})

        # 两两相关：用**原生每块量**在共同网格上算（累积量算相关恒为 1）
        grid_raw = {}
        for mod, (lf, _y, raw) in cur.items():
            grid_raw[mod] = np.interp(GRID, lf, raw)
        mods = [m for m in MODALITIES if m in grid_raw]
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = mods[i], mods[j]
                rho = float(spearmanr(grid_raw[a], grid_raw[b])[0])
                tau = float(kendalltau(grid_raw[a], grid_raw[b])[0])
                ta = _t_at(cur[a][0], cur[a][1], .50)
                tb = _t_at(cur[b][0], cur[b][1], .50)
                pairs.append({'gid': gid, 'mod_a': a, 'mod_b': b,
                              'rho': rho, 'tau': tau, 't50_a': ta, 't50_b': tb,
                              'lead_a_pp': (ta - tb) * 100.0})

    return pd.DataFrame(rows), pd.DataFrame(pairs), curves


def report(df1, df2):
    print('\n=== 各模态的 t50 / t85（寿命分数 %）===')
    if df1.empty:
        print('  （空）')
        return
    show = df1.copy()
    for c in ('t25', 't50', 't85'):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))

    if df1.empty or df2.empty:
        return
    piv = df1.pivot(index='gid', columns='modality', values='t50')
    piv = (piv * 100).round(1)
    print('\n=== t50 透视（行=试件，列=模态，单位 %）===')
    print(piv.to_string())
    print('\n每组的「最晚 - 最早」t50 差（pp）：')
    for gid, r in piv.iterrows():
        v = r.dropna()
        if len(v) >= 2:
            print('   %-7s 最早 %-5s %5.1f   最晚 %-5s %5.1f   差 %5.1f'
                  % (gid, v.idxmin(), v.min(), v.idxmax(), v.max(), v.max() - v.min()))

    print('\n=== 两两相关（共同网格上的 Spearman ρ）===')
    print(df2.groupby(['mod_a', 'mod_b'])['rho'].agg(['count', 'median', 'min']).round(3).to_string())


def plot(_curves=None):
    """Fig. 3 —— 四模态在同一批试件上的一致性（区间条形式）。

    旧版是 2x3 子图的曲线叠加（中文、16.5x9 英寸）：信息量大但难比对，
    也看不出「区间宽度」这个关键差异。改成 5 行 x 4 模态的区间条 ——
    横轴统一为寿命分数，每根条画出 t25 到 t85 的范围，t50 用实心点标出。
    版本三张论文图共用 paper_style 的版式与配色。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import paper_style as PS
    PS.apply()

    df = pd.read_csv(os.path.join(RES, 'four_modal.csv'))
    gids = [g for g in GROUPS if g in set(df['gid'])]
    fig, ax = plt.subplots(figsize=(7.0, 2.25))

    for i, gid in enumerate(gids):
        y0 = float(len(gids) - 1 - i)
        for j, m in enumerate(MODALITIES):
            r = df[(df['gid'] == gid) & (df['modality'] == m)]
            if r.empty:
                continue
            lo = float(r['t25'].iloc[0]) * 100.0
            mid = float(r['t50'].iloc[0]) * 100.0
            hi = float(r['t85'].iloc[0]) * 100.0
            y = y0 + (1.5 - j) * 0.17
            ax.plot([lo, hi], [y, y], color=PS.MODALITY[m], lw=1.7,
                    solid_capstyle='butt', zorder=3)
            ax.plot(mid, y, marker='o', ms=2.6, color=PS.MODALITY[m], zorder=4)
        ax.text(-2.0, y0, gid, fontsize=6.3, ha='right', va='center', color='0.2')

    for m in MODALITIES:
        ax.plot([], [], color=PS.MODALITY[m], lw=1.7, label=m)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, len(gids) - 0.38)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('lifetime fraction   (%)', fontsize=6.6, labelpad=2)
    ax.grid(alpha=0.22, axis='x', lw=0.5)
    PS.thin(ax, keep=('bottom',))
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.00), ncol=4,
              fontsize=6.2, handlelength=1.4, columnspacing=1.3)
    out = os.path.join(FIG, 'fig_four_modal.png')
    PS.save(fig, out)
    return out


def main():
    print('四模态并行对照 —— C1 五组：%s' % ' '.join(GROUPS))
    df1, df2, curves = analyse()
    if not df1.empty:
        df1.to_csv(os.path.join(RES, 'four_modal.csv'), index=False, encoding='utf-8-sig')
    if not df2.empty:
        df2.to_csv(os.path.join(RES, 'four_modal_pairs.csv'), index=False, encoding='utf-8-sig')
    report(df1, df2)
    p = plot(curves)
    print('\n已写：%s' % os.path.join(RES, 'four_modal.csv'))
    print('已写：%s' % os.path.join(RES, 'four_modal_pairs.csv'))
    print('已写：%s' % p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
