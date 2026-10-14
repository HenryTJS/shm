# -*- coding: utf-8 -*-
"""生成论文的图 1 至 3（规格见 `docs/论文大纲_ICSSD2027.md` 4.1 至 4.4）。

图 1  三批自归一化 HI 曲线叠图（跨批不变性）
图 2  `m_eff` 散点 + L1-06 / L1-24 的分级产率对照
图 3  `shape_rob25` 逐组排名（标出 L1-14 与 L1-30，二者性质不同）

口径与数据源
------------
· 图 1 的三批 HI 来自**三个不同的物理量**，统一按「自身终值归一化」后比 t85：
    C1  `_l1_hi_L1-*.npz` 的 `hi`（应变偏离）
    C2  `_l1_dfos_hi_L1-*.npz` 的 `hi`（DFOS 剖面 RMSE）
    C3+C4  从 `_l1ae_frames60s_*.npz` 的每帧 hit 数 `n` 累积（AE 累积 hit 数）
  横轴：C1/C2 用循环分数 `cyc / n_f`；C3+C4 无循环轴 ⇒ 用帧时间分数。
· 图 2 数据来自 `results/l1_transfer_rate.csv`（11 组）。
· 图 3 数据来自 `results/_l1fbgprof_*.npz` 的 `shape_rob25`（13 组）。

用法
----
    python l1/make_figures.py            # 写 l1/figures/fig1..3_*.png
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

import ae_cycle as AC                                  # noqa: E402

# 静默期移除统计：[（组号, 移除帧数, 总帧数）]
REMOVED = []

# 🔴 AE 静默判据的门槛，**故意放宽于** `ae_cycle.AE_PAUSE_FRAC` 的默认 0.05。
# 理由：0.05 只能删掉 0.6% 的帧，图 1 里那几条长平台（事件率低但非零）清不掉。
# 先做了门槛扫描（仅 Lab B 的 14 组）：
#
#     frac  移除帧%   t85 中位%   逐组范围
#     0.05    0.6%      93.2      84 至 99
#     0.10    1.2%      93.2      84 至 99
#     0.15    2.8%      93.2      84 至 99
#     0.20    4.4%      93.2      84 至 99
#     0.30    7.1%      93.2      84 至 99
#     0.50   13.4%      93.2      84 至 99
#
# ⇒ **t85 对门槛完全不敏感**（删掉多达 13.4% 的帧，中位与范围一字不变）。
#    这本身就是一个正面结论：被删的帧确实不承载损伤信息。
#    因此取 0.30 作折中（比现口径多删一个量级，但只到最大删除量的一半）。
PAUSE_FRAC = 0.30

# 🔴 C1/C2 与 C3/C4 是**两个独立实验室**的数据，用的是**两套不同的 AE 采集系统**：
#     C1/C2 —— Vallen `.pridb`（见 `l1/ae_io.py`）
#     C3/C4 —— PAC/Mistras `.DTA`（见自研 `l1/ae_dta.py`）
#   两者的事件筛选阀值、分箱粒度也不同（C1/C2 用 500-cycle 箱，C3/C4 用 60 s 帧）。
#   图 1 把它们叠在一起，论点是「**跨机构 + 跨采集系统仍然一致**」；
#   但它们**不是逐点可比的**，图注里必须写清。
C1 = ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-23']
C2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60']
C34 = ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24', 'L1-25', 'L1-27', 'L1-29',
       'L1-30', 'L1-31', 'L1-35', 'L1-36', 'L1-41', 'L1-44']

COL = {'C1': '#1f77b4', 'C2': '#4a9fd8', 'C34': '#2ca02c'}


def _curve(hi, x):
    hi = np.asarray(hi, float)
    ok = np.isfinite(hi)
    hi, x = hi[ok], np.asarray(x, float)[ok]
    if hi.size < 3 or not np.isfinite(hi[-1]) or hi[-1] <= 0:
        return None, None
    return x, hi / hi[-1]          # 自归一化：末值 = 1


def batch_curves():
    """→ {批名: [(x, HI), ...]}。三批统一用「**累积 AE hit 数自归一化**」口径。

    ⚠️ 不要用 `_l1_hi_*.npz` / `_l1_dfos_hi_*.npz` 里的 `hi` 字段：
    实测 C2 那份**全是 1.0**、C1 那份**非单调**（单调比例 0.66），
    拿来做 t85 会得到 70.5% / 0.2% 这种明显错的数。
    正确来源是 `evaluate_l1_hi_ae.hi_of_group()` 的 `cum_hits`（C1/C2），
    以及 `_l1ae_frames60s_*.npz` 的每帧 hit 数 `n` 累积（C3+C4）——
    两者都是 `ae_hi.py` 的 `HI_hit` 口径。
    """
    import evaluate_l1_hi_ae as E

    out = {'C1': [], 'C2': [], 'C34': [], 'L1-23': []}
    for g in C1 + C2:
        try:
            d = E.hi_of_group(g)
        except Exception:                                    # noqa: BLE001
            d = None
        if not d:
            continue
        cum = np.asarray(d['cum_hits'], float)
        if cum.size < 3 or cum[-1] <= 0:
            continue
        x = np.asarray(d['cyc'], float) / float(d['n_f'])
        # ⚠️ L1-23 虽然同属 C1 批，但它是**提载**组（-5/-50 至 -6/-60），
        # 与恒幅四组不同载荷程序 ⇒ 单独画，不混进 C1 的 t85 中位。
        key = 'L1-23' if g == 'L1-23' else ('C1' if g in C1 else 'C2')
        out[key].append((x, cum / cum[-1]))
    for g in C34:
        fp = os.path.join(RES, '_l1ae_frames60s_%s.npz' % g)
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        n = np.asarray(z['n'], float)
        if n.size < 3 or n.sum() <= 0:
            continue
        # 🔴 移除 **AE 静默期**（试验停机）。理由不只是美观：
        #   C1/C2 的横轴是**循环轴**，停机期间没有循环 ⇒ 本来就不占寿命；
        #   C3+C4 只能用时间轴，若不剔除停机时间，等于把停机算进了寿命，
        #   曲线就会出现长平台 —— 两批口径本来就不公平。
        # 判据直接复用 `ae_cycle` 的现成守卫：`_ae_silent()` + 帧长等比缩放的 `_pause_win()`。
        sil = AC._ae_silent(n, win=AC._pause_win(float(z['frame_s'])),
                             frac=PAUSE_FRAC)
        keep = ~sil
        if keep.sum() < 3:
            continue
        cum = np.cumsum(n[keep])
        if cum[-1] <= 0:
            continue
        x = np.arange(keep.sum()) / float(keep.sum() - 1)   # 压缩掉静默期后的归一化时长
        out['C34'].append((x, cum / cum[-1]))
        REMOVED.append((g, int(sil.sum()), int(n.size)))
    return out


def t85(x, y):
    """HI 首次达到 0.85 的横轴值（百分比）。缺则 None。"""
    idx = np.where(y >= 0.85)[0]
    return float(x[idx[0]] * 100.0) if idx.size else None


def fig1():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cur = batch_curves()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    label = {'C1': 'Lab A · Vallen .pridb · constant-amp  (n=%d)' % len(cur['C1']),
             'C2': 'Lab A · Vallen .pridb · constant-amp  (n=%d)' % len(cur['C2']),
             'C34': 'Lab B · PAC/Mistras .DTA · variable / spectrum  (n=%d)'
                    % len(cur['C34'])}
    med_t85 = {}
    for k in ('C1', 'C2', 'C34'):
        for x, y in cur[k]:
            ax.plot(x, y, color=COL[k], lw=0.8, alpha=0.45)
        ts = [t85(x, y) for x, y in cur[k]]
        ts = [v for v in ts if v is not None]
        med_t85[k] = float(np.median(ts)) if ts else float('nan')
        ax.plot([], [], color=COL[k], lw=1.6,
                label='%s   median $t_{85}$ = %.1f%%' % (label[k], med_t85[k]))
    for x, y in cur.get('L1-23', []):
        ax.plot(x, y, color='0.45', lw=1.0, ls='--', alpha=0.9)
    ax.plot([], [], color='0.45', lw=1.0, ls='--',
            label='L1-23  load-stepped (same C1 batch)')
    ax.axhline(0.85, color='k', ls=':', lw=1.0)
    ax.text(0.015, 0.872, '$t_{85}$ = 0.85', fontsize=7, color='k')
    ax.set_xlabel('normalised life  (cycle fraction for Lab A; silent periods REMOVED '
                  'for Lab B)', fontsize=8)
    ax.set_ylabel('cumulative AE hits, self-normalised', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.8, loc='lower right', frameon=True, framealpha=0.92,
              edgecolor='0.8')
    ax.set_title('Figure 1.  Self-normalised cumulative-AE-hit HI: two independent '
                 'laboratories, two acquisition systems', fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig1_hi_overlay.png')
    fig.savefig(p)
    plt.close(fig)
    print('写出', p)
    if REMOVED:
        tot_r = sum(a for _, a, _ in REMOVED)
        tot_n = sum(b for _, _, b in REMOVED)
        print('   已移除静默期：%d / %d 帧（%.1f%%），涉 %d 组'
              % (tot_r, tot_n, 100.0 * tot_r / max(tot_n, 1), len(REMOVED)))
        for g, a, b in REMOVED:
            if a:
                print('      %-7s %5d / %5d  (%.1f%%)' % (g, a, b, 100.0 * a / b))
    return med_t85


def fig2():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(RES, 'l1_transfer_rate.csv'), encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    ratio = np.array([float(r['幅值对比']) for r in rows])
    meff = np.array([float(r['m_eff']) for r in rows])
    rate = np.array([float(r['产率比']) for r in rows])
    gids = [r['组号'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300,
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    ax1.axhspan(3.0, 5.0, color='0.87', zorder=0)
    ax1.text(1.665, 4.05, 'physical\nexpectation\n$m_{eff}$ = 3 to 5',
             fontsize=6.5, color='0.3', va='center', ha='right')
    ax1.scatter(ratio, meff, s=28, c='#d62728', zorder=3)
    for x, y, g in zip(ratio, meff, gids):
        ax1.annotate(g.replace('L1-', ''), (x, y), fontsize=6.5, ha='center',
                     xytext=(0, 8), textcoords='offset points')
    ax1.set_xlabel('load-amplitude ratio (max / min)', fontsize=8)
    ax1.set_ylabel('$m_{eff}$  (fitted rate exponent)', fontsize=8)
    ax1.set_title('(a)  $m_{eff}$ spans -0.20 to +9.05  (n=11)', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.set_xlim(1.06, 1.80)
    ax1.set_ylim(-1.8, 10.4)

    # L1-06 与 L1-24 同为 1.50 倍幅值对比的分级产率
    idx = {g: i for i, g in enumerate(gids)}
    xpos = np.arange(2)
    w = 0.36
    pick = ['L1-06', 'L1-24']
    lo = [float(rows[idx[g]]['产率_低幅']) for g in pick]
    hi = [float(rows[idx[g]]['产率_高幅']) for g in pick]
    rr = [rate[idx[g]] for g in pick]
    ax2.bar(xpos - w / 2, lo, w, label='low level', color='#8c8c8c')
    ax2.bar(xpos + w / 2, hi, w, label='high level', color='#1f77b4')
    for i, (l, h, q) in enumerate(zip(lo, hi, rr)):
        ax2.text(i - w / 2, l + 1.5, '%.1f' % l, ha='center', fontsize=6.5)
        ax2.text(i + w / 2, h + 1.5, '%.1f' % h, ha='center', fontsize=6.5)
        ax2.text(i, max(h, l) + 11, 'x%.2f' % q, ha='center', fontsize=8,
                 color='#d62728')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(['L1-06\nratio 1.50', 'L1-24\nratio 1.50'], fontsize=7)
    ax2.set_ylabel('AE yield rate', fontsize=8)
    ax2.set_ylim(0, max(hi) * 1.52)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6.5, frameon=False, loc='upper left',
               bbox_to_anchor=(0.0, 0.80))
    ax2.set_title('(b)  same 1.50x, 14x apart in amplification', fontsize=8)
    fig.suptitle('Figure 2.  The amplitude route breaks down under variable amplitude',
                 fontsize=9, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig2_meff.png')
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('写出', p)


def fig3():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    rows = []
    for f in sorted(glob.glob(os.path.join(RES, '_l1fbgprof_*.npz'))):
        g = os.path.basename(f).replace('_l1fbgprof_', '').replace('.npz', '')
        z = np.load(f, allow_pickle=True)
        y = np.asarray(z['shape_rob25'], float)
        if y.size < 30:
            continue
        r = float(spearmanr(y, np.arange(y.size)).statistic)
        rows.append((g, r, int(y.size)))
    rows.sort(key=lambda t: t[1])

    weak = {'L1-14': 'genuinely weak:  FBG drift tiny, AE side normal',
            'L1-30': 'NOT assessable:  98.5% of records before impact'}
    WEAK_R = 0.30                      # 与 fbg_tools.classify() 的 |rho| < 0.3 一致
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    names = [g for g, _, _ in rows]
    vals = [r for _, r, _ in rows]
    cols = ['#d62728' if g in weak else '#2ca02c' for g in names]
    ax.barh(np.arange(len(rows)), vals, color=cols, height=0.72)
    ax.axvline(0, color='k', lw=0.8)
    for i, (g, r, n) in enumerate(rows):
        ax.text(r + 0.012, i, '%+.3f   (n=%d)' % (r, n), va='center', fontsize=6.5)
    ax.plot([], [], 's', color='#2ca02c',
            label='tracks damage  ($\\rho \\geq$ %.2f)' % WEAK_R)
    ax.plot([], [], 's', color='#d62728',
            label='weak  L1-14: %s' % weak['L1-14'])
    ax.plot([], [], 's', color='#d62728',
            label='weak  L1-30: %s' % weak['L1-30'])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(r'Spearman $\rho$  ($shape\_rob25$ vs window order)', fontsize=8)
    ax.set_xlim(-0.06, 1.18)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.tick_params(labelsize=7)
    npos = sum(1 for v in vals if v >= WEAK_R)
    nneg = sum(1 for v in vals if v <= -WEAK_R)
    nweak = len(vals) - npos - nneg
    ax.legend(fontsize=6.2, frameon=False, loc='lower right',
              handlelength=1.1, labelspacing=0.45)
    ax.set_title('Figure 3.  $shape\\_rob25$ per specimen:  %d positive / %d negative / '
                 '%d weak' % (npos, nneg, nweak), fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig3_shape_rob25_rank.png')
    fig.savefig(p)
    plt.close(fig)
    print('写出', p)
    print('   排名:', ', '.join('%s %+.3f' % (g, r) for g, r, _ in rows[::-1]))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    m = fig1()
    print('   t85 中位：C1 %.1f%%   C2 %.1f%%   C3+C4 %.1f%%'
          % (m.get('C1', float('nan')), m.get('C2', float('nan')),
             m.get('C34', float('nan'))))
    fig2()
    fig3()


if __name__ == '__main__':
    main()
