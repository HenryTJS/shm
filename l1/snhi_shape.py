# -*- coding: utf-8 -*-
"""SNHI 形状不变性诊断 —— 算法的**前提检验**，先于算法本身。

为什么必须先做这一步
--------------------
「把 t85 当形状模板、解寿命轴变换」这件事成立的前提是：
**不同试件的 AE 累积曲线，在归一化寿命轴上形状相似。**
这只是我上一轮的口头断言（依据是 t85 群体稳定在 89 至 94 个百分点）。
本脚本把它变成可证伪的量；不成立则后面整条路不用走。

三个量，缺一不可
----------------
1. 分位点散布：每条曲线累积到 q 时的寿命比例 t_q（q = .25/.5/.75/.85/.95）。
   跨组极差就是「形状一致到什么程度」的直接读数。
2. 逐点变异系数 CV(u)：归一化**前**（绝对累积量）vs 归一化**后**。
   论证的核心是对比 —— 只看归一化后的 CV 说明不了什么，
   必须证明**归一化把离散度压下去了**，否则「归一化」是安慰剂。
3. 可辨识性：曲线在早期（前 30 个百分点的寿命）到底长了多少「形状」。
   若 t50 已接近 50 个百分点，说明早期有信息；
   若所有分位点都挤在 90 个百分点之后（累积曲线近乎阶跃），
   则早期观测里没有形状可配准 —— 配准会病态，算法必败。

数据
----
C3（变幅VA，6 级）5 组 + C4（谱载）10 组 = 15 组，均为 Lab B `.DTA`。
只用 `_l1ae_frames_*.npz`（600 s 帧的 hit 计数）与 `_l1cyc_*.npz`（循环轴），
不重读任何原始大文件。口径与 `level_vs_shape.py` 的 `curve()` 保持一致。

只读 l1/results/*.npz；写 l1/results/snhi_shape.csv 与 l1/figures/fig_snhi_shape.*
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

NB = 100                      # 统一寿命网格：100 格，每格 1 个百分点的寿命
Q = [0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]

CAMPAIGN = {}
for _g in ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']:
    CAMPAIGN[_g] = 'C3'
for _g in ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35',
           'L1-36', 'L1-41', 'L1-44']:
    CAMPAIGN[_g] = 'C4'

COL = {'C3': '#1f77b4', 'C4': '#d62728'}
LBL = {'C3': 'C3  变幅VA（6 级，n=5）', 'C4': 'C4  谱载（n=10）'}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def curve(gid, nb=NB):
    """-> (u[%], hit[nb], n_f) 或 None。与 level_vs_shape.curve 同口径，仅改变箱数。"""
    fp = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    with np.load(fp, allow_pickle=True) as z:
        fr = z['frame'].astype(np.int64)
        n = z['n'].astype(float)
    with np.load(cp, allow_pickle=True) as z:
        cf = z['frame'].astype(np.int64)
        cyc = z['cycle'].astype(float)
        n_f = float(z['n_f'])
    m = dict(zip(cf.tolist(), cyc.tolist()))
    keep = np.array([int(x) in m for x in fr])
    fr, n = fr[keep], n[keep]
    c = np.array([m[int(x)] for x in fr], float)
    ok = np.isfinite(c) & (c >= 0)
    c, n = c[ok], n[ok]
    if c.size == 0:
        return None
    o = np.argsort(c)
    c, n = c[o], n[o]
    u = c / n_f                                    # 寿命比例（0 至 1）
    bi = np.clip((u * nb).astype(int), 0, nb - 1)
    hit = np.bincount(bi, weights=n, minlength=nb)[:nb]
    uu = (np.arange(nb) + 0.5) / nb * 100.0        # 用百分点作横轴，图上好读
    return uu, hit, n_f


def main():
    C = {}
    for g in CAMPAIGN:
        r = curve(g)
        if r is not None:
            C[g] = r
    print('可用组 %d（C3 %d / C4 %d），网格 %d 格\n'
          % (len(C), sum(1 for g in C if CAMPAIGN[g] == 'C3'),
             sum(1 for g in C if CAMPAIGN[g] == 'C4'), NB))

    # 每组三条曲线：
    #   hit   每格 hit 数（绝对量级，不可迁移）
    #   cum   累积（未归一化，绝对量级）
    #   sn    累积 / 总量（归一化形状，0 至 1）—— 这就是 SNHI 声称可比的东西
    rows = {}
    for g, (uu, hit, n_f) in C.items():
        cum = np.cumsum(hit)
        tot = cum[-1]
        rows[g] = dict(u=uu, hit=hit, cum=cum, sn=cum / tot if tot > 0 else np.zeros_like(cum),
                       tot=tot, n_f=n_f)

    # ---------------- 1) 分位点散布
    print('各曲线累积到 q 时所在的寿命百分位 t_q：')
    hdr = '  %-8s %-3s' % ('组', '批') + ''.join('%9s' % ('q=%.2f' % q) for q in Q) + '%11s' % 'n_f'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    tq = {}
    for g in sorted(C, key=lambda x: (CAMPAIGN[x], x)):
        uu, sn, tot = rows[g]['u'], rows[g]['sn'], rows[g]['tot']
        v = []
        for q in Q:
            v.append(float(uu[min(int(np.searchsorted(sn, q)), len(uu) - 1)]))
        tq[g] = v
        print('  %-8s %-3s' % (g, CAMPAIGN[g]) + ''.join('%9.1f' % x for x in v)
              + '%11.0f' % rows[g]['n_f'])
    print()
    # 短寿组（n_f 小于 20000）的曲线只有 2 至 3 个非零箱，形状无从谈起，
    # 会把极差撑大；单列一行以便看清「形状不变」到底覆盖了谁。
    SHORT = [g for g in C if rows[g]['n_f'] < 20000]
    GOOD = [g for g in sorted(C) if rows[g]['n_f'] >= 20000]
    if SHORT:
        print('  短寿组（n_f 小于 2e4，曲线仅 2 至 3 个非零箱）：%s'
              % ', '.join('%s(%.0f)' % (g, rows[g]['n_f']) for g in sorted(SHORT)))
        print()
    for tag, gs in (('全 %d 组' % len(C), sorted(C)),
                    ('%d 组（排除短寿）' % len(GOOD), GOOD)):
        A = np.array([tq[g] for g in gs], float)
        print('  [%s]' % tag)
        for j, q in enumerate(Q):
            col = A[:, j]
            print('    q=%.2f  t_q 中位 %5.1f   极差 %5.1f   IQR %5.1f'
                  % (q, np.median(col), col.max() - col.min(),
                     np.percentile(col, 75) - np.percentile(col, 25)))
    A = np.array([tq[g] for g in sorted(C)], float)

    # ---------------- 2) 归一化前 vs 后的逐点离散度
    print('\n逐点变异系数（对 15 条曲线取 sd / mean）：')
    print('  %-10s %9s %9s' % ('寿命段', '绝对累积', '归一化后'))
    print('  ' + '-' * 30)
    for tag, gs in (('全 %d 组' % len(C), sorted(C)),
                    ('%d 组（排除短寿）' % len(GOOD), GOOD)):
        cum = np.stack([rows[g]['cum'] for g in gs])
        sn = np.stack([rows[g]['sn'] for g in gs])
        print('  [%s]' % tag)
        for lo, hi in [(0, 20), (20, 50), (50, 80), (80, 100)]:
            idx = slice(int(lo * NB / 100), int(hi * NB / 100))
            cv_abs = float(np.median(cum[:, idx].std(0)
                                     / np.maximum(cum[:, idx].mean(0), 1e-9)))
            cv_sn = float(np.median(sn[:, idx].std(0)
                                    / np.maximum(sn[:, idx].mean(0), 1e-9)))
            print('    %2d 至 %3d%% %9.3f %9.3f' % (lo, hi, cv_abs, cv_sn))

    # ---------------- 3) 可辨识性：早期有多少形状
    print('\n可辨识性：t_q 越接近 q 越好（曲线越接近线性，早期信息越多）')
    for j, q in enumerate(Q):
        m = float(np.median(A[:, j]))
        print('  q=%.2f  t_q 中位 %5.1f  →  偏离线性 %+5.1f 个百分点'
              % (q, m, m - q * 100))

    # ---------------- 图
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    for g in sorted(C):
        a.plot(rows[g]['u'], rows[g]['sn'], lw=1.1, alpha=0.85,
               color=COL[CAMPAIGN[g]])
    a.plot([0, 100], [0, 1], 'k--', lw=1.0, alpha=0.5, label='线性参考')
    a.set_xlim(0, 100)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel('寿命 / %')
    a.set_ylabel('累积 hit / 自身总量')
    a.set_title('(a) 归一化后：15 条曲线\n形状是否叠在一起')
    a.grid(alpha=0.25)
    a.legend(loc='upper left', fontsize=9)
    a.text(0.98, 0.06, 'C3 蓝  C4 红', transform=a.transAxes, ha='right',
           fontsize=9, color='#555555')

    b = ax[1]
    for g in sorted(C):
        b.semilogy(rows[g]['u'], np.maximum(rows[g]['cum'], 1), lw=1.1, alpha=0.85,
                   color=COL[CAMPAIGN[g]])
    b.set_xlim(0, 100)
    b.set_xlabel('寿命 / %')
    b.set_ylabel('累积 hit（绝对，对数轴）')
    b.set_title('(b) 归一化前：同一批数据\n量级差多少')
    b.grid(alpha=0.25, which='both')

    c = ax[2]
    ys = np.arange(len(C))
    for i, g in enumerate(sorted(C)):
        c.plot(tq[g], [i] * len(Q), 'o', ms=4, color=COL[CAMPAIGN[g]])
        c.hlines(i, tq[g][0], tq[g][-1], color=COL[CAMPAIGN[g]], lw=1.0, alpha=0.4)
    c.set_yticks([])
    c.set_xlim(0, 100)
    c.set_xlabel('寿命 / %')
    c.set_ylabel('每个试件一行')
    c.set_title('(c) 分位点位置（q = .10 .25 .50 .75 .85 .90 .95）\n点位越齐，形状越不变')
    c.grid(alpha=0.25, axis='x')
    for q in Q:
        c.axvline(q * 100, color='#999999', ls=':', lw=0.8)
    for j, q in enumerate(Q):
        c.text(q * 100, len(C) - 0.3, '%.2f' % q, fontsize=7.5, color='#666666',
               ha='center')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, 'fig_snhi_shape.' + ext), dpi=170,
                    bbox_inches='tight')
    plt.close(fig)
    print('\n图已存 %s' % os.path.join(FIG, 'fig_snhi_shape.png'))

    fp = os.path.join(RES, 'snhi_shape.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['gid', 'campaign', 'n_f', 'total_hits', 't50', 't85', 't95'])
        for g in sorted(C, key=lambda x: (CAMPAIGN[x], x)):
            uu, sn = rows[g]['u'], rows[g]['sn']
            f = lambda q: float(uu[min(int(np.searchsorted(sn, q)), len(uu) - 1)])
            w.writerow([g, CAMPAIGN[g], '%.0f' % rows[g]['n_f'],
                        '%.0f' % rows[g]['tot'], '%.2f' % f(0.50),
                        '%.2f' % f(0.85), '%.2f' % f(0.95)])
    print('已写 %s' % fp)


if __name__ == '__main__':
    main()
