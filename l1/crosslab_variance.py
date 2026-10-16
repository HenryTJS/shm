#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨机构可比性的方差占比检验 —— 为什么不用「批间/组内离差比」。

背景（2026-10-15）
------------------
论文原先用「批间离差 / 组内离差」的比值做主要证据（AE 0.33、导波 0.29）。
置换检验显示这个统计量**不具判别力**：

- 把批次标签随机打乱、保持每批样本量，该比值的零分布**中位是 0.44**，
  且 **99.5% 的抽样都小于 1** —— 「比值小于 1」是估计量的构造性质，
  不是实验发现（批中位数的极差天然小于组内极差的中位）；
- 观测值 0.33 落在零分布中部，p = 0.18（批间极差口径）/ 0.21（比值口径）。

改用**方差占比 η²**：`η² = SS_between / (SS_between + SS_within)`，
再用同样的打乱方案做置换检验。这个量有正确的零分布。

已算出的结果（论文 §3.3 / §2.2 引用）
--------------------------------------
| 指标              | 分组     | η²     | 零分布中位 | p(观测更小) |
| ----------------- | -------- | ------ | ---------- | ----------- |
| AE t50            | 2 所大学 | 0.0000 | 0.0166     | 0.027       |
| PZT 斜率          | 2 个实验室 | 0.0000 | 0.0300     | 0.004       |
| AE t85            | 2 所大学 | 0.0570 | 0.0184     | 0.783       |
| AE t85            | 4 个活动 | 0.0711 | 0.0897     | 0.393       |
| 对照：AE 累积能量 t85 | 4 个活动 | 0.1170 | 0.0928     | 0.639       |

**必须避开的坑**：`pzt_causal_summary.csv` 的 `onset_pct` 含 NaN。
`np.bincount(weights=v)` 把 NaN 当 0 累加，而 `(v**2).sum()` 变 NaN，
于是 `eta = b/(b+w)` 的判据 `(b+w) > 0` 在 NaN 上为假、**静默返回 0** ——
第一版结果里 14 个配置的 η² 全为 0.0000 就是这么来的。本脚本对每个分组强制 `dropna`。

用法
    python l1/crosslab_variance.py
    python l1/crosslab_variance.py --perm 50000
输出
    l1/results/crosslab_variance.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
SEED = 20261015
LAB_OF = {'C1': 'A', 'C2': 'A', 'C3': 'B', 'C4': 'B'}


def decomp(v, g):
    """方差分解 -> (SS_between, SS_within)。用 bincount 以便在置换循环里足够快。"""
    if not np.all(np.isfinite(v)):
        raise ValueError('v 含 NaN/Inf —— 必须先 dropna')
    K = int(g.max()) + 1
    cnt = np.bincount(g, minlength=K).astype(float)
    ssum = np.bincount(g, weights=v, minlength=K)
    ok = cnt > 0
    mean = np.where(ok, ssum / np.maximum(cnt, 1), 0.0)
    grand = v.mean()
    ss_b = float((cnt[ok] * (mean[ok] - grand) ** 2).sum())
    ss_w = float((v ** 2).sum() - (ssum[ok] ** 2 / cnt[ok]).sum())
    return ss_b, ss_w


def eta2_perm(v, labels, n_perm, rng):
    """返回 (eta2, 零分布中位, p_le, p_ge)。

    p_le = 随机重排给出更小 η² 的比例；p_ge = 给出更大 η² 的比例。
    想说明「分组效应小」看 p_le；想说明「分组效应大」看 p_ge。
    """
    _, gi = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    b, w = decomp(v, gi)
    eta = b / (b + w) if (b + w) > 0 else 0.0
    gp = gi.copy()
    lo = hi = 0
    draws = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(gp)
        bb, ww = decomp(v, gp)
        e = bb / (bb + ww) if (bb + ww) > 0 else 0.0
        draws[i] = e
        if e <= eta:
            lo += 1
        if e >= eta:
            hi += 1
    return eta, float(np.median(draws)), (lo + 1) / (n_perm + 1), (hi + 1) / (n_perm + 1)


def add(tag, v, group_labels, univ_labels, n_perm, rng, out):
    v = np.asarray(v, dtype=float)
    m = np.isfinite(v)
    v, gl, ul = v[m], np.asarray(group_labels, dtype=str)[m], np.asarray(univ_labels, dtype=str)[m]
    for scheme, lab in (('batch', gl), ('university', ul)):
        if len(np.unique(lab)) < 2:
            continue
        e, med, p_le, p_ge = eta2_perm(v, lab, n_perm, rng)
        out.append(dict(quantity=tag, scheme=scheme, n=len(v),
                        n_groups=int(len(np.unique(lab))),
                        eta2=e, null_median=med, p_le=p_le, p_ge=p_ge))
        print('  %-26s %-11s n=%2d  eta2=%.4f  null_med=%.4f  p_le=%.4f  p_ge=%.4f'
              % (tag, scheme, len(v), e, med, p_le, p_ge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perm', type=int, default=20000)
    a = ap.parse_args()
    n_perm = a.perm
    rng = np.random.default_rng(SEED)
    out = []

    print('== AE：跨 4 个试验活动 / 2 所大学（口径 = 自归一化累积命中数的 t 值）')
    ae = pd.read_csv(os.path.join(RES, 'l1_hi_crosslab.csv'))
    ae['univ'] = ae.batch.map(LAB_OF)
    for variant in ('cum_hits', 'cum_energy'):
        sub = ae[ae.variant == variant]
        if sub.empty:
            continue
        for col in ('t50_pct', 't85_pct'):
            add('AE %s %s' % (variant, col), sub[col].values,
                sub.batch.values, sub.univ.values, n_perm, rng, out)

    print('\n== 导波 PZT：跨 3 个批次 / 2 个实验室')
    pz = pd.read_csv(os.path.join(RES, 'pzt_crosslab_summary.csv'))
    for col in ('tof_slope_ts', 't50', 't85'):
        if col in pz.columns:
            add('PZT %s' % col, pz[col].values, pz.batch.values,
                pz.batch.map(LAB_OF).values, n_perm, rng, out)

    print('\n== PZT 因果阈值配置（§2.2 的反例证据，逐配置）')
    cs = pd.read_csv(os.path.join(RES, 'pzt_causal_summary.csv'))
    for (rule, thr), g in cs.groupby(['rule', 'threshold']):
        g = g.dropna(subset=['onset_pct'])          # <== NaN 必须先去
        if len(g) < 6:
            continue
        e, med, p_le, p_ge = eta2_perm(g.onset_pct.values.astype(float),
                                       g.batch.values, n_perm, rng)
        out.append(dict(quantity='causal %s@%.2f' % (rule, thr), scheme='batch',
                        n=len(g), n_groups=int(g.batch.nunique()),
                        eta2=e, null_median=med, p_le=p_le, p_ge=p_ge))
        print('  %-26s n=%2d  eta2=%.4f  null_med=%.4f  p_ge=%.4f'
              % ('%s@%.2f' % (rule, thr), len(g), e, med, p_ge))

    df = pd.DataFrame(out)
    fp = os.path.join(RES, 'crosslab_variance.csv')
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    print('\n已写 %s（%d 行，%d 次置换/项）' % (fp, len(df), n_perm))

    # 自检：核心结论必须复现
    key = df[(df.quantity == 'AE cum_hits t50_pct') & (df.scheme == 'university')]
    key2 = df[(df.quantity == 'PZT tof_slope_ts') & (df.scheme == 'university')]
    print('\n[自检] 论文 §3.3 的核心数字：')
    for name, k in (('AE t50 / 2 所大学', key), ('PZT 斜率 / 2 个实验室', key2)):
        if len(k):
            r = k.iloc[0]
            ok = (r.eta2 < 0.01) and (r.p_le < 0.05)
            print('  %-22s eta2=%.4f  p_le=%.4f  %s'
                  % (name, r.eta2, r.p_le, 'OK' if ok else '**与论文不符，检查数据**'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
