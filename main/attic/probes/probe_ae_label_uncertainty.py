# -*- coding: utf-8 -*-
"""只用声发射，算 b1 / b3 自身的不确定度。

背景
----
之前的 b1 由「能量累积曲线的台阶」单一路径给出，但三套不同实现之间能差
25 至 36 个百分点。若这个不确定度与分级带宽（±15）同量级，那么
A / E / D 这些标签就没有可辨识性，讨论「方法对不对」是空转。

做法
----
声发射内部取 3 条独立通道 × 2 种检测算法 = 6 个估计：
    通道: 能量(块 Σ MeanSquare) / 频谱(块中位 RMSFrequency) / 冲激(块中位 Kurtosis)
    算法: 台阶(argmax 崩塌前有效台阶) / 簇(前 1% 高能命中的密度簇)
对每个估计取「首次损伤」与「崩塌」两个位置，然后报 6 个估计的中位与四分位距。
全程只用 AE，不碰应变与光纤。

用法
    python main/probe_ae_label_uncertainty.py
产物
    main/results/ae_label_uncertainty.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pipeline as P                                          # noqa: E402

GIDS = ['%03d' % i for i in range(15, 28)]
NB = P.NB_BLOCK
SM = P.SM_BLOCK


def channel_series(ae):
    """三条通道的块级序列。"""
    n = len(ae)
    idx = np.array_split(np.arange(n), NB)
    out = {}
    col = P._pick_col(ae, 'MeanSquare', ['Peak'])
    if col is not None:
        e = np.maximum(ae[col].to_numpy(float), 0.0)
        if col == 'Peak':
            e = e ** 2
        s = np.array([float(np.sum(e[i])) if i.size else np.nan for i in idx])
        out['energy'] = P._smooth_edge(np.log10(np.maximum(s, 1e-12)), SM)
        out['_e'] = e
    for name, c, fb in (('freq', 'RMSFrequency', ['MeanSquaredFrequency']),
                        ('impuls', 'Kurtosis', ['PeakFactor', 'ImpulseFactor'])):
        use = P._pick_col(ae, c, fb)
        if use is None:
            continue
        v = ae[use].to_numpy(float)
        s = np.array([float(np.median(v[i][np.isfinite(v[i])]))
                      if i.size and np.isfinite(v[i]).any() else np.nan for i in idx])
        out[name] = P._smooth_edge(s, SM)
    return out


def main():
    rows = []
    for gid in GIDS:
        dl = P._load_ae_strain(gid)
        ae = dl.ae
        if ae is None or len(ae) < 400:
            print('%s 无 AE，跳过' % gid)
            continue
        ser = channel_series(ae)
        e = ser.get('_e')
        b1s, b3s, src = [], [], []
        for name in ('energy', 'freq', 'impuls'):
            s = ser.get(name)
            if s is None:
                continue
            # --- 算法一：台阶 ---
            p2, p3 = P._step_labels(s)
            if p2 is not None:
                b1s.append(p2); src.append('%s-step' % name)
            if p3 is not None:
                b3s.append(p3)
            # --- 算法二：密度簇（仅能量通道有「能量」含义，其余用同一套块序列）---
            if name == 'energy' and e is not None:
                cl = [c for c in P._energy_clusters(e)
                      if c['n'] >= P.CLUSTER_MIN_HITS
                      and c['center'] >= P.CLUSTER_START_MIN]
                cl.sort(key=lambda c: c['center'])
                if cl:
                    b3s.append(cl[-1]['center'])
                    if len(cl) >= 2:
                        b1s.append(cl[0]['center']); src.append('energy-cluster')
        b1s = np.array(b1s) if b1s else np.array([])
        b3s = np.array(b3s) if b3s else np.array([])

        def stat(a):
            if a.size == 0:
                return (np.nan, np.nan, np.nan, 0)
            q1, q3 = (np.percentile(a, 25), np.percentile(a, 75)) if a.size >= 4 else (a.min(), a.max())
            return (float(np.median(a)), float(a.min()), float(a.max()), a.size)

        m1, lo1, hi1, n1 = stat(b1s)
        m3, lo3, hi3, n3 = stat(b3s)
        rows.append(dict(gid=gid, b1_med=m1, b1_lo=lo1, b1_hi=hi1, b1_n=n1,
                         b1_iqr=hi1 - lo1 if n1 >= 2 else np.nan,
                         b3_med=m3, b3_lo=lo3, b3_hi=hi3, b3_n=n3,
                         b3_span=hi3 - lo3 if n3 >= 2 else np.nan))
        print('%s  b1 %5s [%5s,%5s] n=%d 跨度%5s | b3 %5s [%5s,%5s] n=%d 跨度%5s'
              % (gid,
                 '-' if np.isnan(m1) else '%.1f' % m1,
                 '-' if np.isnan(lo1) else '%.1f' % lo1,
                 '-' if np.isnan(hi1) else '%.1f' % hi1, n1,
                 '-' if n1 < 2 else '%.1f' % (hi1 - lo1),
                 '-' if np.isnan(m3) else '%.1f' % m3,
                 '-' if np.isnan(lo3) else '%.1f' % lo3,
                 '-' if np.isnan(hi3) else '%.1f' % hi3, n3,
                 '-' if n3 < 2 else '%.1f' % (hi3 - lo3)))

    df = pd.DataFrame(rows)
    out = os.path.join(_HERE, 'results', 'ae_label_uncertainty.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    v = df.dropna(subset=['b1_iqr'])
    if len(v):
        print('\nb1 跨度（仅 n>=2 的组，共 %d 组）：中位 %.1f、最小 %.1f、最大 %.1f 个百分点'
              % (len(v), v['b1_iqr'].median(), v['b1_iqr'].min(), v['b1_iqr'].max()))
    v3 = df.dropna(subset=['b3_span'])
    if len(v3):
        print('b3 跨度（共 %d 组）：中位 %.1f、最小 %.1f、最大 %.1f 个百分点'
              % (len(v3), v3['b3_span'].median(), v3['b3_span'].min(), v3['b3_span'].max()))
    print('\n产物 ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
