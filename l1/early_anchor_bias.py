# -*- coding: utf-8 -*-
"""早锚定偏差 + Saxena 三指标的定量（回答审稿意见 2 与 3）。

两个问题
--------
问题 2  用早期数据归一化（早锚定）能否近似末端归一化？代价多大？
问题 3  SNHI 与单调性 / 趋势性 / 可预测性这三把尺子是什么关系？

数据口径
--------
C2 九组：`_l1_ae_shape_<gid>.npz` 的 life（0 至 1）与 hit_rate（100 点）
C3/C4 十五组：`_l1ae_frames_<gid>.npz` 的 frame 与 n（每帧命中数），
             配 `_l1cyc_<gid>.npz` 的 frame / cycle / n_f，得 x = cycle / n_f
C1 五组无曲线缓存，不在本统计内（结论里须声明覆盖 24 组而非 29 组）。

问题 2 的量
-----------
share(f) = C(f) / C(end)，即寿命分数 f 时已累积的份额。
它决定早锚定能否用固定系数补偿：若跨试件相差几个数量级，则不能。

问题 3 的三把尺子（口径写死，便于复核）
--------------------------------------
单调性    每组 |Spearman(累积曲线, 寿命分数)|，报中位
趋势性    两两试件在公共寿命网格上累积曲线的 Spearman，报中位与最小
可预测性  exp(-std(t85) / mean(t85))，t85 为累积曲线达终值 85% 的寿命分数

用法
    python l1/early_anchor_bias.py
写 l1/results/early_anchor_bias.csv 与 l1/results/early_anchor_bias.md
"""
import os
import statistics as st
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

C2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60']
C34 = ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24',
       'L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31',
       'L1-34', 'L1-35', 'L1-36', 'L1-41', 'L1-44']
FR = (0.05, 0.10, 0.20, 0.30, 0.50)
GRID = np.linspace(0.01, 0.99, 99)


def curve_c2(gid):
    with np.load(os.path.join(RES, '_l1_ae_shape_%s.npz' % gid), allow_pickle=True) as z:
        life = z['life'].astype(float)
        hr = np.nan_to_num(z['hit_rate'].astype(float), nan=0.0)
    return life, np.cumsum(hr), hr


def curve_c34(gid):
    with np.load(os.path.join(RES, '_l1ae_frames_%s.npz' % gid), allow_pickle=True) as z:
        fr = z['frame'].astype(np.int64)
        n = np.nan_to_num(z['n'].astype(float), nan=0.0)
    with np.load(os.path.join(RES, '_l1cyc_%s.npz' % gid), allow_pickle=True) as z:
        cf = z['frame'].astype(np.int64)
        cyc = z['cycle'].astype(float)
        n_f = float(z['n_f'])
    m = dict(zip(cf.tolist(), cyc.tolist()))
    keep = np.array([int(f) in m for f in fr])
    if keep.sum() < 5 or not (n_f > 0):
        return None, None, None
    c = np.array([m[int(f)] for f in fr[keep]], float)
    h = n[keep]
    ok = np.isfinite(c) & (c >= 0)
    o = np.argsort(c[ok])
    return c[ok][o] / n_f, np.cumsum(h[ok][o]), h[ok][o]


def at(life, cum, frac, ref):
    tgt = frac * ref
    i = int(np.searchsorted(cum, tgt))
    if i >= cum.size:
        return np.nan
    if i == 0:
        return float(life[0])
    c0, c1 = cum[i - 1], cum[i]
    if c1 <= c0:
        return float(life[i])
    return float(life[i - 1] + (tgt - c0) / (c1 - c0) * (life[i] - life[i - 1]))


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows, curves, rates = [], {}, {}
    for gid in C2 + C34:
        if gid in C2:
            life, cum, rate = curve_c2(gid)
        else:
            life, cum, rate = curve_c34(gid)
        if life is None or life.size < 8 or not (cum[-1] > 0):
            print('[%s] 曲线不可用，跳过' % gid)
            continue
        rec = {'gid': gid, 'campaign': 'C2' if gid in C2 else ('C3' if gid in C34[:5] else 'C4'),
               'n_pt': int(life.size)}
        for f in FR:
            k = max(np.searchsorted(life, f, side='right') - 1, 0)
            rec['share_%.2f' % f] = cum[k] / cum[-1]
        rec['monotonicity'] = abs(spearman(cum, life))
        rec['mon_rate'] = abs(spearman(rate, life))
        rec['t85'] = at(life, cum, 0.85, cum[-1])
        rows.append(rec)
        curves[gid] = np.interp(GRID, life, cum / cum[-1])
        m = np.mean(rate) or 1.0
        rates[gid] = (life, rate / m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'early_anchor_bias.csv'), index=False, encoding='utf-8-sig')

    L = ['# 早锚定偏差与 Saxena 三指标\n',
         '覆盖 %d 组（C2 九组 + C3/C4 十五组中的可用者），C1 五组无曲线缓存不在内。\n' % len(df),
         '## 1. 寿命分数 f 时已累积的份额 C(f)/C(end)\n',
         '| f | 中位 | 最小 | 最大 | 跨试件倍数 | CV |', '| --- | --- | --- | --- | --- | --- |']
    for f in FR:
        v = df['share_%.2f' % f].to_numpy()
        L.append('| %.2f | %.3f | %.3f | %.3f | %.1f | %.2f |'
                 % (f, st.median(v), v.min(), v.max(), v.max() / max(v.min(), 1e-9),
                    np.std(v) / np.mean(v)))

    t85 = df['t85'].to_numpy()
    prog = float(np.exp(-np.std(t85) / np.mean(t85)))
    pairs = []
    ks = list(curves)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r = spearman(curves[ks[i]], curves[ks[j]])
            if np.isfinite(r):
                pairs.append(r)
    rpairs = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = rates[ks[i]], rates[ks[j]]
            g = np.linspace(0.05, 0.95, 40)
            r = spearman(np.interp(g, a[0], a[1]), np.interp(g, b[0], b[1]))
            if np.isfinite(r):
                rpairs.append(r)

    L.append('\n## 2. Saxena 三指标\n')
    L.append('分两个口径报，因为累积曲线上的单调性是**退化的**：\n')
    L.append('| 指标 | 累积曲线 | 速率信号（每箱命中数） | 口径 |')
    L.append('| --- | --- | --- | --- |')
    L.append('| 单调性（中位） | %.3f | %.3f | 每组 \\|Spearman(信号, 寿命)\\| |'
             % (st.median(df['monotonicity']), st.median(df['mon_rate'])))
    L.append('| 趋势性（中位） | %.3f | %.3f | 两两试件在公共网格上的 Spearman |'
             % (st.median(pairs), st.median(rpairs)))
    L.append('| 趋势性（最小） | %.3f | %.3f | 同上，取最小 |' % (min(pairs), min(rpairs)))
    L.append('| 可预测性 | %.3f | 同 | exp(−std(t85)/mean(t85)) |' % prog)
    L.append('\n对照：t85 的中位 %.1f%%，范围 %.1f%% 至 %.1f%%。\n'
             % (100 * st.median(t85), 100 * np.nanmin(t85), 100 * np.nanmax(t85)))

    L.append('\n## 3. 结论\n')
    L.append('1. 早锚定无法近似末端归一：寿命 5%% 时已累积份额跨试件相差 '
             '%.0f 倍，到寿命一半仍相差 %.1f 倍。任何固定修正系数都无法补偿。'
             % (df['share_0.05'].max() / max(df['share_0.05'].min(), 1e-9),
                df['share_0.50'].max() / max(df['share_0.50'].min(), 1e-9)))
    L.append('2. **累积曲线上的 Saxena 指标是退化的**：自归一化后的累积曲线必然单调、'
             '两两必然高相关，实测两项都是 %.3f 与 %.3f，无信息量。'
             '三把尺子应作用于**生成 SNHI 的速率信号**，那里才有区分度（单调性中位 %.3f）。'
             % (st.median(df['monotonicity']), st.median(pairs), st.median(df['mon_rate'])))
    L.append('3. 可预测性 %.3f，对应 t85 的离散（中位 %.1f%%，范围 %.1f%% 至 %.1f%%）。'
             '这是三把尺子里唯一在累积曲线上仍有意义的。'
             % (prog, 100 * st.median(t85), 100 * np.nanmin(t85), 100 * np.nanmax(t85)))

    open(os.path.join(RES, 'early_anchor_bias.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\n覆盖 %d 组；明细 -> results/early_anchor_bias.csv' % len(df))
    return 0


if __name__ == '__main__':
    sys.exit(main())
