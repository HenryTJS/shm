#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通道时间偏置探针 —— 残差是不是系统性地偏向某几个通道？

动机
----
`loc_vel_calib.py` 的结果指向「参数族不足」，而不是「声速没调好」：

  · 三组（L1-49 / L1-54 / L1-55）的最优**全部落在搜索边界**；
  · L1-55 标定出 **vy < vx**（5550 / 4500），正交各向异性板纵向必然更快，
    该解物理上讲不通 —— 它只是把残差硬凑下去了。

`ae_locate.py` 的 `audit()` 早已指出症结：**某探头长期偏晚**会产生
「稳定偏置而非随机误差」。但 `locate()` 的模型里**没有通道时间偏置项**
    t(P->S_k) = T(P, S_k) + d_k
这个偏置没处可去，就被最小二乘摊到 (x, y) 上，造成系统偏移。

探针
----
对每个簇定位后取**残差向量** r_k = (模型预测的 TDOA) - (实测 TDOA)。
因为 TDOA 以「最早到达通道」为参考，该通道残差恒为 0；
其余通道若**系统性同号**（且量级明显），就说明存在 δ_k。

判读
----
  · |中位残差| < 0.5 µs  → 无显著偏置；
  · 0.5 至 2 µs           → 有偏置，但大致被位置吸收；
  · > 2 µs                → 偏置主导，现行模型在该组不可用。

输出
    l1/results/loc_bias_probe.csv  每组 × 每通道：残差中位 / IQR / 出现次数

用法
    python l1/loc_bias_probe.py                      # L1-49 / L1-54 / L1-55
    python l1/loc_bias_probe.py --groups L1-55,L1-51
    python l1/loc_bias_probe.py --n-sample 800
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_locate as L                                     # noqa: E402

RES = L.RES
DEFAULT_GROUPS = ['L1-49', 'L1-54', 'L1-55']


def locate_res(t_us, ch, tmap, gx, gy):
    """同 ae_locate.locate，但额外返回最优点的**残差向量**（µs）。"""
    dt = t_us - t_us[0]
    pred = tmap[np.asarray(ch) - 1]
    pred = pred - pred[0][None, :, :]
    res = pred - dt[:, None, None]
    cost = np.sqrt(np.mean(res * res, axis=0))
    j = int(np.argmin(cost))
    iy, ix = np.unravel_index(j, (gy.size, gx.size))
    return (float(gx[ix]), float(gy[iy]), float(cost[iy, ix]),
            res[:, iy, ix], int(ch[0]))


def clusters_of(gid, n_sample, seed):
    """与 ae_locate.run 完全一致的取簇逻辑，返回 [(t_us, ch), ...]。"""
    a = L.read_hits(gid)
    t_s = a[:, 0]
    chan = a[:, 1].astype(np.int32)
    starts, ends = L.cluster_runs(t_s, L.WIN_US * 1e-6)
    cid = np.repeat(np.arange(starts.size), ends - starts)
    order = np.lexsort((t_s, chan, cid))
    c_s, ch_s = cid[order], chan[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = (c_s[1:] != c_s[:-1]) | (ch_s[1:] != ch_s[:-1])
    sel = order[first]
    nch = np.bincount(cid[sel], minlength=starts.size)
    ok = np.nonzero(nch >= L.MIN_CH)[0]
    if ok.size == 0:
        return [], 0
    n_all = ok.size
    if n_all > n_sample:
        rng = np.random.default_rng(seed)
        ok = np.sort(rng.choice(ok, n_sample, replace=False))
    sel_c, sel_ch = cid[sel], chan[sel]
    pos = np.searchsorted(ok, sel_c)
    m = (pos < ok.size) & (ok[np.clip(pos, 0, ok.size - 1)] == sel_c)
    sel2, grp_c, grp_ch = sel[m], sel_c[m], sel_ch[m]
    bnd = np.nonzero(np.diff(grp_c))[0] + 1
    st = np.concatenate(([0], bnd))
    en = np.concatenate((bnd, [grp_c.size]))
    out = []
    for i0, i1 in zip(st, en):
        tt = t_s[sel2[i0:i1]]
        cc = grp_ch[i0:i1]
        o = np.argsort(tt)
        out.append((tt[o] * 1e6, cc[o]))
    return out, n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(DEFAULT_GROUPS))
    ap.add_argument('--n-sample', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    gx = np.arange(0.0, L.LX + 1e-9, 2.0)
    gy = np.arange(0.0, L.LY + 1e-9, 2.0)
    rows = []

    for gid in a.groups.split(','):
        t0 = time.time()
        vx, vy = L.VEL.get(gid, (L.VX_DEF, L.VY_DEF))
        clus, n_all = clusters_of(gid, a.n_sample, a.seed)
        if not clus:
            print('== %s  无可用簇，跳过' % gid)
            continue
        tmap = L.make_tmap(gx, gy, vx, vy)

        per_ch = {c: [] for c in sorted(L.SENSORS)}
        rms_all = []
        for t_us, ch in clus:
            _x, _y, r, res, c0 = locate_res(t_us, ch, tmap, gx, gy)
            rms_all.append(r)
            for kk, c in enumerate(ch):
                per_ch[int(c)].append(res[kk])       # c0 的那个恒为 0

        print('== %s   vx=%.0f vy=%.0f   簇=%d（总 %d）  rms中位=%.2f µs  （%.0fs）'
              % (gid, vx, vy, len(clus), n_all, np.median(rms_all), time.time() - t0))
        for c in sorted(per_ch):
            v = np.asarray(per_ch[c], dtype=float)
            if v.size == 0:
                print('     通道 %d   未参与' % c)
                continue
            med = float(np.median(v))
            iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
            flag = ('偏置主导' if abs(med) > 2 else
                    '有偏置' if abs(med) > 0.5 else '无显著偏置')
            print('     通道 %d   n=%5d   残差中位=%+6.2f µs  IQR=%5.2f   %s'
                  % (c, v.size, med, iqr, flag))
            rows.append({'gid': gid, 'channel': c, 'n': int(v.size),
                         'res_median_us': med, 'res_iqr_us': iqr,
                         'res_mean_us': float(v.mean()), 'flag': flag,
                         'vx': vx, 'vy': vy})
        print()

    if rows:
        out = os.path.join(RES, 'loc_bias_probe.csv')
        pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
        print('已写 %s' % out)
        df = pd.DataFrame(rows)
        worst = df.reindex(df['res_median_us'].abs().sort_values(ascending=False).index)
        print('\n=== 偏置最大的 8 条 ===')
        print(worst.head(8).to_string(index=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
