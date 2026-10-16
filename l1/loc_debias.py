#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通道时间偏置的去偏标定 —— 把 δₖ 显式建进走时模型。

动机（由上一步的探针证实）
--------------------------
`loc_bias_probe.py` 显示：三组的**残差与通道强相关**，且偏置量级 = 残差量级：

    L1-49   通道 1/2/4 = −4.35 / +3.37 / +1.47 µs   （rms 3.74）
    L1-54   通道 1/4   = −1.66 / +1.64 µs           （rms 1.53）
    L1-55   通道 4     = −23.75 µs                   （rms 12.07）

按 6600 m/s 折算，L1-55 的通道 4 相当于 **约 157 mm 的等效距离误差** ——
这就是「Y 向定位差」的真正来源，而不是几何或声速。

`loc_vel_calib.py` 也印证了这点：调声速虽能把 rms 压下去，
但三组最优**全部落在搜索边界**，且 L1-55 标出 vy < vx（物理上讲不通）
—— 因为它在用错误的自由度（各向异性）去补偿真正的自由度（通道延迟）。

模型
----
    t(P → S_k) = T_k(P; vx, vy) + δ_k + t0
其中 t0 是事件绝对时刻（每簇不同，被 TDOA 差掉），δ_k 是**通道固定延迟**
（跨事件共享，这正是「探头长期偏晚」的数学形式）。

标定 δ 的做法（交替迭代）
-------------------------
1. 令 δ = 0，用现行模型定位每个簇；
2. 在最优位置处算**去均值后的残差** r_k = (T_k − mean T) − (t_k − mean t)。
   由模型可知 r_k = δ_k − mean(δ)，即**去均值残差直接就是 δ 的零均值版本**；
3. 取全体事件 r_k 的中位作为 δ̂_k（含一个不可辨的常数，拟合前再减均值）；
4. 用 δ̂ 修正 t_k 后回到第 1 步，迭代若干轮。

输出
    l1/results/loc_debias.csv       每组：δ̂ / 去偏前后 rms / 改善
    l1/results/loc_debias_delta.csv 每组 × 每通道的 δ̂
    l1/figures/fig_loc_debias.png   去偏前后 rms 对比

用法
    python l1/loc_debias.py                       # 默认三组
    python l1/loc_debias.py --groups L1-55 --iters 5
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
FIG = L.FIG
DEFAULT_GROUPS = list(L.GROUPS)      # 全 9 组（与 ae_locate 一致），不再只取 3 组试点
CHS = sorted(L.SENSORS)


def locate_debiased(t_us, ch, tmap, gx, gy, delta):
    """带通道偏置的定位：先按 δ 修正实测到时，再照常做 TDOA 最小二乘。"""
    t_fix = t_us - np.array([delta[int(c)] for c in ch])
    o = np.argsort(t_fix)                      # 修正后重新排序（最早通道可能变）
    t_fix, ch_s = t_fix[o], ch[o]
    dt = t_fix - t_fix[0]
    pred = tmap[np.asarray(ch_s) - 1]
    pred = pred - pred[0][None, :, :]
    res = pred - dt[:, None, None]
    cost = np.sqrt(np.mean(res * res, axis=0))
    j = int(np.argmin(cost))
    iy, ix = np.unravel_index(j, (gy.size, gx.size))
    # 去均值残差（按通道原编号，供 δ 估计）
    rmean = res[:, iy, ix] - res[:, iy, ix].mean()
    rk = {int(c): float(v) for c, v in zip(ch_s, rmean)}
    return (float(gx[ix]), float(gy[iy]), float(cost[iy, ix]), rk)


def clusters_of(gid, n_sample, seed):
    """与 ae_locate.run 完全一致的取簇逻辑。"""
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
        out.append((tt[o] * 1e6, cc[o].astype(np.int32)))
    return out, n_all


def rms_of(clus, tmap, gx, gy, delta):
    r = [locate_debiased(t, c, tmap, gx, gy, delta)[2] for t, c in clus]
    return float(np.median(r)), np.asarray(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(DEFAULT_GROUPS))
    ap.add_argument('--n-sample', type=int, default=600)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    gx = np.arange(0.0, L.LX + 1e-9, 2.0)
    gy = np.arange(0.0, L.LY + 1e-9, 2.0)
    rows, drows = [], []

    for gid in a.groups.split(','):
        t0 = time.time()
        vx, vy = L.VEL.get(gid, (L.VX_DEF, L.VY_DEF))
        clus, n_all = clusters_of(gid, a.n_sample, a.seed)
        if not clus:
            print('== %s  无可用簇，跳过' % gid)
            continue
        tmap = L.make_tmap(gx, gy, vx, vy)
        print('== %s   vx=%.0f vy=%.0f   簇=%d（总 %d）'
              % (gid, vx, vy, len(clus), n_all))

        delta = {c: 0.0 for c in CHS}
        r_base, _ = rms_of(clus, tmap, gx, gy, delta)
        print('   去偏前 rms 中位 = %.2f µs' % r_base)

        hist = [r_base]
        for it in range(a.iters):
            acc = {c: [] for c in CHS}
            for t_us, ch in clus:
                _x, _y, _r, rk = locate_debiased(t_us, ch, tmap, gx, gy, delta)
                for c, v in rk.items():
                    acc[c].append(v)
            new = {}
            for c in CHS:
                v = np.asarray(acc[c], dtype=float)
                new[c] = delta[c] - (float(np.median(v)) if v.size else 0.0)
            m = np.mean([new[c] for c in CHS])       # 只差一个不可辨常数，去掉
            for c in CHS:
                new[c] -= m
            shift = max(abs(new[c] - delta[c]) for c in CHS)
            delta = new
            r_it, _ = rms_of(clus, tmap, gx, gy, delta)
            hist.append(r_it)
            print('   迭代 %d：rms=%.2f µs   δ=%s   （Δmax=%.2f）'
                  % (it + 1, r_it, {c: round(delta[c], 2) for c in CHS}, shift))

        r_new = hist[-1]
        for c in CHS:
            drows.append({'gid': gid, 'channel': c, 'delta_us': delta[c]})
        rows.append({'gid': gid, 'n_total': n_all, 'n_used': len(clus),
                     'vx': vx, 'vy': vy,
                     'rms_before': r_base, 'rms_after': r_new,
                     'drop_pct': 100.0 * (r_base - r_new) / r_base if r_base > 0 else 0.0,
                     'delta_absmax_us': max(abs(delta[c]) for c in CHS)})
        print('   → rms %.2f → %.2f µs（降 %.0f%%）  用时 %.0fs\n'
              % (r_base, r_new, rows[-1]['drop_pct'], time.time() - t0))

    if not rows:
        return
    df = pd.DataFrame(rows)
    dd = pd.DataFrame(drows)
    df.to_csv(os.path.join(RES, 'loc_debias.csv'), index=False, encoding='utf-8-sig')
    dd.to_csv(os.path.join(RES, 'loc_debias_delta.csv'), index=False, encoding='utf-8-sig')

    show = df.copy()
    for c in ('rms_before', 'rms_after', 'drop_pct', 'delta_absmax_us'):
        show[c] = show[c].round(2)
    print('=== 去偏汇总 ===')
    print(show.to_string(index=False))
    print('\n=== 标定出的通道延迟 δₖ (µs) ===')
    print(dd.pivot(index='gid', columns='channel', values='delta_us').round(2).to_string())
    print('\n已写 %s / %s'
          % (os.path.join(RES, 'loc_debias.csv'), os.path.join(RES, 'loc_debias_delta.csv')))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        g = list(df['gid'])
        x = np.arange(len(g))
        axes[0].bar(x - .2, df['rms_before'], .4, label='去偏前', color='#d62728')
        axes[0].bar(x + .2, df['rms_after'], .4, label='去偏后', color='#2ca02c')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(g)
        axes[0].set_ylabel('走时残差中位 (µs)')
        axes[0].set_title('去偏前后的定位残差')
        axes[0].legend()
        axes[0].grid(alpha=.3, axis='y')
        piv = dd.pivot(index='gid', columns='channel', values='delta_us')
        w = .8 / piv.shape[1]
        for i, c in enumerate(piv.columns):
            axes[1].bar(x + (i - piv.shape[1] / 2 + .5) * w, piv[c], w, label='通道 %d' % c)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(g)
        axes[1].set_ylabel('标定延迟 $\\delta_k$ (µs)')
        axes[1].set_title('各通道的固定延迟')
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=.3, axis='y')
        fig.tight_layout()
        os.makedirs(FIG, exist_ok=True)
        out = os.path.join(FIG, 'fig_loc_debias.png')
        fig.savefig(out, dpi=200)
        fig.savefig(out.replace('.png', '.pdf'))
        plt.close(fig)
        print('已写 %s' % out)
    except Exception as e:                              # noqa: BLE001
        print('  画图跳过：%s' % e)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
