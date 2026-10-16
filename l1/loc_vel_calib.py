#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AE 定位的速度标定 —— PDF 给的声速到底可不可信？

背景
----
`ae_locate.py` 的走时模型是**各向异性椭圆**
    t(P->S) = hypot((xP - xS) / vx, (yP - yS) / vy)
其中 (vx, vy) 直接取自各组自带 PDF。两个疑点：

1. C1 批的 PDF 原文注明「manual PZT recording performed after specimen failure,
   but considering sensor failure, these measurements might not be useful」
   —— 数据方自己说这组声速可能没用（`ae_locate.py` 已因此故意不登记 C1）。
2. **L1-55 的 PDF 值 vx = 2933 m/s，而其余 8 组都落在 4000 至 4290**；
   该组的走时残差中位 13.29 µs，其余组 0.85 至 3.9 —— `audit()` 因此判它不可用。

指标（关键）
------------
用**走时残差 rms**（`locate()` 的返回值）。它只衡量「几何 + 声速模型是否自洽」，
**不依赖真值**，也**不受源迁移影响**。

⚠️ 不要拿定位云的 σ 当精度指标。`run()` 打印的 `σ = (xs.std(), ys.std())` 是
「同一试件内所有事件定位点的空间散布」，主要来自**损伤源随寿命的迁移**
（`docs/待办与未决问题.md`：总迁移中位 139 mm），不是模型误差。
README 旧版的「X 向 σ14 / Y 向 σ46」就是这么读错的。

做法
----
固定几何与聚类窗口（与 `ae_locate.py` 完全一致），对每组在 (vx, vy) 上扫描：
先粗网格，再在最优点附近精化，目标函数 = 抽样簇的**中位 rms**。
最后与 PDF 值对比，看标定值能把残差压到什么程度。

输出
    l1/results/loc_vel_calib.csv   每组：PDF 速度 / 标定速度 / rms 对比 / 改善
    l1/results/loc_vel_curves.csv  每个扫描点的 (gid, vx, vy, rms)  —— 供画曲面
    l1/figures/fig_loc_vel.png     rms 曲面（每组一个小图）

用法
    python l1/loc_vel_calib.py                       # 全部 9 组（约 2 分钟）
    python l1/loc_vel_calib.py --groups L1-55        # 只做一组
    python l1/loc_vel_calib.py --n-sample 300        # 更省时
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

COARSE_VX = np.arange(3000.0, 5250.1, 250.0)
COARSE_VY = np.arange(4800.0, 7300.1, 250.0)
FINE_SPAN = 300.0
FINE_STEP = 100.0


def prep(gid, n_sample, seed):
    """把某组的簇表与搜索网格准备好（与 ae_locate.run 的取簇逻辑一致）。"""
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
    n_all = ok.size
    if n_all == 0:
        return None, None, None, 0, 0
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

    clus = []
    for i0, i1 in zip(st, en):
        tt = t_s[sel2[i0:i1]]
        cc = grp_ch[i0:i1]
        o = np.argsort(tt)
        clus.append((tt[o] * 1e6, cc[o]))

    gx = np.arange(0.0, L.LX + 1e-9, 2.0)
    gy = np.arange(0.0, L.LY + 1e-9, 2.0)
    return clus, gx, gy, n_all, len(clus)


def rms_med(clus, gx, gy, vx, vy):
    tmap = L.make_tmap(gx, gy, vx, vy)
    r = [L.locate(t, c, tmap, gx, gy)[2] for t, c in clus]
    return float(np.median(r))


def sweep(clus, gx, gy, vxs, vys, tag, rows, gid, t0):
    best = (np.inf, None, None)
    for vx in vxs:
        for vy in vys:
            v = rms_med(clus, gx, gy, vx, vy)
            rows.append({'gid': gid, 'vx': float(vx), 'vy': float(vy), 'rms': v,
                         'stage': tag})
            if v < best[0]:
                best = (v, float(vx), float(vy))
    print('    %s 最优 rms=%.2f µs  @ vx=%.0f vy=%.0f   （%.0fs）'
          % (tag, best[0], best[1], best[2], time.time() - t0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=None, help='逗号分隔，缺省=全部')
    ap.add_argument('--n-sample', type=int, default=600, help='标定用簇数')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    groups = (a.groups.split(',') if a.groups else L.GROUPS)
    out_rows, curve_rows = [], []

    for gid in groups:
        t0 = time.time()
        pdf_v = L.VEL.get(gid, (L.VX_DEF, L.VY_DEF))
        print('== %s   PDF 声速 vx=%.0f vy=%.0f' % (gid, pdf_v[0], pdf_v[1]))
        clus, gx, gy, n_all, n_use = prep(gid, a.n_sample, a.seed)
        if clus is None:
            print('   ≥%d 通道的簇为 0，跳过' % L.MIN_CH)
            continue
        print('   ≥4 通道的簇 = %d，抽样 %d 个用于标定' % (n_all, n_use))

        r_pdf = rms_med(clus, gx, gy, *pdf_v)
        print('   PDF 速度下的中位 rms = %.2f µs' % r_pdf)

        _, bx, by = sweep(clus, gx, gy, COARSE_VX, COARSE_VY, '粗扫', curve_rows, gid, t0)
        fvx = np.arange(bx - FINE_SPAN, bx + FINE_SPAN + 1e-9, FINE_STEP)
        fvy = np.arange(by - FINE_SPAN, by + FINE_SPAN + 1e-9, FINE_STEP)
        best, bx2, by2 = sweep(clus, gx, gy, fvx, fvy, '精扫', curve_rows, gid, t0)
        r_new = rms_med(clus, gx, gy, bx2, by2)

        at_edge = (bx2 - COARSE_VX.min() <= FINE_SPAN or COARSE_VX.max() - bx2 <= FINE_SPAN
                   or by2 - COARSE_VY.min() <= FINE_SPAN or COARSE_VY.max() - by2 <= FINE_SPAN)
        out_rows.append({
            'gid': gid, 'n_clusters': n_all, 'n_used': n_use,
            'vx_pdf': pdf_v[0], 'vy_pdf': pdf_v[1], 'rms_pdf': r_pdf,
            'vx_cal': bx2, 'vy_cal': by2, 'rms_cal': r_new,
            'dvx_pct': 100.0 * (bx2 - pdf_v[0]) / pdf_v[0],
            'dvy_pct': 100.0 * (by2 - pdf_v[1]) / pdf_v[1],
            'rms_drop_pct': 100.0 * (r_pdf - r_new) / r_pdf if r_pdf > 0 else 0.0,
            'at_edge': bool(at_edge),
        })
        print('   → 标定 vx=%.0f vy=%.0f   rms %.2f → %.2f µs（降 %.0f%%）%s\n'
              % (bx2, by2, r_pdf, r_new, out_rows[-1]['rms_drop_pct'],
                 '   ⚠️ 最优落在搜索边界，参数族可能不适用' if at_edge else ''))

    if not out_rows:
        return
    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(RES, 'loc_vel_calib.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(curve_rows).to_csv(os.path.join(RES, 'loc_vel_curves.csv'),
                                    index=False, encoding='utf-8-sig')

    show = df.copy()
    for c in ('vx_pdf', 'vy_pdf', 'vx_cal', 'vy_cal'):
        show[c] = show[c].round(0)
    for c in ('rms_pdf', 'rms_cal', 'dvx_pct', 'dvy_pct', 'rms_drop_pct'):
        show[c] = show[c].round(2)
    print('=== 标定汇总 ===')
    print(show.to_string(index=False))
    print('\n中位 rms：PDF %.2f → 标定 %.2f µs'
          % (df['rms_pdf'].median(), df['rms_cal'].median()))
    print('已写 %s' % os.path.join(RES, 'loc_vel_calib.csv'))

    try:
        plot(df, pd.DataFrame(curve_rows))
    except Exception as e:                              # noqa: BLE001
        print('  画图跳过：%s' % e)


def plot(df, cur):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    gids = list(df['gid'])
    n = len(gids)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow), squeeze=False)
    for k, gid in enumerate(gids):
        ax = axes[k // ncol][k % ncol]
        s = cur[(cur['gid'] == gid)].drop_duplicates(subset=['vx', 'vy'])
        piv = s.pivot_table(index='vy', columns='vx', values='rms', aggfunc='min')
        im = ax.imshow(piv.values, origin='lower', aspect='auto', cmap='viridis_r',
                       extent=(piv.columns.min(), piv.columns.max(),
                               piv.index.min(), piv.index.max()))
        r = df[df['gid'] == gid].iloc[0]
        ax.plot([r['vx_pdf']], [r['vy_pdf']], 'w*', ms=14, label='PDF')
        ax.plot([r['vx_cal']], [r['vy_cal']], 'r+', ms=14, mew=2.5, label='标定')
        ax.set_title('%s  rms %.2f → %.2f µs' % (gid, r['rms_pdf'], r['rms_cal']),
                     fontsize=11)
        ax.set_xlabel('$v_x$ (m/s)')
        if k % ncol == 0:
            ax.set_ylabel('$v_y$ (m/s)')
        ax.legend(fontsize=9, loc='upper right')
        fig.colorbar(im, ax=ax, label='中位 rms (µs)')
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle('AE 定位速度标定：走时残差曲面（★ = PDF 值，+ = 标定值）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, 'fig_loc_vel.png')
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print('已写 %s' % out)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
