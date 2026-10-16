#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通道延迟 δ 是否随寿命漂移 —— 分窗标定 + 随机分组零假设对照。

动机
----
`loc_debias.py` 在**全寿命**上标定出一组固定 δ，把 9 组的残差中位从
0.84–12.07 µs 压到 0.45–2.37 µs。但「δ 固定」这个前提本身需要检验：

若探头耦合、线缆或采集前端的状态在试验中变化，δ 就会随寿命漂移，
那么固定标定只是「把漂移平均掉」，而不是真正修正 —— 定位在早期和
晚期会分别偏在相反方向。

方法
----
把每组的簇按**首达时刻**排成 k 个等簇数窗口，在每个窗口内独立标定 δ：

* `life`   方案：按寿命顺序切窗。若 δ 有漂移，窗口间应呈系统性趋势；
* `random` 方案：把同一批簇**随机**分成等量 k 组。这是**零假设基线**，
  其窗口间波动只来自有限样本，不含任何物理漂移。

判据
----
只有当 `life` 的窗口间波动**明显超过** `random` 基线时，才能说 δ 随寿命
漂移。直接看 `life` 曲线的起伏会误判 —— 有限样本本身就带来抖动。

注意：δ 只有**相对值**可辨（全体 δ 加同一常数会被 TDOA 差掉），
故所有 δ 均已去均值。这也意味着「所有通道同步漂移同一个量」不可见，
但它同样不影响定位精度（被事件时刻 t₀ 吸收），所以关注相对漂移是对的。

输出
    l1/results/loc_debias_drift.csv       每组 × 方案 × 窗口的 δ 与 rms
    l1/figures/fig_loc_debias_drift.png   9 组子图：δ 随寿命的走势 + 随机基线带

用法
    python l1/loc_debias_drift.py
    python l1/loc_debias_drift.py --groups L1-55 --windows 8
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_locate as L                                     # noqa: E402
import loc_debias as D                                    # noqa: E402

RES = L.RES
FIG = L.FIG
CHS = sorted(L.SENSORS)
DCOLS = ['d%d' % c for c in CHS]


def calib(clus, tmap, gx, gy, iters=4):
    """在给定簇集合上交替迭代标定 δ。返回 (delta, rms_before, rms_after)。

    与 `loc_debias.main` 内的迭代完全一致：去均值残差 = δ 的零均值版本，
    故更新量为 `δ − median(r_k)`，再整体去均值消掉不可辨常数。
    """
    zero = {c: 0.0 for c in CHS}
    r0 = D.rms_of(clus, tmap, gx, gy, zero)[0]
    delta = dict(zero)
    for _ in range(iters):
        acc = {c: [] for c in CHS}
        for t_us, ch in clus:
            _x, _y, _r, rk = D.locate_debiased(t_us, ch, tmap, gx, gy, delta)
            for c, v in rk.items():
                acc[c].append(v)
        new = {}
        for c in CHS:
            v = np.asarray(acc[c], dtype=float)
            new[c] = delta[c] - (float(np.median(v)) if v.size else 0.0)
        m = float(np.mean([new[c] for c in CHS]))
        for c in CHS:
            new[c] -= m
        delta = new
    r1 = D.rms_of(clus, tmap, gx, gy, delta)[0]
    return delta, r0, r1


def windows(clus, k, scheme, seed=0):
    """按首达时刻的 k 等分位切窗，返回 (窗口列表, 每窗 (n, t_lo, t_hi))。

    **为何不用「等簇数」切**：事件在时间上高度聚集 —— 实测 L1-49 的 2000
    个簇里有 1200 个挤在寿命的 0.56–0.63 之间。等簇数切会让中间若干窗口
    只覆盖 1% 的寿命跨度，它们的 δ 差异其实全是噪声，却会被误读成
    「漂移」。按时间等分位切才能保证每个窗口真正对应一段寿命。

    scheme='life'   真实寿命窗；
    scheme='random' 用**同样的窗口大小序列**做随机分配（零假设基线）。
    """
    t0 = np.array([float(c[0][0]) for c in clus])
    n = len(clus)
    ed = np.quantile(t0, np.linspace(0.0, 1.0, k + 1))
    ed[0] -= 1.0
    ed[-1] += 1.0
    wid = np.clip(np.searchsorted(ed, t0, side='right') - 1, 0, k - 1)
    if scheme == 'random':
        sizes = np.bincount(wid, minlength=k)
        perm = np.random.default_rng(seed).permutation(n)
        cuts = np.concatenate(([0], np.cumsum(sizes)))
        picks = [perm[cuts[i]:cuts[i + 1]] for i in range(k)]
    else:
        picks = [np.nonzero(wid == i)[0] for i in range(k)]
    wins, infos = [], []
    for g in picks:
        wins.append(([clus[i] for i in g],
                     float(np.mean(t0[g])) if g.size else float('nan')))
        infos.append((int(g.size),
                      float(t0[g].min()) if g.size else float('nan'),
                      float(t0[g].max()) if g.size else float('nan')))
    return wins, infos


def spatial_stratified(clus, tmap, gx, gy, n_grid=3, min_n=40, ref=1):
    """固定空间网格，比较同位置早期/晚期的通道间到时差 —— 与 δ 解耦的漂移估计。

    为何需要它
    ----------
    分窗标定出的 δ 差异**混入了损伤源迁移**：δ 的估计依赖于定位位置，而
    位置随寿命迁移（实测 L1-55 质心从 (50, 16) 跑到 (90, 134)），于是模型在
    不同位置的偏差被算进了 δ，看起来很像是「探头延迟漂移」。

    做法
    ----
    用**未去偏模型**（δ≡0，对早/晚一视同仁）定位，再在同一空间网格内比较
    `t_k − t_ref` 的中位。位置偏差在早/晚是**同一个函数**，故同一网格内的
    差异只能来自 δ 的真实变化。

    返回 [(cell_ix, cell_iy, ch, n_early, n_late, dt_early, dt_late, diff), ...]
    """
    zero = {c: 0.0 for c in CHS}
    xe = np.linspace(0.0, L.LX, n_grid + 1)
    ye = np.linspace(0.0, L.LY, n_grid + 1)
    recs = []
    for t_us, ch in clus:
        x, y, _r, _rk = D.locate_debiased(t_us, ch, tmap, gx, gy, zero)
        d = {int(c): float(t) for c, t in zip(ch, t_us)}
        if ref not in d:
            continue
        ix = int(np.clip(np.searchsorted(xe, x, side='right') - 1, 0, n_grid - 1))
        iy = int(np.clip(np.searchsorted(ye, y, side='right') - 1, 0, n_grid - 1))
        recs.append((d[min(d)], ix, iy,
                     {c: v - d[ref] for c, v in d.items() if c != ref}))
    if not recs:
        return []
    t_abs = np.array([r[0] for r in recs])
    med = float(np.median(t_abs))
    out = []
    for iy in range(n_grid):
        for ix in range(n_grid):
            sel = [i for i, r in enumerate(recs) if r[1] == ix and r[2] == iy]
            e = [i for i in sel if t_abs[i] <= med]
            late = [i for i in sel if t_abs[i] > med]
            if len(e) < min_n or len(late) < min_n:
                continue
            for c in CHS:
                if c == ref:
                    continue
                ve = np.array([recs[i][3][c] for i in e if c in recs[i][3]])
                vl = np.array([recs[i][3][c] for i in late if c in recs[i][3]])
                if ve.size < min_n or vl.size < min_n:
                    continue
                a0, b0 = float(np.median(ve)), float(np.median(vl))
                out.append((ix, iy, c, int(ve.size), int(vl.size), a0, b0, b0 - a0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(D.DEFAULT_GROUPS))
    ap.add_argument('--windows', type=int, default=5)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--n-sample', type=int, default=4000,
                    help='每组抽多少簇（再按时间等分位切成 --windows 份）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-spatial', dest='spatial', action='store_false',
                    help='跳过空间分层对照')
    ap.add_argument('--grid', type=int, default=3, help='空间分层网格数（默认 3×3）')
    ap.add_argument('--min-cell', type=int, default=40,
                    help='单个网格内早/晚各至少多少簇才纳入')
    a = ap.parse_args()

    gx = np.arange(0.0, L.LX + 1e-9, 2.0)
    gy = np.arange(0.0, L.LY + 1e-9, 2.0)
    rows = []
    cl_cache = {}          # gid -> clus，供空间分层段复用（避免重做全量分簇排序）

    for gid in [g.strip() for g in a.groups.split(',') if g.strip()]:
        t_start = time.time()
        vx, vy = L.VEL.get(gid, (L.VX_DEF, L.VY_DEF))
        clus, n_all = D.clusters_of(gid, a.n_sample, a.seed)
        if not clus:
            print('== %s  无可用簇，跳过' % gid)
            continue
        cl_cache[gid] = clus
        tmap = L.make_tmap(gx, gy, vx, vy)
        taus = np.array([float(c[0][0]) for c in clus])
        lo, hi = float(taus.min()), float(taus.max())
        span = max(hi - lo, 1e-9)
        print('== %s  簇=%d（总 %d）  寿命 %.0f→%.0f s'
              % (gid, len(clus), n_all, lo / 1e6, hi / 1e6))

        for scheme in ('life', 'random'):
            wins, infos = windows(clus, a.windows, scheme, a.seed)
            for wi, (sub, tm) in enumerate(wins):
                if len(sub) < 20:
                    continue
                delta, r0, r1 = calib(sub, tmap, gx, gy, a.iters)
                row = {'gid': gid, 'scheme': scheme, 'window': wi,
                       'n': len(sub), 'life_mid': (tm - lo) / span,
                       't_lo': (infos[wi][1] - lo) / span,
                       't_hi': (infos[wi][2] - lo) / span,
                       'rms_before': r0, 'rms_after': r1}
                for c in CHS:
                    row['d%d' % c] = delta[c]
                rows.append(row)
            if scheme == 'life':
                sub_df = pd.DataFrame([r for r in rows if r['gid'] == gid
                                       and r['scheme'] == 'life'])
                print(sub_df[['window', 'n', 't_lo', 't_hi', 'life_mid',
                              'rms_before', 'rms_after']
                             + DCOLS].round(2).to_string(index=False))
        print('   用时 %.0fs\n' % (time.time() - t_start))

    if not rows:
        return
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'loc_debias_drift.csv'), index=False,
              encoding='utf-8-sig')

    # ---- 判据汇总：life 的窗口间波动 vs random 基线 ----
    print('=== 漂移判据：life 窗口间标准差 / random 基线标准差 ===')
    print('%-8s %-6s %10s %10s %8s  %s'
          % ('gid', '通道', 'life_std', 'rand_std', '比值', '判定'))
    for gid in dict.fromkeys(df['gid']):
        for c in CHS:
            k = 'd%d' % c
            sl = df[(df.gid == gid) & (df.scheme == 'life')][k].std()
            sr = df[(df.gid == gid) & (df.scheme == 'random')][k].std()
            ratio = sl / sr if sr > 1e-9 else np.nan
            verdict = '漂移' if (ratio > 2.0 and sl > 0.5) else '无显著漂移'
            print('%-8s %-6d %10.2f %10.2f %8.2f  %s'
                  % (gid, c, sl, sr, ratio, verdict))

    # ---- 空间分层对照：把损伤源迁移造成的伪漂移剥掉 ----
    if a.spatial:
        print('\n=== 空间分层对照：同一网格内 早期 vs 晚期 的通道间到时差 ===')
        print('（用 δ≡0 定位；位置偏差在早/晚是同一函数，故同格差异只能来自 δ 真实变化）')
        srows = []
        for gid in dict.fromkeys(df['gid']):
            vx, vy = L.VEL.get(gid, (L.VX_DEF, L.VY_DEF))
            clus = cl_cache.get(gid)
            if not clus:
                continue
            tmap = L.make_tmap(gx, gy, vx, vy)
            grow = spatial_stratified(clus, tmap, gx, gy,
                                      n_grid=a.grid, min_n=a.min_cell)
            for r in grow:
                srows.append(dict(gid=gid, cell_x=r[0], cell_y=r[1], channel=r[2],
                                  n_early=r[3], n_late=r[4],
                                  dt_early_us=r[5], dt_late_us=r[6],
                                  diff_us=r[7]))
            ncell = len({(r[0], r[1]) for r in grow})
            print('  %-7s 有效网格 %d 个' % (gid, ncell), end='')
            for c in CHS:
                ds = [r[7] for r in grow if r[2] == c]
                if len(ds) < 2:
                    continue
                print('\n     通道 %d  网格数 %d  (晚−早) 中位 %+6.2f µs  范围 [%+.2f, %+.2f]'
                      % (c, len(ds), float(np.median(ds)), min(ds), max(ds)), end='')
            print()
        if srows:
            sp = os.path.join(RES, 'loc_debias_drift_spatial.csv')
            pd.DataFrame(srows).to_csv(sp, index=False, encoding='utf-8-sig')
            print('已写 %s' % sp)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        gids = list(dict.fromkeys(df['gid']))
        nc = 3
        nr = int(np.ceil(len(gids) / nc))
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
        fig, axes = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.3 * nr),
                                 squeeze=False)
        for i, gid in enumerate(gids):
            ax = axes[i // nc][i % nc]
            sl = df[(df.gid == gid) & (df.scheme == 'life')].sort_values('life_mid')
            sr = df[(df.gid == gid) & (df.scheme == 'random')]
            for j, c in enumerate(CHS):
                k = 'd%d' % c
                ax.errorbar(sl['life_mid'], sl[k],
                            xerr=[(sl['life_mid'] - sl['t_lo']).values,
                                  (sl['t_hi'] - sl['life_mid']).values],
                            fmt='o-', color=colors[j], ms=3.5, lw=1.2,
                            capsize=2, elinewidth=0.8, label='通道 %d' % c)
                mu, sd = sr[k].mean(), sr[k].std()
                ax.axhspan(mu - sd, mu + sd, color=colors[j], alpha=0.09)
                ax.axhline(mu, color=colors[j], lw=0.8, ls=':')
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(gid)
            ax.set_xlabel('寿命分数')
            ax.set_ylabel(r'$\delta_k-\bar\delta$  (µs)')
            ax.grid(alpha=.3)
            if i == 0:
                ax.legend(fontsize=8, ncol=2)
        for j in range(len(gids), nr * nc):
            axes[j // nc][j % nc].axis('off')
        fig.suptitle('通道延迟随寿命的走势（实线，横棒 = 该窗实际覆盖的寿命区间）'
                     ' vs 随机分窗基线（阴影 = ±1σ）', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        os.makedirs(FIG, exist_ok=True)
        out = os.path.join(FIG, 'fig_loc_debias_drift.png')
        fig.savefig(out, dpi=180)
        fig.savefig(out.replace('.png', '.pdf'))
        plt.close(fig)
        print('\n已写 %s' % out)
    except Exception as e:                                # noqa: BLE001
        print('  画图跳过：%s' % e)

    print('已写 %s' % os.path.join(RES, 'loc_debias_drift.csv'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
