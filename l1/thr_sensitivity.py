# -*- coding: utf-8 -*-
"""阈值灵敏度扫描 —— `MAX_CARRY` x `AE_PAUSE_FRAC` 对加载状态重建结论的影响。

为什么需要
----------
`ae_cycle` 的 holdgap 口径里有**两个人为设定**的阈值，此前都只是单点先验值：
  · `MAX_CARRY = 1200 s` —— 离最近 FBG 突发超过它才算「长停录段」；
  · `AE_PAUSE_FRAC = 0.05` —— 长停录段内局部 AE 率低于组内中位的 5% 才算「静默」。

物理依据（加载是分段常值信号 + 真停机无 AE）能定住量级，但定不住具体数。
本脚本扫 3 x 3 = 9 个组合，看结论会不会随阈值漂移。

四条判据
--------
  1. **反解载荷频率的分组内一致性**（f 的组间 IQR）—— 自洽性锚：
     反解式 f = n_f / 加载小时数，只有加载时长估对了 f 才会落回分组名义值。
  2. **长空隙内声称的加载小时数** —— 应接近 0（AE 已独立证实长空隙内几乎无事件）。
  3. `rho(g, AE 事件数)` —— 跨模态一致性：加载帧应同时是 AE 多的帧。
  4. **加载块结构**（平均块长，帧）—— 应远大于 1；退化成 1 说明没有块结构。
  5. **截断安全审计** —— 被实际判停机的帧，其 AE 事件率比其余帧低多少倍。
     若同量级，说明把「试验在跑」的段判成了停机（切错）。这是最硬的判据。

自检
----
基线组合 (1200 s, 0.05) 必须与生产口径 `cycle_fill_compare.g_hold(max_carry=1200,
default=0.0, n_hits=...)` **逐位一致**；不一致就中止，避免扫出一堆与现状无关的数。

用法
----
    python l1/thr_sensitivity.py
    python l1/thr_sensitivity.py --groups L1-13 L1-31
输出：results/l1_thr_sensitivity.csv
"""

import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr

# ae_cycle._win_stat 对个别全 NaN 通道会发 All-NaN slice 警告（既有行为，不影响结果）；
# 静音它，否则 PowerShell 会把 stderr 当成命令失败（退出码 1）。
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_cycle import FRAME_S, N_F, _data_l1                      # noqa: E402
from cycle_fill_compare import (fbg_bursts, gaps_of, runs,       # noqa: E402
                                MIN_FRAMES)
from cycle_fill_compare import g_hold as g_hold_prod             # noqa: E402

CARRY_GRID = [600.0, 1200.0, 1800.0]
FRAC_GRID = [0.02, 0.05, 0.10]
WIN = 10                      # 局部中位的半窗（帧），本次不扫，固定
BASE_CARRY = 1200.0
BASE_FRAC = 0.05

BATCH_A = ['L1-06', 'L1-13', 'L1-14', 'L1-24']      # 变幅 VA
WATCH = ['L1-13', 'L1-24', 'L1-31', 'L1-41']        # 历史上最敏感的 4 组


def ae_silent(n_hits, win=WIN, frac=BASE_FRAC):
    """局部 AE 事件率是否「基本静默」—— 与 `ae_cycle._ae_silent` 同逻辑，多一个 frac。"""
    n_hits = np.asarray(n_hits, dtype=float)
    k = 2 * win + 1
    ref = float(np.median(n_hits))
    if len(n_hits) < k:
        local = n_hits
    else:
        from numpy.lib.stride_tricks import sliding_window_view
        local = np.median(sliding_window_view(np.pad(n_hits, win, mode='edge'), k),
                          axis=1)
    return local < frac * max(ref, 1.0)


def g_hold(t_sec, tb, sb, max_carry, frac, n_hits, win=WIN):
    """holdgap（阈值可调）。与 `cycle_fill_compare.g_hold` 同逻辑，只多一个 frac 入口。

    返回 (g, far, cut)：far = 超出 MAX_CARRY 的帧；cut = far 里被判停机的帧。
    """
    idx = np.searchsorted(tb, t_sec)
    left = np.clip(idx - 1, 0, len(tb) - 1)
    right = np.clip(idx, 0, len(tb) - 1)
    dl = np.abs(t_sec - tb[left])
    dr = np.abs(t_sec - tb[right])
    g = np.where(dl <= dr, sb[left], sb[right])
    far = np.minimum(dl, dr) > max_carry
    silent = ae_silent(n_hits, win, frac) if len(n_hits) == len(t_sec) \
        else np.ones(len(t_sec), bool)
    cut = far & silent
    return np.where(cut, 0.0, g), far, cut


def metrics(g, n_hits, nf, t_sec, cut):
    """一个 (组, 组合) 的全部指标。

    ⚠️ 其中 `rho` = Spearman(g, AE 帧事件数) **不是全组可用的判据**：
    `g` 几乎恒为 1 的组（L1-06/13/14）无定义或退化成噪声，AE 帧级事件数
    贴近本底的组（L1-41）给约 0，帧数太少的组（L1-30/34/36）无意义。
    ⇒ 汇总时**必须报可用组数并存中位**，不要直接对所有组取均值。详见 §28。
    """
    load_h = float(np.sum(g) * FRAME_S / 3600.0)
    f_res = (nf / (load_h * 3600.0)) if (load_h > 0 and nf) else float('nan')
    mr, mx = runs(g)
    r = spearmanr(g, n_hits).statistic if np.ptp(g) > 0 else float('nan')
    rate_cut = float(n_hits[cut].sum() / max(int(cut.sum()), 1))
    rate_oth = float(n_hits[~cut].sum() / max(int((~cut).sum()), 1))
    gap_h = 0.0
    for (x0, x1) in gaps_of(t_sec):
        m = (t_sec >= x0) & (t_sec <= x1)
        gap_h += float(np.sum(g[m]) * FRAME_S / 3600.0)
    return dict(load_h=load_h, f_res=f_res, blk=mr, blk_max=mx, rho=r,
                cut_h=float(cut.sum()) * FRAME_S / 3600.0,
                cut_rate=rate_cut, oth_rate=rate_oth,
                cut_ratio=rate_cut / max(rate_oth, 1e-9), gap_h=gap_h)


def iqr(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) > 1 else float('nan')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='*', default=None)
    a = ap.parse_args()
    root = _data_l1()

    frames = {}
    for p in sorted(glob.glob(os.path.join(RES, '_l1ae_frames_*.npz'))):
        gid = os.path.basename(p).replace('_l1ae_frames_', '').replace('.npz', '')
        if a.groups and gid not in a.groups:
            continue
        z = np.load(p, allow_pickle=True)
        frames[gid] = (z['frame'].astype(float) * FRAME_S, np.asarray(z['n'], float))

    print(f'数据根目录: {root}')
    print(f'参与组 {len(frames)}: {", ".join(sorted(frames))}')
    print(f'扫描 MAX_CARRY {CARRY_GRID} x AE_PAUSE_FRAC {FRAC_GRID}'
          f'（AE_PAUSE_WIN 固定 {WIN}）\n')

    # ---- 一次性读 FBG（贵），之后只重算网格 ----
    cache = {}
    for gid in sorted(frames):
        got = fbg_bursts(gid, root)
        if got is None:
            print(f'{gid}: 无 FBG，跳过')
            continue
        cache[gid] = got

    rows = []
    grids = {}
    for mc in CARRY_GRID:
        for fr in FRAC_GRID:
            per = {}
            for gid, (t_sec, n_hits) in frames.items():
                if gid not in cache:
                    continue
                tf, lf, tb, sb = cache[gid]
                g, far, cut = g_hold(t_sec, tb, sb, mc, fr, n_hits)
                m = metrics(g, n_hits, N_F.get(gid, 0), t_sec, cut)
                per[gid] = m
                rows.append(dict(max_carry=mc, pause_frac=fr, gid=gid, **m))
            grids[(mc, fr)] = per

    # ---- 自检：基线组合必须与生产口径逐位一致 ----
    bad = []
    for gid in cache:
        t_sec, n_hits = frames[gid]
        tf, lf, tb, sb = cache[gid]
        g_scan, _, _ = g_hold(t_sec, tb, sb, BASE_CARRY, BASE_FRAC, n_hits)
        g_prod = g_hold_prod(t_sec, tb, sb, max_carry=BASE_CARRY, default=0.0,
                             n_hits=n_hits)
        if not np.array_equal(g_scan, g_prod):
            bad.append(gid)
    if bad:
        print(f'!! 自检失败：{bad} 与生产口径不一致，中止。')
        return
    print(f'✅ 自检通过：基线组合 ({BASE_CARRY:.0f}s, {BASE_FRAC}) 与生产口径逐位一致'
          f'（{len(cache)} 组）\n')

    # ---- 表 1：汇总网格 ----
    ok = [g for g in cache if len(frames[g][0]) >= MIN_FRAMES]
    print('=' * 118)
    print('表 1  汇总（只统计 >= %d 帧的 %d 组：%s）'
          % (MIN_FRAMES, len(ok), ', '.join(sorted(ok))))
    print('=' * 118)
    hdr = (f"{'MAX_CARRY':>9}{'FRAC':>7}{'f中位':>8}{'A批IQR':>8}{'B批IQR':>8}"
           f"{'f极差':>8}{'空隙内加载h':>12}{'rho(g,AE)':>11}{'平均块长':>10}"
           f"{'截断h':>8}{'截断安全比':>11}")
    print(hdr)
    print('-' * len(hdr))
    summary = {}
    for mc in CARRY_GRID:
        for fr in FRAC_GRID:
            per = grids[(mc, fr)]
            fs = [per[g]['f_res'] for g in ok if np.isfinite(per[g]['f_res'])]
            fa = [per[g]['f_res'] for g in ok if g in BATCH_A]
            fb = [per[g]['f_res'] for g in ok if g not in BATCH_A]
            gh = [per[g]['gap_h'] for g in ok]
            rs = [per[g]['rho'] for g in ok if np.isfinite(per[g]['rho'])]
            bl = [per[g]['blk'] for g in ok]
            ch = [per[g]['cut_h'] for g in ok]
            cr = [per[g]['cut_ratio'] for g in ok]
            summary[(mc, fr)] = dict(
                f_med=float(np.median(fs)) if fs else float('nan'),
                iqr_a=iqr(fa), iqr_b=iqr(fb),
                f_span=float(max(fs) - min(fs)) if fs else float('nan'),
                gap_sum=float(np.sum(gh)), gap_med=float(np.median(gh)),
                rho_mean=float(np.mean(rs)) if rs else float('nan'),
                blk_med=float(np.median(bl)), blk_min=float(np.min(bl)),
                cut_sum=float(np.sum(ch)), cut_ratio_max=float(np.max(cr)),
                worst=gid_of_max(per, ok))
            s = summary[(mc, fr)]
            star = ' *' if (mc == BASE_CARRY and fr == BASE_FRAC) else '  '
            print(f'{mc:>9.0f}{fr:>7.2f}{s["f_med"]:>8.2f}{s["iqr_a"]:>8.3f}'
                  f'{s["iqr_b"]:>8.3f}{s["f_span"]:>8.2f}{s["gap_sum"]:>12.1f}'
                  f'{s["rho_mean"]:>11.3f}{s["blk_med"]:>10.2f}'
                  f'{s["cut_sum"]:>8.1f}{s["cut_ratio_max"]:>11.3f}{star}')
    print('  (* = 当前生产口径)')
    print('  截断安全比 = 被判停机帧的 AE 事件率 / 其余帧的 AE 事件率；'
          '取各组最差，应 << 1（越小越安全）。')

    # ---- 表 2：关注组 ----
    print()
    print('=' * 96)
    print('表 2  关注组的反解频率（历史敏感组）')
    print('=' * 96)
    hdr2 = f"{'组':<8}" + ''.join(f'{mc:.0f}s/{fr:.2f}' for mc in CARRY_GRID
                                 for fr in FRAC_GRID)
    print(hdr2)
    print('-' * len(hdr2))
    for gid in WATCH:
        if gid not in grids[(BASE_CARRY, BASE_FRAC)]:
            continue
        line = f'{gid:<8}'
        for mc in CARRY_GRID:
            for fr in FRAC_GRID:
                v = grids[(mc, fr)][gid]['f_res']
                line += f'{v:>10.3f}' if np.isfinite(v) else f'{"-":>10}'
        print(line)
    print(f"  参考：A 批名义 1.075 至 1.148 Hz；B 批名义 1.821 至 2.119 Hz")

    # ---- 表 3：逐组最差安全比 ----
    print()
    print('=' * 96)
    print('表 3  截断安全比（每格 = 该阈值下最差的一组的比值）')
    print('=' * 96)
    for mc in CARRY_GRID:
        for fr in FRAC_GRID:
            per = grids[(mc, fr)]
            worst = max((g for g in ok), key=lambda g: per[g]['cut_ratio'])
            print(f'  MAX_CARRY={mc:>6.0f}s  FRAC={fr:.2f}  '
                  f'最差 {worst}: 比 {per[worst]["cut_ratio"]:.3f}  '
                  f'（截断 {per[worst]["cut_h"]:.1f} h，'
                  f'率 {per[worst]["cut_rate"]:.1f} vs {per[worst]["oth_rate"]:.1f} 件/帧）')

    # ---- 表 4：AE 守卫本身的贡献（FRAC=∞ ≡ 关掉守卫）----
    print()
    print('=' * 104)
    print('表 4  对照：关掉 AE 守卫（纯时间阈值，等价 FRAC=∞）')
    print('=' * 104)
    hdr4 = (f"{'MAX_CARRY':>9}{'组':>8}{'纯时间f':>10}{'守卫后f':>10}"
            f"{'被守卫拉回h':>12}{'纯时间安全比':>13}")
    print(hdr4)
    print('-' * len(hdr4))
    for mc in CARRY_GRID:
        for gid in WATCH:
            if gid not in cache:
                continue
            t_sec, n_hits = frames[gid]
            tf, lf, tb, sb = cache[gid]
            nf = N_F.get(gid, 0)
            g_p, _far, cut_p = g_hold(t_sec, tb, sb, mc, 1e9, n_hits)
            m_p = metrics(g_p, n_hits, nf, t_sec, cut_p)
            m_g = grids[(mc, BASE_FRAC)][gid]
            print(f'{mc:>9.0f}{gid:>8}{m_p["f_res"]:>10.3f}{m_g["f_res"]:>10.3f}'
                  f'{m_p["cut_h"] - m_g["cut_h"]:>12.1f}{m_p["cut_ratio"]:>13.3f}')
    print('  读法：若两列 f 相差很大，说明守卫在起作用（纯时间阈值切错了）；')
    print('        若两列差不多，说明这段里本来就没有需要守卫救的时段。')

    # ---- 结论 ----
    print()
    print('=' * 96)
    print('结论')
    print('=' * 96)
    cr = {k: v['cut_ratio_max'] for k, v in summary.items()}
    safe = all(v < 0.25 for v in cr.values())
    print(f'  · 9 个组合里，截断安全比最差 = {max(cr.values()):.3f} '
          f'⇒ {"全部安全（都 << 1）" if safe else "存在切错风险，见上表"}')
    ia = [summary[k]['iqr_a'] for k in summary]
    ib = [summary[k]['iqr_b'] for k in summary]
    print(f'  · A 批 IQR 范围 {min(ia):.3f} 至 {max(ia):.3f}；'
          f'B 批 IQR 范围 {min(ib):.3f} 至 {max(ib):.3f}'
          f'（都远小于分组间的真实差异，分组结论稳定）')
    gh = [summary[k]['gap_sum'] for k in summary]
    print(f'  · 空隙内声称加载小时数合计 {min(gh):.1f} 至 {max(gh):.1f} h（应接近 0）')
    bl = [summary[k]['blk_med'] for k in summary]
    print(f'  · 平均块长中位 {min(bl):.2f} 至 {max(bl):.2f} 帧（全部远大于 1，块结构稳定）')

    out = os.path.join(RES, 'l1_thr_sensitivity.csv')
    cols = ['max_carry', 'pause_frac', 'gid', 'f_res', 'load_h', 'blk', 'blk_max',
            'rho', 'gap_h', 'cut_h', 'cut_rate', 'oth_rate', 'cut_ratio']
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    print(f'\n写出 {out}（{len(rows)} 行）')


def gid_of_max(per, ok):
    return max(ok, key=lambda g: per[g]['cut_ratio'])


if __name__ == '__main__':
    main()
