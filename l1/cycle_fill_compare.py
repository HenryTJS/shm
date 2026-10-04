# -*- coding: utf-8 -*-
"""比较「加载状态外推」的几种口径，用物理性判据选优胜者。

背景
----
`ae_cycle._ratio_on_grid()` 要把 FBG 的「加载 / 停机」标签铺到 600 s 的 AE 帧网格上。
每个 600 s 帧在 ±300 s 内应罩到约 1.4 个 FBG 文件 = 约 2.9 个窗，但原实现的采信门槛
是 **>= 3 个窗**，于是约一半的帧拿不到实测值，被赋上**全局平均加载占比**：

    if m.sum() >= 3:  g[i] = lf[m].mean()
    g[~np.isfinite(g)] = float(lf.mean())        # 其余帧 → 一个常数

后果：`g_load` 退化成少数几个离散值，且「被赋常数」的帧占 50% 至 73%，
循环轴在这些区段变成**线性时间斜坡** —— 而真实加载是「几小时连续加载 + 停机」
的**块状**结构，线性斜坡在物理上说不通。

候选口径
--------
  mean3     现有实现（>=3 个窗取局部均值，否则全局均值）
  mean1     门槛降到 >=1（能测就测，仍以全局均值兜底）
  hold      最近一次 FBG 突发的状态（零阶保持 / Voronoi，无兜底常数）
  holdcut   同 hold，但离最近突发超过 MAX_CARRY 的帧一律判 0（纯时间阈值）
  holdgap   **采用**：同 holdcut，但只对「AE 也静默」的帧判 0
            （见 l1/check_carry.py：纯时间阈值会在 L1-13 / L1-41 上切错）
  aeact     FBG 没罩到的帧改用 **AE 该帧有无事件** 判定（跨模态兜底）

判据（越靠后越硬）
------------------
  1. 反解频率的**组内一致性**：同批次同设备的组，反解 f 应聚在一起。
  2. 取值结构：离散取值数、众数占比、**加载块的平均连续长度**（块状才物理）。
  3. 跨模态一致性：与 AE 帧事件的秩相关（加载时事件多）。
  4. **长空隙惩罚**：AE 已证实长空隙内无活动（L1-41 空隙内 AE 事件为 0），
     任何方法都不该在那里标「加载」。此列应接近 0。

用法
----
    python l1/cycle_fill_compare.py
    python l1/cycle_fill_compare.py --groups L1-31 L1-29 --variants mean3 hold
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_cycle import (scan_fbg, P2P_LOAD, N_F, FRAME_S, F_NOMINAL,  # noqa: E402
                      _data_l1, AE_PAUSE_WIN, AE_PAUSE_FRAC)

BURST_GAP = 60.0        # 相邻窗前隔超过它就分成两个「突发」
MAX_CARRY = 1200.0      # holdgap：离最近突发超过它就判为停机
GAP_H = 2.0             # 认定为「FBG 长空隙」的门槛（小时）
BIN = 0.5               # g > BIN 记为「加载」
MIN_FRAMES = 300        # 参与批次一致性统计的最少帧数


def fbg_bursts(gid, root):
    """把 FBG 窗聚成「突发」（同一文件内的连续窗），返回 (突发时刻, 突发加载占比)。"""
    pts = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        pts.extend(scan_fbg(f))
    pts.sort(key=lambda x: x[0])
    if not pts:
        return None
    tf = np.array([p[0] for p in pts])
    lf = np.array([1.0 if p[1] > P2P_LOAD else 0.0 for p in pts])
    tb, sb = [], []
    i = 0
    while i < len(tf):
        j = i
        while j + 1 < len(tf) and tf[j + 1] - tf[j] <= BURST_GAP:
            j += 1
        tb.append(tf[i:j + 1].mean())
        sb.append(lf[i:j + 1].mean())
        i = j + 1
    return tf, lf, np.array(tb), np.array(sb)


def g_mean(t_sec, tf, lf, minw):
    g = np.full(len(t_sec), np.nan)
    for i, t in enumerate(t_sec):
        m = (tf >= t - FRAME_S / 2) & (tf < t + FRAME_S / 2)
        if m.sum() >= minw:
            g[i] = lf[m].mean()
    g[~np.isfinite(g)] = float(lf.mean())
    return g


def _ae_silent(n_hits, win=AE_PAUSE_WIN, frac=AE_PAUSE_FRAC):
    """局部 AE 事件率是否「基本静默」—— 与 ae_cycle._ae_silent 同口径。"""
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


def g_hold(t_sec, tb, sb, max_carry=None, default=np.nan, n_hits=None):
    """最近突发的状态（零阶保持）。

    max_carry 不为 None 时，超过该距离的帧按 default 处理；
    若同时给了 n_hits，则仅对**AE 静默**的长停录帧用 default
    （AE 仍在活动说明试验在跑，FBG 只是没录上）。
    """
    idx = np.searchsorted(tb, t_sec)
    left = np.clip(idx - 1, 0, len(tb) - 1)
    right = np.clip(idx, 0, len(tb) - 1)
    dl = np.abs(t_sec - tb[left])
    dr = np.abs(t_sec - tb[right])
    use_left = dl <= dr
    g = np.where(use_left, sb[left], sb[right])
    if max_carry is not None:
        far = np.minimum(dl, dr) > max_carry
        if n_hits is not None and len(n_hits) == len(t_sec):
            g = np.where(far & _ae_silent(n_hits), default, g)
        else:
            g = np.where(far, default, g)
    return g


def g_aeact(t_sec, tf, lf, n_hits):
    """有 FBG 罩住的帧取局部均值；没罩住的用 AE 有无事件判定。"""
    g = np.full(len(t_sec), np.nan)
    for i, t in enumerate(t_sec):
        m = (tf >= t - FRAME_S / 2) & (tf < t + FRAME_S / 2)
        if m.sum() >= 1:
            g[i] = lf[m].mean()
    miss = ~np.isfinite(g)
    g[miss] = (n_hits[miss] > 0).astype(float)
    return g


def runs(g):
    """二值化后最长/平均连续段长度（帧数）。"""
    b = g > BIN
    if len(b) == 0:
        return 0.0, 0
    chg = np.flatnonzero(np.diff(b.astype(int)) != 0) + 1
    seg = np.diff(np.concatenate([[0], chg, [len(b)]]))
    lab = b[np.concatenate([[0], chg])]
    L = seg[lab]
    return (float(L.mean()) if len(L) else 0.0), int(L.max() if len(L) else 0)


def gaps_of(t_sec):
    """时间轴上的长空隙区间（> GAP_H 小时）。"""
    d = np.diff(t_sec)
    idx = np.where(d > GAP_H * 3600)[0]
    return [(t_sec[i], t_sec[i + 1]) for i in idx]


def run_variant(name, t_sec, tf, lf, tb, sb, n_hits):
    if name == 'mean3':
        return g_mean(t_sec, tf, lf, 3)
    if name == 'mean1':
        return g_mean(t_sec, tf, lf, 1)
    if name == 'hold':
        return g_hold(t_sec, tb, sb)
    if name == 'holdcut':
        # 纯时间阈值（不看 AE）—— 已被 check_carry.py 证实在 L1-13/L1-41 上切错
        return g_hold(t_sec, tb, sb, max_carry=MAX_CARRY, default=0.0)
    if name == 'holdgap':
        # 采用口径：时间阈值 + AE 静默守卫
        return g_hold(t_sec, tb, sb, max_carry=MAX_CARRY, default=0.0,
                      n_hits=n_hits)
    if name == 'aeact':
        return g_aeact(t_sec, tf, lf, n_hits)
    raise ValueError(name)


VARIANTS = ['mean3', 'mean1', 'hold', 'holdcut', 'holdgap', 'aeact']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='*', default=None)
    ap.add_argument('--variants', nargs='*', default=VARIANTS)
    ap.add_argument('--root', default=None)
    a = ap.parse_args()
    root = a.root or _data_l1()

    frames = {}
    for p in sorted(glob.glob(os.path.join(RES, '_l1ae_frames_*.npz'))):
        gid = os.path.basename(p).replace('_l1ae_frames_', '').replace('.npz', '')
        if a.groups and gid not in a.groups:
            continue
        z = np.load(p, allow_pickle=True)
        frames[gid] = (z['frame'].astype(float) * FRAME_S, np.asarray(z['n'], float))
    gids = list(frames)

    print(f'数据根目录: {root}')
    print(f'参与组: {", ".join(gids)}')
    print(f'口径: {", ".join(a.variants)}')
    print(f'（名义载荷频率 {F_NOMINAL} Hz；>= {MIN_FRAMES} 帧的组才计入批次一致性）\n')

    detail = {}
    cache = {}
    for gid in gids:
        t_sec, n_hits = frames[gid]
        got = fbg_bursts(gid, root)
        if got is None:
            print(f'{gid}: 无 FBG，跳过')
            continue
        tf, lf, tb, sb = got
        cache[gid] = got
        nf = N_F.get(gid, 0)
        gp = float(lf.mean())
        gaps = gaps_of(t_sec)
        nfrm = len(t_sec)
        detail[gid] = {'n': nfrm, 'gp': gp, 'gaps': {}, 'nf': nf,
                       'g': {}, 'gh': {}}

        print(f'=== {gid}  帧 {nfrm}  FBG 突發 {len(tb)}  窗 {len(tf)}  '
              f'实测加载占比 gp={gp:.3f}  n_f={nf}  长空隙 {len(gaps)} 个 ===')
        hdr = (f"  {'口径':<9}{'加载h':>9}{'反解f':>8}{'离散值':>8}{'众数占比':>9}"
               f"{'平均块长':>9}{'最长块':>8}{'空隙内加载h':>12}{'rho(g,AE)':>11}")
        print(hdr)
        for v in a.variants:
            g = run_variant(v, t_sec, tf, lf, tb, sb, n_hits)
            load_h = float(np.sum(g) * FRAME_S / 3600.0)
            f_res = nf / (load_h * 3600.0) if load_h > 0 and nf else np.nan
            uq = len(np.unique(np.round(g, 4)))
            vals, cnts = np.unique(np.round(g, 4), return_counts=True)
            mode_share = 100.0 * cnts.max() / len(g)
            mr, mx = runs(g)
            gap_h = 0.0
            for (x0, x1) in gaps:
                m = (t_sec >= x0) & (t_sec <= x1)
                # 用 sum(g) 而非二值化：全局均值补值的常数（如 0.474）低于
                # BIN 时会「逃过惩罚」，但它在物理上依然是在凭空声称加载
                gap_h += float(np.sum(g[m]) * FRAME_S / 3600.0)
            rr = spearmanr(g, n_hits).statistic if np.ptp(g) > 0 else np.nan
            detail[gid]['g'][v] = g
            detail[gid]['gh'][v] = gap_h
            print(f'  {v:<11}{load_h:>9.1f}{f_res:>8.2f}{uq:>8}{mode_share:>8.1f}%'
                  f'{mr:>9.1f}{mx:>8}{gap_h:>12.1f}{rr:>11.3f}')
        print()

    # 汇总
    print('=' * 100)
    print('汇总（只统计 >= %d 帧的组）' % MIN_FRAMES)
    print('=' * 100)
    hdr = (f"{'口径':<9}{'组数':>5}{'反解f中位':>10}{'f组间极差':>10}{'f组间IQR':>10}"
           f"{'空隙内加载h合计':>16}{'空隙内加载h中位':>16}{'平均rho(g,AE)':>14}")
    print(hdr)
    print('-' * len(hdr))
    for v in a.variants:
        fs, ghs, rs = [], [], []
        for gid, d in detail.items():
            if d['n'] < MIN_FRAMES or v not in d['g']:
                continue
            g = d['g'][v]
            load_h = float(np.sum(g) * FRAME_S / 3600.0)
            if load_h > 0 and d['nf']:
                fs.append(d['nf'] / (load_h * 3600.0))
            ghs.append(d['gh'][v])
            r = spearmanr(g, frames[gid][1]).statistic if np.ptp(g) > 0 else np.nan
            if np.isfinite(r):
                rs.append(r)
        if not fs:
            continue
        fs = np.array(fs)
        ghs = np.array(ghs)
        print(f'{v:<11}{len(fs):>3}{np.median(fs):>10.2f}'
              f'{fs.max() - fs.min():>10.2f}'
              f'{np.percentile(fs, 75) - np.percentile(fs, 25):>10.2f}'
              f'{ghs.sum():>16.1f}{np.median(ghs):>16.1f}'
              f'{np.mean(rs) if rs else np.nan:>14.3f}')
    print('\n判据解读：')
    print('  · 「反解f中位」应接近名义 2 Hz，「f组间极差 / IQR」越小越好。')
    print('  · 「空隙内加载h」应接近 0 —— AE 已证实长空隙内无活动（最硬判据）。')
    print('    注意：该列用 sum(g) 累加而非二值化，因为「凭空声称 47% 时间在加载」')
    print('    与「声称 100%」同样不物理，只是数值大小不同。')
    print('  · 「rho(g,AE)」越高越好 —— 加载帧应同时是 AE 事件多的帧（跨模态）。')


if __name__ == '__main__':
    main()
