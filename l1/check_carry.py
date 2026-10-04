# -*- coding: utf-8 -*-
"""审查 MAX_CARRY 截断是否切错了：把「被截断判停机」的时段与 AE 活动对照。

背景
----
`ae_cycle` 的 holdgap 口径规定：离最近 FBG 突发超过 `MAX_CARRY`（20 min）的
AE 帧判为停机。这条规则的物理依据是「FBG 停录 = 试验暂停」，而它成立与否
**可以独立检验**：若被判停机的时段里 AE 事件依然密集，说明试验其实在跑，
是截断切错了，而不是试验暂停。

本脚本对每组做三件事：
  1. 列出「离最近突发 > MAX_CARRY」的时段（连续段），给出总时长；
  2. 统计这些时段内的 AE 事件数与事件率；
  3. 与「正常加载帧」的事件率对比 —— 若二者同量级，则截断判错。

用法
----
    python l1/check_carry.py                  # 全部组
    python l1/check_carry.py L1-13 L1-31      # 指定组
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_cycle import (scan_fbg, P2P_LOAD, FRAME_S, BURST_GAP, MAX_CARRY,  # noqa: E402
                      _data_l1, _bursts)


def audit(gid, root, max_carry=MAX_CARRY):
    p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        print(f'{gid}: 无 AE 帧表')
        return None
    z = np.load(p, allow_pickle=True)
    t_sec = z['frame'].astype(float) * FRAME_S
    n = np.asarray(z['n'], dtype=float)

    pts = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        pts.extend(scan_fbg(f))
    if not pts:
        print(f'{gid}: 无 FBG 窗')
        return None
    pts.sort(key=lambda x: x[0])
    tf = np.array([x[0] for x in pts])
    lf = np.array([1.0 if x[1] > P2P_LOAD else 0.0 for x in pts])
    tb, sb = _bursts(tf, lf)

    idx = np.searchsorted(tb, t_sec)
    left = np.clip(idx - 1, 0, len(tb) - 1)
    right = np.clip(idx, 0, len(tb) - 1)
    d = np.minimum(np.abs(t_sec - tb[left]), np.abs(t_sec - tb[right]))
    cut = d > max_carry                       # 被截断判停机的帧

    # 连续段
    segs = []
    if cut.any():
        chg = np.flatnonzero(np.diff(cut.astype(int)) != 0) + 1
        bounds = np.concatenate([[0], chg, [len(cut)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if cut[a]:
                segs.append((a, b))

    span = (t_sec[-1] - t_sec[0]) / 3600.0
    tb_med = float(np.median(np.diff(tb))) if len(tb) > 1 else float('nan')
    print(f'=== {gid} ===')
    print(f'  帧 {len(t_sec)}  跨度 {span:.1f} h  FBG 突发 {len(tb)}  '
          f'突发间隔中位 {tb_med:.0f} s  MAX_CARRY={max_carry:.0f} s')
    if not cut.any():
        print('  无被截断的帧')
        return None
    print(f'  被截断判停机的帧 {int(cut.sum())} ({100.0*cut.sum()/len(t_sec):.1f}%), '
          f'共 {len(segs)} 段, 合计 {cut.sum()*FRAME_S/3600:.1f} h')
    rate_cut = n[cut].sum() / max(cut.sum(), 1)
    rate_load = n[~cut].sum() / max((~cut).sum(), 1)
    print(f'  AE 事件率: 被截断帧 {rate_cut:9.1f} 件/帧   其余帧 {rate_load:9.1f} 件/帧'
          f'   比值 {rate_cut/max(rate_load, 1e-9):.3f}')
    print('  最大 5 段（时长 h / 段内 AE 事件 / 事件率 件/帧）：')
    segs.sort(key=lambda s: s[0] - s[1])
    for a, b in segs[:5]:
        m = slice(a, b)
        print(f'    {t_sec[a]:.0f}s 起  {(b-a)*FRAME_S/3600:7.1f} h  '
              f'AE {n[m].sum():9.0f} 件  率 {n[m].mean():9.1f} 件/帧')
    print('  判读: 若「被截断帧」的事件率与其余帧同量级 -> 截断切错了(试验在跑);'
          ' 若近零 -> 确实是停机。')
    print()
    return rate_cut, rate_load


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap_groups = sys.argv[1:]
    root = _data_l1()
    print(f'数据根目录: {root}\n')
    gids = ap_groups or [
        os.path.basename(p).replace('_l1ae_frames_', '').replace('.npz', '')
        for p in sorted(glob.glob(os.path.join(RES, '_l1ae_frames_*.npz')))]
    for gid in gids:
        audit(gid, root)


if __name__ == '__main__':
    main()
