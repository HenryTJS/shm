# -*- coding: utf-8 -*-
"""跨模态独立验证 —— 用 FBG 的**应变空间分布**独立核对 AE 的损伤指标。

为什么这样设计
--------------
老组 C 批做过「AE × DFOS 交叉验证」（8/9 正相关），但那批有 DFOS。新组没有 DFOS，
只有 FBG 的 10 个应变通道（R1 至 R5 在 A 侧、L1 至 L5 在 B 侧，沿加筋条排列）。

关键技巧：**用「形状」而不是「幅值」**。
  同一次加载窗内，10 个通道的应变峰峰值受载荷级支配（级间差 12 至 15%，还会波动
  30%）⇒ 绝对值不可比；但把它**按窗归一化**成一个 10 维分布向量后，
  载荷级的整体缩放被消掉，剩下的就是**应变剖面的形状**。
  脱粘／分层扩展会让载荷路径改变 ⇒ 剖面形状漂移，这正是老组 B 批用 DFOS
  验证过的物理量。

指标（每加载窗一个值）
  · `shape_l1`：与**首窗**归一化剖面的 L1 距离（初版口径）
  · `shape_rob10`：与**前 10% 加载窗中位剖面**的 L1 距离（稳健基线）
  · `shape_rob25`：与**前 25% 加载窗中位剖面**的 L1 距离（**推荐**）
  · `centroid`：剖面质心位置（应变重心沿筋向移动）
  · `argmax`：峰值通道序号
  · `asym`：(A 侧和 减 B 侧和) / 总和（左右不对称）

⚠️ **基线选择很关键**（详见 `fbg_variants.py`）：
  单首窗口径有 1 个反例（L1-31 得 -0.56）；换成中位基线后反例归零，
  |rho| 中位由 0.64 升到 0.83（全窗）。推荐口径 `shape_rob25`：
  全窗与暖机后均为 10 正 / 0 负 / 2 弱，|rho| 中位 0.73 / 0.71。

用法
----
    python l1/fbg_profile.py                 # 全部有 FBG 的组
    python l1/fbg_profile.py L1-29 L1-41
输出：results/_l1fbgprof_{gid}.npz + results/l1_fbgprof_rank.csv
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_cycle import scan_fbg, P2P_LOAD                    # noqa: E402

MIN_WIN = 5


def profile_of(buf):
    """一个窗的 10 通道峰峰值向量 → 归一化分布 + 特征。"""
    a = np.asarray(buf, dtype=float)
    with np.errstate(all='ignore'):
        p2p = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
    p2p = np.where(np.isfinite(p2p) & (p2p > 0), p2p, 0.0)
    s = p2p.sum()
    if s <= 0:
        return None
    v = p2p / s
    A = v[:5].sum()
    B = v[5:].sum()
    idx = np.arange(len(v), dtype=float)
    return {'prof': v, 'centroid': float((v * idx).sum()),
            'argmax': int(np.argmax(v)), 'asym': float((A - B))}


def group_profile(gid, root=None):
    """该组的加载窗剖面序列 → 特征数组。"""
    root = root or _data_l1()
    recs = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        # 逐窗：scan_fbg 已按窗给出 (t, p2p)；这里要 10 通道，单独再扫一遍
        recs.extend(_scan_windows(f))
    if len(recs) < MIN_WIN:
        return None
    recs.sort(key=lambda r: r['t'])
    prof = np.asarray([r['prof'] for r in recs])
    base = prof[0]
    l1 = np.abs(prof - base).sum(axis=1)
    # 稳健基线：前 10% / 25% 加载窗的**中位剖面**，抗单窗异常。
    # 实测（见 fbg_variants.py）：单首窗口径有 1 个反例（L1-31 为 -0.56），
    # 换中位基线后反例归零，|rho| 中位由 0.64 升到 0.83（全窗）/ 0.80（暖机后）。
    n = prof.shape[0]
    k10 = max(1, int(round(n * 0.10)))
    k25 = max(1, int(round(n * 0.25)))
    rob10 = np.abs(prof - np.median(prof[:k10], axis=0)).sum(axis=1)
    rob25 = np.abs(prof - np.median(prof[:k25], axis=0)).sum(axis=1)
    return {'t': np.asarray([r['t'] for r in recs]),
            'p2p_med': np.asarray([r['p2p'] for r in recs]),
            'p2p_ch': np.asarray([r['p2p_ch'] for r in recs]),
            'shape_l1': l1,
            'shape_rob10': rob10,
            'shape_rob25': rob25,
            'centroid': np.asarray([r['centroid'] for r in recs]),
            'argmax': np.asarray([r['argmax'] for r in recs]),
            'asym': np.asarray([r['asym'] for r in recs]),
            'prof': prof}


def _scan_windows(path, window_lines=140):
    """与 ae_cycle.scan_fbg 同样的分窗方式，但保留 10 通道向量。

    表头按**内容**识别（首个能解析出时间戳且第 3 到 12 列均为浮点的行）。
    注意：标定式行有 17 列，**不能**用「字段数 >= 12」当数据判据。
    实测结果与旧版硬编码 `i < 60` 逐位相同（数据确实从第 60 行起）。
    """
    from ae_cycle import _file_dt, _parse_ts
    ref = _file_dt(path)
    out, buf, t0 = [], [], None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 12:
                continue
            try:
                vals = [float(x) for x in p[2:12]]
            except ValueError:
                continue
            ts = _parse_ts(p[0] + ' ' + p[1], ref)
            if ts is None:               # 表头残留行
                continue
            if t0 is None:
                t0 = ts
            buf.append(vals)
            if len(buf) >= window_lines:
                r = _win(buf, t0)
                if r:
                    out.append(r)
                buf, t0 = [], None
    if buf:
        r = _win(buf, t0)
        if r:
            out.append(r)
    return out


def _win(buf, t0):
    if t0 is None or not buf:
        return None
    d = profile_of(buf)
    if d is None:
        return None
    a = np.asarray(buf, dtype=float)
    with np.errstate(all='ignore'):
        p2 = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
    # 逐通道峰峰值要**在过滤前**留副本：过滤后长度可能不足 10，
    # 而看板的 FBG 面板要求固定 10 个通道。非有限值补 0。
    p2_ch = np.where(np.isfinite(p2), p2, 0.0)
    p2 = p2[np.isfinite(p2)]
    if len(p2) == 0:
        return None
    med = float(np.median(p2))
    if med <= P2P_LOAD:                # 只留加载窗
        return None
    r = {'t': t0, 'p2p': med, 'p2p_ch': p2_ch}
    r.update(d)
    return r


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = _data_l1()
    gids = argv or sorted(d for d in os.listdir(root)
                          if d.startswith('L1-')
                          and glob.glob(os.path.join(root, d, 'FBG', '*.txt')))
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        spearmanr = None
    print('%-7s %7s %10s %10s %10s %10s' % (
        '组', '加载窗', 'shapeL1中位', '末段L1', 'rho(L1,窗序)', '质心漂移'))
    rows = []
    for gid in gids:
        d = group_profile(gid, root)
        if d is None:
            print('%-7s %7s' % (gid, '窗数不足'))
            continue
        np.savez_compressed(os.path.join(RES, '_l1fbgprof_%s.npz' % gid),
                            t=d['t'], p2p_med=d['p2p_med'], p2p_ch=d['p2p_ch'],
                            shape_l1=d['shape_l1'],
                            shape_rob10=d['shape_rob10'],
                            shape_rob25=d['shape_rob25'],
                            centroid=d['centroid'], argmax=d['argmax'],
                            asym=d['asym'], prof=d['prof'])
        n = len(d['t'])
        idx = np.arange(n, dtype=float)
        rho = np.nan
        if spearmanr is not None and n >= 8:
            rho = float(spearmanr(idx, d['shape_l1'])[0])
        k = max(int(n * 0.3), 1)
        rows.append((gid, n, float(np.median(d['shape_l1'])),
                     float(np.median(d['shape_l1'][-k:])), rho,
                     float(d['centroid'][-k:].mean() - d['centroid'][:k].mean())))
        print('%-7s %7d %10.3f %10.3f %10.3f %10.3f'
              % (gid, n, np.median(d['shape_l1']), np.median(d['shape_l1'][-k:]),
                 rho, rows[-1][5]))
    if rows:
        import csv
        out = os.path.join(RES, 'l1_fbgprof_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '加载窗', 'shapeL1中位', '末段L1', 'rho_shapeL1_vs_窗序',
                        '质心漂移'])
            w.writerows(rows)
        rr = [r[4] for r in rows if np.isfinite(r[4])]
        if rr:
            print('\nrho(形状漂移, 窗序) 方向：正 %d / 负 %d / 弱 %d（共 %d 组），|rho| 中位 %.2f'
                  % (sum(1 for v in rr if v > 0.3), sum(1 for v in rr if v < -0.3),
                     sum(1 for v in rr if abs(v) <= 0.3), len(rr),
                     float(np.median(np.abs(rr)))))
        print('已写出 ->', out)
    return 0


def _data_l1():
    here = HERE
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                        # noqa: BLE001
        pass
    return here


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
