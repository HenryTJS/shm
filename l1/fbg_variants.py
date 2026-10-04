# -*- coding: utf-8 -*-
"""FBG 剖面指标升级版 —— 多口径横向对比 + 单组诊断。

背景
----
初版 `fbg_profile.py` 只给一个形状距离：

    shape_l1 = |prof_k - prof_0|_1          （prof 为 10 通道按窗归一化剖面）

跨 13 组得到 rho(shape_l1, 窗序) 正 7 / 负 1 / 弱 5，|rho| 中位 0.59（全场最高），
但是**基线只取首窗**，且只有一种距离度量。本轮做两件事（用户选择 D + B）：

  D) 在同一份剖面上横向对比 10 类口径，区分「全窗」与「暖机后」两个评价区间；
  B) 诊断唯一明显反例 **L1-31**（shape_l1 rho = -0.559）。

结论（2026-10-02 实测）
----------------------

   口径           全窗 P/N/W   |rho|中位    暖机后 P/N/W   |rho|中位
   l1_first         7/1/4       0.64        8/1/3        0.58   ← 初版
   l1_ref10         8/0/4       0.83        8/0/4        0.77
   l1_ref25        10/0/2       0.73       11/0/1        0.80   ← 推荐
   cos_first        7/1/4       0.67        7/1/4        0.68
   js_first         8/1/3       0.52        8/2/2        0.48
   l1_last         0/11/1       0.87         ——          ——     ← 伪
   l1_mean          1/4/7       0.16         ——          ——     ← 伪
   path            12/0/0       1.00         ——          ——     ← 构造性单调
   cent_d           3/3/6       0.38         ——          ——
   asym_d           3/4/5       0.34         ——          ——

⇒ **把基线从「单个首窗」换成「前 25% 加载窗的中位剖面」，反例归零，
   |rho| 中位 0.58 升到 0.80，并同时修好 L1-27（-0.23 → +0.72）
   与 L1-31（-0.82 → +0.70）。**

⚠️ 三条必须写进交付的边界
  1. `path` 的 rho=1.00 是**累积路径长度只增不减**的数学必然，不是独立证据。
  2. `l1_last` / `l1_mean` 的强负相关同理（离终值/均值越近，绝对值必然越小）。
  3. Spearman 只看秩 ⇒ **单调变换不改变秩**，换轴后排名不变是数学必然，
     不构成稳健性证据。

混杂检验（重要）：`shape_l1` 会不会只是在跟踪载荷水平（p2p）？
  |rho(l1_first, p2p)| 中位 0.262；|rho(l1_ref10, p2p)| 中位 0.326。
  均远低于与寿命的相关 ⇒ **形状漂移不是载荷水平的替身**。

口径清单（每窗一个标量）
  l1_first   |p_k - p_0|_1                      初版口径
  l1_ref10   |p_k - median(p[:10%])|_1          **稳健基线**（首 10% 中位，抗单窗异常）
  l1_ref25   |p_k - median(p[:25%])|_1          更宽的稳健基线
  l1_last    |p_k - p_{N-1}|_1                  非因果（上界参考）
  l1_mean    |p_k - mean(p)|_1                  非因果（上界参考）
  cos_first  1 - cos(p_k, p_0)                  角度距离
  js_first   Jensen-Shannon(p_k, p_0)           分布距离（对单通道尖峰更敏感）
  path       cumsum |p_k - p_{k-1}|_1           累积路径长度（必须单调不减）
  cent_d     |centroid_k - centroid_0|          质心漂移（老组 B 批口径）
  asym_d     |asym_k - asym_0|                  左右不对称漂移

按窗序算 Spearman rho（与初版口径一致，便于直接对比）。
强 = |rho| >= 0.5，中 = 0.3 至 0.5，弱 = < 0.3。

只读 `results/_l1fbgprof_*.npz`，**不需要 E: 盘**。

用法
----
    python l1/fbg_variants.py                 # 全部组，出对比表
    python l1/fbg_variants.py --dump L1-31    # 单组逐窗诊断
    python l1/fbg_variants.py --qc            # 窗质量 / 载荷级匹配诊断（L1-14 查因）
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

VARIANTS = ['l1_first', 'l1_ref10', 'l1_ref25', 'l1_last', 'l1_mean',
            'cos_first', 'js_first', 'path', 'cent_d', 'asym_d']

# 因果可用 = 只用过去数据（基线取自前段），才能真正在线
CAUSAL = {'l1_first', 'l1_ref10', 'l1_ref25', 'path', 'cos_first', 'js_first'}

# 伪口径 = 用终值/均值当基准，或累积量本身只增不减 ⇒ rho 是数学必然，无独立证据价值
DEGENERATE = {'l1_last', 'l1_mean', 'path'}

WARM = 0.25          # 推荐暖机比例（见文件头结论表）

# 窗质量控制门槛：p2p 低于组内中位的这个比例 ⇒ 不是干净的全载循环窗
QC_REL = 0.80
# 载荷级匹配带宽（相对中位 p2p）
LM_TOL = 0.10


def _paths():
    return sorted(glob.glob(os.path.join(RES, '_l1fbgprof_*.npz')))


def _gid(path):
    return os.path.basename(path).replace('_l1fbgprof_', '').replace('.npz', '')


def load_prof(path):
    with np.load(path) as z:
        if 'prof' not in z.files:
            return None
        prof = np.asarray(z['prof'], dtype=float)
        t = np.asarray(z['t'], dtype=float) if 't' in z.files else None
        p2p = np.asarray(z['p2p_med'], dtype=float) if 'p2p_med' in z.files else None
    ok = np.isfinite(prof).all(axis=1)
    prof = prof[ok]
    if t is not None:
        t = t[ok]
    if p2p is not None:
        p2p = p2p[ok]
    return prof, t, p2p


def _l1(a, b):
    return np.abs(a - b).sum(axis=-1)


def variants(prof):
    """剖面矩阵 (N,10) → 各口径标量序列。"""
    n = prof.shape[0]
    p0 = prof[0]
    k10 = max(1, int(round(n * 0.10)))
    k25 = max(1, int(round(n * 0.25)))
    ref10 = np.median(prof[:k10], axis=0)
    ref25 = np.median(prof[:k25], axis=0)
    out = {}
    out['l1_first'] = _l1(prof, p0)
    out['l1_ref10'] = _l1(prof, ref10)
    out['l1_ref25'] = _l1(prof, ref25)
    out['l1_last'] = _l1(prof, prof[-1])
    out['l1_mean'] = _l1(prof, prof.mean(axis=0))
    with np.errstate(all='ignore'):
        out['cos_first'] = 1.0 - (prof @ p0) / (np.linalg.norm(prof, axis=1)
                                                * np.linalg.norm(p0))
        js = np.array([jensenshannon(prof[i], p0, base=2.0) for i in range(n)])
    # 剖面里必然有 0 通道（p2p 为 0 的通道）⇒ JS 可能在「两边同为 0」处产生 nan，
    # 用稀疏度补：nan 视为 0 距离（该窗剖面与基线完全同支撑）
    out['js_first'] = np.nan_to_num(js, nan=0.0, posinf=0.0)
    d = np.abs(np.diff(prof, axis=0)).sum(axis=1)
    out['path'] = np.concatenate([[0.0], np.cumsum(d)])
    idx = np.arange(prof.shape[1], dtype=float)
    cent = (prof * idx).sum(axis=1)
    asym = prof[:, :5].sum(axis=1) - prof[:, 5:].sum(axis=1)
    out['cent_d'] = np.abs(cent - cent[0])
    out['asym_d'] = np.abs(asym - asym[0])
    return out, {'centroid': cent, 'asym': asym}


def rho_of(y):
    n = len(y)
    if n < 8 or not np.isfinite(y).all() or np.ptp(y) == 0:
        return np.nan
    r = spearmanr(y, np.arange(n)).statistic
    return float(r) if np.isfinite(r) else np.nan


def classify(r):
    if not np.isfinite(r):
        return 'n/a'
    a = abs(r)
    if a < 0.3:
        return '弱'
    return '正' if r > 0 else '负'


def _collect():
    """读全部组的剖面矩阵与各口径标量。"""
    data = {}
    for p in _paths():
        prof, t, p2p = load_prof(p)
        if prof is None or prof.shape[0] < 30:
            continue
        v, extra = variants(prof)
        data[_gid(p)] = (prof, t, p2p, v, extra)
    return data


def _table(rows, title):
    print(f'\n=== {title} ===')
    hdr = f"{'口径':<10}{'正':>4}{'负':>4}{'弱':>4}   {'|rho|中位':>9}   {'因果':>5}"
    print(hdr + '\n' + '-' * len(hdr))
    for k in VARIANTS:
        rs = np.array([r[k] for r in rows], dtype=float)
        cl = [classify(x) for x in rs]
        print(f"{k:<12}{cl.count('正'):>3}{cl.count('负'):>4}{cl.count('弱'):>4}"
              f"   {float(np.nanmedian(np.abs(rs))):>9.2f}   "
              f"{'是' if k in CAUSAL else '否':>4}")


def analyze(warm=WARM, save=True):
    """全窗 + 暖机后双口径对比。"""
    data = _collect()
    if not data:
        print('没有可用的 _l1fbgprof_*.npz（需先跑 fbg_profile.py 生成 prof 字段）')
        return None
    gids = list(data)
    allrows, postrows, confrows = [], [], []

    for g in gids:
        prof, t, p2p, v, extra = data[g]
        n = prof.shape[0]
        k0 = max(1, int(round(n * warm)))
        ra = {k: rho_of(v[k]) for k in VARIANTS}
        rp = {k: (rho_of(v[k][k0:]) if n - k0 >= 8 else np.nan) for k in VARIANTS}
        rc = {k: (float(spearmanr(v[k], p2p).statistic)
                  if p2p is not None and np.ptp(p2p) > 0 else np.nan)
              for k in VARIANTS}
        allrows.append(ra)
        postrows.append(rp)
        confrows.append(rc)
        if save:
            np.savez_compressed(
                os.path.join(RES, '_l1fbgvar_%s.npz' % g),
                prof=prof, warm_k=k0,
                t=t if t is not None else np.array([]),
                p2p_med=p2p if p2p is not None else np.array([]),
                **{k: v[k] for k in VARIANTS})

    print(f'组数 {len(gids)}：{", ".join(gids)}')
    _table(allrows, '全窗评价（k 从 0 起，与初版口径可直接对比）')
    _table(postrows, f'暖机后评价（k >= {warm:.0%} 寿命，真正的在线区间）')

    print(f'\n>>> 暖机后 |rho| 中位最高的因果口径：')
    cand = [k for k in VARIANTS if k in CAUSAL and k not in DEGENERATE]
    rank = sorted(cand, key=lambda k: -float(np.nanmedian(
        np.abs([r[k] for r in postrows]))))
    for k in rank[:3]:
        rs = np.array([r[k] for r in postrows], dtype=float)
        cl = [classify(x) for x in rs]
        print(f"    {k:<12}|rho|中位 {float(np.nanmedian(np.abs(rs))):.2f}   "
              f"正 {cl.count('正')} / 负 {cl.count('负')} / 弱 {cl.count('弱')}")

    print('\n--- 反例（rho < -0.3）---')
    for tag, rows in (('全窗', allrows), ('暖机后', postrows)):
        out = []
        for k in VARIANTS:
            neg = [f"{gids[i]}({rows[i][k]:+.2f})" for i in range(len(gids))
                   if np.isfinite(rows[i][k]) and rows[i][k] < -0.3]
            if neg:
                out.append(f'    {k:<10} {len(neg):>2} 组: {", ".join(neg)}')
        print(f'  [{tag}]' + ('\n' + '\n'.join(out) if out else ' 无'))

    print('\n--- 逐组 rho（暖机后）---')
    print(f"{'组':<8}{'窗数':>7}" + ''.join(f'{k:>11}' for k in cand))
    for i, g in enumerate(gids):
        print(f"{g:<8}{data[g][0].shape[0]:>7}"
              + ''.join(f"{postrows[i][k]:>11.3f}" for k in cand))

    print('\n--- 混杂检验：指标是否在跟踪载荷水平 p2p_med ---')
    print(f"{'口径':<11}{'|rho(指标,p2p)|中位':>20}{'最大':>8}{'>=0.5组数':>11}")
    for k in VARIANTS:
        c = np.array([r[k] for r in confrows], dtype=float)
        print(f"{k:<13}{np.nanmedian(np.abs(c)):>18.3f}"
              f"{np.nanmax(np.abs(c)):>8.3f}{int((np.abs(c) >= 0.5).sum()):>11}")

    if save:
        out = os.path.join(RES, 'l1_fbgvariant_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '窗数'] + [f'{k}_全窗' for k in VARIANTS]
                       + [f'{k}_暖机后' for k in VARIANTS])
            for i, g in enumerate(gids):
                w.writerow([g, data[g][0].shape[0]]
                           + [allrows[i][k] for k in VARIANTS]
                           + [postrows[i][k] for k in VARIANTS])
        print(f'\n写出 {out}')
        print(f'写出 _l1fbgvar_*.npz（含剖面矩阵与全部口径）x {len(gids)}')
    return allrows, postrows


def dump(gid):
    """单组逐窗诊断：分位段看剖面漂移方向与首窗是否异常。"""
    p = os.path.join(RES, f'_l1fbgprof_{gid}.npz')
    if not os.path.exists(p):
        print(f'缺 {p}')
        return
    prof, t, p2p = load_prof(p)
    n = prof.shape[0]
    v, extra = variants(prof)
    print(f'=== {gid}  n={n} 窗口 ===')
    if t is not None:
        print(f'时间跨度 {((t[-1] - t[0]) / 3600.0):.2f} h')
    if p2p is not None:
        print(f'加载窗 p2p 中位 {np.median(p2p):.1f} (min {p2p.min():.1f} / max {p2p.max():.1f})')

    print('\n各口径 rho(与窗序)：')
    for k in VARIANTS:
        print(f'  {k:<12}{v[k][-1]:>10.4f} 末值   rho={rho_of(v[k]):+.3f}')

    print('\n剖面（每 10% 寿命取中位，10 通道）：')
    print('  ' + 'seg'.ljust(12) + ''.join(f'{i:>8}' for i in range(10)))
    for lo in np.linspace(0, 0.9, 10):
        a, b = int(lo * n), max(int(lo * n) + 1, int((lo + 0.1) * n))
        seg = prof[a:b].mean(axis=0)
        print(f'  {lo * 100:>3.0f}%+'.ljust(12) + ''.join(f'{x:>8.4f}' for x in seg))

    print('\nl1_first / l1_ref10 / l1_ref25 / p2p 沿寿命走势（每 10% 中位）：')
    print('  ' + 'seg'.ljust(10) + f"{'l1_first':>10}{'l1_ref10':>10}{'l1_ref25':>10}"
          f"{'p2p':>10}{'centroid':>10}{'asym':>10}")
    for lo in np.linspace(0, 0.9, 10):
        a, b = int(lo * n), max(int(lo * n) + 1, int((lo + 0.1) * n))
        ptxt = (f"{np.median(p2p[a:b]):>10.1f}" if p2p is not None
                else f"{'-':>10}")
        print(f'  {lo * 100:>3.0f}%+'.ljust(10)
              + f"{np.median(v['l1_first'][a:b]):>10.4f}"
              + f"{np.median(v['l1_ref10'][a:b]):>10.4f}"
              + f"{np.median(v['l1_ref25'][a:b]):>10.4f}"
              + ptxt
              + f"{np.median(extra['centroid'][a:b]):>10.3f}"
              + f"{np.median(extra['asym'][a:b]):>10.4f}")

    # 首窗 vs 首 10% 中位的距离 → 判断首窗是否离群
    k10 = max(1, int(round(n * 0.10)))
    ref10 = np.median(prof[:k10], axis=0)
    d0 = np.abs(prof[0] - ref10).sum()
    dall = np.abs(prof[:k10] - ref10).sum(axis=1)
    print(f'\n首窗偏离「首 10% 中位」的距离 = {d0:.4f}；'
          f'该段内中位偏离 = {np.median(dall):.4f}，最大 = {dall.max():.4f}')
    print(f'首窗是否离群: {"是" if d0 > 2 * np.median(dall) else "否"}')

    print('\n首 6 窗剖面：')
    for i in range(min(6, n)):
        print(f'  w{i}: ' + ' '.join(f'{x:.4f}' for x in prof[i]))
    print('\n分位处剖面：')
    for lo in [0.0, 0.25, 0.5, 0.75, 0.9]:
        i = min(n - 1, int(lo * n))
        print(f'  {int(lo * 100):>3}%: ' + ' '.join(f'{x:.4f}' for x in prof[i]))


def _rho(x):
    """序列与自身序号的 Spearman 相关；无定义时返回 0。"""
    r = spearmanr(x, np.arange(len(x)))[0]
    return 0.0 if not np.isfinite(r) else float(r)


def _rob25(prof):
    """稳健基线口径：与「前 25% 中位剖面」的 L1 距离。"""
    k = max(2, int(round(len(prof) * 0.25)))
    return _l1(prof, np.median(prof[:k], axis=0))


def qc_scan(tol=LM_TOL, rel=QC_REL, null_n=400, seed=20261004):
    """窗质量控制 / 载荷级匹配诊断（L1-14 查因，2026-10-04）。

    动机：L1-14 的 `shape_rob25` 与窗序 rho 只有 +0.10，是全批真弱组之一。
    诊断发现它的剖面**寿命漂移只有 0.039（13 组最小，次小 L1-27 是 0.129）**，
    而这点方差几乎全部来自一批 **低 p2p 窗**（占 13.5%）：那批窗的剖面更平坦、
    离基线约 0.10，而 86% 的正常窗只有 0.011 ⇒ rho(shape, p2p) = -0.80（13 组最强）。
    它的载荷程序只有 3 级、幅度仅差 25%（-4/-40 → -5/-50 kN），
    但实测 p2p 跳度达 4 倍 ⇒ p2p 的变化**不能**归因于载荷级，只能归因于**窗质量**。

    两种过滤：
      A. 窗质量控制 `p2p > rel * 中位`：只砍掉非全载窗。
      B. 载荷级匹配 `|p2p - 中位| < tol * 中位`：更窄，等价于「同载比较」。

    零假设检验：B 把样本量砍到 k，而「随机砍到同样样本量」本身也可能改变 rho，
    故对每组抽 null_n 次**同样本量随机子样本**，走完全相同的流程，
    看观测值在零分布上的位置（p 值）。

    结论（11 组，n>=200）：
      A 几乎无害（|Δrho| <= 0.02，10 组），把 L1-14 从 0.104 提到 0.208；
      B 是双刃剑：L1-14 0.104→0.511、L1-25 0.506→0.913（p<0.001 的真实提升），
        但 L1-41 0.821→0.461（显著变差）⇒ **不可全局替换交付口径**。
    """
    rng = np.random.default_rng(seed)
    print('%-7s %5s %8s | %18s | %20s' % (
        '组', '窗数', '全窗', 'A: p2p>%.2f*中位' % rel, 'B: |p2p-中位|<%.0f%%' % (tol * 100)))
    rows = []
    for p in _paths():
        prof, _t, p2p = load_prof(p)
        if prof is None or p2p is None or len(p2p) < 200:
            continue
        gid = _gid(p)
        n = len(p2p)
        full = _rho(_rob25(prof))
        m = p2p > rel * np.median(p2p)
        ra = _rho(_rob25(prof[m])) if m.sum() >= 30 else float('nan')
        ref = np.median(p2p)
        mb = np.abs(p2p - ref) < tol * ref
        kb = int(mb.sum())
        rb, pv, nmean = float('nan'), float('nan'), float('nan')
        if kb >= 30:
            rb = _rho(_rob25(prof[mb]))
            null = np.asarray([
                _rho(_rob25(prof[np.sort(rng.choice(n, kb, replace=False))]))
                for _ in range(null_n)])
            pv = float((null >= rb).mean())
            nmean = float(null.mean())
        rows.append((gid, n, full, ra, int(m.sum()), rb, kb, pv, nmean))
        print('%-7s %5d %8.3f | %7.3f (n=%4d) | %7.3f (n=%4d) p=%.3f 零均值 %+.3f' % (
            gid, n, full, ra, int(m.sum()), rb, kb, pv, nmean))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None, help='单组诊断，如 L1-31')
    ap.add_argument('--warm', type=float, default=WARM, help='暖机比例，默认 0.25')
    ap.add_argument('--qc', action='store_true',
                    help='窗质量控制 / 载荷级匹配诊断（L1-14 查因）')
    ap.add_argument('--lm-tol', type=float, default=LM_TOL,
                    help='载荷级匹配带宽（相对中位 p2p），默认 0.10')
    a = ap.parse_args()
    if a.dump:
        dump(a.dump)
    elif a.qc:
        qc_scan(tol=a.lm_tol)
    else:
        analyze(warm=a.warm)


if __name__ == '__main__':
    main()
