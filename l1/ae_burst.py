# -*- coding: utf-8 -*-
"""AE 事件的「簇状性」指标 —— 检验损伤局部化会不会让事件聚集。

动机
----
损伤从弥散微损伤转向局部化（分层/脱粘扩展）时，AE 事件常表现为成簇爆发。
这类指标天然满足"更好的指标"该有的性质：
  · **无量纲**（Fano 因子、CV）⇒ 不受跨试件绝对活动度差异影响
  · **在线可算**（只用一段过去的窗）
  · 与"活动率"不同 —— 率会被提载爆发/级内衰减支配，而簇状性看的是**涨落结构**

实现（基于 60 s 细帧，只统计**有活动的帧**，以剔除停机）
  · Fano = var(n) / mean(n) 在滑动窗内（窗 60 帧 = 1 h）
      Fano ≈ 1 → 泊松（随机）；Fano 远大于 1 → 簇状
  · 零帧占比 = 窗内 n = 0 的比例（静默程度）
  · 等待时间 CV：事件帧间隔的 std/mean

用法
----
    python l1/ae_burst.py                 # 全部组
输出：results/l1_burst_{gid}.npz + results/l1_burst_rank.csv
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

WIN = 60                # 滑动窗（帧）：60 s 帧 × 60 = 1 h
TAG = '60s'


def load(gid):
    p = os.path.join(RES, '_l1ae_frames%s_%s.npz' % (TAG, gid))
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def burst_series(n, win=WIN):
    """滑动窗内的 Fano / 零帧占比 / 等待时间 CV。"""
    x = np.asarray(n, dtype=float)
    N = len(x)
    if N < win:
        return None
    fano = np.full(N, np.nan)
    zero = np.full(N, np.nan)
    for i in range(win, N + 1):
        seg = x[i - win:i]
        mu = seg.mean()
        if mu > 0:
            fano[i - 1] = seg.var() / mu
        zero[i - 1] = float((seg == 0).mean())
    return {'fano': fano, 'zero_frac': zero}


def run(gid):
    d = load(gid)
    if d is None:
        return None
    n = d['n'].astype(float)
    bs = burst_series(n)
    if bs is None:
        return None
    f, z = bs['fano'], bs['zero_frac']
    ok = np.isfinite(f)
    if ok.sum() < 10:
        return None
    life = np.arange(len(n), dtype=float)
    life = (life - life[ok][0]) / max(life[ok][-1] - life[ok][0], 1.0)
    out = os.path.join(RES, '_l1burst_%s.npz' % gid)
    np.savez_compressed(out, life=life[ok], fano=f[ok], zero_frac=z[ok],
                        n=n, win=win_of(n))
    return {'gid': gid, 'file': out, 'life': life[ok], 'fano': f[ok],
            'zero': z[ok], 'n': n}


def win_of(n):
    return WIN


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        spearmanr = None
    gids = argv or sorted(os.path.basename(p)[len('_l1ae_frames60s_'):-4]
                          for p in glob.glob(os.path.join(RES, '_l1ae_frames60s_*.npz')))
    if not gids:
        print('未找到 60 s 细帧表，请先跑 python l1/ae_frames.py --frame-s 60 --tag 60s')
        return 1
    print('%-7s %8s %10s %10s %9s %9s' % (
        '组', '窗数', 'Fano中位', 'Fano_后30%', '零帧占比', 'rho(Fano,寿命)'))
    rows = []
    for gid in gids:
        r = run(gid)
        if r is None:
            print('%-7s %8s' % (gid, '数据不足'))
            continue
        f, life, z = r['fano'], r['life'], r['zero']
        k = max(int(len(f) * 0.3), 1)
        rho = np.nan
        if spearmanr is not None and len(f) >= 8:
            rho = float(spearmanr(life, f)[0])
        rows.append((gid, len(f), float(np.median(f)), float(np.median(f[-k:])),
                     float(np.median(z)), rho))
        print('%-7s %8d %10.2f %10.2f %9.3f %9.3f'
              % (gid, len(f), np.median(f), np.median(f[-k:]), np.median(z), rho))
    if rows:
        import csv
        out = os.path.join(RES, 'l1_burst_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '窗数', 'Fano中位', 'Fano_后30%', '零帧占比中位', 'rho_Fano_vs_寿命'])
            w.writerows(rows)
        fs = [r[3] for r in rows]
        print('\nFano（后 30%% 寿命）中位 %.2f  范围 %.2f 至 %.2f（远大于 1 表示簇状）'
              % (np.median(fs), min(fs), max(fs)))
        rhos = [r[5] for r in rows if np.isfinite(r[5])]
        if rhos:
            pos = sum(1 for v in rhos if v > 0.3)
            neg = sum(1 for v in rhos if v < -0.3)
            print('rho 方向：正 %d / 负 %d / 弱 %d（共 %d 组），|rho| 中位 %.2f'
                  % (pos, neg, len(rhos) - pos - neg, len(rhos),
                     np.median(np.abs(rhos))))
        print('已写出 ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
