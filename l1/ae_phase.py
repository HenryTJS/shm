# -*- coding: utf-8 -*-
"""二维损伤相图 —— 把「量级」（HI）与「形态」（Fano 簇状性）组合起来看。

思路
----
单指标筛选的结论是：累积量（HI_hit）与簇状性（Fano）各自零反例，但机理不同 ——
一个看**累积了多少**，一个看**事件扎不扎堆**。把两者放进同一张图（横轴 HI，
纵轴 Fano），试件就沿着一条轨迹走。要回答三个问题：

  1. 不同试件的轨迹是否**形状相似**（若相似 ⇒ 指标组合稳定，有普适性）
  2. 末期是否**聚到同一区域**（损伤终态是否一致）
  3. **预置脱粘**（L1-41 = D20x20、L1-44 = D25x20，没做冲击）与**冲击 BVID** 组
     是否落在相图的**不同区域** ⇒ 若能分开，说明这个组合对**损伤类型**敏感

寿命轴统一用循环分数（cycle / n_f），两类帧表（600 s 与 60 s）都插值到同一根循环轴上。

用法
----
    python l1/ae_phase.py                  # 全部组
    python l1/ae_phase.py L1-41 L1-44
输出：results/l1_phase_{gid}.csv + results/l1_phase_summary.csv
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

N_BIN = 20
PRISTINE_DISBOND = {'L1-41', 'L1-44'}      # 预置脱粘（非冲击）


def _life_on_cycle(gid, frame_arr, frame_s):
    """把某套帧表的时间轴插值到循环轴 → 寿命分数 cycle/n_f。"""
    cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not os.path.exists(cp):
        return None
    c = np.load(cp, allow_pickle=True)
    t_grid = c['t_epoch']
    cyc = c['cycle']
    t_here = np.asarray(frame_arr, dtype=float) * frame_s
    y = np.interp(t_here, t_grid, cyc)
    top = float(np.nanmax(cyc))
    return y / top if top > 0 else None


def phase(gid, n_bin=N_BIN):
    fp6 = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    fp1 = os.path.join(RES, '_l1ae_frames60s_%s.npz' % gid)
    if not os.path.exists(fp6):
        return None
    z6 = np.load(fp6, allow_pickle=True)
    life6 = _life_on_cycle(gid, z6['frame'], 600.0)
    if life6 is None:
        return None
    n6 = z6['n'].astype(float)

    # Fano 序列（60 s 帧）
    life1 = fano1 = None
    if os.path.exists(fp1):
        z1 = np.load(fp1, allow_pickle=True)
        life1 = _life_on_cycle(gid, z1['frame'], 60.0)
        from ae_burst import burst_series, WIN
        bs = burst_series(z1['n'].astype(float), WIN)
        if bs is not None:
            fano1 = bs['fano']

    edges = np.linspace(0, 1, n_bin + 1)
    hi_cum, share, fano_bin, hit_sum, life_mid = [], [], [], [], []
    tot = n6.sum() if n6.sum() > 0 else 1.0
    for k in range(n_bin):
        m = (life6 >= edges[k]) & (life6 < edges[k + 1])
        s = n6[m].sum()
        hit_sum.append(s)
        share.append(s / tot)
        hi_cum.append(np.cumsum(hit_sum)[-1] / tot)
        life_mid.append((edges[k] + edges[k + 1]) / 2)
        if fano1 is not None:
            m1 = (life1 >= edges[k]) & (life1 < edges[k + 1])
            f = fano1[m1]
            f = f[np.isfinite(f)]
            fano_bin.append(float(np.median(f)) if len(f) else np.nan)
        else:
            fano_bin.append(np.nan)
    d = {'life': np.asarray(life_mid), 'share': np.asarray(share),
         'hi_cum': np.asarray(hi_cum), 'fano': np.asarray(fano_bin),
         'kind': '预置脱粘' if gid in PRISTINE_DISBOND else '冲击'}
    out = os.path.join(RES, 'l1_phase_%s.csv' % gid)
    np.savetxt(out, np.column_stack([d['life'], d['share'], d['hi_cum'], d['fano']]),
               delimiter=',', header='life,share,hi_cum,fano', comments='')
    return d


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    gids = argv or sorted(os.path.basename(p)[len('_l1ae_frames_'):-4]
                          for p in glob.glob(os.path.join(RES, '_l1ae_frames_*.npz'))
                          if '60s' not in p)
    rows = []
    print('%-7s %-8s %10s %10s %10s %10s %10s'
          % ('组', '类型', 'Fano@20%', 'Fano@50%', 'Fano@80%', '末期Fano', '末期HI'))
    for gid in gids:
        d = phase(gid)
        if d is None:
            continue
        f = d['fano']
        if not np.isfinite(f).any():
            continue
        def at(frac):
            i = min(int(frac * len(f)), len(f) - 1)
            return f[i]
        rows.append({'组号': gid, '类型': d['kind'],
                     'Fano@20%': round(float(at(0.2)), 1) if np.isfinite(at(0.2)) else None,
                     'Fano@50%': round(float(at(0.5)), 1) if np.isfinite(at(0.5)) else None,
                     'Fano@80%': round(float(at(0.8)), 1) if np.isfinite(at(0.8)) else None,
                     '末期Fano': round(float(np.nanmedian(f[-int(len(f) * 0.2):])), 1),
                     '末期HI': round(float(d['hi_cum'][-1]), 3)})
        r = rows[-1]
        print('%-7s %-8s %10s %10s %10s %10s %10s'
              % (gid, r['类型'], r['Fano@20%'], r['Fano@50%'], r['Fano@80%'],
                 r['末期Fano'], r['末期HI']))

    if rows:
        import csv
        out = os.path.join(RES, 'l1_phase_summary.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # 判据 1：末期 Fano 随寿命是否整体抬升（Fano@20% → 末期）
        up = sum(1 for r in rows if r['Fano@20%'] and r['末期Fano'] > r['Fano@20%'])
        print('\n[判据 1] 末期 Fano 高于 20%% 寿命处：%d / %d 组' % (up, len(rows)))
        # 判据 2：类型可分性 —— 末期 Fano 的组间分布
        dis = [r['末期Fano'] for r in rows if r['类型'] == '预置脱粘']
        imp = [r['末期Fano'] for r in rows if r['类型'] == '冲击']
        print('[判据 2] 末期 Fano：预置脱粘 %s（n=%d） vs 冲击 中位 %.1f（n=%d）'
              % (dis, len(dis), float(np.median(imp)), len(imp)))
        print('已写出 ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
