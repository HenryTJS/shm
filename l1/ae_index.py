# -*- coding: utf-8 -*-
"""候选损伤指标的**筛选器** —— 在 L1 新组帧表上逐个指标算「与寿命的相关 + 跨组一致性」。

为什么这么做
------------
老组 C 批的结论是「AE 活动度跨试件差 384 倍 + 阈值跨 campaign 不同 ⇒ 绝对活动度不可用」。
新组恰好相反：26 卷里 21 卷采集配置**完全相同**（阈值 62 dB / 增益 0 dB / 1 MHz），
所以可以正经地筛「跨试件可比的指标」，而不用先赌一个融合权重。

判据（只看数据，不看标签以外的先验）
------------------------------------
对每个指标、每个试件：
  1. 取有效帧（hit ≥ MIN_HITS），算与寿命分数 life = (t−t0)/(t_end−t0) 的 **Spearman ρ**；
  2. 汇总 **同向组数**（|ρ| 显著且方向一致的组数）与 **|ρ| 中位**；
  3. 跨组"方向一致 + 相关强" ⇒ 候选；方向混杂 ⇒ 淘汰（这正是老组 b 值 ρ=−0.62~0.63 的失败形态）。

用法
----
    python l1/ae_index.py                     # 读 results/_l1ae_frames_*.npz
    python l1/ae_index.py --min-hits 50
输出：results/l1_ae_index_rank.csv + 终端明细
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

# 帧表里可直接用的列
RAW_COLS = ['n', 'n_ch1', 'n_ch2', 'amp_mean', 'amp_p50', 'amp_p90', 'amp_max',
            'a95_a50', 'frac_ge80', 'frac_ge90', 'b_value', 'amp_entropy',
            'ener_sum', 'ener_per_hit', 'sig_sum', 'sig_per_hit',
            'dur_p50', 'dur_mean', 'af_p50', 'ra_p50']

CN = {'n': 'hit 数（活动度）', 'n_ch1': '通道1 hit 数', 'n_ch2': '通道2 hit 数',
      'amp_mean': '幅值均值', 'amp_p50': '幅值中位', 'amp_p90': '幅值P90',
      'amp_max': '幅值最大', 'a95_a50': '幅值分位差 A95−A50', 'frac_ge80': '≥80dB 占比',
      'frac_ge90': '≥90dB 占比', 'b_value': 'b 值（G-R）', 'amp_entropy': '幅值熵',
      'ener_sum': '帧能量和', 'ener_per_hit': '平均事件能量',
      'sig_sum': '帧信号强度和', 'sig_per_hit': '平均信号强度',
      'dur_p50': '持续时间中位', 'dur_mean': '持续时间均值',
      'af_p50': 'AF 中位', 'ra_p50': 'RA 中位',
      'shear_frac': '剪切型占比（RA–AF 标准化）',
      'ch_asym': '通道不对称度', 'ener_asym': '通道能量不对称度'}


def load(gid):
    p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    return d


def derive(d):
    """从帧表派生无量纲指标（全部只用到本帧/本组已有信息，除首帧基线外不看未来）。"""
    n1, n2 = d['n_ch1'], d['n_ch2']
    tot = np.maximum(n1 + n2, 1)
    # 通道不对称度：两通道 hit 数的相对差（新组只有 2 通道，无法定位，用它做"损伤偏向"代理）
    d['ch_asym'] = np.abs(n1 - n2) / tot
    # RA–AF 标准化：以「首个有效帧」的 med/IQR 为基线（因果，不看未来）
    ok0 = np.where(d['n'] >= 20)[0]
    if len(ok0) == 0:
        d['shear_frac'] = np.full_like(d['n'], np.nan, dtype=float)
        return d
    i0 = ok0[0]
    ra0 = d['ra_p50'][i0]
    af0 = d['af_p50'][i0]
    iqr_ra = max(np.nanpercentile(d['ra_p50'][ok0], 75)
                 - np.nanpercentile(d['ra_p50'][ok0], 25), 1e-6)
    iqr_af = max(np.nanpercentile(d['af_p50'][ok0], 75)
                 - np.nanpercentile(d['af_p50'][ok0], 25), 1e-6)
    score = (d['af_p50'] - af0) / iqr_af - (d['ra_p50'] - ra0) / iqr_ra
    d['shear_frac'] = (score < 0).astype(float)
    return d


def load_cycle(gid):
    """读循环轴（若有）。返回 (frame, cycle) 或 None。"""
    p = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z['frame'], z['cycle']


def _life_from_cycle(gid, d, mask):
    """寿命分数用**累计循环数**（离线：需要 n_f 归一化）。"""
    c = load_cycle(gid)
    if c is None:
        return None
    cf, cyc = c
    idx = {int(f): i for i, f in enumerate(cf)}
    sel = [idx.get(int(f), -1) for f in d['frame']]
    sel = np.asarray(sel)
    y = np.full(len(d['frame']), np.nan)
    ok = sel >= 0
    y[ok] = cyc[sel[ok]]
    top = np.nanmax(y)
    if not np.isfinite(top) or top <= 0:
        return None
    return y[mask] / top


def analyze(gids, min_hits=20, min_frames=8, axis='cycle'):
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        print('需要 scipy.stats.spearmanr')
        return None
    rows = []
    per = {}
    for gid in gids:
        d = load(gid)
        if d is None:
            continue
        d = derive(d)
        m = d['n'] >= min_hits
        if m.sum() < min_frames:
            print('  %s 有效帧不足（%d），跳过' % (gid, int(m.sum())))
            continue
        life = None
        if axis == 'cycle':
            life = _life_from_cycle(gid, d, m)
        if life is None:
            life = _life(d, m)
        rec = {}
        for c in RAW_COLS + ['shear_frac', 'ch_asym']:
            x = np.asarray(d[c], dtype=float)[m]
            y = life
            if not np.isfinite(x).all() or np.nanstd(x) < 1e-12:
                rec[c] = (np.nan, np.nan)
                continue
            r, p = spearmanr(y, x)
            rec[c] = (float(r), float(p))
        rec['_n'] = int(m.sum())
        rec['_n_f'] = len(d['n'])
        per[gid] = rec

    for c in RAW_COLS + ['shear_frac', 'ch_asym']:
        rs = [v[c][0] for v in per.values() if np.isfinite(v[c][0])]
        if not rs:
            continue
        rs = np.asarray(rs)
        pos = int((rs > 0.3).sum())
        neg = int((rs < -0.3).sum())
        weak = int((np.abs(rs) <= 0.3).sum())
        rows.append({'指标': c, '中文': CN.get(c, c), '可用组数': len(rs),
                     '同向_正': pos, '同向_负': neg, '弱相关': weak,
                     '|rho|中位': float(np.median(np.abs(rs))),
                     'rho中位': float(np.median(rs)),
                     '一致性': max(pos, neg) / len(rs)})
    rows.sort(key=lambda r: (-r['一致性'], -r['|rho|中位']))
    return rows, per


def _life(d, mask=None):
    """寿命分数代理。

    用**活跃帧的序号**而不是日历时间：这批试验有大量停机夜歇
    （最长组跨度 365 h 但实际加载时间远小于此），若用日历时间，
    停机会把「进度」稀释，导致明明随损伤上升的量被判为下降。
    """
    n = len(d['frame'])
    idx = np.arange(n, dtype=float)
    if mask is None:
        span = idx[-1] - idx[0]
        return (idx - idx[0]) / span if span > 0 else np.full(n, np.nan)
    ok = idx[np.asarray(mask, dtype=bool)]
    if len(ok) < 2:
        return np.full(len(ok), np.nan)
    return (ok - ok[0]) / max(ok[-1] - ok[0], 1.0)


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    min_hits = 20
    axis = 'cycle'
    if '--min-hits' in argv:
        min_hits = int(argv[argv.index('--min-hits') + 1])
    if '--axis' in argv:
        axis = argv[argv.index('--axis') + 1]
    print('寿命轴 =', axis)
    gids = sorted(os.path.basename(p)[len('_l1ae_frames_'):-4]
                  for p in glob.glob(os.path.join(RES, '_l1ae_frames_*.npz')))
    if not gids:
        print('未找到帧表，请先跑 python l1/ae_frames.py')
        return 1
    print('组数 %d：%s' % (len(gids), ' '.join(gids)))
    out = analyze(gids, min_hits=min_hits, axis=axis)
    if out is None:
        return 1
    rows, per = out
    print('\n== 指标排名（按跨组一致性；同向 = |rho|>0.3 且方向一致）==')
    print('%-34s %s %s %s %s %s' % ('指标', '可用组', '正', '负', '弱', '一致性/|rho|中位'))
    for r in rows:
        print('%-34s %4d %4d %4d %4d   %.2f / %.2f'
              % ('%s (%s)' % (r['中文'], r['指标']), r['可用组数'], r['同向_正'],
                 r['同向_负'], r['弱相关'], r['一致性'], r['|rho|中位']))
    print('\n== 逐组明细（Spearman rho vs 寿命分数）==')
    hdr = ['组'] + [r['指标'] for r in rows]
    print('%-8s' % '组' + ''.join('%-9s' % h[:8] for h in hdr[1:]))
    for gid, rec in per.items():
        line = '%-8s' % gid
        for r in rows:
            v = rec[r['指标']][0]
            line += '%-9s' % ('%.2f' % v if np.isfinite(v) else '—')
        print(line)

    import csv
    out_csv = os.path.join(RES, 'l1_ae_index_rank.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n已写出 ->', out_csv)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
