"""跨机构 HI 一致性：把两所大学 / 两套采集系统的 28 组放进同一口径。

- C1 + C2（Lab A，Vallen .pridb）  ← l1/results/l1_hi_ae.csv（--batch both 产物）
- C3 + C4（Lab B，PAC/Mistras .DTA）← 帧表 + 重建循环轴，累积口径现算

口径（两侧完全相同的定义）：
    累积 AE 量自归一化曲线达到终值 x% 时，处于寿命的百分之几 → t50 / t85
  该定义与帧宽、箱宽、采样率、记录格式均无关，因此可跨系统比较。
  速率型（win1_* / win10_*）依赖分箱宽度，**不**参与跨机构比较。

同时做随机化检验：把组随机重排到各批，看各批中位数的极差能否小于观测值。

只写一个新文件 l1/results/l1_hi_crosslab.csv，不改动任何既有产物。
"""
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

RES = r'l1/results'
RNG = np.random.default_rng(20261011)
NPERM = 20000

BATCH = {}
for g in ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-23']:
    BATCH[g] = ('C1', 'A', 'Vallen .pridb')
for g in ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60']:
    BATCH[g] = ('C2', 'A', 'Vallen .pridb')
for g in ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']:
    BATCH[g] = ('C3', 'B', 'PAC/Mistras .DTA')
for g in ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35', 'L1-36', 'L1-41', 'L1-44']:
    BATCH[g] = ('C4', 'B', 'PAC/Mistras .DTA')

# ---------------------------------------------------------------- Lab A 侧
def load_labA(variant):
    """读 l1_hi_ae.csv（--batch both 产物）。"""
    out = {}
    with open(os.path.join(RES, 'l1_hi_ae.csv'), encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['variant'] != variant:
                continue
            g = r['gid']
            if g not in BATCH or BATCH[g][1] != 'A':
                continue
            out[g] = {'gid': g, 'n_f': float(r['n_f']), 'cov': float(r['cov']),
                      'mon': float(r['mon']),
                      't50': float(r['t50_pct']), 't85': float(r['t85_pct'])}
    # L1-23 不在 l1_hi_ae.csv：它的 AE 导出是 4.11 GB 事件级文件，
    # 由 l1/hi_l123.py 单独流式处理（口径相同），产物为 l1_hi_l123.csv。
    p23 = os.path.join(RES, 'l1_hi_l123.csv')
    if os.path.exists(p23):
        with open(p23, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['variant'] != variant:
                    continue
                out[r['gid']] = {'gid': r['gid'], 'n_f': float(r['n_f']),
                                 'cov': float(r['cov']), 'mon': np.nan,
                                 't50': float(r['t50_pct']), 't85': float(r['t85_pct'])}
    return out


# ---------------------------------------------------------------- Lab B 侧
def curves(gid):
    fp = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    with np.load(fp, allow_pickle=True) as z:
        fr, n = z['frame'].astype(np.int64), z['n'].astype(float)
        en = z['ener_sum'].astype(float)
    with np.load(cp, allow_pickle=True) as z:
        cf, cyc, n_f = z['frame'].astype(np.int64), z['cycle'].astype(float), float(z['n_f'])
    m = dict(zip(cf.tolist(), cyc.tolist()))
    keep = np.array([int(x) in m for x in fr])
    fr, n, en = fr[keep], n[keep], en[keep]
    c = np.array([m[int(x)] for x in fr], float)
    ok = np.isfinite(c) & (c >= 0)
    c, n, en = c[ok], n[ok], en[ok]
    if c.size == 0:
        return None
    o = np.argsort(c)
    return {'n_f': n_f, 'x': c[o] / n_f * 100, 'n': n[o], 'en': en[o],
            'cov': float(c.max() / n_f)}


def t_at(x, cum, frac):
    tot = cum[-1]
    if not (tot > 0):
        return np.nan
    j = min(int(np.searchsorted(cum, frac * tot)), len(x) - 1)
    return float(x[j])


def load_labB(variant):
    out = {}
    for g in BATCH:
        if BATCH[g][1] != 'B':
            continue
        d = curves(g)
        if d is None:
            print('  [%s] 缺帧表或循环轴 → 跳过' % g)
            continue
        cum = np.cumsum(d['n'] if variant == 'cum_hits' else d['en'])
        out[g] = {'gid': g, 'n_f': d['n_f'], 'cov': d['cov'], 'mon': np.nan,
                  't50': t_at(d['x'], cum, 0.50), 't85': t_at(d['x'], cum, 0.85)}
    return out


# ---------------------------------------------------------------- 统计
def summarise(rows, col):
    """逐批中位 + 全批极差。"""
    med, rng = {}, {}
    for b in ('C1', 'C2', 'C3', 'C4'):
        v = [r[col] for r in rows if r['batch'] == b and np.isfinite(r[col])]
        if v:
            med[b] = float(np.median(v))
            rng[b] = (float(min(v)), float(max(v)), len(v))
    if len(med) < 2:
        return med, rng, np.nan
    return med, rng, float(max(med.values()) - min(med.values()))


def rand_test(rows, col):
    """观测到的批间极差 vs 随机重排。"""
    vals = np.array([r[col] for r in rows if np.isfinite(r[col])], float)
    labs = np.array([r['batch'] for r in rows if np.isfinite(r[col])])
    if len(vals) < 8:
        return np.nan, np.nan
    sizes = {b: int((labs == b).sum()) for b in np.unique(labs)}

    def spread(lab_arr, v):
        out = []
        for b, k in sizes.items():
            idx = np.where(lab_arr == b)[0]
            if len(idx):
                out.append(np.median(v[idx]))
        return max(out) - min(out) if len(out) > 1 else np.nan

    obs = spread(labs, vals)
    cnt = 0
    labp = labs.copy()
    for _ in range(NPERM):
        RNG.shuffle(labp)
        if spread(labp, vals) <= obs:
            cnt += 1
    return obs, (cnt + 1) / (NPERM + 1)


BATCH_LAB = {'C1': 'A', 'C2': 'A', 'C3': 'B', 'C4': 'B'}

print('=' * 104)
print('跨机构 HI 一致性：2 所大学 · 2 套采集系统 · 4 个试验活动 · 28 个试件')
print('=' * 104)

allrows = []
for variant in ('cum_hits', 'cum_energy'):
    a = load_labA(variant)
    b = load_labB(variant)
    for src in (a, b):
        for g, r in src.items():
            bt, lab, sysname = BATCH[g]
            allrows.append(dict(batch=bt, lab=lab, system=sysname,
                                variant=variant, **r))
    print()
    print('--- %s ---' % variant)
    rows = [r for r in allrows if r['variant'] == variant]
    for col in ('t50', 't85'):
        med, rng, sp = summarise(rows, col)
        print('  %s：' % col.upper())
        for b in ('C1', 'C2', 'C3', 'C4'):
            if b in med:
                lo, hi, n = rng[b]
                print('     %-3s (Lab %s) n=%2d  中位 %5.1f%%   范围 %5.1f 至 %5.1f'
                      % (b, BATCH_LAB[b], n, med[b], lo, hi))
        obs, p = rand_test(rows, col)
        print('     批间极差 %.1f 个百分点   随机重排 p = %.4f' % (sp, p))

# 写出
fp = os.path.join(RES, 'l1_hi_crosslab.csv')
with open(fp, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh)
    w.writerow(['gid', 'batch', 'lab', 'system', 'variant', 'n_f', 'cov',
                'mon', 't50_pct', 't85_pct'])
    for r in sorted(allrows, key=lambda z: (z['variant'], z['batch'], z['gid'])):
        w.writerow([r['gid'], r['batch'], r['lab'], r['system'], r['variant'],
                    '%.0f' % r['n_f'], '%.3f' % r['cov'],
                    '' if not np.isfinite(r['mon']) else '%.3f' % r['mon'],
                    '%.1f' % r['t50'], '%.1f' % r['t85']])
print()
print('已写 %s（%d 行）' % (fp, len(allrows)))
