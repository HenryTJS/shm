"""补第 29 组：L1-23（C1 的第五组）纳入同一跨机构口径。

L1-23 的 `声发射.csv` 是 4.11 GB / 约 38 M 行的事件级导出（12 列），
直接 pd.read_csv 会吃 4 至 6 GB 内存 ⇒ 这里分块流式只取 time / energy。

口径与 l1_hi_crosslab.py 完全一致：累积量达到终值 x% 时的寿命百分位。
"""
import csv
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

L1 = os.path.dirname(os.path.abspath(__file__))
ROOT = r'E:\l1'
GID = 'L1-23'

# 复用 evaluate_l1_hi_ae 的循环锚（FBG 块锚）
spec = importlib.util.spec_from_file_location('_ev', os.path.join(L1, 'evaluate_l1_hi_ae.py'))
ev = importlib.util.module_from_spec(spec)
sys.modules['_ev'] = ev
spec.loader.exec_module(ev)

n_f = ev.load_meta(GID)['n_f']
print('%s  n_f = %.0f' % (GID, n_f))

fp = os.path.join(ROOT, GID, '%s声发射.csv' % GID)
print('流式读取 %s（%.2f GB）...' % (os.path.basename(fp), os.path.getsize(fp) / 1024**3))

ts, ens, nrow = [], [], 0
for ch in pd.read_csv(fp, encoding='utf-8-sig', usecols=['time', 'energy'],
                      dtype={'time': np.float64, 'energy': np.float64},
                      chunksize=5_000_000):
    ts.append(ch['time'].to_numpy())
    ens.append(ch['energy'].to_numpy())
    nrow += len(ch)
    print('  已读 %6.1f M 行' % (nrow / 1e6), end='\r')
t = np.concatenate(ts)
en = np.maximum(np.concatenate(ens), 0.0)
del ts, ens
print('\n  共 %d 个事件' % nrow)

ac = np.asarray(ev._cycle_of(GID, t, n_f), float)
ok = np.isfinite(ac)
ac, en = ac[ok], en[ok]
print('  落在寿命内的事件 %d 个（%.1f%%）' % (ac.size, 100.0 * ac.size / max(nrow, 1)))
o = np.argsort(ac)
ac, en = ac[o], en[o]
x = ac / n_f * 100.0


def t_at(cum, frac):
    tot = cum[-1]
    if not (tot > 0):
        return np.nan
    j = min(int(np.searchsorted(cum, frac * tot)), len(x) - 1)
    return float(x[j])


n_hits = np.ones(ac.size, float)
ch_cum = np.cumsum(n_hits)
ce_cum = np.cumsum(en)
print()
print('  %-12s t50 = %5.1f%%   t85 = %5.1f%%' % ('cum_hits', t_at(ch_cum, .50), t_at(ch_cum, .85)))
print('  %-12s t50 = %5.1f%%   t85 = %5.1f%%' % ('cum_energy', t_at(ce_cum, .50), t_at(ce_cum, .85)))
print()
print('参照：C1 其余四组 cum_hits t85 中位 92.6%%（86.4 至 97.4）')

fp_out = os.path.join(L1, 'results', 'l1_hi_l123.csv')
with open(fp_out, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh)
    w.writerow(['gid', 'variant', 'n_f', 'cov', 't50_pct', 't85_pct'])
    for v, cum in (('cum_hits', ch_cum), ('cum_energy', ce_cum)):
        w.writerow([GID, v, '%.0f' % n_f, '%.3f' % (ac.max() / n_f),
                    '%.1f' % t_at(cum, .50), '%.1f' % t_at(cum, .85)])
print('已写 %s' % fp_out)
