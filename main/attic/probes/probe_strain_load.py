# -*- coding: utf-8 -*-
"""验证一个假设：024 的应变 std 早抬是否由「载荷在变化」造成。

前面的 AE 门控与上限耦合都失败了（见 early_alarm_023_024.md）。
剩下来最可能的判别量不是「AE 在不在」，而是「载荷在不在变」：
应变块 std 在加载步上升时天然变大，与损伤发散同形态。

做法：把应变按 2.5% 分块，同时打印块**均值**（≈载荷水平）与块 **std**。
  · 若 std 抬升处 mean 也在明显抬升 → 载荷在变，属伪发散；
  · 若 std 抬升处 mean 平坦 → 真正发散。

用法
    python main/probe_strain_load.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pipeline as P                                          # noqa: E402
from eval_common import ref_stage                             # noqa: E402

GIDS = ['022', '023', '024', '025']
NB = 40          # 2.5% 每块


def main():
    refs = ref_stage()
    for gid in GIDS:
        dl = P._load_ae_strain(gid)
        if dl.strain is None or 'strain' not in dl.strain.columns:
            print('%s: 无应变，跳过' % gid)
            continue
        s = np.asarray(dl.strain['strain'].values, float)
        s = s[np.isfinite(s)]
        n = s.size
        edges = np.linspace(0, n, NB + 1).astype(int)
        mu, sd = [], []
        for i in range(NB):
            v = s[edges[i]:edges[i + 1]]
            mu.append(v.mean() if v.size else np.nan)
            sd.append(v.std() if v.size else np.nan)
        mu, sd = np.array(mu), np.array(sd)
        base = np.nanmedian(sd[max(0, int(.2 * NB)):int(.45 * NB)])
        b1, b3 = refs.get(gid, (np.nan, np.nan))
        print('\n=== %s === b1=%.1f b3=%.1f  基线 std(20-45%%)=%.3g'
              % (gid, b1, b3, base))
        print(' life%    mean      std    std/base   d(mean)/块')
        dmu = np.diff(mu, prepend=mu[0])
        for i in range(NB):
            life = (i + 0.5) / NB * 100
            print('  %5.1f  %8.2f  %7.3g  %7.2f  %10.3g'
                  % (life, mu[i], sd[i], sd[i] / base if base > 0 else np.nan, dmu[i]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
