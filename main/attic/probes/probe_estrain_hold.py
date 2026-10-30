# -*- coding: utf-8 -*-
"""应变「持续发散」判据的交叉验证。

改动：`estrain_hold = 0` 为旧口径；>0 时要求连续 hold 个块的归一化发散
≥ 0.5 才采信应变。目的：把 024 在 28.7% 处「抬两块就回落」的伪发散剔除，
同时不能把 022 / 025 的有效早警一起剔掉。

验证组：022（应变早警有效）、024（伪发散）、025（应变早警有效）
对照：018、020（锚点最稳、应变不参与或轻微参与）

用法
    python main/probe_estrain_hold.py
"""
import os
import sys

# ---- 路径前缀：本文件在 main/attic/ 下 ----
# attic（eval_common / pipeline）← main（config / streaming / damage_index）← 仓库根（paths.py）
_HERE0 = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(os.path.dirname(_HERE0)), os.path.dirname(_HERE0), _HERE0):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pipeline as P                                          # noqa: E402
from damage_index import OnlineDamageIndex                # noqa: E402
from eval_common import onset_of, ref_stage, HOLD_FRAC        # noqa: E402

GIDS = ['018', '020', '022', '024', '025']
HOLDS = [0, 5, 10, 20]
BAND = 15.0


def grade_of(tw, b1):
    if tw is None:
        return 'C'
    if b1 is None or np.isnan(b1):
        return 'N'
    e = tw - b1
    return 'E' if e < -BAND else ('D' if e > BAND else 'A')


def main():
    refs = ref_stage()
    print('gid    b1     ' + ''.join('  hold=%-2d        ' % h for h in HOLDS))
    tally = {h: 0 for h in HOLDS}
    for gid in GIDS:
        b1, _ = refs.get(gid, (np.nan, np.nan))
        cells = []
        for h in HOLDS:
            d = P.eval_stream_d(gid, OnlineDamageIndex({'estrain_hold': h}))
            hold = max(2000, int(len(d) * HOLD_FRAC))
            tw = onset_of(d, hold)
            g = grade_of(tw, b1)
            tally[h] += g == 'A'
            cells.append('%6s %s' % ('-' if tw is None else '%.1f' % tw, g))
        print('%-5s %5.1f ' % (gid, b1) + ''.join('  %-14s' % c for c in cells),
              flush=True)
    print('\nA 级组数：' + '  '.join('hold=%d -> %d' % (h, tally[h]) for h in HOLDS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
