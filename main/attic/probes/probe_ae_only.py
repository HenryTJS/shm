# -*- coding: utf-8 -*-
"""声发射单源（sources=('ae',)）与默认双源的逐组对比。

动机
----
消融显示 024 的过早报警完全由应变辅证造成：去掉应变后 t_warn 从 30.6
移到 72.1，而其 b1 = 65.5（极差 5.4，锚点较稳）。023 同向但幅度小。
本脚本把「去掉应变辅证」这一改动在全部可评估组上跑一遍，确认它是否
全局更优，而不是只对 024 有效。

用法
    python main/probe_ae_only.py
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
from eval_common import onset_of, ref_stage, GROUPS_EVAL, HOLD_FRAC  # noqa: E402

SKIP = {'015', '021', '026', '027'}     # 数据不可加载 / 单阶段，不参与
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
    print('gid     b1     b3   | 双源 t_warn  评分 | 单源 t_warn  评分 | 变化')
    n_a2 = n_a1 = 0
    rows = []
    for gid in GROUPS_EVAL:
        if gid in SKIP:
            continue
        b1, b3 = refs.get(gid, (np.nan, np.nan))
        out = {}
        for tag, kw in (('dual', {}), ('ae', {'sources': ('ae',)})):
            d = P.eval_stream_d(gid, OnlineDamageIndex(kw))
            hold = max(2000, int(len(d) * HOLD_FRAC))
            out[tag] = onset_of(d, hold)
        g2 = grade_of(out['dual'], b1)
        g1 = grade_of(out['ae'], b1)
        n_a2 += g2 == 'A'
        n_a1 += g1 == 'A'
        rows.append((gid, b1, b3, out['dual'], g2, out['ae'], g1))
        print('%-4s %6.1f %6.1f |  %8s  %s    |  %8s  %s    | %s'
              % (gid, b1, b3,
                 '-' if out['dual'] is None else '%.1f' % out['dual'], g2,
                 '-' if out['ae'] is None else '%.1f' % out['ae'], g1,
                 '改善' if g1 == 'A' and g2 != 'A' else
                 ('变差' if g2 == 'A' and g1 != 'A' else '')),
              flush=True)
    print('\nA 级组数：双源 %d / 单源 %d（共 %d 组）' % (n_a2, n_a1, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
