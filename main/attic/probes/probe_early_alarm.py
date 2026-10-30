# -*- coding: utf-8 -*-
"""023 / 024 为什么过早报警：按证据消融定位。

背景
----
两阶段口径下 023 的 b1 = 39.5、024 的 b1 = 65.5，而在线 A 级预警分别在
11.2 与 30.6，偏差 −28.3 与 −34.9，判为「E 过早」。

D 的风险由三部分合成（见 main/damage_index.py 文件头）：
    e_ae     = max(e_dmg, e_full, e_shape)      声发射主源
    e_strain = estrain_w × 应变 std 相对滚动低分位发散    应变辅证
    risk     = max(e_ae, e_strain)（fusion=max）
因此过早只可能来自三处：损伤型事件判据 e_dmg、全能量加速 e_full、
形状证据 e_shape，或应变辅证 e_strain。

做法
----
对同一组数据换 `abl` 重放整条流，比较 t_warn 的移动。哪个消融把 t_warn
推后到 b1 附近，那个分量就是元凶。

    full        完整
    no_strain   去掉应变辅证
    only_strain 只看应变辅证
    no_shape    去掉形状证据
    only_shape  只看形状证据
    no_full     去掉全能量加速
    no_dmg      去掉损伤型事件分类

用法
    python main/probe_early_alarm.py
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


ABLS = ['full', 'no_strain', 'only_strain', 'no_shape', 'only_shape',
        'no_full', 'no_dmg']
GIDS = ['018', '023', '024']


def first_ge(d, th):
    i = np.where(np.asarray(d) >= th)[0]
    return float(i[0]) / len(d) * 100.0 if len(i) else np.nan


def main():
    refs = ref_stage()
    print('gid  abl           t25    t55    t85   t_warn     b1     b3')
    rows = []
    for gid in GIDS:
        b1, b3 = refs.get(gid, (np.nan, np.nan))
        for abl in ABLS:
            d = P.eval_stream_d(gid, OnlineDamageIndex({'abl': abl}))
            n = len(d)
            hold = max(2000, int(n * HOLD_FRAC))
            tw = onset_of(d, hold)
            t25, t55, t85 = first_ge(d, .25), first_ge(d, .55), first_ge(d, .85)
            rows.append((gid, abl, t25, t55, t85, tw, b1, b3))
            print('%-4s %-12s %6.1f %6.1f %6.1f  %7s  %6.1f %6.1f'
                  % (gid, abl, t25, t55, t85,
                     '-' if tw is None else '%.1f' % tw,
                     -999 if np.isnan(b1) else b1, -999 if np.isnan(b3) else b3),
                  flush=True)
        print('', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
