# -*- coding: utf-8 -*-
"""已归档模块 `evaluate_l1_degree` 的最小桩（仅用于回放旧 L1 看板导出器）。

背景：2026-10-07「L1 在线口径下线」时该脚本被归档，磁盘上已不存在；
但 `l1/__pycache__/export_dashboard_l1.cpython-312.pyc` 仍在，重建 C1 看板包时会被 import。

回放时该导出器实际只用到：
  * `META[gid]['n_f']`      → 失效循环数（= l1_meta.load_meta）
  * `CYCS_PER_BLK / CYCS_PER_PT / PTS_PER_BLK` → 块划分常量
  * `run_group(...)['cyc'] / ['strain']`       → 点级循环轴与应变
  * `run_group(...)['peak']`                   → 帧级 AE 峰值（给 NaN ⇒ 看板留空，不伪造）

**主曲线不用这里给出的任何量**：导出器在 HI 可用时用
`evaluate_l1_hi_ae.hi_of_group()` 的离线复评 HI 覆盖 D/risk/eae/est/c0。
"""
from __future__ import annotations

import os

import numpy as np

from l1_meta import load_meta
import _l1_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09']


class _Meta(dict):
    """按需补全：归档导出器会直接 `_deg.META[gid]` 取 n_f；
    新增组（如 L1-23，2026-10-09 补入 C1）不在上面的 GROUPS 里，这里即时解析。"""

    def __missing__(self, gid):
        self[gid] = load_meta(gid)
        return self[gid]


META = _Meta({g: load_meta(g) for g in GROUPS})

CYCS_PER_BLK = _l1_grid.CYCS_PER_BLK
CYCS_PER_PT = _l1_grid.CYCS_PER_PT
PTS_PER_BLK = CYCS_PER_BLK // CYCS_PER_PT


def run_group(gid, *_a, **_k):
    """只提供循环轴与应变；其余字段给中性值（HI 可用的组会全部被覆盖）。"""
    nf = META.get(gid, {}).get('n_f') or load_meta(gid)['n_f']
    cyc = _l1_grid.cyc_grid(nf)
    nb = cyc.size
    return dict(
        cyc=cyc,
        D=np.zeros(nb), risk=np.zeros(nb), eae=np.zeros(nb), est=np.zeros(nb),
        c0=0.0,
        strain=_l1_grid.dfos_mean_strain(gid, cyc, nf),
        peak=np.full(nb, np.nan),          # 无点级 AE 峰值 ⇒ 看板 AE 峰值线留空
    )
