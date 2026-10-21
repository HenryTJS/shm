# -*- coding: utf-8 -*-
"""已归档模块 `evaluate_l1_degree_v2` 的最小桩（仅用于回放旧 L1 看板导出器）。

同 `evaluate_l1_degree.py`：2026-10-07 已归档，但字节码仍在
`l1/__pycache__/export_dashboard_l1_v2.cpython-312.pyc`。

回放时该导出器只用 `group_series(gid) -> (cyc, strain)` 重建点级网格；
主曲线（HI_AE）由 `evaluate_l1_hi_ae.hi_of_group()` 现算。
"""
from __future__ import annotations

import numpy as np

from l1_meta import load_meta
import _l1_grid

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
          'L1-55', 'L1-56', 'L1-59', 'L1-60']


def group_series(gid, *_a, **_k):
    nf = load_meta(gid)['n_f']
    if not nf:
        return None
    cyc = _l1_grid.cyc_grid(nf)
    return cyc, _l1_grid.dfos_mean_strain(gid, cyc, nf)
