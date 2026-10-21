# -*- coding: utf-8 -*-
"""重建 L1 看板包时用的共享工具（**只服务归档字节码回放，不参与任何结论**）。

用途：`evaluate_l1_degree` / `evaluate_l1_degree_v2` 两个脚本 2026-10-07 已归档，
但其 `l1/__pycache__/*.pyc` 仍在。回放这些字节码时，它们需要：

  * `cyc`   —— 点级循环轴（旧口径：10 cycle/点）
  * `strain` —— 点级应变（喂给看板的 STRAIN 子面板）

这里用**现存且未归档**的模块重建这两样：
  * 循环轴按 `n_f` 铺 10 cycle 网格（看板 frameDt = STEP × 10 = 50 cycle/帧，与旧包一致）
  * 应变取 DFOS 每行**空间均值**，时间→循环用 `evaluate_l1_hi_ae._cycle_of`
    （C2 = DFOS 段锚；C1 = FBG 块锚），再线性插到循环轴

⚠️ 这里**不重建**任何 D(t)/风险量：L1 的在线口径已下线，主曲线一律用
`evaluate_l1_hi_ae.hi_of_group()` 的离线复评 HI。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CYCS_PER_BLK = 5000          # FBG 测量块 ≈5000 cycles
CYCS_PER_PT = 10             # 旧口径点间距（cycle）


def cyc_grid(nf):
    """点级循环轴：10 cycle/点，覆盖 [10, 10·nb]，nb = floor(n_f / 10)。"""
    nb = max(2, int(float(nf) // CYCS_PER_PT))
    return (np.arange(nb) + 1.0) * CYCS_PER_PT


def dfos_mean_strain(gid, cyc, nf):
    """DFOS 每行空间均值（µε）→ 插值到给定循环轴；缺数据返回全 NaN。"""
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变.csv')
    if not os.path.exists(fp):
        return np.full(np.shape(cyc), np.nan, float)
    df = pd.read_csv(fp, encoding='utf-8-sig')
    if df.shape[1] < 2 or not len(df):
        return np.full(np.shape(cyc), np.nan, float)
    t = df['timestamp'].to_numpy(float)
    M = df.iloc[:, 1:].to_numpy(float)
    fin = np.isfinite(M)
    cnt = fin.sum(axis=1)
    s = np.where(fin, M, 0.0).sum(axis=1)
    y = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    from evaluate_l1_hi_ae import _cycle_of       # 已存在的正式实现，按组自动选锚
    ac = np.asarray(_cycle_of(gid, t, nf), float)
    ok = np.isfinite(ac) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(np.shape(cyc), np.nan, float)
    x, yy = ac[ok], y[ok]
    o = np.argsort(x, kind='stable')
    x, yy = x[o], yy[o]
    ux, first = np.unique(x, return_index=True)   # np.interp 要求严格递增
    return np.interp(cyc, ux, yy[first])
