# -*- coding: utf-8 -*-
"""L1 看板数据包导出（统一入口）
=================================================================

把 L1 公开集**三批**试件的看板数据包写到 `dashboard/data/`，供 `dashboard/index.html`
加载。三批数据合并在本文件内，用 `--ds` 选分组：

| --ds   | 分组              | 试件                  | AE 格式            | 应变            | 主曲线         |
| ------ | ----------------- | --------------------- | ------------------ | --------------- | -------------- |
| `l1`   | L1 恒幅+FBG+DFOS | L1-03/04/05/09        | Vallen `.pridb`    | FBG 块级 + DFOS | D(t)           |
| `l1v2` | L1 恒幅+DFOS | L1-49 至 L1-60（9 组） | Vallen `.pridb`   | DFOS 段级       | 离线复评 HI_AE |
| `l1v3` | L1 变幅VA/谱载   | 新 14 组（变幅/谱载）  | PAC/Mistras `.DTA` | FBG(sm130)      | 离线复评 HI_hit |

⚠️ L1 恒幅+DFOS 与 L1 变幅VA/谱载 的主曲线是**离线复评（非因果）**：`unity01` 需要全寿命最大值，
这正是它不能在线用的原因。看板已把 `ds.name` / `pkg.mode` / `chans[engine].mode`
三处标注为「离线复评」，因果通道（`eae` / `est`）只作观察，不作阈值报警。

各批口径与推导链见 `docs/details.md` §12.8 至 §12.13 与 §32。

输出
----
    dashboard/data/{gid}.js           window.SHM_DATA["{gid}"]
    dashboard/data/index_l1.js        window.SHM_DATASETS["l1"]
    dashboard/data/index_l1v2.js      window.SHM_DATASETS["l1v2"]
    dashboard/data/index_l1v3.js      window.SHM_DATASETS["l1v3"]

用法
----
    python l1/export_dashboard.py --ds l1                 # L1 恒幅+FBG+DFOS 4 组
    python l1/export_dashboard.py --ds l1v2 L1-49 L1-55   # L1 恒幅+DFOS 指定组
    python l1/export_dashboard.py --ds l1v3               # L1 变幅VA/谱载全部可用组
    python l1/export_dashboard.py --ds all                # 三批全部
    python l1/export_dashboard.py --ds l1 --out <dir>     # 自定义输出目录

说明
----
本文件由 `l1/export_dashboard_l1.py`(588 行) / `export_dashboard_l1_v2.py`(461) /
`export_dashboard_l1_v3.py`(446) 合并而来：三段流水线的**函数体逐字保留**，只对
跨分组重名的常量/函数加了 `V1_` / `V2_` / `V3_` 前缀（逐字相同的最多一组去重保留），
公共脚手架（`write_js` / `main` / JS 包装 / 清单）只写一份。合并前后 `pack()` 的
返回值已逐字段比对一致，见 `docs/details.md` §33。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # <项目根>\l1
ROOT = HERE
PROJ = os.path.dirname(HERE)                               # 项目根
RES = os.path.join(HERE, 'results')
sys.path.insert(0, PROJ)                                   # shm
sys.path.insert(0, ROOT)                                   # l1_meta / evaluate_* 等同级模块
_CWD0 = os.getcwd()            # 调用者所在目录：--out 的相对路径按它解析
os.chdir(HERE)                                             # 与合并前的第一/二批一致

import evaluate_l1_degree as _deg
import evaluate_l1_degree_v2 as _v2
import evaluate_l1_dfos as _dfos
import evaluate_l1_hi_ae as _hiae
import reproduce_broer_l1 as _l4
import l1_time_align as ta
from l1_meta import load_meta
from shm.datasets import L1_CAMPAIGNS, campaign_members
os.makedirs(RES, exist_ok=True)



# ==================================================================
# 公共工具（三段流水线共用）
# ==================================================================
def write_js(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return os.path.getsize(path) / 1024.0


# ==================================================================
# L1 恒幅+FBG+DFOS（L1 恒幅+FBG+DFOS，L1-03/04/05/09）：FBG 块级 + DFOS 块级，Vallen .pridb
# ==================================================================
GAP_S = 300.0                 # FBG 测量块间隔阈值(s)
MIN_ROWS = 500                # 一个有效块的最少采样行
CYCS_PER_BLK = _deg.CYCS_PER_BLK
V1_CYCS_PER_PT = _deg.CYCS_PER_PT
PTS_PER_BLK = _deg.PTS_PER_BLK
V1_STEP = 5                      # 降采样步长(点): 1 帧 = 5 点 = 50 cycle
LEVELS = [0.25, 0.55, 0.85]
V1_AE_EMPTY = -99999
PARAMS = {'rise': 0.05}       # = run.py 的 l1/degree 推荐配置
FUSION = 'max'


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def _i100(x):
    return int(round(float(x) * 100.0))


def _i1000(x):
    return int(round(float(x) * 1000.0))


def fbg_strain_cols(df):
    """全部 FBG 应变通道(表头自适应); 排除 FBG_A*/FBG_B* 波长列。"""
    return [c for c in df.columns
            if (c.startswith('fbg') and c[3:].isdigit())
            or (c.startswith('b') and c[1:].isdigit())]


def fill_nan(v):
    """线性插值填补 NaN; 全 NaN 返回 None。"""
    v = np.asarray(v, float).copy()
    m = np.isfinite(v)
    if m.sum() == 0:
        return None
    if m.sum() < v.size:
        idx = np.arange(v.size)
        v = np.interp(idx, idx[m], v[m])
    return v


def fbg_block_channels(gid):
    """每个 FBG 测量块的**逐通道**应变均值 → (块 cycle, 通道名, {通道: 值数组})。

    抛掉全 NaN 或有效块不足半数的通道(如 L1 里常年无数据的 fbg2-5 / 波长列)。
    """
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    cols = fbg_strain_cols(df)
    gap = np.where(np.diff(t) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(t)]])
    cyc, vals = [], {c: [] for c in cols}
    k = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < MIN_ROWS:
            continue
        cyc.append((k + 0.5) * CYCS_PER_BLK)
        for c in cols:
            seg = df[c].to_numpy(float)[a:b]
            vals[c].append(float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan)
        k += 1
    cyc = np.asarray(cyc, float)
    keep, out = [], {}
    for c in cols:
        v = fill_nan(vals[c])
        if v is None or np.isfinite(v).sum() < max(2, 0.5 * len(cyc)):
            continue                                   # 无效通道 → 不展示
        keep.append(c)
        out[c] = v
    return cyc, keep, out


def ae_series(gid):
    """AE 事件(按 time 去重取 max energy) → (time, energy, 通道数)。"""
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig', usecols=['time', 'energy', 'channel'])
    n_ch = int(df['channel'].nunique())
    g = df.groupby('time', sort=True)['energy'].max()
    return g.index.to_numpy(float), g.to_numpy(float), n_ch


def dfos_block_series(gid):
    """DFOS 块级序列(局部峰 / rmse / HI / 空间点数) —— 复用 evaluate_l1_dfos 的缓存。"""
    cache = os.path.join(_deg.RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(cache):
        _dfos.analyze(gid)
    z = np.load(cache, allow_pickle=True)
    pos = np.asarray(z['pos'], float)
    return (np.asarray(z['cyc'], float), np.asarray(z['local'], float),
            np.asarray(z['rmse'], float), np.asarray(z['hi'], float), int(pos.size))


def dfos_profiles(gid, max_pos=420):
    """DFOS **逐块空间分布**(块内逐点中位曲线) → (位置, prof[nblk,npos], 块 cycle, 修复点数)。

    步骤: 全分辨率建块分布 → 沿块轴补 NaN → **共用 `evaluate_l1_dfos.despike_row` 去尖峰**
    (与 §4.2.4 的 DFOS HI 指标同一判据) → 空间降采样至 ~max_pos 点(控制包体积)。
    """
    td, pos, M = _dfos.load_dfos(gid)
    blk = _dfos.load_fbg_blocks(gid)
    nblk = len(blk)
    row_blk = _dfos.distribute_rows_to_blocks(td, blk)
    idx = np.arange(nblk)
    prof = np.full((nblk, pos.size), np.nan)
    n_fix = 0
    for k in range(nblk):
        pr = _dfos.block_profile(M, row_blk == k)
        if pr is None:
            continue
        pr, nfx = _dfos.despike_row(pr)          # 先去尖峰(与 HI 指标同判据/同顺序)
        n_fix += nfx
        prof[k] = pr
    # 再沿块轴补 NaN —— 仅为显示连续; 必须在去尖峰之后(否则插值会造出假尖峰)
    for j in range(prof.shape[1]):
        col = prof[:, j]
        m = np.isfinite(col)
        prof[:, j] = np.interp(idx, idx[m], col[m]) if m.any() else 0.0
    step = max(1, int(np.ceil(pos.size / float(max_pos))))
    sel = np.arange(0, pos.size, step)
    return pos[sel], prof[:, sel], (idx + 0.5) * CYCS_PER_BLK, n_fix


def fbg_block_p2p(gid):
    """FBG 逐块的**逐通道应变峰峰值** → (块 cycle, {通道: 值数组}）。

    与 `fbg_block_channels` 用同一套分块规则（时间间隔 > GAP_S、块内 >= MIN_ROWS 行），
    但取块内**峰峰值**而非均值 —— 均值含应变基线漂移与温度项，不适合做形状指标；
    L1 变幅VA/谱载的 `shape_rob25` 用的也是峰峰值口径。
    """
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    cols = fbg_strain_cols(df)
    gap = np.where(np.diff(t) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(t)]])
    cyc, vals = [], {c: [] for c in cols}
    k = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < MIN_ROWS:
            continue
        cyc.append((k + 0.5) * CYCS_PER_BLK)
        for c in cols:
            seg = df[c].to_numpy(float)[a:b]
            v = seg[np.isfinite(seg)]
            vals[c].append(float(v.max() - v.min()) if v.size else np.nan)
        k += 1
    cyc = np.asarray(cyc, float)
    out = {}
    for c in cols:
        v = np.asarray(vals[c], float)
        if np.isfinite(v).sum() < max(2, 0.5 * len(cyc)):
            continue
        out[c] = fill_nan(v)
    return cyc, out


def shape_rob25_of(P):
    """块级 10 通道剖面（按块归一化）与**前 25% 块中位剖面**的 L1 距离。

    与 `fbg_tools.py profile` 的 `shape_rob25` **同定义**（只是分辨率从加载窗换成 FBG 测量块）：
    先在块内按通道求和归一化（消掉整体缩放），再取 L1 形状距离。
    """
    P = np.asarray(P, float)
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    P = P / s
    n = len(P)
    k25 = max(1, int(round(n * 0.25)))
    ref = np.median(P[:k25], axis=0)
    return np.abs(P - ref).sum(axis=1)


def fano_of_ae(gid, nf, bin_s=60.0, win=60):
    """AE 簇状性 Fano = 滑窗内 var(n)/mean(n)，60 s 分箱 + 1 h（60 箱）滑窗。

    与L1 变幅VA/谱载（`ae_burst.py` 及 `export_dashboard.py --ds l1v3`）**同口径**。
    返回 (cycle, fano)；横轴用 `_hiae._cycle_of` 把箱时间映射到 cycle，
    使簇状性可与主曲线同轴比较。
    """
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if not os.path.exists(fp):
        return None
    t = pd.read_csv(fp, encoding='utf-8-sig',
                    usecols=['time'])['time'].to_numpy(float)
    if t.size == 0:
        return None
    t0, t1 = float(t.min()), float(t.max())
    nb = max(1, int(np.ceil((t1 - t0) / bin_s)))
    cnt, _ = np.histogram(t, bins=nb, range=(t0, t0 + nb * bin_s))
    x = cnt.astype(float)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    fano = np.full(nb, np.nan)
    for i in range(win, nb + 1):
        s1 = c1[i] - c1[i - win]
        s2 = c2[i] - c2[i - win]
        mu = s1 / win
        if mu > 0:
            fano[i - 1] = (s2 / win - mu * mu) / mu
    ok = np.isfinite(fano)
    if ok.sum() < 10:
        return None
    tb = t0 + (np.arange(nb) + 0.5) * bin_s
    # `_cycle_of` 是 evaluate_l1_hi_ae 的内部函数（本仓库内复用，避免另造一套锚）
    cycb = np.asarray(_hiae._cycle_of(gid, tb[ok], nf), float)
    good = np.isfinite(cycb)
    if good.sum() < 10:
        return None
    return cycb[good], fano[ok][good]


def V1_levels_from(D):
    """D → 单调分级(在线语义: 级别只升不降)。"""
    lv = np.zeros(len(D), dtype=np.int8)
    cur = 0
    for i, v in enumerate(D):
        while cur < len(LEVELS) and v >= LEVELS[cur]:
            cur += 1
        lv[i] = cur
    return lv


def to_block_frames(arr, cyc_frame):
    """块级数组 → 逐帧数组(按 cycle 就近取块); NaN → 0。"""
    if len(arr) == 0:
        return np.zeros(len(cyc_frame))
    a = np.nan_to_num(np.asarray(arr, float), nan=0.0)
    b = np.clip((cyc_frame / CYCS_PER_BLK).astype(int), 0, len(a) - 1)
    return a[b]


# ------------------------------------------------------------------
# 打包一组
# ------------------------------------------------------------------
# 每块(EXT_BLOCK_PTS=500 点) 对应的帧数 = 500 / V1_STEP
V1_BLK_FRAMES = 500 // V1_STEP


def V1_de_step(arr, blk=V1_BLK_FRAMES):
    """块级阶跃 → 块内线性插值（消除 D(t)/risk 的阶梯观感）。

    OnlineDamageIndex 每 EXT_BLOCK_PTS=500 点才结算一次块级统计, 故 D/risk/e_ae/e_st
    在时间轴上呈"每块一个平台"的阶梯。此处把每块内部线性铺开, 使曲线连续:
      - 块末值保持不变（只把"块末那次跳变"摊回整块内）;
      - 阈值跨越时刻最多提前不到一个块。
    仅用于展示; 分级 lv 由插值后的 D 重算, 保持内部一致。
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n <= blk:
        return a
    out = a.copy()
    for s in range(blk, n, blk):
        e = min(s + blk, n)
        out[s:e] = np.linspace(a[s - 1], a[s], e - s + 1)[1:]
    return out


def pack_v1(gid):
    print(f'\n=== 导出 {gid} ===')
    nf = _deg.META[gid]['n_f']

    # --- 网格与应变/AE 逐点序列仍复用原口径（D(t) 的运行只用来取网格与 strain/peak）---
    r = _deg.run_group(gid, dict(PARAMS), baseline=True, strain_ev=True, fusion=FUSION)
    cyc, nb = r['cyc'], len(r['cyc'])
    nfr = int(np.ceil(nb / V1_STEP))

    # --- 主指标：改用与第二/三批**同源**的文献口径 HI_hit（AE 累积 hits 自归一化）---
    # 为何不再用 D(t)：`evaluate_l1_hi_ae.py` 头部已论证 —— L1 是「冲击后疲劳」，
    # 0 cycle 即带 BVID ⇒ D 的 e_ae 一抬头就冲上 0.85，与真实寿命无关；
    # 该项目结论是「**仅保留离线复评 `evaluate_l1_hi_ae.py`**」。
    # 实测L1 恒幅+FBG+DFOS 4 组 cum_hits 的 t85 中位 92.6%，与第二/三批（92.1% / 93.9%）一致。
    hi = _hiae.hi_of_group(gid)
    if hi is None:
        print(f'  [warn] {gid} HI_hit 不可用 → 回退 D(t)')
        D = np.asarray(r['D'], float)
        risk_pt = np.asarray(r['risk'], float)
        eae_pt = np.asarray(r['eae'], float)
        est_pt = np.asarray(r['est'], float)
        c0_pt = float(r['c0'])
        main_name = '损伤度 D(t)（回退）'
    else:
        hc = np.asarray(hi['cyc'], float)
        D = np.interp(cyc, hc, _hiae.unity01(hi['cum_hits']))
        risk_pt = D.copy()
        # 因果证据 1：AE 活动度（5000 cycle 窗内事件量）
        eae_pt = np.interp(cyc, hc, _hiae.unity01(hi['win10_hits']))
        # 因果证据 2：FBG 剖面漂移（块级，与L1 变幅VA/谱载 shape_rob25 同定义）
        bcyc2, prof2 = fbg_block_p2p(gid)
        if prof2:
            names = list(prof2.keys())
            P = np.column_stack([prof2[c] for c in names])
            est_pt = np.interp(cyc, bcyc2, _hiae.unity01(shape_rob25_of(P)))
        else:
            est_pt = np.zeros_like(cyc)
        c0_pt = 0.0                       # c0 是 D(t) 专用锚，换口径后不再成立
        main_name = '离线复评 HI_hit（非在线）'

    # --- FBG 逐通道(块级 → 逐点插值) + AE 事件级 ---
    bcyc, fo_cols, fo_blk = fbg_block_channels(gid)
    strain = np.nan_to_num(np.asarray(r['strain'], float), nan=0.0)
    fo_pt = {c: np.interp(cyc, bcyc, v, left=v[0], right=v[-1]) for c, v in fo_blk.items()}
    tae, eae_ev, n_ae_ch = ae_series(gid)
    dfos_cyc, dfos_local, dfos_rmse, dfos_hi, n_pos = dfos_block_series(gid)
    dpos, dprof, dcyc, n_spike = dfos_profiles(gid)

    # --- 与L1 变幅VA/谱载同源的第三路证据：Fano 簇状性（AE 事件 60 s 分箱 + 1 h 滑窗）---
    fa = fano_of_ae(gid, nf)
    fano_pt = (np.interp(cyc, fa[0], _hiae.unity01(fa[1])) if fa is not None
               else np.zeros_like(cyc))
    fano_raw = None
    if fa is not None:
        fano_raw = (round(float(np.min(fa[1])), 1), round(float(np.max(fa[1])), 1))

    # --- 逐点 → 逐帧(取每帧最后一个点, 与主样本导出器同口径) ---
    j = np.minimum((np.arange(nfr) + 1) * V1_STEP - 1, nb - 1)
    cyc_f = cyc[j]

    D_f = D[j]
    risk_f = risk_pt[j]
    eae_f = eae_pt[j]
    est_f = est_pt[j]
    fano_f = fano_pt[j]
    st_f = strain[j]
    lv_f = V1_levels_from(D)[j]

    peak_pt = np.asarray(r['peak'], float)
    peak_f = peak_pt[j]
    aen = np.zeros(nfr, dtype=np.int32)
    ael = np.full(nfr, V1_AE_EMPTY, dtype=np.int32)
    for f in range(nfr):
        lo = f * V1_STEP
        hi = min(lo + V1_STEP, nb)
        seg = peak_pt[lo:hi]
        m = seg > 0
        if m.any():
            aen[f] = int(m.sum())
            ael[f] = _i1000(np.log10(float(seg[m].max())))
    fo_f = {c: v[j] for c, v in fo_pt.items()}
    dl_f = to_block_frames(dfos_local, cyc_f)
    dr_f = to_block_frames(dfos_rmse, cyc_f)
    dh_f = to_block_frames(dfos_hi, cyc_f)
    nblk_p = dprof.shape[0]
    blk_of = np.clip((cyc_f / CYCS_PER_BLK).astype(int), 0, nblk_p - 1)

    # --- 摘要 ---
    def first_ge_pct(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    # --- 论文 HI_F（离线参考曲线；方案A：与在线 D(t) **并列**展示，不替换）---
    # HI_F = 0.5·HI_AE + 0.5·HI_OF（Broer 2021 式 7，两项各 unity01 归一化到 [0,1]）。
    # ⚠️ 归一化需全寿命最大值 ⇒ **非因果/离线**，仅作参考基线。
    # 叠加的意义：既保留在线因果 D(t) 作为主指标，又把「与论文口径的差异」量化且可见。
    hiF_f, hi85_pct = None, None
    try:
        r4 = _l4.level4(gid)
        c4 = np.asarray(r4['cyc'], float)
        h4 = np.asarray(r4['HI_F'], float)
        o = np.argsort(c4)
        hiF_f = np.interp(cyc_f, c4[o], h4[o])
        ix = np.where(hiF_f >= 0.85)[0]
        hi85_pct = round(float(cyc_f[ix[0]]) / nf * 100.0, 1) if len(ix) else None
    except Exception as e:                       # noqa: BLE001
        print(f'  [warn] {gid} HI_F 参考曲线不可用: {e}')
    d85_pct = first_ge_pct(0.85)
    hi_lead = (round(hi85_pct - d85_pct, 1)      # 正 = D(t) 早于论文 HI_F
               if (d85_pct is not None and hi85_pct is not None) else None)

    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge_pct(0.25), 't55': first_ge_pct(0.55), 't85': d85_pct,
        'b2': None,                                  # L1 无弱标签 b2(前端自动隐藏)
        'b3': 100.0,                                 # 失效锚 = n_f
        'c0Pct': None,                               # c0 是 D(t) 专用锚，换口径后不再成立
        'refs': [{'pct': round(c / nf * 100.0, 1), 'label': lab}
                 for lab, c in _deg.META[gid]['refs']],
        'aeEvents': int(aen.sum()),
        'nFo': len(fo_cols),
        'nDfos': n_pos,
        # --- 论文 HI_F 离线参考（方案A）---
        'hiF85': hi85_pct,        # HI_F 达 0.85 的寿命% (None = 未达)
        'hiLeadPt': hi_lead,      # HI_hit 相对论文 HI_F 的提前量(百分点, 正 = 更早)
        'mainName': main_name,
        'fanoRaw': fano_raw,      # Fano 原始量程(显示用的 0至1 缩放是离线归一)
    }

    # --- 块级阶跃 → 块内线性插值(展示连续化; 块级口径不变) ---
    D_f = V1_de_step(D_f)
    risk_f = V1_de_step(risk_f)
    eae_f = V1_de_step(eae_f)
    est_f = V1_de_step(est_f)
    fano_f = V1_de_step(fano_f)
    lv_f = V1_levels_from(D_f)

    # --- 分级升级事件(逐帧, 在线语义) ---
    warn = []
    last = 0
    for f in range(nfr):
        lv = int(lv_f[f])
        if lv > last:
            for g in range(last + 1, lv + 1):
                warn.append({'f': f, 't': round(float(cyc_f[f]), 1), 'lv': g})
            last = lv

    # --- AE log 峰值纵轴范围(自动) ---
    fin = ael[ael != V1_AE_EMPTY]
    if fin.size:
        ax_ael = [round(float(fin.min()) / 1000.0 - 0.3, 2),
                  round(float(fin.max()) / 1000.0 + 0.3, 2)]
    else:
        ax_ael = [-1.0, 1.0]
    rate_hi = max(2.0, float(aen.max()) * 1.2)

    chans = [
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流', 'n': str(n_ae_ch), 'unit': '通道'},
        {'key': 'fo', 'name': '光纤光栅 FBG', 'mode': '5000 cyc/块', 'n': str(len(fo_cols)), 'unit': '通道'},
        {'key': 'dfos', 'name': '分布式应变 DFOS', 'mode': '5000 cyc/块', 'n': str(n_pos), 'unit': '点'},
        {'key': 'engine', 'name': '离线复评 HI_hit', 'mode': 'AE 累积·需全寿命归一',
         'n': None, 'unit': ''},
    ]

    # --- 证据通道声明（与L1 变幅VA/谱载同结构；前端 EVIDENCE 面板由它渲染）---
    indics = [
        {'key': 'e_ae', 'name': 'e_ae AE 活动度（5000 cycle 窗）',
         'color': 'rgba(255,208,138,.9)',
         'series': [_i1000(x) for x in eae_f]},
        {'key': 'e_st', 'name': 'e_st FBG 剖面漂移（块级，在线）',
         'color': 'rgba(0,227,154,.85)',
         'series': [_i1000(x) for x in est_f]},
        {'key': 'e_fano', 'name': 'e_fano Fano 簇状性（1 h 窗 var/mean）',
         'color': 'rgba(176,124,255,.9)',
         'series': [_i1000(x) for x in fano_f]},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳 (BVID 后)',
        'mode': '多源同步 · 5000 cycle/块 · 主曲线 = 离线复评 HI_hit',
        'indexName': main_name,
        'trendName': 'HI_hit 趋势 / 阈值 0.25 · 0.55 · 0.85',
        'labels': {'unit': 'HI_hit (0~1)', 'legend': 'HI_hit',
                   'margin': '1 − HI_hit', 'thrName': 'HI_hit',
                   'peak': '峰值 HI_hit',
                   'foot': 'HI_hit ∈ [0,1] · 阈值 0.25 / 0.55 / 0.85'},
        'n': int(nb), 'nfr': int(nfr), 'step': V1_STEP,
        'frameDt': float(V1_STEP * V1_CYCS_PER_PT), 'dt': float(V1_STEP * V1_CYCS_PER_PT),
        'dur': float(nf), 'c0': c0_pt,
        'foCols': list(fo_cols), 'chans': chans, 'indics': indics,
        'meta': meta, 'warn': warn,
        # --- 波形数组(长度均 = nfr) ---
        'D': [_i1000(x) for x in D_f],
        'hiF': ([_i1000(x) for x in hiF_f] if hiF_f is not None else None),
        'risk': [_i1000(x) for x in risk_f],
        'eae': [_i1000(x) for x in eae_f],
        'est': [_i1000(x) for x in est_f],
        'lv': [int(x) for x in lv_f],
        'st': [_i100(x) for x in st_f],
        'ael': [int(x) for x in ael],
        'aen': [int(x) for x in aen],
        't': [round(float(x), 1) for x in cyc_f],
        'fo': {c: [_i100(x) for x in v] for c, v in fo_f.items()},
        'dfos': {
            'local': [_i100(x) for x in dl_f],
            'rmse': [_i100(x) for x in dr_f],
            'hi': [_i1000(x) for x in dh_f],
            'nPos': n_pos,
            # --- 空间分布(逐块中位曲线) ---
            'nblk': int(nblk_p), 'npos': int(dprof.shape[1]),
            'pos': [round(float(x), 1) for x in dpos],
            'cyc': [round(float(x), 1) for x in dcyc],
            'base': [_i100(x) for x in dprof[0]],
            'prof': [_i100(x) for x in dprof.reshape(-1)],
            'blkOf': [int(x) for x in blk_of],
            'cycPerBlk': CYCS_PER_BLK,
            'spikeN': int(n_spike),          # 被空间去尖峰修复的采样点数
        },
        'ax': {'ael': ax_ael, 'rate': round(rate_hi, 1)},
    }
    return pkg

# ==================================================================
# L1 恒幅+DFOS（L1 恒幅+DFOS，L1-49…L1-60，无 FBG）：DFOS 段级，Vallen .pridb
# ==================================================================
V2_GROUPS = list(_v2.GROUPS)
V2_STEP = 5                      # 降采样: 5 点(10-cycle 网格) = 50 cycle/帧 —— 与 L1 恒幅+FBG+DFOS 一致
V2_CYCS_PER_PT = 10              # v2 网格步长(cycle/点), 见 evaluate_l1_degree_v2.group_series
FRAME_CYC = V2_STEP * V2_CYCS_PER_PT     # 每帧代表的 cycle 数 = 50
V2_AE_EMPTY = -99999             # ael 空值标记(与前端约定一致)
MAX_POS = 420                 # DFOS 空间降采样点数(与 L1 恒幅+FBG+DFOS 一致, 控制包体积)
MAX_SEG = 200                 # DFOS 段轴降采样上限(L1 恒幅+DFOS 每 500 cycle 一段, 可上千段)
V2_BLK_FRAMES = 500 // V2_STEP      # 每块(EXT_BLOCK_PTS=500 点) 对应的帧数


def V2_de_step(arr, blk=V2_BLK_FRAMES):
    """块级阶跃 → 块内线性插值（消除 D(t)/risk 的阶梯观感）。

    OnlineDamageIndex 每 EXT_BLOCK_PTS=500 点才结算一次块级统计, 故 D/risk/e_ae/e_st
    在时间轴上呈"每块一个平台"的阶梯。此处把每块内部线性铺开, 使曲线连续:
      - 块末值保持不变（只把"块末那次跳变"摊回整块内）;
      - 阈值跨越时刻最多提前不到一个块。
    仅用于展示; 分级 lv 由插值后的 D 重算, 保持内部一致。
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n <= blk:
        return a
    out = a.copy()
    for s in range(blk, n, blk):
        e = min(s + blk, n)
        out[s:e] = np.linspace(a[s - 1], a[s], e - s + 1)[1:]
    return out


# ------------------------------------------------------------------
# 基础工具（与L1 恒幅+FBG+DFOS export_dashboard.py --ds l1 同口径）
# ------------------------------------------------------------------




def V2_unity01(x):
    """unity 归一化到 [0,1]（文献口径式1）。⚠️ 用全局 min/max → **非因果**。"""
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)


def hi_ae_series(gid, cyc_grid):
    """离线复评主指标 HI_AE，插值到给定 cycle 网格；缺 AE 返回 None。

    HI_AE = V2_unity01(AE 累积事件数)（500-cycle 箱）。详见模块 docstring 与 docs §12.8。
    """
    s = _hiae.hi_of_group(gid)
    if s is None:
        return None
    return np.interp(cyc_grid, s['cyc'], V2_unity01(s['cum_hits']), left=0.0, right=1.0)


def ae_rate_series(gid, cyc_grid):
    """证据通道 eae：V2_unity01(最近 5000 cycle 的 AE 事件数)（**因果**）。"""
    s = _hiae.hi_of_group(gid)
    if s is None:
        return None
    v = V2_unity01(_hiae._roll_sum(s['hit_bin'], 10))
    return np.interp(cyc_grid, s['cyc'], v, left=0.0, right=1.0)


def V2_levels_from(D):
    """D → 单调分级(在线语义: 级别只升不降)。"""
    lv = np.zeros(len(D), dtype=np.int8)
    cur = 0
    for i, v in enumerate(D):
        while cur < len(LEVELS) and v >= LEVELS[cur]:
            cur += 1
        lv[i] = cur
    return lv


def to_seg_frames(arr, seg_cyc, cyc_frame):
    """段级数组 → 逐帧数组（取"cycle ≤ 当前帧"的最近一段）；NaN → 0。"""
    if arr is None or len(arr) == 0:
        return np.zeros(len(cyc_frame))
    a = np.nan_to_num(np.asarray(arr, float), nan=0.0)
    seg = np.asarray(seg_cyc, float)
    b = np.clip(np.searchsorted(seg, cyc_frame, side='right') - 1, 0, len(a) - 1)
    return a[b]


# ------------------------------------------------------------------
# 数据提取
# ------------------------------------------------------------------
def dfos_seg_series(gid):
    """段级 local / rmse / HI（复用 evaluate_l1_dfos 的缓存, 缺则现算）。"""
    cache = os.path.join(RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(cache):
        _dfos.analyze(gid)
    z = np.load(cache, allow_pickle=True)
    pos = np.asarray(z['pos'], float)
    return (np.asarray(z['cyc'], float), np.asarray(z['local'], float),
            np.asarray(z['rmse'], float), np.asarray(z['hi'], float), int(pos.size))


def dfos_profiles_v2(gid, max_pos=MAX_POS, max_seg=MAX_SEG):
    """段级空间分布 → (位置, prof[nseg,npos], 段 cycle, 去尖峰点数)。

    L1 恒幅+DFOS 组**段 = 块**：`step0.py --batch c2` 已把每段压成"段内逐点中位数"一行，
    故直接取该矩阵（只保留 AI 段, 排除冲击前 BI 段），
    逐段用与 §4.2.4 同一判据去尖峰，再沿段轴补 NaN（仅为显示连续），
    最后**段轴**降采样到 max_seg、**空间轴**降采样到 max_pos 点。
    """
    _t, pos, M = _dfos.load_dfos(gid)
    nf = load_meta(gid)['n_f']
    cyc_all, keep = ta.dfos_cycles(gid, nf, include_bi=True)
    if cyc_all.size != M.shape[0]:
        print(f'  [{gid}] 段数({cyc_all.size}) ≠ DFOS 行数({M.shape[0]}) → 跳过空间分布')
        return None
    M = M[keep]
    cyc_all = np.asarray(cyc_all, float)[keep]
    ai = cyc_all >= 0                                  # 只用冲击后(AI)段
    M, cyc_ai = M[ai], cyc_all[ai]
    if M.shape[0] < 2:
        return None
    nseg = M.shape[0]
    prof = np.full(M.shape, np.nan)
    n_fix = 0
    for k in range(nseg):
        prof[k], nfx = _dfos.despike_row(M[k])
        n_fix += nfx
    idx = np.arange(nseg)
    for jj in range(prof.shape[1]):
        col = prof[:, jj]
        m = np.isfinite(col)
        prof[:, jj] = np.interp(idx, idx[m], col[m]) if m.any() else 0.0
    # 段轴降采样 —— L1 恒幅+DFOS段数可达上千（每 500 cycle 一段），全存既臃肿、热图也无法分辨
    if nseg > max_seg:
        ksel = np.unique(np.linspace(0, nseg - 1, max_seg).astype(int))
        prof = prof[ksel]
        cyc_ai = cyc_ai[ksel]
    # 空间轴降采样
    pstep = max(1, int(np.ceil(pos.size / float(max_pos))))
    psel = np.arange(0, pos.size, pstep)
    return pos[psel], prof[:, psel], cyc_ai, int(n_fix)


# ------------------------------------------------------------------
# 打包一组
# ------------------------------------------------------------------
def pack_v2(gid):
    print(f'\n=== 导出 {gid} ===')
    nf = load_meta(gid)['n_f']

    # --- 主指标：离线复评 HI_AE（docs §12.8）---
    #   ⚠️ 非因果（V2_unity01 需全寿命最大值）→ 本数据集定位为「离线复评」，界面已标注。
    #   `group_series` 仍用于取 10-cycle 网格与 DFOS 脚部应变（st 通道）。
    r = _v2.group_series(gid)
    if r is None:
        print(f'  [{gid}] 缺数据 → 跳过')
        return None
    cyc, strain = r[0], r[1]
    nb = len(cyc)
    nfr = int(np.ceil(nb / V2_STEP))

    D = hi_ae_series(gid, cyc)                  # 主曲线 = 离线复评 HI_AE
    if D is None:
        print(f'  [{gid}] 缺 AE → 跳过')
        return None
    risk = D.copy()                             # 前端分级/发光逻辑沿用 risk
    eae = ae_rate_series(gid, cyc)              # 因果证据：AE 活动度（5000 cycle 窗）
    if eae is None:
        eae = np.zeros(nb)
    # 因果证据：DFOS 空间脱粘特征（段级，下面 dfos_seg_series 取到后填）
    est = None

    # --- AE（1 s bin, 经 markers 墙钟映射到 cycle）---
    ae_cyc = np.zeros(0)
    ae_e = np.zeros(0)
    ae_n = np.zeros(0)
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if os.path.exists(fp):
        ae = pd.read_csv(fp, encoding='utf-8-sig')
        if len(ae):
            ae_cyc = np.asarray(ta.time_to_cycle(gid, ae['time'].to_numpy(float), nf,
                                                 clip_out_of_life=True), float)
            ae_e = np.sqrt(np.maximum(ae['energy'].to_numpy(float), 0.0))
            ae_n = ae['n_hits'].to_numpy(float) if 'n_hits' in ae.columns \
                else np.ones(len(ae), float)
            ok = np.isfinite(ae_cyc)          # 剔除冲击前(BI)事件, 与 D(t) 同口径
            ae_cyc, ae_e, ae_n = ae_cyc[ok], ae_e[ok], ae_n[ok]

    # --- 段级 DFOS 指标 + 空间分布 ---
    dfos_cyc, dfos_local, dfos_rmse, dfos_hi, n_pos = dfos_seg_series(gid)
    profs = dfos_profiles_v2(gid)
    if profs is None:
        dpos = np.zeros(0)
        dprof = np.zeros((1, 1))
        dcyc = np.zeros(1)
        n_spike = 0
    else:
        dpos, dprof, dcyc, n_spike = profs

    # --- 逐点 → 逐帧（取每帧末点, 与L1 恒幅+FBG+DFOS同口径）---
    j = np.minimum((np.arange(nfr) + 1) * V2_STEP - 1, nb - 1)
    cyc_f = cyc[j]
    D_f, risk_f = D[j], risk[j]
    eae_f = eae[j]
    est_f = to_seg_frames(V2_unity01(dfos_local) if dfos_local is not None else None,
                          dfos_cyc, cyc_f)
    st_f = strain[j]
    lv_f = V2_levels_from(D)[j]

    # --- 帧内 AE 聚合: 事件数求和 + log 峰值取最大 ---
    aen = np.zeros(nfr, dtype=np.int32)
    ael = np.full(nfr, V2_AE_EMPTY, dtype=np.int32)
    if ae_cyc.size:
        order = np.argsort(ae_cyc)
        ac, ae_s, an_s = ae_cyc[order], ae_e[order], ae_n[order]
        lo = np.searchsorted(ac, cyc_f - FRAME_CYC / 2.0, side='left')
        hi = np.searchsorted(ac, cyc_f + FRAME_CYC / 2.0, side='right')
        for f in range(nfr):
            if hi[f] > lo[f]:
                aen[f] = int(round(float(an_s[lo[f]:hi[f]].sum())))
                pk = float(ae_s[lo[f]:hi[f]].max())
                if pk > 0:
                    ael[f] = _i1000(np.log10(pk))

    dl_f = to_seg_frames(dfos_local, dfos_cyc, cyc_f)
    dr_f = to_seg_frames(dfos_rmse, dfos_cyc, cyc_f)
    dh_f = to_seg_frames(dfos_hi, dfos_cyc, cyc_f)
    nblk_p = dprof.shape[0]
    blk_of = np.clip(np.searchsorted(np.asarray(dcyc, float), cyc_f, side='right') - 1,
                     0, nblk_p - 1) if nblk_p > 1 else np.zeros(nfr, int)

    # --- 摘要 ---
    def first_ge_pct(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge_pct(0.25), 't55': first_ge_pct(0.55), 't85': first_ge_pct(0.85),
        'b2': None,                       # L1 恒幅+DFOS 无弱标签 b2 → 前端自动隐藏
        'b3': 100.0,                      # 失效锚 = n_f
        'c0Pct': None,                    # L1 恒幅+DFOS 不做基线重定义 → 无 c0 锚
        'refs': [],                       # 无论文检测点(仅 03/04/05 收录于 Broer 2021)
        'aeEvents': int(aen.sum()),
        'nFo': 0,
        'nDfos': int(n_pos),
    }

    # --- 块级阶跃 → 块内线性插值(展示连续化) ---
    # 注: 主曲线 HI_AE 已由 np.interp 插值到 10-cycle 网格（无块级台阶），无需 V2_de_step。
    lv_f = V2_levels_from(D_f)

    # --- 分级升级事件(逐帧, 在线语义) ---
    warn = []
    last = 0
    for f in range(nfr):
        lv = int(lv_f[f])
        if lv > last:
            for g in range(last + 1, lv + 1):
                warn.append({'f': f, 't': round(float(cyc_f[f]), 1), 'lv': g})
            last = lv

    # --- AE log 峰值纵轴范围(自动) ---
    fin = ael[ael != V2_AE_EMPTY]
    if fin.size:
        ax_ael = [round(float(fin.min()) / 1000.0 - 0.3, 2),
                  round(float(fin.max()) / 1000.0 + 0.3, 2)]
    else:
        ax_ael = [-1.0, 1.0]
    rate_hi = max(2.0, float(aen.max()) * 1.2)

    chans = [                        # 前端按此渲染通道健康表(无 FBG 行)
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流(1 s bin)', 'n': '4', 'unit': '通道'},
        {'key': 'dfos', 'name': '分布式应变 DFOS', 'mode': '500 cycle/段',
         'n': str(n_pos), 'unit': '点'},
        {'key': 'engine', 'name': '离线复评 HI_AE', 'mode': 'AE 累积·需全寿命归一',
         'n': None, 'unit': ''},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1v2', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳 (BVID 后, 无 FBG)',
        'mode': 'AE + DFOS · 离线复评（非在线）',
        # 主指标名称（覆盖前端默认的“损伤度 D(t)”）
        'indexName': '离线复评 HI_AE（非在线）',
        'trendName': 'HI_AE 趋势 / 阈值 0.25 · 0.55 · 0.85',
        'labels': {'unit': 'HI_AE (0~1)', 'legend': 'HI_AE',
                   'margin': '1 − HI_AE', 'thrName': 'HI_AE',
                   'peak': '峰值 HI_AE',
                   'foot': 'HI_AE ∈ [0,1] · 阈值 0.25 / 0.55 / 0.85'},
        'n': int(nb), 'nfr': int(nfr), 'step': V2_STEP,
        'frameDt': float(FRAME_CYC), 'dt': float(FRAME_CYC),
        'dur': float(nf), 'c0': 0.0,
        'foCols': [],                 # 无 FBG → 前端 FBG 面板显示"未接入"
        'chans': chans,
        'meta': meta, 'warn': warn,
        # --- 波形数组(长度均 = nfr) ---
        'D': [_i1000(x) for x in D_f],
        'risk': [_i1000(x) for x in risk_f],
        'eae': [_i1000(x) for x in eae_f],
        'est': [_i1000(x) for x in est_f],
        'lv': [int(x) for x in lv_f],
        'st': [_i100(x) for x in st_f],
        'ael': [int(x) for x in ael],
        'aen': [int(x) for x in aen],
        't': [round(float(x), 1) for x in cyc_f],
        'fo': {},
        'dfos': {
            'local': [_i100(x) for x in dl_f],
            'rmse': [_i100(x) for x in dr_f],
            'hi': [_i1000(x) for x in dh_f],
            'nPos': int(n_pos),
            # --- 空间分布(逐段中位曲线) ---
            'nblk': int(nblk_p), 'npos': int(dprof.shape[1]),
            'pos': [round(float(x), 1) for x in dpos],
            'cyc': [round(float(x), 1) for x in dcyc],
            'base': [_i100(x) for x in dprof[0]],
            'prof': [_i100(x) for x in dprof.reshape(-1)],
            'blkOf': [int(x) for x in blk_of],
            'cycPerBlk': float(np.median(np.diff(np.asarray(dcyc, float))))
            if len(dcyc) > 1 else float(FRAME_CYC),
            'spikeN': int(n_spike),
        },
        'ax': {'ael': ax_ael, 'rate': round(rate_hi, 1)},
    }
    return pkg

# ==================================================================
# L1 变幅VA/谱载（L1 变幅VA+FBG + L1 谱载+FBG，新 14 组）：FBG(sm130)，PAC/Mistras .DTA
# ==================================================================
V3_GROUPS = (list(L1_CAMPAIGNS['L1 变幅VA+FBG']['members'])
          + list(L1_CAMPAIGNS['L1 谱载+FBG']['members']))

MIN_FRAMES = 20               # 帧数太少的组不出版本包（L1-34=2、L1-36=5、L1-30=13）
V3_AE_EMPTY = -99999             # 前端约定的 ael 空值标记
HIT_WIN_CYC = 5000.0          # eae 的滚动窗（cycle）
V3_LEVELS = (0.25, 0.55, 0.85)
FO_COLS = ['R1', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5']
FBG_WINDOW_S = 14.0           # 一个 FBG 统计窗的时长（140 行 x 0.1 s）
FBG_BURST_GAP_S = 60.0        # 相邻窗前隔超过它算两个突发


# --------------------------------------------------------------- 工具
def V3__i1000(x):
    if x is None or not np.isfinite(x):
        return 0
    return int(round(float(x) * 1000))


def V3__i100(x):
    if x is None or not np.isfinite(x):
        return 0
    return int(round(float(x) * 100))


def V3_unity01(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def V3_levels_from(d):
    d = np.asarray(d, dtype=float)
    lv = np.zeros(len(d), dtype=int)
    for i, th in enumerate(V3_LEVELS):
        lv[d >= th] = i + 1
    return lv


def roll_sum(x, win):
    """滚动求和（含当前位置），窗宽 win 个样本。"""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(len(x))
    for i in range(len(x)):
        j = max(0, i - win + 1)
        out[i] = c[i + 1] - c[j]
    return out


def zoh_to(target_t, src_t, src_v):
    """把 (src_t, src_v) 零阶保持到 target_t（取最近样本）。"""
    if len(src_t) == 0:
        return np.zeros(len(target_t))
    idx = np.searchsorted(src_t, target_t)
    left = np.clip(idx - 1, 0, len(src_t) - 1)
    right = np.clip(idx, 0, len(src_t) - 1)
    use_left = np.abs(target_t - src_t[left]) <= np.abs(target_t - src_t[right])
    return np.where(use_left, src_v[left], src_v[right])


def ffill(a):
    """前向填充 NaN（首部则后向填），避免前端画线时出现断点。"""
    a = np.asarray(a, dtype=float).copy()
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(ok)
    out[:first] = a[first]
    return out


FANO_WIN = 60          # 60 s 帧 x 60 = 1 h（与 ae_burst.py 同口径）


def fano_series(gid, target_t):
    """从 60 s 细帧表算 Fano 簇状性，ZOH 到 600 s 帧网格。

    Fano = 滑动窗内 var(n) / mean(n)，窗宽 60 个 60 s 帧 = 1 h。
    与 `l1/ae_burst.py` 完全同口径，避免两处定义不一致。
    """
    p = os.path.join(RES, '_l1ae_frames60s_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        t60 = z['frame'].astype(float) * 60.0
        x = np.asarray(z['n'], dtype=float)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    N = len(x)
    fano = np.full(N, np.nan)
    for i in range(FANO_WIN, N + 1):
        s1 = c1[i] - c1[i - FANO_WIN]
        s2 = c2[i] - c2[i - FANO_WIN]
        mu = s1 / FANO_WIN
        if mu > 0:
            fano[i - 1] = (s2 / FANO_WIN - mu * mu) / mu
    ok = np.isfinite(fano)
    if ok.sum() < 10:
        return None
    return zoh_to(target_t, t60[ok], fano[ok])


# --------------------------------------------------------------- 单组打包
def load_group(gid):
    p_ae = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    p_cy = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(p_ae) and os.path.exists(p_cy)):
        return None
    with np.load(p_ae, allow_pickle=True) as z:
        ae = {k: z[k] for k in z.files}
    with np.load(p_cy, allow_pickle=True) as z:
        cy = {k: z[k] for k in z.files}
    fb = None
    p_fb = os.path.join(RES, '_l1fbgprof_%s.npz' % gid)
    if os.path.exists(p_fb):
        with np.load(p_fb, allow_pickle=True) as z:
            fb = {k: z[k] for k in z.files}
    return ae, cy, fb


def pack_v3(gid):
    got = load_group(gid)
    if got is None:
        print(f'  [{gid}] 缺 AE 帧表或循环轴 → 跳过')
        return None
    ae, cy, fb = got

    nfr = len(np.asarray(cy['cycle'], dtype=float))
    if nfr < MIN_FRAMES:
        print(f'  [{gid}] 仅 {nfr} 帧（< {MIN_FRAMES}）→ 跳过')
        return None

    cyc = np.asarray(cy['cycle'], dtype=float)          # 横轴：cycle
    g_load = np.asarray(cy['g_load'], dtype=float)
    nf = float(cy['n_f']) if cy.get('n_f') is not None else float(cyc[-1])
    f_hz = float(cy['f_hz'])
    t_rel = np.asarray(cy['t_epoch'], dtype=float)      # 帧 x 600 s（绝对 epoch 近似）
    aen = np.asarray(ae['n'], dtype=float)

    if not (len(aen) == nfr == len(t_rel)):
        # 极点情况：帧表与循环轴不齐（理论上不会），按较短者截断
        m = min(len(aen), nfr, len(t_rel))
        print(f'  [{gid}] 帧表与循环轴长度不一致（{len(aen)}/{nfr}/{len(t_rel)}）'
              f'→ 截断到 {m}')
        nfr = m
        aen, cyc, t_rel, g_load = aen[:m], cyc[:m], t_rel[:m], g_load[:m]

    # --- 主曲线：离线复评 HI_hit（非因果）---
    D = V3_unity01(np.cumsum(aen))
    risk = D.copy()
    lv_f = V3_levels_from(D)

    # --- 因果证据 1：AE 活动度（最近 5000 cycle 窗内的事件数）---
    dcyc = np.diff(cyc, prepend=cyc[0])
    # 每帧代表的 cycle 数 -> 换算成"最近 5000 cycle 相当于多少帧"
    med_dc = float(np.median(dcyc[dcyc > 0])) if (dcyc > 0).any() else 1.0
    win_fr = max(1, int(round(HIT_WIN_CYC / max(med_dc, 1e-9))))
    eae_raw = roll_sum(aen, win_fr)
    eae = V3_unity01(eae_raw)

    # --- 因果证据 2：FBG shape_rob25（在线可用）---
    # ⚠️ 归一化到 0~1 只是**显示需要**（前端按固定 0~1 轴画条），
    #    用到了全寿命极值，因而这一层缩放本身非因果；
    #    但 `shape_rob25` 指标本体是因果的（基线取自前 25% 加载窗）。
    #    原始量程记入 meta，便于反推物理值。
    est_raw = np.zeros(nfr)
    if fb is not None and 'prof' in fb and 'shape_rob25' in fb:
        ft = np.asarray(fb['t'], dtype=float)
        s25 = np.asarray(fb['shape_rob25'], dtype=float)
        p2p_med = np.asarray(fb['p2p_med'], dtype=float)
        p2p_ch = np.asarray(fb['p2p_ch'], dtype=float) \
            if 'p2p_ch' in fb else None
        st = zoh_to(t_rel, ft, p2p_med)
        est_raw = zoh_to(t_rel, ft, s25)
        est = V3_unity01(est_raw)
        if p2p_ch is not None and p2p_ch.ndim == 2 and p2p_ch.shape[1] == len(FO_COLS):
            fo = {FO_COLS[k]: zoh_to(t_rel, ft, p2p_ch[:, k])
                  for k in range(len(FO_COLS))}
        else:
            fo = {}
        n_win = len(ft)
        fbg_span = (ft[-1] - ft[0]) / 3600.0 if len(ft) > 1 else 0.0
    else:
        st = np.zeros(nfr)
        est = np.zeros(nfr)
        fo = {}
        n_win, fbg_span = 0, 0.0

    # --- AE 幅值纵轴（dB）---
    amp = np.asarray(ae['amp_p90'], dtype=float) if 'amp_p90' in ae else None
    if amp is None or len(amp) != nfr:
        ael = np.full(nfr, V3_AE_EMPTY, dtype=np.int64)
    else:
        ael = np.array([V3__i1000(a) if np.isfinite(a) and a > 0 else V3_AE_EMPTY
                        for a in amp], dtype=np.int64)
    fin = ael[ael != V3_AE_EMPTY]
    if fin.size:
        ax_ael = [round(float(fin.min()) / 1000.0 - 0.3, 2),
                  round(float(fin.max()) / 1000.0 + 0.3, 2)]
    else:
        ax_ael = [-1.0, 1.0]

    # --- 证据通道声明（前端 EVIDENCE 面板由它渲染，未声明则回落原三行）---
    # 三个通道各自分工：AE 活动度（量级）/ FBG 剖面漂移（独立模态，在线）/
    # Fano 簇状性（无量纲、看涨落结构，与上面两个机理独立）
    indics = [
        {'key': 'e_ae', 'name': 'e_ae AE 活动度（最近 5000 cycle）',
         'color': 'rgba(255,208,138,.9)',
         'series': [V3__i1000(x) for x in eae]},
        {'key': 'e_st', 'name': 'e_st FBG 剖面漂移 shape_rob25（在线）',
         'color': 'rgba(0,227,154,.85)',
         'series': [V3__i1000(x) for x in est]},
    ]
    fano_raw = fano_series(gid, t_rel)
    if fano_raw is not None:
        fano_raw = ffill(fano_raw)
        indics.append({'key': 'e_fano', 'name': 'e_fano Fano 簇状性（1 h 窗 var/mean）',
                       'color': 'rgba(176,124,255,.9)',
                       'series': [V3__i1000(x) for x in V3_unity01(fano_raw)]})

    # --- 摘要 ---
    def first_ge(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    win_hits = int(round(float(np.sum(aen))))
    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge(0.25), 't55': first_ge(0.55), 't85': first_ge(0.85),
        'b2': None,          # 无弱标签 → 前端自动隐藏
        'b3': 100.0,         # 失效锚 = n_f
        'c0Pct': None,
        'refs': [],
        'aeEvents': win_hits,
        'nFo': len(fo),
        'nDfos': 0,          # 本批无 DFOS
        'f_hz': round(f_hz, 3),
        'load_h': round(float(cy['load_h']), 1),
        'span_h': round(float(cy['span_h']), 1),
        'fill': str(cy['fill']) if 'fill' in cy else '?',
        'fbgWin': int(n_win),
        'fbgSpanH': round(fbg_span, 1),
        # 原始量程：est/eae 显示用的 0~1 缩放是「全寿命极值归一」，
        # 记下量程便于反推物理值（指标本体仍是因果的）
        'estRawMin': round(float(np.min(est_raw)), 4),
        'estRawMax': round(float(np.max(est_raw)), 4),
        'eaeWinCyc': float(HIT_WIN_CYC),
        'eaeWinFrames': int(win_fr),
        'eaeRawMax': int(round(float(np.max(eae_raw)))),
        'nIndics': len(indics),
        'fanoRawMin': (round(float(np.min(fano_raw)), 1)
                       if fano_raw is not None else None),
        'fanoRawMax': (round(float(np.max(fano_raw)), 1)
                       if fano_raw is not None else None),
    }

    warn = []
    last = 0
    for f in range(nfr):
        lvv = int(lv_f[f])
        if lvv > last:
            for g in range(last + 1, lvv + 1):
                warn.append({'f': f, 't': round(float(cyc[f]), 1), 'lv': g})
            last = lvv

    chans = [
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流(%d s 帧)' % 600,
         'n': '2', 'unit': '通道'},
        {'key': 'fbg', 'name': '光纤光栅 FBG', 'mode': '突发式(每 420/240 s 采 20 s)',
         'n': str(len(fo)) if fo else '0', 'unit': '通道'},
        {'key': 'engine', 'name': '离线复评 HI_hit', 'mode': 'AE 累积·需全寿命归一',
         'n': None, 'unit': ''},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1v3', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳（%s）' % ('变幅 VA' if gid in
                                          ('L1-06', 'L1-13', 'L1-14', 'L1-24')
                                          else '谱载'),
        'mode': 'AE(.DTA) + FBG · 离线复评主曲线 + 在线应变剖面',
        'indexName': '离线复评 HI_hit（非在线）',
        'trendName': 'HI_hit 趋势 / 阈值 0.25 · 0.55 · 0.85',
        'labels': {'unit': 'HI_hit (0~1)', 'legend': 'HI_hit',
                   'margin': '1 − HI_hit', 'thrName': 'HI_hit',
                   'peak': '峰值 HI_hit',
                   'foot': 'HI_hit ∈ [0,1] · 阈值 0.25 / 0.55 / 0.85',
                   # 面板副标题覆盖：本批没有 DFOS，只有 FBG，
                   # 否则前端 isCycle() 会写成「分布式应变」（DFOS 措辞）
                   'ae': '声发射 · 事件率 / 峰值（600 s 帧）',
                   'fo': '光纤光栅 · 10 通道（应变峰峰值）',
                   'st': '光纤光栅应变 · 峰峰值 / 波动'},
        'n': int(nfr), 'nfr': int(nfr), 'step': 1,
        'frameDt': float(med_dc), 'dt': float(med_dc),
        'dur': float(nf), 'c0': 0.0,
        'foCols': list(fo.keys()),
        'chans': chans,
        'indics': indics,
        'meta': meta, 'warn': warn,
        # --- 波形数组（长度均 = nfr）---
        'D': [V3__i1000(x) for x in D],
        'risk': [V3__i1000(x) for x in risk],
        'eae': [V3__i1000(x) for x in eae],
        'est': [V3__i1000(x) for x in est],
        'lv': [int(x) for x in lv_f],
        'st': [V3__i100(x) for x in st],
        'ael': [int(x) for x in ael],
        'aen': [int(round(x)) for x in aen],
        't': [round(float(x), 1) for x in cyc],
        'fo': {k: [V3__i100(x) for x in v] for k, v in fo.items()},
        'ax': {'ael': ax_ael,
               'rate': round(max(2.0, float(np.max(aen)) * 1.2), 1)},
    }
    return pkg

# ==================================================================
# 统一入口
# ==================================================================
# L1 变幅VA+FBG批成员（给 l1v3 的清单打 spec 标）；**唯一来源 = shm.datasets**
V3_VA_GROUPS = tuple(campaign_members('C3'))

_SPECS = {
    'l1': {
        'batch': 'l1 · L1 恒幅+FBG+DFOS（L1-03/04/05/09）',
        'name': 'ReMAP / TU-Delft L1',
        'groups': None,                     # None → _deg.META.keys()
        'pack': pack_v1,
        'comment': None,
        'pre': False,
    },
    'l1v2': {
        'batch': 'l1v2 · L1 恒幅+DFOS（L1-49 至 L1-60，无 FBG）',
        'name': 'L1 恒幅+DFOS · 离线复评(非在线)',
        'groups': V2_GROUPS,
        'pack': pack_v2,
        'comment': None,
        'pre': False,
    },
    'l1v3': {
        'batch': 'l1v3 · L1 变幅VA+FBG 与 L1 谱载+FBG（FBG(sm130) / .DTA）',
        'name': 'L1 变幅VA/谱载 · 新 14 组（AE .DTA + FBG）',
        'groups': V3_GROUPS,
        'pack': pack_v3,
        'comment': '/* 自动生成，勿手改：l1/export_dashboard.py --ds l1v3 */',
        'pre': True,
    },
}


def _entry(ds, gid, pkg):
    """清单里的一组条目（各批字段顺序沿用合并前，前端不依赖顺序）。"""
    if ds == 'l1v3':
        return {'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'],
                'n': pkg['nfr'], 'unit': pkg['unit'],
                'frameDt': pkg['frameDt'], 'meta': pkg['meta'],
                'foCols': pkg['foCols'], 'warn': pkg['warn'],
                'spec': {'kind': ('变幅 VA' if gid in V3_VA_GROUPS else '谱载')}}
    return {'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'], 'n': pkg['n'],
            'meta': pkg['meta'], 'foCols': pkg['foCols'], 'warn': pkg['warn'],
            'unit': 'cycle', 'frameDt': pkg['frameDt']}


def _js_text(gid, pkg, comment):
    """单组 JS。L1 恒幅+FBG+DFOS/L1 恒幅+DFOS一行; L1 变幅VA/谱载带"勿手改"注释头并分三行。"""
    head = (comment + '\n') if comment else ''
    nl = '\n' if comment else ''
    return (head + 'window.SHM_DATA=window.SHM_DATA||{};' + nl +
            'window.SHM_DATA[%s]=%s;\n'
            % (json.dumps(gid),
               json.dumps(pkg, ensure_ascii=False, separators=(',', ':'))))


def _log(ds, gid, pkg, size):
    if ds == 'l1':
        print(f'  写出 {gid}.js  ({size:.0f} KB)  帧数={pkg["nfr"]}  '
              f'D_end={pkg["meta"]["D_end"]}  t85={pkg["meta"]["t85"]}%  '
              f'c0={pkg["meta"]["c0Pct"]}%  AE事件={pkg["meta"]["aeEvents"]}  '
              f'FBG={pkg["meta"]["nFo"]}ch  DFOS={pkg["meta"]["nDfos"]}pt  '
              f'空间块={pkg["dfos"]["nblk"]}×{pkg["dfos"]["npos"]}  '
              f'去尖峰={pkg["dfos"]["spikeN"]}点  ael轴={pkg["ax"]["ael"]}')
    elif ds == 'l1v2':
        print(f'  写出 {gid}.js  ({size:.0f} KB)  帧数={pkg["nfr"]}  '
              f'n_f={pkg["dur"]:.0f}  D_end={pkg["meta"]["D_end"]}  '
              f't85={pkg["meta"]["t85"]}%  AE事件={pkg["meta"]["aeEvents"]}  '
              f'DFOS={pkg["meta"]["nDfos"]}pt  '
              f'空间段={pkg["dfos"]["nblk"]}×{pkg["dfos"]["npos"]}  '
              f'去尖峰={pkg["dfos"]["spikeN"]}点  ael轴={pkg["ax"]["ael"]}')
    else:
        m = pkg['meta']
        print('  %-7s 帧 %5d  n_f %9.0f  f %.3f Hz  FBG窗 %5d  通道 %2d  '
              't85 %s  包 %.0f KB'
              % (gid, pkg['nfr'], pkg['dur'], m['f_hz'], m['fbgWin'],
                 len(pkg['foCols']), m['t85'], size))


def run(ds, groups, out):
    spec = _SPECS[ds]
    if groups:
        gid_list = list(groups)
    elif spec['groups'] is not None:
        gid_list = list(spec['groups'])
    else:
        gid_list = list(_deg.META.keys())
    outdir = os.path.abspath(out if os.path.isabs(out)
                             else os.path.join(_CWD0, out))
    os.makedirs(outdir, exist_ok=True)
    print('输出目录:', outdir)

    index = []
    for gid in gid_list:
        if spec['pre']:
            print(f'[{gid}]')
        pkg = spec['pack'](gid)
        if pkg is None:                       # 帧数太少的组不出包
            continue
        size = write_js(os.path.join(outdir, f'{gid}.js'),
                        _js_text(gid, pkg, spec['comment']))
        _log(ds, gid, pkg, size)
        index.append(_entry(ds, gid, pkg))

    ds_pkg = {'id': ds, 'name': spec['name'], 'unit': 'cycle',
              'path': 'data/', 'groups': index}
    ipath = os.path.join(outdir, f'index_{ds}.js')
    write_js(ipath, 'window.SHM_DATASETS=window.SHM_DATASETS||{};'
             'window.SHM_DATASETS["%s"]=%s;\n'
             % (ds, json.dumps(ds_pkg, ensure_ascii=False,
                               separators=(',', ':'))))
    if ds == 'l1v3':
        print(f'\n共 {len(index)} 组 → index_{ds}.js')
        print('组：', ' '.join(g['gid'] for g in index))
    else:
        print(f'\n写出清单 {ipath}')
        print(f'提示: dashboard/index.html 已内置 '
              f'<script src="data/index_{ds}.js">，刷新即见。')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(
        description='L1 三批数据集 → 看板 JS 数据包（统一入口）')
    ap.add_argument('--ds', default='l1', choices=sorted(_SPECS) + ['all'],
                    help='数据分组: l1=L1 恒幅+FBG+DFOS / l1v2=L1 恒幅+DFOS / l1v3=L1 变幅VA/谱载 / all')
    ap.add_argument('groups', nargs='*', default=None,
                    help='试件号(省略=该批全部)')
    ap.add_argument('--out', default=os.path.join(PROJ, 'dashboard', 'data'),
                    help='输出目录(默认 dashboard/data/)')
    a = ap.parse_args()
    for ds in (sorted(_SPECS) if a.ds == 'all' else [a.ds]):
        print(f'\n=== {ds}  {_SPECS[ds]["batch"]} ===')
        run(ds, a.groups, a.out)


if __name__ == '__main__':
    main()
