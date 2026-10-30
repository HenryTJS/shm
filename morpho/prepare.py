# -*- coding: utf-8 -*-
"""MORPHO FOD7 数据集的初步处理：一个 7.96 GB 的 HDF5 → 若干可直接用 Excel 看的 CSV。

数据集：H2020 MORPHO 项目，LEAP 发动机风扇叶片前缘的 FOD 板（3D 编织复材 + 钢前缘，
约 800 × 350 mm 曲面，厚度沿宽度 6.3 → 10.8 → 2.8 mm）。先落锤冲击（约 55 J）引入
目视勉强可见的损伤，再做四点弯曲多级疲劳直至破坏。

原始文件 MORPHO_FOD7.h5 内部只有一个 HDF5，五类数据各占一个顶层组（属性见 ReadMe.pdf）：

    1_Impedance  FOD7_Healthy / FOD7_Failed × PZT_5…29 → Z / phase / freq（801 点，
                 100 Hz–200 kHz，激励 1 V）
    2_FBG        21 个载荷工况 × FBG_1…16 → FBG_data（µm/m）/ time（s）
    3_Impact     PZT_1…29 → PZT_data（1 MHz 采，10 s = 1e7 点）
    4_MTS        21 个载荷工况 → displacement / load / time
    5_Active     25 个损伤工况 × 8 个激励频率 × Actionneur1…4 × 10 次重复
                 → measured_data_rep_L.mat，形状 (30, 5001)：行 0 = 时间，行 1 = 激励，
                   行 2…29 = 除作动器外的 28 个 PZT 的响应

本脚本只做**结构化解包**，不改任何物理口径：每个模态写到自己的目录，宽表（时间 × 通道）
优先，另配一张 summary 便于快速扫。产物默认落在 `data_root/morpho/processed/`。

用法
----
    python prepare.py                     # 全部（约 4–8 分钟，读满 7.9 GB）
    python prepare.py --only index fbg    # 只做指定几类
    python prepare.py --only active --limit-active 2   # 每工况只抽 2 个重复（冒烟用）
    python prepare.py --dry-run           # 只报结构，不落盘

产物（全部 UTF-8 with BOM，Excel 直接打开不乱码）
----
    00_report.md                     本次处理的清单、口径与已发现的异常
    01_inventory/tree.csv            完整组树：每个 dataset 的路径 / 形状 / dtype / 字节
    01_inventory/cases.csv           25 个工况 × 各模态有无 × 名义载荷/循环
    01_inventory/channels_pzt.csv    29 个 PZT 的类型（1–4 标准陶瓷 / 5–29 印刷）与坐标
    02_impedance/imp_healthy_Z.csv    宽表：freq + PZT_5…29 的 |Z|（Ω）
    02_impedance/imp_failed_Z.csv     同上，失效后
    02_impedance/imp_*_phase.csv      同上，相位（°）
    02_impedance/imp_delta.csv        每个 PZT 的谐振峰迁移与频段 RMS 比
    03_fbg/fbg_<工况>.csv             21 个：time + FBG_1…16（µm/m）
    03_fbg/fbg_summary.csv            工况 × FBG 的时长/采样/均值/峰峰值
    04_mts/mts_<工况>.csv             21 个：time, load_N, load_kN, disp_mm
    04_mts/mts_summary.csv            工况：峰谷、实测循环数、割线刚度
    05_impact/impact_summary.csv      29 通道：峰值、峰时、RMS、能量、重复通道
    05_impact/impact_zoom.csv         冲击窗 ±10 ms 的 29 通道原始波形（1 MHz）
    06_active/active_index.csv        8000 条信号的索引（工况/频率/作动器/重复）
    06_active/active_features.csv     每条信号 × 28 个接收通道的峰值/RMS/首波到时
    06_active/wave_<工况>_<频率>_A<k>.csv  若干条示例波形（宽的 30 列）
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)                       # 仓库根（paths.py）
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths                                       # noqa: E402

H5_NAME = 'MORPHO_FOD7.h5'
FREQS = ('30kHz', '50kHz', '60kHz', '90kHz', '100kHz', '150kHz', '200kHz', '250kHz')
CSV_KW = dict(index=False, encoding='utf-8-sig', float_format='%.6g')
PZT_STD = (1, 2, 3, 4)                             # 标准陶瓷，也是唯一的 4 个作动器


# ----------------------------------------------------------------- 路径与工具
def default_h5():
    """原始 HDF5 的位置：优先仓库内 junction，其次 data_root/morpho。"""
    cand = [os.path.join(HERE, 'data', H5_NAME),
            os.path.join(paths.data_root(), 'morpho', H5_NAME)]
    for p in cand:
        if os.path.exists(p):
            return p
    return cand[0]


def default_out():
    return os.path.join(paths.data_root(), 'morpho', 'processed')


def say(msg=''):
    print(msg, flush=True)


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def pzt_id(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


def case_key(name):
    """'12kN_400' → (12.0, 400)；baseline 工况放到最后并按名字排。"""
    m = re.match(r'(\d+)kN_(\d+)$', name)
    if m:
        return (0, float(m.group(1)), int(m.group(2)))
    return (1, 999.0, name)


def load_cases(f):
    """FBG / MTS / Active 三处共有的疲劳工况（顺序 = 载荷升序、循环升序）。"""
    common = set(f['2_FBG']) & set(f['4_MTS']) & set(f['5_Active'])
    return sorted(common, key=case_key)


def active_cases(f):
    """5_Active 的全部工况（含 4 个基线状态，不含 image）。"""
    return [c for c in f['5_Active'] if c != 'image']


# ----------------------------------------------------------------- 01 索引
def task_index(f, out):
    d = ensure(os.path.join(out, '01_inventory'))

    rows = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            rows.append(dict(路径='/' + name, 类型='dataset', 形状='x'.join(map(str, obj.shape)),
                             dtype=str(obj.dtype), 字节=int(obj.size * obj.dtype.itemsize)))
        else:
            rows.append(dict(路径='/' + name, 类型='group', 形状='', dtype='',
                             字节=int(len(obj))))
    f.visititems(walk)
    tree = pd.DataFrame(rows)
    tree.to_csv(os.path.join(d, 'tree.csv'), **CSV_KW)
    say('  tree.csv           %d 行' % len(tree))

    fbg, mts, act, imp = set(f['2_FBG']), set(f['4_MTS']), set(active_cases(f)), set()
    BASE = ('Healthy_Clamped', 'Healthy_Unclamped', 'AfterImpact_Clamped', 'AfterImpact_Unclamped')
    for c in active_cases(f):
        if c not in BASE and c in f['5_Active']:
            imp.add(c)
    cases = []
    for c in sorted(fbg | mts | act, key=case_key):
        m = re.match(r'(\d+)kN_(\d+)$', c)
        cases.append(dict(
            工况=c,
            类别='基线' if c in BASE else '疲劳',
            名义载荷kN=float(m.group(1)) if m else '',
            名义累计循环数=int(m.group(2)) if m else '',
            有FBG=int(c in fbg), 有MTS=int(c in mts), 有Active=int(c in act),
            频率数=len(f['5_Active'][c]) if c in act else 0,
            作动器数=len(f['5_Active'][c][FREQS[0]]) if c in act and FREQS[0] in f['5_Active'][c] else 0,
            重复数=10 if c in act else 0,
        ))
    pd.DataFrame(cases).to_csv(os.path.join(d, 'cases.csv'), **CSV_KW)
    say('  cases.csv          %d 行' % len(cases))

    geo = np.asarray(f['5_Active'].attrs['PZT_info_geo'])
    ch = [dict(
        PZT='PZT_%d' % i, 编号=i,
        类型='标准陶瓷' if i in PZT_STD else '印刷',
        可作动=int(i in PZT_STD),
        x_m=float(geo[0, i - 1]), y_m=float(geo[1, i - 1]),
    ) for i in range(1, 30)]
    pd.DataFrame(ch).to_csv(os.path.join(d, 'channels_pzt.csv'), **CSV_KW)
    say('  channels_pzt.csv   29 行')


# ----------------------------------------------------------------- 02 阻抗
def _imp_wide(f, state, quantity='Z'):
    """把一个状态的全部 PZT 拼成宽表：index = freq，columns = PZT_5…29。"""
    g = f['1_Impedance'][state]
    ids = sorted(g, key=pzt_id)
    freq = g[ids[0]]['freq'][()].ravel()
    cols = {'freq_Hz': freq}
    for k in ids:
        z = g[k]['Z'][()].ravel()
        if quantity == 'Z':
            cols[k] = z
        elif quantity == 'phase':
            cols[k] = g[k]['phase'][()].ravel()
        else:                                      # Re(Z) = |Z|·cos(phase)
            cols[k] = z * np.cos(np.deg2rad(g[k]['phase'][()].ravel()))
    return pd.DataFrame(cols), ids


def _anchors(freq, y, anchors=(100.0, 1e3, 1e4, 5e4, 1e5)):
    return [float(np.interp(a, freq, y)) for a in anchors]


def _resonances(freq, y, lo=5e3, hi=2e5, k=3):
    """实部阻抗在 [lo, hi] 内的局部极大（EMI 的谐振峰），按幅度取前 k 个。"""
    m = (freq >= lo) & (freq <= hi)
    fr, yy = freq[m], y[m]
    peaks = []
    for i in range(1, len(yy) - 1):
        if yy[i] > yy[i - 1] and yy[i] >= yy[i + 1]:
            peaks.append((float(fr[i]), float(yy[i])))
    peaks.sort(key=lambda t: -t[1])
    return peaks[:k]


def task_impedance(f, out):
    d = ensure(os.path.join(out, '02_impedance'))
    say('  读 1_Impedance …')
    zh, ids = _imp_wide(f, 'FOD7_Healthy')
    zf, _ = _imp_wide(f, 'FOD7_Failed')
    rh, _ = _imp_wide(f, 'FOD7_Healthy', 'ReZ')
    rf, _ = _imp_wide(f, 'FOD7_Failed', 'ReZ')
    zh.to_csv(os.path.join(d, 'imp_healthy_Z.csv'), **CSV_KW)
    zf.to_csv(os.path.join(d, 'imp_failed_Z.csv'), **CSV_KW)
    rh.to_csv(os.path.join(d, 'imp_healthy_ReZ.csv'), **CSV_KW)
    rf.to_csv(os.path.join(d, 'imp_failed_ReZ.csv'), **CSV_KW)
    _imp_wide(f, 'FOD7_Healthy', 'phase')[0].to_csv(
        os.path.join(d, 'imp_healthy_phase.csv'), **CSV_KW)
    _imp_wide(f, 'FOD7_Failed', 'phase')[0].to_csv(
        os.path.join(d, 'imp_failed_phase.csv'), **CSV_KW)

    freq = zh['freq_Hz'].values
    band = (freq >= 1e3) & (freq <= 1e5)          # 1–100 kHz：印刷 PZT 的主工作段
    anchors = (100.0, 1e3, 1e4, 5e4, 1e5)
    a_name = ('100Hz', '1kHz', '10kHz', '50kHz', '100kHz')

    rows = []
    for k in ids:
        hz, fz = zh[k].values, zf[k].values
        hr, fr_ = rh[k].values, rf[k].values
        r1 = a_name[1]
        row = dict(PZT=k)
        for nm, a, b in zip(a_name, _anchors(freq, hz), _anchors(freq, fz)):
            row['Z_健康_' + nm] = a
            row['Z_失效_' + nm] = b
        row['ReZ_RMS_1_100k_健康'] = float(np.sqrt(np.mean(hr[band] ** 2)))
        row['ReZ_RMS_1_100k_失效'] = float(np.sqrt(np.mean(fr_[band] ** 2)))
        row['ReZ_RMS比'] = (row['ReZ_RMS_1_100k_失效']
                            / row['ReZ_RMS_1_100k_健康']
                            if row['ReZ_RMS_1_100k_健康'] else np.nan)
        row['ReZ_相关_1_100k'] = float(np.corrcoef(hr[band], fr_[band])[0, 1])

        pk_h = _resonances(freq, hr)
        pk_f = _resonances(freq, fr_)
        row['谐振峰数_健康'] = len(pk_h)
        row['谐振峰数_失效'] = len(pk_f)
        for i in range(3):
            row['谐振%d_健康_Hz' % (i + 1)] = pk_h[i][0] if i < len(pk_h) else np.nan
            row['谐振%d_健康_ohm' % (i + 1)] = pk_h[i][1] if i < len(pk_h) else np.nan
            row['谐振%d_失效_Hz' % (i + 1)] = pk_f[i][0] if i < len(pk_f) else np.nan
            row['谐振%d_失效_ohm' % (i + 1)] = pk_f[i][1] if i < len(pk_f) else np.nan
        # 最强谐振峰：按频率就近匹配（±1 kHz），无匹配则留空——失效后最强峰可能换了模态
        row['谐振峰漂移_Hz'] = np.nan
        row['谐振峰幅值比'] = np.nan
        if pk_h and pk_f:
            fh_, ah_ = pk_h[0]
            cand = [(abs(fq - fh_), fq, aq) for fq, aq in pk_f if abs(fq - fh_) <= 1e3]
            if cand:
                _, fq, aq = min(cand)
                row['谐振峰漂移_Hz'] = fq - fh_
                row['谐振峰幅值比'] = aq / ah_ if ah_ else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(d, 'imp_delta.csv'), **CSV_KW)
    say('  imp_*.csv          2 状态 × (|Z|, ReZ, phase) 宽表 + delta（25 个印刷 PZT）')
    return r1


# ----------------------------------------------------------------- 03 FBG
def _load_mask(f, case, t, frac=0.5):
    """用 MTS 的载荷把 FBG 记录里的「加载段」挑出来（两条记录共用同一测试时钟）。"""
    g = f['4_MTS'].get(case)
    if g is None:
        return None
    tm = g['time'][()].ravel()
    ld = np.abs(g['load'][()].ravel())
    if tm.size < 2 or ld.size < 2:
        return None
    return np.interp(t, tm, ld) >= frac * float(np.max(ld)), float(tm[-1] - tm[0])


def task_fbg(f, out, cases):
    d = ensure(os.path.join(out, '03_fbg'))
    rows = []
    for case in cases:
        g = f['2_FBG'].get(case)
        if g is None:
            continue
        ids = sorted(g, key=pzt_id)
        ser, tser = {}, {}
        for k in ids:
            ser[k] = g[k]['FBG_data'][()].ravel()
            tser[k] = g[k]['time'][()].ravel()
        lens = {len(v) for v in ser.values()}
        same_t = len({len(v) for v in tser.values()}) == 1

        if len(lens) == 1 and same_t:
            t = tser[ids[0]]
            df = pd.DataFrame({'time_s': t})
            for k in ids:
                df[k] = ser[k]
            df.to_csv(os.path.join(d, 'fbg_%s.csv' % case), **CSV_KW)
        else:
            # 长度不一致的工况退化为长表（极少见，保留原始时间轴）
            long = []
            for k in ids:
                long.append(pd.DataFrame({'time_s': tser[k], 'FBG': k,
                                          'strain_um_per_m': ser[k]}))
            pd.concat(long).to_csv(os.path.join(d, 'fbg_%s_long.csv' % case), **CSV_KW)

        t0 = tser[ids[0]]
        dt = float(np.median(np.diff(t0))) if t0.size > 1 else np.nan
        mask, mts_span = _load_mask(f, case, t0)
        for k in ids:
            v = ser[k]
            tk = tser[k]
            row = dict(
                工况=case, FBG=k, 点数=int(v.size),
                t起_s=float(tk[0]), t止_s=float(tk[-1]),
                中位dt_s=dt, 采样率_Hz=1.0 / dt if dt else np.nan,
                均值=float(v.mean()), 标准差=float(v.std()),
                最小=float(v.min()), 最大=float(v.max()),
                峰峰值=float(v.max() - v.min()),
            )
            if mask is not None and mask.size == v.size and mask.any():
                a = v[mask]
                row.update(
                    加载段点数=int(mask.sum()),
                    加载段占比=float(mask.mean()),
                    加载段均值=float(a.mean()),
                    加载段标准差=float(a.std()),
                    加载段峰峰值=float(a.max() - a.min()),
                    MTS时长_s=mts_span,
                    时长比_FBG对MTS=(row['t止_s'] - row['t起_s']) / mts_span if mts_span else np.nan,
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(d, 'fbg_summary.csv'), **CSV_KW)
    say('  fbg_*.csv          %d 个工况 + summary（%d 行，含 MTS 加载段掩码统计）'
        % (len(cases), len(rows)))


# ----------------------------------------------------------------- 04 MTS
def _up_crossings(y):
    """过均值的上行穿越次数 ≈ 载荷循环数。"""
    s = np.sign(y - np.mean(y))
    return int(np.sum((s[1:] > 0) & (s[:-1] <= 0)))


def task_mts(f, out, cases):
    d = ensure(os.path.join(out, '04_mts'))
    rows = []
    prev_nominal = {}
    for case in cases:
        g = f['4_MTS'].get(case)
        if g is None:
            continue
        t = g['time'][()].ravel()
        load = g['load'][()].ravel()
        disp = g['displacement'][()].ravel()
        df = pd.DataFrame({'time_s': t, 'load_N': load, 'load_kN': load / 1e3,
                           'disp_mm': disp})
        df.to_csv(os.path.join(d, 'mts_%s.csv' % case), **CSV_KW)

        m = re.match(r'(\d+)kN_(\d+)$', case) or [None, '', '']
        slope = float(np.polyfit(disp, load, 1)[0]) if disp.size > 2 else np.nan
        measured = _up_crossings(load)
        nominal = int(m.group(2)) if m.group(2) else np.nan
        rows.append(dict(
            工况=case,
            名义载荷kN=float(m.group(1)) if m.group(1) else np.nan,
            名义累计循环数=nominal,
            本次实测循环数=measured,
            本次循环数_推定=(nominal - prev_nominal.get(m.group(1), 0)
                              if nominal == nominal else np.nan),
            点数=int(t.size), 时长_s=float(t[-1] - t[0]),
            中位dt_s=float(np.median(np.diff(t))),
            载荷min_N=float(load.min()), 载荷max_N=float(load.max()),
            载荷幅值_N=float(load.max() - load.min()),
            位移min_mm=float(disp.min()), 位移max_mm=float(disp.max()),
            割线刚度_N_per_mm=slope,
        ))
        if nominal == nominal:
            prev_nominal[m.group(1)] = nominal
    pd.DataFrame(rows).to_csv(os.path.join(d, 'mts_summary.csv'), **CSV_KW)
    say('  mts_*.csv          %d 个工况 + summary' % len(rows))


# ----------------------------------------------------------------- 05 冲击
def task_impact(f, out, zoom_ms=10.0):
    d = ensure(os.path.join(out, '05_impact'))
    g = f['3_Impact']
    ids = sorted(g, key=pzt_id)
    fs = float(g.attrs['f_s'][0])
    say('  读 3_Impact（%d 通道 × %s 点，1 MHz）…' % (len(ids), '1e7'))

    sig, rows = [], []
    for k in ids:
        a = g[k + '/PZT_data'][:, 0]
        i = int(np.abs(a).argmax())
        sig.append(a)
        rows.append(dict(
            PZT=k, 点数=int(a.size), 时长_s=float(a.size / fs),
            峰值绝对_V=float(np.abs(a).max()), 峰值时刻_s=float(i / fs),
            RMS_V=float(np.sqrt(np.mean(a ** 2))),
            能量_V2s=float(np.sum(a ** 2) / fs),
            md5=hashlib.md5(a.tobytes()).hexdigest(),
        ))
    df = pd.DataFrame(rows)

    # 完全相同的通道（实测存在：PZT_26…29 与 PZT_2…5 逐点相同）
    dup = {}
    for h, sub in df.groupby('md5'):
        if len(sub) > 1:
            for name in sub['PZT']:
                if name != sub['PZT'].iloc[0]:
                    dup[name] = sub['PZT'].iloc[0]
    df['重复于'] = df['PZT'].map(dup).fillna('')
    df.drop(columns=['md5']).to_csv(os.path.join(d, 'impact_summary.csv'), **CSV_KW)

    # 冲击窗：以全体通道最猛的那个时刻为中心
    i0 = int(np.argmax([np.abs(a).max() for a in sig]))
    t0 = float(np.argmax(np.abs(sig[i0])) / fs)
    half = int(zoom_ms * 1e-3 * fs)
    lo, hi = max(0, int(t0 * fs) - half), min(sig[0].size, int(t0 * fs) + half)
    zoom = pd.DataFrame({'time_s': np.arange(lo, hi) / fs})
    for k, a in zip(ids, sig):
        zoom[k] = a[lo:hi]
    zoom.to_csv(os.path.join(d, 'impact_zoom.csv'), **CSV_KW)
    say('  impact_summary.csv 29 行（重复通道 %d 个）  zoom 中心 %.4f s，±%.0f ms，%d 点'
        % (len(dup), t0, zoom_ms, hi - lo))
    return df


# ----------------------------------------------------------------- 06 主动导波
def _active_map(act_id):
    """作动器 → 列 2…29 对应的 PZT 号（作动器自身被排除，按编号升序）。"""
    return [p for p in range(1, 30) if p != act_id]


def _features(x, fs):
    """单通道响应特征。

    ⚠️ 不给「绝对首波到时」：这套数据的接收通道会拾取驱动信号的电串扰，
    记录开头 0–50 µs 的能量与真实到达波同量级（实测多个通道 50% 包络阈值直接落在死区边界），
    阈值法得到的"到时"没有物理意义。要做得用与基线工况的互相关/差分，属于二次分析。
    这里给三个稳健量：包络峰值时刻、包络能量重心、SNR。
    """
    env = np.abs(hilbert(x))
    e2 = env ** 2
    t = np.arange(x.size) / fs
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    energy = float(np.sum(x ** 2) / fs)
    t_peak = float(np.argmax(env) / fs)
    centroid = float(np.sum(t * e2) / np.sum(e2)) if np.sum(e2) > 0 else np.nan
    noise = x[-500:]
    sigma = float(np.std(noise))
    snr = float(env.max() / sigma) if sigma > 0 else np.nan
    return peak, rms, energy, t_peak, centroid, snr


SAMPLE_CASES = (('Healthy_Clamped', '100kHz', 1),
                ('AfterImpact_Clamped', '100kHz', 1),
                ('4kN_200', '100kHz', 1),
                ('24kN_100', '100kHz', 1),
                ('Healthy_Clamped', '50kHz', 2),
                ('24kN_100', '250kHz', 1))


def task_active(f, out, limit_rep=None, samples=True):
    d = ensure(os.path.join(out, '06_active'))
    g5 = f['5_Active']
    cases = active_cases(f)

    idx_rows, feat_rows, chan_rows = [], [], []
    for act_id in PZT_STD:
        for col, p in enumerate(_active_map(act_id), start=2):
            chan_rows.append(dict(作动器='Actionneur%d' % act_id, 列号=col, PZT='PZT_%d' % p))
    pd.DataFrame(chan_rows).to_csv(os.path.join(d, 'active_channel_map.csv'), **CSV_KW)

    n_sig = 0
    t_start = time.time()
    for ci, case in enumerate(cases, 1):
        gc = g5[case]
        freqs = [fr for fr in FREQS if fr in gc]
        for fr in freqs:
            gf = gc[fr]
            fs = float(gf.attrs.get('fs', [1e6])[0])
            for act in sorted(gf, key=pzt_id):
                act_id = pzt_id(act)
                reps = sorted(gf[act])
                if limit_rep:
                    reps = reps[:limit_rep]
                for rep in reps:
                    path = '%s/%s/%s/%s' % (case, fr, act, rep)
                    arr = gf[act][rep][()]
                    nch, ns = arr.shape
                    t = arr[0]
                    dt = float(np.median(np.diff(t))) if ns > 1 else np.nan
                    idx_rows.append(dict(
                        工况=case, 频率_kHz=float(fr.replace('kHz', '')),
                        作动器='Actionneur%d' % act_id, 重复=rep,
                        形状='%dx%d' % arr.shape, 通道数=int(nch), 点数=int(ns),
                        dt_us=dt * 1e6 if dt == dt else np.nan,
                        时长_ms=float(ns * dt * 1e3) if dt == dt else np.nan,
                        路径=path,
                    ))
                    for col, p in enumerate(_active_map(act_id), start=2):
                        peak, rms, energy, t_peak, centroid, snr = _features(arr[col], fs)
                        feat_rows.append(dict(
                            工况=case, 频率_kHz=float(fr.replace('kHz', '')),
                            作动器='Actionneur%d' % act_id, 重复=rep,
                            列号=col, PZT='PZT_%d' % p,
                            峰值_V=peak, RMS_V=rms, 能量_V2s=energy,
                            峰时_us=t_peak * 1e6, 包络重心_us=centroid * 1e6,
                            SNR=snr,
                        ))
                    n_sig += 1
        if ci % 5 == 0 or ci == len(cases):
            say('    %2d/%d 工况  %5d 条信号  %.0f s' % (ci, len(cases), n_sig, time.time() - t_start))

    pd.DataFrame(idx_rows).to_csv(os.path.join(d, 'active_index.csv'), **CSV_KW)
    pd.DataFrame(feat_rows).to_csv(os.path.join(d, 'active_features.csv'), **CSV_KW)
    say('  active_index.csv   %d 行；active_features.csv  %d 行' % (len(idx_rows), len(feat_rows)))

    if not samples:
        return
    made = 0
    for case, fr, act_id in SAMPLE_CASES:
        if case not in g5 or fr not in g5[case]:
            continue
        act = 'Actionneur%d' % act_id
        if act not in g5[case][fr]:
            continue
        rep = 'measured_data_rep_1.mat'
        if rep not in g5[case][fr][act]:
            continue
        arr = g5[case][fr][act][rep][()]
        cols = {'time_s': arr[0], '激励_V': arr[1]}
        for col, p in enumerate(_active_map(act_id), start=2):
            cols['PZT_%d' % p] = arr[col]
        df = pd.DataFrame(cols)
        fn = 'wave_%s_%s_A%d.csv' % (case, fr, act_id)
        df.to_csv(os.path.join(d, fn), **CSV_KW)
        made += 1
    say('  示例波形           %d 个（宽表：time + 激励 + 28 通道）' % made)


# ----------------------------------------------------------------- 报告
def write_report(out, h5, info):
    p = os.path.join(out, '00_report.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('# MORPHO FOD7 初级处理报告\n\n')
        fh.write('- 源文件：`%s`（%.2f GB）\n' % (h5, os.path.getsize(h5) / 2 ** 30))
        fh.write('- 生成时刻：%s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
        fh.write('- 数据根：`%s`\n\n' % paths.data_root())
        fh.write(info)

        # 产物清单
        fh.write('## 产物清单\n\n| 目录 | 文件数 | 合计 | 内容 |\n| --- | --- | --- | --- |\n')
        NOTE = {
            '01_inventory': '组树 / 工况表 / PZT 坐标',
            '02_impedance': '电化学阻抗宽表 + 健康↔失效 delta',
            '03_fbg': 'FBG 时序 + 加载段统计',
            '04_mts': '试验机载荷-位移 + 循环数/刚度',
            '05_impact': '冲击通道汇总 + ±10 ms 波形窗',
            '06_active': '导波信号索引/特征 + 示例波形',
        }
        total_n = total_b = 0
        for d in sorted(NOTE):
            dp = os.path.join(out, d)
            if not os.path.isdir(dp):
                continue
            files = [os.path.join(dp, f) for f in os.listdir(dp)]
            n = len(files)
            b = sum(os.path.getsize(f) for f in files)
            total_n += n
            total_b += b
            fh.write('| `%s/` | %d | %.1f MB | %s |\n' % (d, n, b / 2 ** 20, NOTE[d]))
        fh.write('| **合计** | **%d** | **%.1f MB** | |\n' % (total_n, total_b / 2 ** 20))
    return p


# ----------------------------------------------------------------- 入口
TASKS = ('index', 'impedance', 'fbg', 'mts', 'impact', 'active')


def main(argv=None):
    ap = argparse.ArgumentParser(description='MORPHO_FOD7.h5 → 多个 CSV')
    ap.add_argument('--h5', default=None, help='原始 HDF5（默认自动定位）')
    ap.add_argument('--out', default=None, help='输出目录（默认 data_root/morpho/processed）')
    ap.add_argument('--only', nargs='+', choices=TASKS, default=list(TASKS))
    ap.add_argument('--limit-active', type=int, default=None,
                    help='每个 作动器 只取前 N 个重复（冒烟用）')
    ap.add_argument('--no-samples', action='store_true', help='不出示例波形 CSV')
    ap.add_argument('--dry-run', action='store_true', help='只打印结构，不落盘')
    a = ap.parse_args(argv)

    h5 = a.h5 or default_h5()
    out = a.out or default_out()
    if not os.path.exists(h5):
        say('[错误] 找不到 %s\n       数据应在 E:\\morpho，仓库内 morpho\\data 是 junction' % h5)
        return 2

    say('源文件 %s（%.2f GB）' % (h5, os.path.getsize(h5) / 2 ** 30))
    say('输出   %s' % out)
    say('任务   %s\n' % ', '.join(a.only))

    t0 = time.time()
    with h5py.File(h5, 'r') as f:
        if a.dry_run:
            for top in f:
                o = f[top]
                say('%-14s %s  children=%d' % (top, type(o).__name__,
                                               len(o) if isinstance(o, h5py.Group) else 0))
            say('\n[dry-run] 结构已列出，未写文件。')
            return 0

        ensure(out)
        cases = load_cases(f)
        if 'index' in a.only:
            say('[1/6] 索引')
            task_index(f, out)
        if 'impedance' in a.only:
            say('[2/6] 阻抗')
            task_impedance(f, out)
        if 'fbg' in a.only:
            say('[3/6] FBG')
            task_fbg(f, out, cases)
        if 'mts' in a.only:
            say('[4/6] MTS')
            task_mts(f, out, cases)
        imp = None
        if 'impact' in a.only:
            say('[5/6] 冲击')
            imp = task_impact(f, out)
        if 'active' in a.only:
            say('[6/6] 主动导波')
            task_active(f, out, a.limit_active, samples=not a.no_samples)

    info = []
    imp_p = os.path.join(out, '05_impact', 'impact_summary.csv')
    if os.path.exists(imp_p):
        s = pd.read_csv(imp_p)
        s['重复于'] = s['重复于'].fillna('').astype(str).str.strip()
        dup = s[s['重复于'] != '']
        if len(dup):
            info.append('## 已发现的异常\n\n')
            info.append('**冲击数据里只有 8 个独立通道**：PZT_9…29 是 PZT_1…8 的逐点复制。\n\n')
            info.append('| 通道 | 与谁逐点相同 |\n| --- | --- |\n')
            for r in dup.itertuples():
                info.append('| %s | %s |\n' % (r.PZT, r.重复于))
            info.append('\n用 md5 比对确认是同一份字节，不是"看起来像"。'
                        '做冲击定位/能量标定时只能当 %d 个通道用。\n\n' % (29 - len(dup)))
    mts_p = os.path.join(out, '04_mts', 'mts_summary.csv')
    if os.path.exists(mts_p):
        s = pd.read_csv(mts_p)
        gap = s[s['本次实测循环数'] < s['本次循环数_推定']]
        if len(gap):
            info.append('**MTS 部分工况的实测循环数少于该级计划值**：\n\n')
            info.append('| 工况 | 该级计划 | 实测 |\n| --- | --- | --- |\n')
            for r in gap.itertuples():
                info.append('| %s | %d | %d |\n'
                            % (r.工况, r.本次循环数_推定, r.本次实测循环数))
            info.append('\n循环数按「载荷过均值的上行穿越」计。工况名里的 N 是**累计**循环数'
                        '（每级各跑固定圈数：≤18 kN 每级 200 圈，≥20 kN 每级 100 圈），'
                        '所以 `20kN_100/200/300/400` 各录到 100 圈是正确的；'
                        '实测少于计划值说明该级中途停机、或试件在 24 kN 级提前破坏。\n\n')
    info.append('## 口径提醒\n\n')
    info.append('- **载荷单位是 N 不是 kN**：ReadMe 写 kN，但 4kN_200 的 `load` 峰值约 3900、'
                '谷值约 -4008，即 ±4 kN ⇒ 原始列是 N。CSV 里同时给了 `load_N` 与 `load_kN`。\n')
    info.append('- 载荷全为**负值（压）**，四点弯曲压-压疲劳；`disp_mm` 同样为负。\n')
    info.append('- 主动导波矩阵在 HDF5 里是 (30, 5001)：**行 0 时间、行 1 激励、行 2…29 响应**；'
                '作动器自身不在这 28 个响应里，列号↔PZT 见 `06_active/active_channel_map.csv`。'
                '该映射与 ReadMe 一致，并用「最近邻传感器到时」抽检过（Actionneur2 的最近 4 个'
                '传感器给出 1.8–2.0 km/s，量级合理）。\n')
    info.append('- 主动导波**不给绝对首波到时**：接收通道会拾取驱动信号的电串扰，记录开头'
                ' 0–50 µs 的能量与真实到达波同量级，阈值法得到的"到时"没有物理意义。'
                '`active_features.csv` 给的是稳健量：`峰时_us`（包络峰值时刻）、'
                '`包络重心_us`（包络能量重心）、`SNR`（尾部噪声为底）。'
                '真要做走时/成像，请用同通道跨工况的差分或互相关，那属于二次分析。\n')
    info.append('- FBG 采样约 **2 Hz**（中位 dt≈0.497 s），不是连续高速采集；单条记录常跨越'
                '数小时，中间含大量停机段。`fbg_summary.csv` 因此额外给出「加载段」统计：'
                '用同工况的 MTS 载荷把 |load| ≥ 50% 峰值的时段挑出来，两列都在。'
                '比较不同工况时应看 `加载段均值`，直接看 `均值` 会把停机段算进去。\n')
    info.append('- 阻抗只覆盖 PZT_5…29（25 个印刷 PZT），标准陶瓷 1…4 无阻抗数据；'
                '记录是 801 点、100 Hz–200 kHz。|Z| 在低频被容性主导（100 Hz 处约 0.6–0.9 MΩ），'
                '因此不要拿 |Z| 的最大值当"谐振峰"——`imp_delta.csv` 用的是实部 Re(Z)=|Z|cos(φ)'
                '的局部极大。\n')
    p = write_report(out, h5, ''.join(info))
    say('\n完成：%.1f s；报告 %s' % (time.time() - t0, p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
