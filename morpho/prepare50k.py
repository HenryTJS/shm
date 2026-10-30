# -*- coding: utf-8 -*-
"""MORPHO 50 kHz 派生集（5 块板 FOD3–FOD7）的初步处理：一个 93 MB HDF5 → 多个 CSV。

来源：Zenodo `10.5281/zenodo.20120070`（Stamatelatos / Dimitriou / Loutas，Patras），
是原始 MORPHO 采集的**滤波子集**：只留 50 kHz、每个作动器只留 3 个最近的非驱动接收器
（驱动通道被排除），2001 点 @ 1 MHz。

结构::

    /<panel>/50kHz/act<1..4>/recv<0..24>/cycle<累计循环数>/data   (n_reps, 2001) float32

⚠️ 与 `MORPHO_FOD7.h5` 的区别（都会影响结论，别混用）：
    - **只有导波**：没有阻抗 / FBG / 冲击 / MTS；
    - **只有 50 kHz**（原始有 8 个频率），信号长 2.001 ms（原始 5.001 ms）；
    - **只有 3 个接收器**（原始 28 个），且是 0 基编号 `recv0…recv3`；
    - 多出 4 块板 FOD3–FOD6 的 50 kHz 切片，这是它们唯一的公开形态。

本脚本只做**结构化解包 + 无监督特征**，不训练模型、不给损伤结论（那在 `analyze.py`）。

用法
----
    python prepare50k.py                      # 全部（约 2–3 分钟）
    python prepare50k.py --only index features
    python prepare50k.py --waves              # 额外导出 60 个波形宽表（约 60 MB）

产物（UTF-8 with BOM，Excel 直接打开）
----
    11_wave50/groups.csv        每 (板, 作动器, 接收器, 循环) 一行，跨重复聚合的特征
    11_wave50/reps.csv          每一条信号一行（12504 行），含人工标签 GOOD/BAD
    11_wave50/panel_cycle.csv   每板 × 每循环一行：12 个通道的中位特征（跨试件对比用）
    11_wave50/inventory.csv     组清单：板 / 作动器 / 接收器 / 循环 / 重复数 / 信号长度
    11_wave50/baseline.csv      每 (板, 作动器, 接收器) 的 cycle0 基线质量体检
    11_wave50/waves/            每 (板, 作动器, 接收器) 一个宽表：行 = 循环，列 = 时间采样
"""
from __future__ import annotations

import argparse
import json
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
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import paths                                       # noqa: E402

H5_NAME = 'morpho_50khz_full.h5'
LABELS_NAME = 'paper_labels.json'
CSV_KW = dict(index=False, encoding='utf-8-sig', float_format='%.6g')
FS = 1e6
OUTDIR = '11_wave50'

# 作动器 → 驱动通道（0 基），其余三个即接收器
DRIVE = {1: 0, 2: 1, 3: 2, 4: 3}


# ----------------------------------------------------------------- 路径
def default_h5():
    for p in (os.path.join(HERE, 'data', H5_NAME),
              os.path.join(paths.data_root(), 'morpho', H5_NAME)):
        if os.path.exists(p):
            return p
    return os.path.join(paths.data_root(), 'morpho', H5_NAME)


def default_out():
    return os.path.join(paths.data_root(), 'morpho', 'processed')


def say(msg=''):
    print(msg, flush=True)


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_labels(h5):
    """人工标签（GOOD/BAD），键 = (面板, 作动器, 接收器, 循环, 重复)。"""
    for p in (os.path.join(os.path.dirname(h5), LABELS_NAME),
              os.path.join(HERE, 'data', LABELS_NAME)):
        if os.path.exists(p):
            d = json.load(open(p, encoding='utf-8'))
            out = {}
            for x in d['labels']:
                out[(x['panel'], int(x['actuator']), int(x['receiver']),
                     int(x['cycle']), int(x['rep_idx']))] = x['label']
            return out
    return {}


# ----------------------------------------------------------------- 特征
def _good_reps(d, frac=0.2):
    """挑出幅值正常的重复。

    实测这批数据里存在**近零幅值的坏重复**：例如 FOD3 每组 4–8 次重复里只有 1–2 次是正常的
    （其余低 10–100 倍），FOD4 的 cycle0 有 80 次重复但约 1/3 是近零。
    判定：单次重复的 RMS 低于该组最大 RMS 的 `frac` 就丢掉。
    """
    r = np.sqrt((d ** 2).mean(axis=1))
    m = float(r.max())
    if m <= 0:
        return d, np.zeros(len(d), dtype=bool)
    keep = r > frac * m
    return d[keep], keep


def _chan_features(x):
    """单条信号的时域/包络特征。x: (2001,)。"""
    env = np.abs(hilbert(x))
    e2 = env ** 2
    t = np.arange(x.size) / FS
    s2 = float(np.sum(e2))
    return dict(
        peak=float(np.abs(x).max()),
        rms=float(np.sqrt(np.mean(x ** 2))),
        energy=float(np.sum(x ** 2) / FS),
        env_peak_us=float(np.argmax(env) / FS * 1e6),
        env_centroid_us=float(np.sum(t * e2) / s2 * 1e6) if s2 > 0 else np.nan,
        snr=float(env.max() / np.std(x[-200:])) if np.std(x[-200:]) > 0 else np.nan,
    )


def _band_ratio(x, edges=(0, 50e3, 100e3, 200e3, 500e3)):
    """频段能量占比（对 2001 点做 rFFT）。"""
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / FS)
    tot = float(spec.sum())
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        out.append(float(spec[m].sum()) / tot if tot > 0 else np.nan)
    return out


def _align(a, b, max_lag=200):
    """归一化互相关峰与最佳时移（样点）。"""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan, 0
    n = a.size + b.size
    nfft = 1 << int(np.ceil(np.log2(n)))
    c = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    c = np.concatenate((c[-max_lag:], c[:max_lag + 1])) / (na * nb)
    i = int(np.argmax(c))
    return float(c[i]), i - max_lag


# ----------------------------------------------------------------- 主任务
def task_index(f, out):
    d = ensure(os.path.join(out, OUTDIR))
    rows = []
    for panel in f:
        g = f[panel]['50kHz']
        for act in g:
            a = int(re.search(r'\d+', act).group())
            for rcv in g[act]:
                r = int(re.search(r'\d+', rcv).group())
                for c in g[act][rcv]:
                    ds = g[act][rcv][c]['data']
                    rows.append(dict(
                        面板=panel, 作动器=a, 接收器=r,
                        驱动通道=int(DRIVE.get(a, -1)),
                        循环=int(re.search(r'\d+', c).group()),
                        重复数=int(ds.shape[0]), 信号长度=int(ds.shape[1]),
                        采样率_Hz=int(FS),
                    ))
    df = pd.DataFrame(rows).sort_values(['面板', '作动器', '接收器', '循环'])
    df.to_csv(os.path.join(d, 'inventory.csv'), **CSV_KW)
    say('  inventory.csv      %d 组' % len(df))

    # 每板循环轴
    print_axis = []
    for panel in f:
        cs = sorted({int(re.search(r'\d+', c).group())
                     for act in f[panel]['50kHz'] for rcv in f[panel]['50kHz'][act]
                     for c in f[panel]['50kHz'][act][rcv]})
        print_axis.append('%s(%d): %s' % (panel, len(cs), cs[:6] + ['...'] + cs[-3:]))
    for s in print_axis:
        say('    ' + s)
    return df


def task_features(f, out, labels, waves=False):
    d = ensure(os.path.join(out, OUTDIR))
    wdir = ensure(os.path.join(d, 'waves')) if waves else None
    grow, prow, brow, drop = [], [], [], []

    for panel in f:
        gc = f[panel]['50kHz']
        for act in sorted(gc, key=lambda s: int(re.search(r'\d+', s).group())):
            a = int(re.search(r'\d+', act).group())
            for rcv in sorted(gc[act], key=lambda s: int(re.search(r'\d+', s).group())):
                r = int(re.search(r'\d+', rcv).group())
                grp = gc[act][rcv]
                cycles = sorted((int(re.search(r'\d+', c).group()), c) for c in grp)
                med = {}
                for cyc, name in cycles:
                    arr = grp[name]['data'][()].astype(np.float64)
                    good, keep = _good_reps(arr)
                    med[cyc] = np.median(good, axis=0) if len(good) else np.full(arr.shape[1], np.nan)
                    feats = [_chan_features(good[i]) for i in range(len(good))]
                    idx = np.flatnonzero(keep)
                    for k, i in enumerate(idx):
                        prow.append(dict(
                            面板=panel, 作动器=a, 接收器=r, 循环=cyc, 重复=int(i),
                            **feats[k],
                            人工标签=labels.get((panel, a, r, cyc, int(i)), ''),
                        ))
                    for i in np.flatnonzero(~keep):
                        rr_ = float(np.sqrt((arr[i] ** 2).mean()))
                        drop.append(dict(
                            面板=panel, 作动器=a, 接收器=r, 循环=cyc, 重复=int(i),
                            RMS_V=rr_,
                            人工标签=labels.get((panel, a, r, cyc, int(i)), ''),
                        ))
                    # 被剔除的坏重复只记数不记特征（特征无意义），便于统计
                    rr = np.array([x['rms'] for x in feats]) if feats else np.array([np.nan])
                    pk = np.array([x['peak'] for x in feats]) if feats else np.array([np.nan])
                    grow.append(dict(
                        面板=panel, 作动器=a, 接收器=r, 循环=cyc,
                        重复数=int(arr.shape[0]), 有效重复数=int(len(good)),
                        峰值_V=float(np.nanmedian(pk)),
                        RMS_V=float(np.nanmedian(rr)),
                        能量_V2s=float(np.nanmedian([x['energy'] for x in feats])),
                        峰时_us=float(np.nanmedian([x['env_peak_us'] for x in feats])),
                        包络重心_us=float(np.nanmedian([x['env_centroid_us'] for x in feats])),
                        SNR=float(np.nanmedian([x['snr'] for x in feats])),
                        重复CV=float(np.nanstd(rr) / np.nanmean(rr))
                        if len(rr) > 1 and np.nanmean(rr) > 0 else np.nan,
                    ))

                base = med.get(0)
                base_ok = base is not None and np.isfinite(base).all()
                g_row_base_rms = grow[-len(cycles)]['RMS_V'] if cycles else np.nan
                if base is None:
                    brow.append(dict(面板=panel, 作动器=a, 接收器=r, 有基线=0, 基线有效重复=np.nan,
                                     基线RMS=np.nan, 基线峰时_us=np.nan,
                                     基线SNR=np.nan, 说明='缺 cycle0'))
                else:
                    fb = _chan_features(base)
                    n0 = grow[-len(cycles)]['有效重复数']
                    brow.append(dict(
                        面板=panel, 作动器=a, 接收器=r, 有基线=int(base_ok),
                        基线有效重复=n0,
                        基线RMS=fb['rms'], 基线峰时_us=fb['env_peak_us'],
                        基线SNR=fb['snr'],
                        说明='' if n0 >= 2 else 'cycle0 只有 1 次有效重复，基线不稳',
                    ))

                # 以 cycle0 为参考的相关与频段（对每个循环）
                for g_row, cyc in zip(grow[-len(cycles):], [c for c, _ in cycles]):
                    x = med[cyc]
                    if not np.isfinite(x).all():
                        g_row.update({'xcorr_基线': np.nan, '时移_us': np.nan,
                                      '相对RMS': np.nan})
                        continue
                    bands = _band_ratio(x)
                    for i, nm in enumerate(('频段0_50k', '频段50_100k',
                                            '频段100_200k', '频段200_500k')):
                        g_row[nm] = bands[i]
                    if base_ok and base is not x:
                        cc, lag = _align(x, base)
                        g_row['xcorr_基线'] = cc
                        g_row['时移_us'] = lag / FS * 1e6
                        br = g_row_base_rms
                        g_row['相对RMS'] = g_row['RMS_V'] / br if br and br > 0 else np.nan
                    else:
                        g_row['xcorr_基线'] = 1.0 if base_ok else np.nan
                        g_row['时移_us'] = 0.0 if base_ok else np.nan
                        g_row['相对RMS'] = 1.0 if base_ok else np.nan

                if waves:
                    cols = {'循环': [c for c, _ in cycles],
                            't_us': np.arange(2001) / FS * 1e6}
                    for cyc, _ in cycles:
                        cols['c%d' % cyc] = med[cyc]
                    # 宽表：行 = 时间采样，列 = 循环（便于直接用 Excel/绘图看波形演进）
                    wdf = pd.DataFrame({'t_us': np.arange(2001) / FS * 1e6})
                    for cyc, _ in cycles:
                        wdf['c%d' % cyc] = med[cyc]
                    wdf.to_csv(os.path.join(wdir, 'wave_%s_act%d_recv%d.csv' % (panel, a, r)),
                               **CSV_KW)

    gdf = pd.DataFrame(grow).sort_values(['面板', '作动器', '接收器', '循环'])
    gdf.to_csv(os.path.join(d, 'groups.csv'), **CSV_KW)
    pdf = pd.DataFrame(prow).sort_values(['面板', '作动器', '接收器', '循环', '重复'])
    pdf.to_csv(os.path.join(d, 'reps.csv'), **CSV_KW)
    ddf = pd.DataFrame(drop).sort_values(['面板', '作动器', '接收器', '循环', '重复'])
    ddf.to_csv(os.path.join(d, 'reps_dropped.csv'), **CSV_KW)
    pd.DataFrame(brow).to_csv(os.path.join(d, 'baseline.csv'), **CSV_KW)
    say('  groups.csv         %d 组' % len(gdf))
    say('  reps.csv           %d 条有效信号（含人工标签 %d 条）'
        % (len(pdf), int((pdf['人工标签'].fillna('') != '').sum())))
    say('  reps_dropped.csv   %d 条近零幅值重复被剔除（占 %.0f%%）'
        % (len(ddf), 100.0 * len(ddf) / max(len(pdf) + len(ddf), 1)))
    say('  baseline.csv       %d 组；基线只有 1 次有效重复的 %d 组'
        % (len(brow), sum(1 for b in brow if str(b.get('说明', '')) != '')))

    # 每板 × 每循环的中位（跨 12 个通道）
    agg = (gdf.groupby(['面板', '循环'])
           .agg(通道数=('接收器', 'size'),
                相对RMS=('相对RMS', 'median'),
                xcorr_基线=('xcorr_基线', 'median'),
                包络重心_us=('包络重心_us', 'median'),
                峰时_us=('峰时_us', 'median'),
                SNR=('SNR', 'median'),
                重复CV=('重复CV', 'median'))
           .reset_index())
    agg.to_csv(os.path.join(d, 'panel_cycle.csv'), **CSV_KW)
    say('  panel_cycle.csv    %d 行' % len(agg))
    return gdf


# ----------------------------------------------------------------- 入口
TASKS = ('index', 'features')


def main(argv=None):
    ap = argparse.ArgumentParser(description='morpho_50khz_full.h5 → CSV')
    ap.add_argument('--h5', default=None,
                    help='派生集 HDF5（默认 data_root/morpho/%s）' % H5_NAME)
    ap.add_argument('--out', default=None, help='输出根（默认 data_root/morpho/processed）')
    ap.add_argument('--only', nargs='+', choices=TASKS, default=list(TASKS))
    ap.add_argument('--waves', action='store_true', help='额外导出波形宽表（约 60 MB）')
    a = ap.parse_args(argv)

    h5 = a.h5 or default_h5()
    out = a.out or default_out()
    if not os.path.exists(h5):
        say('[错误] 找不到 %s' % h5)
        return 2
    say('源文件 %s（%.1f MB）' % (h5, os.path.getsize(h5) / 2 ** 20))
    say('输出   %s\\%s' % (out, OUTDIR))
    say('任务   %s%s\n' % (', '.join(a.only), ' + waves' if a.waves else ''))

    t0 = time.time()
    labels = load_labels(h5)
    say('人工标签 %d 条' % len(labels))
    with h5py.File(h5, 'r') as f:
        ensure(out)
        if 'index' in a.only:
            say('[1/2] 索引')
            task_index(f, out)
        if 'features' in a.only:
            say('[2/2] 特征')
            task_features(f, out, labels, waves=a.waves)
    say('\n完成：%.1f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
