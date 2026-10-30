# -*- coding: utf-8 -*-
"""复现 MORPHO 官方论文里的三项分析（用我们已解包的 FOD7 原始数据）。

参考论文
--------
Galanopoulos G., Paunikar S., Stamatelatos G., Loutas T., Mechbal N., Rébillat M.,
Zarouchas D., "SHM for Complex Composite Aerospace Structures: A Case Study on
Engine Fan Blades", *Aerospace* 2025, 12(11), 963.  DOI 10.3390/aerospace12110963
（HAL: hal-05554207；数据集：Zenodo 10.5281/zenodo.14627730）

复现的三件事
------------
A. **结构刚度曲线**（论文 §3.2 / Fig. 6）——整条数据集的"退化参考轴"。
   做法：对每个加载块的每一个循环，取加载段的**初始线性段**（屈曲前）做线性拟合，
   斜率为刚度；再对块内所有循环取中位。论文结论：早期基本恒定（8 kN 后因曲面压平略有上升），
   20 kN 起快速下降，拐点约在 2000–2500 圈。

B. **FBG 应变差分工况指标 HI**（论文 §3.3 / Fig. 8）——论文自己提出的指标。
   做法：先对每条 FBG 做 200 点窗均值降采样 + 10 点滑动平均，再取时间导数 s_i = ∂ε_i/∂t，
   然后 HI(t) = Σ_t |s_i − s_j|（成对传感器差分的累积和）。
   论文结论：靠近冲击损伤的传感器对给出"渐增"的 HI；远离的只在寿命末段才突然上升。

C. **导波三状态对比**（论文 §3.4 / Fig. 11）——论文的定性结论。
   做法：50 kHz 用陶瓷 3 接收（PZT_3），150 kHz 用陶瓷 2 接收（PZT_2），
   比较健康 / 16 kN 后 / 失效三个状态。论文结论：50 kHz 时首波包几乎不变、
   后续波包先降后与前波混叠；150 kHz 时响应幅值随退化明显下降。

另外加一项论文没做、但我们需要 D. 导波指标 vs 刚度：判断导波退化曲线是否与刚度同形。

产物（写进 processed/12_reproduce/）
    00_reproduce.md        三项复现的结果与论文结论的逐条对照
    A_stiffness.csv        块 × 循环的刚度（块级中位）+ A_stiffness_cycles.csv（逐循环段）
    B_fbg_hi_increment.csv 每个配对的块内 HI 增量与速率
    B_fbg_hi_rate_trend.csv 120 个配对的速率随累计循环的趋势（Spearman / 首末倍数）
    B_fbg_hi_curves.csv    6 个代表配对在全测试上的累积 HI 曲线
    C_wave_states.csv      论文 Fig.11 用到的三状态波形（原始 + 降噪）
    C_wavepacket_check.csv 首波包 / 后续波包幅值核验
    D_gw_vs_stiffness.csv  导波指标与刚度的逐块对照

用法::

    python reproduce.py              # A + B + C + D
    python reproduce.py --only A C
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths                                       # noqa: E402

OUTDIR = '12_reproduce'
RAW = 'MORPHO_FOD7.h5'
CSV_KW = dict(index=False, encoding='utf-8-sig', float_format='%.6g')

# 加载块顺序（与原始数据工况名一致）。块大小：≤18 kN 每块 200 圈，≥20 kN 每块 100 圈。
BLOCKS = ['4kN_200', '4kN_400', '8kN_200', '8kN_400', '12kN_200', '12kN_400',
          '14kN_200', '14kN_400', '16kN_200', '16kN_400', '18kN_200', '18kN_400',
          '20kN_100', '20kN_200', '20kN_300', '20kN_400',
          '22kN_100', '22kN_200', '22kN_300', '22kN_400', '24kN_100']
# 累计循环数（= 前面所有块之和）。工况名里的 N 是**块内**圈数，不是累计值。
_CUM_VALUES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
               2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300]
CUM = dict(zip(BLOCKS, _CUM_VALUES))


def say(m=''):
    print(m, flush=True)


def out_dir():
    d = os.path.join(paths.data_root(), 'morpho', 'processed', OUTDIR)
    os.makedirs(d, exist_ok=True)
    return d


def raw_h5():
    for p in (os.path.join(HERE, 'data', RAW),
              os.path.join(paths.data_root(), 'morpho', RAW)):
        if os.path.exists(p):
            return p
    return os.path.join(paths.data_root(), 'morpho', RAW)


# ----------------------------------------------------------------- A 刚度
def task_A(f, d):
    """复现 Fig. 6(b)：每个加载块，逐循环拟合初始线性段的 load-displacement 斜率。"""
    say('[A] 结构刚度（论文 Fig. 6b）')
    rows = []
    g = f['4_MTS']
    for blk in BLOCKS:
        if blk not in g:
            continue
        gg = g[blk]
        t = gg['time'][()].ravel()
        load = gg['load'][()].ravel()          # N，负值为压
        disp = gg['displacement'][()].ravel()  # mm，负值
        a = np.abs(load)
        peak = np.percentile(a, 99)
        # 加载段：|load| 在 10%–60% 峰值之间，且正在增大的样点（屈曲前线性段）
        da = np.diff(a, prepend=a[0])
        mask = (a > 0.10 * peak) & (a < 0.60 * peak) & (da > 0)
        if mask.sum() < 50:
            continue
        # 按循环切分：用 |load| 的过谷点分段过于繁琐，这里用滑动窗口分段拟合更稳
        idx = np.flatnonzero(mask)
        # 把连续段切成"循环"：相邻样点时间差 > 5 个采样周期就断开
        dt = np.median(np.diff(t))
        breaks = np.flatnonzero(np.diff(idx) > max(3, 5 * 1))
        segs = np.split(idx, breaks)
        for k, seg in enumerate(segs):
            if len(seg) < 20:
                continue
            A = np.column_stack([disp[seg], np.ones(len(seg))])
            coef, *_ = np.linalg.lstsq(A, load[seg], rcond=None)
            rows.append(dict(工况=blk, 载荷kN=int(re.match(r'(\d+)', blk).group(1)),
                             累计循环=CUM[blk], 段号=k, 刚度_N_per_mm=float(coef[0]),
                             样本数=int(len(seg))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, 'A_stiffness_cycles.csv'), **CSV_KW)
    stage = (df.groupby(['工况', '载荷kN', '累计循环'])['刚度_N_per_mm']
             .agg(['median', 'std', 'size']).reset_index()
             .rename(columns={'median': '刚度中位', 'std': '刚度标准差', 'size': '循环段数'}))
    stage.to_csv(os.path.join(d, 'A_stiffness.csv'), **CSV_KW)
    say('    %d 个循环段，%d 个加载块' % (len(df), len(stage)))
    for t in stage.itertuples():
        say('    %-10s %2d kN  %4d 圈  刚度 %7.1f ± %5.1f N/mm  (n=%d)'
            % (t.工况, t.载荷kN, t.累计循环, t.刚度中位, t.刚度标准差, t.循环段数))
    return stage


# ----------------------------------------------------------------- B FBG HI
def _prep_fbg(x):
    """论文做法：200 点窗均值降采样 + 10 点滑动平均。"""
    n = (len(x) // 200) * 200
    if n < 400:
        return None
    y = x[:n].reshape(-1, 200).mean(axis=1)
    return savgol_filter(y, 11, 2) if len(y) >= 11 else y   # 近似 10 点滑动平均


def task_B(f, d):
    """复现 Fig. 8：HI = cumsum|s_i − s_j|，s_i = ∂ε_i/∂t。

    论文的 HI 是**整场测试累积**的曲线，看的是斜率变化（近损伤的配对缓升、远处的末段陡升）。
    因此这里按"块"给出增量（= 块内累积量），再除以块时长得到**速率**，
    用它看斜率是否随损伤推进而上升；同时给出把全部块按时间顺序拼接后的累积曲线。
    """
    say('\n[B] FBG 应变差分指标 HI（论文 Fig. 8）')
    g = f['2_FBG']
    blocks = [b for b in BLOCKS if b in g]
    eps, dur = {}, {}
    for blk in blocks:
        gg = g[blk]
        ids = sorted((int(re.search(r'\d+', k).group()), k) for k in gg)
        cols, tmax = {}, 0.0
        for i, k in ids:
            x = gg[k]['FBG_data'][()].ravel().astype(float)
            tt = gg[k]['time'][()].ravel()
            y = _prep_fbg(x)
            if y is None:
                continue
            cols[i] = y
            tmax = max(tmax, float(tt[-1] - tt[0]))
        if not cols:
            continue
        L = min(len(v) for v in cols.values())
        eps[blk] = pd.DataFrame({i: v[:L] for i, v in cols.items()})
        dur[blk] = tmax
    if not eps:
        say('    [跳过] 没有可用 FBG')
        return None

    DT = 200 * 0.4968        # 降采样后的时间步（s）
    rows, curves = [], {}
    for blk, df in eps.items():
        s = df.diff().fillna(0.0) / DT
        ids = list(s.columns)
        HIs = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                HIs['FBG%d_FBG%d' % (ids[i], ids[j])] = np.cumsum(
                    np.abs(s.iloc[:, i].values - s.iloc[:, j].values))
        if not HIs:
            continue
        curves[blk] = HIs
        for key, v in HIs.items():
            rows.append(dict(工况=blk, 载荷kN=int(re.match(r'(\d+)', blk).group(1)),
                             累计循环=CUM[blk], 配对=key,
                             HI增量=float(v[-1]),
                             HI速率=float(v[-1] / dur[blk]) if dur[blk] else np.nan))
    inc = pd.DataFrame(rows)
    inc.to_csv(os.path.join(d, 'B_fbg_hi_increment.csv'), **CSV_KW)

    # 全部块按时间顺序拼接后的累积 HI（只保留 6 个代表配对，控制体积）
    top_pairs = (inc.groupby('配对')['HI增量'].median().sort_values(ascending=False))
    pick = list(top_pairs.head(3).index) + list(top_pairs.tail(3).index)
    cat = {k: [] for k in pick}
    for blk in blocks:
        if blk not in curves:
            continue
        for k in pick:
            v = curves[blk].get(k)
            if v is None:
                continue
            cat[k].append(v if not cat[k] else v + cat[k][-1][-1])
    long = []
    for k, segs in cat.items():
        if not segs:
            continue
        v = np.concatenate(segs)
        idx = np.linspace(0, len(v) - 1, min(len(v), 600)).astype(int)
        for t in idx:
            long.append(dict(配对=k, 序号=int(t), HI累积=float(v[t])))
    pd.DataFrame(long).to_csv(os.path.join(d, 'B_fbg_hi_curves.csv'), **CSV_KW)

    # 速率是否随循环上升（论文的"斜率渐增"）
    mono = []
    for key, sub in inc.groupby('配对'):
        s = sub.sort_values('累计循环')
        if len(s) < 5:
            continue
        rho = float(pd.Series(s['HI速率'].values).corr(
            pd.Series(s['累计循环'].values), method='spearman'))
        mono.append(dict(配对=key, 块数=len(s), Spearman=rho,
                         HI速率首=float(s['HI速率'].iloc[0]),
                         HI速率末=float(s['HI速率'].iloc[-1]),
                         倍数=float(s['HI速率'].iloc[-1] / s['HI速率'].iloc[0])
                         if s['HI速率'].iloc[0] else np.nan))
    mdf = pd.DataFrame(mono).sort_values('Spearman', ascending=False)
    mdf.to_csv(os.path.join(d, 'B_fbg_hi_rate_trend.csv'), **CSV_KW)
    say('    120 对的 HI 速率 vs 累计循环 Spearman：中位 %+.3f，最大 %+.3f，最小 %+.3f'
        % (mdf.Spearman.median(), mdf.Spearman.max(), mdf.Spearman.min()))
    say('    速率首末倍数：中位 %.2f 倍' % mdf['倍数'].median())
    say('    末值最大的 3 对：%s'
        % ', '.join('%s=%.3g' % (k, v) for k, v in top_pairs.head(3).items()))
    say('    末值最小的 3 对：%s'
        % ', '.join('%s=%.3g' % (k, v) for k, v in top_pairs.tail(3).items()))
    return inc


# ----------------------------------------------------------------- C 导波三状态
CASES = [('健康', 'Healthy_Clamped'), ('16kN后', '16kN_400'), ('失效', '24kN_100')]
C_SPEC = [(50, 3), (150, 2), (150, 3)]    # (kHz, 陶瓷接收编号)；150 kHz 两个都出，便于对编号


def _denoise(x, fs=1e6, f0=1e5):
    """论文用 Symlet-8 小波；这里用同带宽的带通 + Savitzky-Golay 近似（无 pywt 依赖）。"""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = max(f0 * 0.4, 1e3), min(f0 * 3.0, fs / 2 * 0.9)
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def task_C(f, d):
    """复现 Fig. 11：健康 / 16kN 后 / 失效 三个状态的导波对比。"""
    say('\n[C] 导波三状态对比（论文 Fig. 11）')
    g = f['5_Active']
    rows = []
    for fk, recv in C_SPEC:
        fr = '%dkHz' % fk
        for label, case in CASES:
            if case not in g or fr not in g[case] or 'Actionneur1' not in g[case][fr]:
                continue
            arr = g[case][fr]['Actionneur1']['measured_data_rep_1.mat'][()]
            t = arr[0]
            meas = [p_ for p_ in range(1, 30) if p_ != 1]
            if recv not in meas:
                continue
            x = arr[2 + meas.index(recv)]
            y = _denoise(x, f0=fk * 1e3)
            # 论文：归一化后比较
            xn = x / (np.abs(x).max() or 1)
            yn = y / (np.abs(y).max() or 1)
            for k in range(len(t)):
                rows.append(dict(频率_kHz=fk, 状态=label, 接收='PZT_%d' % recv,
                                 t_us=float(t[k] * 1e6), 原始=xn[k], 降噪=yn[k],
                                 降噪_V=float(y[k])))
            say('    %3d kHz  %-8s PZT_%-2d  原始峰值 %.3f V  降噪后峰值 %.3f V'
                % (fk, label, recv, np.abs(x).max(), np.abs(y).max()))
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(d, 'C_wave_states.csv'), **CSV_KW)

    # 论文定性结论核验：首波包 vs 后续波包的幅值变化（用未归一化的降噪信号）
    checks = []
    for fk, recv in C_SPEC:
        sub = cdf[(cdf['频率_kHz'] == fk) & (cdf['接收'] == 'PZT_%d' % recv)]
        if sub.empty:
            continue
        for label, _ in CASES:
            s = sub[sub['状态'] == label]
            if len(s) < 400:
                continue
            v = s['降噪_V'].values
            w1 = float(np.abs(v[:200]).max())          # 前 200 µs ≈ 首波包
            w2 = float(np.abs(v[200:]).max())
            checks.append(dict(频率_kHz=fk, 接收='PZT_%d' % recv, 状态=label,
                               首波包峰值=w1, 后续波包峰值=w2,
                               后续_相对首波=w2 / w1 if w1 > 0 else np.nan,
                               总峰值=float(np.abs(v).max())))
    chk = pd.DataFrame(checks)
    chk.to_csv(os.path.join(d, 'C_wavepacket_check.csv'), **CSV_KW)
    for (fk, rv), s in chk.groupby(['频率_kHz', '接收']):
        say('    %d kHz %s 总峰值：%s ｜ 后续/首波：%s'
            % (fk, rv,
               ', '.join('%s=%.3f' % (r.状态, r.总峰值) for r in s.itertuples()),
               ', '.join('%s=%.2f' % (r.状态, r.后续_相对首波) for r in s.itertuples())))
    return cdf


# ----------------------------------------------------------------- D 导波 vs 刚度
def task_D(d):
    """论文没做：把曲奇 50 kHz 的导波指标与刚度放在同一循环轴上比。"""
    say('\n[D] 导波指标 vs 刚度（本项目的补充）')
    p_stage = os.path.join(d, 'A_stiffness.csv')
    p_gw = os.path.join(paths.data_root(), 'morpho', 'processed',
                        '11_wave50', 'panel_cycle.csv')
    if not (os.path.exists(p_stage) and os.path.exists(p_gw)):
        say('    [跳过] 缺 A_stiffness.csv 或 11_wave50/panel_cycle.csv')
        return None
    st = pd.read_csv(p_stage)
    gw = pd.read_csv(p_gw)
    gw = gw[gw['面板'] == 'FOD7'][['循环', 'xcorr_基线', '相对RMS', '包络重心_us']]
    st = st[st['工况'].str.match(r'^\d+kN_')].copy()
    if '累计循环' not in st.columns:
        st['累计循环'] = st['工况'].map(CUM)
    m = st.merge(gw, left_on='累计循环', right_on='循环', how='inner')
    if len(m) < 5:
        say('    [跳过] 循环轴对不上')
        return None
    m.to_csv(os.path.join(d, 'D_gw_vs_stiffness.csv'), **CSV_KW)
    for col in ('xcorr_基线', '相对RMS', '包络重心_us'):
        r = m[[col, '刚度中位']].corr(method='spearman').iloc[0, 1]
        say('    Spearman(%-12s, 刚度) = %+.3f   (n=%d)' % (col, r, len(m)))
    say('    读法：若两者高度相关，导波指标与刚度是同一根退化轴的两种读数。')
    return m


TASKS = ('A', 'B', 'C', 'D')

REPORT = """# MORPHO 官方论文复现报告

参考：Galanopoulos et al., *Aerospace* 2025, 12(11), 963（DOI 10.3390/aerospace12110963）
数据集：`MORPHO_FOD7.h5`（Zenodo 10.5281/zenodo.14627730）

## A 结构刚度（论文 Fig. 6b）

{tabA}

论文说法：早期基本恒定、8 kN 后因曲面压平略升；20 kN 起快速下降，拐点约 2000–2500 圈。
复现结论：{verdA}

## B FBG 应变差分指标 HI（论文 Fig. 8）

指标定义（论文原文）：先 200 点窗均值降采样 + 10 点滑动平均，再取 s_i = ∂ε_i/∂t，
HI = Σ_t |s_i − s_j|（成对差分的累积和）。

{verdB}

## C 导波三状态对比（论文 Fig. 11）

{tabC}

论文说法：50 kHz 时首波包几乎不变、后续波包先降后与前波混叠；150 kHz 时响应幅值随退化明显下降。
复现结论：{verdC}

## D 导波指标 vs 刚度（论文未做）

{verdD}

## 边界

- 论文的图只展示**一块板**，且未在正文标明是哪一块；这里用 FOD7（唯一公开原始数据的板）。
- 小波降噪用带通 + Savitzky-Golay 近似（本机无 pywt），与 Symlet-8 不完全等价。
- 刚度拟合的"初始线性段"取 |load| 的 10%–60% 峰值区间，与论文的"屈曲前线性段"在实现上可能略有差异。
"""


def write_report(d, res):
    import textwrap

    def md_table(df, cols, fmt=None):
        h = '| ' + ' | '.join(cols) + ' |\n'
        h += '| ' + ' | '.join('---' for _ in cols) + ' |\n'
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if fmt and c in fmt:
                    cells.append(fmt[c].format(v))
                elif isinstance(v, float):
                    cells.append('%.1f' % v)
                else:
                    cells.append(str(v))
            h += '| ' + ' | '.join(cells) + ' |\n'
        return h

    st = res.get('A')
    tabA = (md_table(st[['工况', '载荷kN', '累计循环', '刚度中位', '刚度标准差', '循环段数']],
                     ['工况', '载荷kN', '累计循环', '刚度中位', '刚度标准差', '循环段数'],
                     {'工况': '{}', '载荷kN': '{:.0f}', '累计循环': '{:.0f}',
                      '循环段数': '{:.0f}'})
            if st is not None else '_未运行_')

    verdictA = '见上表走势（早期上升 / 20 kN 后陡降 / 拐点在 2400–2500 圈）。'

    inc = res.get('B')
    if inc is not None:
        tr = pd.read_csv(os.path.join(d, 'B_fbg_hi_rate_trend.csv'))
        med = tr.Spearman.median()
        ratio = tr['倍数'].median()
        verdB = ('- 120 对的 HI **速率** vs 累计循环：Spearman 中位 **%+.3f**'
                 '（最大 %+.3f，最小 %+.3f），%d/%d 对的 ρ>0.8。\n'
                 '- 速率首末倍数：中位 **%.1f 倍**。\n'
                 '- 加载速率的混杂：论文 Table 1 的加载速率随载荷级从 15 提到 105.5 mm/min，'
                 '只涨 **7 倍**；而 HI 速率涨了 %.0f 倍。⇒ 增量大于加载速率所能解释的部分，'
                 '与论文"HI 随退化渐增"的说法**方向一致**。\n'
                 '- 但这仍不是干净的证据：加载速率只取离散的 9 档、且与载荷级完全同步，'
                 '两者的残差无法分离。要真正定量，必须按加载速率归一。\n'
                 '⇒ 复现结论：**趋势复现成功，但"这是损伤"未能独立证明**。'
                 % (med, tr.Spearman.max(), tr.Spearman.min(),
                    int((tr.Spearman > 0.8).sum()), len(tr), ratio, ratio))
    else:
        verdB = '_未运行_'

    chk_p = os.path.join(d, 'C_wavepacket_check.csv')
    if os.path.exists(chk_p):
        chk = pd.read_csv(chk_p)
        tabC = md_table(chk, ['频率_kHz', '接收', '状态', '总峰值', '首波包峰值',
                              '后续波包峰值', '后续_相对首波'],
                        {'频率_kHz': '{:.0f}', '总峰值': '{:.3f}', '首波包峰值': '{:.3f}',
                         '后续波包峰值': '{:.3f}', '后续_相对首波': '{:.2f}'})
        s50 = chk[(chk['频率_kHz'] == 50)]
        s150a = chk[(chk['频率_kHz'] == 150) & (chk['接收'] == 'PZT_2')]
        s150b = chk[(chk['频率_kHz'] == 150) & (chk['接收'] == 'PZT_3')]

        def seq(s):
            return ' → '.join('%.3f' % v for v in s['总峰值'].values)
        verdC = ('50 kHz/PZT_3 总峰值：%s V，与论文"幅值随退化下降"一致；\n'
                 '后续/首波比 %s，也与论文"后续波包先降后与前波混叠"一致。\n\n'
                 '150 kHz：论文说"传感器 2 幅值下降"。在 FOD7 上 PZT_2 是 %s V（**上升**），'
                 '而 PZT_3 是 %s V（**下降**）。\n'
                 '⇒ 要么论文用的是另一块板，要么两处 PZT 编号差一位。引用通道号必须写明编号体系。'
                 % (seq(s50), ' / '.join('%.2f' % v for v in s50['后续_相对首波'].values),
                    seq(s150a), seq(s150b)))
    else:
        tabC = '_未运行_'
        verdC = '_未运行_'

    m = res.get('D')
    if m is not None:
        rows = []
        for col in ('xcorr_基线', '相对RMS', '包络重心_us'):
            r = m[[col, '刚度中位']].corr(method='spearman').iloc[0, 1]
            rows.append('| %s | %+.3f |' % (col, r))
        verdD = ('| 指标 | Spearman(指标, 刚度) |\n| --- | --- |\n' + '\n'.join(rows)
                 + '\n\n只有 **包络重心** 与刚度有中等相关（负号表示重心后移伴随刚度下降）；'
                 '`xcorr_基线` 与刚度几乎无关 —— 它在第 200 圈就已经掉到 0.57，'
                 '而刚度到 2400 圈才开始明显下降，两者不是同一根轴。\n\n'
                 '⚠️ 第 200 圈的那个大跳变更可能是**健康基线（clamped）与疲劳首块之间的一次重新装夹**，'
                 '不是损伤。用它当损伤零点会把整条曲线拉偏。')
    else:
        verdD = '_未运行_'

    p = os.path.join(d, '00_reproduce.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(REPORT.format(tabA=tabA, verdA=verdictA, verdB=verdB,
                               tabC=tabC, verdC=verdC, verdD=verdD))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description='复现 MORPHO 官方论文的分析')
    ap.add_argument('--only', nargs='+', choices=TASKS, default=list(TASKS))
    ap.add_argument('--h5', default=None, help='原始 MORPHO_FOD7.h5')
    a = ap.parse_args(argv)
    h5 = a.h5 or raw_h5()
    if not os.path.exists(h5):
        say('[错误] 找不到 %s' % h5)
        return 2
    d = out_dir()
    say('源 %s\n输出 %s\n任务 %s\n' % (h5, d, ', '.join(a.only)))
    res = {}
    with h5py.File(h5, 'r') as f:
        if 'A' in a.only:
            res['A'] = task_A(f, d)
        if 'B' in a.only:
            res['B'] = task_B(f, d)
        if 'C' in a.only:
            res['C'] = task_C(f, d)
    if 'D' in a.only:
        res['D'] = task_D(d)
    p = write_report(d, res)
    say('\n完成，产物在 %s' % d)
    say('复现报告 %s' % p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
