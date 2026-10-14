"""PZT 跨机构一致性：把各组的退相关累积量放到同一条「寿命分数」轴上。

口径（与 AE 侧 cum_hits t85 **完全同构**，这是刻意的）：
    D_i    = 相邻两个 PZT 状态之间、直达波群紧窗内 56 条路径的退相关中位
    D_cum  = cumsum(D_i)                       （构造上单调，对应 AE 的累积 hits）
    t50/t85 = D_cum 达到终值 50%/85% 时所处的寿命百分数

为什么用「相邻状态」而不是「相对 REF」：
    State_1 取在加载/热瞬态之前，拿它当基准会把整个寿命压平（实测：前 15 状态掉到 0.3，
    其后 165 状态一直平在 0.1）。相邻差分天然免疫基准漂移，且它是**因果**的。

寿命轴（两所大学用不同来源，但都是绝对墙钟）：
    Lab B（C3/C4）  l1/results/_l1cyc_<gid>.npz      t_epoch -> cycle，n_f 同文件
    Lab A 中的 C2   l1_time_align.dfos_anchor()      wall_s  -> cyc_seg
    Lab A 中的 C1   无绝对墙钟锚（只有相对时间的 FBG 块锚）-> 本脚本不纳入

状态集合：一个组下**所有** PZT-N 会话合并（它们共享同一绝对时钟后按时间排序），
    而不是挑一个会话 —— 否则 L1-25（3 个会话）与 L1-35（8 个会话）会被截断。

只写 l1/results/pzt_crosslab.csv 与 pzt_crosslab_summary.csv。
"""
import csv
import datetime as dt
import importlib.util
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

FREQ = '250kHz_5cycles'
REP = 1
MIN_STATES = 12           # 少于 12 个状态无法构成有意义的累积曲线

# 抽样步长。全量运行受限于外接盘的**首次访问成本**（实测 86 ms/文件，
# 而缓存命中只要 0.7 ms）—— 1972 个状态 × 8 个文件要 22.5 分钟。
# stride = k 表示每 k 个状态只读一个；**寿命序号仍按完整序列计**，故寿命轴不受影响。
# 只影响 D（相邻状态改为相邻**被抽样**状态）与逐点采样密度，不影响斜率。
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1

# 寿命轴：**采集设计文档**里的 guided-wave 测量间隔（比时间戳可靠，不依赖任何时钟）
#   Lab A (C1, L1-03.pdf)："Guided waves Yes 5000 cycles"
#   Lab B (C3/C4, L1-xx.txt)："PZT measurements are performed every 5200 cycle"
INTERVAL = {'A': 5000.0, 'B': 5200.0}

BATCH = {}
for g in ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-23']:
    BATCH[g] = ('C1', 'A')
for g in ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60']:
    BATCH[g] = ('C2', 'A')
for g in ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']:
    BATCH[g] = ('C3', 'B')
for g in ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35', 'L1-36', 'L1-41', 'L1-44']:
    BATCH[g] = ('C4', 'B')

TS = re.compile(r'State_(\d+)_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# 复用已验证的预处理：pzt_index 里的 rx_of / bandpass / prep / rn / 紧窗常数
PI = _load('pzt_index', os.path.join(HERE, 'pzt_index.py'))
TA = _load('l1_time_align', os.path.join(HERE, 'l1_time_align.py'))
RB = _load('reproduce_broer_l1', os.path.join(HERE, 'reproduce_broer_l1.py'))


def epoch_of(state):
    """状态目录名 -> 绝对墙钟秒。

    约定必须与 l1_time_align.read_markers 的 pd.Timestamp(wall_text).value/1e9 一致：
    pandas 把 naive 字符串当 UTC，故这里也用 tzinfo=UTC。两边同约定，时区自然抵消。
    """
    m = TS.match(state)
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[1:])
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()


def _sessnum(name):
    """PZT-N 的 N。字符串序会把 PZT-10 排到 PZT-2 前，必须按数字。"""
    try:
        return int(name.split('-')[1])
    except (IndexError, ValueError):
        return 0


def pzt_states(gid):
    """合并该组全部 PZT-N 会话的状态，按绝对时间排序 -> [(epoch, pztN, state)]。

    注意：这个排序只用于**取时间先后**（把多会话串成一条序列），
    真正的主寿命轴用的是「DAM 状态序号 × 采集间隔 / n_f」，不依赖时刻。
    """
    gdir = os.path.join(HERE, gid)
    if not os.path.isdir(gdir):
        return []
    out = []
    for e in sorted(os.scandir(gdir), key=lambda x: (_sessnum(x.name), x.name)):
        if not (e.is_dir() and e.name.startswith('PZT')):
            continue
        for s in os.scandir(e.path):
            if not (s.is_dir() and s.name.startswith('State_')):
                continue
            if not TS.match(s.name):
                continue
            p = os.path.join(s.path, FREQ, 'Actionneur1', 'measured_data_rep_%d.mat' % REP)
            if os.path.exists(p):
                out.append((epoch_of(s.name), e.name, s.name))
    out.sort()
    return out


ROW = re.compile(
    r"'([^']*)'\s+'([^']*)'\s+\[\s*([-\d.eE]+|NaN)\s+([-\d.eE]+|NaN)\s*\]\s+"
    r"\[\s*([-\d.eE]+|NaN)\s+([-\d.eE]+|NaN)\s*\]\s+([-\d.eE]+|NaN)\s+"
    r"([-\d.eE]+|NaN)\s+'([^']*)'")


def dam_order(gid):
    """按会话、按表内顺序列出标为 DAM 的状态 -> [(epoch, pztN, state)]。

    类别取自各组自带 structure_info.m 的 damage_info 表（每行 state 名 + 'REF'/'DAM'）。
    同一会话内表内顺序即采集顺序（实测 State_1, State_2, ...）。
    """
    gdir = os.path.join(HERE, gid)
    if not os.path.isdir(gdir):
        return []
    out = []
    for e in sorted(os.scandir(gdir), key=lambda x: (_sessnum(x.name), x.name)):
        if not (e.is_dir() and e.name.startswith('PZT')):
            continue
        p = os.path.join(e.path, 'structure_info.m')
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8', errors='replace') as fh:
            txt = fh.read()
        for m in ROW.finditer(txt):
            state, cat = m.group(1), m.group(9)
            if cat.upper() != 'DAM' or not TS.match(state):
                continue
            f = os.path.join(e.path, state, FREQ, 'Actionneur1', 'measured_data_rep_%d.mat' % REP)
            if os.path.exists(f):
                out.append((epoch_of(state), e.name, state))
    # 按状态**时间戳**排序，而不是按会话名：多会话组只有这样才能得到正确的先后。
    # （字符串序会把 PZT-10 排到 PZT-2 之前，实测使 L1-35 的寿命轴错乱）
    out.sort(key=lambda r: (r[0], _sessnum(r[1])))
    return out


def nf_of(gid, lab):
    """n_f：Lab B 取自 _l1cyc_*.npz；Lab A 取自 l1_hi_ae.csv。"""
    if lab == 'B':
        p = os.path.join(RES, '_l1cyc_%s.npz' % gid)
        if os.path.exists(p):
            with np.load(p, allow_pickle=True) as z:
                return float(z['n_f'])
        return np.nan
    p = os.path.join(RES, 'l1_hi_ae.csv')
    if os.path.exists(p):
        with open(p, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['gid'] == gid:
                    return float(r['n_f'])
    # L1-23 不在 l1_hi_ae.csv：其 AE 是 4.11 GB 事件级文件，单独流式处理
    p23 = os.path.join(RES, 'l1_hi_l123.csv')
    if os.path.exists(p23):
        with open(p23, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['gid'] == gid:
                    return float(r['n_f'])
    return np.nan


def life_map(gid, lab, nf):
    """返回 (墙钟秒[], cycle[]) 供 np.interp 把绝对墙钟换算成寿命百分数。

    Lab B（C3/C4）：_l1cyc_*.npz 自带 t_epoch → cycle
    Lab A + 有 dfos_anchor.csv（C2）：dfos_anchor 的 wall_s → cyc_seg
    Lab A 其余（C1）：两级链 —— 墙钟 → AE 相对时间（Resume 锚，段内 1:1）
                      → cycle（FBG 块锚）
    """
    if lab == 'B':
        p = os.path.join(RES, '_l1cyc_%s.npz' % gid)
        if not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=True) as z:
            te = z['t_epoch'].astype(float)
            cy = z['cycle'].astype(float)
        o = np.argsort(te)
        return te[o], cy[o]
    if os.path.exists(os.path.join(HERE, gid, '%sdfos_anchor.csv' % gid)):
        anc = TA.dfos_anchor(gid, n_f=nf, mode='uniform')
        if anc is None or anc.empty:
            return None
        return (anc['wall_s'].to_numpy(float), anc['cyc_seg'].to_numpy(float))
    # --- C1：墙钟 -> AE 相对时间 -> cycle ---
    mk = TA.read_markers(gid)
    if mk.empty:
        return None
    res = mk[mk['kind'] == 'Resume'].dropna(subset=['wall_s', 'ae_s'])
    if not len(res):
        return None
    res = res.sort_values('wall_s')
    w = res['wall_s'].to_numpy(float)
    a = res['ae_s'].to_numpy(float)
    cyc = np.asarray(RB.time_to_cycle(gid, a), float)
    ok = np.isfinite(w) & np.isfinite(cyc)
    return (w[ok], cyc[ok]) if ok.sum() >= 2 else None


def signals(gdir, pztN, state, ba):
    """一个状态下 56 条路径的紧窗信号 + 相对幅值 + argmax TOF。"""
    out, amps, tofs = {}, {}, {}
    for k in range(1, 9):
        p = os.path.join(gdir, pztN, state, FREQ, 'Actionneur%d' % k,
                         'measured_data_rep_%d.mat' % REP)
        if not os.path.exists(p):
            return None, None, None
        tr = sio.loadmat(p)['Time_Response']
        drv = float(np.abs(tr[:, 1]).max())
        for j, m in enumerate(PI.rx_of(k)):
            x = PI.prep(tr[:, 2 + j], ba)
            out[(k, m)] = PI.rn(x)
            amps[(k, m)] = float(np.abs(x).max()) / max(drv, 1e-30)
            tofs[(k, m)] = (int(np.argmax(np.abs(x))) + PI.WS + PI.TRIM) / PI.FS * 1e6
    return out, amps, tofs


LAGMAX = 50          # ±25 µs，远大于观察到的漂移，避免搜到伪峰


def delay_us(x, ref):
    """x 相对 ref 的时延（µs）：互相关 + 抛物线亚样点插值。

    实测（L1-44 状态 145，作动器1/5/7 共 21 条路径 × 10 次重复）：
        同状态重复之间的 sd 中位  0.190 µs（最大 0.312）
        而 argmax 同类 sd 中位    1.378 µs（最大 15.06）
    即把估计噪声从「与漂移同量级（3~19 µs）」降到「远小于漂移」。
    """
    xc = np.correlate(x, ref, 'full')
    c = len(ref) - 1
    lo, hi = max(0, c - LAGMAX), min(len(xc), c + LAGMAX + 1)
    j = int(np.argmax(xc[lo:hi])) + lo
    if lo < j < hi - 1:
        a, b, cc = xc[j - 1], xc[j], xc[j + 1]
        den = a - 2 * b + cc
        off = 0.5 * (a - cc) / den if abs(den) > 1e-30 else 0.0
    else:
        off = 0.0
    return ((j - c) + off) / PI.FS * 1e6


def run_group(gid):
    batch, lab = BATCH[gid]
    st = dam_order(gid)
    n_sel = (len(st) + STRIDE - 1) // STRIDE
    if n_sel < MIN_STATES:
        return None, '抽样后状态不足 %d（全量 %d，stride=%d → %d）' % (
            MIN_STATES, len(st), STRIDE, n_sel)
    nf = nf_of(gid, lab)
    if not np.isfinite(nf):
        return None, '拿不到 n_f'
    dtc = INTERVAL[lab]
    lm = life_map(gid, lab, nf)          # 仅作交叉校验，允许为 None
    ba = PI.bandpass(float(FREQ.split('kHz')[0]))
    gdir = os.path.join(HERE, gid)

    rows, prev, tmpl = [], None, None
    for j, (ep, pztN, state) in enumerate(st, start=1):
        if (j - 1) % STRIDE:
            continue                      # 抽样：跳过，但 j 仍是完整序列中的真实序号
        sig, amps, tofs = signals(gdir, pztN, state, ba)
        if sig is None:
            continue
        if tmpl is None:
            tmpl = sig                     # 模板 = 首个 DAM 状态（只决定零点，不影响漂移形状）
        row = {'gid': gid, 'batch': batch, 'lab': lab, 'state': state, 'pzt': pztN,
               'idx': j, 'epoch': ep, 'cycle': j * dtc,
               'life_pct': min(j * dtc / nf * 100.0, 100.0),
               'tof_us': float(np.median(list(tofs.values()))),
               'tof_x_us': float(np.median([delay_us(sig[p], tmpl[p])
                                            for p in sig if p in tmpl])),
               'amp': float(np.median(list(amps.values())))}
        if lm is not None:
            row['life_pct_ts'] = float(np.interp(ep, lm[0], lm[1], left=np.nan,
                                                right=lm[1][-1]) / nf * 100.0)
        if prev is not None:
            row['D'] = float(np.median([1.0 - float(np.dot(sig[jj], prev[jj]))
                                        for jj in sig if jj in prev]))
        rows.append(row)
        prev = sig
    return rows, None


def t_at(x, cum, frac):
    tot = cum[-1]
    if not (tot > 0):
        return np.nan
    j = min(int(np.searchsorted(cum, frac * tot)), len(x) - 1)
    return float(x[j])


def metrics(gid, rows):
    """从逐状态记录算出该组的全部汇总指标。

    只依赖 (life_pct, D, tof_x_us, epoch, life_pct_ts)，**不读任何 .mat**，
    因此可直接对已写出的逐状态 CSV 复算（见 `sum` 模式）。
    """
    d = np.array([r.get('D', np.nan) for r in rows], float)
    x = np.array([r['life_pct'] for r in rows], float)
    ok = np.isfinite(d) & np.isfinite(x)
    x, d = x[ok], d[ok]
    keep = (x >= 0) & (x <= 100)
    x, d = x[keep], d[keep]
    o = np.argsort(x)
    x, d = x[o], d[o]
    cum = np.cumsum(d)
    s = {'gid': gid, 'batch': BATCH[gid][0], 'lab': BATCH[gid][1],
         'n_states': len(x), 'life_min': float(x.min()) if len(x) else np.nan,
         'life_max': float(x.max()) if len(x) else np.nan,
         'D_sum': float(cum[-1]) if len(x) else np.nan,
         't50': t_at(x, cum, 0.50), 't85': t_at(x, cum, 0.85),
         'tof_first_us': float(np.nanmedian([r['tof_x_us'] for r in rows][:5])),
         'tof_last_us': float(np.nanmedian([r['tof_x_us'] for r in rows][-5:]))}
    s['tof_drift_us'] = s['tof_last_us'] - s['tof_first_us']
    # ---- 漂移趋势：OLS 会被晚期重尾跳变拖累，主指标用 Theil-Sen ----
    # 实测晚期逐点跳变中位 0.755 µs，而早期只有 0.163 µs（损伤累积后波场变复杂）。
    # Theil-Sen 使 L1-13 从 OLS 0.0014（看似无漂移）变回 0.0045（CI 不含 0）。
    yt = np.array([r['tof_x_us'] for r in rows], float)[ok][keep][o]
    s['tof_slope_ols'] = float(np.polyfit(x, yt, 1)[0]) if len(x) > 2 else np.nan
    if len(x) > 2:
        tsv, _, tslo, tshi = stats.theilslopes(yt, x, 0.95)
        tau, pv = stats.kendalltau(x, yt)
    else:
        tsv = tslo = tshi = tau = pv = np.nan
    s['tof_slope_ts'] = float(tsv)
    s['tof_slope_ts_lo'] = float(tslo)
    s['tof_slope_ts_hi'] = float(tshi)
    s['kendall_tau'] = float(tau)
    s['kendall_p'] = float(pv)
    jd = np.abs(np.diff(yt))
    s['jump_early_us'] = float(jd[x[1:] <= 15.0].mean()) if (x[1:] <= 15.0).any() else np.nan
    s['jump_late_us'] = float(jd[x[1:] > 15.0].mean()) if (x[1:] > 15.0).any() else np.nan
    return s


def print_report(s, rows):
    print('  [%s/%s] 状态 %3d  寿命覆盖 %5.1f%%~%5.1f%%  t50=%5.1f  t85=%5.1f  '
          'dTOF=%+6.1f µs  TS斜率=%+.4f [%+.4f,%+.4f]'
          % (s['gid'], s['batch'], s['n_states'], s['life_min'], s['life_max'],
             s['t50'], s['t85'], s['tof_drift_us'], s['tof_slope_ts'],
             s['tof_slope_ts_lo'], s['tof_slope_ts_hi']))
    va = [(r['life_pct'], r['life_pct_ts']) for r in rows
          if r.get('life_pct_ts') is not None and np.isfinite(r.get('life_pct_ts', np.nan))]
    if len(va) >= 5:
        a1 = np.array([t[0] for t in va])
        a2 = np.array([t[1] for t in va])
        print('        轴校对（时钟轴 − 序号轴）：中位 %+5.1f pp  最大绝对差 %5.1f pp  '
              '秩相关 %+.3f'
              % (np.median(a2 - a1), np.max(np.abs(a2 - a1)),
                 float(np.corrcoef(a1, a2)[0, 1])))


def _read_rows(path):
    """读逐状态 CSV -> {gid: [rowdict]}，数值字段转 float。"""
    groups = {}
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if v == '':
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            groups.setdefault(r['gid'], []).append(row)
    return groups


def main(src_csv=None):
    """src_csv 非空时：不读 .mat，直接从已写出的逐状态 CSV 复算汇总。"""
    allrows, summ, skipped = [], [], []
    groups = _read_rows(src_csv) if src_csv else None
    for gid in sorted(BATCH, key=lambda g: (BATCH[g][1], BATCH[g][0], g)):
        if groups is not None:
            if gid not in groups:
                continue
            rows = groups[gid]
        else:
            rows, err = run_group(gid)
            if rows is None:
                skipped.append((gid, BATCH[gid][0], err))
                print('  [%s/%s] 跳过：%s' % (gid, BATCH[gid][0], err))
                continue
        s = metrics(gid, rows)
        summ.append(s)
        for r in rows:
            allrows.append(r)
        print_report(s, rows)

    os.makedirs(RES, exist_ok=True)
    # 复算模式与全量模式都写规范文件名；只有 stride>1 的迭代跑加后缀
    sfx = '' if (src_csv or STRIDE == 1) else '_s%d' % STRIDE
    fp1 = os.path.join(RES, 'pzt_crosslab%s.csv' % sfx)
    cols = ['gid', 'batch', 'lab', 'state', 'pzt', 'idx', 'epoch', 'cycle', 'life_pct',
            'life_pct_ts', 'D', 'tof_us', 'tof_x_us', 'amp']
    if src_csv:
        print('（复算模式：不重写逐状态表 %s）' % src_csv)
    else:
        with open(fp1, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in allrows:
                w.writerow(r)
        print('\n已写 %s（%d 行）' % (fp1, len(allrows)))
    fp2 = os.path.join(RES, 'pzt_crosslab_summary%s.csv' % sfx)
    scols = list(summ[0].keys()) if summ else ['gid']
    with open(fp2, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=scols, extrasaction='ignore')
        w.writeheader()
        for s in summ:
            w.writerow(s)
    print('\n已写 %s（%d 行）' % (fp1, len(allrows)))
    print('已写 %s（%d 组）' % (fp2, len(summ)))

    print('\n--- 逐批汇总（TOF 漂移的 Theil-Sen 斜率，µs / 100%% 寿命）---')
    for b in sorted({s['batch'] for s in summ}):
        v = [s['tof_slope_ts'] for s in summ if s['batch'] == b and np.isfinite(s['tof_slope_ts'])]
        sig = [s for s in summ if s['batch'] == b and np.isfinite(s['tof_slope_ts_lo'])
               and s['tof_slope_ts_lo'] > 0]
        if v:
            print('  %-4s n=%2d  斜率中位 %+.4f  范围 %+.4f~%+.4f  （组内跨 %.0f 倍）  '
                  'CI 下界>0 的 %d/%d'
                  % (b, len(v), np.median(v), min(v), max(v),
                     (max(v) / min(v)) if min(v) > 0 else float('nan'),
                     len(sig), len(v)))
    med = {b: np.median([s['tof_slope_ts'] for s in summ
                         if s['batch'] == b and np.isfinite(s['tof_slope_ts'])])
           for b in sorted({s['batch'] for s in summ})}
    allv = [s['tof_slope_ts'] for s in summ if np.isfinite(s['tof_slope_ts'])]
    nsig = sum(1 for s in summ if np.isfinite(s['tof_slope_ts_lo']) and s['tof_slope_ts_lo'] > 0)
    print('  全体 %d 组：斜率 > 0 的 %d 组；TS 95%% 置信下界 > 0 的 %d 组'
          % (len(allv), sum(1 for v in allv if v > 0), nsig))
    if len(med) > 1:
        print('  机构间差 = %.4f µs/100%%（相对 %.0f%%）；组内跨 %.0f 倍%s'
              % (max(med.values()) - min(med.values()),
                 (max(med.values()) - min(med.values())) / max(med.values()) * 100,
                 max(allv) / min(allv) if min(allv) > 0 else float('nan'),
                 ''))
    for gid, b, err in skipped:
        print('  跳过 %-7s (%s)：%s' % (gid, b, err))


if __name__ == '__main__':
    _a = sys.argv[1] if len(sys.argv) > 1 else None
    if _a == 'sum':
        main(os.path.join(RES, 'pzt_crosslab.csv'))
    else:
        main()
