# -*- coding: utf-8 -*-
"""`P2P_LOAD = 200 µε` 绝对阈值体检 —— 该判据成立吗？改成相对判据影响多大？

判据（本工具的核心）
--------------------
把 **AE 帧**按「最近 FBG 突发的 p2p」分箱，看每箱的 **AE 事件率**。
若 200 µε 真的分开了「加载 / 停机」，那么 p2p <= 200 的箱其事件率
应比加载箱低 **2 至 3 个数量级**（实测真停机段就是这么低，L1-24 = 0.001）。
反之若低 p2p 箱的事件率与加载箱同量级，说明这个绝对阈值在切别的东西
（程序块谱的低幅值段、部分失效通道、数据边界窗……），不能当加载判据。

口径与生产一致
--------------
  · 窗：`ae_cycle.scan_fbg`（140 数据行为一窗）；
  · 突发：相邻窗间隔 <= `BURST_GAP`，突发状态 = 其窗的**均值**（`_bursts` 同源）；
  · 帧 → 突发：取**最近**突发（`_ratio_on_grid` 的 holdgap 就是这么赋值的）。
另附 `dist <= MAX_CARRY` 的子集统计，以剥离「长停录段 + AE 静默守卫」的干扰。

输出
----
  results/l1_p2p_load_check.csv    每组每候选阈值的分箱与事件率
  results/_logs/l1_p2p_load_check.txt  人类可读报告

用法
----
    python l1/p2p_load_check.py                # 全部有帧且有 FBG 的组
    python l1/p2p_load_check.py --groups L1-41
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ae_cycle as ac                                          # noqa: E402

RES = ac.RES
LOGDIR = os.path.join(RES, '_logs')
os.makedirs(LOGDIR, exist_ok=True)
OUT_CSV = os.path.join(RES, 'l1_p2p_load_check.csv')
OUT_TXT = os.path.join(LOGDIR, 'l1_p2p_load_check.txt')
OUT_CH_TXT = os.path.join(LOGDIR, 'l1_p2p_channels.txt')
OUT_PF_TXT = os.path.join(LOGDIR, 'l1_p2p_profile.txt')
OUT_AL_TXT = os.path.join(LOGDIR, 'l1_p2p_align.txt')
OUT_AMP_TXT = os.path.join(LOGDIR, 'l1_p2p_amp.txt')

# 分箱上界（µε）；最后一箱为 +inf。低幅值段用密箱，加载段用疏箱。
EDGES = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, np.inf]
# 候选阈值（相对判据用组内分位，绝对判据用 µε）
ABS_THRS = [200.0, 100.0, 50.0, 25.0]
REL_QS = [0.02, 0.05, 0.10, 0.20]


def bursts_with_p2p(tf, pf, gap=ac.BURST_GAP):
    """按生产口径聚突发，返回 (突发代表时刻, 突发代表 p2p)。"""
    tb, pb = [], []
    i = 0
    while i < len(tf):
        j = i
        while j + 1 < len(tf) and tf[j + 1] - tf[j] <= gap:
            j += 1
        tb.append(float(tf[i:j + 1].mean()))
        pb.append(float(np.mean(pf[i:j + 1])))
        i = j + 1
    return np.asarray(tb), np.asarray(pb)


def nearest_burst(t_sec, tb):
    """每个 AE 帧 -> (最近突发的下标, 时间距离)。"""
    idx = np.searchsorted(tb, t_sec)
    left = np.clip(idx - 1, 0, len(tb) - 1)
    right = np.clip(idx, 0, len(tb) - 1)
    dl = np.abs(t_sec - tb[left])
    dr = np.abs(t_sec - tb[right])
    take_left = dl <= dr
    return np.where(take_left, left, right), np.minimum(dl, dr)


def analyse(gid):
    fr = ac._load_frames(gid, ac.FRAME_S)
    if fr is None:
        return None, '跳过：无 AE 帧表'
    frame, t_sec, n_hits = fr
    pts = ac.group_load(gid)
    if not pts:
        return None, '跳过：无 FBG'
    if n_hits is None:
        return None, '跳过：帧表无事件数'
    tf = np.asarray([p[0] for p in pts], dtype=float)
    pf = np.asarray([p[1] for p in pts], dtype=float)
    order = np.argsort(tf)
    tf, pf = tf[order], pf[order]

    tb, pb = bursts_with_p2p(tf, pf)
    near, dist = nearest_burst(t_sec, tb)
    p_ae = pb[near]                       # 每个 AE 帧拿到的 p2p
    nh = np.asarray(n_hits, dtype=float)
    good = np.isfinite(nh) & np.isfinite(p_ae)
    p_ae, nh, dist = p_ae[good], nh[good], dist[good]

    span_h = float(t_sec[-1] - t_sec[0]) / 3600.0
    n_f = ac.N_F.get(gid)
    quiet = dist <= ac.MAX_CARRY          # 剥离长停录段 + AE 静默守卫的干扰

    rec = {'gid': gid, 'n_win': len(tf), 'n_burst': len(tb), 'span_h': span_h,
           'n_f': n_f, 'n_frame': int(len(nh)),
           'p2p_q': {q: float(np.quantile(pf, q))
                     for q in (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9)},
           'frac_burst_le200': float(np.mean(pb <= 200.0)),
           'frac_frame_le200': float(np.mean(p_ae <= 200.0)),
           'bins': [], 'thrs': []}

    # ① 分箱事件率（判据体检）
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (p_ae > lo) & (p_ae <= hi)
        mm = m & quiet
        rec['bins'].append({
            'lo': lo, 'hi': hi, 'n': int(m.sum()), 'n_q': int(mm.sum()),
            'rate': float(np.median(nh[m])) if m.sum() >= 10 else float('nan'),
            'rate_q': float(np.median(nh[mm])) if mm.sum() >= 10 else float('nan'),
        })

    # ② 候选阈值：绝对 + 相对（相对分位取自**该组自身的 p2p 分布**）
    hi_ref = float(np.median(nh[p_ae > 800.0])) if (p_ae > 800.0).sum() >= 10 else np.nan
    cands = [('abs', t, t) for t in ABS_THRS]
    cands += [('rel', q, float(np.quantile(pb, q))) for q in REL_QS]
    for kind, key, thr in cands:
        lo_m = p_ae <= thr
        if lo_m.sum() < 10 or (~lo_m).sum() < 10:
            continue
        lo_rate = float(np.median(nh[lo_m]))
        hi_rate = float(np.median(nh[~lo_m]))
        frac_load = float(np.mean(p_ae > thr))            # 帧加权加载占比
        load_h = frac_load * span_h
        f_hz = (n_f / (load_h * 3600.0)) if (n_f and load_h > 0) else float('nan')
        rec['thrs'].append({
            'kind': kind, 'key': key, 'thr': thr,
            'frac_load': frac_load, 'load_h': load_h, 'f_hz': f_hz,
            'rate_lo': lo_rate, 'rate_hi': hi_rate,
            'ratio': (lo_rate / hi_rate) if hi_rate > 0 else np.nan,
            'ratio_ref': (lo_rate / hi_ref) if hi_ref > 0 else np.nan,
        })

    # ③ 低 p2p 突发在寿命里的位置（是否集中在某段）
    low = pb <= 200.0
    if low.sum():
        rel = (tb[low] - t_sec[0]) / max(t_sec[-1] - t_sec[0], 1.0)
        cnt = np.histogram(rel, bins=np.linspace(0, 1, 11))[0]
        rec['low_pos'] = '/'.join(str(int(c)) for c in cnt)
    else:
        rec['low_pos'] = ''
    return rec, None


def fmt(rec):
    L = []
    q = rec['p2p_q']
    L.append('=' * 78)
    L.append('%s  窗 %d / 突发 %d / 帧 %d' % (rec['gid'], rec['n_win'],
                                             rec['n_burst'], rec['n_frame']))
    L.append('  寿命跨度 %.1f h，n_f = %s' % (rec['span_h'], rec['n_f']))
    L.append('  窗 p2p 分位：  1%%=%.0f  5%%=%.0f  10%%=%.0f  25%%=%.0f  '
             '中位=%.0f  75%%=%.0f  90%%=%.0f'
             % (q[0.01], q[0.05], q[0.1], q[0.25], q[0.5], q[0.75], q[0.9]))
    L.append('  <= 200 µε 的占比：突发 %.1f%%，AE 帧 %.1f%%'
             % (100 * rec['frac_burst_le200'], 100 * rec['frac_frame_le200']))
    L.append('  分箱（AE 帧数 / 每帧事件数中位；括号内=限制 dist<=MAX_CARRY）：')
    for b in rec['bins']:
        tag = '[%.0f, %s)' % (b['lo'], ('inf' if not np.isfinite(b['hi']) else '%.0f' % b['hi']))
        L.append('    %-16s n=%-7d rate=%-8.2f (n=%-7d rate=%-.2f)'
                 % (tag, b['n'], b['rate'], b['n_q'], b['rate_q']))
    L.append('  候选阈值：')
    L.append('    %-4s %-6s %-9s %-9s %-9s %-8s %-8s %-8s %s'
             % ('类型', '取值', '阈值µε', '加载占比', 'load_h', 'f_Hz',
                '未载率', '加载率', '比值'))
    for t in rec['thrs']:
        L.append('    %-4s %-6s %-9.1f %8.1f%% %9.2f %8.3f %8.2f %8.2f %.3f'
                 % (t['kind'], ('%g' % t['key']), t['thr'], 100 * t['frac_load'],
                    t['load_h'], t['f_hz'], t['rate_lo'], t['rate_hi'],
                    t['ratio']))
    if rec['low_pos']:
        L.append('  低 p2p 突发在寿命十等分里的分布（左=早）：%s' % rec['low_pos'])
    return '\n'.join(L)


CH_NAMES = ['R1', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5']


def channels_report(gid, root=None):
    """逐窗逐通道 p2p —— 判断低 p2p 窗是「真停机」还是「坏通道拉低中位数」。

    这是本工具最决定性的一步：
      · 若低 p2p 窗里**所有** 10 个通道都低（几 µε） -> 停顿是真的，
        200 µε 这个阈值没错，问题在「停机时 AE 竟然还在记事件」；
      · 若低 p2p 窗里**部分**通道仍然很高（> 800 µε） -> 停顿是假的，
        是中位数被失效通道拉低，应该换成「至少 k 个通道同时低」的判据。
    """
    root = root or ac._data_l1()
    rows = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        rows.extend(ac.scan_fbg(f, channels=True))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    t = np.asarray([r[0] for r in rows], dtype=float)
    full = np.asarray([len(r[1]) == 10 for r in rows])
    V = np.full((len(rows), 10), np.nan)
    for i, r in enumerate(rows):
        if len(r[1]) == 10:
            V[i, :] = r[1]
    med = np.asarray([float(np.median(r[1])) for r in rows])
    lo = med <= ac.P2P_LOAD
    hi = ~lo
    row = {'gid': gid, 'n': len(rows), 'n_full': int(full.sum()),
           'n_lo': int(lo.sum()), 'n_hi': int(hi.sum())}
    good = full
    if lo.sum() and good[lo].sum():
        Vlo = V[lo & good]
        row['lo_ch_med'] = [float(np.median(Vlo[:, c])) for c in range(10)]
        row['lo_ch_le200'] = [float(np.mean(Vlo[:, c] <= ac.P2P_LOAD)) for c in range(10)]
        row['lo_ch_ge800'] = [float(np.mean(Vlo[:, c] >= 800.0)) for c in range(10)]
        row['lo_any_ge800'] = float(np.mean(np.max(Vlo, axis=1) >= 800.0))
        row['lo_all_le200'] = float(np.mean(np.max(Vlo, axis=1) <= ac.P2P_LOAD))
    if hi.sum() and good[hi].sum():
        Vhi = V[hi & good]
        row['hi_ch_med'] = [float(np.median(Vhi[:, c])) for c in range(10)]
        row['hi_ch_le200'] = [float(np.mean(Vhi[:, c] <= ac.P2P_LOAD)) for c in range(10)]
    return row


def fmt_channels(r):
    if r is None:
        return '（无数据）'
    L = ['=' * 78,
         '%s  窗 %d（10 通道齐全 %d）；低组(中位<=%g µε) %d，高组 %d'
         % (r['gid'], r['n'], r['n_full'], ac.P2P_LOAD, r['n_lo'], r['n_hi'])]
    if 'lo_ch_med' not in r:
        L.append('  低组为空（该组所有窗都 > 阈值）')
        return '\n'.join(L)
    L.append('  低组逐通道 p2p 中位：  ' + '  '.join(
        '%s=%g' % (CH_NAMES[c], round(r['lo_ch_med'][c])) for c in range(10)))
    L.append('  低组逐通道 <=200 占比：' + '  '.join(
        '%s=%.0f%%' % (CH_NAMES[c], 100 * r['lo_ch_le200'][c]) for c in range(10)))
    L.append('  低组逐通道 >=800 占比：' + '  '.join(
        '%s=%.1f%%' % (CH_NAMES[c], 100 * r['lo_ch_ge800'][c]) for c in range(10)))
    L.append('  ★ 低组「所有通道都 <=200」的窗占比 = %.1f%%   '
             '「至少一道 >=800」的窗占比 = %.1f%%'
             % (100 * r['lo_all_le200'], 100 * r['lo_any_ge800']))
    if 'hi_ch_med' in r:
        L.append('  高组逐通道 p2p 中位：  ' + '  '.join(
            '%s=%g' % (CH_NAMES[c], round(r['hi_ch_med'][c])) for c in range(10)))
    return '\n'.join(L)


def profile_report(gid, nb=20):
    """按寿命等分对照：AE 事件率 vs FBG 判停机占比 vs 最近突发 p2p。

    用途：若 AE 事件率在所有等分里都差不多，而 FBG 在某些等分里大量判停机，
    则说明「停机时 AE 仍在记事件」——问题出在 AE 侧（噪声底 / 时钟），
    而不是 FBG 阈值。
    """
    fr = ac._load_frames(gid, ac.FRAME_S)
    if fr is None:
        return None
    frame, t_sec, n_hits = fr
    pts = ac.group_load(gid)
    if not pts or n_hits is None:
        return None
    tf = np.asarray([p[0] for p in pts], dtype=float)
    pf = np.asarray([p[1] for p in pts], dtype=float)
    o = np.argsort(tf)
    tf, pf = tf[o], pf[o]
    tb, pb = bursts_with_p2p(tf, pf)
    near, _ = nearest_burst(t_sec, tb)
    p_ae = pb[near]
    nh = np.asarray(n_hits, dtype=float)
    edges = np.linspace(t_sec[0], t_sec[-1], nb + 1)
    rows = []
    for i in range(nb):
        m = (t_sec >= edges[i]) & (t_sec < edges[i + 1])
        if m.sum() == 0:
            continue
        lo = p_ae[m] <= ac.P2P_LOAD
        rows.append({'i': i, 'n': int(m.sum()),
                     'rate': float(np.median(nh[m])),
                     'frac_pause': float(np.mean(lo)),
                     'p2p_med': float(np.median(p_ae[m])),
                     'nburst': int(((tb >= edges[i]) & (tb < edges[i + 1])).sum())})
    return {'gid': gid, 'span_h': float(t_sec[-1] - t_sec[0]) / 3600.0,
            'rows': rows}


def fmt_profile(r):
    if r is None:
        return '（无数据）'
    L = ['=' * 78, '%s  寿命 %.1f h（20 等分）' % (r['gid'], r['span_h']),
         '  等分   月殁数  FBG突发数  AE率中位   判停机帧占比  最近突发p2p中位']
    for x in r['rows']:
        L.append('  %-5d %6d %10d %11.0f %13.1f%% %15.0f'
                 % (x['i'], x['n'], x['nburst'], x['rate'],
                    100 * x['frac_pause'], x['p2p_med']))
    return '\n'.join(L)


def align_report(gid, max_h=8.0, step_min=10.0):
    """AE 帧轴 vs FBG 突发的时间对齐 + 帧轴稀疏度。

    关键假设：AE 帧表**只在有事件时才建帧**（帧轴稀疏）。
    若如此，则帧自然集中在**加载期**，所以把 AE 平移 δ 后统计
    「最近突发为停机」的帧占比，**正确的 δ 应当使该占比最小**。
    若最小占比仍很高，说明两者不是同一个时间基准（或 AE 在停机时也在记事件）。
    """
    fr = ac._load_frames(gid, ac.FRAME_S)
    if fr is None:
        return None
    frame, t_sec, n_hits = fr
    pts = ac.group_load(gid)
    if not pts:
        return None
    tf = np.asarray([p[0] for p in pts], dtype=float)
    pf = np.asarray([p[1] for p in pts], dtype=float)
    o = np.argsort(tf)
    tf, pf = tf[o], pf[o]
    tb, pb = bursts_with_p2p(tf, pf)

    span = float(t_sec[-1] - t_sec[0])
    dense = int(frame[-1] - frame[0] + 1)
    gaps = np.diff(t_sec)
    rows = []
    for d in np.arange(-max_h * 3600.0, max_h * 3600.0 + 1e-9, step_min * 60.0):
        near, _ = nearest_burst(t_sec + d, tb)
        p = pb[near]
        rows.append((d / 3600.0, float(np.mean(p <= ac.P2P_LOAD)),
                     float(np.median(p))))
    return {'gid': gid, 'n_frame': len(t_sec), 'frame_span_n': dense,
            'span_h': span / 3600.0, 'n_burst': len(tb),
            'gap_med_h': float(np.median(gaps)) / 3600.0,
            'gap_max_h': float(np.max(gaps)) / 3600.0,
            'gap_ge12h': int((gaps > 12 * 3600).sum()),
            'rows': rows}


def fmt_align(r):
    if r is None:
        return '（无数据）'
    best = min(r['rows'], key=lambda x: x[1])
    L = ['=' * 78, '%s' % r['gid'],
         '  帧 %d；帧号跨度 %d（差 %d = 缺失帧）'
         % (r['n_frame'], r['frame_span_n'], r['frame_span_n'] - r['n_frame']),
         '  帧时刻跨度 %.1f h，FBG 突发 %d 个' % (r['span_h'], r['n_burst']),
         '  相邻帧间隔：中位 %.2f h，最大 %.2f h，>12 h 的缺口 %d 处'
         % (r['gap_med_h'], r['gap_max_h'], r['gap_ge12h']),
         '  偏移(h)   判停机帧占比   [最近突发 p2p 中位]']
    for d, f, m in r['rows']:
        mark = '  <<< 最优' if (d, f, m) == best else ''
        L.append('  %+7.2f %13.1f%% %18.0f%s' % (d, 100 * f, m, mark))
    return '\n'.join(L)


AMP_KEYS = ['n', 'amp_mean', 'amp_p50', 'amp_p90', 'amp_max', 'frac_ge80',
            'frac_ge90', 'b_value', 'ener_per_hit', 'sig_per_hit',
            'dur_p50', 'af_p50', 'ra_p50']


def amp_report(gid):
    """停机帧 vs 加载帧的 AE 幅值特征对照。

    若停机帧的事件是**背景噪声**，其幅值应该明显更低（`amp_p90` 低、
    `frac_ge90` 小、`b_value` 高）；若两者幅值分布相近，则那些事件是真事件，
    那就不能再用「AE 静默」当停机证据，得改用幅值加权的判据。
    """
    p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    fr = ac._load_frames(gid, ac.FRAME_S)
    pts = ac.group_load(gid)
    if fr is None or not pts:
        return None
    _, t_sec, _ = fr
    tf = np.asarray([q[0] for q in pts], dtype=float)
    pf = np.asarray([q[1] for q in pts], dtype=float)
    o = np.argsort(tf)
    tf, pf = tf[o], pf[o]
    tb, pb = bursts_with_p2p(tf, pf)
    near, dist = nearest_burst(t_sec, tb)
    p_ae = pb[near]
    lo = (p_ae <= ac.P2P_LOAD) & (dist <= ac.MAX_CARRY)
    hi = (p_ae > ac.P2P_LOAD) & (dist <= ac.MAX_CARRY)
    out = {'gid': gid, 'n_lo': int(lo.sum()), 'n_hi': int(hi.sum()), 'lo': {}, 'hi': {}}
    for k in AMP_KEYS:
        if k not in z.files:
            continue
        a = np.asarray(z[k], dtype=float)
        if len(a) != len(lo):
            continue
        out['lo'][k] = (float(np.median(a[lo])) if lo.sum() else float('nan'), lo.sum())
        out['hi'][k] = (float(np.median(a[hi])) if hi.sum() else float('nan'), hi.sum())
    return out


def fmt_amp(r):
    if r is None:
        return '（无数据）'
    L = ['=' * 78, '%s  停机帧 %d / 加载帧 %d（限 dist<=MAX_CARRY）'
         % (r['gid'], r['n_lo'], r['n_hi']), '  特征                停机帧        加载帧       比值']
    for k in AMP_KEYS:
        if k not in r['lo']:
            continue
        a, b = r['lo'][k][0], r['hi'][k][0]
        ratio = (a / b) if (b and np.isfinite(b) and b != 0) else float('nan')
        L.append('  %-16s %12.3f %12.3f %10.3f' % (k, a, b, ratio))
    return '\n'.join(L)


OUT_AMP_TXT = os.path.join(LOGDIR, 'l1_p2p_amp.txt')
OUT_KIND_TXT = os.path.join(LOGDIR, 'l1_p2p_kind.txt')
OUT_LV_TXT = os.path.join(LOGDIR, 'l1_p2p_level.txt')


def level_report(gid, root=None):
    """区分「真停机」与「保载」：逐窗看**振荡幅度**与**应变绝对水平**。

    `P2P_LOAD` 只看幅度，而停机与保载（恒定载荷）都给出 p2p 接近 0，
    因此必须补看水平：
      · 低 p2p 且**水平趋近 0** -> 真停机（无载荷，应变释放）
      · 低 p2p 且**水平仍在载荷量级** -> 保载 / 恒定载荷（试件仍在受力）
    水平以该组**加载窗的中位水平**为参考尺度（未做绝对标定，只看相对）。
    """
    root = root or ac._data_l1()
    rows = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        rows.extend(ac.scan_fbg_ex(f))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    p2 = np.asarray([float(np.median(r[1])) for r in rows])
    lv = np.asarray([float(np.median(r[2])) for r in rows])
    lo = p2 <= ac.P2P_LOAD
    hi = ~lo
    ref_lv = float(np.median(lv[hi])) if hi.sum() else float('nan')
    out = {'gid': gid, 'n': len(rows), 'n_lo': int(lo.sum()), 'n_hi': int(hi.sum()),
           'ref_lv': ref_lv,
           'spread_lv': float(np.quantile(lv, 0.95) - np.quantile(lv, 0.05)),
           'p2p_hi_med': float(np.median(p2[hi])) if hi.sum() else float('nan'),
           'span_h': (rows[-1][0] - rows[0][0]) / 3600.0,
           'n_f': ac.N_F.get(gid),
           'lv_lo': None, 'lv_hi': None}
    if lo.sum() >= 5 and hi.sum() >= 5:
        out['lv_lo'] = [float(np.quantile(lv[lo], q)) for q in (0.1, 0.5, 0.9)]
        out['lv_hi'] = [float(np.quantile(lv[hi], q)) for q in (0.1, 0.5, 0.9)]
        scale = max(out['spread_lv'], 1e-9)
        drop = np.abs(lv[lo] - ref_lv)
        out['frac_hold'] = float(np.mean(drop < 0.2 * scale))
        out['frac_idle'] = float(np.mean(drop > 0.5 * scale))
        # 对照：只用 p2p 判据 vs 加上水平判据后的加载占比与反解 f
        idx_lo = np.where(lo)[0]
        new_unload = np.zeros_like(lo)
        new_unload[idx_lo[drop > 0.5 * scale]] = True    # 幅度低且水平已离开 => 真不加载
        # 加载窗水平在寿命各等分里的中位（用来看基线/热漂移的量级）
        t0, t1 = rows[0][0], rows[-1][0]
        edges = np.linspace(t0, t1, 11)
        dec = []
        for i in range(10):
            m = hi & (np.asarray([r[0] for r in rows]) >= edges[i]) \
                & (np.asarray([r[0] for r in rows]) < edges[i + 1])
            dec.append(float(np.median(lv[m])) if m.sum() else float('nan'))
        out['lv_hi_dec'] = dec
        for tag, unl in (('仅 p2p（现口径）', lo), ('p2p + 水平（新口径）', new_unload)):
            fl = 1.0 - float(np.mean(unl))
            lh = fl * out['span_h']
            f = (out['n_f'] / (lh * 3600.0)) if (out['n_f'] and lh > 0) else float('nan')
            out.setdefault('cmp', []).append((tag, fl, lh, f))
    return out


def fmt_level(r):
    if r is None:
        return '（无数据）'
    L = ['=' * 78, '%s  窗 %d（低 p2p %d / 高 p2p %d）'
         % (r['gid'], r['n'], r['n_lo'], r['n_hi']),
         '  加载窗水平中位 = %.1f（全组水平 5%% 至 95%% 跨度 = %.1f），加载窗 p2p 中位 = %.1f'
         % (r['ref_lv'], r['spread_lv'], r['p2p_hi_med'])]
    if r['lv_lo'] is None:
        L.append('  低 p2p 窗不足，跳过')
        return '\n'.join(L)
    L.append('  低 p2p 窗的水平 p10/p50/p90 = %.1f / %.1f / %.1f' % tuple(r['lv_lo']))
    L.append('  高 p2p 窗的水平 p10/p50/p90 = %.1f / %.1f / %.1f' % tuple(r['lv_hi']))
    L.append('  ★ 低 p2p 窗里「水平仍接近加载态」= %.1f%%；「水平已明显离开加载态」= %.1f%%'
             % (100 * r['frac_hold'], 100 * r['frac_idle']))
    L.append('     判读：「接近加载态」高 => **保载**（试件仍受力）；'
             '「离开加载态」高 => **真停机**')
    if r.get('lv_hi_dec'):
        L.append('  加载窗水平在寿命十等分里的中位（看漂移量级）：')
        L.append('    ' + ' / '.join('%.0f' % v for v in r['lv_hi_dec']))
    if r.get('cmp'):
        L.append('  对照（帧加权，不含 AE 静默守卫，仅供参考方向）：')
        L.append('      口径                加载占比   load_h(h)  反解 f(Hz)')
        for tag, fl, lh, f in r['cmp']:
            L.append('      %-20s %6.1f%% %9.2f %10.3f' % (tag, 100 * fl, lh, f))
    return '\n'.join(L)

KIND_FEATS = ['n', 'amp_p50', 'amp_p90', 'amp_max', 'a95_a50', 'frac_ge80',
              'frac_ge90', 'b_value', 'amp_entropy', 'ener_per_hit',
              'sig_per_hit', 'dur_p50', 'af_p50', 'ra_p50']


def _auc(a, b):
    """P(a > b) + 0.5 P(a == b)（Mann-Whitney 的 AUC 形式）。

    ≈ 0.5 表示两组无法区分（同一总体）；越远离 0.5 越可分。
    """
    a = np.sort(np.asarray(a, dtype=float))
    a = a[np.isfinite(a)]
    b = np.sort(np.asarray(b, dtype=float))
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    gt = eq = 0
    for v in a:
        lo = int(np.searchsorted(b, v, side='left'))
        hi = int(np.searchsorted(b, v, side='right'))
        gt += len(b) - hi
        eq += hi - lo
    return (gt + 0.5 * eq) / (len(a) * len(b))


def kind_report(gid):
    """停机帧 vs 加载帧的「事件性质」对照（三分量 + 分布 + 局部性）。

    三层证据：
      (a) 逐特征 AUC：停机帧与加载帧是**同一总体**还是两个总体；
      (b) RA-AF 象限：AE 源机制判别（低 RA + 高 AF = 张拉型裂纹；
          高 RA + 低 AF = 剪切/摩擦；高 RA + 高 AF = 噪声常落在此）；
      (c) **局部性**：直接看「FBG 快照所在的那一帧」的事件数 ——
          不经过最近突发传播，是这批事件到底真不真的最直接证据。
    """
    p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    frame_s = float(z['frame_s']) if 'frame_s' in z.files else ac.FRAME_S
    frame = np.asarray(z['frame'], dtype=np.int64)
    t_sec = frame.astype(float) * frame_s
    pts = ac.group_load(gid)
    if not pts:
        return None
    tf = np.asarray([q[0] for q in pts], dtype=float)
    pf = np.asarray([q[1] for q in pts], dtype=float)
    o = np.argsort(tf)
    tf, pf = tf[o], pf[o]
    tb, pb = bursts_with_p2p(tf, pf)
    near, dist = nearest_burst(t_sec, tb)
    p_ae = pb[near]
    lo = (p_ae <= ac.P2P_LOAD) & (dist <= ac.MAX_CARRY)
    hi = (p_ae > ac.P2P_LOAD) & (dist <= ac.MAX_CARRY)
    out = {'gid': gid, 'n_lo': int(lo.sum()), 'n_hi': int(hi.sum()),
           'feats': [], 'quad': None, 'local': None}
    if out['n_lo'] < 5 or out['n_hi'] < 5:
        return out

    # (a) 逐特征 AUC 与分位
    for k in KIND_FEATS:
        if k not in z.files:
            continue
        a = np.asarray(z[k], dtype=float)
        if len(a) != len(lo):
            continue
        al, ah = _auc(a[lo], a[hi]), None
        ah = _auc(a[hi], a[lo])
        out['feats'].append({
            'k': k, 'auc': al,
            'lo': [float(np.quantile(a[lo][np.isfinite(a[lo])], q)) if np.isfinite(a[lo]).sum() else float('nan')
                   for q in (0.1, 0.5, 0.9)],
            'hi': [float(np.quantile(a[hi][np.isfinite(a[hi])], q)) if np.isfinite(a[hi]).sum() else float('nan')
                   for q in (0.1, 0.5, 0.9)],
            'auc_rev': ah})

    # (b) RA-AF 象限
    if 'ra_p50' in z.files and 'af_p50' in z.files:
        ra = np.asarray(z['ra_p50'], dtype=float)
        af = np.asarray(z['af_p50'], dtype=float)
        good = np.isfinite(ra) & np.isfinite(af)
        m = good
        ra_m = float(np.median(ra[m]))
        af_m = float(np.median(af[m]))
        q = {}
        for tag, sel in (('lo', lo & good), ('hi', hi & good)):
            if sel.sum() == 0:
                q[tag] = None
                continue
            q[tag] = {
                'n': int(sel.sum()), 'ra_m': ra_m, 'af_m': af_m,
                '低RA高AF': float(np.mean((ra[sel] < ra_m) & (af[sel] >= af_m))),
                '高RA高AF': float(np.mean((ra[sel] >= ra_m) & (af[sel] >= af_m))),
                '低RA低AF': float(np.mean((ra[sel] < ra_m) & (af[sel] < af_m))),
                '高RA低AF': float(np.mean((ra[sel] >= ra_m) & (af[sel] < af_m))),
            }
        out['quad'] = q

    # (c) 局部性：FBG 快照所在那一帧的事件数
    if 'n' in z.files:
        n_in = np.asarray(z['n'], dtype=float)
        pos = {int(v): i for i, v in enumerate(frame)}
        idx = np.array([pos.get(int(f), -1) for f in (tb / frame_s).astype(np.int64)])
        ok = idx >= 0
        burst_pause = pb <= ac.P2P_LOAD
        nb = n_in[idx[ok]]
        bp = burst_pause[ok]
        med_all = float(np.median(n_in))
        out['local'] = {
            'n_pause': int(bp.sum()), 'n_load': int((~bp).sum()),
            'med_pause': float(np.median(nb[bp])) if bp.sum() else float('nan'),
            'med_load': float(np.median(nb[~bp])) if (~bp).sum() else float('nan'),
            'silent_pause': float(np.mean(nb[bp] <= 1.0)) if bp.sum() else float('nan'),
            'silent_load': float(np.mean(nb[~bp] <= 1.0)) if (~bp).sum() else float('nan'),
            'med_all_frame': med_all,
        }
        # (d) 列联表：「静默」与「停机标签」到底重不重合
        thr = max(5.0, 0.01 * med_all)
        sil = n_in <= thr
        out['ct'] = {
            'thr': thr, 'n_sil': int(sil.sum()), 'n_all': int(len(sil)),
            'sil_pause': int((sil & lo).sum()), 'sil_load': int((sil & hi).sum()),
            'p_pause_given_sil': (float(np.mean(lo[sil])) if sil.sum() else float('nan')),
            'p_pause_given_loud': (float(np.mean(lo[~sil])) if (~sil).sum() else float('nan')),
            'p_sil_given_pause': (float(np.mean(sil[lo])) if lo.sum() else float('nan')),
            'p_sil_given_load': (float(np.mean(sil[hi])) if hi.sum() else float('nan')),
        }
    return out


def fmt_kind(r):
    if r is None:
        return '（无数据）'
    L = ['=' * 78, '%s' % r['gid']]
    if r['n_lo'] < 5 or r['n_hi'] < 5:
        L.append('  样本不足（停机帧 %d / 加载帧 %d），跳过' % (r['n_lo'], r['n_hi']))
        return '\n'.join(L)
    L.append('  停机帧 %d / 加载帧 %d' % (r['n_lo'], r['n_hi']))
    L.append('  (a) 逐特征可分性（AUC 离 0.5 越远越可分）')
    L.append('      特征             停机帧 p10/p50/p90        加载帧 p10/p50/p90        AUC')
    for f in r['feats']:
        L.append('      %-14s %8.2f %8.2f %8.2f   %8.2f %8.2f %8.2f   %6.3f'
                 % (f['k'], f['lo'][0], f['lo'][1], f['lo'][2],
                    f['hi'][0], f['hi'][1], f['hi'][2], f['auc']))
    if r['quad']:
        q = r['quad']
        L.append('  (b) RA-AF 象限占比（以合并中位切分：RA 中位 %g，AF 中位 %g）'
                 % (q['lo']['ra_m'], q['lo']['af_m']))
        for tag, name in (('低RA高AF（张拉型裂纹）', '低RA高AF'), ('高RA高AF', '高RA高AF'),
                          ('低RA低AF', '低RA低AF'), ('高RA低AF（剪切/摩擦）', '高RA低AF')):
            L.append('      %-22s 停机 %6.1f%%   加载 %6.1f%%'
                     % (tag, 100 * q['lo'][name], 100 * q['hi'][name]))
    if r['local']:
        c = r['local']
        L.append('  (c) 局部性：FBG 快照**所在那一帧**的事件数')
        L.append('      停机快照 %d 个 -> 事件数中位 %.0f（%.1f%% 的快照其所在帧 <= 1 个事件）'
                 % (c['n_pause'], c['med_pause'], 100 * c['silent_pause']))
        L.append('      加载快照 %d 个 -> 事件数中位 %.0f（%.1f%% 的快照其所在帧 <= 1 个事件）'
                 % (c['n_load'], c['med_load'], 100 * c['silent_load']))
        L.append('      全体帧事件数中位 %.0f' % c['med_all_frame'])
    if r.get('ct'):
        c = r['ct']
        L.append('  (d) 列联表：静默（事件数 <= %.0f）与停机标签的重合' % c['thr'])
        L.append('      静默帧 %d 个（占全部 %d 帧的 %.1f%%）：其中标签停机 %d、加载 %d'
                 % (c['n_sil'], c['n_all'], 100 * c['n_sil'] / max(c['n_all'], 1),
                    c['sil_pause'], c['sil_load']))
        L.append('      P(标签=停机 | 静默) = %.1f%%     P(标签=停机 | 非静默) = %.1f%%'
                 % (100 * c['p_pause_given_sil'], 100 * c['p_pause_given_loud']))
        L.append('      P(静默 | 标签=停机) = %.1f%%      P(静默 | 标签=加载) = %.1f%%'
                 % (100 * c['p_sil_given_pause'], 100 * c['p_sil_given_load']))
    return '\n'.join(L)


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    gids = []
    only_ch = False
    only_pf = False
    only_al = False
    only_amp = False
    only_kind = False
    only_lv = False
    for i, a in enumerate(argv):
        if a == '--groups':
            gids = argv[i + 1].split(',')
        elif a == '--channels':
            only_ch = True
        elif a == '--profile':
            only_pf = True
        elif a == '--align':
            only_al = True
        elif a == '--amp':
            only_amp = True
        elif a == '--kind':
            only_kind = True
        elif a == '--level':
            only_lv = True
    if not gids:
        gids = ac.groups_with_frames()

    if only_lv:
        blocks = [fmt_level(level_report(g)) for g in gids]
        with open(OUT_LV_TXT, 'w', encoding='utf-8') as fh:
            fh.write('停机 vs 保载：逐窗幅度与水平对照\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_LV_TXT)
        print('\n'.join(blocks))
        return

    if only_kind:
        blocks = [fmt_kind(kind_report(g)) for g in gids]
        with open(OUT_KIND_TXT, 'w', encoding='utf-8') as fh:
            fh.write('停机帧 vs 加载帧的事件性质定性\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_KIND_TXT)
        print('\n'.join(blocks))
        return

    if only_amp:
        blocks = [fmt_amp(amp_report(g)) for g in gids]
        with open(OUT_AMP_TXT, 'w', encoding='utf-8') as fh:
            fh.write('停机帧 vs 加载帧的 AE 幅值特征\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_AMP_TXT)
        print('\n'.join(blocks))
        return

    if only_al:
        blocks = [fmt_align(align_report(g)) for g in gids]
        with open(OUT_AL_TXT, 'w', encoding='utf-8') as fh:
            fh.write('AE / FBG 时间对齐与帧轴稀疏度\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_AL_TXT)
        print('\n'.join(blocks))
        return

    if only_pf:
        blocks = [fmt_profile(profile_report(g)) for g in gids]
        with open(OUT_PF_TXT, 'w', encoding='utf-8') as fh:
            fh.write('AE 与 FBG 的时间剖面\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_PF_TXT)
        print('\n'.join(blocks))
        return

    if only_ch:
        blocks = []
        for gid in gids:
            r = channels_report(gid)
            blocks.append(fmt_channels(r) if r else '%s  跳过：无 FBG' % gid)
        with open(OUT_CH_TXT, 'w', encoding='utf-8') as fh:
            fh.write('逐通道 p2p 诊断\n' + '\n'.join(blocks) + '\n')
        print('已写 %s' % OUT_CH_TXT)
        print('\n'.join(blocks))
        return

    rows, blocks = [], []
    for gid in gids:
        rec, why = analyse(gid)
        if rec is None:
            blocks.append('%s  %s' % (gid, why))
            continue
        blocks.append(fmt(rec))
        for b in rec['bins']:
            rows.append([rec['gid'], 'bin', '%.0f' % b['lo'],
                         '%.1f' % b['rate'] if np.isfinite(b['rate']) else '',
                         '%.1f' % b['rate_q'] if np.isfinite(b['rate_q']) else '',
                         b['n'], b['n_q'], rec['span_h']])
        for t in rec['thrs']:
            rows.append([rec['gid'], t['kind'], '%g' % t['key'],
                         '%.2f' % (100 * t['frac_load']), '%.3f' % t['ratio'],
                         '', '', '%.3f' % t['f_hz']])

    head = 'gid,kind,key,frac_load_pct,ratio,n,n_q,col8'
    with open(OUT_CSV, 'w', encoding='utf-8') as fh:
        fh.write(head + '\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')
    with open(OUT_TXT, 'w', encoding='utf-8') as fh:
        fh.write('P2P_LOAD 体检报告（%s）\n' % ac.P2P_LOAD)
        fh.write('\n'.join(blocks) + '\n')
    print('已写 %s\n     %s' % (OUT_CSV, OUT_TXT))
    print('\n'.join(blocks))


if __name__ == '__main__':
    main(sys.argv[1:])
