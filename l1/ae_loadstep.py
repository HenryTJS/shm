# -*- coding: utf-8 -*-
"""重启响应（Restart Response）—— 把 Kaiser / Felicity 效应量化成**跨试件可比**的指标。

为什么是它
----------
这批试验每天有大量「停机 → 重启」（夜歇／换级），而循环轴里已经存了每帧的加载占比
`g_load`（FBG 判出来的），所以停机段与重启点可以**纯数据驱动**地找出来，不需要载荷信号。

物理含义（教科书级）：
  · 重启后**活动几乎不变** ⇒ 纯 **Kaiser 效应**（载荷未超过历史最大值，无新损伤）
  · 重启后**活动明显跳升** ⇒ **Felicity 效应**（提载／新损伤）⇒ 跃升比越大，新增损伤越多

    跃升比 R = 重启后活动水平 / 停机前活动水平

R 是**比值**，天然抵消了跨试件的绝对活动度差异（那正是老组失败的根因），
且只需"当前与刚过去的一小段"⇒ **在线可用**。

用法
----
    python l1/ae_loadstep.py                      # 全部组
    python l1/ae_loadstep.py L1-29 L1-25
输出：results/l1_restart_{gid}.npz（逐次重启事件）+ 终端汇总
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
if HERE not in sys.path:
    sys.path.insert(0, HERE)

STOP_G = 0.30          # 加载占比低于此值 → 判为停机帧
RUN_G = 0.70           # 高于此值 → 判为加载帧
MIN_STOP = 2           # 停机段至少这么多 FBG 窗（2×420 s = 14 min）
K_FRAME = 3            # 重启后 / 停机前各取几帧做水平估计（3×600 s = 30 min）


def fbg_stop_spans(gid):
    """从 FBG **原始采样序列**找停机段 → [(t0, t1, n_win), ...]。

    之前用 AE 帧网格上的加载占比判停机行不通：AE 帧长 600 s，而 FBG 采样间隔
    420 s，一个窗口里平均只有 1 至 2 个 FBG 帧 ⇒ 占比被量化成 0 / 0.5 / 1。
    直接在 FBG 序列上找连续非加载窗才准。

    ⚠️ **2026-10-05 补充：上面这段理由只对旧的 `fill='mean'` 口径成立。**
    循环轴现在用 `holdgap` + AE 静默守卫（`ae_cycle._ratio_on_grid`），
    真停机段会在帧网格上给出 `g=0` 的**连续段**，所以 `find_restarts` 反而是对的 ——
    但它**从未被调用**（死代码，保留作参照）。
    两法实测结论一致：批次 A（L1-06/13/14/24）两法都是 0 至 1 段，
    因为那 4 组**确实没有停机段**（AE 事件率自身也无低谷，见 `--diag`、`details.md` §26）。
    另一条要注意：批次 A 的 FBG **覆盖率只有 13 % 至 17 %**，
    且缺失部分**没有 >1 h 的缺口**（都是大量中等间隙）⇒ 本函数对它们**天然盲**。
    """
    try:
        from ae_cycle import group_load, P2P_LOAD
    except Exception:                                        # noqa: BLE001
        return []
    pts = group_load(gid)
    if len(pts) < 3:
        return []
    t = np.asarray([p[0] for p in pts], dtype=float)
    isload = np.asarray([p[1] > P2P_LOAD for p in pts])
    spans = []
    i = 0
    while i < len(t):
        if not isload[i]:
            j = i
            while j < len(t) and not isload[j]:
                j += 1
            if (j - i) >= MIN_STOP and j < len(t):
                spans.append((t[i], t[j - 1], j - i))
            i = j
        else:
            i += 1
    return spans


def find_restarts(g_load, min_stop=MIN_STOP):
    """返回 [(stop_i0, stop_i1, restart_i), ...]：停机段起止 + 重启后首帧。"""
    out = []
    n = len(g_load)
    i = 0
    while i < n:
        if g_load[i] < STOP_G:
            j = i
            while j < n and g_load[j] < STOP_G:
                j += 1
            if (j - i) >= min_stop and j < n:      # 停机段后确实重启了
                out.append((i, j - 1, j))
            i = j
        else:
            i += 1
    return out


def _level(n_seq, lo, hi):
    """[lo, hi) 区间内活动水平的稳健估计（中位），至少 1 帧。"""
    lo = max(lo, 0)
    hi = min(hi, len(n_seq))
    if hi - lo < 1:
        return np.nan
    return float(np.median(n_seq[lo:hi]))


def run(gid):
    fp = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    z = np.load(fp, allow_pickle=True)
    c = np.load(cp, allow_pickle=True)
    idx = {int(f): i for i, f in enumerate(c['frame'])}
    sel = np.asarray([idx.get(int(f), -1) for f in z['frame']])
    ok = sel >= 0
    if ok.sum() < 10:
        return None
    t_ec = np.full(len(z['n']), np.nan)
    t_ec[ok] = c['t_epoch'][sel[ok]]
    cyc = np.full(len(z['n']), np.nan)
    cyc[ok] = c['cycle'][sel[ok]]
    top = np.nanmax(cyc) if np.isfinite(cyc).any() else 0.0
    n_seq = z['n'].astype(float)

    spans = fbg_stop_spans(gid)
    ev = []
    for (t0, t1, nw) in spans:
        # 停机段映射到 AE 帧：停机前最后一帧 / 重启后第一帧
        pre = np.where(t_ec < t0)[0]
        post = np.where(t_ec > t1)[0]
        if len(pre) == 0 or len(post) == 0:
            continue
        i_pre, r = pre[-1], post[0]
        before = _level(n_seq, i_pre - K_FRAME + 1, i_pre + 1)
        after = _level(n_seq, r, r + K_FRAME)
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        ratio = (after + 1.0) / (before + 1.0)          # 加 1 平滑，避免除零
        ev.append({'i_stop': i_pre, 'i_restart': r, 'stop_win': nw,
                   'before': before, 'after': after, 'ratio': ratio,
                   'life': (cyc[i_pre] / top) if top > 0 else np.nan})
    if not ev:
        return None
    d = {'life': np.asarray([e['life'] for e in ev]),
         'ratio': np.asarray([e['ratio'] for e in ev]),
         'before': np.asarray([e['before'] for e in ev]),
         'after': np.asarray([e['after'] for e in ev]),
         'stop_win': np.asarray([e['stop_win'] for e in ev]),
         'frame_stop': z['frame'][[e['i_stop'] for e in ev]]}
    np.savez_compressed(os.path.join(RES, '_l1restart_%s.npz' % gid), **d)
    return d


def ae_dip_runs(n_hits, radius=2, frac=0.30, min_run=2):
    """直接在 **AE 事件率**上找低谷段 —— 不依赖 FBG 覆盖，也不依赖 `g`。

    动机：批次 A 的 `g` 恒为 1（FBG 只看得到 13 % 至 17 % 的时间，
    而 AE 静默守卫从未触发）⇒ `g` 从未被任何证据否证过。
    若试验真的停过，AE 事件率应该出现**低谷**，而 `g` 看不出来。
    返回 (低谷段数, 低谷帧占比)。
    """
    n = np.asarray(n_hits, dtype=float)
    if n.size < 5:
        return 0, float('nan')
    # 滑动中位（滑窗 2*radius+1），对孤立低帧不敏感，避免把噪声当停机
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * radius + 1
    if n.size < k:
        return 0, float('nan')
    sm = np.median(sliding_window_view(np.pad(n, radius, mode='edge'), k), axis=1)
    low = sm < frac * float(np.median(n))
    runs, i = 0, 0
    while i < low.size:
        if low[i]:
            j = i
            while j < low.size and low[j]:
                j += 1
            if (j - i) >= min_run:
                runs += 1
            i = j
        else:
            i += 1
    return runs, float(low.mean())


def diag(gids):
    """为什么有些组「无重启事件」—— 先查 FBG 采样本身能不能看见停机（待办 1.2）。

    三种可能，必须先用数据分开，否则会把数据问题当成物理结论：
      (a) **真的没有停机段** —— 试验连续跑完（那么 `load_h` 应该接近 `span_h`）；
      (b) **FBG 只在加载时采集** —— 停机段变成**大时间缺口**，而
          `fbg_stop_spans` 只认「连续非加载窗」，**看不见缺口**；
      (c) `P2P_LOAD` 把停机窗误判成加载窗（§15 ① 已排除阈值问题）。

    逐组输出：窗数、窗长估计、FBG 记录跨度与覆盖率、缺口统计（>1 h 的个数与合计）、
    非加载窗数，以及现有 `fbg_stop_spans` 能找到几个停机段。
    """
    try:
        from ae_cycle import group_load, P2P_LOAD
    except Exception:                                        # noqa: BLE001
        print('无法导入 ae_cycle')
        return None
    print('%-7s %6s %7s %9s %9s %8s %9s %9s %7s %5s %7s %6s %8s %8s' % (
        '组', '窗数', '窗长s', 'FBG跨度h', '覆盖h', '覆盖率', '缺口>1h', '缺口合计h',
        '非载窗', 'FBG停机', 'g<0.3比', '网格停机', 'AE低谷段', '低谷帧比'))
    rows = []
    for gid in gids:
        pts = group_load(gid)
        if len(pts) < 3:
            print('%-7s  无 FBG' % gid)
            continue
        t = np.asarray([p[0] for p in pts], dtype=float)
        p2p = np.asarray([p[1] for p in pts], dtype=float)
        n = t.size
        dt = np.diff(t)
        small = dt[dt < 120.0]
        win_s = float(np.median(small)) if small.size else float('nan')
        span_h = (t[-1] - t[0]) / 3600.0
        cover_h = n * win_s / 3600.0
        gaps = dt[dt > 3600.0]
        nonload = int((p2p <= P2P_LOAD).sum())
        nspan = len(fbg_stop_spans(gid))
        # 另一个检测器：AE 帧网格上的 g < STOP_G 连续段
        # （`find_restarts`，它考虑了 holdgap 的零阶保持 + AE 静默守卫）
        cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
        fgrid, gzero = -1, float('nan')
        if os.path.exists(cp):
            c = np.load(cp, allow_pickle=True)
            gg = np.asarray(c['g_load'], dtype=float)
            fgrid = len(find_restarts(gg))
            gzero = float((gg < STOP_G).mean())
        # AE 事件率自身的低谷（不依赖 FBG 覆盖，也不依赖 g）
        nrun, nlow = 0, float('nan')
        fpa = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
        if os.path.exists(fpa):
            zz = np.load(fpa, allow_pickle=True)
            if 'n' in zz.files:
                nrun, nlow = ae_dip_runs(zz['n'].astype(float))
        rows.append((gid, n, win_s, span_h, cover_h, nonload, nspan,
                     int(gaps.size), float(gaps.sum()) / 3600.0, fgrid, gzero,
                     nrun, nlow))
        print('%-7s %6d %7.1f %9.1f %9.1f %7.0f%% %9d %9.1f %7d %5d %6.0f%% %6d %8d %7.0f%%'
              % (gid, n, win_s, span_h, cover_h, 100 * cover_h / max(span_h, 1e-9),
                 gaps.size, float(gaps.sum()) / 3600.0, nonload, nspan,
                 100 * gzero, fgrid, nrun, 100 * nlow))
    print('\n判读：若「缺口>1h」多且「缺口合计」与 (FBG跨度 − 覆盖) 同量级，')
    print('      说明停机段是被**采样停掉**的（情形 b），不是真的没有停机。')
    print('      「FBG停机」= fbg_stop_spans（连续非加载窗）；')
    print('      「网格停机」= find_restarts（AE 帧网格上 g<0.3 的连续段）；')
    print('      「AE低谷段」= 事件率自身低于中位 30%% 且 >=2 帧的连续段（不看 FBG、不看 g）。')
    return rows


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    allg = sorted(os.path.basename(p)[len('_l1ae_frames_'):-4]
                  for p in glob.glob(os.path.join(RES, '_l1ae_frames_*.npz')))
    gids = [a for a in argv if not a.startswith('--')] or allg
    if '--diag' in argv:
        return 0 if diag(gids) is not None else 1
    print('%-7s %6s %8s %9s %9s %9s %9s' % (
        '组', '重启次数', '停机窗中位', 'R中位', 'R_前25%', 'R_后25%', 'rho(R,寿命)'))
    rows = []
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        spearmanr = None
    for gid in gids:
        d = run(gid)
        if d is None:
            print('%-7s %6s' % (gid, '无重启事件或无循环轴'))
            continue
        r = d['ratio']
        life = d['life']
        med = float(np.median(r))
        q1 = float(np.median(r[:max(1, len(r) // 4)]))
        q4 = float(np.median(r[-max(1, len(r) // 4):]))
        rho = np.nan
        if spearmanr is not None and len(r) >= 5 and np.isfinite(life).sum() >= 5:
            m = np.isfinite(life)
            rho = float(spearmanr(life[m], r[m])[0])
        rows.append((gid, len(r), med, q1, q4, rho))
        print('%-7s %6d %8.0f %9.3f %9.3f %9.3f %9.3f'
              % (gid, len(r), np.median(d['stop_win']), med, q1, q4, rho))
    if rows:
        import csv
        out = os.path.join(RES, 'l1_restart_summary.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '重启次数', 'R中位', 'R_前25%', 'R_后25%', 'rho_R_vs_寿命'])
            w.writerows(rows)
        print('\n已写出 ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
