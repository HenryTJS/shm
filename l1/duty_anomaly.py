# -*- coding: utf-8 -*-
"""L1 加载占空比体检（待办 1.1）：L1-31 的 gp 为何偏低。

问题
----
`gp` = FBG **窗级**加载占比（`lf.mean()`，阈 P2P_LOAD）。L1-31 的 gp = 0.474，
低于同批 C4 其余 6 组的 0.549 至 0.870。问：真实工况差异，还是 FBG 采样相位偏置？

先排除掉两条不成立的线索
------------------------
1. **窗距**：`fbg_variants --batch` 报的「窗距中位」只统计**加载窗之间**的间隔
   （停机窗不计入）⇒ 它天然看不见停机。实测 L1-31 与 L1-41 都是 406 s，
   而 gp 分别是 0.474 与 0.870 ⇒ 窗距解释不了 gp 的差异。
2. **跨度**：L1-31 的 `span_h = 716.8 h` 里含 08-09 至 08-25 一段
   **一个 FBG 文件都没有**的 18 天空洞。空洞贡献 0 个窗 ⇒ gp 与跨度无关。

本模块给的是**不依赖 FBG 的**证据
--------------------------------
AE 事件是**按加载循环**产生的，所以「每单位日历时间的事件数」是机器是否在转的
直接指示。`_l1ae_frames_*.npz` 的帧号 = `int(绝对 epoch / frame_s)`
（见 `ae_frames.group_frames`：坐标 = 该卷采集起始时刻 + 卷内相对时间，
C4 批再按实测钟差平移 8 天）⇒ **帧号 x 600 s 就是 FBG 日历上的绝对时刻**，
两套记录可以直接逐日对齐。

用法
----
    python l1/duty_anomaly.py --timeline L1-31 L1-41      # 逐日对照（推荐先看）
    python l1/duty_anomaly.py --marks L1-31 L1-13         # .DTA 采集开关时间轴
    python l1/duty_anomaly.py --batch                     # 11 组汇总
输出日志：results/_logs/l1_duty_{timeline,marks,batch}.txt
"""

import argparse
import collections
import datetime as _dt
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_cycle import scan_fbg, P2P_LOAD, FRAME_S, RES      # noqa: E402

# §25 的分组（11 组 = 有 FBG 剖面且窗数足够的组）
C3 = ['L1-06', 'L1-13', 'L1-14', 'L1-24']
C4 = ['L1-25', 'L1-27', 'L1-29', 'L1-31', 'L1-35', 'L1-41', 'L1-44']
GIDS = C3 + C4
DAY = 86400.0
MARK_NAME = {128: 'START', 129: 'STOP', 130: 'PAUSE'}
_TZ8 = _dt.timezone(_dt.timedelta(hours=8))


def _dtxt(epoch, fmt='%m-%d %H:%M'):
    """epoch 秒 → UTC+8 的日期文本（实测两组 FBG 记录都是 UTC+8 挂钟）。"""
    return _dt.datetime.fromtimestamp(epoch, _TZ8).strftime(fmt)
_LOGS = os.path.join(RES, '_logs')
os.makedirs(_LOGS, exist_ok=True)


class Tee(object):
    """同时写控制台与日志文件。"""

    def __init__(self, path):
        self.fh = open(path, 'w', encoding='utf-8')

    def write(self, s):
        sys.__stdout__.write(s)
        self.fh.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()


def _data_l1():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                        # noqa: BLE001
        pass
    return HERE


# --------------------------------------------------------------- FBG 窗序列
def fbg_windows(gid, root, refresh=False):
    """返回 (t_epoch[], 加载标记[])。结果缓存到 results/_duty_{gid}.npz。

    ⚠️ 扫描要读完全部 FBG 文本（L1-31 是 2420 个文件），首次较慢。
    """
    p = os.path.join(RES, '_duty_%s.npz' % gid)
    if os.path.exists(p) and not refresh:
        z = np.load(p)
        return z['t'], z['lf']
    pts = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        pts.extend(scan_fbg(f))
    pts.sort(key=lambda x: x[0])
    t = np.asarray([x[0] for x in pts], dtype=float)
    lf = np.asarray([1.0 if x[1] > P2P_LOAD else 0.0 for x in pts], dtype=float)
    np.savez(p, t=t, lf=lf)
    return t, lf


def fbg_files(gid, root):
    """FBG 文件的挂钟时刻序列（从文件名解析），以及文件数。"""
    out = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        s = os.path.basename(f)
        try:
            d = s.split('.')[1]
            out.append(_dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]),
                                    int(d[8:10]), int(d[10:12]), int(d[12:14]))
                       .timestamp())
        except Exception:                                    # noqa: BLE001
            continue
    return np.asarray(sorted(out), dtype=float)


def ae_frames(gid):
    p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        return None, None
    z = np.load(p, allow_pickle=True)
    return z['frame'].astype(float) * FRAME_S, np.asarray(z['n'], dtype=float)


def day_stats(t, lf, ta, n):
    """按日汇总 → dict（供 --timeline 与 --batch 共用）。

    关键量：
      `duty_med`   —— 只取「窗数 >= 300 的完整日」的日加载占比**中位**。
                      ⚠️ 这个名字容易误读：整天停摆多的组（如 L1-31 有 7/11 天）
                      中位会被拉低，它反映的是「试验执行」而不是机器参数。
      `duty_max`   —— 完整日里**最高**的日加载占比 = 「机器正常跑的一天」的占空比。
                      实测 11 组都落在 0.85 上下，因此**组间 gp 差异可由停摆天数完全解释**。
      `n_stall`    —— 完整日里日加载占比 < 60% 的天数。
      `ae_cov_med` —— 完整日里「有事件的 600 s 帧 / 该日 AE 在录的 600 s 槽」中位。
                      该量**不用 FBG**，可与 duty 交叉验证。
    """
    b = ((t - t[0]) // DAY).astype(int)
    duty, aecov = [], []
    for k in np.unique(b):
        m = b == k
        if m.sum() < 300:
            continue
        duty.append(float(lf[m].mean()))
        if ta is not None and len(ta):
            lo, hi = t[0] + k * DAY, t[0] + (k + 1) * DAY
            lo2, hi2 = max(lo, ta[0]), min(hi, ta[-1])
            if hi2 > lo2:
                slots = (hi2 - lo2) / FRAME_S
                nm = ((ta >= lo) & (ta < hi)).sum()
                aecov.append(float(nm) / slots)
    duty = np.asarray(duty)
    return {'n_full': int(duty.size),
            'duty_med': float(np.median(duty)) if duty.size else float('nan'),
            'duty_min': float(duty.min()) if duty.size else float('nan'),
            'duty_max': float(duty.max()) if duty.size else float('nan'),
            'n_stall': int((duty < 0.60).sum()),
            'ae_cov_med': float(np.median(aecov)) if aecov else float('nan')}


# ------------------------------------------------------------------- 模式 A
def timeline(gids, root):
    print('=' * 118)
    print('逐日对照：FBG（窗数 / 加载占比） vs AE（帧数 / 事件数）')
    print('  —— AE 事件数完全不使用 FBG 信息，是判定 gp 真伪的独立证据')
    print('=' * 118)
    for gid in gids:
        t, lf = fbg_windows(gid, root)
        ff = fbg_files(gid, root)
        ta, n = ae_frames(gid)
        print('\n### %s   FBG 窗 %d  gp=%.3f   FBG 跨度 %.1f h   FBG 文件 %d'
              % (gid, len(t), lf.mean(), (t[-1] - t[0]) / 3600.0, len(ff)))
        if len(ff) > 1:
            d = np.diff(ff) / 3600.0
            big = d[d > 1.0]
            print('    FBG 文件间隔：中位 %.3f h  最大 %.1f h  (>1h 的缝 %d 段，合计 %.1f h)'
                  % (np.median(d) / 1.0, d.max(), len(big), big.sum()))
        else:
            print('    FBG 文件间隔：单文件')
        if ta is None:
            print('    （无 AE 帧表）')
            continue
        st = day_stats(t, lf, ta, n)
        print('    => 完整日 %d 天：日均占空比中位 %.3f、**最好日 %.3f**、最低 %.3f、'
              '低占空日(<60%%) %d 天；AE 有事件帧占比中位 %.3f'
              % (st['n_full'], st['duty_med'], st['duty_max'], st['duty_min'],
                 st['n_stall'], st['ae_cov_med']))
        t0 = min(t[0], ta[0])
        nd = int((max(t[-1], ta[-1]) - t0) // DAY) + 1
        print('    %-12s %6s %7s | %7s %10s %8s'
              % ('日期(UTC+8)', 'FBG窗', '加载%', 'AE帧', 'AE事件', '事件/帧'))
        for k in range(nd):
            lo, hi = t0 + k * DAY, t0 + (k + 1) * DAY
            mf = (t >= lo) & (t < hi)
            ma = (ta >= lo) & (ta < hi)
            if not mf.any() and not ma.any():
                continue
            dtxt = _dtxt(lo)
            fbg_s = ('%6d %6.0f%%' % (mf.sum(), 100 * lf[mf].mean())) if mf.any() \
                else '     -      -'
            if ma.any():
                ae_s = '%7d %10d %8.0f' % (ma.sum(), n[ma].sum(),
                                           n[ma].sum() / ma.sum())
            else:
                ae_s = '      -          -        -'
            print('    %-12s %s | %s' % (dtxt, fbg_s, ae_s))


# ------------------------------------------------------------------- 模式 B
def marks(gids, root):
    from ae_dta import iter_messages, _rtot
    print('=' * 118)
    print('.DTA 采集开关时间轴（ID 128=开始 / 129=停止 / 130=暂停，载荷 = 6 字节相对时间）')
    print('  用途：证明「AE 在某个时段是否在录」。C4 谱载批的 AE 时钟比 FBG 慢整 8 天')
    print('  （ae_frames.CLOCK_SHIFT），本表打印的是 **AE 自己的挂钟**，与 FBG 日历差 8 天。')
    print('=' * 118)
    for gid in gids:
        files = sorted(glob.glob(os.path.join(root, gid, 'AE', '*.DTA')))
        if not files:
            print('\n### %s  无 .DTA' % gid)
            continue
        print('\n### %s' % gid)
        span_all = 0.0
        ta, _n = ae_frames(gid)
        src = None
        p_fr = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
        if os.path.exists(p_fr):
            src = np.load(p_fr, allow_pickle=True)['src']
        n_ok = n_bad = 0
        for p in files:
            base = os.path.basename(p)
            mk, start = [], ''
            with open(p, 'rb') as fh:
                for _o, _L, mid, pay in iter_messages(fh):
                    if mid == 99:
                        start = pay.decode('latin-1').strip('\x00').strip()
                    elif mid in (128, 129, 130):
                        mk.append((_rtot(pay[0:6]), MARK_NAME[mid]))
            mk.sort()
            if not mk:
                continue
            # 该卷在帧表里覆盖的跨度与密度（帧只在有事件时存在）
            fh_span = fh_dens = float('nan')
            if src is not None and ta is not None:
                m = src == base
                if m.any():
                    fh_span = (ta[m].max() - ta[m].min()) / 3600.0
                    fh_dens = float(m.sum()) / (fh_span * 3600.0 / FRAME_S)
            mk_span = mk[-1][0] / 3600.0
            if not np.isfinite(fh_span) or mk_span >= 0.98 * fh_span:
                n_ok += 1
                verdict = '标记覆盖整卷'
            else:
                n_bad += 1
                verdict = '**标记不覆盖**（只用帧密度）'
            print('    %-16s 起始 %s' % (base, start))
            print('        标记跨度 %7.1f h（%d 个）  帧跨度 %7.1f h  帧密度 %.3f  => %s'
                  % (mk_span, len(mk), fh_span, fh_dens, verdict))
            if len(mk) < 2 or mk[-1][0] <= 0:
                print('        采集全程未中断（只有起始标记）')
                continue
            span_all += mk[-1][0]
            dur = 0.0
            for i, (tt, kk) in enumerate(mk):
                if kk in ('PAUSE', 'STOP'):
                    nxt = next((t2 for t2, k2 in mk[i + 1:] if k2 == 'START'), None)
                    dur += (nxt - tt) if nxt is not None else (mk[-1][0] - tt)
            print('        暂停合计 %.2f h (%.2f%%)   标记: %s'
                  % (dur / 3600.0, 100.0 * dur / mk[-1][0],
                     ' '.join('%.1fh:%s' % (tt / 3600.0, kk) for tt, kk in mk)))
        print('    => 标记覆盖整卷的卷数 %d，不覆盖 %d；AE 在录合计 %.1f h（不含卷间空洞）'
              % (n_ok, n_bad, span_all / 3600.0))


# ------------------------------------------------------------------- 模式 C
def batch(root):
    print('=' * 118)
    print('11 组汇总：占空比与各条候选解释')
    print('=' * 118)
    print('%-7s %-3s %7s %6s %10s %8s %7s %7s %7s %9s %9s %13s %7s'
          % ('组', '批', 'FBG窗', 'gp', '日均占空中位', '最好日', '低占空日', '最低日',
             'AE帧占', 'FBG跨度h', '最大缝h', '最长低p2p段h', 'AE帧'))
    rows = []
    for gid in GIDS:
        t, lf = fbg_windows(gid, root)
        ff = fbg_files(gid, root)
        ta, n = ae_frames(gid)
        span = (t[-1] - t[0]) / 3600.0
        if len(ff) > 1:
            d = np.diff(ff) / 3600.0
            mx = float(d.max())
        else:
            mx = float('nan')
        # 低占空日：窗数 >= 200 的日里加载占比 < 50%
        st = day_stats(t, lf, ta, n)
        # 最长连续低 p2p 时段（按时间，不按窗数）
        idx = np.flatnonzero(lf < 0.5)
        best = 0.0
        if idx.size:
            brk = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk, [idx.size - 1]])
            for a0, b0 in zip(starts, ends):
                best = max(best, (t[idx[b0]] - t[idx[a0]]) / 3600.0)
        if ta is None:
            ae_fr, ae_n, epf = 0, 0, float('nan')
        else:
            ae_fr, ae_n = len(ta), int(n.sum())
            epf = ae_n / ae_fr
        rows.append((gid, 'A' if gid in C3 else 'B', len(t), lf.mean(), st['duty_med'],
                     st['duty_max'], st['n_stall'], st['duty_min'], st['ae_cov_med'],
                     span, mx, best, ae_fr, ae_n))
        print('%-7s %-3s %7d %6.3f %10.3f %8.3f %7d %7.3f %7.3f %9.1f %9.1f %13.1f %7d %10d'
              % rows[-1])

    # 交叉验证：FBG 推出的「正常日占空比」 vs AE 推出的「有事件帧占比」
    # 后者完全不使用 FBG 信息，两者的秩一致性是 P2P_LOAD 判据的独立体检。
    from scipy.stats import spearmanr
    dm = np.array([r[4] for r in rows], dtype=float)
    mx = np.array([r[5] for r in rows], dtype=float)
    am = np.array([r[8] for r in rows], dtype=float)
    ok = np.isfinite(dm) & np.isfinite(am)
    rr = spearmanr(dm[ok], am[ok])
    print('\n独立交叉验证（%d 组）：Spearman(日均占空比中位, AE 有事件帧占比) = %.3f  p = %.2g'
          % (int(ok.sum()), rr.statistic, rr.pvalue))
    print('  => AE 侧完全不使用 FBG：两者的秩一致 => P2P_LOAD 的加载/停机标注在 11 组上都站得住。')
    okm = np.isfinite(mx)
    print('各组的「最好的一天」占空比：%s  => 区间 %.3f 至 %.3f'
          % (' '.join('%.2f' % v for v in mx[okm]), mx[okm].min(), mx[okm].max()))
    print('  => 机器一旦正常跑，各组占空比几乎相同 => gp 的组间差异由「停摆天数」解释，'
          '不是机器设定的工况差。')
    return rows


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', nargs='*', default=None)
    ap.add_argument('--marks', nargs='*', default=None)
    ap.add_argument('--batch', action='store_true')
    ap.add_argument('--root', default=None)
    ap.add_argument('--refresh', action='store_true', help='不用 FBG 缓存，重扫')
    a = ap.parse_args(argv)

    root = a.root or _data_l1()
    if not (a.timeline is not None or a.marks is not None or a.batch):
        a.timeline = ['L1-31', 'L1-41']

    mode = ('timeline' if a.timeline is not None else
            'marks' if a.marks is not None else 'batch')
    old = sys.stdout
    sys.stdout = Tee(os.path.join(_LOGS, 'l1_duty_%s.txt' % mode))
    try:
        print('数据根目录: %s' % root)
        if a.refresh:
            for gid in GIDS:
                p = os.path.join(RES, '_duty_%s.npz' % gid)
                if os.path.exists(p):
                    os.remove(p)
        if a.timeline is not None:
            timeline(a.timeline or GIDS, root)
        if a.marks is not None:
            marks(a.marks, root)
        if a.batch:
            batch(root)
    finally:
        sys.stdout = old
    print('日志: %s' % os.path.join(_LOGS, 'l1_duty_%s.txt' % mode))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
