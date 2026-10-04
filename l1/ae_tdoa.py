# -*- coding: utf-8 -*-
"""TDOA 一维方位指标 —— 给这套数据补上「空间」这个维度。

为什么能做
----------
这批 `.DTA` **没有波形**，所以做不了波形互相关；但每条 hit 都带高精度触发时刻
（RTOT，0.25 µs 分辨率）。同一物理事件被两个传感器先后拾取 ⇒ 两条 hit 记录，
其时刻差就是 TDOA。

**配对真实性已验证**（`L1-29`，见会话结论）：
    |dt| ≤ 5 / 10 / 20 / 50 µs 的配对数分别占 23% / 47% / 54% / 75%，
    配对后两通道**幅值相关 r = +0.89 至 +0.92**，而打乱对照 r ≈ 0
⇒ 配对的是同一事件，不是巧合（该组通道 1 事件率仅 4 次/s，随机配对概率约 0.008%）。

指标（每帧 = 600 s，只统计配对事件）
------------------------------------
  · **dt_med**：TDOA 中位数 —— 声源偏向哪一侧（符号）与偏离多少（大小）
  · **dt_iqr**：TDOA 分散度 —— 声源在连线方向上有多"散"
  · **n_pair**：配对数（帧内事件量）
  · **r_amp**：配对幅值相关系数（**数据自检**：健康帧应维持高相关）

物理含义：脱粘／分层扩展会改变主要声源的位置 ⇒ `dt_med` 随寿命漂移。
注意：绝对位置需要传感器间距 D 与声速 c（PDF 未给 AE 传感器布置），
所以这里只做**相对方位**（组内可比较、可归一化），不报毫米坐标。

用法
----
    python l1/ae_tdoa.py                      # 全部有 .DTA 的组
    python l1/ae_tdoa.py --frame-s 600 L1-29
输出：results/_l1tdoa_{gid}.npz + results/l1_tdoa_rank.csv
"""

import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_dta import iter_messages, _rtot, CHID_NBYTES          # noqa: E402

FRAME_S = 600.0
PAIR_US = 20.0            # 配对阈值（µs）；实测 20 µs 内仍保持 r≈0.90
MIN_PAIR = 30             # 帧内配对数少于该值不算


def _chid_of(pay):
    """从 ID 42 载荷里取特征表（用于定位 AMP 字段偏移）。"""
    L = pay[2:]
    q = 0
    while q + 3 <= len(L):
        LS = int.from_bytes(L[q:q + 2], 'little')
        if LS <= 0 or q + LS + 2 > len(L):
            break
        if L[q + 2] == 5:
            return list(L[q + 4:q + 4 + L[q + 3]])
        q += LS + 2
    return None


def _amp_offset(chid):
    o = 7
    for c in chid:
        if c == 6:
            return o
        o += CHID_NBYTES[c]
    return None


def scan(path, frame_s=FRAME_S, pair_us=PAIR_US):
    """流式扫一卷 → 每帧一行统计。帧内配对，内存只装一帧。"""
    chid, off_amp = None, None
    buckets = {}
    cur = None

    def close(f, keep):
        if keep is None:
            return None
        n1 = len(keep[1]['t'])
        n2 = len(keep[2]['t'])
        if n1 < MIN_PAIR // 2 or n2 < MIN_PAIR // 2 or n1 + n2 < MIN_PAIR:
            return None
        t1 = np.asarray(keep[1]['t']); a1 = np.asarray(keep[1]['a'])
        t2 = np.asarray(keep[2]['t']); a2 = np.asarray(keep[2]['a'])
        if len(t1) == 0 or len(t2) == 0:
            return None
        # 对通道 2 的每条，找通道 1 最近的
        idx = np.clip(np.searchsorted(t1, t2), 1, len(t1) - 1)
        lo = t2 - t1[idx - 1]; hi = t2 - t1[idx]
        m_lo = np.abs(lo) <= np.abs(hi)
        d = np.where(m_lo, lo, hi)
        j = np.where(m_lo, idx - 1, idx)
        m = np.abs(d) * 1e6 <= pair_us
        if m.sum() < MIN_PAIR:
            return None
        dd = d[m]
        x = a1[j[m]].astype(float); y = a2[m].astype(float)
        r = np.corrcoef(x, y)[0, 1] if (x.std() > 0 and y.std() > 0) else np.nan
        rec = {'frame': f, 'n_pair': int(m.sum()),
               'dt_med': float(np.median(dd)) * 1e6,
               'dt_iqr': float(np.percentile(dd, 75) - np.percentile(dd, 25)) * 1e6,
               'amp_med1': float(np.median(x)), 'amp_med2': float(np.median(y)),
               'r_amp': float(r)}
        # 只用高幅值事件（主损伤源应产生大事件）—— 更聚焦于“主声源方位”
        amean = (x + y) / 2.0
        for tag, thr, min_n in (('75', 75.0, 15), ('85', 85.0, 8)):
            mm = amean >= thr
            rec['n_p' + tag] = int(mm.sum())
            rec['dt_med_p' + tag] = (float(np.median(dd[mm])) * 1e6
                                     if mm.sum() >= min_n else np.nan)
        return rec

    out = []
    with open(path, 'rb') as fh:
        for _off, LEN, mid, pay in iter_messages(fh):
            if mid == 42 and chid is None:
                chid = _chid_of(pay)
                off_amp = _amp_offset(chid) if chid else None
            elif mid == 1:
                if off_amp is None:
                    continue
                ci = pay[6]
                if ci not in (1, 2):
                    continue
                t = _rtot(pay[0:6])
                f = int(t // frame_s)
                if f != cur:
                    r = close(cur, buckets.pop(cur, None)) if cur is not None else None
                    if r:
                        out.append(r)
                    cur = f
                    buckets[cur] = {1: {'t': [], 'a': []}, 2: {'t': [], 'a': []}}
                b = buckets[cur][ci]
                b['t'].append(t)
                b['a'].append(pay[off_amp])
    r = close(cur, buckets.pop(cur, None)) if cur is not None else None
    if r:
        out.append(r)
    if not out:
        return None
    keys = list(out[0].keys())
    return {k: np.asarray([o[k] for o in out]) for k in keys}


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    frame_s = FRAME_S
    gids = []
    i = 0
    while i < len(argv):
        if argv[i] == '--frame-s':
            frame_s = float(argv[i + 1]); i += 2
        else:
            gids.append(argv[i]); i += 1
    root = _data_l1()
    gids = gids or sorted(d for d in os.listdir(root)
                          if d.startswith('L1-') and glob.glob(os.path.join(root, d, 'AE', '*.DTA')))
    rows = []
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        spearmanr = None
    print('%-7s %7s %9s %11s %11s %9s %9s %11s %11s' % (
        '组', '有效帧', '配对中位', 'dt_med中位us', 'dt_iqr中位us', '幅值相关',
        'rho(dt,帧)', 'rho(>=75dB)', 'rho(>=85dB)'))
    for gid in gids:
        acc = None
        for f in sorted(glob.glob(os.path.join(root, gid, 'AE', '*.DTA'))):
            d = scan(f, frame_s=frame_s)
            if d is None:
                continue
            acc = d if acc is None else {k: np.concatenate([acc[k], d[k]]) for k in d}
        if acc is None:
            print('%-7s %7s' % (gid, '无有效配对帧'))
            continue
        np.savez_compressed(os.path.join(RES, '_l1tdoa_%s.npz' % gid), **acc)
        fr = acc['frame'].astype(float)
        life = (fr - fr[0]) / max(fr[-1] - fr[0], 1.0)

        def rho_of(key):
            if spearmanr is None or key not in acc:
                return np.nan
            v = acc[key]
            ok = np.isfinite(v)
            if ok.sum() < 8:
                return np.nan
            return float(spearmanr(life[ok], v[ok])[0])

        rho = rho_of('dt_med')
        r75 = rho_of('dt_med_p75')
        r85 = rho_of('dt_med_p85')
        rows.append((gid, len(fr), float(np.median(acc['n_pair'])),
                     float(np.median(acc['dt_med'])), float(np.median(acc['dt_iqr'])),
                     float(np.nanmedian(acc['r_amp'])), rho, r75, r85))
        print('%-7s %7d %9.0f %11.2f %11.2f %9.3f %9.3f %11.3f %11.3f'
              % (gid, len(fr), np.median(acc['n_pair']), np.median(acc['dt_med']),
                 np.median(acc['dt_iqr']), np.nanmedian(acc['r_amp']), rho, r75, r85))
    if rows:
        import csv
        out = os.path.join(RES, 'l1_tdoa_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '有效帧', '配对中位', 'dt_med中位us', 'dt_iqr中位us',
                    '幅值相关中位', 'rho_dtmed_vs_帧序', 'rho_>=75dB', 'rho_>=85dB'])
        w.writerows(rows)
        for ci, name in ((6, '全部事件'), (7, '仅 >=75dB'), (8, '仅 >=85dB')):
            rr = [r[ci] for r in rows if np.isfinite(r[ci])]
            if rr:
                print('rho(dt_med, 寿命) [%s] 正 %d / 负 %d / 弱 %d（%d 组），|rho| 中位 %.2f'
                      % (name, sum(1 for v in rr if v > 0.3),
                         sum(1 for v in rr if v < -0.3),
                         sum(1 for v in rr if abs(v) <= 0.3), len(rr),
                         float(np.median(np.abs(rr)))))
        print('数据自检：各帧配对幅值相关中位 %.3f（应维持高值）'
              % float(np.median([r[5] for r in rows])))
        print('已写出 ->', out)
    return 0


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


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
