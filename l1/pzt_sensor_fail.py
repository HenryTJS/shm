# -*- coding: utf-8 -*-
"""传感器失效剔除的鲁棒性检查 —— 回答审稿人最可能问的那一问。

要回答的问题
------------
`pzt_crosslab.py` 的 `tof_x_us` 是**全部 56 条路径** TOF 的中位数
（见该脚本 `run_group` 里 `np.median([...for p in sig if p in tmpl])`）。
数据集自带文档明确记录了若干传感器在特定循环数断裂 / 脱粘 ⇒
**失效之后的路径仍在参与统计**。若这些路径在断裂后输出噪声，
中位数就被污染，`17/17 组斜率为正、16/17 组 CI 下界 > 0` 这条结论就不可信。

本脚本按文档剔除失效路径后重算，并给出**逐组前后对比**。

判据（**预先定死**）
-------------------
稳健   剔除后仍 17/17 组斜率为正、16/17 组 95% 置信下界 > 0
需交代 有任何一组符号翻转或 CI 越零 ⇒ 正文里必须单独说明是哪一组

失效表来源（**全部引自数据集自带文档，非推测**）
------------------------------------------------
C1（`L1-xx.pdf` 第 1 页试验表 + 备注）：
  L1-04  "After 75,000 cycles it was noted that PZT sensor 7 has disbanded
          and PZT sensor 8 has failed."
  L1-05  "PZT sensor 8 was noted to have failed at 80,800 cycles."
  L1-23  "PZT sensor 3 disbond noted at 356700 cycles"
  L1-03 / L1-09 无传感器失效记录。
C4（`L1-xx.txt`）：
  L1-25  304k cycles (PZT 6 broke); 439k cycles (PZT 7 disbonded)
  L1-29  60k cycles (PZT 8 disbonded); 326k cycles (PZT 1,3,4,8 broke)
  L1-31  (Impact broke 2-3 PZT 2, 6, probably 3); 453k cycles (PZT 1,3,4,7,8 broke)
  L1-44  420k cycles (PZT 3, 5 broke)

**未给编号、无法精确剔除的组**（正文须单独交代，本脚本不动它们）：
  L1-27  "PZT error"，未给编号
  L1-30  "(Impact – broke 4 PZT)"，未给编号
  L1-34  "(Impact broke 2 PZT)"，未给编号
  L1-35  "348k cycles (2 PZT were not connected)"，未给编号
  L1-36  "(Impact-1 PZT broke)"，未给编号
注：L1-41 "PZT error at 965k cycles - Matlab restarted" 与 L1-44 的
"somewhere prior of 900k" 都是**采集中断**（缺数据点），不是传感器断裂 ⇒ 不需要剔路径。

用法
----
    python l1/pzt_sensor_fail.py 4      # stride=4，约 2 分钟
    python l1/pzt_sensor_fail.py        # 默认也走 stride=4
只读原始 `.mat`；写 l1/results/pzt_sensor_fail.csv 与 _summary.csv。
"""
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 隔离 argv：pzt_crosslab 在 import 时会读 sys.argv[1] 当 stride
_argv = list(sys.argv)
sys.argv = [sys.argv[0]]
import pzt_crosslab as PC              # noqa: E402
sys.argv = _argv

PI = PC.PI
RES = PC.RES

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4

FAIL = {
    'L1-04': [(75000.0, [7, 8])],
    'L1-05': [(80800.0, [8])],
    'L1-23': [(356700.0, [3])],
    'L1-25': [(304000.0, [6]), (439000.0, [7])],
    'L1-29': [(60000.0, [8]), (326000.0, [1, 3, 4, 8])],
    'L1-31': [(0.0, [2, 6]), (453000.0, [1, 3, 4, 7, 8])],
    'L1-44': [(420000.0, [3, 5])],
}
UNKNOWN = ['L1-27', 'L1-30', 'L1-34', 'L1-35', 'L1-36']


def live(p, cyc, fs):
    """路径 p = (作动器, 接收 PZT 编号) 在循环数 cyc 时，两条腿是否都还活着。"""
    k, m = p
    for c0, lst in fs:
        if cyc >= c0 and (k in lst or m in lst):
            return False
    return True


def run_group(gid):
    batch, lab = PC.BATCH[gid]
    st = PC.dam_order(gid)
    if not st:
        return None, '无 DAM 状态'
    nf = PC.nf_of(gid, lab)
    if not np.isfinite(nf):
        return None, '拿不到 n_f'
    dtc = PC.INTERVAL[lab]
    ba = PI.bandpass(float(PC.FREQ.split('kHz')[0]))
    gdir = os.path.join(HERE, gid)
    fs = FAIL.get(gid, [])

    rows, prev, tmpl = [], None, None
    for j, (ep, pztN, state) in enumerate(st, start=1):
        if (j - 1) % STRIDE:
            continue
        sig, amps, tofs = PC.signals(gdir, pztN, state, ba)
        if sig is None:
            continue
        if tmpl is None:
            tmpl = sig
        cyc = j * dtc
        common = [p for p in sig if p in tmpl]
        if not common:
            prev = sig
            continue
        alive_ = [p for p in common if live(p, cyc, fs)]
        ta = [PC.delay_us(sig[p], tmpl[p]) for p in common]
        tl = [PC.delay_us(sig[p], tmpl[p]) for p in alive_]
        r = dict(gid=gid, batch=batch, lab=lab, idx=j, cycle=cyc,
                 life_pct=min(cyc / nf * 100.0, 100.0),
                 tof_all=float(np.median(ta)),
                 tof_live=float(np.median(tl)) if tl else np.nan,
                 n_all=len(common), n_live=len(alive_), D=np.nan)
        # D 必须算：生产口径 metrics() 用「D 有限」过滤行，而第一个状态的
        # tof_x_us 是 0（模板就是它自己，自相关延迟为 0）—— 那个 0 必须被丢掉。
        if prev is not None:
            dd = [1.0 - float(np.dot(sig[q], prev[q]))
                  for q in common if q in prev]
            if dd:
                r['D'] = float(np.median(dd))
        rows.append(r)
        prev = sig
    return rows, None


def slope(rows, key):
    x = np.array([r['life_pct'] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    d = np.array([r.get('D', np.nan) for r in rows], float)
    ok = np.isfinite(d) & np.isfinite(x)          # 与 metrics() 同口径
    x, y = x[ok], y[ok]
    keep = (x >= 0) & (x <= 100) & np.isfinite(y)
    x, y = x[keep], y[keep]
    o = np.argsort(x)
    x, y = x[o], y[o]
    if len(x) < 6:
        return np.nan, np.nan, np.nan, len(x)
    v, _, lo, hi = stats.theilslopes(y, x, 0.95)
    return float(v), float(lo), float(hi), len(x)


def main():
    print('传感器失效剔除的鲁棒性检查   stride=%d' % STRIDE)
    print('=' * 100)
    print('%-8s %-3s %6s %7s %8s | %10s %10s %9s | %10s %10s %9s | %s'
          % ('组', '批', '状态', '寿命%', '首次剔%',
             '斜率全路径', 'CI下界', 'CI上界', '斜率存活', 'CI下界', 'CI上界', '判定'))
    print('-' * 100)

    detail, summ = [], []
    for gid in sorted(PC.BATCH):
        if not PC.BATCH[gid][0] in ('C1', 'C3', 'C4'):
            continue
        rows, err = run_group(gid)
        if rows is None or len(rows) < 6:
            print('%-8s %-3s   --  %s' % (gid, PC.BATCH[gid][0], err or '状态不足'))
            continue
        v0, l0, h0, n0 = slope(rows, 'tof_all')
        v1, l1, h1, n1 = slope(rows, 'tof_live')
        fs = FAIL.get(gid, [])
        first = min((c0 / PC.nf_of(gid, PC.BATCH[gid][1]) * 100.0 for c0, _ in fs),
                    default=np.nan)
        flip = '翻转!' if (np.isfinite(v0) and np.isfinite(v1)
                          and np.sign(v0) != np.sign(v1)) else ''
        if not fs:
            verdict = '无失效记录'
        elif flip:
            verdict = '符号翻转'
        elif not (np.isfinite(l1) and l1 > 0):
            verdict = 'CI 越零'
        else:
            verdict = '稳健'
        print('%-8s %-3s %6d %7.0f %8s | %+10.4f %+10.4f %+9.4f | %+10.4f %+10.4f %+9.4f | %s'
              % (gid, PC.BATCH[gid][0], n0, max(r['life_pct'] for r in rows),
                 '--' if not fs else '%.1f' % first,
                 v0, l0, h0, v1, l1, h1,
                 verdict + (' ' + flip if flip else '')))
        summ.append(dict(gid=gid, batch=PC.BATCH[gid][0], n=n0,
                         life_max=max(r['life_pct'] for r in rows),
                         first_fail_pct=first,
                         slope_all=v0, lo_all=l0, hi_all=h0,
                         slope_live=v1, lo_live=l1, hi_live=h1,
                         has_fail=len(fs) > 0, verdict=verdict))
        detail += rows

    if not summ:
        print('没有可用的组')
        return

    # ---- 自检：无失效记录的组，全路径列必须与生产口径逐位一致 ----
    # 这是本脚本唯一的口径防线。若这里报不一致，下面的剔除结果全部不可用。
    fp0 = os.path.join(RES, 'pzt_crosslab_summary.csv')
    if STRIDE == 1 and os.path.exists(fp0):
        ref = {}
        with open(fp0, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                ref[r['gid']] = r
        print('\n[自检] 与生产口径 pzt_crosslab_summary.csv 对账（只比无失效记录的组）')
        bad = cnt = 0
        for s in summ:
            if s['has_fail'] or s['gid'] not in ref:
                continue
            cnt += 1
            rv = float(ref[s['gid']]['tof_slope_ts'])
            rel = abs(s['slope_all'] - rv) / max(abs(rv), 1e-12)
            okd = rel < 0.02
            bad += 0 if okd else 1
            print('  %-8s 本脚本 %+.6f   生产 %+.6f   相对差 %6.3f%%  %s'
                  % (s['gid'], s['slope_all'], rv, rel * 100,
                     'OK' if okd else '<== 不一致'))
        print('  比对 %d 组，不一致 %d 组%s'
              % (cnt, bad, '' if bad == 0 else '   <<< 口径未对齐，结果不可用'))
    else:
        print('\n[自检] stride=%d：生产口径只存了 stride=1 的汇总，跳过对账' % STRIDE)

    print('-' * 100)
    a_pos = sum(1 for s in summ if np.isfinite(s['slope_all']) and s['slope_all'] > 0)
    l_pos = sum(1 for s in summ if np.isfinite(s['slope_live']) and s['slope_live'] > 0)
    a_ci = sum(1 for s in summ if np.isfinite(s['lo_all']) and s['lo_all'] > 0)
    l_ci = sum(1 for s in summ if np.isfinite(s['lo_live']) and s['lo_live'] > 0)
    N = len(summ)
    print('斜率 > 0     ：全路径 %d/%d  →  存活路径 %d/%d' % (a_pos, N, l_pos, N))
    print('CI 下界 > 0  ：全路径 %d/%d  →  存活路径 %d/%d' % (a_ci, N, l_ci, N))
    nf_grp = [s for s in summ if s['has_fail']]
    if nf_grp:
        print('\n有失效记录的 %d 组：' % len(nf_grp))
        for s in nf_grp:
            print('  %-8s 首次剔除在寿命 %5.1f%%   斜率 %+.4f → %+.4f   %s'
                  % (s['gid'], s['first_fail_pct'], s['slope_all'],
                     s['slope_live'], s['verdict']))
    if UNKNOWN:
        print('\n[!] 以下 %d 组文档给了失效事件但未给编号，无法精确剔除（正文须交代）：'
              % len(UNKNOWN))
        print('   ' + ', '.join(UNKNOWN))

    fp = os.path.join(RES, 'pzt_sensor_fail.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['gid', 'batch', 'lab', 'idx', 'cycle',
                                           'life_pct', 'tof_all', 'tof_live',
                                           'n_all', 'n_live', 'D'])
        w.writeheader()
        w.writerows(detail)
    sp = os.path.join(RES, 'pzt_sensor_fail_summary.csv')
    with open(sp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print('\n已写 %s\n     %s' % (fp, sp))


if __name__ == '__main__':
    import io as _io

    class _Tee:                      # PowerShell 管道会把中文输出毁掉，自己落盘一份
        def __init__(self, *ss):
            self.ss = ss

        def write(self, x):
            for s in self.ss:
                s.write(x)

        def flush(self):
            for s in self.ss:
                s.flush()

    _b = _io.StringIO()
    _o = sys.stdout
    sys.stdout = _Tee(_o, _b)
    try:
        main()
    finally:
        sys.stdout = _o
        with open(os.path.join(RES, '_pzt_fail_log.txt'), 'w',
                  encoding='utf-8') as _fh:
            _fh.write(_b.getvalue())
