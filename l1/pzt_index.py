"""PZT 损伤指标管线 v2（取代早期的单组验证版；旧版已于 2026-10-14 清理）。

v1 的三处错误（已由诊断脚本证伪）：
  ① 列含义：Time_Response = [时间 | 激励参考 | 7 个接收通道]，
     第 1 列是激励监测（所有作动器下都 ≈9.0），不是接收器。
  ② 比较窗口：不能用整段记录（200~600 点），那里面绝大部分是尾波，
     尾波对波速极敏感 ⇒ 那是温度计不是损伤计。只用直达波群紧窗 [90, 240] µs。
  ③ 基准：不能只用 State_1。它取在加载/热瞬态之前，会把整个寿命压平。
     ⇒ 主指标用「相邻状态局部退相关率」，累积量仅作参考。

约定：
  作动器 k 发射时，7 个接收通道按 PZT 序号升序跳过 k：
     k=1 → [2,3,4,5,6,7,8]   k=5 → [1,2,3,4,6,7,8]
  路径标识 = (k, m)。
"""
import json
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

sys.stdout.reconfigure(encoding='utf-8')

R = os.environ.get('L1_ROOT', 'E:\\l1')
TS = re.compile(r'State_(\d+)_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})')

FS = 2.0e6
WS, WE = 180, 480          # 直达波群窗（点）= 90~240 µs
TRIM = 100                 # 窗内两侧各裁掉的点数，抑制窗边缘效应


def states_of(d):
    """按时间戳排序的状态目录。"""
    out = []
    for e in os.scandir(d):
        if e.is_dir() and e.name.startswith('State_'):
            m = TS.match(e.name)
            if m:
                yrs = tuple(int(x) for x in m.groups()[1:])
                out.append((yrs, int(m.group(1)), e.name))
    out.sort()
    return out


def ref_state(d):
    """从 structure_info.m 的 damage_info 表里找 REF 类别的状态名。

    行格式：'State_A' 'State_B' [NaN NaN] [NaN NaN] NaN NaN 'REF';
    """
    p = os.path.join(d, 'structure_info.m')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8', errors='replace') as fh:
        txt = fh.read()
    for m in re.finditer(r"'(State_\d+_[0-9_]+)'\s+'[^']*'[^;\n]*?'(REF|DAM)'", txt):
        if m.group(2) == 'REF':
            return m.group(1)
    return None


def rx_of(k):
    """作动器 k 发射时，列 -> PZT 序号的映射。"""
    return [i for i in range(1, 9) if i != k]


def bandpass(freq_khz, order=4):
    lo = max(0.3 * freq_khz, 20.0) * 1e3
    hi = min(2.0 * freq_khz, 900.0) * 1e3
    return butter(order, [lo / (FS / 2), hi / (FS / 2)], btype='band')


def prep(x, ba):
    b, a = ba
    y = filtfilt(b, a, x - x.mean())
    return y[WS + TRIM:WE - TRIM]


def rn(x):
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-30)


def scan_group(d, freq, rep=1, stride=1, max_states=None):
    """扫描一个 PZT-N 目录，返回逐状态逐路径的特征表。"""
    st = states_of(d)
    refname = ref_state(d)
    if max_states:
        st = st[:max_states]
    st = st[::stride]
    fk = float(re.match(r'(\d+)kHz', freq).group(1))
    ba = bandpass(fk)
    rows = []
    prev = None
    for i, (_, num, name) in enumerate(st):
        cur = {}
        ok = True
        for k in range(1, 9):
            p = os.path.join(d, name, freq, 'Actionneur%d' % k,
                             'measured_data_rep_%d.mat' % rep)
            if not os.path.exists(p):
                ok = False
                break
            tr = sio.loadmat(p)['Time_Response']
            drv = tr[:, 1]
            R7 = tr[:, 2:]
            dpk = float(np.abs(drv).max())
            for j, m in enumerate(rx_of(k)):
                x = prep(R7[:, j], ba)
                cur[(k, m)] = (rn(x), float(np.abs(x).max()) / max(dpk, 1e-30),
                               (int(np.argmax(np.abs(x))) + WS + TRIM) / FS * 1e6)
        if not ok:
            continue
        row = {'i': i, 'num': num, 'state': name, 'is_ref': (name == refname)}
        for key, (x, amp, tof) in cur.items():
            row['path_%d_%d' % key] = True
            row['amp_%d_%d' % key] = amp
            row['tof_%d_%d' % key] = tof
            if prev is not None and key in prev:
                row['dc_%d_%d' % key] = 1.0 - float(np.dot(x, prev[key][0]))
        rows.append((row, cur))
        prev = cur
        if (i + 1) % 20 == 0 or i == len(st) - 1:
            print('    已处理 %d/%d' % (i + 1, len(st)))
    return rows, refname


def summarise(rows, title):
    keys = [k for k in rows[-1][0] if k.startswith('dc_')]
    print()
    print('--- %s ---' % title)
    print('  状态 %d 个，pitch-catch 路径 %d 条' % (len(rows), len(keys)))
    step = max(1, len(rows) // 12)
    print('  %-6s %-9s %10s %10s %10s' % ('状态序', 'State', '局部D中位', '幅值比中位', 'TOF中位us'))
    for r, _ in rows[::step]:
        dcs = [r[k] for k in keys if k in r]
        amps = [r[k] for k in r if k.startswith('amp_')]
        tofs = [r[k] for k in r if k.startswith('tof_')]
        print('  %-6d %-9s %10.3f %10.3f %10.1f'
              % (r['i'], r['state'][:9],
                 np.median(dcs) if dcs else float('nan'),
                 np.median(amps), np.median(tofs)))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print('用法: pzt_index.py <组号> <PZT-N> [频率] [rep] [stride] [max_states]')
        return
    gid, pn = sys.argv[1], sys.argv[2]
    freq = sys.argv[3] if len(sys.argv) > 3 else '250kHz_5cycles'
    rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    mx = int(sys.argv[6]) if len(sys.argv) > 6 else None

    d = os.path.join(R, gid, pn)
    print('%s / %s / %s   (rep=%d, stride=%d)' % (gid, pn, freq, rep, stride))
    print('  窗 = %.0f~%.0f µs（点 %d~%d，内缩 %d）'
          % ((WS + TRIM) / FS * 1e6, (WE - TRIM) / FS * 1e6, WS, WE, TRIM))
    rows, refname = scan_group(d, freq, rep, stride, mx)
    print('  REF 状态 = %s' % (refname or '(未标注)'))
    tag = '%s_%s_%s' % (gid, pn, freq.split('kHz')[0])
    out = os.path.join('l1', 'results', 'pzt_%s.csv' % tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = sorted({k for r, _ in rows for k in r})
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(','.join(cols) + '\n')
        for r, _ in rows:
            fh.write(','.join('' if r.get(c) is None else
                              (r[c] if isinstance(r[c], str) else '%.6g' % r[c])
                              for c in cols) + '\n')
    print('  已写 %s（%d 行）' % (out, len(rows)))
    summarise(rows, '%s / %s / %s' % (gid, pn, freq))


if __name__ == '__main__':
    main()
