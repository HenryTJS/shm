"""PZT 的**因果（在线）**版本：能否给出一个不需要逐件标定的达阈时刻？

背景（为什么必须做这一步）
    声发射侧的因果路已经被堵死：以寿命 5%/10%/20%/30%/50% 处的 HI 作锚点归一后，
    失效值 CV 仍为 1.622 / 1.317 / 0.967 / 0.986 / 1.145 —— 即「自校准」并未实现。
    导波侧还没试过。它的量是**速率**且 17/17 组符号一致，因此有可能不同。

规则（纯因果，不使用未来信息）
    锚点   TOF0 = 该试件寿命前 10% 内所有状态的 TOF 中位（只用到最早的一小段）
    指标   Δ(t) = TOF(t) − TOF0      （t 为当前寿命百分数）
    报警   Δ(t) ≥ Δ* 的**首次**时刻，记为达阈时刻 t_on

关键：Δ* 是一个**物理量纲的绝对阈值**（µs）。为了不挑数，对 Δ* 做**灵敏度扫描**，
    并报告每个 Δ* 下的批间离散 / 组内离散；同时给出「有多少试件根本没达到」。
    Δ* 的下限取估计噪声的 3 倍（互相关估计的同状态重复间 sd ≈ 0.19 µs ⇒ 3σ ≈ 0.57 µs），
    低于此值的达阈时刻没有意义。

只读 l1/results/pzt_crosslab.csv，只写 l1/results/pzt_causal_summary.csv。
"""
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

ANCHOR_PCT = 10.0            # 锚点：寿命前 10%
SIGMA = 0.190                # 互相关 TOF 估计的同状态重复间 sd（µs，实测）
D_STAR = [3 * SIGMA, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0]
ROLL_W = 20.0                # 滚动斜率窗宽（寿命百分点）
S_STAR = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]      # µs / 100% 寿命
BATCHES = ['C1', 'C3', 'C4']


def read_groups():
    g = {}
    with open(os.path.join(RES, 'pzt_crosslab.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            g.setdefault(r['gid'], []).append(r)
    out = {}
    for gid, rows in g.items():
        rows = [r for r in rows if r.get('tof_x_us') not in (None, '')]
        if len(rows) < 5:
            continue
        x = np.array([float(r['life_pct']) for r in rows], float)
        y = np.array([float(r['tof_x_us']) for r in rows], float)
        o = np.argsort(x)
        out[gid] = (rows[0]['batch'], x[o], y[o])
    return out


def onset(x, y, thr):
    """首次 Δ >= thr 的寿命百分数；未达到返回 nan。"""
    m = x > ANCHOR_PCT
    if not m.any():
        return np.nan
    xa, ya = x[m], y[m]
    d = ya - np.median(y[x <= ANCHOR_PCT])
    hit = np.where(d >= thr)[0]
    return float(xa[hit[0]]) if hit.size else np.nan


def roll_slope(x, y):
    """因果滚动斜率：对每个状态，用它前面 [t−W, t] 窗内的点做 Theil-Sen。"""
    xs, ss = [], []
    for i in range(len(x)):
        m = (x >= x[i] - ROLL_W) & (x <= x[i]) & (x > ANCHOR_PCT)
        if m.sum() >= 4:
            k = np.polyfit(x[m], y[m], 1)[0]
            xs.append(x[i])
            ss.append(float(k))
    return np.array(xs), np.array(ss)


def onset_s(xs, ss, thr):
    hit = np.where(ss >= thr)[0]
    return float(xs[hit[0]]) if hit.size else np.nan


def table(title, per_batch_onset, thresholds):
    """per_batch_onset(thr) -> [(gid, batch, onset)]；逐阈值打印离散度。"""
    print('\n=== %s ===' % title)
    print('%-9s %7s %8s %8s %8s %9s %11s %7s'
          % ('阈值', '达标组', 'C1中位', 'C3中位', 'C4中位', '批间pp', '组内中位pp', '比值'))
    print('-' * 76)
    for thr in thresholds:
        items = per_batch_onset(thr)
        med, within, n_ok = {}, [], 0
        for b in BATCHES:
            v = np.array([o for _, bb, o in items if bb == b], float)
            v = v[np.isfinite(v)]
            if len(v):
                med[b] = float(np.median(v))
                within.append(v.max() - v.min())
            n_ok += len(v)
        if len(med) < 2:
            continue
        bt = max(med.values()) - min(med.values())
        wi = float(np.median(within)) if within else np.nan
        print('%-9.3g %7d %8s %8s %8s %9.1f %11.1f %7.2f'
              % (thr, n_ok,
                 *['%.1f' % med[b] if b in med else '--' for b in BATCHES],
                 bt, wi, bt / wi if wi else np.nan))


def main():
    G = read_groups()
    print('组数 %d；锚点 = 寿命前 %.0f%%；Δ* 下限 = 3σ = %.2f µs'
          % (len(G), ANCHOR_PCT, 3 * SIGMA))

    # ---- 规则 A：绝对增量阈值（最直接的可用规则）
    per = {t: [(g, b, onset(x, y, t)) for g, (b, x, y) in G.items()] for t in D_STAR}
    print()
    for t in D_STAR:
        n = sum(1 for _, _, o in per[t] if np.isfinite(o))
        print('  Δ* = %5.2f µs：%2d/%d 个试件达到（未达到 = 该试件全寿命增量不足此值）'
              % (t, n, len(G)))
    table('规则 A：Δ(t) = TOF(t) − TOF(前 10%% 中位)，达阈时刻（寿命%%）',
          lambda t: per[t], D_STAR)

    # ---- 规则 B：滚动斜率阈值
    rows = []
    for g, (b, x, y) in G.items():
        xs, ss = roll_slope(x, y)
        for t in S_STAR:
            rows.append((g, b, t, onset_s(xs, ss, t)))
    table('规则 B：滚动 Theil-Sen 斜率（窗宽 %.0f 寿命%%），达阈时刻' % ROLL_W,
          lambda t: [(g, b, o) for g, b, tt, o in rows if tt == t], S_STAR)

    # ---- 逐试件明细（用中间的 Δ* = 2.0 µs 作代表）
    print('\n=== 逐试件明细（Δ* = 2.0 µs）===')
    for b in BATCHES:
        for g, bb, o in sorted(per[2.0]):
            if bb == b:
                print('  %-7s %s  达阈时刻 %s' % (g, b, '未达到' if not np.isfinite(o) else '%.1f%%' % o))

    # ---- 落盘
    fp = os.path.join(RES, 'pzt_causal_summary.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['rule', 'threshold', 'gid', 'batch', 'onset_pct'])
        for t in D_STAR:
            for g, b, o in per[t]:
                w.writerow(['delta_abs', t, g, b, '' if not np.isfinite(o) else '%.4f' % o])
        for g, b, t, o in rows:
            w.writerow(['roll_slope', t, g, b, '' if not np.isfinite(o) else '%.4f' % o])
    print('\n已写 %s' % fp)


if __name__ == '__main__':
    main()
