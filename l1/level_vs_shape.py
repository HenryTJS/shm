"""对照实验：绝对阈值法（现行做法）vs 形状归一化指标（本文做法）。

为什么要这个对照
    ICSSD 议程集里几乎所有方法型论文都做了同类对照（提议方法 vs 一个基线）。
    我们缺这一步，所以「SNHI 更好」一直只是断言。

设计要点：让两种方法产出**同一种量**，才可比
    两者最终都给出「该试件在寿命的百分之几触发/判定」：
      (A) 阈值法：HI = 每 500 cycle 的 AE hit 数（速率型，最自然的报警量）。
          阈值 θ 由**留一批**定：θ = 其余批各试件「失效时 HI」的中位。
          达阈时刻 = 该试件 3 箱滑动均值首次超过 θ 的寿命百分数。
      (B) 形状法（SNHI）：累积 hit 曲线自归一化后达到 85% 时，所处的寿命百分数（t85）。

    统一用同一个统计量评价两者：**批间极差 / 组内极差中位**（越小越可迁移）。

数据
    仅用 C3 + C4 的 15 个组（Lab B），因为这两批有逐帧产物 `_l1ae_frames_*.npz`
    与循环轴 `_l1cyc_*.npz`，无需重读任何原始大文件。
    Lab A 侧的阈值对照已有既有产物 `l1_alarm_summary.csv`（9 组，跨组离散 46 至 66 pp），
    作为旁证引用，不在本脚本重算。

只读 l1/results/*.npz，只写 l1/results/level_vs_shape.csv。
"""
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

CYCS_BIN = 500.0          # 与 AE 侧分箱一致
PERSIST = 3               # 滑动持续箱数（抑制单箱假报）
TAIL_FRAC = 0.05          # 「失效时 HI」取寿命最后 5% 的均值

BATCH = {}
for g in ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']:
    BATCH[g] = 'C3'
for g in ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35', 'L1-36',
          'L1-41', 'L1-44']:
    BATCH[g] = 'C4'


def curve(gid):
    """-> (life%, hit_per_bin, cum_hits) 或 None。"""
    fp = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    cp = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    with np.load(fp, allow_pickle=True) as z:
        fr = z['frame'].astype(np.int64)
        n = z['n'].astype(float)
    with np.load(cp, allow_pickle=True) as z:
        cf = z['frame'].astype(np.int64)
        cyc = z['cycle'].astype(float)
        n_f = float(z['n_f'])
    m = dict(zip(cf.tolist(), cyc.tolist()))
    keep = np.array([int(x) in m for x in fr])
    fr, n = fr[keep], n[keep]
    c = np.array([m[int(x)] for x in fr], float)
    ok = np.isfinite(c) & (c >= 0)
    c, n = c[ok], n[ok]
    if c.size == 0:
        return None
    o = np.argsort(c)
    c, n = c[o], n[o]
    nbin = max(2, int(np.floor(n_f / CYCS_BIN)))
    bi = np.clip((c / CYCS_BIN).astype(int), 0, nbin - 1)
    hit = np.bincount(bi, weights=n, minlength=nbin)[:nbin]
    life = (np.arange(nbin) + 0.5) / nbin * 100.0
    return life, hit, np.cumsum(hit), n_f


def roll(x, w):
    if w <= 1:
        return x
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return (c[i + 1] - c[np.maximum(0, i + 1 - w)]) / np.minimum(i + 1, w)


def run():
    C = {}
    for g in BATCH:
        r = curve(g)
        if r is not None:
            C[g] = r
    print('可用组 %d 个（C3 %d，C4 %d）\n'
          % (len(C), sum(1 for g in C if BATCH[g] == 'C3'),
             sum(1 for g in C if BATCH[g] == 'C4')))

    # 失效时的 HI（寿命最后 TAIL_FRAC 的均值）
    fail_hi = {}
    for g, (life, hit, cum, n_f) in C.items():
        m = life >= 100 * (1 - TAIL_FRAC)
        v = hit[m]
        v = v[np.isfinite(v)]
        fail_hi[g] = float(v.mean()) if v.size else np.nan

    # 直接量化「阈值为什么不可迁移」：失效值本身跨试件差多少倍
    print('失效时的 AE hit 数（每 500 cycle）：')
    for b in ('C3', 'C4'):
        v = np.array([fail_hi[g] for g in C if BATCH[g] == b], float)
        v = v[np.isfinite(v)]
        if v.size:
            print('  %s n=%2d  中位 %10.0f   最小 %10.0f   最大 %10.0f   **跨度 %.1f 倍**'
                  % (b, v.size, np.median(v), v.min(), v.max(), v.max() / max(v.min(), 1e-9)))
    nan_g = [g for g in C if not np.isfinite(fail_hi[g])]
    if nan_g:
        print('  （无有效尾段的组，已从训练集剔除：%s）' % ', '.join(sorted(nan_g)))
    print()

    # ---- (A) 留一批的绝对阈值法
    onset_A = {}
    for hold in ('C3', 'C4'):
        train = [fail_hi[g] for g in C if BATCH[g] != hold and np.isfinite(fail_hi[g])]
        theta = float(np.median(train))
        for g in C:
            if BATCH[g] != hold:
                continue
            life, hit, cum, n_f = C[g]
            s = roll(hit, PERSIST)
            idx = np.where(s > theta)[0]
            # 达阈值还需该试件本身出现过有限值（否则只能记未触发）
            onset_A[g] = float(life[idx[0]]) if idx.size else np.nan
        print('  留出 %s：阈值 θ = %.0f（由另一批 %d 个有效试件定）' % (hold, theta, len(train)))

    # ---- (B) 形状法
    onset_B = {}
    for g, (life, hit, cum, n_f) in C.items():
        tot = cum[-1]
        onset_B[g] = (float(life[min(int(np.searchsorted(cum, 0.85 * tot)), len(life) - 1)])
                      if tot > 0 else np.nan)

    def stat(onset):
        med, within, miss = {}, [], 0
        for b in ('C3', 'C4'):
            v = np.array([onset[g] for g in C if BATCH[g] == b], float)
            miss += int((~np.isfinite(v)).sum())
            v = v[np.isfinite(v)]
            if len(v):
                med[b] = float(np.median(v))
                within.append(v.max() - v.min())
        bt = max(med.values()) - min(med.values()) if len(med) > 1 else np.nan
        wi = float(np.median(within)) if within else np.nan
        return med, bt, wi, miss

    print('\n%-26s %8s %8s %9s %11s %7s %6s'
          % ('方法', 'C3 中位', 'C4 中位', '批间 pp', '组内中位 pp', '比值', '未触发'))
    print('-' * 80)
    out = {}
    for name, onset in [('(A) 绝对阈值（留一批）', onset_A), ('(B) 形状归一化 SNHI', onset_B)]:
        med, bt, wi, miss = stat(onset)
        out[name] = (med, bt, wi, miss)
        print('%-26s %8s %8s %9.1f %11.1f %7.2f %6d'
              % (name, *['%.1f' % med.get(b, np.nan) for b in ('C3', 'C4')],
                 bt, wi, bt / wi if wi else np.nan, miss))

    print('\n逐试件：（A）达阈时刻 / （B）t85')
    for b in ('C3', 'C4'):
        for g in sorted(C, key=lambda x: (BATCH[x], x)):
            if BATCH[g] != b:
                continue
            print('  %-7s %s   (A) %8s   (B) %6.1f'
                  % (g, b,
                     '未触发' if not np.isfinite(onset_A[g]) else '%7.1f%%' % onset_A[g],
                     onset_B[g]))

    fp = os.path.join(RES, 'level_vs_shape.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['gid', 'batch', 'onset_level_pct', 'onset_shape_pct', 'fail_hi'])
        for g in sorted(C, key=lambda x: (BATCH[x], x)):
            w.writerow([g, BATCH[g],
                        '' if not np.isfinite(onset_A[g]) else '%.4f' % onset_A[g],
                        '%.4f' % onset_B[g], '%.2f' % fail_hi[g]])
    print('\n已写 %s' % fp)


if __name__ == '__main__':
    run()
