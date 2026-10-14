"""两套独立传感系统给出同一结构结论：批间效应远小于试件间离散。

(a) AE（声发射）：cum_hits 的 t85 —— 累积 AE 命中数达 85% 时的寿命百分数
(b) PZT（导波）：波速漂移的 Theil-Sen 斜率 —— µs / 100% 寿命
(c) 把两套不同量纲的指标化到同一尺度：批间极差 与 组内极差 的比较（组内归一为 1）

为什么两套系统可并置：它们物理上无关（声发射 vs 主动导波）、仪器不同、
采集间隔不同（5000 vs 5200 cycles）、机构也不同。
若两者都给出「批间 ≪ 组内」，则「绝对阈值不可迁移、可迁移的是方向」这一判断
就不是单一系统的偶然。

只读 l1/results 下已存在的 CSV，只写 l1/figures/fig_two_systems.png。
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

BATCHES = ['C1', 'C2', 'C3', 'C4']
COL = {'C1': '#1f77b4', 'C2': '#7fa9d4', 'C3': '#d1701a', 'C4': '#c0392b'}
LABEL = {'C1': 'C1  (Lab A)', 'C2': 'C2  (Lab A)',
         'C3': 'C3  (Lab B)', 'C4': 'C4  (Lab B)'}
AE_VARIANT = 'cum_hits'
NPERM = 20000


def read_ae():
    """AE 侧逐试件 t85（寿命%）。"""
    out = {}
    with open(os.path.join(RES, 'l1_hi_crosslab.csv'), encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['variant'] != AE_VARIANT:
                continue
            v = float(r['t85_pct'])
            if np.isfinite(v):
                out[r['gid']] = (r['batch'], v)
    return out


def read_pzt():
    """PZT 侧逐试件 Theil-Sen 斜率（µs / 100% 寿命）。"""
    out = {}
    with open(os.path.join(RES, 'pzt_crosslab_summary.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['tof_slope_ts'] == '':
                continue
            out[r['gid']] = (r['batch'], float(r['tof_slope_ts']))
    return out


def spreads(d):
    """-> 批中位表, 批间极差, 逐批组内极差。"""
    med, rng = {}, {}
    for b in BATCHES:
        v = np.array([x[1] for x in d.values() if x[0] == b], float)
        if len(v):
            med[b] = float(np.median(v))
            rng[b] = (float(v.min()), float(v.max()), len(v))
    both = list(med.values())
    between = max(both) - min(both) if len(both) > 1 else np.nan
    within = float(np.median([r[1] - r[0] for r in rng.values()]))
    return med, between, within, rng


def perm_p(d, rng_seed=20261011, nperm=NPERM):
    """批间中位极差的随机化检验（与 hi_crosslab.rand_test 同口径）。

    统计量 = 各批中位数的极差。计数方向为 `spread_perm <= spread_obs`，
    即 **p 越小 = 观测到的批间极差比随机重排更小 = 越不像存在批效应**。
    （反向计数得到的是它的补数，两者容易搞混，引用时必须写明方向。）
    """
    vals = np.array([x[1] for x in d.values()], float)
    labs = np.array([x[0] for x in d.values()])
    obs = spreads(d)[1]
    sizes = {b: int((labs == b).sum()) for b in np.unique(labs)}
    if len(sizes) < 2:
        return obs, np.nan
    rng = np.random.default_rng(rng_seed)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(labs)
        m = [np.median(vals[p == b]) for b in sizes if (p == b).any()]
        if max(m) - min(m) <= obs + 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (nperm + 1)


def strip(ax, d, batches, ylab, title, pval=None):
    med, between, within, rng = spreads(d)
    for i, b in enumerate(batches):
        v = np.array([x[1] for x in d.values() if x[0] == b], float)
        if not len(v):
            continue
        jit = (np.random.default_rng(7).random(len(v)) - 0.5) * 0.30
        ax.scatter(np.full(len(v), i) + jit, v, s=16, color=COL[b],
                   edgecolor='white', linewidth=0.4, zorder=3)
        ax.hlines(med[b], i - 0.30, i + 0.30, color='k', lw=2.0, zorder=4)
        ax.vlines(i, rng[b][0], rng[b][1], color=COL[b], lw=0.8, alpha=0.35, zorder=2)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(['%s\n%s' % (b, LABEL[b].split('(')[1].rstrip(')'))
                        for b in batches], fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.25, axis='y')
    ax.set_xlim(-0.6, len(batches) - 0.4)
    txt = '批间极差 %s\n组内极差中位 %s' % (fmt(between), fmt(within))
    if pval is not None and np.isfinite(pval):
        txt += '\np(重排≤观测) = %.3f' % pval
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va='top', fontsize=6.5,
            bbox=dict(fc='white', ec='0.8', lw=0.4, alpha=0.9))
    return med, between, within


def fmt(v):
    """三位有效数字，避免 0.0122 被四舍五入成 0.01 而失去意义。"""
    return '%.3g' % v


def main():
    os.makedirs(FIG, exist_ok=True)
    ae = read_ae()
    pz = read_pzt()
    print('AE 试件 %d 个：%s' % (len(ae), {b: sum(1 for x in ae.values() if x[0] == b)
                                          for b in BATCHES}))
    print('PZT 试件 %d 个：%s' % (len(pz), {b: sum(1 for x in pz.values() if x[0] == b)
                                           for b in BATCHES}))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85))

    ae_pv = perm_p(ae)[1]
    pz_pv = perm_p(pz)[1]
    strip(axes[0], ae, ['C1', 'C2', 'C3', 'C4'],
          't85：寿命位置 [%]',
          '(a) 声发射 AE：累积 hits 达 85% 时的寿命', ae_pv)
    strip(axes[1], pz, ['C1', 'C3', 'C4'],
          '波速漂移率 [µs / 100% 寿命]',
          '(b) 导波 PZT：TOF 漂移的稳健斜率', pz_pv)
    axes[1].set_ylim(0, max(x[1] for x in pz.values()) * 1.18)

    # (c) 无量纲对比：批间极差 vs 组内极差中位（后者归一为 1）
    ae_between, ae_within = spreads(ae)[1], spreads(ae)[2]
    pz_between, pz_within = spreads(pz)[1], spreads(pz)[2]
    x = np.arange(2)
    w = 0.36
    axes[2].bar(x - w / 2, [ae_between / ae_within, pz_between / pz_within],
                w, color='#c0392b', label='批间极差')
    axes[2].bar(x + w / 2, [1.0, 1.0], w, color='#4c78a8', label='组内极差（中位，归一）')
    for i, (bb, ww) in enumerate([(ae_between, ae_within), (pz_between, pz_within)]):
        axes[2].text(i - w / 2, bb / ww + 0.04, '%.2f×' % (bb / ww),
                     ha='center', fontsize=7.5, color='#c0392b', fontweight='bold')
        axes[2].text(i + w / 2, 1.04, '1.00×', ha='center', fontsize=7.5, color='#4c78a8')
        axes[2].text(i - w / 2, 0.06, '%s' % fmt(bb), ha='center', fontsize=6, color='w')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['AE\nn=%d' % len(ae), 'PZT\nn=%d' % len(pz)], fontsize=7)
    axes[2].set_ylabel('相对组内极差', fontsize=7.5)
    axes[2].set_title('(c) 两套系统的离散分解（无量纲）', fontsize=8.5)
    axes[2].legend(fontsize=6.5, frameon=False, loc='upper center', ncol=1)
    axes[2].axhline(1.0, color='0.6', ls=':', lw=0.8)
    axes[2].grid(alpha=0.25, axis='y')
    axes[2].set_ylim(0, 1.30)
    axes[2].set_xlabel('（红色数字为批间极差原始值，单位同各自纵轴）', fontsize=6)

    fig.tight_layout(pad=0.6)
    fp = os.path.join(FIG, 'fig_two_systems.png')
    fig.savefig(fp, dpi=300)
    fig.savefig(fp.replace('.png', '.pdf'))
    plt.close(fig)

    # 置换检验
    ob, pv = perm_p(ae)
    print('\nAE  批间极差 %.2f pp，组内极差中位 %.2f pp，比值 %.2f×' % (ae_between, ae_within, ae_between / ae_within))
    print('    p(随机重排的批间极差 <= 观测值) = %.3f   越小 = 批间越相似、越不像有批效应' % pv)
    ob2, pv2 = perm_p(pz)
    print('PZT 批间极差 %.4f，组内极差中位 %.4f，比值 %.2f×' % (pz_between, pz_within, pz_between / pz_within))
    print('    p(随机重排的批间极差 <= 观测值) = %.3f' % pv2)
    print('\n已写', fp, '与同名 .pdf')
    print('\n逐批中位：')
    for b in BATCHES:
        a = [x[1] for x in ae.values() if x[0] == b]
        p = [x[1] for x in pz.values() if x[0] == b]
        print('  %-3s AE n=%2d 中位 %6.1f%%   |   PZT n=%2d 中位 %+.4f'
              % (b, len(a), np.median(a) if a else np.nan,
                 len(p), np.median(p) if p else np.nan))


if __name__ == '__main__':
    main()
