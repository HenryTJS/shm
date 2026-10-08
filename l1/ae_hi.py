# -*- coding: utf-8 -*-
"""在**循环轴**上复算 HI（健康指标）与 t85 —— 与 Broer 2021 / 老组口径直接可比。

口径
----
  · HI_hit  = 累积 AE hit 数 自归一化（Broer 2021 / Galanopoulos 2021 的写法，老组 C 批的交付口径）
  · HI_eng  = 累积 ABS-ENERGY 自归一化（能量口径的对照）
  · t85     = HI 达到 0.85 时的**寿命分数**（横轴用循环分数 cycle/n_f；若缺循环轴则用活跃帧序号）

**通道维度（2026-10-04 新增；⚠️ 2026-10-05 更正，勿再照抄旧结论）**
指标筛选实测（`results/l1_ae_index_rank.csv`，12 组）显示：

    口径          一致性   |rho|中位   反例
    全部 hit n      0.50      0.30       0
    **通道2 n_ch2    0.67      0.54       0**
    通道1 n_ch1     0.50      0.31       0

旧版本据此写「通道 2 明显优于全部 hit，主口径应改用通道 2」——
**这句只在「逐帧活动度」口径下成立；换成累积口径后结论反转**
（本脚本实测，与 `docs/details.md` §7 / §29 一致）：

    累积口径             t85 中位   范围           极差
    HI_hit（全部 hit）     93.9%    83.3 至 100.0   16.7   <= 交付主曲线
    HI_ch1（通道 1）       92.0%    85.1 至 100.0   14.9
    **HI_ch2（通道 2）      95.4%    77.8 至 100.0   22.2**  <= 反而最分散
    HI_eng（能量，已弃）    90.6%    21.1 至  98.4   77.3

机制：筛选用的判据是「逐帧活动度与寿命的秩相关」（通道 2 = 0.54、全部 hit = 0.30），
秩相关高意味着**速率随寿命上升更陡** ⇒ 累积曲线更凸 ⇒ t85 更分散。
这不是巧合，而是两者之间的必然联系。
**两个判据回答的是不同问题，不能互相替代。**

**选型规则（待办 1.6）：先定交付形态（逐帧秩相关 或 累积 t85），再据此选指标；
绝不能拿逐帧口径的优胜去挑累积口径的主曲线。**
本脚本输出三种累积口径**只是为了对比**，**交付主曲线始终是 `HI_hit`（全部 hit）**：
它与老组 B 批 92.6% / C 批 92.1% 同源可比，且 t85 极差最小（16.7）。

⚠️ 口径说明：累积量自归一化**必然用到终值** ⇒ 这是**离线复评**，不是在线预警。
老组（B 批 4 组 t85 中位 92.6%、C 批 9 组 92.1%）就是这个口径，所以这里的结果与它们可比。

帧格回落（2026-10-08 新增）
--------------------------
总循环数极少的组（L1-34 只有 1400 cycle，试验仅 0.2 h）在 **600 s 帧格**上只落
**2 帧**，会被「至少 3 个有效帧」的门槛挡掉 —— 这是**帧格分辨率**问题，不是数据问题。
现改为**逐级回落**：600 s 不足 3 帧就自动改用 **60 s 细格**（需先跑
`ae_frames.py --frame-s 60 --tag 60s` 与 `ae_cycle.py --frame-s 60 --tag 60s`）。
输出多一列 `帧长s` 标明用的是哪一格；细格的 `反解f_Hz` 会略高（格越细，
`load_h` 的离散化误差越小），但 `cycle` 末值恒等于 `n_f`（由构造保证）。

用法
----
    python l1/ae_hi.py                # 全部组
    python l1/ae_hi.py L1-29 L1-41
输出：results/l1_ae_hi.csv
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

# n_f：**唯一来源 = shm.datasets**（这里原本还有一份重复的硬编码，已删）
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from shm.datasets import N_F                                        # noqa: E402

# 帧格候选 (tag, 帧长s)：先用 600 s 生产口径，有效帧不足则回落到 60 s 细格。
# 见模块 docstring「帧格回落」。
GRIDS = (('', 600.0), ('60s', 60.0))
MIN_FRAMES = 3              # 「有效帧」下限；低于它就换更细的帧格


def t_at(hi, axis, level):
    """HI 首次达到 level 时的横轴值（横轴 = 寿命分数，单调不减）。"""
    i = np.argmax(hi >= level)
    if hi[i] < level:
        return np.nan
    return float(axis[i])


def mono_tail(hi, frac=0.3):
    """尾段（末 30%）单调不回落的比例 —— 交付口径里用来判"不回落"。"""
    n = len(hi)
    k = max(int(n * frac), 2)
    seg = hi[-k:]
    d = np.diff(seg)
    return float((d >= -1e-9).mean()) if len(d) else np.nan


def _hi_metrics(counts, axis, label):
    """一种累积口径的 t50 / t85 / 尾段单调。"""
    counts = np.asarray(counts, dtype=float)
    tot = np.cumsum(counts)
    if not np.isfinite(tot[-1]) or tot[-1] <= 0 or not np.isfinite(axis).any():
        return {label + '_t50': None, label + '_t85': None, label + '_尾段单调': None}
    hi = tot / tot[-1]
    a = axis / np.nanmax(axis)
    return {label + '_t50': round(t_at(hi, a, 0.50) * 100, 1),
            label + '_t85': round(t_at(hi, a, 0.85) * 100, 1),
            label + '_尾段单调': round(mono_tail(hi), 3)}


def run(gid, min_hits=1, min_frames=MIN_FRAMES):
    """逐级回落，取第一个「有效帧足够」的帧格；都不够则返回 None。"""
    for tag, frame_s in GRIDS:
        d = _run_grid(gid, tag, frame_s, min_hits, min_frames)
        if d is not None:
            return d
    return None


def _run_grid(gid, tag, frame_s, min_hits, min_frames):
    """在指定帧格上算一遍；帧表缺失或有效帧不足则返回 None。"""
    fp = os.path.join(RES, '_l1ae_frames%s_%s.npz' % (tag, gid))
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    n = z['n'].astype(float)
    m = n >= min_hits
    if m.sum() < min_frames:
        return None
    cp = os.path.join(RES, '_l1cyc%s_%s.npz' % (tag, gid))
    if tag and not os.path.exists(cp):
        # 回落到了细格但细格循环轴还没建 —— 横轴会退化成帧序号（不是循环分数），
        # 必须显式喊出来，否则 t85 会被当成循环口径引用。
        print('  ⚠️ %s 回落到 tag=%r 帧格，但缺 %s ⇒ 横轴退化为帧序号；'
              '请先跑 ae_cycle.py --frame-s 60 --tag 60s --groups %s'
              % (gid, tag, os.path.basename(cp), gid))
    if os.path.exists(cp):
        c = np.load(cp, allow_pickle=True)
        idx = {int(f): i for i, f in enumerate(c['frame'])}
        sel = np.asarray([idx.get(int(f), -1) for f in z['frame']])
        cyc = np.full(len(n), np.nan)
        ok = sel >= 0
        cyc[ok] = c['cycle'][sel[ok]]
        axis = cyc
        axis_name = 'cycle'
        f_hz = float(c['f_hz'])
    else:
        axis = z['frame'].astype(float)
        axis_name = 'frame'
        f_hz = np.nan
    hi_hit = np.cumsum(n) / np.cumsum(n)[-1]
    hi_eng = np.cumsum(z['ener_sum']) / np.cumsum(z['ener_sum'])[-1]
    a = axis / np.nanmax(axis)
    d = {'组号': gid, 'n_f': N_F.get(gid), '轴': axis_name, '反解f_Hz': round(f_hz, 3),
         '有效帧': int(m.sum()), '帧长s': int(frame_s)}
    # 三种累积口径并列输出**只作对比**：交付主曲线是 HI_hit（全部 hit）。
    # 通道 2 在「逐帧活动度」口径下最优，但在累积 t85 上极差最大（§7 / §29）。
    for key, lab in (('n', 'HI_hit'), ('n_ch1', 'HI_ch1'), ('n_ch2', 'HI_ch2')):
        if key in z.files:
            d.update(_hi_metrics(z[key], axis, lab))
    d.update(_hi_metrics(z['ener_sum'], axis, 'HI_eng'))
    return d


def _groups_with_frames():
    """有 600 s 帧表（生产口径）的组。

    ⚠️ **不能用前缀切片**：`_l1ae_frames60s_L1-06.npz` 会切出 `60s_L1-06` 这种
    假组名，再交给 run() 就静默返回 None。必须按正则严格匹配。
    """
    pat = re.compile(r'^_l1ae_frames_(L1-\d+)\.npz$')
    out = []
    for p in glob.glob(os.path.join(RES, '_l1ae_frames*_L1-*.npz')):
        m = pat.match(os.path.basename(p))
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    gids = argv or _groups_with_frames()
    rows = []
    print('%-7s %-6s %8s %10s %10s %10s %10s %10s %8s' % (
        '组', '轴', 'n_f', 'HI_hit_t85', 'HI_ch1_t85', 'HI_ch2_t85',
        'HI_eng_t85', 'CH2_t50', '尾段单调'))
    for gid in gids:
        d = run(gid)
        if d is None:
            print('%-7s 两档帧格的有效帧都不足，跳过' % gid)
            continue
        rows.append(d)
        print('%-7s %-6s %8s %10s %10s %10s %10s %10s %8s  %s'
              % (gid, d['轴'], d['n_f'], d.get('HI_hit_t85'), d.get('HI_ch1_t85'),
                 d.get('HI_ch2_t85'), d.get('HI_eng_t85'), d.get('HI_ch2_t50'),
                 d.get('HI_ch2_尾段单调'),
                 '' if d['帧长s'] == 600 else '<- 60 s 细格回落'))
    if rows:
        import csv
        # ⚠️ 同名互覆护栏（2026-10-08）：带组名只跑**子集**时，绝不能覆盖全量交付产物
        # `l1_ae_hi.csv` —— 这个坑在 `docs/待办与未决问题.md` §7 记了很久，实测真的会踩
        # （子集跑一次就把 14 行缩成 3 行）。子集结果改写 `_l1_ae_hi_partial.csv`。
        all_g = sorted(_groups_with_frames())
        subset = sorted(r['组号'] for r in rows)
        full = bool(all_g) and subset == all_g
        out = os.path.join(RES, 'l1_ae_hi.csv' if full
                           else '_l1_ae_hi_partial.csv')
        if not full:
            print('\n⚠️ 本次只跑了 %d / %d 组 ⇒ 结果写 `%s`，**不动**交付产物 `l1_ae_hi.csv`'
                  % (len(rows), len(all_g), os.path.basename(out)))
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in keys})
        print('\n── 四种累积口径的跨组一致性（极差越小 = 跨组越稳）──')
        for lab in ('HI_hit', 'HI_ch1', 'HI_ch2', 'HI_eng'):
            v = [r[lab + '_t85'] for r in rows
                 if r.get(lab + '_t85') is not None]
            if v:
                tag = ('   <= 交付主口径' if lab == 'HI_hit' else
                       '   （已弃用：能量口径不稳）' if lab == 'HI_eng' else '')
                print('  %-8s t85 中位 %5.1f%%   范围 %5.1f 至 %5.1f（极差 %.1f）%s'
                      % (lab, np.median(v), min(v), max(v), max(v) - min(v), tag))
        print('  ⚠️ **不要只挑极差最小的当主口径** —— 必须先固定交付形态（逐帧还是累积），'
              '再据此选指标；两种判据不可互推（§7 / §29）。')
        mono = [r for r in rows if r.get('HI_ch2_尾段单调') is not None]
        if mono:
            nm = sum(1 for r in mono if r['HI_ch2_尾段单调'] >= 0.99)
            print('  通道2 尾段单调不回落：%d / %d 组' % (nm, len(mono)))
        print('已写出 ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
