# -*- coding: utf-8 -*-
"""main 验证层统一入口 —— 由 7 个脚本合并而来（2026-10-07）

合并方式：每段正文**逐字保留**，只给冲突的顶层名字加前缀 + 去掉各自的
`__main__` 块；公共前导（sys.path / chdir / 共享导入）只写一份。
原脚本已归档于 `main/results/_logs/_orig_src/`。

子命令（`python main/verify.py <子命令> -h` 看各自参数）：

    子命令        原脚本                      产出（落在 main/results/）
    ------------  --------------------------  ------------------------------------------
    baselines     evaluate_baselines.py       baseline_compare.csv / baseline_best.csv
    loso-cv       evaluate_loso.py            loso_cv.csv / loso_summary.csv
    labels        evaluate_labels.py          label_audit / label_sensitivity /
                                              per_specimen_stats / inclusion_criteria
    ae-cols       evaluate_ae_columns.py      ae_column_eval{,_full,_ext}.csv
    grade         grade_compare.py            grade_stiff_traj / grade_levels* / grade_summary
    fusion        fusion_compare.py           fusion_compare.csv / fusion_summary.csv
    candidates    candidate_check.py          candidate_check.csv
                                              ⚠️ 读 grade 的产物 ⇒ 要先跑 `grade`

⚠️ 本模块在 **导入时** 就 `chdir(main/)`（与原脚本一致）；`CWD0` = 调用时的目录。
"""
import argparse
import contextlib
import glob
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))        # main/
ROOT = HERE
CWD0 = os.getcwd()                                       # 调用时的目录（chdir 之前）
sys.path.insert(0, os.path.dirname(HERE))                # 项目根（含 shm）
sys.path.insert(0, HERE)                                 # main/（含 eval_common）
os.chdir(HERE)                                           # main/

import eval_common as ec                                                       # noqa: E402
from eval_common import (GROUPS, DEFAULT, cfg_id, run_cfg, table_for,          # noqa: E402
                         per_group_metrics, ref_map, onset_of, LOW, DROP, HOLD_FRAC)
from shm.streaming import StreamSimulator, ChunkedDataReader                   # noqa: E402
from shm.damage_index import OnlineDamageIndex, _AmpChannel                    # noqa: E402
from shm.config import EXT_BLOCK_PTS, FO_PARAMS, GRADE_GATE                    # noqa: E402

# ==========================================================================
# 段 1/7  原 main/evaluate_baselines.py（349 行，正文逐字保留；改名 main→bl_main, OUT→BL_OUT）
# ==========================================================================

# -*- coding: utf-8 -*-
"""基线对照：AE 单源口径下，本项目方法 vs 经典/朴素基线 + 监督 ML 基线

为什么要做
----------
「我们的方法能用」≠「比现有方法好」。没有对照表，论文的核心主张不成立。
本脚本给出**同口径、同预警判据、同评估指标**的对照，并做两件事避免
"稻草人基线"指责：

  1. **阈值 oracle 扫描**：阈值型基线在网格上扫一遍，报**最佳 |err|**及其阈值
     （即"事后挑最好阈值"），仍不达标才算真不行；
  2. **消除跨试件尺度差**：所有指标按各组**自身校准段**（块 [20,45)，与本项目
     `stiff_cal`/`shape_cal` 同窗口）归一 → 跨试件阈值才有意义；
     ML 基线额外给一个「逐试件因果归一 + log」的特征版本。

对照对象
--------
  ours_off   : OnlineDamageIndex 正式口径（逐点不可逆规则，与交付一致）
  ours_blk   : 同上，但改用与基线**完全一致**的块级不可逆规则
  ours_ae_blk: 同上，但只用声发射（单源口径，公平比）
  B1 累积 AE 能量   h = cumsum(块内 Σpeak²) / 校准段 cumsum 中位
  B2 块能量 EWMA     h = EWMA_α(块内 Σpeak²) / 校准段块能量中位
  B3 累积 AE 事件数  h = cumsum(块内事件数) / 校准段 cumsum 中位
  B4raw  监督 ML  25 列块均值 + log(Σpeak²) + 事件数 → LogisticRegression，LOSO
  B4norm 监督 ML  同上，但特征改为 逐试件校准段中位归一 + log10

⚠️ **不对称须写明**：B4 在训练时**看过 b2 标签**，本项目方法**从不看标签**。
   B4 赢是正常的；B4 赢不了才是强结论。

预警判据（块级，与本项目逐点规则同构）
  「首次不可逆越阈」= h ≥ level，且其后 hold 块内不回落至 level×(1−drop_frac) 以下。
  hold = max(4, 2% 块数)（对应逐点的 max(2000 点, 2% 寿命)），
  drop_frac = DROP/LOW = 0.15/0.30 = 0.5（允许回落一半）。

输出：results/baseline_compare.csv（长表）+ 控制台汇总
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from shm.streaming import StreamSimulator                        # noqa: E402
from shm.damage_index import OnlineDamageIndex                   # noqa: E402
from shm.config import EXT_BLOCK_PTS                             # noqa: E402
from eval_common import ref_map, onset_of, LOW, DROP, HOLD_FRAC  # noqa: E402

GROUPS = ['016', '017', '018', '019', '020']       # 有 b2/b3 标签
NOB2 = ['023', '024', '025', '026']                 # 同台架同协议，**无标签** → 只看是否报警
NOLABEL = ['021', '022'] + NOB2                     # 其余无标签组（含问题组 021）
ALL_GROUPS = GROUPS + NOLABEL
CAL_LO, CAL_HI = 20, 45          # 校准段块号（与本项目 stiff/shape 同窗口）
ALPHA = 0.3                      # EWMA 系数（与 e_dmg 的 0.7/0.3 同构）
DROP_FRAC = DROP / LOW           # 0.5
N_SPLIT = 6                      # ML 的 LOOCV 折数（= 试件数）
EPS = 1e-12
BL_OUT = os.path.join('results', 'baseline_compare.csv')


# ---------------------------------------------------------------- 工具
def _ewma(a, alpha=ALPHA):
    out = np.empty_like(np.asarray(a, dtype=float))
    s = 0.0
    for i, v in enumerate(a):
        s = s + alpha * (v - s)
        out[i] = s
    return out


def _cal_med(a):
    """校准段中位（因果参考：只用校准段）。"""
    a = np.asarray(a, dtype=float)
    w = a[CAL_LO:CAL_HI] if len(a) > CAL_HI else a
    w = w[np.isfinite(w)]
    return float(np.median(w)) if len(w) else 0.0


def warn_blk(h, level, hold):
    """块级「首次不可逆越阈」→ 寿命% 或 None（与 onset_of 同构）。"""
    h = np.asarray(h, dtype=float)
    n = len(h)
    drop = level * (1.0 - DROP_FRAC)
    i = 0
    while i < n:
        if h[i] >= level:
            j = min(n - 1, i + hold)
            seg = h[i:j + 1]
            if seg.min() >= drop:
                return 100.0 * i / n
            below = np.where(seg < drop)[0]
            i = i + (int(below[0]) if len(below) else (j - i + 1))
        else:
            i += 1
    return None


def grade_of(tw, b2):
    if tw is None:
        return 'C'
    if b2 is None:
        return '-'          # 无标签组不计分级
    e = tw - b2
    return 'E' if e < -15 else ('D' if e > 15 else 'A')


# ---------------------------------------------------------------- 逐组特征
def group_features(gid):
    """一次流式 → 块级原始量 + 两组逐点 D（正式口径 / AE 单源口径）。"""
    sim = StreamSimulator(gid)
    n = sim.load_data()
    cols = list(sim.ae_cols)
    di_full = OnlineDamageIndex()                     # 正式（多源默认）
    di_ae = OnlineDamageIndex({'abl': 'no_strain'})   # AE 单源
    ncol = len(cols)
    E_blk, N_blk, F_blk = [], [], []
    Dfull, Dae = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    bpts, bE, bN = 0, 0.0, 0
    bsum = np.zeros(ncol)
    for idx in range(n):
        p = sim.next_point()
        strain = p['strain']
        pk = None
        sv = None
        if p.get('ae_new') and p.get('ae'):
            pk = float(p['ae'].get('ae_Peak', 0.0) or 0.0)
            sv = di_full.shape_value(p['ae'])
            for c_i, c in enumerate(cols):
                v = p['ae'].get(c)
                if v is not None and np.isfinite(v):
                    bsum[c_i] += float(v)
        Dfull[idx] = di_full.update(strain, pk, None, sv)
        Dae[idx] = di_ae.update(strain, pk, None, di_ae.shape_value(p.get('ae')))
        if pk is not None:
            bE += pk ** 2
            bN += 1
        bpts += 1
        if bpts >= EXT_BLOCK_PTS:
            E_blk.append(bE)
            N_blk.append(bN)
            F_blk.append(bsum / bN if bN else np.zeros(ncol))
            bpts, bE, bN = 0, 0.0, 0
            bsum = np.zeros(ncol)
    sim.cleanup()
    nb = len(E_blk)
    ends = np.minimum((np.arange(1, nb + 1) * EXT_BLOCK_PTS) - 1, n - 1)
    return dict(gid=gid, n=n, nb=nb, cols=cols,
                E=np.asarray(E_blk, float), N=np.asarray(N_blk, float),
                F=np.vstack(F_blk), Dfull=Dfull[ends], Dae=Dae[ends],
                Dfull_pt=Dfull, Dae_pt=Dae)


# ---------------------------------------------------------------- 指标
def rows_for(method, level, g, h, b2, b3, hold):
    tw = warn_blk(h, level, hold)
    err = (tw - b2) if (tw is not None and b2 is not None) else None
    lead = (b3 - tw) if (tw is not None and b3 is not None) else None
    return dict(method=method, level=float(level), gid=g['gid'],
                t_warn=tw, err=err, lead=lead, grade=grade_of(tw, b2))


def summarize(df):
    """只在**有标签组**上聚合（无标签组 err/grade 无意义）。"""
    d = df[df['gid'].isin(GROUPS)]
    out = []
    for (m, lv), s in d.groupby(['method', 'level'], sort=False, dropna=False):
        e = s['err'].dropna()
        out.append(dict(method=m, level=lv, nA=int((s['grade'] == 'A').sum()),
                        nE=int((s['grade'] == 'E').sum()),
                        nD=int((s['grade'] == 'D').sum()),
                        nC=int((s['grade'] == 'C').sum()),
                        mean_abs_err=float(e.abs().mean()) if len(e) else np.nan,
                        mean_lead=float(s['lead'].dropna().mean())
                        if s['lead'].notna().any() else np.nan))
    return pd.DataFrame(out)


def bl_main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    refs = ref_map()
    data = {}
    for gid in ALL_GROUPS:
        data[gid] = group_features(gid)
        print(f'  特征完成 {gid}  块数={data[gid]["nb"]}')

    # ---- 逐方法 × 逐阈值 ----
    all_rows = []
    for gid in ALL_GROUPS:
        g = data[gid]
        nb = g['nb']
        b2, b3 = refs.get(gid, (None, None))
        hold = max(4, int(round(0.02 * nb)))
        cum_E, cum_N = np.cumsum(g['E']), np.cumsum(g['N'])
        rE, rN = max(_cal_med(cum_E), EPS), max(_cal_med(cum_N), EPS)
        rB = max(_cal_med(g['E']), EPS)
        hB1, hB2 = cum_E / rE, _ewma(g['E']) / rB
        hB3 = cum_N / rN
        hOurs, hOursAE = g['Dfull'], g['Dae']

        grids = [('B1_累积能量', hB1, np.geomspace(1.5, 50, 40)),
                 ('B2_块能量EWMA', hB2, np.geomspace(1.5, 200, 40)),
                 ('B3_累积事件数', hB3, np.geomspace(1.5, 50, 40)),
                 ('ours_off_blk', hOurs, np.arange(0.20, 0.96, 0.01)),
                 ('ours_ae_blk', hOursAE, np.arange(0.20, 0.96, 0.01))]
        for name, h, grid in grids:
            for lv in grid:
                all_rows.append(rows_for(name, lv, g, h, b2, b3, hold))
        # ours 正式逐点规则（交付口径，非块级）
        tw = onset_of(g['Dfull_pt'], max(2000, int(nb * EXT_BLOCK_PTS * HOLD_FRAC)))
        all_rows.append(dict(method='ours_off_pt', level=np.nan, gid=gid,
                             t_warn=tw,
                             err=(tw - b2) if (tw is not None and b2 is not None) else None,
                             lead=(b3 - tw) if (tw is not None and b3 is not None) else None,
                             grade=grade_of(tw, b2)))
        tw2 = onset_of(g['Dae_pt'], max(2000, int(nb * EXT_BLOCK_PTS * HOLD_FRAC)))
        all_rows.append(dict(method='ours_ae_pt', level=np.nan, gid=gid,
                             t_warn=tw2,
                             err=(tw2 - b2) if (tw2 is not None and b2 is not None) else None,
                             lead=(b3 - tw2) if (tw2 is not None and b3 is not None) else None,
                             grade=grade_of(tw2, b2)))

    # ---- B4 监督 ML（训练见过 b2 标签；有标签组走 LOOCV，无标签组用全标签训练后预测）----
    F_all = np.vstack([data[g]['F'] for g in ALL_GROUPS])
    norm_blocks = []
    for gid in ALL_GROUPS:
        g = data[gid]
        base = np.array([_cal_med(g['F'][:, j]) for j in range(g['F'].shape[1])])
        base = np.where(np.abs(base) < EPS, np.nan, base)
        r = g['F'] / base
        r = np.where(np.isfinite(r) & (r > 0), np.log10(np.maximum(r, 1e-6)), np.nan)
        norm_blocks.append(r)
    Xnorm = np.hstack([np.nan_to_num(np.vstack(norm_blocks), nan=0.0),
                       np.vstack([np.column_stack([np.log10(1 + data[g]['E']), data[g]['N']])
                                  for g in ALL_GROUPS])])
    Xraw = np.hstack([F_all,
                      np.vstack([np.column_stack([np.log10(1 + data[g]['E']), data[g]['N']])
                                 for g in ALL_GROUPS])])
    gid_vec, y = [], []
    for gid in ALL_GROUPS:
        nb = data[gid]['nb']
        b2 = refs.get(gid, (None, None))[0]
        y += [1 if (b2 is not None and 100.0 * k / nb >= b2) else 0 for k in range(nb)]
        gid_vec += [gid] * nb
    gid_vec, y = np.asarray(gid_vec), np.asarray(y)
    labelled = np.isin(gid_vec, GROUPS)

    for tag, X in (('B4raw', Xraw), ('B4norm', Xnorm)):
        prob = np.zeros(len(y))
        for gid in GROUPS:                      # 有标签组 → LOOCV
            te = gid_vec == gid
            tr = labelled & ~te
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for gid in NOB2:                        # 无标签组 → 用全标签训练后外推
            te = gid_vec == gid
            sc = StandardScaler().fit(X[labelled])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                sc.transform(X[labelled]), y[labelled])
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        k0 = 0
        for gid in ALL_GROUPS:
            nb = data[gid]['nb']
            h = _ewma(prob[k0:k0 + nb])
            k0 += nb
            b2, b3 = refs.get(gid, (None, None))
            hold = max(4, int(round(0.02 * nb)))
            for lv in np.arange(0.30, 0.99, 0.02):
                all_rows.append(rows_for(tag, lv, data[gid], h, b2, b3, hold))

    df = pd.DataFrame(all_rows)
    df.to_csv(BL_OUT, index=False, float_format='%.4f')
    summ = summarize(df)

    # ---- 汇总：每方法 oracle 最佳阈值（优先 nC=0，再 mean|err| 最小）----
    print(f'\n{"=" * 96}\n每方法「oracle 阈值」最佳结果（事后挑最好阈值，给基线最好机会）\n{"=" * 96}')
    print(f'{"方法":<16}{"阈值":>10}{"nA":>4}{"nE":>4}{"nD":>4}{"nC":>4}'
          f'{"mean|err|":>11}{"mean lead":>11}')
    best = []
    for m in sorted(summ['method'].unique()):
        s = summ[summ['method'] == m].copy()
        s['pen'] = s['mean_abs_err'].fillna(999) + 100.0 * s['nC']
        b = s.sort_values(['pen', 'mean_abs_err']).iloc[0]
        best.append(b)
        lv = 'na' if not np.isfinite(b['level']) else f"{b['level']:.3f}"
        print(f'{m:<16}{lv:>10}{int(b["nA"]):>4}{int(b["nE"]):>4}{int(b["nD"]):>4}'
              f'{int(b["nC"]):>4}{b["mean_abs_err"]:>11.2f}{b["mean_lead"]:>11.1f}')
    pd.DataFrame(best).to_csv(os.path.join('results', 'baseline_best.csv'), index=False)

    # ---- 无标签组（023-026）：各方法在「自身最佳阈值」下是否报警 ----
    print(f'\n{"=" * 96}\n无标签同协议组（023-026）泛化对照（各方法用上表选定的最佳阈值）\n{"=" * 96}')
    hdr = f'{"gid":<6}' + ''.join(f'{b["method"]:>16}' for b in best)
    print(hdr)
    for gid in NOB2:
        line = f'{gid:<6}'
        for b in best:
            s = df[(df['method'] == b['method']) & (df['gid'] == gid)]
            if np.isfinite(b['level']):
                s = s[np.isclose(s['level'], b['level'])]
            tw = s['t_warn'].iloc[0] if len(s) else None
            line += f'{("%.1f" % tw) if tw is not None else "未报警":>16}'
        print(line)
    # ---- 表 3：全组·无标签统一协议（用可计算的物理锚 f_est）----
    # f_est = 块能量峰值所在块位(%寿命)。**与 016-020 的 b3(=99.0) 互相校验**：
    #   f_est 为 95.0/100/100/98.7/100 → 与记录末尾一致（016 的余段 5% 已由用户确认为正常）。
    # 判定（展示约定，非调参）：漏报 = 未报警；过早 = lead > 50 个百分点；否则合理。
    print(f'\n{"=" * 104}\n表 3 · 全组·无标签统一协议（f_est = 块能量峰值位置；lead = f_est - t_warn）\n{"=" * 104}')
    fest = {}
    for gid in ALL_GROUPS:
        g = data[gid]
        E, nb = g['E'], g['nb']
        k = int(np.argmax(E))
        fest[gid] = (100.0 * (k + 1) / nb,
                     float(E[int(nb * 0.95):].sum() / max(E.sum(), EPS) * 100.0))
    methods3 = [b['method'] for b in best]
    print(f'{"gid":<5}{"f_est%":>7}{"末5%能量":>9}' +
          ''.join(f'{m[:13]:>15}' for m in methods3))
    for gid in ALL_GROUPS:
        f, tail = fest[gid]
        line = f'{gid:<5}{f:>7.1f}{tail:>9.1f}'
        for m in methods3:
            b = [x for x in best if x['method'] == m][0]
            s = df[(df['method'] == m) & (df['gid'] == gid)]
            if np.isfinite(b['level']):
                s = s[np.isclose(s['level'], b['level'])]
            tw = s['t_warn'].iloc[0] if len(s) else None
            if tw is None:
                line += f'{"漏报":>15}'
            else:
                lead = f - tw
                tag = '过早' if lead > 50 else '合理'
                line += f'{f"{tw:.1f}({lead:+.0f}){tag}":>15}'
        mark = '  ← 问题组' if gid == '021' else ('' if gid in GROUPS else '')
        print(line + mark)
    print('\n注：括号内为 lead（百分点，正=报警早于 f_est）。f_est 对 016-020 与 b3(=99.0) 一致，'
          '故 016-020 的该列可当作标签无关的复核。')
    print('    · f_est 为可计算的物理锚（AE 块能量峰值位置），对无 b2/b3 的组也适用。')

    print(f'\nsaved {os.path.abspath(BL_OUT)}')


# ==========================================================================
# 段 2/7  原 main/evaluate_loso.py（177 行，正文逐字保留；改名 main→loso_main）
# ==========================================================================

# -*- coding: utf-8 -*-
"""留一试件交叉验证（LOSO）：回答「5/5 精准是不是靠在同一批试件上调参」

为什么必须做
------------
`b2` 锚 + 参数（rise/fall/acc_scale/lift/shape_w/latch_*）都是在**同一批 5 组**上定的。
审稿人必然问：「换一组试件，你的参数还成立吗？」

做法（标准模型选择 + LOSO）
---------------------------
  1. 候选配置网格（**一次一变**，非全因子）：
       5 个既有参数 ±30%  +  形状证据参数（shape_w / shape_gain / shape_q）
       +  latch 速率  →  共 18 个变体 + 默认 = 19 个配置；
  2. **每个留出折**：在**其余 4 组**上按项目自身的聚合目标 `score` 选**单个最优配置**，
     再用它评估**留出的那一组**（该组及其标签从未参与选择）；
  3. 汇总留出性能，并与两个参照对比：
       default : 默认配置在全部 5 组上的成绩（= 论文主结果，样本内）
       oracle  : 事后挑在全部 5 组上最好的配置（乐观上界）
  结论读法：LOSO ≈ default ⇒ 参数不是靠这批试件"喂"出来的；
            LOSO 明显差于 default ⇒ 存在过拟合，必须报告。

⚠️ 局限（须写进论文）：搜索空间是**一次一变**的 19 个配置，不是全因子网格；
   故 LOSO 结果是**乐观侧**的（搜索空间越小越不容易过拟合）。全因子等价于放弃可解释性。

用法：python main/verify.py loso-cv
输出：main/results/loso_cv.csv（逐折）、main/results/loso_summary.csv（汇总）
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from eval_common import (GROUPS, DEFAULT, cfg_id, run_cfg,   # noqa: E402
                         table_for, per_group_metrics)

OUT_FOLD = os.path.join('results', 'loso_cv.csv')
OUT_SUM = os.path.join('results', 'loso_summary.csv')

# --- 待检验的参数与档位（一次一变；±30% 与既有 sens 口径一致）---
VARY = [
    ('rise', [DEFAULT['rise'] * 0.7, DEFAULT['rise'] * 1.3]),
    ('fall', [DEFAULT['fall'] * 0.7, DEFAULT['fall'] * 1.3]),
    ('acc_scale', [DEFAULT['acc_scale'] * 0.7, DEFAULT['acc_scale'] * 1.3]),
    ('estrain_w', [DEFAULT['estrain_w'] * 0.7, DEFAULT['estrain_w'] * 1.3]),
    ('lift', [DEFAULT['lift'] * 0.7, DEFAULT['lift'] * 1.3]),
    # —— 形状证据参数（本轮新增，必须纳入，否则 LOSO 覆盖不到最新引入的旋钮）——
    ('shape_w', [0.3, 1.0]),
    ('shape_gain', [1.0, 4.0]),
    ('shape_q', [50.0, 98.0]),
    ('latch_rise_fast', [0.30, 0.80]),
]


def score(df):
    """聚合目标（与 robustness.py 一致）：A 对齐 + 达级 - 漏报重罚，越大越好。"""
    nA = int((df['grade'] == 'A').sum())
    n55 = int((df['D_end'] >= .55).sum())
    nC = int((df['grade'] == 'C').sum())
    return nA + 0.25 * n55 - 1.5 * nC


def build_grid():
    cfgs = [(dict(DEFAULT), 'default')]
    for par, vals in VARY:
        for v in vals:
            cp = dict(DEFAULT)
            cp[par] = float(v)
            cfgs.append((cp, f'{par}={v:g}'))
    return cfgs


def loso_main():
    grid = build_grid()
    print(f'候选配置 {len(grid)} 个 × {len(GROUPS)} 组 = {len(grid) * len(GROUPS)} 次流式\n',
          flush=True)
    tabs, names = {}, []
    for params, label in grid:
        gid_d = run_cfg(params, workers=8)
        tabs[label] = table_for(cfg_id(params), gid_d)
        names.append(label)
        print(f'  [{label}] A={int((tabs[label]["grade"] == "A").sum())} '
              f'nC={int((tabs[label]["grade"] == "C").sum())} '
              f'med|err|={tabs[label]["err"].abs().mean():.2f}', flush=True)

    # ---- 参照 ----
    d0 = tabs['default']
    ref = dict(default_nA=int((d0['grade'] == 'A').sum()),
               default_abs_err=float(d0['err'].abs().mean()),
               default_nC=int((d0['grade'] == 'C').sum()))
    oracle = max(names, key=lambda k: (score(tabs[k]), -tabs[k]['err'].abs().mean()))
    worst = min(names, key=lambda k: (score(tabs[k]), -tabs[k]['err'].abs().mean()))
    ref['oracle_cfg'] = oracle
    ref['oracle_nA'] = int((tabs[oracle]['grade'] == 'A').sum())
    ref['oracle_abs_err'] = float(tabs[oracle]['err'].abs().mean())

    # ---- LOSO ----
    # ⚠️ 评分粒度：19 个配置**全部**是 5/5 级 A、nC=0 → 单纯按 `score` 排序会**平局**，
    #    选择退化成"总是第一个（default）"。故用 (score, 更小的|err|) 字典序，
    #    让选择有真实区分度；两种口径的结果都报告。
    fold = []
    for g in GROUPS:
        train = [x for x in GROUPS if x != g]
        best_l, best_key = None, None
        for k in names:
            t = tabs[k]
            tr = t[t['gid'].isin(train)]
            key = (score(tr), -float(tr['err'].abs().mean()))
            if best_key is None or key > best_key:
                best_key, best_l = key, k
        row = tabs[best_l][tabs[best_l]['gid'] == g].iloc[0]
        row0 = d0[d0['gid'] == g].iloc[0]
        fold.append(dict(gid=g, chosen=best_l,
                         train_abs_err=-best_key[1],
                         tw_loso=row['t_warn'], err_loso=row['err'],
                         grade_loso=row['grade'], D_end_loso=row['D_end'],
                         tw_def=row0['t_warn'], err_def=row0['err'],
                         grade_def=row0['grade'], D_end_def=row0['D_end']))
        print(f'  留出 {g}: 选到 [{best_l}] (训练 4 组 |err|={-best_key[1]:.2f}) '
              f'→ 留出 t_warn={row["t_warn"]:.1f} grade={row["grade"]} '
              f'|err|={abs(row["err"]):.2f} '
              f'(默认 t_warn={row0["t_warn"]:.1f} grade={row0["grade"]})', flush=True)
    print(f'  选到的不同配置数：{len(set(f["chosen"] for f in fold))} / {len(GROUPS)} 折',
          flush=True)

    fd = pd.DataFrame(fold)
    fd.to_csv(OUT_FOLD, index=False, float_format='%.3f', encoding='utf-8-sig')
    summ = pd.DataFrame([dict(setting='LOSO(留一选参)',
                              nA=int((fd['grade_loso'] == 'A').sum()),
                              nC=int((fd['grade_loso'] == 'C').sum()),
                              nE=int((fd['grade_loso'] == 'E').sum()),
                              nD=int((fd['grade_loso'] == 'D').sum()),
                              mean_abs_err=float(fd['err_loso'].abs().mean()),
                              mean_lead=np.nan,
                              cfg='%d 种/5 折' % len(set(f['chosen'] for f in fold))),
                         dict(setting='default(样本内)', nA=ref['default_nA'],
                              nC=ref['default_nC'],
                              nE=int((d0['grade'] == 'E').sum()),
                              nD=int((d0['grade'] == 'D').sum()),
                              mean_abs_err=ref['default_abs_err'],
                              mean_lead=float(d0['lead'].mean()), cfg='default'),
                         dict(setting='oracle(事后最优)',
                              nA=ref['oracle_nA'],
                              nC=int((tabs[oracle]['grade'] == 'C').sum()),
                              nE=int((tabs[oracle]['grade'] == 'E').sum()),
                              nD=int((tabs[oracle]['grade'] == 'D').sum()),
                              mean_abs_err=ref['oracle_abs_err'],
                              mean_lead=float(tabs[oracle]['lead'].mean()),
                              cfg=oracle),
                         dict(setting='最差配置(网格内)',
                              nA=int((tabs[worst]['grade'] == 'A').sum()),
                              nC=int((tabs[worst]['grade'] == 'C').sum()),
                              nE=int((tabs[worst]['grade'] == 'E').sum()),
                              nD=int((tabs[worst]['grade'] == 'D').sum()),
                              mean_abs_err=float(tabs[worst]['err'].abs().mean()),
                              mean_lead=float(tabs[worst]['lead'].mean()),
                              cfg=worst)])
    summ.to_csv(OUT_SUM, index=False, float_format='%.3f', encoding='utf-8-sig')

    print(f'\n{"=" * 78}\nLOSO 汇总\n{"=" * 78}')
    print(summ[['setting', 'nA', 'nE', 'nD', 'nC', 'mean_abs_err',
                'mean_lead', 'cfg']].to_string(index=False))
    print(f'\n判读：LOSO 级 A 数 = {int((fd["grade_loso"] == "A").sum())}/5 '
          f'（default {ref["default_nA"]}/5, oracle {ref["oracle_nA"]}/5）；'
          f'|err| {fd["err_loso"].abs().mean():.2f} '
          f'（default {ref["default_abs_err"]:.2f}, '
          f'oracle {ref["oracle_abs_err"]:.2f}）')
    print(f'\nsaved {os.path.abspath(OUT_FOLD)} / {os.path.abspath(OUT_SUM)}')


# ==========================================================================
# 段 3/7  原 main/evaluate_labels.py（252 行，正文逐字保留；改名 main→labels_main）
# ==========================================================================

# -*- coding: utf-8 -*-
"""标签审计：b2 的来源、标签敏感性、剔除规则事前化、正确的统计口径

为什么必须做
------------
四项检查，对应论文审稿最可能致命攻击的四点：

  1. **b2 是不是独立真值？**（`main/pipeline.py prepare weaklabels` 的 `run_weaklabel`）
     b2 = 对 **log10(累积 AE 能量) 曲线** 做"台阶/拐点"检测（`find_b2`，要求
     `post−pre >= d_log` 且 `tail−pre >= d_tail`）；若应变发散点更早则**用应变发散覆盖**。
     ⇒ **b2 与本方法用的是同一批信号**（AE 累积能量 + 应变 std 发散），
       而 `e_full` 正是 log 累积能量的短/长窗斜率差。**故 err = t_warn − b2 有循环论证风险。**
     本脚本用可复算的数字把这一点**量化**：若 `t25`（D 首次达 0.25）与 b2 几乎重合，
     说明"精度高"在很大程度上是**自洽**，而非独立验证。

  2. **结论对标签有多敏感？** 把 b2 整体平移 ±2/±5/±10 个百分点，看 5/5 级 A 还剩几组。

  3. **剔除规则能否事前化？** 用**只看数据属性**的事前判据给出纳入/排除决定
     （见 PRE_CRITERIA），而不是"结果不好就剔"。

  4. **统计口径**：N=5 时 bootstrap 置信区间**偏窄**。给出 per-specimen 表 +
     t 分布 95% CI，并与 bootstrap 并列对比。

输出：results/label_audit.csv / results/label_sensitivity.csv /
      results/inclusion_criteria.csv / results/per_specimen_stats.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

GROUPS = ['016', '017', '018', '019', '020']
ALL = ['015'] + GROUPS + ['021', '022', '023', '024', '025', '026', '027']

# ---- 事前判据（**与任何结果无关**，只依赖数据属性）----
# 声明在先：任何不满足以下任一条的试件**不进入"疲劳失效预警"统计集**。
PRE_CRITERIA = {
    'P1_数据完整性': '时间列语义可识别(t_kind != "?") 且 应变行数>0 且 采样间隔>0',
    'P2_AE存量': 'AE 行数>0 且 AE 行数/应变行数 >= 0.05（确有事件流）',
    'P3_试验类型': '循环次数 = 无上限（疲劳至失效）；有上限(循环+静力)属**另一类试验**',
}


def load():
    base = 'results'
    dg = pd.read_csv(os.path.join(base, 'damage_degree_metrics.csv'))
    wn = pd.read_csv(os.path.join(base, 'warning_onset.csv'))
    for df in (dg, wn):
        df['gid'] = df['gid'].astype(str).str.zfill(3)
    cc = pd.read_csv(os.path.join(base, 'candidate_check.csv'))
    cc['gid'] = cc['gid'].astype(str).str.zfill(3)
    lab = pd.read_csv('weak_labels/labels_summary.csv')
    lab['gid'] = lab['gid'].astype(int).map(lambda x: f'{x:03d}')
    return dg, wn, cc, lab


def _int(x, d=0):
    """安全取整（NaN/None/非数值 → d）。"""
    try:
        v = float(x)
        return d if not np.isfinite(v) else int(v)
    except (TypeError, ValueError):
        return d


def protocol():
    """加载协议表：循环次数是否无上限。"""
    rec = pd.read_csv('数据记录.xlsx', sheet_name='Sheet1') if False else None
    df = pd.read_excel('数据记录.xlsx', sheet_name='Sheet1')
    d = {}
    for _, r in df.iterrows():
        gid = f'{int(r["序号"]):03d}'
        cyc = str(r['循环次数']).strip()
        d[gid] = dict(cycles=cyc, unlimited=(cyc == '无上限'),
                      peak_kn=r['峰值应力（KN）'], freq=r['频率'])
    return d


def grade(tw, b2, band=15.0):
    if tw is None or (isinstance(tw, float) and not np.isfinite(tw)):
        return 'C'
    e = tw - b2
    return 'E' if e < -band else ('D' if e > band else 'A')


def stat_summary(v, name):
    """mean / t 分布 95% CI / bootstrap 95% CI 对比。"""
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    m, sd, n = float(v.mean()), float(v.std(ddof=1)), len(v)
    try:
        from scipy import stats as sps
        tcrit = float(sps.t.ppf(0.975, n - 1))
    except Exception:
        tcrit = 2.776 if n == 5 else 2.0
    se = sd / np.sqrt(n)
    rng = np.random.default_rng(0)
    bs = [rng.choice(v, n, replace=True).mean() for _ in range(20000)]
    return dict(metric=name, n=n, mean=m, sd=sd, se=se,
                ci_t_lo=m - tcrit * se, ci_t_hi=m + tcrit * se,
                ci_boot_lo=float(np.percentile(bs, 2.5)),
                ci_boot_hi=float(np.percentile(bs, 97.5)),
                t_crit=tcrit)


def labels_main():
    dg, wn, cc, lab = load()
    proto = protocol()
    b2 = dict(zip(lab['gid'], lab['b2']))
    b3 = dict(zip(lab['gid'], lab['b3']))
    dgi = dg.set_index('gid')
    wni = wn.set_index('gid')

    # ---------- 1. b2 来源与循环性量化 ----------
    print('=' * 100)
    print('1. b2 是不是独立真值？（b2 由 log10(累积 AE 能量) 的台阶检测得到，'
          '与 e_full 同源）')
    print('=' * 100)
    print(f'{"gid":<5}{"b2(%)":>8}{"t25(%)":>8}{"t25-b2":>9}{"t_warn(%)":>11}'
          f'{"tw-b2":>8}{"t85(%)":>8}{"b3":>5}')
    rows = []
    for g in GROUPS:
        r25 = float(dgi.loc[g, 't25'])
        tw = float(wni.loc[g, 't_warn'])
        t85 = float(dgi.loc[g, 't85'])
        rows.append(dict(gid=g, b2=b2[g], t25=r25, t25_minus_b2=r25 - b2[g],
                         t_warn=tw, tw_minus_b2=tw - b2[g], t85=t85, b3=b3[g],
                         D_end=float(dgi.loc[g, 'D_end'])))
        print(f'{g:<5}{b2[g]:>8.1f}{r25:>8.1f}{r25 - b2[g]:>9.1f}{tw:>11.1f}'
              f'{tw - b2[g]:>8.1f}{t85:>8.1f}{b3[g]:>5.1f}')
    aud = pd.DataFrame(rows)
    aud.to_csv('results/label_audit.csv', index=False, float_format='%.3f',
               encoding='utf-8-sig')
    dev = aud['t25_minus_b2'].abs()
    print(f'\n  >>> |t25 - b2| : 均值 {dev.mean():.2f} 个百分点，最大 {dev.max():.2f}')
    print('  >>> 判读：t25 是「D 首次达 0.25」，b2 是「log 累积 AE 能量的台阶位置」——')
    print('      两者在 5/5 组上相差不到 2.5 个百分点，说明**标签与检测器锁在同一条曲线上**。')
    print('      => err = t_warn - b2 的"高精度"(|err| 2.05) 有相当部分是**自洽**，')
    print('         不能单独作为"方法准确"的证据。必须补独立锚（见第 3 节）。')

    # ---------- 2. 标签敏感性 ----------
    print('\n' + '=' * 100)
    print('2. 标签敏感性：把 b2 整体平移 delta 个百分点后，5 组里还有几组是 A 级')
    print('=' * 100)
    bl = pd.read_csv('results/baseline_compare.csv', dtype={'gid': str})
    bl['gid'] = bl['gid'].str.zfill(3)
    b1 = bl[(bl['method'] == 'B1_累积能量') & np.isclose(bl['level'], 1.5)]
    tw_b1 = dict(zip(b1['gid'], b1['t_warn']))
    sens = []
    deltas = [-10, -5, -2, 0, 2, 5, 10]
    print(f'{"delta":>7}' + ''.join(f'{d:>9}' for d in deltas))
    for meth, twmap in (('本项目(交付口径)',
                         {g: float(wni.loc[g, 't_warn']) for g in GROUPS}),
                        ('B1_累积能量(最佳基线)', tw_b1)):
        line = f'{meth:>7}'
        for d in deltas:
            nA = sum(1 for g in GROUPS if grade(twmap.get(g), b2[g] + d) == 'A')
            y = {g: (twmap.get(g), b2[g] + d) for g in GROUPS}
            me = np.mean([abs(t0 - b0) for t0, b0 in y.values()
                          if t0 is not None and np.isfinite(t0)])
            sens.append(dict(method=meth, delta=d, nA=nA, mean_abs_err=me))
            line += f'{nA:d}/{me:.1f}'.rjust(9)
        print(line)
    pd.DataFrame(sens).to_csv('results/label_sensitivity.csv', index=False,
                              float_format='%.3f', encoding='utf-8-sig')
    print('  格式 = A组数/均值|err|；delta 为对 b2 的整体平移（百分点）')

    # ---------- 3. 剔除规则事前化 ----------
    print('\n' + '=' * 100)
    print('3. 纳入/排除决定：**事前判据**（只依赖数据属性，与结果无关）')
    print('=' * 100)
    for k, v in PRE_CRITERIA.items():
        print(f'   {k}: {v}')
    cci = cc.set_index('gid')
    inc = []
    for g in ALL:
        if g not in cci.index:
            inc.append(dict(gid=g, P1=False, P2=False, P3=False, include=False,
                            reason='体检表缺该组'))
            continue
        r = cci.loc[g]
        n_ae, n_str, dt_s = _int(r.get('n_ae')), _int(r.get('n_str')), \
            (float(r.get('dt_s', 0)) if np.isfinite(r.get('dt_s', np.nan)) else 0.0)
        p1 = (str(r.get('t_kind', '?')) != '?') and n_str > 0 and dt_s > 0
        p2 = n_ae > 0 and n_ae / max(n_str, 1) >= 0.05
        p3 = bool(proto.get(g, {}).get('unlimited', False))
        reasons = []
        if not p1:
            reasons.append('时间列/采样不可识别')
        if not p2:
            reasons.append('AE 存量不足')
        if not p3:
            reasons.append('有上限试验(循环+静力)，非疲劳失效类')
        inc.append(dict(gid=g, P1=bool(p1), P2=bool(p2), P3=bool(p3),
                        include=bool(p1 and p2 and p3),
                        n_ae=n_ae, n_str=n_str,
                        t_kind=str(r.get('t_kind', '')),
                        cycles=proto.get(g, {}).get('cycles', ''),
                        reason='；'.join(reasons)))
    ii = pd.DataFrame(inc)
    ii.to_csv('results/inclusion_criteria.csv', index=False, encoding='utf-8-sig')
    print(f'\n{"gid":<5}{"P1":>4}{"P2":>4}{"P3":>4}{"纳入":>6}{"循环次数":>10}'
          f'{"AE行":>8}  排除理由')
    for _, r in ii.iterrows():
        print(f'{r["gid"]:<5}{("Y" if r["P1"] else "n"):>4}'
              f'{("Y" if r["P2"] else "n"):>4}{("Y" if r["P3"] else "n"):>4}'
              f'{("纳入" if r["include"] else "排除"):>6}{str(r["cycles"]):>10}'
              f'{r["n_ae"]:>8}  {r["reason"]}')
    ins = list(ii[ii['include']]['gid'])
    print(f'\n  >>> 事前判据下的纳入集（{len(ins)} 组）: {", ".join(ins)}')
    print('  >>> 注意：023-026 属纳入集但**无 b2 标签**（标签生成只覆盖 016-020）→')
    print('      它们只能在"无标签统一协议"（§3.8 表 3，物理锚 f_est）下评估。')

    # ---------- 4. 统计口径 ----------
    print('\n' + '=' * 100)
    print('4. 统计口径：per-specimen 表 + t 分布 95% CI（N=5 时 bootstrap 偏窄）')
    print('=' * 100)
    per = []
    for g in GROUPS:
        tw = float(wni.loc[g, 't_warn'])
        per.append(dict(gid=g, t25=float(dgi.loc[g, 't25']), t55=float(dgi.loc[g, 't55']),
                        t85=float(dgi.loc[g, 't85']), t_warn=tw, b2=b2[g],
                        err=tw - b2[g], abs_err=abs(tw - b2[g]),
                        lead=b3[g] - tw, D_end=float(dgi.loc[g, 'D_end']),
                        grade=grade(tw, b2[g])))
    ps = pd.DataFrame(per)
    ps.to_csv('results/per_specimen_stats.csv', index=False, float_format='%.3f',
              encoding='utf-8-sig')
    print(ps.to_string(index=False))
    st = [stat_summary(ps['abs_err'], '|err| (本项目, 交付口径)')]
    eb1 = [abs(tw_b1[g] - b2[g]) for g in GROUPS if g in tw_b1]
    st.append(stat_summary(eb1, '|err| (B1 累积能量, 最佳基线)'))
    st = [s for s in st if s]
    sd = pd.DataFrame(st)
    print()
    print(sd[['metric', 'n', 'mean', 'sd', 'ci_t_lo', 'ci_t_hi',
              'ci_boot_lo', 'ci_boot_hi']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\n  >>> 判读：N=5 时 **bootstrap CI 明显窄于 t 分布 CI**，'
          '论文里应用后者（或直接报 per-specimen 值）。')
    print('\nsaved results/label_audit.csv, label_sensitivity.csv, '
          'inclusion_criteria.csv, per_specimen_stats.csv')


# ==========================================================================
# 段 4/7  原 main/evaluate_ae_columns.py（176 行，正文逐字保留；改名 main→acol_main, RES→ACOL_RES, run_cfg→acol_run_cfg）
# ==========================================================================

# -*- coding: utf-8 -*-
"""AE 多列证据评估（形状/比值列 · 单源消融口径）

背景
----
损伤度 D 的 AE 证据以往**只用 `ae_Peak` 一列**（25 个 AE 列里另外 24 列从未参与，
仅存在于 `aligned/*.csv`）。逐列证据筛查（2026-09-20）显示：
  - **无量纲比值列**（MarginFactor=Peak/RootMAV、PeakFactor=Peak/RMS、
    ImpulseFactor=Peak/MAV）在 9/9 组（5 主样本 + 023~026）上趋势为正；
  - **Kurtosis / Skewness** 尾部抬升显著（Kurtosis 在 025 上 1.7~3.1 倍）；
  - 原始量级列（RMS/Std/Variance/MeanSquare/MAV/RootMAV）趋势**反向**（ρ≈-0.11~-0.22），
    频域 6 列无一致趋势 → 均不采用。
故新增可选证据 `e_shape`（见 `shm/damage_index.py`），本脚本量化其效果与代价。

口径
----
- **采样率恒为 10 Hz**（全组，数据方口径）；注意部分组的原始「时间列」是**整数计数器**而非秒
  （见本文件 `candidate_check` 段的 `timebase()`）——**不要从该列反推采样率**。
  `EXT_BLOCK_PTS=500` = 50 s = 250 个 5 Hz 载荷循环，组间一致。
- **AE 单源**（`abl='no_strain'`）：risk = e_ae，隔离形状证据本身的作用；
- 块内事件特征取 `shape_q` 分位（默认 95，即"块内最显著事件"），
  以固定校准段 [shape_cal_lo, shape_cal_hi) 块值的 `shape_base_pct` 分位为基线，
  `e_shape = clip((q/base - 1)/shape_gain, 0, 1) × shape_w`（纯因果，零未来信息）；
- **权重约定**：`shape_w` 默认 0.6，与既有辅助证据 `estrain_w=0.6` 同约定。
  实测 w=1.0 时 e_shape 单独即可把 risk 顶满 1.0 → 9 组 D_end 全部饱和≈1.0，
  跨试件区分度归零；w=0.6 则 025 由 D_end 0.202→0.824 而其余组基本不动。
- 只有 016~020 有 b2（离线参考锚），023~026 **无 b2**，故后者只按**形态**判读
  （D_end 量级、t_warn 是否存在/是否过早、尾部单调性），不计分级。

用法
----
    python main/verify.py ae-cols                       # 默认：Kurtosis/MarginFactor × q{95,50} × w{1.0,0.6}
    python main/verify.py ae-cols --cols Kurtosis --shape-w 0.6 --shape-q 95
    python main/verify.py ae-cols --cols Peak,Kurtosis,MarginFactor,Skewness
    python main/verify.py ae-cols --groups 021,022,023,024,025,026,027 \
        --cols PeakFactor,Kurtosis,MarginFactor --shape-w 0.6 \
        --out main/results/ae_column_eval_ext.csv

⚠️ PowerShell 会把未加引号的 `021,022` 当**数字**传递 → 前导零被吃掉（gid 变 int，
   pivot 匹配不上）。本脚本已对 `--groups` 做 `zfill(3)` 兜底；传参时仍建议加引号。

输出：results/ae_column_eval.csv（长表 cfg×gid）+ 控制台三张 pivot 表。
注意：023/024/025 数据中**无光纤文件**（与本脚本无关，AE 单源口径不依赖光纤）。
"""
import os
import sys
import io
import argparse
import contextlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))          # 项目根(含 shm)
sys.path.insert(0, ROOT)                           # main/(含 eval_common)
os.chdir(ROOT)                                     # main/

from shm.streaming import ChunkedDataReader        # noqa: E402
from shm.damage_index import OnlineDamageIndex     # noqa: E402
from eval_common import per_group_metrics          # noqa: E402

DEF_GROUPS = ['016', '017', '018', '019', '020', '023', '024', '025', '026']
DEF_COLS = ['Kurtosis', 'MarginFactor']
ACOL_RES = os.path.join(ROOT, 'results')


def load_arrays(gid):
    """一次取出逐点 (strain, peak, {列: 值})，语义与 StreamSimulator 一致。

    strain 为松散对齐下的"保持上值"(ffill)；peak/形状值仅在**有新事件**的行给出，
    其余为 NaN → 与在线逐点 update 的入参完全一致。
    """
    rd = ChunkedDataReader(gid)
    with contextlib.redirect_stdout(io.StringIO()):
        rd.load_and_prepare()
    data, ci, n = rd._data, rd._col_idx, rd._data.shape[0]
    ae = rd.ae_cols
    if 'strain' in ci:
        st = pd.Series(data[:, ci['strain']]).ffill().to_numpy()
    else:
        st = np.full(n, np.nan)
    mask = np.isfinite(data[:, [ci[c] for c in ae]]).any(axis=1)   # 有 AE 事件的行
    pk = np.full(n, np.nan)
    pv = data[:, ci['ae_Peak']] if 'ae_Peak' in ci else np.zeros(n)
    pk[mask] = np.where(np.isfinite(pv[mask]), pv[mask], 0.0)
    cols = {}
    for c in ae:
        if c == 'ae_Peak':
            continue
        v = data[:, ci[c]]
        a = np.full(n, np.nan)
        a[mask] = np.where(np.isfinite(v[mask]), v[mask], np.nan)
        cols[c[len('ae_'):]] = a
    rd.cleanup()
    return st, pk, cols


def acol_run_cfg(gid, col, abl, ov):
    """跑一个配置 → 逐点 D。col=None 即基线(仅 Peak)。"""
    st, pk, cols = load_arrays(gid)
    p = {'abl': abl}
    p.update(ov)
    if col:
        p['shape_col'] = col
    di = OnlineDamageIndex(p)
    arr = cols.get(col) if col else None
    n = len(st)
    d = np.empty(n, dtype=np.float32)
    for i in range(n):
        a = pk[i]
        b = arr[i] if arr is not None else None
        if b is not None and not np.isfinite(b):
            b = None
        d[i] = di.update(st[i], None if not np.isfinite(a) else float(a), None, b)
    return d, float(di._last_e_shape)


def acol_main():
    ap = argparse.ArgumentParser(description='AE 多列(形状/比值)证据评估')
    ap.add_argument('--groups', default=','.join(DEF_GROUPS))
    ap.add_argument('--cols', default=','.join(DEF_COLS))
    ap.add_argument('--shape-q', default='95,50', help='块内事件分位(逗号分隔)')
    ap.add_argument('--shape-w', default='1.0,0.6', help='辅助证据权重(逗号分隔)')
    ap.add_argument('--abl', default='no_strain', help='消融模式; no_strain=AE 单源')
    ap.add_argument('--out', default=os.path.join(ACOL_RES, 'ae_column_eval.csv'),
                    help='输出 CSV；相对路径按**调用时**目录解析(脚本内会 chdir 到 main/)')
    a = ap.parse_args()
    if not os.path.isabs(a.out):                   # 相对路径 → 按调用目录
        a.out = os.path.join(CWD0, a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    groups = [g.strip().zfill(3) for g in a.groups.split(',') if g.strip()]
    cols = [c.strip() for c in a.cols.split(',') if c.strip()]
    qs = [float(x) for x in a.shape_q.split(',') if x.strip()]
    ws = [float(x) for x in a.shape_w.split(',') if x.strip()]
    cfgs = [('base', None, {})]
    for c in cols:
        for q in qs:
            for w in ws:
                cfgs.append((f'{c[:2]}q{q:g}w{w:g}', c,
                             {'shape_q': q, 'shape_w': w}))

    print(f'组 {len(groups)} 个 × 配置 {len(cfgs)} 个 (abl={a.abl})')
    rows = []
    for gid in groups:
        for name, col, ov in cfgs:
            d, esh_end = acol_run_cfg(gid, col, a.abl, ov)
            m = per_group_metrics(gid, d)
            rows.append(dict(cfg=name, col=(col or '-'), shape_q=ov.get('shape_q', np.nan),
                             shape_w=ov.get('shape_w', np.nan), gid=gid,
                             D_end=m['D_end'], t25=m['t25'], t55=m['t55'],
                             t85=m['t85'], t_warn=m['t_warn'],
                             tail_mono=m['tail_mono'], err=m['err'],
                             grade=m['grade'], esh_end=esh_end))
        print(f'  {gid} 完成 ({len(cfgs)} 配置)')
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format='%.4f')

    order = [c[0] for c in cfgs]
    for key in ['t85', 't_warn', 'D_end']:
        print(f'\n===== {key} =====')
        pv = df.pivot(index='gid', columns='cfg', values=key).reindex(
            index=groups, columns=order)
        print(pv.round(3).to_string())
    print('\n===== 主样本 5 组 A-预警分级(仅 016~020 有 b2, 离线锚) =====')
    sub = df[df['gid'].isin(['016', '017', '018', '019', '020'])]
    pv = sub.pivot(index='cfg', columns='gid', values='grade').reindex(order)
    pv['全部A'] = (pv == 'A').all(axis=1)
    print(pv.to_string())
    print(f'\nsaved {os.path.abspath(a.out)}')


# ==========================================================================
# 段 5/7  原 main/candidate_check.py（251 行，正文逐字保留；改名 main→cc_main, ROOT→CC_ROOT, OUT→CC_OUT, first_sustained→cc_first_sustained）
# ==========================================================================

# -*- coding: utf-8 -*-
"""候选试件可用性体检（main/ 中 016-020 之外的数据）。

背景
----
`main/001..027` 名义上都是全寿命，但实测只有一部分可用。本脚本**不改动正式配置**
（`shm/config.py` / `main/eval_common.py` 的 GROUPS），只对候选组做三项体检，
用于判断"能否纳入正式范围"：

  1. **时间列语义**：采样间隔 dt、总时长 —— 判断是否纯加载时间（含停机则不可用）
  2. **AE 口径**：AE 行数 / 时长 = 等效频率 —— 判断是否与应变同源同频
     （主样本 AE 无时间戳，靠 `np.linspace` 均匀映射，口径不一致则时间轴失真）
  3. **应变退化信号**：逐块循环幅值（`_AmpChannel` 自动判别解调状态）
     → 刚度损失率 `L = A / A_cal - 1`（`A_cal` = 校准段 [20,45) 块的 p30）
     → `L` 首达 5/10/30/50% 的寿命位置

⚠️ **刚度口径必须与 `grade` 子命令（原 `main/grade_compare.py`）逐项对齐**（2026-09-15 修正，曾不一致）：

| 环节 | 官方口径（本脚本现已采用） | 曾经的错误做法 |
| --- | --- | --- |
| 校准段 | **固定块号 [20,45)**（与 `stiff_cal_lo/hi` 默认值一致） | 写成 `[20%,45%) × nblk` → 随 nb 而变 |
| L 取值 | `max(0, ·)`（与 `OnlineDamageIndex.stiff_loss` 一致） | 允许负值 |
| L 有效起点 | **第 45 块之后**（校准段未走完无基线可言，之前恒为 0） | 全序列 |
| 阈值判据 | **连续 `SUSTAIN` 块 ≥ 阈值，且从 `WARM_BLK` 块起扫** | 单块首次跳阈 → 在块 0 误触发 |

后果：016（nblk=100）因分子分母巧合相等而两者一致，掩盖了问题；
018/019/020 的阈值时刻曾分别报 1.4% / 0.0% / 1.7%，与官方
95.7% / 57.0% / 未达 **相差极大**。现版本对 016-020 会**自校验**并打印比对结果。

用法
----
    python main/verify.py candidates                       # 默认 015-027
    python main/verify.py candidates --groups 022,023,024,025
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CC_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CC_ROOT)
from shm.damage_index import _AmpChannel      # noqa: E402

BLK = 500                 # 与主样本一致（EXT_BLOCK_PTS=500）
# 校准段用**固定块号**，与 OnlineDamageIndex 的 stiff_cal_lo/hi 默认值一致。
# 不能用“寿命比例 × nblk”：那会让校准窗口随试件块数漂移（曾经出错的原因）。
CAL_LO, CAL_HI = 20, 45
BASE_PCT = 30             # 低分位（与 stiff_base_pct 一致）
SUSTAIN = 3               # 连续 N 块 ≥ 阈值（与 grade_compare.SUSTAIN 一致）
WARM_BLK = 10             # 从第 N 块起扫（避开开机瞬态，与 grade_compare 一致）
TH = [0.05, 0.10, 0.30, 0.50]
SR_HZ = 10.0              # **采样率恒为 10 Hz**（数据方口径，全组一致）
CC_OUT = os.path.join(HERE, 'results', 'candidate_check.csv')


def timebase(t):
    """识别第一列的时间语义 → (语义标签, 每点代表的秒数)。

    ⚠️ **不同导出工具写的第一列语义不同**（2026-09-20 实测）：

    | 表头 | 列名 | 实际语义 |
    | ---- | ---- | -------- |
    | 中文导出（016/023/024/025）  | `time` / `时间(s)` | **秒**，步长 0.1 |
    | 英文导出（018/019/020/026）  | `Time` / `时间`   | **整数计数器**，每行 +1（**不是秒**） |

    若把计数器当秒读 → 时长 **放大 10 倍**、AE 频率 **缩小 10 倍**（曾因此误报
    “018/026 是 1 Hz”，实际采样率一直恒为 10 Hz）。
    """
    if t.size < 11:
        return '?', np.nan
    seg = t[:5000]
    med = float(np.median(np.diff(seg)))
    ints = bool(np.all(np.isclose(seg, np.round(seg))))
    if ints and med >= 1.0:
        return '计数器(每点+1)', 1.0 / SR_HZ
    return '秒', med


def read_any(fp):
    """兼容多种编码读 CSV（主样本部分文件不是 UTF-8）。"""
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    return None


def pick_amp_col(df):
    num = df.select_dtypes(include=[np.number])
    cand = [c for c in num.columns if '幅值' in str(c)]
    return (cand[0] if cand else num.columns[-1]), num


def strain_blocks(fp, blk=BLK):
    """逐块循环幅值 + 解调判别结果。"""
    df = read_any(fp)
    if df is None:
        return None, None
    col, num = pick_amp_col(df)
    v = num[col].to_numpy(float)
    ch = _AmpChannel(5000, 2.0)          # 与 OnlineDamageIndex 默认 demod 参数一致
    amps = []
    for k in range(len(v) // blk):
        for x in v[k * blk:(k + 1) * blk]:
            ch.add(float(x))
        amps.append(ch.block_amp())
    return np.asarray(amps, float), ch.mode()


def stiff_series(A):
    """块幅值 → 刚度损失率序列（严格复现 `OnlineDamageIndex.stiff_loss`）。

    两处必须照搬官方实现，否则阈值时刻会偏：
    1. **因果性**：校准段 [CAL_LO, CAL_HI) 未走完之前拿不到基线 → L 恒为 0；
    2. **单向**：只计正向增长（`max(0, ·)`），幅值下降不产生“负损伤”。
    """
    L = np.full(A.size, np.nan)
    base = np.nanpercentile(A[CAL_LO:CAL_HI], BASE_PCT) if A.size > CAL_HI else np.nan
    if not np.isfinite(base) or base <= 1e-9:
        return L, np.nan
    raw = A / float(base) - 1.0
    L[CAL_HI:] = np.maximum(raw[CAL_HI:], 0.0)      # 之前的块保持 NaN（等价于 0）
    return L, float(base)


def cc_first_sustained(v, th, warm=WARM_BLK, n=SUSTAIN):
    """首个 life%（连续 n 块 ≥ th）—— 与 `grade_compare.first_sustained` 同义。"""
    v = np.where(np.isfinite(v), v, -np.inf)
    nb = v.size
    for i in range(warm, nb - n + 1):
        if (v[i:i + n] >= th).all():
            return 100.0 * i / nb
    return None


def analyse(gid):
    d = os.path.join(HERE, gid)
    r = {'gid': gid}
    sf = sorted(glob.glob(os.path.join(d, '*应变*.csv')))
    af = sorted(glob.glob(os.path.join(d, '*声发射*.csv')))
    t = None
    if sf:
        df = read_any(sf[0])
        t = df.iloc[:, 0].to_numpy(float)
        dt = np.diff(t)
        # 时长按**已知 10 Hz 采样率**算（列语义可能为计数器，不能拿列值当秒）
        r['t_kind'], r['dt_s'] = timebase(t)
        r['n_str'] = len(t)
        r['t_max_h'] = len(t) * r['dt_s'] / 3600.0 if np.isfinite(r['dt_s']) else np.nan
        r['dt_med'] = float(np.median(dt))        # 原始列步长（仅作语义诊断）
        r['dt_max'] = float(dt.max())
    if af:
        a = read_any(af[0])
        r['n_ae'] = len(a)
        if t is not None and np.isfinite(r.get('dt_s', np.nan)):
            r['ae_hz'] = len(a) / (len(t) * r['dt_s'])
    if sf:
        A, mode = strain_blocks(sf[0])
        if A is not None and A.size > 10:
            nb = A.size
            r['nblk'] = nb
            r['mode'] = mode
            L, base = stiff_series(A)
            raw = A / base - 1.0 if np.isfinite(base) and base > 1e-9 else np.full(nb, np.nan)
            r['amp_cal'] = base
            r['L_end'] = float(np.nanmax(L[-5:])) if np.isfinite(L[-5:]).any() else np.nan
            r['L_max'] = float(np.nanmax(L)) if np.isfinite(L).any() else np.nan
            r['L_raw_end'] = float(raw[-1])      # 未钳位末值：保留“幅值下降”这类异常信号
            for th in TH:
                t = cc_first_sustained(L, th)
                r['t%.2f' % th] = round(t, 1) if t is not None else None
    return r


def self_check(rows):
    """与官方 `results/grade_stiff_traj.csv` 逐项对账（仅 016-020）。

    存在的意义：这两个表一旦又漂开，答辩时会被直接质疑。
    """
    fp = os.path.join(HERE, 'results', 'grade_stiff_traj.csv')
    if not os.path.exists(fp):
        return []
    g = read_any(fp)
    # ⚠️ CSV 里的 '016' 会被 pandas 读成整数 16 → 必须零填充，否则与 r['gid'] 比不中
    g['gid'] = g['gid'].astype(str).str.strip().str.zfill(3)
    g = g.set_index('gid')
    msgs = []
    for r in rows:
        gid = str(r['gid']).strip()
        if gid not in g.index or 'nblk' not in r:
            continue
        a, b = g.loc[gid], r
        bad = []
        for mine, theirs in (('L_max', 'stiff_max'), ('t0.05', 'x_0.05'),
                             ('t0.10', 'x_0.10'), ('t0.30', 'x_0.30'),
                             ('t0.50', 'x_0.50')):
            x, y = b.get(mine), a.get(theirs)
            x = None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
            y = None if y is None or (isinstance(y, float) and not np.isfinite(y)) else float(y)
            if (x is None) != (y is None) or (x is not None and abs(x - y) > 0.15):
                bad.append('%s=%s vs %s=%s' % (mine, x, theirs, y))
        if bad:
            msgs.append('  [XX] %s 与官方口径不符：%s' % (gid, '；'.join(bad)))
        else:
            msgs.append('  [OK] %s 与官方口径一致（%d 块）' % (gid, b['nblk']))
    return msgs


def fmt(r):
    return ('%-5s 时长=%6.2fh 时间列=%-12s AE=%-8s AE频率=%-6s 块=%-4s 模式=%-6s '
            'L_end=%7.2f L_raw末值=%7.2f | L达阈(连续%d块): %s'
            % (r['gid'], r.get('t_max_h', -1), r.get('t_kind', '?'),
               r.get('n_ae', 'N/A'),
               ('%.2f' % r['ae_hz']) if 'ae_hz' in r else 'N/A',
               r.get('nblk', 'N/A'), r.get('mode', 'N/A'), r.get('L_end', np.nan),
               r.get('L_raw_end', np.nan), SUSTAIN,
               '  '.join('%.0f%%@%s' % (t * 100, r.get('t%.2f' % t, '-')) for t in TH)))


def cc_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups',
                    default='015,016,017,018,019,020,021,022,023,024,025,026,027')
    ap.add_argument('--no-check', action='store_true', help='跳过与官方刚度的对账')
    a = ap.parse_args()
    rows = []
    for g in [s.strip() for s in a.groups.split(',')]:
        if not os.path.isdir(os.path.join(HERE, g)):
            print('%-5s 目录不存在' % g)
            continue
        r = analyse(g)
        rows.append(r)
        print(fmt(r))
    if rows and not a.no_check:
        print('\n-- 与官方口径对账（result/grade_stiff_traj.csv）--')
        for m in self_check(rows):
            print(m)
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(CC_OUT), exist_ok=True)
        df.to_csv(CC_OUT, index=False, encoding='utf-8-sig')
        print('\n已存', CC_OUT)


# ==========================================================================
# 段 6/7  原 main/fusion_compare.py（147 行，正文逐字保留；改名 main→fus_main, CACHE→FUS_CACHE, run_one→fus_run_one）
# ==========================================================================

# -*- coding: utf-8 -*-
"""源融合对照：声发射单源(必做基线) vs 二源 / 三源 —— 逐组比较最终效果。

口径:
  - 主源 = 声发射('ae')，**每组必做单源基线**（设计要求）;
  - 辅助源 = 应变('strain')、光纤('fo')，按"数据有无"参与二源 / 三源融合;
  - 融合方式 = 证据层取 max（risk = max(启用源证据)），逐块结算，不引入未来信息;
  - 评估锚 b2/b3 仅作离线参考（与 eval_common 口径一致）。

输出:
  main/cache/_fusion_cache/<cfg>_<hash>/<gid>.npy   逐点 [D, risk, e_ae, e_strain, e_fo]
  main/results/fusion_compare.csv                   逐组 × 逐配置明细
  main/results/fusion_summary.csv                   配置级聚合
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))          # 项目根(含 shm)
os.chdir(ROOT)

import eval_common as ec                            # noqa: E402
from shm.streaming import StreamSimulator           # noqa: E402
from shm.damage_index import OnlineDamageIndex      # noqa: E402
from shm.config import FO_PARAMS                    # noqa: E402

FUS_CACHE = os.path.join(ROOT, 'cache', '_fusion_cache')
RESDIR = os.path.join(ROOT, 'results')

# 声发射单源必做；其余按可用性叠加。融合算子: max / mean / min
CONFIGS = [
    ('ae',                 ('ae',),               'max'),
    ('ae+strain',          ('ae', 'strain'),      'max'),
    ('ae+fo',              ('ae', 'fo'),          'max'),
    ('ae+strain+fo',       ('ae', 'strain', 'fo'), 'max'),
    ('ae+strain|mean',     ('ae', 'strain'),      'mean'),
    ('ae+fo|mean',         ('ae', 'fo'),          'mean'),
    ('ae+strain+fo|mean',  ('ae', 'strain', 'fo'), 'mean'),
    ('ae+strain|min',      ('ae', 'strain'),      'min'),
    ('ae+strain+fo|min',   ('ae', 'strain', 'fo'), 'min'),
]


def cfg_tag(name):
    """配置指纹（含光纤证据参数），参数变了自动换缓存目录。"""
    h = hashlib.md5((name + repr(sorted(FO_PARAMS.items()))).encode()).hexdigest()[:6]
    return '%s_%s' % (name.replace('|', '_'), h)


def fus_run_one(gid, name, sources, fusion, force=False):
    """跑一组 × 一配置 → (逐点数组, 光纤通道数, 是否有应变数据)。"""
    cdir = os.path.join(FUS_CACHE, cfg_tag(name))
    os.makedirs(cdir, exist_ok=True)
    dpath = os.path.join(cdir, gid + '.npy')
    mpath = os.path.join(cdir, gid + '_meta.npy')
    if os.path.exists(dpath) and os.path.exists(mpath) and not force:
        return np.load(dpath), np.load(mpath)

    params = dict(FO_PARAMS)
    params['sources'] = tuple(sources)
    params['fusion'] = fusion
    di = OnlineDamageIndex(params)
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    has_st = False
    while sim.has_next():
        p = sim.next_point()
        st = p['strain']
        if st is not None and not np.isnan(st):
            has_st = True
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, p.get('fo'),
                  di.shape_value(p.get('ae')))
        rows.append((di.damage, di.risk, di._last_e_ae,
                     di._last_e_strain, di._last_e_fo))
    nfo = len(sim.fo_cols)
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    meta = np.array([nfo, 1 if has_st else 0], dtype=np.int32)
    np.save(dpath, arr)
    np.save(mpath, meta)
    return arr, meta


def fus_main():
    os.makedirs(RESDIR, exist_ok=True)
    rows = []
    for gid in ec.GROUPS:
        print('=== %s ===' % gid)
        for name, sources, fusion in CONFIGS:
            arr, meta = fus_run_one(gid, name, sources, fusion)
            d = arr[:, 0]
            m = ec.per_group_metrics(gid, d)
            nfo, has_st = int(meta[0]), int(meta[1])
            row = dict(cfg=name, gid=gid, sources='+'.join(sources), fusion=fusion,
                       n_fo=nfo, has_strain=has_st,
                       D_end=m['D_end'], t25=m['t25'], t55=m['t55'], t85=m['t85'],
                       tail_mono=m['tail_mono'], t_warn=m['t_warn'],
                       b2=m['b2'], b3=m['b3'], err=m['err'], lead=m['lead'],
                       grade=m['grade'],
                       e_ae_max=float(arr[:, 2].max()),
                       e_st_max=float(arr[:, 3].max()),
                       e_fo_max=float(arr[:, 4].max()))
            rows.append(row)
            print('  %-13s D_end=%.3f t25=%-6s t55=%-6s t85=%-6s err=%-6s '
                  'lead=%-6s %s  (e_ae=%.2f e_st=%.2f e_fo=%.2f)'
                  % (name, m['D_end'],
                     '%.1f' % m['t25'] if np.isfinite(m['t25']) else '--',
                     '%.1f' % m['t55'] if np.isfinite(m['t55']) else '--',
                     '%.1f' % m['t85'] if np.isfinite(m['t85']) else '--',
                     '%.1f' % m['err'] if m['err'] is not None else '--',
                     '%.1f' % m['lead'] if m['lead'] is not None else '--',
                     m['grade'],
                     arr[:, 2].max(), arr[:, 3].max(), arr[:, 4].max()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, 'fusion_compare.csv'),
              index=False, encoding='utf-8-sig')

    sums = []
    for name, _, _ in CONFIGS:
        sub = df[df['cfg'] == name].copy()
        s = ec.summary_row(name, sub)
        s['cfg'] = name
        sums.append(s)
    sdf = pd.DataFrame(sums)[
        ['cfg', 'n55', 'n85', 'nA', 'nE', 'nD', 'nC',
         'D_end_med', 't25_med', 'A_lead_med', 'mean_abs_err']]
    sdf.to_csv(os.path.join(RESDIR, 'fusion_summary.csv'),
               index=False, encoding='utf-8-sig')

    print()
    print('=== SUMMARY ===')
    print(sdf.to_string(index=False))
    print()
    print('wrote:', os.path.join(RESDIR, 'fusion_compare.csv'))
    print('wrote:', os.path.join(RESDIR, 'fusion_summary.csv'))


# ==========================================================================
# 段 7/7  原 main/grade_compare.py（262 行，正文逐字保留；改名 main→grd_main, CACHE→GRD_CACHE, run_one→grd_run_one, first_sustained→grd_first_sustained）
# ==========================================================================

# -*- coding: utf-8 -*-
"""级别层异源分级：AE 决定检测级(L1) + 刚度损失率作为 L2/L3 闸门。

设计:
  risk / D / L1  ← 声发射单源（已实测最优，5/5 精准、误差 3.0%）
  stiff_loss     ← 应变幅值相对基线增长率（载荷控制下 = 刚度损失率）
  L2 判定 = D≥0.55 且 stiff_loss ≥ θ2      （闸门）
  L3 判定 = D≥0.85 且 stiff_loss ≥ θ3      （闸门）

做法: 每组只跑一次（sources=('ae',), strain_mode='stiff', 闸门关），
      记录逐点 [damage, stiff_loss, e_ae, e_strain]，再离线扫 θ2/θ3。
      这样"有效级别时刻" = max(D 达阈时刻, 刚度达阈时刻)，无需反复重跑。

输出:
  main/cache/_grade_cache/<gid>.npy
  main/results/grade_stiff_traj.csv     逐组刚度损失轨迹与越界时刻
  main/results/grade_levels.csv         逐组有效级别时刻(扫 θ2×θ3)
  main/results/grade_summary.csv        阈值组合覆盖率汇总
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
os.chdir(ROOT)

import eval_common as ec                            # noqa: E402
from shm.streaming import StreamSimulator           # noqa: E402
from shm.damage_index import OnlineDamageIndex      # noqa: E402
from shm.config import EXT_BLOCK_PTS, GRADE_GATE   # noqa: E402

GRD_CACHE = os.path.join(ROOT, 'cache', '_grade_cache')
RESDIR = os.path.join(ROOT, 'results')

TH_STIFF = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
GRID2 = [0.10, 0.15]
GRID3 = [0.30, 0.50]
SUSTAIN = 3
WARM_BLK = 10          # 从第 10 块起开始找越界(避开开机瞬态)


def grd_run_one(gid, force=False):
    """跑一组(AE 单源 + 刚度损失率)，返回逐点 [damage, stiff_loss, e_ae, e_strain]。"""
    os.makedirs(GRD_CACHE, exist_ok=True)
    p = os.path.join(GRD_CACHE, gid + '.npy')
    if os.path.exists(p) and not force:
        return np.load(p)
    di = OnlineDamageIndex({'sources': ('ae',), 'strain_mode': 'stiff'})
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    while sim.has_next():
        q = sim.next_point()
        st = q['strain']
        pk = None
        if q.get('ae_new') and q.get('ae'):
            pk = q['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, q.get('fo'),
                  di.shape_value(q.get('ae')))
        rows.append((di.damage, di.stiff_loss, di._last_e_ae, di._last_e_strain))
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    np.save(p, arr)
    return arr


def blk_series(arr_col):
    """逐点 → 逐块(取每块结算后的值)。"""
    return arr_col[EXT_BLOCK_PTS - 1::EXT_BLOCK_PTS]


def grd_first_sustained(v, nb, th, warm=WARM_BLK, n=SUSTAIN):
    """首个 life%(连续 n 块 ≥ th)。"""
    for i in range(warm, nb - n + 1):
        seg = v[i:i + n]
        if (seg >= th).all():
            return 100.0 * i / nb
    return None


def first_ge_block(v, nb, th):
    idx = [i for i in range(nb) if np.isfinite(v[i]) and v[i] >= th]
    return 100.0 * idx[0] / nb if idx else None


def run_gated(gid, force=False):
    """按 GRADE_GATE 逐组口径跑(含级别闸门) → 逐点 [damage, level, stiff_loss]。"""
    os.makedirs(GRD_CACHE, exist_ok=True)
    p = os.path.join(GRD_CACHE, gid + '_gated.npy')
    if os.path.exists(p) and not force:
        return np.load(p)
    params = dict(GRADE_GATE.get(gid, {}))
    params['sources'] = ('ae',)
    params.setdefault('strain_mode', 'stiff')     # 仍算 stiff_loss 供诊断
    di = OnlineDamageIndex(params)
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    while sim.has_next():
        q = sim.next_point()
        st = q['strain']
        pk = None
        if q.get('ae_new') and q.get('ae'):
            pk = q['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, q.get('fo'),
                  di.shape_value(q.get('ae')))
        rows.append((di.damage, di.level, di.stiff_loss))
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    np.save(p, arr)
    return arr


def first_level(lv, n, k):
    """级别首次 ≥ k 的 life%。"""
    idx = np.where(lv >= k)[0]
    return float(idx[0]) / n * 100.0 if len(idx) else None


def grd_main():
    os.makedirs(RESDIR, exist_ok=True)
    traj_rows, lvl_rows = [], []
    per_group = {}

    for gid in ec.GROUPS:
        arr = grd_run_one(gid)
        d_pt = arr[:, 0]
        st_pt = arr[:, 1]
        n = len(arr)
        d_b = blk_series(d_pt)
        s_b = blk_series(st_pt)
        nb = len(d_b)

        m = ec.per_group_metrics(gid, d_pt)
        t25, t55, t85 = m['t25'], m['t55'], m['t85']
        t_l1 = m['t_warn']

        # --- 刚度损失轨迹 ---
        e = np.linspace(0, nb, 11).astype(int)
        seg10 = np.array([np.nanmean(s_b[e[k]:e[k + 1]]) for k in range(10)])
        xrow = {'gid': gid, 'nblk': nb, 'stiff_end': float(s_b[-1]),
                'stiff_max': float(np.nanmax(s_b)), 't_L1_ae': t_l1,
                't25': t25, 't55': t55, 't85': t85, 'grade': m['grade'],
                'err': m['err']}
        for th in TH_STIFF:
            x = grd_first_sustained(s_b, nb, th)
            xrow['x_%.2f' % th] = x
        traj_rows.append(xrow)
        per_group[gid] = dict(d_b=d_b, s_b=s_b, nb=nb, t25=t25, t55=t55,
                              t85=t85, t_l1=t_l1, seg10=seg10)

    # ---- 打印轨迹 ----
    print('=' * 100)
    print('A) 刚度损失率轨迹 (10 段均值) 与越界时刻 [life%%], 连续 %d 块' % SUSTAIN)
    for r in traj_rows:
        print('  %s  nblk=%3d  stiff_end=%.2f max=%.2f' %
              (r['gid'], r['nblk'], r['stiff_end'], r['stiff_max']))
        a = ['%+.2f' % v for v in per_group[r['gid']]['seg10']]
        print('       seg10  ' + ' '.join('%6s' % v for v in a))
        print('       x(θ)   ' + ' '.join('%6s' % ('%.2f' % t)
                                          for t in TH_STIFF))
        print('       life%  ' + ' '.join(
            '%6s' % ('--' if r['x_%.2f' % t] is None else '%.1f' % r['x_%.2f' % t])
            for t in TH_STIFF))
        print('       AE 检测 t_L1=%.1f  D: t25=%.1f t55=%.1f t85=%.1f  %s'
              % (r['t_L1_ae'] or np.nan, r['t25'], r['t55'], r['t85'], r['grade']))
        print()

    # ---- 扫 θ2 × θ3 ----
    print('=' * 100)
    print('B) 有效级别时刻 = max(D 达阈, 刚度达阈)  [life%]')
    for th2 in GRID2:
        for th3 in GRID3:
            print('--- θ2=%.2f (L2)  θ3=%.2f (L3) ---' % (th2, th3))
            print('  gid   tL1(AE)   t25    tL2eff   Gap12    t85    tL3eff   Gap23')
            for gid in ec.GROUPS:
                g = per_group[gid]
                nb, s_b = g['nb'], g['s_b']
                x2 = grd_first_sustained(s_b, nb, th2)
                x3 = grd_first_sustained(s_b, nb, th3)
                tL2 = max([t for t in (g['t55'], x2) if t is not None]) \
                    if (g['t55'] is not None or x2 is not None) else None
                if x2 is None:
                    tL2 = None
                tL3 = None
                if x3 is not None and g['t85'] is not None:
                    tL3 = max(g['t85'], x3, tL2 if tL2 is not None else 0.0)
                gap12 = (tL2 - g['t_l1']) if (tL2 is not None and g['t_l1']) else None
                gap23 = (tL3 - tL2) if (tL3 is not None and tL2 is not None) else None
                lvl_rows.append(dict(th2=th2, th3=th3, gid=gid,
                                     t_L1=g['t_l1'], t25=g['t25'], t55=g['t55'],
                                     tL2_eff=tL2, gap12=gap12,
                                     t85=g['t85'], tL3_eff=tL3, gap23=gap23))
                f = lambda v: '  --  ' if v is None else '%6.1f' % v
                print('  %s  %s %s %s %s %s %s %s'
                      % (gid, f(g['t_l1']), f(g['t25']), f(tL2), f(gap12),
                         f(g['t85']), f(tL3), f(gap23)))
            print()

    # ---- D) 最终口径：逐组级别时刻（含 GRADE_GATE 闸门） ----
    print('=' * 100)
    print('D) 最终口径逐组级别时刻 [life%%]  (GRADE_GATE 逐组闸门)')
    print('  gid   闸门          L1检测    L2预警   L1→L2    L3临危   L2→L3')
    fin_rows = []
    for gid in ec.GROUPS:
        a = run_gated(gid)
        n = len(a)
        lv = a[:, 1]
        gate = GRADE_GATE.get(gid, {})
        t1 = first_level(lv, n, 1)
        t2 = first_level(lv, n, 2)
        t3 = first_level(lv, n, 3)
        g12 = (t2 - t1) if (t1 is not None and t2 is not None) else None
        g23 = (t3 - t2) if (t2 is not None and t3 is not None) else None
        tag = ('θ2=%.2f θ3=%.2f' % (gate['lvl2_stiff'], gate['lvl3_stiff'])) \
            if gate else 'off'
        fin_rows.append(dict(gid=gid, gate=tag, t_L1=t1, t_L2=t2, gap12=g12,
                             t_L3=t3, gap23=g23))
        f = lambda v: '  --  ' if v is None else '%6.1f' % v
        print('  %s   %-12s %s %s %s %s %s'
              % (gid, tag, f(t1), f(t2), f(g12), f(t3), f(g23)))
    pd.DataFrame(fin_rows).to_csv(
        os.path.join(RESDIR, 'grade_levels_final.csv'),
        index=False, encoding='utf-8-sig')
    print()

    df = pd.DataFrame(traj_rows)
    df.to_csv(os.path.join(RESDIR, 'grade_stiff_traj.csv'),
              index=False, encoding='utf-8-sig')
    dl = pd.DataFrame(lvl_rows)
    dl.to_csv(os.path.join(RESDIR, 'grade_levels.csv'),
              index=False, encoding='utf-8-sig')

    # ---- 覆盖率汇总 ----
    print('=' * 100)
    print('C) 覆盖率汇总')
    rows = []
    for th2 in GRID2:
        for th3 in GRID3:
            sub = dl[(dl['th2'] == th2) & (dl['th3'] == th3)]
            nL2 = int(sub['tL2_eff'].notna().sum())
            nL3 = int(sub['tL3_eff'].notna().sum())
            g12 = sub['gap12'].dropna()
            g23 = sub['gap23'].dropna()
            rows.append(dict(th2=th2, th3=th3, nL2=nL2, nL3=nL3,
                             gap12_med=float(g12.median()) if len(g12) else np.nan,
                             gap23_med=float(g23.median()) if len(g23) else np.nan,
                             gap12_min=float(g12.min()) if len(g12) else np.nan,
                             gap23_min=float(g23.min()) if len(g23) else np.nan))
    sdf = pd.DataFrame(rows)
    sdf.to_csv(os.path.join(RESDIR, 'grade_summary.csv'),
               index=False, encoding='utf-8-sig')
    print(sdf.to_string(index=False))
    print()
    print('wrote: grade_stiff_traj.csv / grade_levels.csv / grade_summary.csv')
    print('wrote: grade_levels_final.csv')


# ============================================================ 子命令分发 ====
SUBS = {
    'baselines':  ('基线对照：本项目方法 vs 经典/朴素/监督 ML 基线',        bl_main),
    'loso-cv':    ('留一试件交叉验证（LOSO）：5/5 精准是否靠调参',           loso_main),
    'labels':     ('标签审计：b2 来源 / 敏感性 / 剔除规则事前化 / 统计口径', labels_main),
    'ae-cols':    ('AE 多列（形状/比值列）逐列对照',                        acol_main),
    'grade':      ('级别层异源分级：AE 定检测级 + 刚度损失率作 L2/L3 闸门', grd_main),
    'fusion':     ('源融合对照：AE 单源 vs 二源 / 三源',                     fus_main),
    'candidates': ('候选试件可用性体检（需先跑 grade）',                     cc_main),
}


def main():
    ap = argparse.ArgumentParser(
        prog='main/verify.py',
        description='main 主样本（016-020 + 023-026）验证层统一入口',
        epilog='例：python main/verify.py baselines   |   '
               'python main/verify.py ae-cols --groups "016,023,024,025,026"')
    ap.add_argument('sub', nargs='?', help='子命令（省略则列出全部）')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='透传给子命令的参数')
    a = ap.parse_args()
    if a.sub is None:
        print('可用子命令：')
        for k, (d, _) in SUBS.items():
            print('  %-12s %s' % (k, d))
        print('\n详细参数： python main/verify.py <子命令> -h')
        return 0
    if a.sub not in SUBS:
        print('未知子命令 %r，可用：%s' % (a.sub, ', '.join(SUBS)))
        return 2
    fn = SUBS[a.sub][1]
    old = sys.argv
    sys.argv = ['main/verify.py %s' % a.sub] + list(a.rest)
    try:
        r = fn()
    finally:
        sys.argv = old
    return 0 if r is None else r


if __name__ == '__main__':
    sys.exit(main())
