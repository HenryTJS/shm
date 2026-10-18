# -*- coding: utf-8 -*-
"""分布式应变能否定位损伤 —— 峰位迁移与左右脚对比。

背景
----
现有 `l1_dfos_metrics.py` 把整条光纤压成一个标量 HI（t50/t85），
空间维度被丢掉了。ODiSi-B 沿两条加强筋脚每 0.65 mm 一个测点，
这是四模态里唯一天生带空间分辨的量。本脚本问三件事：

  问题 1  应变剖面的**变化**是否集中在受损的那只脚（方向性检验）
  问题 2  峰位在脚内的相对位置是否稳定，还是随机游走
  问题 3  Δ 剖面在空间上是否可复现（相邻寿命箱的相关系数）

数据形态
--------
`{gid}分布式应变.csv` 是 LUNA 逐行落盘：每行一次测量，整段寿命只有
490 至 1612 行（约每 500 循环一次），列是空间点，间距 0.65 mm。
FBG 的 `{gid}光纤.csv` 是高采样率连续记录（27 万至 85 万行）。
两者时间基相同，但**不能用 FBG 的间隔去切 DFOS**，所以按 DFOS 自身
的测量序号等频分箱。

判据（预先定死）
----------------
方向性检验只在有明确脚归属的组上做（L1-03 右脚、L1-05 左脚）。
  命中率 = 受损脚 |Δ| 更大的寿命箱占比
  中位比 = R/L 的箱中位
  正面  至少一组命中率 ≥ 0.7 且中位比偏离 1 超过 1.5 倍
  反面  两组都接近 0.5 且中位比落在 0.7 至 1.4
问题 2、3 只作描述量报告，不设通过线。

用法
    python l1/dfos_peak_migration.py
写 l1/results/dfos_peak_migration.csv 与 _summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from l1_meta import load_meta          # noqa: E402

ROOT = r'E:\l1'
RES = os.path.join(HERE, 'results')
NBIN = 20
MARGIN = 30.0

import csv                        # noqa: E402
import re                         # noqa: E402


def impact_side():
    """从 l1_impact_truth.csv 的 raw_desc 解析受损加强筋在哪一侧。

    只认「加强筋」的左右，不认「距左边缘」这类字样，也不认蒙皮冲击：
        "stiffener foot (right) ..."        -> R
        "on stiffener (left stiffener) ..." -> L
        "on stiffener (centre) ..."         -> None（未说左右）
        "on skin near right side ..."       -> None（冲击不在加强筋上）
    返回 {gid: 'L'|'R'|None}
    """
    fp = os.path.join(RES, 'l1_impact_truth.csv')
    out = {}
    with open(fp, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            d = (r.get('raw_desc') or '').lower()
            if not d:
                continue
            if 'skin' in d:
                out[r['gid']] = None
                continue
            if re.search(r'(?:stiffener foot|stiffener)\s*\(\s*(?:the\s+)?left', d) or \
               re.search(r'left fatigue stiffener', d):
                out[r['gid']] = 'L'
            elif re.search(r'(?:stiffener foot|stiffener)\s*\(\s*(?:the\s+)?right', d) or \
                    re.search(r'right fatigue stiffener', d):
                out[r['gid']] = 'R'
            else:
                out[r['gid']] = None
    return out


EXPECT = impact_side()
GROUPS = [g for g in sorted(EXPECT) if EXPECT[g]] + \
         [g for g in ('L1-04', 'L1-09', 'L1-23') if g in EXPECT]
print('按位置真值解析出的组与期望侧：')
for g in GROUPS:
    print('   %-7s %s' % (g, EXPECT.get(g) or '不设期望'))
print()


def load_dfos(gid):
    fp = os.path.join(ROOT, gid, '%s分布式应变.csv' % gid)
    cols = list(pd.read_csv(fp, nrows=0, encoding='utf-8-sig').columns)
    pos = np.array([float(c[:-2]) for c in cols[1:]])
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = pd.to_numeric(df['timestamp'], errors='coerce').to_numpy(float)
    M = df.iloc[:, 1:].to_numpy(np.float32)
    ok = np.isfinite(t)
    return t[ok], pos, M[ok]


def safe_corr(a, b):
    if a.size < 8:
        return np.nan
    sa, sb = np.nanstd(a), np.nanstd(b)
    if not (sa > 0 and sb > 0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


SMOOTH_W = 21          # 空间中值滤波窗口 ≈ 13.6 mm（0.65 mm 一格）


def smooth_spatial(a, w=SMOOTH_W):
    """中值滤波。光纤掉点是单点非物理跳变，最高可达数千微应变，
    直接取 max|Δ| 会让峰位锁在掉点上，所以先平滑再找峰。"""
    return pd.Series(a).rolling(w, center=True,
                                min_periods=max(3, w // 3)).median().to_numpy()


def main():
    rows, summ = [], []
    for gid in GROUPS:
        m = load_meta(gid)
        fl, fr = m.get('foot_L'), m.get('foot_R')
        if not fl or not fr:
            print('[%s] 缺脚段定义，跳过' % gid)
            continue
        t, pos, M = load_dfos(gid)
        n = t.size
        if n < NBIN:
            print('[%s] 测量次数仅 %d，跳过' % (gid, n))
            continue

        inL = (pos >= fl[0]) & (pos <= fl[1])
        inR = (pos >= fr[0]) & (pos <= fr[1])
        inF = inL | inR
        outF = ~((pos >= fl[0] - MARGIN) & (pos <= fl[1] + MARGIN)) & \
               ~((pos >= fr[0] - MARGIN) & (pos <= fr[1] + MARGIN))

        edges = np.linspace(0, n, NBIN + 1).astype(int)
        profs = []
        for k in range(NBIN):
            a, b = edges[k], edges[k + 1]
            profs.append(np.nanmedian(M[a:b], axis=0))
        profs = np.array(profs)
        D = profs - profs[0]

        prev = None
        for k in range(NBIN):
            d = np.abs(D[k])
            ds = np.abs(smooth_spatial(D[k]))
            dL, dR = float(np.nanmax(ds[inL])), float(np.nanmax(ds[inR]))
            jL, jR = int(np.nanargmax(ds[inL])), int(np.nanargmax(ds[inR]))
            jAll = int(np.nanargmax(ds))
            rho = np.nan if prev is None else safe_corr(D[k][inF], prev[inF])
            prev = D[k]
            rows.append(dict(
                gid=gid, bin=k, t_frac=round(k / NBIN, 3),
                dL=round(dL, 2), dR=round(dR, 2), dom='R' if dR > dL else 'L',
                rel_L=round(float((pos[inL][jL] - fl[0]) / (fl[1] - fl[0])), 4),
                rel_R=round(float((pos[inR][jR] - fr[0]) / (fr[1] - fr[0])), 4),
                peak_on_foot=bool(inF[jAll]),
                d_out=round(float(np.nanmax(ds[outF])), 2) if outF.any() else np.nan,
                rho_prev=rho, d_raw_max=round(float(np.nanmax(d)), 2),
            ))

        sub = pd.DataFrame([r for r in rows if r['gid'] == gid])
        exp = EXPECT[gid]
        nR = int((sub['dom'] == 'R').sum())
        hit = (nR / len(sub)) if exp == 'R' else \
              (1 - nR / len(sub)) if exp == 'L' else np.nan
        ratio = float(np.median(sub['dR'] / np.maximum(sub['dL'], 1e-9)))
        rel = sub['rel_R'] if exp == 'R' else sub['rel_L'] if exp == 'L' \
            else pd.concat([sub['rel_L'], sub['rel_R']])
        summ.append(dict(
            gid=gid, n_meas=n, n_bin=len(sub),
            foot_L=str(tuple(fl)), foot_R=str(tuple(fr)), expect=exp or '-',
            hit_frac=hit, med_ratio_R_over_L=ratio,
            peak_on_foot_frac=float(sub['peak_on_foot'].mean()),
            rel_pos_std=float(rel.std()),
            rho_prev_med=float(np.nanmedian(sub['rho_prev'])),
            d_out_med=float(np.nanmedian(sub['d_out'])),
            d_foot_med=float(np.nanmedian(np.maximum(sub['dL'], sub['dR']))),
        ))
        print('%-6s 测量 %4d  箱 %2d  期望 %s  命中 %s  R/L %5.2f  '
              '峰在脚上 %3.0f%%  脚内位置 std %.2f  相邻箱 rho %+.2f  '
              '脚内/脚外 %6.1f'
              % (gid, n, len(sub), exp or '-',
                 ('%.0f%%' % (100 * hit)) if exp else '-', ratio,
                 100 * sub['peak_on_foot'].mean(),
                 float(rel.std()), np.nanmedian(sub['rho_prev']),
                 summ[-1]['d_foot_med'] / max(summ[-1]['d_out_med'], 1e-9)))

    pd.DataFrame(rows).to_csv(os.path.join(RES, 'dfos_peak_migration.csv'),
                              index=False, encoding='utf-8-sig')
    df = pd.DataFrame(summ)
    df.to_csv(os.path.join(RES, 'dfos_peak_migration_summary.csv'),
              index=False, encoding='utf-8-sig')
    print()
    print('自检  组数 %d，寿命箱总数 %d' % (len(df), len(rows)))
    sc = df[df['expect'] != '-'].copy()
    if not len(sc):
        return 0
    # 交叉表：位置真值说哪一侧  vs  数据里哪只脚的变化更大
    sc['dom_side'] = np.where(sc['med_ratio_R_over_L'] > 1.0, 'R', 'L')
    ct = pd.crosstab(sc['expect'], sc['dom_side'])
    print('\n交叉表（行 = 位置真值的侧，列 = 变化更大的脚）')
    print(ct.to_string())
    n_ok = int(sum((r['expect'] == r['dom_side']) for _, r in sc.iterrows()))
    print('\n一致 %d / %d 组' % (n_ok, len(sc)))
    print('命中率中位 %.2f   R/L 中位比范围 %.2f 至 %.2f'
          % (sc['hit_frac'].median(), sc['med_ratio_R_over_L'].min(),
             sc['med_ratio_R_over_L'].max()))
    print('峰落在脚上的比例中位 %.0f%%   相邻箱 rho 中位 %+.2f'
          % (100 * sc['peak_on_foot_frac'].median(),
             sc['rho_prev_med'].median()))
    if n_ok >= 0.8 * len(sc):
        print('判定  A 定位可用：真值侧与数据侧一致率达 %d/%d' % (n_ok, len(sc)))
    elif n_ok <= 0.2 * len(sc):
        print('判定  左右口径整体镜像：一致 %d/%d，需反过来读 ODiSi 的 L/R 标签'
              % (n_ok, len(sc)))
    else:
        print('判定  B 定位不可用：一致率 %d/%d 接近随机' % (n_ok, len(sc)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
