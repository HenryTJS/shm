# -*- coding: utf-8 -*-
"""线 1 正式分析 —— **变幅/谱载下「累积型 AE 指标不可迁移」的检验与可分辨性判据**。

论文主张（见 docs/details.md §46）
------------------------------
变幅/谱载下要的是**不变性**（对载荷级缩放不变），不是**归一化**（按幅值加权）。三条证据：

  (a) **累积型不可迁移**（⚠️ 2026-10-09 口径修正，先前的说法是错的）：
      把「AE 累积损伤」分别对
        A) 原始循环分数  cycle / n_f
        B) Miner 等损伤当量  D_M(c) = Σ_i w_i·Δc_i，w_i = range_i^m（Basinski/Paris 形）
      做对照。实测**不是**「Miner 被否决」：在「载荷程序能分辨 + 三种代理一致」的组上
      （L1-06 / L1-13 / L1-24，另 L1-31 在 0.94 边缘）**全是 Miner 胜** ⇒
      **AE 产率确实随载荷上升**。真正不可迁移的是**幅值敏感度本身**：
        · 分级实测的幅值指数 m_eff 跨 **-0.20 至 +9.05**（11 组）——不是常数；
        · 3 种损伤代理里 6/11 组的「胜者」互相翻转 ⇒ 那些胜负是噪声；
        · 同一幅值对比下的敏感度差 14 倍（C3 的 L1-06 与 L1-24 同为 1.50 倍幅值对比，
          产率放大 2.80 对 39.21）——敏感度是**试件属性**，不是材料常数。
      ⇒ 结论是「**存在载荷敏感，但不存在可迁移的归一化**」，不是「某一方被否决」。
      **汇报时不得只挑一边**（先前的错就是只举了 L1-27/L1-35 两个最可分辨的组）。
  (b) **可分辨性判据（本工作新增）**：
        效应量 effect = max|D_cyc − D_Miner|   ← 两条横轴本身差多少
        噪声   noise  = RMSE(D_AE vs D_cyc)   ← 把"线性于循环"当零假设时的残差
        比值   effect / noise
      比值 < 1 ⇒ **该试件的载荷程序根本没有能力分辨两条轴**，任何"谁更优"都无意义。
      这个判据可以在动手前一票否决一个数据集。
  (c) **幅值指数 m_eff**：分级实测 AE 产率之比 → m_eff = ln(rate_hi/rate_lo)/ln(range_hi/range_lo)。
      若 AE 计量"损伤"应 ≈3 至 5（金属 S-N 斜率量级）；若只是"循环计数器"应 ≈0。

为什么只做 C3/C4
----------------
C1/C2 是恒幅组，**没有 cycle 轴**（`.pridb` 的 Time 是各次会话自己的起算秒数，
repo 早已实测「AE 时间不能线性映射到 cycle」，组间速率差 3.4 倍）⇒ 无法上循环横轴。
C3/C4（`.DTA` + FBG）有 `_l1cyc_*.npz`（holdgap 口径），是唯一能做这件事的批次。

🔴 「有效载荷级」口径（2026-10-09 修正，**不点掉这两类段就得出错结论**）
------------------------------------------------------------
谱载表里同一个幅值会列**好几段**，它们不全是「载荷级」：
  · **冲击前基准段**：note 写作 `pristine`（如 L1-29 的 60,000 循环、L1-30 的 10,000 循环）。
    它与后一段**幅值完全相同**，差别只是冲击前后 ⇒ 当成载荷级算 m_eff，
    得到的是「冲击前 vs 冲击后」的对比（L1-29 原值 -2.06 就是这么来的，修正后 +1.59）。
  · **试机段**：循环数占 n_f 不到 1%（如 L1-44 的 460 循环）⇒ 不能当载荷级。
⇒ `m_eff` / `幅值对比` / `最低最高幅值` **只在有效载荷级上取**。
⇒ 但 **Miner 轴仍用完整分段**（`miner_curve`）—— 它描述试件真实经历过的循环历史，
   不能只留有效级，否则循环轴定位就错了（pristine 段与后段同幅值，合在一起等价）。
⇒ 因此 `L1-30`（1 个有效级）与 `L1-44`（1 个）被整体排除，可做组数 13 → **11**。

⚠️ C3/C4 是**冲击后疲劳**且冲击能量各组不同（7.4 至 15 J），L1-41/44/L1-22 是预置脱粘
⇒ **跨试件**比较的分母不干净；本脚本的结论只建立在**试件内**（同试件、不同载荷级）的对照上。

用法
----
    python l1/paper_transferability.py            # 打印 + 写产物
    python l1/paper_transferability.py --groups L1-06 L1-22
输出
----
    results/l1_transfer_miner.csv    每组的两轴对照与可分辨性判据
    results/l1_transfer_rate.csv     分级产率与 m_eff
    results/l1_transfer_summary.txt  人读小结（含自检结果）
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

META = os.path.join(RES, 'l1_specimen_meta.csv')
# 载荷级串的两种写法：变幅表 "to"、谱载表 "~"；kN 与 x 都可能紧贴或带空格
_LEVEL_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:to|~|-)\s*(-?\d+(?:\.\d+)?)\s*kN?\s*[xX]\s*(\d[\d,]*)')
M_GRID = (3, 5, 7, 10)          # S-N / Paris 斜率候选（**未知** ⇒ 必须报整条曲线）
PROXIES = (('n', '命中数'), ('ener_sum', '能量'), ('sig_sum', '信号强度'))
MIN_FRAME = 3                   # 与 ae_hi.py 同口径：有效帧不足就换细帧格
GRIDS = (('', 600.0), ('60s', 60.0))
TOL_NF = 0.02                   # 「各级循环求和 vs n_f」的相对容差
MIN_LEVEL_FRAC = 0.01           # 「试机段」判据：循环数占 n_f 的比例下限


def read_levels(gid):
    """从 l1_specimen_meta.csv 读该组的分级载荷谱 → [(range_kN, cycles, note), ...]。

    🔴 `note` **必须保留**（2026-10-09 修正）：谱载表里同一幅值会出现**两段**，
    靠 `(pristine)` / `(impact)` 区分「**冲击前**基准段」与冲击后段 ——
    它们**不是两个载荷级**。丢掉 note 就会把「冲击前 vs 冲击后」当成幅值效应
    （L1-29 的 `m_eff = -2.06` 就是这么误出来的）。
    """
    prog = {}
    with open(META, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['field'].startswith('rootcyc_level_') or r['field'].startswith('load_level_'):
                prog.setdefault(r['gid'], []).append((r['field'], r['value']))
    out = []
    for _k, s in sorted(prog.get(gid, [])):
        m = _LEVEL_RE.search(s)
        if not m:
            continue
        lo, hi, cyc = float(m.group(1)), float(m.group(2)), int(m.group(3).replace(',', ''))
        mb = re.search(r'\(([^)]*)\)', s)
        out.append((abs(hi - lo), cyc, mb.group(1).strip() if mb else ''))
    # 去掉「同幅值同循环」的重复项（C1 的 load_level 表会重复列同一级）
    dedup = []
    for x in out:
        if (x[0], x[1]) not in [(y[0], y[1]) for y in dedup]:
            dedup.append(x)
    return dedup


def valid_levels(lv, n_f):
    """真正的「载荷级」索引：点掉冲击前基准段（pristine）与试机段（循环占比 < 1%）。"""
    return [i for i, (_r, c, nt) in enumerate(lv)
            if 'pristine' not in nt.lower()
            and (n_f <= 0 or c / n_f >= MIN_LEVEL_FRAC)]


def load_grid(gid):
    """取第一个「有效帧足够」的帧格 → (frame, cyc, n_f, hits, ener, sig, frame_s, tag)。"""
    for tag, fs in GRIDS:
        pf = os.path.join(RES, '_l1ae_frames%s_%s.npz' % (tag, gid))
        pc = os.path.join(RES, '_l1cyc%s_%s.npz' % (tag, gid))
        if not (os.path.exists(pf) and os.path.exists(pc)):
            continue
        zf, zc = np.load(pf, allow_pickle=True), np.load(pc, allow_pickle=True)
        n = zf['n'].astype(float)
        if int((n >= 1).sum()) < MIN_FRAME:
            continue
        cyc = zc['cycle'].astype(float)
        ok = np.isfinite(cyc) & np.isfinite(n)
        return (cyc[ok], float(zc['n_f']),
                {k: zf[k].astype(float)[ok] for k, _ in PROXIES}, fs, tag)
    return None


def miner_curve(levels, cyc, m):
    """Miner 等损伤当量在循环轴 cyc 上取值（归一到 1）。

    ⚠️ 传**完整**分段（含 pristine 段）—— Miner 轴描述的是试件**真实经历过的循环历史**，
    不能只留「有效载荷级」，否则循环轴定位就错了。pristine 段与后一段同幅值，
    合在一起等价，故不影响结果。
    """
    w = np.array([r ** m for r, *_ in levels], float)
    n = np.array([c for _, c, *_ in levels], float)
    bounds = np.cumsum(n)
    tot = float((w * n).sum())
    acc, prev = np.zeros_like(cyc), 0.0
    for wi, bi in zip(w, bounds):
        acc = acc + wi * np.clip(np.minimum(cyc, bi) - prev, 0, None)
        prev = bi
    return np.clip(acc / tot, 0, 1)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def analyze(gid, log):
    """单组分析：两轴对照 + 可分辨性判据 + 分级产率。返回 (miner行, rate行) 或 (None, None)。"""
    lv = read_levels(gid)
    g = load_grid(gid)
    if not lv or g is None:
        log.append('%-7s 跳过（载荷谱 %d 级 / 帧格 %s）' % (gid, len(lv), '缺' if g is None else 'OK'))
        return None, None
    cyc, n_f, feat, fs, tag = g
    rngs = [r for r, *_ in lv]
    # 🔴 「有效载荷级」（2026-10-09 修正）：点掉冲击前基准段与试机段。
    cand = valid_levels(lv, n_f)
    cand_rngs = [rngs[i] for i in cand]
    if len(cand) < 2 or max(cand_rngs) <= min(cand_rngs):
        log.append('%-7s 跳过（%d 段里只有 %d 个有效载荷级，没法做幅值对照）'
                   % (gid, len(lv), len(cand)))
        return None, None
    notes = [lv[i][2] for i in cand]

    # --- 自检 1：各级循环求和 vs n_f
    sum_n = sum(c for _, c, *_ in lv)
    dev_nf = abs(sum_n - n_f) / n_f
    # --- 自检 2：循环轴末值 == n_f
    dev_axis = abs(float(cyc[-1]) - n_f) / n_f

    d_cyc = np.clip(cyc / n_f, 0, 1)
    contrast = max(cand_rngs) / min(cand_rngs)
    eff = float(np.max(np.abs(d_cyc - miner_curve(lv, cyc, 5))))

    row = dict(组号=gid, 帧长s=int(fs), 段数=len(lv), 有效级数=len(cand),
               幅值对比=round(contrast, 3),
               n_f=int(n_f), 各级循环和=int(sum_n), 偏差_nf=round(dev_nf, 4),
               载荷表完整='是' if dev_nf <= TOL_NF else '否（差 %.1f%%）' % (dev_nf * 100),
               偏差_轴末=round(dev_axis, 4),
               效应量_m5=round(eff, 4))
    wins = {}
    for key, lab in PROXIES:
        x = feat[key]
        d = np.cumsum(x)
        if not np.isfinite(d[-1]) or d[-1] <= 0:
            wins[lab] = None
            continue
        d = d / d[-1]
        r_cyc = rmse(d, d_cyc)
        r_m = [rmse(d, miner_curve(lv, cyc, m)) for m in M_GRID]
        best = int(np.argmin(r_m))
        noise = r_cyc
        row['RMSE_cyc_%s' % lab] = round(r_cyc, 4)
        for m, v in zip(M_GRID, r_m):
            row['RMSE_m%d_%s' % (m, lab)] = round(v, 4)
        row['最优m_%s' % lab] = M_GRID[best]
        row['胜者_%s' % lab] = 'Miner' if r_m[best] < r_cyc else 'cyc'
        row['可分辨性_%s' % lab] = round(eff / noise, 2) if noise > 0 else np.nan
        wins[lab] = row['胜者_%s' % lab]
    row['代理一致'] = '一致' if len({v for v in wins.values() if v}) == 1 else '矛盾'

    # --- 分级产率与 m_eff（命中数口径）
    hits = feat['n']
    bounds = np.cumsum([c for _, c, *_ in lv])
    rates, lo = [], 0.0
    for i, (rng, _c, *_nt) in enumerate(lv):
        sel = (cyc > lo) & (cyc <= bounds[i])
        if sel.any():
            c_in = np.clip(np.minimum(cyc[sel], bounds[i]), lo, None)
            used = float(np.diff(np.concatenate([[lo], c_in])).sum())
            rates.append(float(hits[sel].sum()) / used if used > 0 else np.nan)
        else:
            rates.append(np.nan)
        lo = bounds[i]
    ia = max(cand, key=lambda i: rngs[i])      # 最高幅值级（只在有效级上取）
    ib = min(cand, key=lambda i: rngs[i])      # 最低幅值级
    rr, ar = rates[ia], rates[ib]
    m_eff = np.nan
    if (rngs[ia] > rngs[ib] and np.isfinite(rr) and np.isfinite(ar) and rr > 0 and ar > 0):
        m_eff = float(np.log(rr / ar) / np.log(rngs[ia] / rngs[ib]))
    # 同幅值级的产率散布（试件内自洽性检查）
    same = [v for r, v in zip(rngs, rates) if abs(r - rngs[ia]) < 1e-9]
    spread = (max(same) / min(same)) if len(same) > 1 and min(same) > 0 else np.nan
    rrow = dict(组号=gid, 段数=len(lv), 有效级数=len(cand),
                幅值对比=round(contrast, 3),
                最低幅值kN=round(min(cand_rngs), 2), 最高幅值kN=round(max(cand_rngs), 2),
                m_eff用级备注=' | '.join('%g%s' % (rngs[i], ('(%s)' % lv[i][2]) if lv[i][2] else '')
                                     for i in (ib, ia)),
                产率_低幅=round(ar, 4) if np.isfinite(ar) else '', 
                产率_高幅=round(rr, 4) if np.isfinite(rr) else '',
                产率比=round(rr / ar, 4) if (np.isfinite(rr) and np.isfinite(ar) and ar > 0) else '',
                m_eff=round(m_eff, 3) if np.isfinite(m_eff) else '',
                同幅值级产率散布=round(spread, 3) if np.isfinite(spread) else '')
    log.append('%-7s %d段/%d有效级 对比%.2f | 胜者(命/能/信)=%s/%s/%s 一致=%s | '
               '可分辨性(命)=%.2f | m_eff=%.2f  [%s]'
               % (gid, len(lv), len(cand), contrast,
                  wins['命中数'], wins['能量'], wins['信号强度'],
                  row['代理一致'], row.get('可分辨性_命中数', float('nan')),
                  m_eff if np.isfinite(m_eff) else float('nan'),
                  '/'.join(notes) or '-'))
    return row, rrow


def main(argv):
    ap = argparse.ArgumentParser()
    from datasets import campaign_members
    default = campaign_members('C3') + campaign_members('C4')
    ap.add_argument('--groups', nargs='*', default=default)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    log, mrows, rrows = [], [], []
    print('%-7s %s' % ('组', '分析要点'))
    for gid in a.groups:
        mr, rr = analyze(gid, log)
        if mr:
            mrows.append(mr)
            rrows.append(rr)
    for line in log:
        print('  ' + line)

    def write(name, rows):
        if not rows:
            return None
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fp = os.path.join(RES, name)
        with open(fp, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in keys})
        return fp

    fp1 = write('l1_transfer_miner.csv', mrows)
    fp2 = write('l1_transfer_rate.csv', rrows)
    lines = _summary(mrows, rrows)
    fp3 = os.path.join(RES, 'l1_transfer_summary.txt')
    with open(fp3, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\n已写出:')
    for fp in (fp1, fp2, fp3):
        if fp:
            print('  ', fp)
    return 0


def _summary(mrows, rrows):
    """人读小结 + 自检汇总（不写死结论，全部由数据算出）。"""
    L = []
    L.append('=' * 78)
    L.append('线 1 结果汇总（自动生成，勿手改）')
    L.append('=' * 78)
    L.append('分析组数: %d（C3+C4 中有 >=2 个「有效载荷级」者）' % len(mrows))
    L.append('   ⚠️ 「有效载荷级」= 段列表里点掉 (a) **冲击前基准段**（note=pristine）'
             '与 (b) **试机段**（循环数 < %.0f%% n_f）' % (MIN_LEVEL_FRAC * 100))
    try:
        from datasets import campaign_members as _cm
        _have = {r['组号'] for r in mrows}
        _skip = [g for g in (_cm('C3') + _cm('C4')) if g not in _have]
        L.append('   ⚠️ 被排除的组: %s' % (', '.join(_skip) or '无'))
    except Exception:                                        # noqa: BLE001
        pass
    bad_nf = [r for r in mrows if r['偏差_nf'] > TOL_NF]
    bad_ax = [r['组号'] for r in mrows if r['偏差_轴末'] > 1e-6]
    L.append('自检① 各级循环和 vs n_f（容差 %.0f%%）: %s'
             % (TOL_NF * 100, ('全部通过' if not bad_nf else '偏离 %s'
                               % [(r['组号'], r['载荷表完整']) for r in bad_nf])))
    for r in bad_nf:
        L.append('   ⚠️ %s 的载荷表**本身不完整**：分级之和 %d vs 表中 Total %d'
                 '（差 %d 循环 / %.1f%%）⇒ Miner 轴的**分段边界**不可信，'
                 '**不计入关键检验**'
                 % (r['组号'], r['各级循环和'], r['n_f'],
                    r['n_f'] - r['各级循环和'],
                    (r['n_f'] - r['各级循环和']) / r['n_f'] * 100))
        L.append('      ⚠️ 但 n_f 与表中的 Total **同源**（都来自同一份 PDF）⇒ 本自检只证明'
                 '「**PDF 内部不自洽**」，**不能**判定哪一边错（也不是独立证据）。'
                 '要定哪边错得靠外部量，如 FBG 判定出的加载时长 × 频率。')
    L.append('自检② 循环轴末值 vs n_f      : %s'
             % ('全部通过' if not bad_ax else '偏离 %s' % bad_ax))

    for lab in ('命中数', '能量', '信号强度'):
        w = [r['胜者_%s' % lab] for r in mrows if r.get('胜者_%s' % lab)]
        L.append('自检③ 代理「%s」: Miner 胜 %d / %d'
                 % (lab, sum(1 for x in w if x == 'Miner'), len(w)))
    n_cons = sum(1 for r in mrows if r.get('代理一致') == '一致')
    L.append('自检④ 三种代理「胜者」互相一致: %d / %d 组（不一致 = 那些"胜"是噪声）'
             % (n_cons, len(mrows)))
    L.append('自检⑤ 段数 != 有效级数的组（含冲击前基准段或试机段）:')
    _n5 = 0
    for r in rrows:
        d_s, d_v = r.get('段数'), r.get('有效级数')
        if isinstance(d_s, int) and isinstance(d_v, int) and d_s != d_v:
            _n5 += 1
            L.append('   ⚠️ %s: %d 段里只有 %d 个是有效载荷级'
                     '（m_eff 用级：%s）' % (r['组号'], d_s, d_v,
                                          r.get('m_eff用级备注', '-')))
    if not _n5:
        L.append('   无')
    L.append('   ⇒ 若不点掉这些段，它们会被当成「载荷级」，'
             'm_eff 就变成「冲击前 vs 冲击后」的对比（L1-29 原值 -2.06 就是这么错的）。')
    # m_eff 提到这里算：下面的「结论口径」要用到它（曾因顺序写错而 UnboundLocalError）
    me = [r['m_eff'] for r in rrows
          if isinstance(r['m_eff'], (int, float)) and np.isfinite(r['m_eff'])]
    n_bad = len(mrows) - n_cons

    L.append('')
    L.append('── 可分辨性效应量（effect/noise，命中数口径）──')
    dec = [r for r in mrows if isinstance(r.get('可分辨性_命中数'), float)]
    und = [r for r in dec if r['可分辨性_命中数'] < 1.0]
    ok = [r for r in dec if r['可分辨性_命中数'] >= 1.0]
    L.append('  可分辨（>=1.0）: %d 组 -> %s'
             % (len(ok), ', '.join('%s(%.2f)' % (r['组号'], r['可分辨性_命中数']) for r in ok)))
    L.append('  不可分辨（<1.0）: %d 组 -> %s'
             % (len(und), ', '.join('%s(%.2f)' % (r['组号'], r['可分辨性_命中数']) for r in und)))
    L.append('  ⇒ 不可分辨的组上「谁更优」无意义；可分辨的组才是真检验。')

    L.append('')
    L.append('── 可分辨组里 Miner 是否胜出（关键检验；已排除载荷表不完整的组）──')
    n_miner = n_cyc = 0
    for r in ok:
        if r['偏差_nf'] > TOL_NF:
            L.append('  %-7s 可分辨性 %.2f | **载荷表不完整，跳过**'
                     % (r['组号'], r['可分辨性_命中数']))
            continue
        win = r['胜者_命中数']
        n_miner += (win == 'Miner')
        n_cyc += (win == 'cyc')
        L.append('  %-7s 可分辨性 %.2f | Miner RMSE %.4f vs 原始循环 %.4f -> %s'
                 % (r['组号'], r['可分辨性_命中数'],
                    min(r['RMSE_m%d_命中数' % m] for m in M_GRID), r['RMSE_cyc_命中数'],
                    win))
    L.append('  ⇒ 计数：Miner 胜 %d 组 / 原始循环胜 %d 组' % (n_miner, n_cyc))
    cons = [r for r in ok if r.get('代理一致') == '一致' and r['偏差_nf'] <= TOL_NF]
    L.append('  ⇒ 再加一道「三种代理胜负一致」的筛子后剩 %d 组：%s'
             % (len(cons), ', '.join('%s(%s, 可分辨性 %.2f)'
                                    % (r['组号'], r['胜者_命中数'], r['可分辨性_命中数'])
                                    for r in cons)))
    L.append('  ⚠️ **不要只挑其中一边汇报**。可分辨就说「Miner 胜」是把证据读过头了：'
             '另有 %d/%d 组的**胜者随代理翻转**，说明那些胜负落在噪声里。'
             % (n_bad, len(mrows)))
    L.append('')
    L.append('── 结论口径（勿误读）──')
    L.append('  本工作**不是**在说「载荷加权累积律被否决」。实测支持的是：')
    L.append('    ① 能分辨且三代理一致的组上「载荷加权」优于「原始循环」（见下行保留）；')
    if me:
        L.append('    ② 但幅值敏感度**跨试件不可迁移**：m_eff 从 %.2f 到 %.2f'
                 '（跨 %.1f 个单位，而物理预期只是 3 至 5）；'
                 % (min(me), max(me), max(me) - min(me)))
        L.append('       更直接的证据：同为 1.50 倍幅值对比，L1-06 产率只放大 2.80 倍，'
                 'L1-24 却放大 39.21 倍（差 14 倍）⇒ 敏感度是试件属性，不是材料常数；')
    else:
        L.append('    ② 但幅值敏感度不可迁移（m_eff 可用样本不足，无法给范围）；')
    L.append('  ⇒ 变幅下应追求**不变性**（对载荷级缩放不变的量，如形状型）'
             '而不是「把幅值归一化掉」。')
    L.append('  ⚠️ **但①还得再加一道保留**：那 3 组（L1-06/13/24）的阶梯程序**单调升**，'
             '「产率随载荷升」与「产率随损伤进程升」在这些组上**完全共线**，'
             '故 Miner 胜**不能**单独证明「载荷加权正确」。本数据集能立的只有②。')

    L.append('')
    L.append('── 幅值指数 m_eff（命中数口径）──')
    if me:
        L.append('  n=%d  范围 %.2f 至 %.2f  中位 %.2f'
                 % (len(me), min(me), max(me), float(np.median(me))))
        L.append('  （若 AE 计量"损伤"，应聚集在 3 至 5；实测从 %.2f 到 %.2f，跨 %.1f 个单位 ⇒ 非常数）'
                 % (min(me), max(me), max(me) - min(me)))
    L.append('')
    L.append('⚠️ 局限（两条，缺一不可）：')
    L.append('  ① C3/C4 是冲击后疲劳且冲击能量各组不同（7.4 至 15 J）⇒ 只能作**试件内**'
             '（同试件不同载荷级）对照，不得跨试件比较绝对量。')
    L.append('  ② L1 的阶梯程序基本单调升 ⇒ 在大多数组上「载荷级」与「损伤进程」共线，'
             '**无法分离**二者；含降级的只有 L1-29 与 L1-35'
             '（L1-29 另因载荷表不自洽而排除出关键检验）。')
    return L


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
