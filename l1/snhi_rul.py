# -*- coding: utf-8 -*-
"""SNHI 曲线配准 —— 把「形状不变」落成一个可执行的 N_f 预测，并对照两条基线。

任务
----
已知试件 j 在 c 不超过 c_cut 的 AE 累积 hit 曲线，预测**失效循环数 N_f**。
c_cut = tau * N_f_true（模拟「只观测到寿命的 tau 比例」，评估时用真值构造观测窗）。

三条方法（同一输入、同一输出量 N_f）
------------------------------------
S   SNHI 形状配准：解 (N_f, alpha) 使观测形状贴合模板
        r_hat(c) = S_bar((c/N_f)^alpha) / S_bar((c_cut/N_f)^alpha)
    模板 S_bar = 训练组归一化形状的逐点中位。**不使用任何绝对量级。**
A   绝对量级法（现行阈值法在 RUL 任务上的对应物）
        N_f = c_cut * T_med / C_j(c_cut)，T_med = 训练组总 hit 量中位
    **依赖绝对量级跨试件、跨批次可比** —— 这正是要检验的假设。
B0  常数先验：N_f = 训练组 N_f 中位。**完全不用试件自身数据。**

为什么要三条而不是两条
----------------------
只和 A 比，赢了也说明不了什么（可能两者都差）。
必须和 B0 比：B0 是「零信息基线」，形状法若打不过它，说明形状法没有额外信息。

判据（**预先定死，看到结果后不调**）—— 只认「同批留一」那一栏
-------------------------------------------------------
成功    tau=0.5 时 S 的 MdAPE 比 B0 低 30% 以上，且三个 tau 上都优于 B0
半成功  S 赢了同批零信息基线，但没到 30% 门槛
失败    S 打不过同批零信息基线（实测就是这一档）

为什么必须同时跑「跨批留出」和「同批留一」
-----------------------------------------
跨批那一栏的 B0 会被**批间 n_f 分布差异**拖垮（C3 与 C4 的 n_f 中位差 4.8 倍），
于是「赢」得虚高。同批留一才是基线公平的干净检验，判据只看它。

留一批交叉验证：train = C4 / test = C3，以及反向；两组预测合并统计。

数据：只用 C3 与 C4 的 15 组（`_l1ae_frames_*.npz` + `_l1cyc_*.npz`）。
只读 npz；写 l1/results/snhi_rul.csv。
"""
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

NB = 100                                  # 寿命网格格数
TAUS = [0.30, 0.50, 0.70]                 # 观测窗：寿命的前百分之几
SHORT_NF = 20000.0                        # 短寿组阈值（曲线不足 3 个非零箱）

CAMPAIGN = {}
for _g in ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']:
    CAMPAIGN[_g] = 'C3'
for _g in ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35',
           'L1-36', 'L1-41', 'L1-44']:
    CAMPAIGN[_g] = 'C4'

U = (np.arange(NB) + 0.5) / NB            # 各格中心的寿命比例


def series(gid):
    """-> (c, cum, n_f)：c 升序的逐帧循环数、到该帧的累积 hit、该组总寿命循环数。"""
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
    return c, np.cumsum(n), n_f


def shape_of(gid, nb=NB):
    """归一化形状 S(u)：累积 hit / 自身总量，在 nb 点寿命网格上（np.interp 线性内插）。"""
    r = series(gid)
    if r is None:
        return None, None
    c, cum, n_f = r
    tot = cum[-1]
    if tot <= 0:
        return None, None
    S = np.interp(U, c / n_f, cum, left=0.0) / tot
    return S, n_f


def sbar_at(Sbar, u):
    """模板在寿命比例 u 处的取值（u 可为数组）。"""
    return np.interp(np.clip(u, 0.0, 1.0), U, Sbar, left=0.0, right=1.0)


def fit_shape(c, cum, c_cut, Sbar, ngrid=110, agrid=64):
    """解 (N_f, alpha) 使观测段形状贴合模板。返回 N_f（预测失效循环数）。"""
    tot = float(np.interp(c_cut, c, cum))
    if tot <= 0:
        return np.nan
    sel = c <= c_cut * (1.0 + 1e-9)
    cc = c[sel]
    rr = np.interp(cc, c, cum) / tot
    if cc.size < 3:
        return np.nan
    best = (np.inf, np.nan)
    for Nf in np.logspace(np.log10(c_cut * 1.02), np.log10(c_cut * 40.0), ngrid):
        for al in np.logspace(np.log10(0.3), np.log10(3.0), agrid):
            uc = (c_cut / Nf) ** al
            if uc <= 1e-6:
                continue
            den = float(sbar_at(Sbar, uc))
            if den <= 1e-6:
                continue
            pred = sbar_at(Sbar, (cc / Nf) ** al) / den
            e = float(np.mean((pred - rr) ** 2))
            if e < best[0]:
                best = (e, float(Nf))
    return best[1]


def jsurface(c, cum, c_cut, Sbar, Nf_true):
    """返回 (全局最小 J, 真值处 J, 取到全局最小的 N_f)。

    若真值处 J 与全局最小 J 相差无几，则目标函数**分不出** N_f 的对错，
    形状配准在原理上就不成立（不是调参能救的）。
    """
    tot = float(np.interp(c_cut, c, cum))
    if tot <= 0:
        return np.nan, np.nan, np.nan
    sel = c <= c_cut * (1.0 + 1e-9)
    cc = c[sel]
    rr = np.interp(cc, c, cum) / tot

    def J(Nf, al):
        uc = (c_cut / Nf) ** al
        if uc <= 1e-6:
            return np.inf
        den = float(sbar_at(Sbar, uc))
        if den <= 1e-6:
            return np.inf
        return float(np.mean((sbar_at(Sbar, (cc / Nf) ** al) / den - rr) ** 2))

    Nfg = np.logspace(np.log10(c_cut * 1.02), np.log10(c_cut * 40.0), 80)
    ag = np.logspace(np.log10(0.3), np.log10(3.0), 40)
    best, bnf = np.inf, np.nan
    for Nf in Nfg:
        for al in ag:
            j = J(Nf, al)
            if j < best:
                best, bnf = j, float(Nf)
    jt = min(J(Nf_true, al) for al in ag)
    return best, jt, bnf


def eval_block(tr, te, tag, S, NF):
    """在 (tr -> te) 上跑三种方法，返回该块的逐条记录。"""
    Sbar = np.median(np.stack([S[g] for g in tr]), axis=0)
    T_med = float(np.median([series(g)[1][-1] for g in tr]))
    Nf_med = float(np.median([NF[g] for g in tr]))
    out = []
    for g in te:
        c, cum, n_f = series(g)
        for tau in TAUS:
            cut = min(max(tau * n_f, c[0] * 1.0000001), c[-1] * 0.999)
            nf_s = fit_shape(c, cum, cut, Sbar)
            cur = float(np.interp(cut, c, cum))
            nf_a = cut * T_med / cur if cur > 0 else np.nan
            out.append(dict(gid=g, camp=CAMPAIGN[g], tag=tag, tau=tau, n_f=n_f,
                            S=nf_s, A=nf_a, B0=Nf_med))
    return out


def stats(recs, tau):
    """某切片下三种方法的 MdAPE(%) 与中位 log2 偏置。"""
    sub = [r for r in recs if abs(r['tau'] - tau) < 1e-9]
    ref = np.array([r['n_f'] for r in sub], float)
    res = {}
    for k in ('S', 'A', 'B0'):
        v = np.array([r[k] for r in sub], float)
        m = np.isfinite(v)
        if not m.any():
            res[k], res[k + '_b'] = np.nan, np.nan
            continue
        res[k] = float(np.median(np.abs(v[m] - ref[m]) / ref[m])) * 100
        res[k + '_b'] = float(np.median(np.log2(v[m] / ref[m])))
    return res


def table(recs, title):
    print('\n=== %s ===' % title)
    print('%-34s %9s %9s %9s' % ('方法', 'tau=0.30', 'tau=0.50', 'tau=0.70'))
    print('-' * 66)
    R = {}
    for key, nm in (('S', 'S   SNHI 形状配准'), ('A', 'A   绝对量级外推'),
                    ('B0', 'B0  常数先验')):
        row = [stats(recs, t)[key] for t in TAUS]
        R[key] = row
        print('%-34s %8.1f%% %8.1f%% %8.1f%%' % (nm, *row))
    print('%-34s %8.2f  %8.2f  %8.2f'
          % ('（S 的中位 log2 偏置）', *[stats(recs, t)['S_b'] for t in TAUS]))
    return R


def main():
    global S, NF
    S, NF = {}, {}
    for g in CAMPAIGN:
        s, nf = shape_of(g)
        if s is not None:
            S[g], NF[g] = s, nf
    ok = sorted(S)
    short = [g for g in ok if NF[g] < SHORT_NF]
    long_ = [g for g in ok if NF[g] >= SHORT_NF]
    print('可用组 %d（C3 %d / C4 %d）；短寿组 %s\n'
          % (len(ok), sum(1 for g in ok if CAMPAIGN[g] == 'C3'),
             sum(1 for g in ok if CAMPAIGN[g] == 'C4'),
             ', '.join(short) if short else '无'))

    # 模式 1：跨批留出（基线会被批间 n_f 分布差异拖垮，仅供展示绝对法的迁移失败）
    recs1 = []
    for hold in ('C3', 'C4'):
        tr = [g for g in long_ if CAMPAIGN[g] != hold]
        te = [g for g in ok if CAMPAIGN[g] == hold]
        blk = eval_block(tr, te, 'cross/%s' % hold, S, NF)
        recs1 += blk
        print('留出 %s：训练 %d 组  n_f 中位 %.0f  ->  测试 %d 组  n_f 中位 %.0f'
              % (hold, len(tr), np.median([NF[g] for g in tr]), len(te),
                 np.median([NF[g] for g in te])))
    R1 = table(recs1, '模式 1：跨批留出（基线不公平，仅示意）')

    # 模式 2：同批留一 —— 基线公平，是「形状法是否真有额外信息」的干净检验
    recs2 = []
    for camp in ('C3', 'C4'):
        pool = [g for g in long_ if CAMPAIGN[g] == camp]
        if len(pool) < 4:
            continue
        for g in pool:
            recs2 += eval_block([x for x in pool if x != g], [g],
                                'loo/%s' % camp, S, NF)
    R2 = table(recs2, '模式 2：同批留一（基线公平 —— 判据用这一栏）')

    # 判据
    i50 = TAUS.index(0.50)
    s50, b50, a50 = R2['S'][i50], R2['B0'][i50], R2['A'][i50]
    better_b0_all = all(R2['S'][j] < R2['B0'][j] for j in range(len(TAUS)))
    if s50 < b50 * 0.70 and better_b0_all:
        verdict = '成功 —— 同批公平基线下达标'
    elif s50 < b50:
        verdict = '半成功 —— 赢了同批零信息基线，但没到 30% 门槛'
    else:
        verdict = '失败 —— 形状法打不过同批零信息基线，无额外信息'
    print('\n判据：tau=0.50 时 S %.1f%%  vs  B0 %.1f%%（需低于其 70%% = %.1f%%）'
          % (s50, b50, b50 * 0.70))
    print('结论：%s' % verdict)

    # 闭环诊断：N_f 到底可不可辨识？
    print('\n=== 可辨识性诊断（tau=0.50）===')
    print('  真值处 J / 全局最小 J 接近 1 ⇒ 目标函数分不出 N_f 对错，方法在原理上失效')
    print('  %-8s %-3s %11s %11s %9s %13s'
          % ('组', '批', '全局min J', '真值处 J', '比值', '最优预测/真值'))
    for camp in ('C3', 'C4'):
        pool = [g for g in long_ if CAMPAIGN[g] == camp]
        for g in pool:
            Sbar = np.median(np.stack([S[x] for x in pool if x != g]), axis=0)
            c, cum, n_f = series(g)
            cut = 0.5 * n_f
            jm, jt, bnf = jsurface(c, cum, cut, Sbar, n_f)
            print('  %-8s %-3s %11.2e %11.2e %9.3f %12.2fx'
                  % (g, camp, jm, jt, jt / jm if np.isfinite(jm) and jm > 0 else np.nan,
                     bnf / n_f))

    print('\n模式 2 tau=0.50 逐试件（预测 / 真值）：')
    print('  %-8s %-3s %11s %14s %14s %11s'
          % ('组', '批', '真实 N_f', 'S 形状', 'A 绝对', 'B0 常数'))
    for r in recs2:
        if abs(r['tau'] - 0.50) > 1e-9:
            continue
        f = lambda v: ('%.2fx' % (v / r['n_f'])) if np.isfinite(v) else '--'
        print('  %-8s %-3s %11.0f %9s %4s %9s %4s %10.0f'
              % (r['gid'], r['camp'], r['n_f'],
                 '%.0f' % r['S'] if np.isfinite(r['S']) else '--', f(r['S']),
                 '%.0f' % r['A'] if np.isfinite(r['A']) else '--', f(r['A']),
                 r['B0']))

    fp = os.path.join(RES, 'snhi_rul.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['mode', 'gid', 'campaign', 'tau', 'n_f_true', 'n_f_shape',
                    'n_f_level', 'n_f_const'])
        for tag, rr in (('cross', recs1), ('loo', recs2)):
            for r in rr:
                w.writerow([tag, r['gid'], r['camp'], '%.2f' % r['tau'],
                            '%.0f' % r['n_f'],
                            '%.0f' % r['S'] if np.isfinite(r['S']) else '',
                            '%.0f' % r['A'] if np.isfinite(r['A']) else '',
                            '%.0f' % r['B0']])
    print('\n已写 %s' % fp)


if __name__ == '__main__':
    main()
