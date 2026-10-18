"""阈值稳健性扫描：论文 §3.1 那句「26 条阈值规则、四个族」的数据来源。

为什么需要这个脚本
    §3.1 新增了一句反驳「稻草人基线」的话：把绝对阈值法换成另外 25 条阈值规则，
    弃权数最少也只有 1 个，而且只有把中位达阈位置压到 0.3% 寿命时才做得到。
    这个论断必须可复现，否则就是把一个不可核对的数字写进论文。

    v1 里那个「6/15 未触发」的基线只有一条规则，审稿人自然可以问「是不是挑了个差的」。
    本脚本的作用就是把「一条规则」扩成「一族的规则」，并证明前沿是空的。

四个族（共 26 条规则，含论文现行的那一条）
    A0  leave-one-campaign-out：θ = 其余批各试件「失效时 HI」的中位        （论文现行，1 条）
    A1  同一规则整体缩放：θ = α × A0 的 θ                              （10 条，α 见 ALPHAS）
    A2  另一批「全寿命」逐箱 hit 数的分位：θ = Q_q(其余批所有箱)          （6 条）
    A3  另一批「前 20% 寿命」逐箱 hit 数的分位：θ = Q_q(其余批早期箱)     （3 条）
    A4  自参照：θ = k × 该试件自身前 10% 寿命的箱中位                     （6 条）
        —— 这一族最公平（不需要跨机构标定），所以必须测；它也没成功。

    A1 的 α = 1.00 与 A0 完全相同，故只在 A0 计一次。

评价量（与论文一致）
    abstain      15 个试件里始终未超过阈值的个数
    bt_pp        两批中位达阈位置之差（批间极差，pp 寿命）
    wi_pp        各批内极差的批中位（组内极差，pp 寿命）
    ratio        bt_pp / wi_pp（越小越像可迁移；注意它天然 < 1，只作参考）
    med_onset    全部命中试件的中位达阈位置（% 寿命）—— 用来识别「常开」阈值
    early_lt10   命中里达阈位置 < 10% 寿命的比例

判读方式
    「少弃权」与「位置可控」不可兼得：
    凡是 abstain <= 1 的规则，med_onset 都落在 0.3 至 2.7%，即阈值一进记录就触发。

只读 l1/results/*.npz（经 level_vs_shape.curve），只写 l1/results/threshold_sweep.csv。
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import level_vs_shape as L

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.50, 2.00, 3.00)
QUANTILES_FULL = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
QUANTILES_EARLY = (0.95, 0.99, 0.999)
SCALES_OWN = (2, 3, 5, 10, 20, 50)
EARLY_LIFE_PCT = 20.0      # A3 取「前 20% 寿命」的箱
OWN_BASE_PCT = 10.0        # A4 自身的本底窗


def load():
    C = {}
    for g in L.BATCH:
        r = L.curve(g)
        if r is not None:
            C[g] = r
    R = {}
    for g, (life, hit, cum, n_f) in C.items():
        R[g] = dict(life=life, hit=hit, roll=L.roll(hit, L.PERSIST), cum=cum, n_f=n_f)
    return R


def fail_hi(R, g, tail=0.05):
    """失效时的 HI：寿命最后 tail 段的逐箱 hit 数均值。"""
    d = R[g]
    v = d['hit'][d['life'] >= 100 * (1 - tail)]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def onset_for(R, g, theta):
    """3 箱滑动均值首次超过 theta 的寿命百分数；未超过记 NaN。"""
    if not np.isfinite(theta):
        return np.nan
    idx = np.where(R[g]['roll'] > theta)[0]
    return float(R[g]['life'][idx[0]]) if idx.size else np.nan


def summarise(R, onset):
    med, within, abstain, hits = {}, [], 0, []
    for b in ('C3', 'C4'):
        v = np.array([onset.get(g, np.nan) for g in R if L.BATCH[g] == b], float)
        abstain += int((~np.isfinite(v)).sum())
        v = v[np.isfinite(v)]
        if v.size:
            med[b] = float(np.median(v))
            within.append(float(v.max() - v.min()))
            hits.extend(v.tolist())
    bt = max(med.values()) - min(med.values()) if len(med) > 1 else np.nan
    wi = float(np.median(within)) if within else np.nan
    h = np.array(hits, float)
    return dict(abstain=abstain, bt_pp=bt, wi_pp=wi,
                ratio=(bt / wi if wi else np.nan),
                med_onset=(float(np.median(h)) if h.size else np.nan),
                early_lt10=(float((h < 10).mean()) if h.size else np.nan))


def build_rules(R):
    """-> [(family, rule_label, {gid: onset_pct})]，共 26 条。"""
    rules = []
    holds = ('C3', 'C4')

    # A0 —— 论文现行
    base_theta = {}
    for hold in holds:
        tr = [fail_hi(R, g) for g in R if L.BATCH[g] != hold and np.isfinite(fail_hi(R, g))]
        base_theta[hold] = float(np.median(tr))
    rules.append(('A0 leave-one-campaign-out',
                  'theta = median(failure HI) of the other campaign',
                  {g: onset_for(R, g, base_theta[L.BATCH[g]]) for g in R}))

    # A1 —— 同一规则缩放
    for a in ALPHAS:
        rules.append(('A1 scaled leave-one-out',
                      'alpha = %.2f' % a,
                      {g: onset_for(R, g, a * base_theta[L.BATCH[g]]) for g in R}))

    # A2 —— 另一批全寿命逐箱速率的分位
    for q in QUANTILES_FULL:
        o = {}
        for hold in holds:
            pool = np.concatenate([R[g]['hit'] for g in R if L.BATCH[g] != hold])
            th = float(np.quantile(pool[np.isfinite(pool)], q))
            for g in R:
                if L.BATCH[g] == hold:
                    o[g] = onset_for(R, g, th)
        rules.append(('A2 pooled per-bin quantile', 'q = %.3f' % q, o))

    # A3 —— 另一批早期（前 20% 寿命）分位，抬到健康本底之上
    for q in QUANTILES_EARLY:
        o = {}
        for hold in holds:
            pool = np.concatenate([R[g]['hit'][R[g]['life'] <= EARLY_LIFE_PCT]
                                   for g in R if L.BATCH[g] != hold])
            th = float(np.quantile(pool[np.isfinite(pool)], q))
            for g in R:
                if L.BATCH[g] == hold:
                    o[g] = onset_for(R, g, th)
        rules.append(('A3 early-life quantile', 'q = %.3f' % q, o))

    # A4 —— 自参照（最公平的一族）
    for k in SCALES_OWN:
        o = {}
        for g in R:
            base = R[g]['hit'][R[g]['life'] <= OWN_BASE_PCT]
            base = base[np.isfinite(base)]
            th = k * float(np.median(base)) if base.size else np.nan
            o[g] = onset_for(R, g, th)
        rules.append(('A4 self-referenced', 'k = %g x own first-10%% median' % k, o))
    return rules


def run():
    R = load()
    if not R:
        print('没有可用的帧产物（需要 _l1ae_frames_*.npz 与 _l1cyc_*.npz）')
        return 1
    n_c3 = sum(1 for g in R if L.BATCH[g] == 'C3')
    print('可用试件 %d 个（C3 %d，C4 %d）' % (len(R), n_c3, len(R) - n_c3))

    rules = build_rules(R)
    rows = []
    print('\n%-30s %-38s %5s %8s %8s %6s %9s %8s'
          % ('family', 'rule', 'abst', 'bt_pp', 'wi_pp', 'ratio', 'med_onset', 'early<10'))
    print('-' * 120)
    for fam, label, onset in rules:
        s = summarise(R, onset)
        rows.append([fam, label, s['abstain'], s['bt_pp'], s['wi_pp'],
                     s['ratio'], s['med_onset'], s['early_lt10']])
        print('%-30s %-38s %5d %8.1f %8.1f %6.2f %9.1f %8.2f'
              % (fam, label, s['abstain'],
                 *[np.nan if not np.isfinite(s[k]) else s[k]
                   for k in ('bt_pp', 'wi_pp', 'ratio', 'med_onset', 'early_lt10')]))

    # 自检：论文 §3.1 的论断必须复现
    print('\n--- 自检（论文 §3.1 的论断） ---')
    ok = True
    print('规则总数 %d（应为 26）' % len(rows))
    ok &= len(rows) == 26
    best = min(r[2] for r in rows)
    best_rows = [r for r in rows if r[2] == best]
    print('最少弃权数 = %d（应为 1）' % best)
    ok &= best == 1
    for r in best_rows:
        print('   来自 %s / %s，med_onset = %.1f%% 寿命' % (r[0], r[1], r[6]))
    onset_of_best = max(r[6] for r in best_rows)
    print('这些规则的 med_onset 上界 = %.1f%%（应 <= 3.0%%，即落在寿命前 3%%，属「常开」）'
          % onset_of_best)
    ok &= onset_of_best <= 3.0
    a0 = rows[0]
    print('A0 弃权 %d，组内极差 %.1f pp（应为 6 与 60.7）' % (a0[2], a0[4]))
    ok &= (a0[2] == 6) and abs(a0[4] - 60.7) < 0.05
    print('自检结果：%s' % ('通过' if ok else '🔴 与论文不符，检查数据'))

    fp = os.path.join(RES, 'threshold_sweep.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'rule', 'n_abstain', 'bt_pp', 'wi_pp', 'ratio',
                    'med_onset_pct', 'early_lt10'])
        for fam, label, abst, bt, wi, ratio, mo, early in rows:
            w.writerow([fam, label, abst,
                        '' if not np.isfinite(bt) else '%.4f' % bt,
                        '' if not np.isfinite(wi) else '%.4f' % wi,
                        '' if not np.isfinite(ratio) else '%.4f' % ratio,
                        '' if not np.isfinite(mo) else '%.4f' % mo,
                        '' if not np.isfinite(early) else '%.4f' % early])
    print('\n已写 %s' % fp)
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(run())
