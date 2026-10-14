# -*- coding: utf-8 -*-
"""文献基线的决定性实验：Chen 2025 的 CAE 能不能**跨批**用？

为什么要做
----------
`chen_baseline.py` 显示 Chen 的 ERROR 指标在同数据、同基线、同口径下 b 中位
rho 达到 +0.88，高于本文 `shape_rob25` 的 +0.63。但一个高 rho 有两种来源：
  (a) 真在识别损伤；或
  (b) 只是**记住了源数据的形状**，越晚的窗离源分布越远 ⇒ 重构误差天然单调上升。
区分 (a) 与 (b) 的办法是**跨批迁移**：把在一批上训练的 CAE 原样用到另一批。
  · 若跨批 rho 不降 ⇒ 学到的是通用损伤特征；
  · 若跨批 rho 大幅下降 ⇒ 高 rho 主要来自记忆，不是损伤敏感。

对照口径：`shape_rob25` 是**零训练**的（基线取自该组自己的前 25 % 窗），
所以它没有"迁移"这一步，天然可跨批使用。

批次（L1 数据集自带的载荷程序分组）
------------------------------------
  C3 = 变幅 VA（L1-06/13/14/22/24）
  C4 = 谱载 spectrum（L1-25/27/29/31/35/41/44；L1-30/36 窗数不足已剔除）

用法
----
    python l1/chen_transfer.py
输出
----
    results/l1_chen_transfer.csv
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chen_baseline as CB                             # noqa: E402
import torch                                           # noqa: E402
import torch.nn as nn                                   # noqa: E402
from scipy.stats import spearmanr                       # noqa: E402

RES = CB.RES
C3 = ['L1-06', 'L1-13', 'L1-14', 'L1-22', 'L1-24']
C4 = ['L1-25', 'L1-27', 'L1-29', 'L1-31', 'L1-35', 'L1-41', 'L1-44']


def load_prof(gid):
    fp = os.path.join(RES, '_l1fbgprof_%s.npz' % gid)
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    prof = np.asarray(z['prof'], dtype=np.float64)
    if prof.shape[0] < CB.MIN_WINS:
        return None
    return prof, z['shape_rob25']


def train_on(profs, seed=CB.SEED):
    """用若干组的前 25 % 窗**拼接**训练一个 CAE，返回 (net, mu, sd)。"""
    xs = []
    for p in profs:
        k = max(int(round(CB.WARM * p.shape[0])), 8)
        xs.append(p[:k])
    X = np.concatenate(xs, axis=0)
    mu, sd = X.mean(), X.std() + 1e-12
    Xn = (X - mu) / sd

    torch.manual_seed(seed)
    net = CB.CAE(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=CB.LR)
    lossf = nn.MSELoss()
    tr = torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)
    net.train()
    for _ in range(CB.EPOCHS):
        perm = torch.randperm(tr.shape[0])
        for a in range(0, tr.shape[0], CB.BATCH):
            b = perm[a:a + CB.BATCH]
            opt.zero_grad()
            loss = lossf(net(tr[b]), tr[b])
            loss.backward()
            opt.step()
    net.eval()
    return net, mu, sd


def eval_group(net, mu, sd, prof, rob25):
    Xn = (prof - mu) / sd
    with torch.no_grad():
        rec = net(torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)
                  ).squeeze(1).numpy()
    err = ((Xn - rec) ** 2).mean(axis=1)
    orsr = np.linalg.norm(Xn, axis=1) / (np.linalg.norm(rec, axis=1) + 1e-12)
    ssim = np.array([CB._ssim1d(Xn[i], rec[i]) for i in range(Xn.shape[0])])
    idx = np.arange(len(err))
    return dict(
        rho_error=float(spearmanr(err, idx).statistic),
        rho_ssim=float(spearmanr(ssim, idx).statistic),
        rho_orsr=float(spearmanr(orsr, idx).statistic),
        rho_shape_rob25=float(spearmanr(rob25, idx).statistic))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    data = {}
    for g in C3 + C4:
        r = load_prof(g)
        if r:
            data[g] = r
    print('可用组：C3 %d 组、C4 %d 组\n'
          % (sum(1 for g in C3 if g in data), sum(1 for g in C4 if g in data)))

    models = {}
    for name, gids in (('C3', C3), ('C4', C4)):
        profs = [data[g][0] for g in gids if g in data]
        print('训练 CAE on %s（%d 组前 25%% 窗拼接，共 %d 个样本）...'
              % (name, len(profs), sum(max(int(round(CB.WARM * p.shape[0])), 8)
                                       for p in profs)))
        models[name] = train_on(profs)

    rows = []
    for src in ('C3', 'C4'):
        net, mu, sd = models[src]
        for dst, gids in (('C3', C3), ('C4', C4)):
            for g in gids:
                if g not in data:
                    continue
                prof, rob25 = data[g]
                r = eval_group(net, mu, sd, prof, rob25)
                r.update(gid=g, src_batch=src, dst_batch=dst, n_win=prof.shape[0])
                rows.append(r)

    out = os.path.join(RES, 'l1_chen_transfer.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def med(sub, key):
        v = sorted(x[key] for x in sub)
        return v[len(v) // 2] if v else float('nan')

    print('\n写出 %s（%d 行）\n' % (out, len(rows)))
    print('中位 rho（项目 = 用 src 批训练的 CAE 在 dst 批上评测）')
    print('   %-6s %-6s %-6s %10s %10s %10s %12s'
          % ('src', 'dst', '组数', 'ERROR', 'SSIM', 'ORSR', 'shape_rob25'))
    for src in ('C3', 'C4'):
        for dst in ('C3', 'C4'):
            sub = [r for r in rows if r['src_batch'] == src and r['dst_batch'] == dst]
            if not sub:
                continue
            tag = '批内' if src == dst else '**跨批**'
            print('   %-6s %-6s %-6d %+10.3f %+10.3f %+10.3f %+12.3f   %s'
                  % (src, dst, len(sub), med(sub, 'rho_error'), med(sub, 'rho_ssim'),
                     med(sub, 'rho_orsr'), med(sub, 'rho_shape_rob25'), tag))

    print('\n判读用的关键量（中位 rho）')
    in_ = [r for r in rows if r['src_batch'] == r['dst_batch']]
    cr = [r for r in rows if r['src_batch'] != r['dst_batch']]
    for key, name in (('rho_error', 'ERROR'), ('rho_ssim', 'SSIM'), ('rho_orsr', 'ORSR'),
                      ('rho_shape_rob25', 'shape_rob25')):
        a, b = med(in_, key), med(cr, key)
        drop = ''
        if name != 'shape_rob25' and abs(a) > 1e-9:
            drop = '  衰减 %+.1f%%' % ((abs(b) - abs(a)) / abs(a) * 100.0)
        print('   %-14s 批内 %+.3f   跨批 %+.3f%s' % (name, a, b, drop))
    print('\n   注：`shape_rob25` 是零训练的，它没有 src / dst 之分 ——')
    print('   同一组的 rho 在两行里数值相同，可直接当作「不需迁移」的参照线。')


if __name__ == '__main__':
    main()
