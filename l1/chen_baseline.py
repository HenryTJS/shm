# -*- coding: utf-8 -*-
"""文献基线：把 Chen et al. 2025（ICSSD 2025, pp.163 至 169, doi:10.3233/ATDE251187）
的「CAE + 三指标」移植到 L1 的 FBG 剖面上，与本文的 `shape_rob25` 做**同数据对比**。

Chen 2025 的原始口径（逐字取自原文）
------------------------------------
· 输入：100 点的一维振动段（`an input layer for 100x1 signal segments`）
· 模型：卷积自编码器。编码 `Conv1d(32, k=3)` 与 `Conv1d(64, k=5)`，各接 LeakyReLU
  (scale 0.01)；解码为对称的两层转置卷积；Adam，lr=0.001；MSE；50 epochs
· 训练：**只用健康态数据**（`only healthy-state data trained the model`）
· 三个损伤敏感指标：
    ERROR —— 重构误差（原信号与重构信号的差异）
    SSIM  —— 结构相似性（`evaluates the structural similarity between the two`）
    ORSR  —— 原-重构信号比（`reflects the relative amplitude differences in signal energy`）
  原文**没有给公式**，本脚本按标准定义实现（见下）。
· 判读：三指标构成的 3D 聚类能否把不同损伤等级分开（`"stepped" distribution`）

本文的移植（刻意做到「同数据、同基线」）
----------------------------------------
· 输入：连续 W=10 个加载窗 x 10 个 FBG 通道，展平成 **100 点** —— 与 Chen 的输入尺度一致。
  （L1 的每窗只有 10 个通道，单窗喂不进卷积；按窗口序列展平是最接近的对应。）
· 训练集：每组**前 25 % 窗**，与 `shape_rob25` 的暖机段完全相同 ⇒ 两者拿到的基线数据一样。
· 对比量：Spearman rho(指标, 窗口序)。口径与本文一致，可直接与 `shape_rob25` 的 rho 并列。

用法
----
    python l1/chen_baseline.py              # 跑全部组，写 results/l1_chen_baseline.csv
    python l1/chen_baseline.py --groups L1-06
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

WARM = 0.25          # 暖机比例，与 fbg_tools.WARM 一致
WIN = 10             # 每个样本 = 连续 10 个加载窗
CH = 10              # FBG 通道数
EPOCHS = 50          # 照 Chen 2025
LR = 0.001           # 照 Chen 2025
BATCH = 64
SEED = 20261011
MIN_WINS = 30        # 与 fbg_tools._collect() 的过滤一致


class CAE(nn.Module):
    """Chen 2025 的卷积自编码器（结构逐层照抄）。"""

    def __init__(self, n_in=WIN * CH):
        super().__init__()
        self.n_in = n_in
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.LeakyReLU(0.01),
            nn.Conv1d(32, 64, 5, padding=2), nn.LeakyReLU(0.01))
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 5, padding=2), nn.LeakyReLU(0.01),
            nn.ConvTranspose1d(32, 1, 3, padding=1))

    def forward(self, x):
        return self.dec(self.enc(x))


def _ssim1d(x, y, win=7):
    """一维 SSIM（标准滑窗定义）。输入假定已标准化，故用常规常数。"""
    n = x.shape[0]
    w = min(win, n if n % 2 == 1 else n - 1)
    w = max(w, 3)
    k = np.ones(w) / w
    mx = np.convolve(x, k, 'valid')
    my = np.convolve(y, k, 'valid')
    vx = np.convolve(x * x, k, 'valid') - mx ** 2
    vy = np.convolve(y * y, k, 'valid') - my ** 2
    vxy = np.convolve(x * y, k, 'valid') - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / \
        ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(np.mean(s))


def _samples(prof, mode):
    """剖面序列 (n_win, 10) → (样本矩阵, 每个样本代表的窗口号)。

    mode='window'：**单窗**，输入 10 点。与 `shape_rob25` 严格同口径 ——
                  两者都只看当前窗的形状，没有时间上下文。
    mode='stack' ：连续 10 窗展平成 100 点（除 100 点尺度外，还额外给了 CAE
                  历史上下文，而 `shape_rob25` 没有 ⇒ 只作参考，不作主结果）。
    """
    n = prof.shape[0]
    if mode == 'window':
        return np.asarray(prof, dtype=np.float64), np.arange(n)
    idx = np.arange(n - WIN + 1)
    return (np.stack([prof[i:i + WIN].ravel() for i in idx]), idx + WIN - 1)


def run_group(gid, mode='window', verbose=True):
    fp = os.path.join(RES, '_l1fbgprof_%s.npz' % gid)
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    prof, rob25 = z['prof'], z['shape_rob25']
    if prof.shape[0] < MIN_WINS:
        return None
    X, center = _samples(np.asarray(prof, dtype=np.float64), mode)

    k = max(int(round(WARM * X.shape[0])), 8)          # 暖机段样本数
    mu, sd = X[:k].mean(), X[:k].std() + 1e-12
    Xn = (X - mu) / sd

    torch.manual_seed(SEED)
    net = CAE(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.MSELoss()
    tr = torch.tensor(Xn[:k], dtype=torch.float32).unsqueeze(1)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(tr.shape[0])
        for a in range(0, tr.shape[0], BATCH):
            b = perm[a:a + BATCH]
            opt.zero_grad()
            loss = lossf(net(tr[b]), tr[b])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        te = torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)
        rec = net(te).squeeze(1).numpy()

    err = ((Xn - rec) ** 2).mean(axis=1)
    orsr = (np.linalg.norm(Xn, axis=1)
            / (np.linalg.norm(rec, axis=1) + 1e-12))
    ssim = np.array([_ssim1d(Xn[i], rec[i]) for i in range(Xn.shape[0])])

    # 把"样本序"折回"窗口序"：每个样本的代表窗是它的末窗
    r_rob = float(spearmanr(rob25[center], np.arange(len(center))).statistic)
    out = dict(gid=gid, mode=mode, n_win=int(prof.shape[0]),
               n_sample=int(X.shape[0]), warm_n=k,
               rho_shape_rob25=r_rob,
               rho_error=float(spearmanr(err, np.arange(len(err))).statistic),
               rho_ssim=float(spearmanr(ssim, np.arange(len(ssim))).statistic),
               rho_orsr=float(spearmanr(orsr, np.arange(len(orsr))).statistic),
               mean_err_warm=float(err[:k].mean()),
               mean_err_late=float(err[len(err) // 2:].mean()),
               mean_ssim_warm=float(ssim[:k].mean()),
               mean_ssim_late=float(ssim[len(ssim) // 2:].mean()),
               mean_orsr_warm=float(orsr[:k].mean()),
               mean_orsr_late=float(orsr[len(orsr) // 2:].mean()))
    if verbose:
        print('  %-7s %-6s 样本 %4d  暖机 %3d | rho: rob25 %+.3f  ERROR %+.3f  '
              'SSIM %+.3f  ORSR %+.3f'
              % (gid, mode, out['n_sample'], k, out['rho_shape_rob25'],
                 out['rho_error'], out['rho_ssim'], out['rho_orsr']))
    return out


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='')
    ap.add_argument('--modes', default='window,stack')
    a = ap.parse_args()

    gids = ([g.strip() for g in a.groups.split(',') if g.strip()] if a.groups
            else sorted(os.path.basename(f).replace('_l1fbgprof_', '').replace('.npz', '')
                        for f in glob.glob(os.path.join(RES, '_l1fbgprof_*.npz'))))
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    print('Chen 2025 基线移植：%d 组候选，输入模式 %s' % (len(gids), modes))
    print('  模型：Conv1d(32,k=3) + Conv1d(64,k=5) + 对称转置卷积，LeakyReLU(0.01)；'
          'Adam lr=%g，%d epochs（照 Chen 2025）' % (LR, EPOCHS))
    print('  训练集：每组前 %d%% 窗（= shape_rob25 的暖机段，两者基线完全相同）'
          % int(WARM * 100))
    rows = []
    for mode in modes:
        print('\n[模式 %s] %s' % (mode, '单窗 10 点（与 shape_rob25 同口径）'
                                   if mode == 'window' else '10 窗展平 100 点（带时间上下文）'))
        for g in gids:
            r = run_group(g, mode)
            if r:
                rows.append(r)
    if not rows:
        print('没有可用的组。')
        return

    out = os.path.join(RES, 'l1_chen_baseline.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('\n写出 %s（%d 行 = 组 x 模式）\n' % (out, len(rows)))
    for mode in modes:
        sub = [r for r in rows if r['mode'] == mode]
        if not sub:
            continue
        print('[模式 %s]  %d 组' % (mode, len(sub)))
        print('   %-13s %-12s %s' % ('指标', '正相关组数', '中位 rho'))
        for key, name in (('rho_shape_rob25', 'shape_rob25'), ('rho_error', 'ERROR'),
                          ('rho_ssim', 'SSIM'), ('rho_orsr', 'ORSR')):
            vals = sorted(r[key] for r in sub)
            med = vals[len(vals) // 2]
            npos = sum(1 for r in sub if r[key] > 0)
            print('   %-13s %2d / %-9d %+.3f' % (name, npos, len(sub), med))
        print('   %-7s %10s %9s %9s %9s' % ('组', 'rob25', 'ERROR', 'SSIM', 'ORSR'))
        for r in sub:
            print('   %-7s %+10.3f %+9.3f %+9.3f %+9.3f'
                  % (r['gid'], r['rho_shape_rob25'], r['rho_error'], r['rho_ssim'],
                     r['rho_orsr']))
        print()


if __name__ == '__main__':
    main()
