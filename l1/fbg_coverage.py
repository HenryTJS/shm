# -*- coding: utf-8 -*-
"""FBG 采集覆盖与「补值」体检。

为什么要有这个脚本
------------------
2026-10-02 曾误报「L1-25 / L1-31 的 FBG 覆盖率仅 39%」。复核后确认那是
**分母用错**：拿「首末文件之间的全部日历时间」当分母，把**试验暂停期**
也算成缺数据。正确口径下活跃期文件覆盖率是 98.6% 至 100.1%。

但在复核过程中挖出一个**真问题**：

  FBG 是突发式采集（每文件 20.0 s 数据，文件间隔 420 s 或 240 s），
  而 `ae_cycle._ratio_on_grid()` 要求每个 600 s 的 AE 帧在 ±300 s 内
  凑到 **>= 3 个 FBG 窗**才采信，否则把该帧的「加载占比」用
  **全局均值** 补齐（`g[~isfinite] = lf.mean()`）。

  实测多文件组有 **50% 至 70% 的帧走的是补值**，导致 `g_load` 退化成
  只有 2 至 15 个离散取值、循环轴在这些区段近似**线性时间斜坡**，
  并连带使「反解载荷频率」自检失真（L1-31 得 2.715 Hz / 偏差 -26%）。

本脚本做三件事
--------------
1. **文件级覆盖体检**（快，不需要读数据内容）：间隔、跨度、大空隙、
   活跃期覆盖率、FBG 有效观测时长。
2. **空隙归因**：把每个 > 2 h 的空隙与 AE 帧表对照；若该区间 AE 事件数
   近零，则判定为「试验暂停」而非「丢数据」。
3. **补值体检**（`--load-h`，需扫 FBG 内容）：统计每帧 ±300 s 内的
   FBG 窗数分布、真实全局加载占比 gp、以及只用实测得到的加载时长。

用法
----
    python l1/fbg_coverage.py                      # 1 + 2（快）
    python l1/fbg_coverage.py --load-h L1-31 L1-29 # 追加 3（慢，需扫 FBG）
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FRAME_S = 600.0


def _data_l1():
    if os.path.dirname(HERE) not in sys.path:
        sys.path.insert(0, os.path.dirname(HERE))
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                        # noqa: BLE001
        pass
    return HERE


def _file_times(fs):
    import datetime
    return np.array([
        datetime.datetime.strptime(os.path.basename(f)[8:22],
                                   '%Y%m%d%H%M%S').timestamp() for f in fs])


def survey(root):
    """文件级体检 + 空隙归因。"""
    print('=' * 104)
    print('FBG 文件级覆盖体检（活跃期覆盖率 = 小间隙内的实有文件 / 应有 slot）')
    print('=' * 104)
    hdr = (f"{'组':<8}{'文件':>6}{'间隔s':>7}{'跨度h':>8}{'>2h空隙':>9}{'空隙h':>8}"
           f"{'活跃覆盖率':>11}{'有效观测h':>11}{'空隙内AE事件':>13}{'判定':>12}")
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for gid in sorted(os.listdir(root)):
        fs = sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt')))
        if not fs:
            continue
        if len(fs) < 4:
            # 单文件/双文件大文件组（L1-03/04/05/06/09/13/14/24/34…）：
            # 「文件间隔」和「空隙」概念不适用，整条试验就在文件内容里
            tot = sum(os.path.getsize(f) for f in fs) / 1e6
            print(f'{gid:<8}{len(fs):>6}  （大文件组，共 {tot:.0f} MB，不做覆盖/空隙判定）')
            continue
        tt = _file_times(fs)
        if len(tt) < 2:
            continue
        g = np.diff(tt)
        small = g[g <= 7200]
        iv = float(np.median(small)) if len(small) else float(np.median(g))
        span = (tt[-1] - tt[0]) / 3600.0
        big = g[g > 7200]
        exp = small.sum() / iv
        act = len(fs) - len(big)
        cov = 100.0 * act / max(exp, 1.0)

        # 空隙归因：该区间内 AE 事件数
        ae_ev, ae_tot, note = 0.0, 0.0, '—'
        p = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
        if len(big) and os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            td = np.asarray(z['t_day'], dtype=float)
            n = np.asarray(z['n'], dtype=float)
            ae_tot = float(n.sum())
            idx = np.where(g > 7200)[0]
            for i in idx:
                m = (td >= tt[i] / 86400.0) & (td <= tt[i + 1] / 86400.0)
                ae_ev += float(n[m].sum())
            frac = 100.0 * ae_ev / max(ae_tot, 1.0)
            note = ('试验暂停' if frac <= 2.0 else '**需查**（空隙内有活动 %.2f%%）' % frac)
        rows.append((gid, len(fs), iv, span, len(big), big.sum() / 3600.0, cov,
                     len(fs) * 20.0 / 3600.0, ae_ev, ae_tot, note))

    for r in rows:
        print(f'{r[0]:<8}{r[1]:>6}{r[2]:>7.0f}{r[3]:>8.1f}{r[4]:>9}'
              f'{r[5]:>8.1f}{r[6]:>10.1f}%{r[7]:>11.1f}{r[8]:>13.0f}{r[10]:>12}')

    print('\n读法：')
    print('  · 「活跃覆盖率」才是覆盖率；跨度里的长空隙若同时 AE 事件近零，属试验暂停。')
    print('  · 「有效观测h」= 文件数 x 20 s —— 突发式采集的真实观测时长，通常远小于跨度。')
    return rows


def load_h_check(root, gids):
    """补值体检：扫 FBG 内容，统计真实加载占比与只用实测的加载时长。"""
    from ae_cycle import scan_fbg, P2P_LOAD
    print('\n' + '=' * 104)
    print('补值体检：AE 帧（600 s）与 FBG 突发（420 s / 240 s，每文件 2 个 140 行窗）的对齐')
    print('=' * 104)
    for gid in gids:
        p = os.path.join(RES, '_l1cyc_%s.npz' % gid)
        q = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
        if not os.path.exists(q):
            print(f'{gid}: 缺 AE 帧表')
            continue
        z = np.load(q, allow_pickle=True)
        t_sec = z['frame'].astype(float) * FRAME_S

        pts = []
        n_files = 0
        for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
            n_files += 1
            pts.extend(scan_fbg(f))
        if not pts:
            print(f'{gid}: 无 FBG 窗')
            continue
        pts.sort(key=lambda x: x[0])
        tf = np.array([x[0] for x in pts])
        lf = np.array([1.0 if x[1] > P2P_LOAD else 0.0 for x in pts])
        gp = float(lf.mean())

        cnt = np.array([int(((tf >= t - FRAME_S / 2) &
                             (tf < t + FRAME_S / 2)).sum()) for t in t_sec])
        imp = int((cnt < 3).sum())
        # ⚠️ 有效观测时长必须按**文件数 x 20 s** 算；按窗数算会高估约 2 倍
        obs_h = n_files * 20.0 / 3600.0
        frac_load = float(lf.mean())
        load_h_meas = obs_h * frac_load

        old = np.load(p, allow_pickle=True) if os.path.exists(p) else None
        lh_old = float(old['load_h']) if old is not None else float('nan')
        nf = float(old['n_f']) if (old is not None and old['n_f']) else float('nan')
        f_old = float(old['f_hz']) if old is not None else float('nan')

        print(f'\n--- {gid} ---')
        print(f'  AE 帧 {len(t_sec)}  每帧 ±300 s 内 FBG 窗数中位 {np.median(cnt):.0f}'
              f'  <3 的帧 {imp} ({100.0*imp/len(t_sec):.1f}%)'
              f'  <1 的帧 {int((cnt < 1).sum())}')
        print(f'  FBG 文件 {n_files}  窗 {len(tf)}  加载窗 {int(lf.sum())}  '
              f'真实全局加载占比 gp={gp:.3f}')
        print(f'  FBG 有效观测 {obs_h:.1f} h（= 文件数 x 20 s），其中加载 {load_h_meas:.1f} h')
        print(f'  旧口径 load_h={lh_old:.1f} h（含外推）→ 反解 f={f_old:.3f} Hz')
        if load_h_meas > 0 and np.isfinite(nf):
            fm = nf / (load_h_meas * 3600)
            print(f'  若只用 FBG 实测时长：f = {nf:.0f}/({load_h_meas:.1f} x 3600) '
                  f'= {fm:.2f} Hz  <-- **荒谬值，说明此路不通**')
            need_h = nf / 2.0 / 3600.0
            print(f'  按名义 2 Hz 反推需要的加载时长 = {need_h:.1f} h，'
                  f'而 FBG 只观测到 {obs_h:.1f} h（{100.0*obs_h/need_h:.0f}%）')
            print('  ⇒ 结论：FBG 观测时长远短于加载时长，load_h **不可能**由 FBG 直接测得，'
                  '只能外推；因此本组的「反解载荷频率」自检**不可靠**，'
                  '不能反过来当作数据异常的证据。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-h', nargs='*', default=[],
                    help='对这些组追加补值体检（需扫 FBG，较慢）')
    ap.add_argument('--root', default=None)
    a = ap.parse_args()
    root = a.root or _data_l1()
    print(f'数据根目录: {root}')
    survey(root)
    if a.load_h:
        load_h_check(root, a.load_h)


if __name__ == '__main__':
    main()
