# -*- coding: utf-8 -*-
"""L1 的 FBG（光纤光栅）工具链 —— 剖面指标产线 + 覆盖体检 + 口径对比
=================================================================

三个子命令，覆盖 FBG 这一路的全部工作：

| 子命令     | 干什么                                                                 | 产物 |
| ---------- | ---------------------------------------------------------------------- | ---- |
| `profile`  | 扫 FBG 原始文件（`FBG/*.txt`）→ 每个加载窗的 10 通道**归一化应变剖面**与形状指标 | `results/_l1fbgprof_{gid}.npz` + `results/l1_fbgprof_rank.csv` |
| `coverage` | 文件级覆盖体检 + 空隙归因（不读数据内容）；`--load-h` 追加 `g_load` 补值核查 | 只打印报告 |
| `variants` | 10 个形状口径横向对比 + 单组诊断（`--dump/--qc/--b-diag/--b-why/--qc-causal/--rank-check/--batch`） | `results/_l1fbgvar_*.npz` |

为什么要用「形状」而不是「幅值」：同一加载窗内 10 个通道的应变峰值受载荷级支配
（级间差 12 至 15%，还会波动 30%）⇒ 绝对值不可比；把它**按窗归一化**成 10 维分布向量后，
载荷级的整体缩放被消掉，剩下的就是**应变剖面的形状**。脱粘/分层扩展会改变载荷路径
⇒ 剖面形状漂移，这正是老组 B 批用 DFOS 验证过的物理量。

交付口径：**`shape_rob25`**（与前 25% 加载窗的中位剖面的 L1 距离）。初版 `shape_l1`
（与首窗比）有一个反例（L1-31 得 -0.56）；换中位基线后反例归零、|rho| 中位 0.64 升到
0.83。`variants` 的对比表就是这条结论的证据，见 `docs/details.md`。

用法
----
    python l1/fbg_tools.py profile                  # 全部有 FBG 的组
    python l1/fbg_tools.py profile L1-29 L1-41      # 指定组
    python l1/fbg_tools.py coverage                 # 1 + 2（快）
    python l1/fbg_tools.py coverage --load-h L1-31 L1-29   # 追加 g_load 补值核查（慢）
    python l1/fbg_tools.py variants                 # 全部组，出对比表
    python l1/fbg_tools.py variants --dump L1-31    # 单组逐窗诊断
    python l1/fbg_tools.py variants --qc --b-diag   # 可组合几个诊断开关

每个子命令的完整参数用 `python l1/fbg_tools.py <子命令> -h` 查看。

说明
----
本文件由 `l1/fbg_profile.py`(235 行)、`l1/fbg_coverage.py`(202 行)、
`l1/fbg_variants.py`(807 行) 合并而来：三段函数体**逐字保留**，只把三个 `main`
改名成 `profile_main` / `coverage_main` / `variants_main`（后两个改为接收 argv），
并把两处重名的 `_data_l1` 改名 `_data_l1_prof` / `_data_l1_cov`；公共脚手架
（import / HERE / RES / 子命令分发）只写一份。见 `docs/details.md` §33。
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from ae_cycle import scan_fbg, P2P_LOAD                    # noqa: E402

# ==================================================================
# 子命令 profile —— FBG 应变剖面指标产线（写 results/_l1fbgprof_*.npz）
# ==================================================================
MIN_WIN = 5


def profile_of(buf):
    """一个窗的 10 通道峰峰值向量 → 归一化分布 + 特征。"""
    a = np.asarray(buf, dtype=float)
    with np.errstate(all='ignore'):
        p2p = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
    p2p = np.where(np.isfinite(p2p) & (p2p > 0), p2p, 0.0)
    s = p2p.sum()
    if s <= 0:
        return None
    v = p2p / s
    A = v[:5].sum()
    B = v[5:].sum()
    idx = np.arange(len(v), dtype=float)
    return {'prof': v, 'centroid': float((v * idx).sum()),
            'argmax': int(np.argmax(v)), 'asym': float((A - B))}


def group_profile(gid, root=None):
    """该组的加载窗剖面序列 → 特征数组。"""
    root = root or _data_l1_prof()
    recs = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        # 逐窗：scan_fbg 已按窗给出 (t, p2p)；这里要 10 通道，单独再扫一遍
        recs.extend(_scan_windows(f))
    if len(recs) < MIN_WIN:
        return None
    recs.sort(key=lambda r: r['t'])
    prof = np.asarray([r['prof'] for r in recs])
    base = prof[0]
    l1 = np.abs(prof - base).sum(axis=1)
    # 稳健基线：前 10% / 25% 加载窗的**中位剖面**，抗单窗异常。
    # 实测（见 `fbg_tools.py variants`）：单首窗口径有 1 个反例（L1-31 为 -0.56），
    # 换中位基线后反例归零，|rho| 中位由 0.64 升到 0.83（全窗）/ 0.80（暖机后）。
    n = prof.shape[0]
    k10 = max(1, int(round(n * 0.10)))
    k25 = max(1, int(round(n * 0.25)))
    rob10 = np.abs(prof - np.median(prof[:k10], axis=0)).sum(axis=1)
    rob25 = np.abs(prof - np.median(prof[:k25], axis=0)).sum(axis=1)
    return {'t': np.asarray([r['t'] for r in recs]),
            'p2p_med': np.asarray([r['p2p'] for r in recs]),
            'p2p_ch': np.asarray([r['p2p_ch'] for r in recs]),
            'shape_l1': l1,
            'shape_rob10': rob10,
            'shape_rob25': rob25,
            'centroid': np.asarray([r['centroid'] for r in recs]),
            'argmax': np.asarray([r['argmax'] for r in recs]),
            'asym': np.asarray([r['asym'] for r in recs]),
            'prof': prof}


def _scan_windows(path, window_lines=140):
    """与 ae_cycle.scan_fbg 同样的分窗方式，但保留 10 通道向量。

    表头按**内容**识别（首个能解析出时间戳且第 3 到 12 列均为浮点的行）。
    注意：标定式行有 17 列，**不能**用「字段数 >= 12」当数据判据。
    实测结果与旧版硬编码 `i < 60` 逐位相同（数据确实从第 60 行起）。
    """
    from ae_cycle import _file_dt, _parse_ts
    ref = _file_dt(path)
    out, buf, t0 = [], [], None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 12:
                continue
            try:
                vals = [float(x) for x in p[2:12]]
            except ValueError:
                continue
            ts = _parse_ts(p[0] + ' ' + p[1], ref)
            if ts is None:               # 表头残留行
                continue
            if t0 is None:
                t0 = ts
            buf.append(vals)
            if len(buf) >= window_lines:
                r = _win(buf, t0)
                if r:
                    out.append(r)
                buf, t0 = [], None
    if buf:
        r = _win(buf, t0)
        if r:
            out.append(r)
    return out


def _win(buf, t0):
    if t0 is None or not buf:
        return None
    d = profile_of(buf)
    if d is None:
        return None
    a = np.asarray(buf, dtype=float)
    with np.errstate(all='ignore'):
        p2 = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
    # 逐通道峰峰值要**在过滤前**留副本：过滤后长度可能不足 10，
    # 而看板的 FBG 面板要求固定 10 个通道。非有限值补 0。
    p2_ch = np.where(np.isfinite(p2), p2, 0.0)
    p2 = p2[np.isfinite(p2)]
    if len(p2) == 0:
        return None
    med = float(np.median(p2))
    if med <= P2P_LOAD:                # 只留加载窗
        return None
    r = {'t': t0, 'p2p': med, 'p2p_ch': p2_ch}
    r.update(d)
    return r


def profile_main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = _data_l1_prof()
    gids = argv or sorted(d for d in os.listdir(root)
                          if d.startswith('L1-')
                          and glob.glob(os.path.join(root, d, 'FBG', '*.txt')))
    try:
        from scipy.stats import spearmanr
    except Exception:                                        # noqa: BLE001
        spearmanr = None
    print('%-7s %7s %10s %10s %10s %10s' % (
        '组', '加载窗', 'shapeL1中位', '末段L1', 'rho(L1,窗序)', '质心漂移'))
    rows = []
    for gid in gids:
        d = group_profile(gid, root)
        if d is None:
            print('%-7s %7s' % (gid, '窗数不足'))
            continue
        np.savez_compressed(os.path.join(RES, '_l1fbgprof_%s.npz' % gid),
                            t=d['t'], p2p_med=d['p2p_med'], p2p_ch=d['p2p_ch'],
                            shape_l1=d['shape_l1'],
                            shape_rob10=d['shape_rob10'],
                            shape_rob25=d['shape_rob25'],
                            centroid=d['centroid'], argmax=d['argmax'],
                            asym=d['asym'], prof=d['prof'])
        n = len(d['t'])
        idx = np.arange(n, dtype=float)
        rho = np.nan
        if spearmanr is not None and n >= 8:
            rho = float(spearmanr(idx, d['shape_l1'])[0])
        k = max(int(n * 0.3), 1)
        rows.append((gid, n, float(np.median(d['shape_l1'])),
                     float(np.median(d['shape_l1'][-k:])), rho,
                     float(d['centroid'][-k:].mean() - d['centroid'][:k].mean())))
        print('%-7s %7d %10.3f %10.3f %10.3f %10.3f'
              % (gid, n, np.median(d['shape_l1']), np.median(d['shape_l1'][-k:]),
                 rho, rows[-1][5]))
    if rows:
        import csv
        out = os.path.join(RES, 'l1_fbgprof_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '加载窗', 'shapeL1中位', '末段L1', 'rho_shapeL1_vs_窗序',
                        '质心漂移'])
            w.writerows(rows)
        rr = [r[4] for r in rows if np.isfinite(r[4])]
        if rr:
            print('\nrho(形状漂移, 窗序) 方向：正 %d / 负 %d / 弱 %d（共 %d 组），|rho| 中位 %.2f'
                  % (sum(1 for v in rr if v > 0.3), sum(1 for v in rr if v < -0.3),
                     sum(1 for v in rr if abs(v) <= 0.3), len(rr),
                     float(np.median(np.abs(rr)))))
        print('已写出 ->', out)
    return 0


def _data_l1_prof():
    here = HERE
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                        # noqa: BLE001
        pass
    return here


# ==================================================================
# 子命令 coverage —— FBG 采集覆盖体检与空隙归因（只读文件与 npz，不写产物）
# ==================================================================
FRAME_S = 600.0


def _data_l1_cov():
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


def coverage_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-h', nargs='*', default=[],
                    help='对这些组追加补值体检（需扫 FBG，较慢）')
    ap.add_argument('--root', default=None)
    a = ap.parse_args(argv)
    root = a.root or _data_l1_cov()
    print(f'数据根目录: {root}')
    survey(root)
    if a.load_h:
        load_h_check(root, a.load_h)


# ==================================================================
# 子命令 variants —— 10 个剖面口径的横向对比与诊断（读 _l1fbgprof_*.npz）
# ==================================================================
VARIANTS = ['l1_first', 'l1_ref10', 'l1_ref25', 'l1_last', 'l1_mean',
            'cos_first', 'js_first', 'path', 'cent_d', 'asym_d']

# 因果可用 = 只用过去数据（基线取自前段），才能真正在线
CAUSAL = {'l1_first', 'l1_ref10', 'l1_ref25', 'path', 'cos_first', 'js_first'}

# 伪口径 = 用终值/均值当基准，或累积量本身只增不减 ⇒ rho 是数学必然，无独立证据价值
DEGENERATE = {'l1_last', 'l1_mean', 'path'}

WARM = 0.25          # 推荐暖机比例（见文件头结论表）

# 窗质量控制门槛：p2p 低于组内中位的这个比例 ⇒ 不是干净的全载循环窗
QC_REL = 0.80
# 载荷级匹配带宽（相对中位 p2p）
LM_TOL = 0.10


def _paths():
    return sorted(glob.glob(os.path.join(RES, '_l1fbgprof_*.npz')))


def _gid(path):
    return os.path.basename(path).replace('_l1fbgprof_', '').replace('.npz', '')


def load_prof(path):
    with np.load(path) as z:
        if 'prof' not in z.files:
            return None
        prof = np.asarray(z['prof'], dtype=float)
        t = np.asarray(z['t'], dtype=float) if 't' in z.files else None
        p2p = np.asarray(z['p2p_med'], dtype=float) if 'p2p_med' in z.files else None
    ok = np.isfinite(prof).all(axis=1)
    prof = prof[ok]
    if t is not None:
        t = t[ok]
    if p2p is not None:
        p2p = p2p[ok]
    return prof, t, p2p


def _l1(a, b):
    return np.abs(a - b).sum(axis=-1)


def variants(prof):
    """剖面矩阵 (N,10) → 各口径标量序列。"""
    n = prof.shape[0]
    p0 = prof[0]
    k10 = max(1, int(round(n * 0.10)))
    k25 = max(1, int(round(n * 0.25)))
    ref10 = np.median(prof[:k10], axis=0)
    ref25 = np.median(prof[:k25], axis=0)
    out = {}
    out['l1_first'] = _l1(prof, p0)
    out['l1_ref10'] = _l1(prof, ref10)
    out['l1_ref25'] = _l1(prof, ref25)
    out['l1_last'] = _l1(prof, prof[-1])
    out['l1_mean'] = _l1(prof, prof.mean(axis=0))
    with np.errstate(all='ignore'):
        out['cos_first'] = 1.0 - (prof @ p0) / (np.linalg.norm(prof, axis=1)
                                                * np.linalg.norm(p0))
        js = np.array([jensenshannon(prof[i], p0, base=2.0) for i in range(n)])
    # 剖面里必然有 0 通道（p2p 为 0 的通道）⇒ JS 可能在「两边同为 0」处产生 nan，
    # 用稀疏度补：nan 视为 0 距离（该窗剖面与基线完全同支撑）
    out['js_first'] = np.nan_to_num(js, nan=0.0, posinf=0.0)
    d = np.abs(np.diff(prof, axis=0)).sum(axis=1)
    out['path'] = np.concatenate([[0.0], np.cumsum(d)])
    idx = np.arange(prof.shape[1], dtype=float)
    cent = (prof * idx).sum(axis=1)
    asym = prof[:, :5].sum(axis=1) - prof[:, 5:].sum(axis=1)
    out['cent_d'] = np.abs(cent - cent[0])
    out['asym_d'] = np.abs(asym - asym[0])
    return out, {'centroid': cent, 'asym': asym}


def rho_of(y):
    n = len(y)
    if n < 8 or not np.isfinite(y).all() or np.ptp(y) == 0:
        return np.nan
    r = spearmanr(y, np.arange(n)).statistic
    return float(r) if np.isfinite(r) else np.nan


def classify(r):
    if not np.isfinite(r):
        return 'n/a'
    a = abs(r)
    if a < 0.3:
        return '弱'
    return '正' if r > 0 else '负'


def _collect():
    """读全部组的剖面矩阵与各口径标量。"""
    data = {}
    for p in _paths():
        prof, t, p2p = load_prof(p)
        if prof is None or prof.shape[0] < 30:
            continue
        v, extra = variants(prof)
        data[_gid(p)] = (prof, t, p2p, v, extra)
    return data


def _table(rows, title):
    print(f'\n=== {title} ===')
    hdr = f"{'口径':<10}{'正':>4}{'负':>4}{'弱':>4}   {'|rho|中位':>9}   {'因果':>5}"
    print(hdr + '\n' + '-' * len(hdr))
    for k in VARIANTS:
        rs = np.array([r[k] for r in rows], dtype=float)
        cl = [classify(x) for x in rs]
        print(f"{k:<12}{cl.count('正'):>3}{cl.count('负'):>4}{cl.count('弱'):>4}"
              f"   {float(np.nanmedian(np.abs(rs))):>9.2f}   "
              f"{'是' if k in CAUSAL else '否':>4}")


def analyze(warm=WARM, save=True):
    """全窗 + 暖机后双口径对比。"""
    data = _collect()
    if not data:
        print('没有可用的 _l1fbgprof_*.npz（需先跑 fbg_tools.py profile 生成 prof 字段）')
        return None
    gids = list(data)
    allrows, postrows, confrows = [], [], []

    for g in gids:
        prof, t, p2p, v, extra = data[g]
        n = prof.shape[0]
        k0 = max(1, int(round(n * warm)))
        ra = {k: rho_of(v[k]) for k in VARIANTS}
        rp = {k: (rho_of(v[k][k0:]) if n - k0 >= 8 else np.nan) for k in VARIANTS}
        rc = {k: (float(spearmanr(v[k], p2p).statistic)
                  if p2p is not None and np.ptp(p2p) > 0 else np.nan)
              for k in VARIANTS}
        allrows.append(ra)
        postrows.append(rp)
        confrows.append(rc)
        if save:
            np.savez_compressed(
                os.path.join(RES, '_l1fbgvar_%s.npz' % g),
                prof=prof, warm_k=k0,
                t=t if t is not None else np.array([]),
                p2p_med=p2p if p2p is not None else np.array([]),
                **{k: v[k] for k in VARIANTS})

    print(f'组数 {len(gids)}：{", ".join(gids)}')
    _table(allrows, '全窗评价（k 从 0 起，与初版口径可直接对比）')
    _table(postrows, f'暖机后评价（k >= {warm:.0%} 寿命，真正的在线区间）')

    print(f'\n>>> 暖机后 |rho| 中位最高的因果口径：')
    cand = [k for k in VARIANTS if k in CAUSAL and k not in DEGENERATE]
    rank = sorted(cand, key=lambda k: -float(np.nanmedian(
        np.abs([r[k] for r in postrows]))))
    for k in rank[:3]:
        rs = np.array([r[k] for r in postrows], dtype=float)
        cl = [classify(x) for x in rs]
        print(f"    {k:<12}|rho|中位 {float(np.nanmedian(np.abs(rs))):.2f}   "
              f"正 {cl.count('正')} / 负 {cl.count('负')} / 弱 {cl.count('弱')}")

    print('\n--- 反例（rho < -0.3）---')
    for tag, rows in (('全窗', allrows), ('暖机后', postrows)):
        out = []
        for k in VARIANTS:
            neg = [f"{gids[i]}({rows[i][k]:+.2f})" for i in range(len(gids))
                   if np.isfinite(rows[i][k]) and rows[i][k] < -0.3]
            if neg:
                out.append(f'    {k:<10} {len(neg):>2} 组: {", ".join(neg)}')
        print(f'  [{tag}]' + ('\n' + '\n'.join(out) if out else ' 无'))

    print('\n--- 逐组 rho（暖机后）---')
    print(f"{'组':<8}{'窗数':>7}" + ''.join(f'{k:>11}' for k in cand))
    for i, g in enumerate(gids):
        print(f"{g:<8}{data[g][0].shape[0]:>7}"
              + ''.join(f"{postrows[i][k]:>11.3f}" for k in cand))

    print('\n--- 混杂检验：指标是否在跟踪载荷水平 p2p_med ---')
    print(f"{'口径':<11}{'|rho(指标,p2p)|中位':>20}{'最大':>8}{'>=0.5组数':>11}")
    for k in VARIANTS:
        c = np.array([r[k] for r in confrows], dtype=float)
        print(f"{k:<13}{np.nanmedian(np.abs(c)):>18.3f}"
              f"{np.nanmax(np.abs(c)):>8.3f}{int((np.abs(c) >= 0.5).sum()):>11}")

    if save:
        out = os.path.join(RES, 'l1_fbgvariant_rank.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['组号', '窗数'] + [f'{k}_全窗' for k in VARIANTS]
                       + [f'{k}_暖机后' for k in VARIANTS])
            for i, g in enumerate(gids):
                w.writerow([g, data[g][0].shape[0]]
                           + [allrows[i][k] for k in VARIANTS]
                           + [postrows[i][k] for k in VARIANTS])
        print(f'\n写出 {out}')
        print(f'写出 _l1fbgvar_*.npz（含剖面矩阵与全部口径）x {len(gids)}')
    return allrows, postrows


def dump(gid):
    """单组逐窗诊断：分位段看剖面漂移方向与首窗是否异常。"""
    p = os.path.join(RES, f'_l1fbgprof_{gid}.npz')
    if not os.path.exists(p):
        print(f'缺 {p}')
        return
    prof, t, p2p = load_prof(p)
    n = prof.shape[0]
    v, extra = variants(prof)
    print(f'=== {gid}  n={n} 窗口 ===')
    if t is not None:
        print(f'时间跨度 {((t[-1] - t[0]) / 3600.0):.2f} h')
    if p2p is not None:
        print(f'加载窗 p2p 中位 {np.median(p2p):.1f} (min {p2p.min():.1f} / max {p2p.max():.1f})')

    print('\n各口径 rho(与窗序)：')
    for k in VARIANTS:
        print(f'  {k:<12}{v[k][-1]:>10.4f} 末值   rho={rho_of(v[k]):+.3f}')

    print('\n剖面（每 10% 寿命取中位，10 通道）：')
    print('  ' + 'seg'.ljust(12) + ''.join(f'{i:>8}' for i in range(10)))
    for lo in np.linspace(0, 0.9, 10):
        a, b = int(lo * n), max(int(lo * n) + 1, int((lo + 0.1) * n))
        seg = prof[a:b].mean(axis=0)
        print(f'  {lo * 100:>3.0f}%+'.ljust(12) + ''.join(f'{x:>8.4f}' for x in seg))

    print('\nl1_first / l1_ref10 / l1_ref25 / p2p 沿寿命走势（每 10% 中位）：')
    print('  ' + 'seg'.ljust(10) + f"{'l1_first':>10}{'l1_ref10':>10}{'l1_ref25':>10}"
          f"{'p2p':>10}{'centroid':>10}{'asym':>10}")
    for lo in np.linspace(0, 0.9, 10):
        a, b = int(lo * n), max(int(lo * n) + 1, int((lo + 0.1) * n))
        ptxt = (f"{np.median(p2p[a:b]):>10.1f}" if p2p is not None
                else f"{'-':>10}")
        print(f'  {lo * 100:>3.0f}%+'.ljust(10)
              + f"{np.median(v['l1_first'][a:b]):>10.4f}"
              + f"{np.median(v['l1_ref10'][a:b]):>10.4f}"
              + f"{np.median(v['l1_ref25'][a:b]):>10.4f}"
              + ptxt
              + f"{np.median(extra['centroid'][a:b]):>10.3f}"
              + f"{np.median(extra['asym'][a:b]):>10.4f}")

    # 首窗 vs 首 10% 中位的距离 → 判断首窗是否离群
    k10 = max(1, int(round(n * 0.10)))
    ref10 = np.median(prof[:k10], axis=0)
    d0 = np.abs(prof[0] - ref10).sum()
    dall = np.abs(prof[:k10] - ref10).sum(axis=1)
    print(f'\n首窗偏离「首 10% 中位」的距离 = {d0:.4f}；'
          f'该段内中位偏离 = {np.median(dall):.4f}，最大 = {dall.max():.4f}')
    print(f'首窗是否离群: {"是" if d0 > 2 * np.median(dall) else "否"}')

    print('\n首 6 窗剖面：')
    for i in range(min(6, n)):
        print(f'  w{i}: ' + ' '.join(f'{x:.4f}' for x in prof[i]))
    print('\n分位处剖面：')
    for lo in [0.0, 0.25, 0.5, 0.75, 0.9]:
        i = min(n - 1, int(lo * n))
        print(f'  {int(lo * 100):>3}%: ' + ' '.join(f'{x:.4f}' for x in prof[i]))


def _rho(x):
    """序列与自身序号的 Spearman 相关；无定义时返回 0。"""
    r = spearmanr(x, np.arange(len(x)))[0]
    return 0.0 if not np.isfinite(r) else float(r)


def _rob25(prof):
    """稳健基线口径：与「前 25% 中位剖面」的 L1 距离。"""
    k = max(2, int(round(len(prof) * 0.25)))
    return _l1(prof, np.median(prof[:k], axis=0))


def qc_scan(tol=LM_TOL, rel=QC_REL, null_n=400, seed=20261004):
    """窗质量控制 / 载荷级匹配诊断（L1-14 查因，2026-10-04）。

    动机：L1-14 的 `shape_rob25` 与窗序 rho 只有 +0.10，是全批真弱组之一。
    诊断发现它的剖面**寿命漂移只有 0.039（13 组最小，次小 L1-27 是 0.129）**，
    而这点方差几乎全部来自一批 **低 p2p 窗**（占 13.5%）：那批窗的剖面更平坦、
    离基线约 0.10，而 86% 的正常窗只有 0.011 ⇒ rho(shape, p2p) = -0.80（13 组最强）。
    它的载荷程序只有 3 级、幅度仅差 25%（-4/-40 → -5/-50 kN），
    但实测 p2p 跳度达 4 倍 ⇒ p2p 的变化**不能**归因于载荷级，只能归因于**窗质量**。

    两种过滤：
      A. 窗质量控制 `p2p > rel * 中位`：只砍掉非全载窗。
      B. 载荷级匹配 `|p2p - 中位| < tol * 中位`：更窄，等价于「同载比较」。

    零假设检验：B 把样本量砍到 k，而「随机砍到同样样本量」本身也可能改变 rho，
    故对每组抽 null_n 次**同样本量随机子样本**，走完全相同的流程，
    看观测值在零分布上的位置（p 值）。

    结论（11 组，n>=200，2026-10-04 首次得出）：
      A 几乎无害（|Δrho| <= 0.02，10 组）；B 是双刃剑：
      L1-14 0.104→0.511、L1-25 0.506→0.913，但 L1-41 0.821→0.613（显著变差）。

    ⚠️ **2026-10-05 更正（两条，见 `--b-why` 与 `--qc-causal`）**：
      1. 本函数里的 B 是「子集**重算基线** + **子集序号**」，而 `_rob25` 的 rho
         对这个基线的取法很敏感。把基线固定为全窗基线后：
           L1-14 0.511 → **-0.523**（“提升”消失且反向）
           L1-41 0.613 → ** 0.796**（“变差”基本消失）
           L1-31 0.619 → ** 0.420**（这一组 B 确实损坏了内容）
         ⇒ B 不可做交付口径的结论**不变**，但**理由要改**：不是“窄带匹配破坏
         寿命覆盖”（`--b-diag` 已否证：L1-41 的 B 子集寿命分布是均匀的），
         而是 **`_rob25` 在更小子集上重算基线会让 rho 大幅漂移**。
      2. 上面那对旧数字（L1-41 0.461）已过期 —— 缓存 `_l1fbgprof_*.npz` 重生后
         变为 **0.613**。教训：docstring 里**不要写死数值**（§16 反复强调过）。
      3. A 的门槛是**非因果**的（用全寿命中位）；改成因果形式（`--qc-causal`）后
         |Δrho| 中位仅 0.007 至 0.008，但 **L1-14 的 0.208 掉回 0.065**
         ⇒ A 对 L1-14 的“提升”本身就是非因果门槛的产物，不是真收益。
    """
    rng = np.random.default_rng(seed)
    print('%-7s %5s %8s | %18s | %20s' % (
        '组', '窗数', '全窗', 'A: p2p>%.2f*中位' % rel, 'B: |p2p-中位|<%.0f%%' % (tol * 100)))
    rows = []
    for p in _paths():
        prof, _t, p2p = load_prof(p)
        if prof is None or p2p is None or len(p2p) < 200:
            continue
        gid = _gid(p)
        n = len(p2p)
        full = _rho(_rob25(prof))
        m = p2p > rel * np.median(p2p)
        ra = _rho(_rob25(prof[m])) if m.sum() >= 30 else float('nan')
        ref = np.median(p2p)
        mb = np.abs(p2p - ref) < tol * ref
        kb = int(mb.sum())
        rb, pv, nmean = float('nan'), float('nan'), float('nan')
        if kb >= 30:
            rb = _rho(_rob25(prof[mb]))
            null = np.asarray([
                _rho(_rob25(prof[np.sort(rng.choice(n, kb, replace=False))]))
                for _ in range(null_n)])
            pv = float((null >= rb).mean())
            nmean = float(null.mean())
        rows.append((gid, n, full, ra, int(m.sum()), rb, kb, pv, nmean))
        print('%-7s %5d %8.3f | %7.3f (n=%4d) | %7.3f (n=%4d) p=%.3f 零均值 %+.3f' % (
            gid, n, full, ra, int(m.sum()), rb, kb, pv, nmean))
    return rows


def b_diag(tol=LM_TOL, rel=QC_REL):
    """B 口径（载荷级匹配）为何在部分组把 rho 砸掉 —— 看它选了寿命的哪一段。

    背景：`--qc` 的结论里 B 是双刃剑（L1-14 0.104→0.511，但 L1-41 0.821→0.461）。
    本诊断检验一个很具体的机制：
      **B 的带宽 `|p2p − 中位| < tol·中位` 可能比载荷程序的两级幅度差还窄**，
      于是 B 只选到其中一级 ⇒ 只覆盖寿命的一段 ⇒
      「剖面 vs 寿命序」的 rho 就不再是寿命漂移，而是在一个窄时间带内测别的东西。
    例：L1-41 的 level_1 幅度 18.6 kN、level_2 32.1 kN（差 **1.73 倍**），远大于 ±10%。

    输出每组：全窗/A/B 的 rho、样本量、**寿命分布**（选中窗在寿命十等分里的占比，
    逐位一个数字，以 10% 为单位；`1111111111` = 均匀）以及 p2p 自身随寿命序的 rho
    （若很强，B 必然挑成一段）。
    """
    print('%-7s %5s %8s | %-26s | %-26s | %s' % (
        '组', '窗数', 'rho全窗', 'A: p2p>%.2f*中位' % rel, 'B: |p2p-中位|<%.0f%%' % (tol * 100),
        'p2p 分位(10/25/50/75/90)'))
    rows = []
    for p in _paths():
        prof, _t, p2p = load_prof(p)
        if prof is None or p2p is None or len(p2p) < 200:
            continue
        gid = _gid(p)
        n = len(p2p)
        ref = float(np.median(p2p))
        mA = p2p > rel * ref
        mB = np.abs(p2p - ref) < tol * ref
        full = _rho(_rob25(prof))

        def stat(m):
            idx = np.nonzero(m)[0]
            if idx.size < 30:
                return None
            h = np.histogram(idx / max(n - 1, 1), bins=np.linspace(0, 1, 11))[0]
            # 寿命分布：逐等分占比（以 10% 为单位取整，0 至 9）—— 直接看出密集在哪段
            life = ''.join(str(min(int(round(100.0 * v / idx.size / 10.0)), 9)) for v in h)
            return {'k': int(idx.size), 'life': life,
                    'rho': _rho(_rob25(prof[m]))}
        sA, sB = stat(mA), stat(mB)
        qs = np.quantile(p2p, [0.1, 0.25, 0.5, 0.75, 0.9])
        rho_p2p = _rho(p2p)
        rows.append((gid, n, full, sA, sB, qs, rho_p2p))

        def fmt(s):
            if s is None:
                return 'n<30 跳过'
            return '%7.3f n=%4d 寿命%s' % (s['rho'], s['k'], s['life'])
        print('%-7s %5d %8.3f | %s | %s | %s  rho(p2p,序)=%+.2f'
              % (gid, n, full, fmt(sA), fmt(sB),
                 '/'.join('%.0f' % v for v in qs), rho_p2p))
    return rows


def _partial_rank_shuffle(x, y, z):
    """x、y 的秩偏相关（控制 z）。三者都按秩算，避免量纲问题。"""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = float(np.corrcoef(rx, ry)[0, 1])
    rxz = float(np.corrcoef(rx, rz)[0, 1])
    ryz = float(np.corrcoef(ry, rz)[0, 1])
    den = np.sqrt(max((1.0 - rxz ** 2) * (1.0 - ryz ** 2), 1e-12))
    return (rxy - rxz * ryz) / den


def b_why(tol=LM_TOL, rel=QC_REL):
    """B 口径为何在 L1-41 把 rho 从 0.821 砸到 0.613 —— 拆开两个效应。

    `qc_scan` 的 B 是「子集**重算基线** + **子集序号**」，所以 rho 变化可能有三个来源：
      (a) **子集内容**：选走的窗本身携带多少漂移信号；
      (b) **重算基线**：`_rob25` 在子集的前 25% 上重新取基线；
      (c) **序号归一**：子集序号被重新拉到 0 至 1。
    本诊断固定基线为**全窗的 `_rob25`**（`shape_full`），只用原寿命序，
    于是「子集内容」以外的效应被排除 —— 由此可把 (a) 单独量出来。

    另外算两个相关系数：
      · `rho(shape, p2p)` —— 剖面与窗质量的耦合（L1-14 当年是 -0.801）；
      · `偏rho(shape, 序 | p2p)` —— 扣掉 p2p 后还剩多少寿命漂移。
        若它塌到接近 0，说明「漂移」其实是 p2p 变化的影子。
    """
    print('%-7s %5s | %s' % ('组', '窗数', '全窗（固定基线的 shape_full）'))
    print('%-7s %5s | %8s | %13s | %s' % ('', '', 'rho(序)', 'rho(shape,p2p)',
                                          '偏rho(序|p2p)'))
    for p in _paths():
        prof, _t, p2p = load_prof(p)
        if prof is None or p2p is None or len(p2p) < 200:
            continue
        gid = _gid(p)
        n = len(p2p)
        life = np.arange(n, dtype=float)
        shape_full = _rob25(prof)                 # 固定基线，与子集无关
        ref = float(np.median(p2p))
        mA = p2p > rel * ref
        mB = np.abs(p2p - ref) < tol * ref
        r_ord = _rho(shape_full)
        r_sp = float(spearmanr(shape_full, p2p).statistic)
        r_par = _partial_rank_shuffle(shape_full, life, p2p)
        print('%-7s %5d | %8.3f | %13.3f | %+.3f' % (gid, n, r_ord, r_sp, r_par))

        def row(m, tag):
            k = int(m.sum())
            if k < 30:
                return '  %s: n<30（%d）跳过' % (tag, k)
            rd = _rho(_rob25(prof[m]))            # 交付口径：子集重算基线 + 子集序号
            rf = _rho(shape_full[m])              # 固定基线（内容效应）
            sp = float(spearmanr(shape_full[m], p2p[m]).statistic)
            return ('  %s: rho交付 %6.3f | rho固定基线 %6.3f | rho(shape,p2p) %+.3f'
                    '（n=%d）' % (tag, rd, rf, sp, k))
        print(row(mA, 'A'))
        print(row(mB, 'B'))
    return None


def qc_causal(rel=QC_REL, win=200):
    """窗质量控制 A 的门槛能不能**在线化**（把「全寿命中位」换成因果中位）。

    现状（`qc_scan` 的 A）：`p2p > rel * median(p2p 全寿命)` —— **非因果**，
    用了未来的数据，所以不能声称在线可用。

    两种因果替代（都只用当前及过去）：
      · **expanding**：门槛 = rel * median(p2p[0..i])
      · **sliding**  ：门槛 = rel * median(p2p[i−W..i])，W = `win`
    暖机段（历史不足 W 个窗）没有依据判窗质量，**一律保留** ——
    这与交付口径一致（`shape_rob25` 的基线本来就取自前 25%）。

    判读：若换成两种因果门槛后 `rho(shape_rob25, 序)` 与全寿命门槛基本一致，
    则 A 可以改成在线形式；若明显变差，则 A 只能当离线诊断。
    """
    print('%-7s %5s %8s | %-22s | %-22s | %s' % (
        '组', '窗数', 'rho全窗', 'A 全寿命门槛（现口径）',
        'A expanding 因果', 'A sliding 因果(W=%d)' % win))
    rows = []
    for p in _paths():
        prof, _t, p2p = load_prof(p)
        if prof is None or p2p is None or len(p2p) < 200:
            continue
        gid = _gid(p)
        n = len(p2p)
        s = pd.Series(p2p)
        thr_g = rel * float(np.median(p2p))
        thr_e = (rel * s.expanding(min_periods=win).median()).to_numpy()
        thr_s = (rel * s.rolling(win, min_periods=win).median()).to_numpy()

        def rho_of(thr):
            m = np.where(np.isnan(thr), True, p2p > thr)
            keep = int(m.sum())
            if keep < 30:
                return float('nan'), keep
            return _rho(_rob25(prof[m])), keep
        r0, k0 = rho_of(np.full(n, thr_g))
        r1, k1 = rho_of(thr_e)
        r2, k2 = rho_of(thr_s)
        rows.append((gid, n, r0, r1, k1, r2, k2))
        print('%-7s %5d %8.3f | %7.3f (n=%4d)    | %7.3f (n=%4d)   | %7.3f (n=%4d)'
              % (gid, n, r0, r0, k0, r1, k1, r2, k2))
    dr1 = [r[3] - r[2] for r in rows if np.isfinite(r[3])]
    dr2 = [r[5] - r[2] for r in rows if np.isfinite(r[5])]
    print('|Δrho| 中位：expanding %.3f   sliding %.3f   最大 %.3f / %.3f'
          % (float(np.median(np.abs(dr1))), float(np.median(np.abs(dr2))),
             float(np.max(np.abs(dr1))), float(np.max(np.abs(dr2)))))
    return rows


def rank_check(warm=WARM):
    """复核 §22 的推论会不会影响**现有的指标排名**（待办 1.5）。

    §22 发现「在子集上重算基线」会让 rho 大幅漂移。本开关要回答两件事：

      1. **生产路径用的到底是哪一种？** 不靠读代码，而用**独立复算**核验：
         直接按「全窗前 25% 中位当基线、再取暖机后的序列」算一遍，
         与 `analyze()` 产出的暖机后 rho 比 —— 相等即证明生产是固定基线。
      2. **若当初用的是子集重算版，排名会变多少？** 即 §22 的影响面。
         做法：把剖面先截到暖机后，再在**截断后的矩阵上**重算基线（`variants(prof[k0:])`）。

    输出每组的两个 rho（`l1_first` 与 `l1_ref25`），以及按「暖机后 |rho| 中位」
    排序的前 3 名在两种口径下是否一致。
    """
    data = _collect()
    if not data:
        print('没有可用的 _l1fbgprof_*.npz')
        return None
    gids = list(data)
    print('%-7s %5s | %-19s | %-19s | %s' % (
        '组', '窗数', 'l1_ref25 固定基线', 'l1_ref25 子集重算', 'l1_first 固定/子集'))
    rows = []
    for g in gids:
        prof, _t, _p2p, v, _e = data[g]
        n = prof.shape[0]
        k0 = max(1, int(round(n * warm)))
        if n - k0 < 8:
            continue
        sub = prof[k0:]
        # ① 固定基线（独立复算，不经 variants()）
        ref25_full = np.median(prof[:max(1, int(round(n * 0.25)))], axis=0)
        s25_fix = np.abs(sub - ref25_full).sum(axis=1)
        r25_fix = float(spearmanr(s25_fix, np.arange(len(sub))).statistic)
        s1_fix = np.abs(sub - prof[0]).sum(axis=1)
        r1_fix = float(spearmanr(s1_fix, np.arange(len(sub))).statistic)
        # ② 子集重算基线（截断后重新取前 25% 中位 / 首窗）
        ref25_sub = np.median(sub[:max(1, int(round(len(sub) * 0.25)))], axis=0)
        r25_sub = float(spearmanr(np.abs(sub - ref25_sub).sum(axis=1),
                                  np.arange(len(sub))).statistic)
        r1_sub = float(spearmanr(np.abs(sub - sub[0]).sum(axis=1),
                                 np.arange(len(sub))).statistic)
        # ③ 与生产路径的暖机后值对照（核验 ① 是否就是生产口径）
        r25_prod = rho_of(v['l1_ref25'][k0:])
        rows.append({'gid': g, 'n': n, 'r25_fix': r25_fix, 'r25_sub': r25_sub,
                     'r1_fix': r1_fix, 'r1_sub': r1_sub, 'r25_prod': r25_prod})
        print('%-7s %5d | %7.3f (生产 %6.3f) | %7.3f (差 %+.3f) | %+.3f / %+.3f'
              % (g, n, r25_fix, r25_prod, r25_sub, r25_sub - r25_fix,
                 r1_fix, r1_sub))
    d_prod = np.abs([r['r25_fix'] - r['r25_prod'] for r in rows])
    d_sub = np.abs([r['r25_sub'] - r['r25_fix'] for r in rows])
    print('\n核验：独立复算的固定基线 vs 生产暖机后 rho —— 最大差 **%.2e**'
          % (float(d_prod.max()) if d_prod.size else float('nan')))
    print('      => %s' % ('生产路径就是固定基线口径，排名不受 §22 影响'
                          if float(d_prod.max()) < 1e-9 else
                          '⚠️ 不一致，需查生产线'))
    print('若改子集重算：|Δrho| 中位 %.3f、最大 %.3f（共 %d 组）'
          % (float(np.median(d_sub)), float(d_sub.max()), len(d_sub)))

    # 排名（因果口径：暖机后 |rho| 中位）
    cand = [k for k in VARIANTS if k in CAUSAL and k not in DEGENERATE]
    post = {k: np.array([rho_of(data[g][3][k][max(1, int(round(data[g][0].shape[0] * warm))):])
                         for g in gids], dtype=float) for k in cand}
    rank_fix = sorted(cand, key=lambda k: -float(np.nanmedian(np.abs(post[k]))))
    print('\n暖机后 |rho| 中位排名（生产/固定基线，前 5）：')
    for k in rank_fix[:5]:
        print('    %-12s %6.3f' % (k, float(np.nanmedian(np.abs(post[k])))))
    return rows, rank_fix


BATCH_A = ['L1-06', 'L1-13', 'L1-14', 'L1-24']      # L1 变幅VA+FBG，连续长记录型 FBG


def _acf1(y):
    """滞后 1 自相关（Pearson，用于估有效样本量）。"""
    y = np.asarray(y, dtype=float)
    if y.size < 4 or np.ptp(y) == 0:
        return 0.0
    a, b = y[:-1], y[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _block_perm_rho(shape, n_perm=400, block=20, seed=7):
    """循环分块置换的零分布 —— 保留自相关、只打乱顺序。

    ⚠️ 为什么不能用「随机子样本」当零分布：分组 A 是**连续长记录**上的 28 s 切片，
    相邻窗强自相关，随机打散会破坏这个结构，零分布偏窄 ⇒ p 值偏小。
    循环分块置换把序列按 block 长度切块再随机拼接，**保留了块内结构**。
    """
    rng = np.random.default_rng(seed)
    n = shape.size
    if n < 4 * block:
        block = max(2, n // 4)
    nb = int(np.ceil(n / block))
    comp = np.empty(n)
    out = np.empty(n_perm)
    ar = np.arange(n)
    for i in range(n_perm):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        comp[:] = shape[idx[:n]]
        out[i] = spearmanr(comp, ar).statistic
    return out


def batch_review(warm=WARM, n_perm=400, block=20):
    """按 FBG 记录格式分批评估 `shape_rob25` 的 rho（待办 1.1）。

    动机：分组 A（L1 变幅VA+FBG，L1-06/13/14/24）是**连续长记录**型 FBG ——
    相邻窗是同一条记录上的 28 s 切片，强自相关；
    分组 B（L1 谱载+FBG）是**每 7 分钟 20 秒快照** —— 窗之间近乎独立。
    同一个 rho 在两组的**有效样本量**相差很多，
    所以直接比较「|rho| 中位」或「反例数」并不对等。

    本诊断给出：每组窗数、相邻窗时距中位、剖面序列的滞后 1 自相关、
    有效样本量 n_eff = n(1-a1)/(1+a1)，以及
      · rho —— 固定基线（全窗 `l1_ref25`）口径的暖机后 rho；
      · p_block —— 循环分块置换零分布下 |rho| 的 p 值（保留自相关）。
    再按分组汇总，回答「分组 A 的 4 组是否撑得住同一个结论」。
    """
    data = _collect()
    if not data:
        print('没有可用的 _l1fbgprof_*.npz')
        return None
    print('%-7s %4s %6s %9s %8s %8s | %8s %8s' % (
        '组', '批', '窗数', '窗距中位s', 'acf1', 'n_eff', 'rho', 'p_block'))
    rows = []
    for g in data:
        prof, t, _p2p, v, _e = data[g]
        n = prof.shape[0]
        k0 = max(1, int(round(n * warm)))
        if n - k0 < 8:
            continue
        shape_full = v['l1_ref25']                 # 固定基线，与子集无关
        a1 = _acf1(shape_full)
        # ⚠️ AR(1) 的 n_eff 公式在 acf1 -> 1 时退化（L1-44 会算出 3712 窗 -> n_eff 8）
        # ⇒ 只在 acf1 < 0.95 时当参考值用，否则标 * 并从汇总里排除。
        ok_eff = a1 < 0.95
        neff = (n * max(1.0 - a1, 1e-3) / max(1.0 + a1, 1e-3)) if ok_eff else float('nan')
        sub = shape_full[k0:]
        rho = float(spearmanr(sub, np.arange(sub.size)).statistic)
        null = _block_perm_rho(sub, n_perm=n_perm, block=block)
        p = float((np.abs(null) >= abs(rho)).mean())
        dt = float(np.median(np.diff(t))) if (t is not None and t.size == n) else float('nan')
        b = 'A' if g in BATCH_A else 'B'
        rows.append((g, b, n, dt, a1, neff, rho, p))
        print('%-7s %4s %6d %9.1f %8.3f %8s | %8.3f %8.3f'
              % (g, b, n, dt, a1, ('%.0f' % neff) if ok_eff else '退化*', rho, p))

    print('\n按分组汇总（固定基线口径）')
    print('%-4s %4s %10s %10s %16s %10s %11s' % (
        '批', '组数', '|rho|中位', 'rho 中位', '反例(rho<-0.3)', 'acf1中位', 'n_eff中位'))
    for b in ('A', 'B'):
        rs = [r for r in rows if r[1] == b]
        if not rs:
            continue
        ar = np.abs([r[6] for r in rs])
        neg = sum(1 for r in rs if r[6] < -0.3)
        ne = [r[5] for r in rs if np.isfinite(r[5])]
        print('%-4s %4d %10.3f %10.3f %16d %10.3f %11s' % (
            b, len(rs), float(np.median(ar)), float(np.median([r[6] for r in rs])),
            neg, float(np.median([r[4] for r in rs])),
            ('%.0f（%d/%d 组可用）' % (float(np.median(ne)), len(ne), len(rs)))
            if ne else '全部退化*'))
    print('  * acf1 >= 0.95 时 AR(1) 的 n_eff 公式退化，不给值。')
    pa = [r[7] for r in rows if r[1] == 'A']
    pb = [r[7] for r in rows if r[1] == 'B']
    print('\np_block < 0.05 的组数：批 A %d/%d，批 B %d/%d'
          % (sum(1 for x in pa if x < 0.05), len(pa),
             sum(1 for x in pb if x < 0.05), len(pb)))
    print('窗距中位（s）：A %s；B %s'
          % ('/'.join('%.0f' % r[3] for r in rows if r[1] == 'A'),
             '/'.join('%.0f' % r[3] for r in rows if r[1] == 'B')))
    return rows


def variants_main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None, help='单组诊断，如 L1-31')
    ap.add_argument('--warm', type=float, default=WARM, help='暖机比例，默认 0.25')
    ap.add_argument('--qc', action='store_true',
                    help='窗质量控制 / 载荷级匹配诊断（L1-14 查因）')
    ap.add_argument('--b-diag', dest='b_diag', action='store_true',
                    help='B 口径（载荷级匹配）到底选了寿命的哪一段')
    ap.add_argument('--b-why', dest='b_why', action='store_true',
                    help='拆开 B 口径 rho 变化：子集内容 vs 重算基线 vs 序号归一')
    ap.add_argument('--qc-causal', dest='qc_causal', action='store_true',
                    help='窗质量控制门槛能否在线化（全寿命中位 vs 因果中位）')
    ap.add_argument('--rank-check', dest='rank_check', action='store_true',
                    help='核验指标排名用的是固定基线还是子集重算（待办 1.5）')
    ap.add_argument('--batch', dest='batch', action='store_true',
                    help='按 FBG 记录格式分批评估 rho（带自相关校正的 p 值）')
    ap.add_argument('--qc-win', type=int, default=200,
                    help='因果门槛的滑动/暖机窗长（窗数），默认 200')
    ap.add_argument('--lm-tol', type=float, default=LM_TOL,
                    help='载荷级匹配带宽（相对中位 p2p），默认 0.10')
    ap.add_argument('--qc-rel', type=float, default=QC_REL,
                    help='窗质量控制门槛（相对中位 p2p），默认 0.80')
    a = ap.parse_args(argv)
    if a.dump:
        dump(a.dump)
    elif a.batch:
        batch_review(warm=a.warm)
    elif a.rank_check:
        rank_check(warm=a.warm)
    elif a.qc_causal:
        qc_causal(rel=a.qc_rel, win=a.qc_win)
    elif a.b_why:
        b_why(tol=a.lm_tol, rel=a.qc_rel)
    elif a.b_diag:
        b_diag(tol=a.lm_tol, rel=a.qc_rel)
    elif a.qc:
        qc_scan(tol=a.lm_tol)
    else:
        analyze(warm=a.warm)

# ==================================================================
# 子命令分发
# ==================================================================
USAGE = """L1 的 FBG 工具链 —— 选一个子命令：

  profile    扫 FBG 原始文件 → results/_l1fbgprof_{gid}.npz + l1_fbgprof_rank.csv
             python l1/fbg_tools.py profile [L1-29 L1-41 ...]
  coverage   文件级覆盖体检 + 空隙归因（快，不读数据内容）
             python l1/fbg_tools.py coverage [--load-h L1-31 L1-29] [--root DIR]
  variants   10 个口径横向对比 + 单组诊断
             python l1/fbg_tools.py variants [--dump L1-31 | --qc | --b-diag | --b-why |
                                              --qc-causal | --rank-check | --batch]

每个子命令的完整参数：python l1/fbg_tools.py <子命令> -h
"""

PROFILE_USAGE = """用法: python l1/fbg_tools.py profile [组号 ...]

  不带组号 = 处理全部「FBG 目录下有 .txt」的组。
  产物: results/_l1fbgprof_{gid}.npz 与 results/l1_fbgprof_rank.csv
        （字段: t / p2p_med / p2p_ch / shape_l1 / shape_rob10 / shape_rob25 /
                 centroid / argmax / asym / prof）
"""

_SUBS = {
    'profile': profile_main,
    'coverage': coverage_main,
    'variants': variants_main,
}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help'):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    fn = _SUBS.get(cmd)
    if fn is None:
        print(f'未知子命令: {cmd}')
        print(USAGE)
        return 2
    if cmd == 'profile' and any(x in ('-h', '--help') for x in rest):
        print(PROFILE_USAGE)          # profile 的子参数是「组号」，没有 argparse
        return 0
    r = fn(rest)
    return 0 if r is None else r


if __name__ == '__main__':
    sys.exit(main())
