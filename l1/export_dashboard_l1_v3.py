# -*- coding: utf-8 -*-
"""L1 第三批（新 14 组：C3 变幅 + C4 谱载）→ 在线监测看板数据包。

与第一/二批的差异
------------------
| 项           | 第一批 L1-03/04/05/09        | 第二批 L1-49..60          | **第三批（本批）** |
| ------------ | --------------------------- | ------------------------- | ------------------ |
| AE 格式      | Vallen `.pridb`             | Vallen `.pridb`           | **PAC/Mistras `.DTA`** |
| 应变模态     | FBG 块级                    | DFOS 段级                 | **FBG(sm130)，无 DFOS** |
| 时间锚       | FBG 块 (5000 cycle)         | DFOS 测段 / markers       | **`ae_cycle` 循环轴** |
| 主曲线       | D(t)                        | 离线复评 HI_AE            | **离线复评 HI_hit** |
| 在线指标     | 无                          | 无                        | **FBG `shape_rob25`（新增）** |

主曲线（离线复评，非因果）
--------------------------
    HI_hit = unity01(累积 AE 事件数)        横轴 = `ae_cycle` 的 cycle

⚠️ `unity01` 需要全寿命最大值 ⇒ **非因果**，看板已标注「离线复评」。

因果证据通道（供观察，不作阈值报警）
------------------------------------
    eae = unity01(最近 5000 cycle 窗内的 AE 事件数)     ← AE 活动度
    est = unity01(FBG `shape_rob25`)                     ← **在线可用的应变剖面漂移**
          `shape_rob25` = 10 通道应变剖面（按窗归一化）与
          「前 25% 加载窗中位剖面」的 L1 距离；基线全部取自过去数据 ⇒ 可在线。
          实测 12 组：全窗与暖机后均 10 正 / 0 负 / 2 弱，|rho| 中位 0.71 至 0.73。

数据来源（全部为本地缓存的 npz，不重新解析原始数据）
---------------------------------------------------
    results/_l1ae_frames_{gid}.npz    每帧 AE 统计（600 s 帧）
    results/_l1cyc_{gid}.npz          frame / cycle / g_load / f_hz（口径 holdgap）
    results/_l1fbgprof_{gid}.npz      t / p2p_med / p2p_ch(10) / shape_rob25
    results/l1_ae_hi.csv              t50 / t85
    results/l1_burst_rank.csv         Fano 簇状性

输出
----
    dashboard/data/{gid}.js           window.SHM_DATA['{gid}']
    dashboard/data/index_l1v3.js      window.SHM_DATASETS['l1v3']

⚠️ 前端约定（照 dashboard.js 实现）：
  · 单组包必须挂在 **window.SHM_DATA[gid]**（不是 SHM_PKG），
    因为 loadGroup() 读的是 window.SHM_DATA[gid]；
  · 数据集条目的 groups 必须是**对象数组**（至少含 gid），
    因为 fillGroupSel() 用 g.gid。

用法
----
    python l1/export_dashboard_l1_v3.py                 # 全部可用组
    python l1/export_dashboard_l1_v3.py L1-29 L1-41     # 指定组
    python l1/export_dashboard_l1_v3.py --out <dir>     # 自定义输出目录
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')

# 新 14 组（C3 变幅 + C4 谱载）
GROUPS = ['L1-06', 'L1-13', 'L1-14', 'L1-24', 'L1-25', 'L1-27', 'L1-29',
          'L1-30', 'L1-31', 'L1-34', 'L1-35', 'L1-36', 'L1-41', 'L1-44']

MIN_FRAMES = 20               # 帧数太少的组不出版本包（L1-34=2、L1-36=5、L1-30=13）
AE_EMPTY = -99999             # 前端约定的 ael 空值标记
HIT_WIN_CYC = 5000.0          # eae 的滚动窗（cycle）
LEVELS = (0.25, 0.55, 0.85)
FO_COLS = ['R1', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5']
FBG_WINDOW_S = 14.0           # 一个 FBG 统计窗的时长（140 行 x 0.1 s）
FBG_BURST_GAP_S = 60.0        # 相邻窗前隔超过它算两个突发


# --------------------------------------------------------------- 工具
def _i1000(x):
    if x is None or not np.isfinite(x):
        return 0
    return int(round(float(x) * 1000))


def _i100(x):
    if x is None or not np.isfinite(x):
        return 0
    return int(round(float(x) * 100))


def unity01(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def levels_from(d):
    d = np.asarray(d, dtype=float)
    lv = np.zeros(len(d), dtype=int)
    for i, th in enumerate(LEVELS):
        lv[d >= th] = i + 1
    return lv


def roll_sum(x, win):
    """滚动求和（含当前位置），窗宽 win 个样本。"""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(len(x))
    for i in range(len(x)):
        j = max(0, i - win + 1)
        out[i] = c[i + 1] - c[j]
    return out


def zoh_to(target_t, src_t, src_v):
    """把 (src_t, src_v) 零阶保持到 target_t（取最近样本）。"""
    if len(src_t) == 0:
        return np.zeros(len(target_t))
    idx = np.searchsorted(src_t, target_t)
    left = np.clip(idx - 1, 0, len(src_t) - 1)
    right = np.clip(idx, 0, len(src_t) - 1)
    use_left = np.abs(target_t - src_t[left]) <= np.abs(target_t - src_t[right])
    return np.where(use_left, src_v[left], src_v[right])


def ffill(a):
    """前向填充 NaN（首部则后向填），避免前端画线时出现断点。"""
    a = np.asarray(a, dtype=float).copy()
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(ok)
    out[:first] = a[first]
    return out


FANO_WIN = 60          # 60 s 帧 x 60 = 1 h（与 ae_burst.py 同口径）


def fano_series(gid, target_t):
    """从 60 s 细帧表算 Fano 簇状性，ZOH 到 600 s 帧网格。

    Fano = 滑动窗内 var(n) / mean(n)，窗宽 60 个 60 s 帧 = 1 h。
    与 `l1/ae_burst.py` 完全同口径，避免两处定义不一致。
    """
    p = os.path.join(RES, '_l1ae_frames60s_%s.npz' % gid)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        t60 = z['frame'].astype(float) * 60.0
        x = np.asarray(z['n'], dtype=float)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    N = len(x)
    fano = np.full(N, np.nan)
    for i in range(FANO_WIN, N + 1):
        s1 = c1[i] - c1[i - FANO_WIN]
        s2 = c2[i] - c2[i - FANO_WIN]
        mu = s1 / FANO_WIN
        if mu > 0:
            fano[i - 1] = (s2 / FANO_WIN - mu * mu) / mu
    ok = np.isfinite(fano)
    if ok.sum() < 10:
        return None
    return zoh_to(target_t, t60[ok], fano[ok])


# --------------------------------------------------------------- 单组打包
def load_group(gid):
    p_ae = os.path.join(RES, '_l1ae_frames_%s.npz' % gid)
    p_cy = os.path.join(RES, '_l1cyc_%s.npz' % gid)
    if not (os.path.exists(p_ae) and os.path.exists(p_cy)):
        return None
    with np.load(p_ae, allow_pickle=True) as z:
        ae = {k: z[k] for k in z.files}
    with np.load(p_cy, allow_pickle=True) as z:
        cy = {k: z[k] for k in z.files}
    fb = None
    p_fb = os.path.join(RES, '_l1fbgprof_%s.npz' % gid)
    if os.path.exists(p_fb):
        with np.load(p_fb, allow_pickle=True) as z:
            fb = {k: z[k] for k in z.files}
    return ae, cy, fb


def build_pkg(gid):
    got = load_group(gid)
    if got is None:
        print(f'  [{gid}] 缺 AE 帧表或循环轴 → 跳过')
        return None
    ae, cy, fb = got

    nfr = len(np.asarray(cy['cycle'], dtype=float))
    if nfr < MIN_FRAMES:
        print(f'  [{gid}] 仅 {nfr} 帧（< {MIN_FRAMES}）→ 跳过')
        return None

    cyc = np.asarray(cy['cycle'], dtype=float)          # 横轴：cycle
    g_load = np.asarray(cy['g_load'], dtype=float)
    nf = float(cy['n_f']) if cy.get('n_f') is not None else float(cyc[-1])
    f_hz = float(cy['f_hz'])
    t_rel = np.asarray(cy['t_epoch'], dtype=float)      # 帧 x 600 s（绝对 epoch 近似）
    aen = np.asarray(ae['n'], dtype=float)

    if not (len(aen) == nfr == len(t_rel)):
        # 极点情况：帧表与循环轴不齐（理论上不会），按较短者截断
        m = min(len(aen), nfr, len(t_rel))
        print(f'  [{gid}] 帧表与循环轴长度不一致（{len(aen)}/{nfr}/{len(t_rel)}）'
              f'→ 截断到 {m}')
        nfr = m
        aen, cyc, t_rel, g_load = aen[:m], cyc[:m], t_rel[:m], g_load[:m]

    # --- 主曲线：离线复评 HI_hit（非因果）---
    D = unity01(np.cumsum(aen))
    risk = D.copy()
    lv_f = levels_from(D)

    # --- 因果证据 1：AE 活动度（最近 5000 cycle 窗内的事件数）---
    dcyc = np.diff(cyc, prepend=cyc[0])
    # 每帧代表的 cycle 数 -> 换算成"最近 5000 cycle 相当于多少帧"
    med_dc = float(np.median(dcyc[dcyc > 0])) if (dcyc > 0).any() else 1.0
    win_fr = max(1, int(round(HIT_WIN_CYC / max(med_dc, 1e-9))))
    eae_raw = roll_sum(aen, win_fr)
    eae = unity01(eae_raw)

    # --- 因果证据 2：FBG shape_rob25（在线可用）---
    # ⚠️ 归一化到 0~1 只是**显示需要**（前端按固定 0~1 轴画条），
    #    用到了全寿命极值，因而这一层缩放本身非因果；
    #    但 `shape_rob25` 指标本体是因果的（基线取自前 25% 加载窗）。
    #    原始量程记入 meta，便于反推物理值。
    est_raw = np.zeros(nfr)
    if fb is not None and 'prof' in fb and 'shape_rob25' in fb:
        ft = np.asarray(fb['t'], dtype=float)
        s25 = np.asarray(fb['shape_rob25'], dtype=float)
        p2p_med = np.asarray(fb['p2p_med'], dtype=float)
        p2p_ch = np.asarray(fb['p2p_ch'], dtype=float) \
            if 'p2p_ch' in fb else None
        st = zoh_to(t_rel, ft, p2p_med)
        est_raw = zoh_to(t_rel, ft, s25)
        est = unity01(est_raw)
        if p2p_ch is not None and p2p_ch.ndim == 2 and p2p_ch.shape[1] == len(FO_COLS):
            fo = {FO_COLS[k]: zoh_to(t_rel, ft, p2p_ch[:, k])
                  for k in range(len(FO_COLS))}
        else:
            fo = {}
        n_win = len(ft)
        fbg_span = (ft[-1] - ft[0]) / 3600.0 if len(ft) > 1 else 0.0
    else:
        st = np.zeros(nfr)
        est = np.zeros(nfr)
        fo = {}
        n_win, fbg_span = 0, 0.0

    # --- AE 幅值纵轴（dB）---
    amp = np.asarray(ae['amp_p90'], dtype=float) if 'amp_p90' in ae else None
    if amp is None or len(amp) != nfr:
        ael = np.full(nfr, AE_EMPTY, dtype=np.int64)
    else:
        ael = np.array([_i1000(a) if np.isfinite(a) and a > 0 else AE_EMPTY
                        for a in amp], dtype=np.int64)
    fin = ael[ael != AE_EMPTY]
    if fin.size:
        ax_ael = [round(float(fin.min()) / 1000.0 - 0.3, 2),
                  round(float(fin.max()) / 1000.0 + 0.3, 2)]
    else:
        ax_ael = [-1.0, 1.0]

    # --- 证据通道声明（前端 EVIDENCE 面板由它渲染，未声明则回落原三行）---
    # 三个通道各自分工：AE 活动度（量级）/ FBG 剖面漂移（独立模态，在线）/
    # Fano 簇状性（无量纲、看涨落结构，与上面两个机理独立）
    indics = [
        {'key': 'e_ae', 'name': 'e_ae AE 活动度（最近 5000 cycle）',
         'color': 'rgba(255,208,138,.9)',
         'series': [_i1000(x) for x in eae]},
        {'key': 'e_st', 'name': 'e_st FBG 剖面漂移 shape_rob25（在线）',
         'color': 'rgba(0,227,154,.85)',
         'series': [_i1000(x) for x in est]},
    ]
    fano_raw = fano_series(gid, t_rel)
    if fano_raw is not None:
        fano_raw = ffill(fano_raw)
        indics.append({'key': 'e_fano', 'name': 'e_fano Fano 簇状性（1 h 窗 var/mean）',
                       'color': 'rgba(176,124,255,.9)',
                       'series': [_i1000(x) for x in unity01(fano_raw)]})

    # --- 摘要 ---
    def first_ge(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    win_hits = int(round(float(np.sum(aen))))
    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge(0.25), 't55': first_ge(0.55), 't85': first_ge(0.85),
        'b2': None,          # 无弱标签 → 前端自动隐藏
        'b3': 100.0,         # 失效锚 = n_f
        'c0Pct': None,
        'refs': [],
        'aeEvents': win_hits,
        'nFo': len(fo),
        'nDfos': 0,          # 本批无 DFOS
        'f_hz': round(f_hz, 3),
        'load_h': round(float(cy['load_h']), 1),
        'span_h': round(float(cy['span_h']), 1),
        'fill': str(cy['fill']) if 'fill' in cy else '?',
        'fbgWin': int(n_win),
        'fbgSpanH': round(fbg_span, 1),
        # 原始量程：est/eae 显示用的 0~1 缩放是「全寿命极值归一」，
        # 记下量程便于反推物理值（指标本体仍是因果的）
        'estRawMin': round(float(np.min(est_raw)), 4),
        'estRawMax': round(float(np.max(est_raw)), 4),
        'eaeWinCyc': float(HIT_WIN_CYC),
        'eaeWinFrames': int(win_fr),
        'eaeRawMax': int(round(float(np.max(eae_raw)))),
        'nIndics': len(indics),
        'fanoRawMin': (round(float(np.min(fano_raw)), 1)
                       if fano_raw is not None else None),
        'fanoRawMax': (round(float(np.max(fano_raw)), 1)
                       if fano_raw is not None else None),
    }

    warn = []
    last = 0
    for f in range(nfr):
        lvv = int(lv_f[f])
        if lvv > last:
            for g in range(last + 1, lvv + 1):
                warn.append({'f': f, 't': round(float(cyc[f]), 1), 'lv': g})
            last = lvv

    chans = [
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流(%d s 帧)' % 600,
         'n': '2', 'unit': '通道'},
        {'key': 'fbg', 'name': '光纤光栅 FBG', 'mode': '突发式(每 420/240 s 采 20 s)',
         'n': str(len(fo)) if fo else '0', 'unit': '通道'},
        {'key': 'engine', 'name': '离线复评 HI_hit', 'mode': 'AE 累积·需全寿命归一',
         'n': None, 'unit': ''},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1v3', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳（%s）' % ('变幅 VA' if gid in
                                          ('L1-06', 'L1-13', 'L1-14', 'L1-24')
                                          else '谱载'),
        'mode': 'AE(.DTA) + FBG · 离线复评主曲线 + 在线应变剖面',
        'indexName': '离线复评 HI_hit（非在线）',
        'trendName': 'HI_hit 趋势 / 阈值 0.25 · 0.55 · 0.85',
        'labels': {'unit': 'HI_hit (0~1)', 'legend': 'HI_hit',
                   'margin': '1 − HI_hit', 'thrName': 'HI_hit',
                   'peak': '峰值 HI_hit',
                   'foot': 'HI_hit ∈ [0,1] · 阈值 0.25 / 0.55 / 0.85',
                   # 面板副标题覆盖：本批没有 DFOS，只有 FBG，
                   # 否则前端 isCycle() 会写成「分布式应变」（DFOS 措辞）
                   'ae': '声发射 · 事件率 / 峰值（600 s 帧）',
                   'fo': '光纤光栅 · 10 通道（应变峰峰值）',
                   'st': '光纤光栅应变 · 峰峰值 / 波动'},
        'n': int(nfr), 'nfr': int(nfr), 'step': 1,
        'frameDt': float(med_dc), 'dt': float(med_dc),
        'dur': float(nf), 'c0': 0.0,
        'foCols': list(fo.keys()),
        'chans': chans,
        'indics': indics,
        'meta': meta, 'warn': warn,
        # --- 波形数组（长度均 = nfr）---
        'D': [_i1000(x) for x in D],
        'risk': [_i1000(x) for x in risk],
        'eae': [_i1000(x) for x in eae],
        'est': [_i1000(x) for x in est],
        'lv': [int(x) for x in lv_f],
        'st': [_i100(x) for x in st],
        'ael': [int(x) for x in ael],
        'aen': [int(round(x)) for x in aen],
        't': [round(float(x), 1) for x in cyc],
        'fo': {k: [_i100(x) for x in v] for k, v in fo.items()},
        'ax': {'ael': ax_ael,
               'rate': round(max(2.0, float(np.max(aen)) * 1.2), 1)},
    }
    return pkg


# --------------------------------------------------------------- 主流程
def write_js(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return os.path.getsize(path) / 1024.0


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(PROJ, 'dashboard', 'data'))
    a = ap.parse_args()
    groups = a.groups or GROUPS
    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    print('输出目录:', outdir)

    index = []
    for gid in groups:
        print(f'[{gid}]')
        pkg = build_pkg(gid)
        if pkg is None:
            continue
        js = ('/* 自动生成，勿手改：l1/export_dashboard_l1_v3.py */\n'
              'window.SHM_DATA=window.SHM_DATA||{};\n'
              'window.SHM_DATA[%s]=%s;\n'
              % (json.dumps(gid), json.dumps(pkg, ensure_ascii=False,
                                             separators=(',', ':'))))
        size = write_js(os.path.join(outdir, f'{gid}.js'), js)
        index.append({'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'],
                      'n': pkg['nfr'], 'unit': pkg['unit'],
                      'frameDt': pkg['frameDt'], 'meta': pkg['meta'],
                      'foCols': pkg['foCols'], 'warn': pkg['warn'],
                      'spec': {'kind': ('变幅 VA' if gid in
                                        ('L1-06', 'L1-13', 'L1-14', 'L1-24')
                                        else '谱载')}})
        m = pkg['meta']
        print('  %-7s 帧 %5d  n_f %9.0f  f %.3f Hz  FBG窗 %5d  通道 %2d  '
              't85 %s  包 %.0f KB'
              % (gid, pkg['nfr'], pkg['dur'], m['f_hz'], m['fbgWin'],
                 len(pkg['foCols']), m['t85'], size))

    ds = {'id': 'l1v3', 'name': 'L1 第三批 · 新 14 组（AE .DTA + FBG）',
          'unit': 'cycle', 'path': 'data/',
          'groups': index}
    with open(os.path.join(outdir, 'index_l1v3.js'), 'w', encoding='utf-8') as f:
        f.write('window.SHM_DATASETS=window.SHM_DATASETS||{};'
                'window.SHM_DATASETS["l1v3"]=%s;\n'
                % json.dumps(ds, ensure_ascii=False, separators=(',', ':')))
    print(f'\n共 {len(index)} 组 → index_l1v3.js')
    print('组：', ' '.join(g['gid'] for g in index))


if __name__ == '__main__':
    main()
