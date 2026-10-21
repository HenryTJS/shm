# -*- coding: utf-8 -*-
"""main 主流程统一入口（阶段① 数据准备 / 阶段② 评估出图 / 阶段③ 稳健性）

由 3 个脚本合并而来（2026-10-07）：`prepare_data.py` / `evaluate.py` / `robustness.py`。
合并方式：每段正文**逐字保留**，只给冲突的顶层名字加前缀 + 去掉各自的 `__main__` 块。
原脚本已归档于 `main/results/_logs/_orig_src/`。

用法（第二级 = 原脚本的子命令，可给多个或用 `all`）：

    python main/pipeline.py prepare  align              # 多源对齐 → main/aligned/
    python main/pipeline.py prepare  weaklabels         # 弱标签 b2/b3 → main/weak_labels/
    python main/pipeline.py evaluate degree             # D 达阈/单调 → main/results/
    python main/pipeline.py evaluate warning curves     # 预警 onset + D(t) 出图
    python main/pipeline.py evaluate paper              # 论文四联图（逐点重算，较慢）
    python main/pipeline.py robust   sens loso ablation stats
    python main/pipeline.py <段> all                    # 该段全部子命令

⚠️ 本模块在**导入时**就 `chdir(main/)`（与原脚本一致）。
   `evaluate` / `robust` 还接受 `--workers N`（默认 4 / 8）。
"""
import argparse
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根（含 shm）
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/

from shm.data_loader import DataLoader                                          # noqa: E402
from shm.streaming import StreamSimulator                                       # noqa: E402
from shm.damage_index import OnlineDamageIndex                                  # noqa: E402

# ==========================================================================
# 段 1/3  原 main/prepare_data.py（411 行，正文逐字保留；改名 main→prep_main, TASKS→PREP_TASKS）
#==========================================================================

# -*- coding: utf-8 -*-
"""阶段① 数据准备：多源对齐 + 弱标签（5 组主样本 016-020）

子命令（可组合，如 `python main/pipeline.py prepare align weaklabels` 或 `all`）：
  align       多源数据对齐(AE 网格) → aligned/
  weaklabels  弱标签生成 → weak_labels/
"""
import os, sys, argparse, warnings, traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根(含 shm)
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/

from shm.data_loader import DataLoader   # weaklabels 用

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\main
GROUPS = ['016', '017', '018', '019', '020']   # 主样本 5 组
AE_GROUPS = ['%03d' % i for i in range(15, 28)]   # 有 AE 的全部 13 组，供多通道标签覆盖

# ============================================================
# 子命令 align —— 多源对齐（AE 网格模式）
# ============================================================
OUT = os.path.join(ROOT, "aligned")   # 对齐输出：根目录 aligned\
OUT_D = OUT
os.makedirs(OUT_D, exist_ok=True)


def ts_to_seconds(t_series):
    """时间戳统一换算为秒（等长返回，保留 NaN）。

    采样率 10Hz：光纤/部分应变 dt≈1.0（0.1s 计数单位）需 ×0.1；应变 dt≈0.1 已是秒。
    按中位间隔自动判断（>=0.2s 视为 0.1s 计数单位）。
    """
    t = pd.to_numeric(t_series, errors='coerce').values.astype(float)
    valid = t[~np.isnan(t)]
    if len(valid) < 2:
        return t
    dt = float(np.median(np.diff(valid)))
    if dt >= 0.2:
        return t * 0.1
    return t


def detect_gaps(time_s, mult=1.5, min_gap_s=0.5):
    """检测时间轴缺失段。返回 [(start_s, end_s, lost_s), ...]。"""
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 3:
        return []
    d = np.diff(time_s)
    med = float(np.median(d))
    if med <= 0:
        return []
    gaps = []
    for i, g in enumerate(d):
        if g > mult * med and g >= min_gap_s:
            gaps.append((float(time_s[i]), float(time_s[i + 1]), float(g - med)))
    return gaps


def read_fiber(sid, sdir):
    """读光纤 → DataFrame[time_s, fiber_1..] 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "光纤" in f]
    if not files:
        return None, "无光纤文件"
    fp = os.path.join(sdir, files[0])
    skiprows = 0
    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc, skiprows=skiprows)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"
    cols = list(df.columns)
    if len(cols) < 2:
        return None, f"列数不足: {cols}"
    time_s = ts_to_seconds(df.iloc[:, 0])
    channels = {}
    for i, c in enumerate(cols[1:], 1):
        vals = pd.to_numeric(df[c], errors='coerce').values
        if np.sum(~np.isnan(vals)) > 10:
            channels[f"fiber_{i}"] = vals
    if not channels:
        return None, "无有效通道"
    result = pd.DataFrame({"time_s": time_s})
    for k, v in channels.items():
        result[k] = v
    result = result.dropna(subset=["time_s"])
    ch_cols = list(channels.keys())
    result = result.dropna(subset=ch_cols, how='all')
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_strain(sid, sdir):
    """读应变 → DataFrame[time_s, strain] 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "应变" in f]
    if not files:
        return None, "无应变文件"
    fp = os.path.join(sdir, files[0])
    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"
    if df.shape[1] < 2:
        return None, f"列数不足: {df.shape}"
    time_s = ts_to_seconds(df.iloc[:, 0])
    v = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    result = pd.DataFrame({"time_s": time_s, "strain": v})
    result = result.dropna()
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_ae(sid, sdir):
    """读声发射原始 DataFrame 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "声发射" in f]
    if not files:
        return None, "无声发射文件"
    fp = os.path.join(sdir, files[0])
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(fp, encoding=enc)
            return df, None
        except Exception:
            continue
    return None, "CSV编码无法识别"


def detect_oscillation(strain_df):
    """应变是否循环振荡：前 5000 点零交叉率 >30%。"""
    v = strain_df['strain'].values
    if len(v) < 100:
        return False
    sample = v[:min(len(v), 5000)]
    signs = np.sign(sample)
    signs[signs == 0] = 1
    zero_crossings = np.sum(signs[1:] != signs[:-1])
    return (zero_crossings / len(signs)) > 0.3


def align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T):
    """以声发射事件为网格对齐多源。AE 无时间戳：行号均匀映射 [0,T]。
    光纤/应变线性插值到 AE 事件时刻，仅在有效覆盖范围内（范围外 NaN，不外推）。"""
    if ae_df is None or len(ae_df) < 2:
        return None
    N = len(ae_df)
    ae_norm = np.linspace(0, 1, N)
    ae_t = np.linspace(t_start, t_start + T, N)
    out = pd.DataFrame({"ae_event": np.arange(N), "ae_time_s": ae_t, "ae_norm": ae_norm})
    for c in ae_df.columns:                       # AE 原始特征 100% 保留
        out[f"ae_{c}"] = pd.to_numeric(ae_df[c], errors='coerce').values
    if fb_df is not None and len(fb_df) > 1:      # 光纤插值
        fb_t = fb_df['time_s'].values.astype(float)
        fb_norm = (fb_t - t_start) / T
        for c in [c for c in fb_df.columns if c.startswith("fiber_")]:
            v = pd.to_numeric(fb_df[c], errors='coerce').values.astype(float)
            valid = ~np.isnan(v) & ~np.isnan(fb_norm)
            if valid.sum() > 1:
                interp = np.interp(ae_norm, fb_norm[valid], v[valid])
                lo, hi = float(np.min(fb_norm[valid])), float(np.max(fb_norm[valid]))
                interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
                out[c] = interp
        fb_cols = [c for c in out.columns if c.startswith("fiber_") and c != "fiber_mean"]
        if fb_cols:
            out["fiber_mean"] = out[fb_cols].mean(axis=1)
    if ys_df is not None and len(ys_df) > 1:      # 应变插值
        ys_t = ys_df['time_s'].values.astype(float)
        ys_v = ys_df['strain'].values.astype(float)
        ys_norm = (ys_t - t_start) / T
        valid = ~np.isnan(ys_v) & ~np.isnan(ys_norm)
        if valid.sum() > 1:
            interp = np.interp(ae_norm, ys_norm[valid], ys_v[valid])
            lo, hi = float(np.min(ys_norm[valid])), float(np.max(ys_norm[valid]))
            interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
            out["strain"] = interp
    return out


def process_specimen(sid):
    """处理单个试件 → (ae_grid, meta) 或 (None, error)。"""
    sdir = os.path.join(ROOT, sid)
    meta = {"试件": sid, "问题": []}
    fb_df, fb_err = read_fiber(sid, sdir)
    ys_df, ys_err = read_strain(sid, sdir)
    ae_df, ae_err = read_ae(sid, sdir)
    if fb_err: meta["问题"].append(f"光纤: {fb_err}")
    if ys_err: meta["问题"].append(f"应变: {ys_err}")
    if ae_err: meta["问题"].append(f"声发射: {ae_err}")
    if fb_df is None and ys_df is None:
        return None, "光纤和应变均缺失，无法确定时间轴"
    # 以应变为主时间基准（连续、覆盖完整实验）；无应变时用光纤
    if ys_df is not None and len(ys_df) > 0:
        t_start, t_end, time_ref = float(ys_df['time_s'].iloc[0]), float(ys_df['time_s'].iloc[-1]), "应变"
    elif fb_df is not None and len(fb_df) > 0:
        t_start, t_end, time_ref = float(fb_df['time_s'].iloc[0]), float(fb_df['time_s'].iloc[-1]), "光纤"
    else:
        return None, "光纤和应变均缺失，无法确定时间轴"
    T = t_end - t_start
    meta["实验时长_s"], meta["时间起点_s"], meta["时间基准"] = round(T, 1), round(t_start, 1), time_ref
    for tag, df in (("应变", ys_df), ("光纤", fb_df)):    # 丢数据检测
        if df is not None and len(df) > 1:
            gs = detect_gaps(df['time_s'].values)
            if gs:
                meta[f"{tag}缺失段"] = [f"{a:.1f}-{b:.1f}s(缺{c:.1f}s)" for a, b, c in gs]
                meta[f"{tag}缺失总时长_s"] = round(float(sum(c for _, _, c in gs)), 1)
    if fb_df is not None and len(fb_df) > 0 and ys_df is not None and len(ys_df) > 0:
        fb_dur = float(fb_df['time_s'].iloc[-1] - fb_df['time_s'].iloc[0])
        if fb_dur > T * 1.1:
            meta["问题"].append(f"光纤时长{fb_dur:.0f}s 明显长于应变{T:.0f}s，应变可能提前结束")
        elif fb_dur < T * 0.9:
            meta["问题"].append(f"光纤仅覆盖{fb_dur/T*100:.0f}%实验时长，可能中途中断")
    meta["光纤通道数"] = len([c for c in fb_df.columns if c.startswith("fiber_")]) if fb_df is not None else 0
    if ys_df is not None and len(ys_df) > 0:
        meta["应变振荡"] = detect_oscillation(ys_df)
    meta["声发射行数"] = len(ae_df) if ae_df is not None else 0
    ae_grid = align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T)
    if ae_grid is None:
        meta["问题"].append("声发射数据不足，无法生成AE网格对齐输出")
        return ae_grid, meta
    meta["输出列"] = list(ae_grid.columns)
    meta["数据完整率"] = round(ae_grid.notna().mean().mean(), 3)
    return ae_grid, meta


def cmd_align():
    all_meta, failed = [], []
    for sid in GROUPS:
        try:
            ae_grid, meta = process_specimen(sid)
            if ae_grid is None:
                print(f"[跳过] {sid}: {meta.get('问题', ['未知'])}")
                failed.append((sid, str(meta.get("问题", []))))
                all_meta.append(meta)
                continue
            ae_path = os.path.join(OUT_D, f"{sid}.csv")
            ae_grid.to_csv(ae_path, index=False, encoding="utf-8-sig", float_format="%.6f")
            all_meta.append(meta)
            n_issues = len(meta.get("问题", []))
            miss = meta.get("应变缺失总时长_s", 0) + meta.get("光纤缺失总时长_s", 0)
            print(f"[完成] {sid}: T={meta['实验时长_s']:.0f}s, AE={meta['声发射行数']}行, "
                  f"光纤{meta['光纤通道数']}ch, 问题={n_issues}, 缺失={miss}s")
        except Exception as e:
            traceback.print_exc()
            failed.append((sid, str(e)))
            print(f"[异常] {sid}: {e}")
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(os.path.join(OUT, "对齐元信息.csv"), index=False, encoding="utf-8-sig")
    print(f"\n===== 对齐完成（AE 网格模式） =====")
    print(f"成功: {len(all_meta) - len(failed)}/{len(GROUPS)}")
    print(f"失败/跳过: {len(failed)}")
    for sid, err in failed:
        print(f"  {sid}: {err}")
    print(f"输出目录: {OUT_D}")


# ============================================================
# 子命令 weaklabels —— 弱标签生成
# ============================================================
# 协议：b3=数据末端断裂(≈99%)；b2=损伤扩展(AE 累积能量最大加速或应变发散更早)；
#       b1=损伤萌生(累积 log 能量水平持久抬升；平台型无独立萌生段返回 None)。
def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def find_b2(logc, grid):
    """能量曲线最大正加速点(主要能量台阶)，限 30%~97% → life%"""
    d1 = np.gradient(logc, grid)
    d2 = np.gradient(d1, grid)
    mask = (grid > 0.30) & (grid < 0.97)
    if not mask.any():
        return None
    j = int(np.argmax(d2[mask]))
    return float(grid[mask][j] * 100.0)


def find_b1(logc, grid, b2):
    """b1(萌生)=累积能量水平相对前平台的【持久】抬升最早点。
    条件：post−pre≥0.4(≈2.5×能量) 且 tail−pre≥0.2；平台型无此抬升→None。"""
    if b2 is None:
        return None
    lo, hi = 0.12, (b2 - 1.0) / 100.0
    if hi <= lo:
        return None
    m = (grid >= lo) & (grid <= hi)
    idx = np.where(m)[0]
    d_log, d_tail = 0.4, 0.2
    for ik in idx:
        gk = float(grid[ik])
        i0 = int(np.searchsorted(grid, max(lo, gk - 0.05)))
        i1 = max(ik - 1, i0)
        pre = float(np.median(logc[i0:i1 + 1])) if i1 > i0 else float(logc[ik])
        j1 = int(np.searchsorted(grid, min(hi, gk + 0.05)))
        post = float(np.median(logc[ik:j1 + 1])) if j1 > ik else float(logc[ik])
        tail = float(np.percentile(logc[ik:], 50))
        if post - pre >= d_log and tail - pre >= d_tail:
            return gk * 100.0
    return None


# ------------------------------------------------------------
# 多通道弱标签 v2
# ------------------------------------------------------------
# 动机：v1 只用 AE 的 Peak 一列。Peak 取单次波形的极值，对离群点最敏感；
# 诊断脚本 main/probe_ae_channels.py 给出三条事实：
#   1) 25 列 PCA 有效维度中位 3.9，21 列是代数冗余（MS=RMS²、MSF=RMSF² 等）；
#   2) 幅度族内换一列，变点最多移动 75 个百分点（016: Peak 13.8 vs MeanSquare 88.2）；
#   3) 真正独立的通道为 幅度能量 / 冲激性 / 频谱质心 / 命中率，
#      命中率跟随加载排程（018 组 16.8% 就跳）而非损伤，予以排除。
# v2 协议：三条独立通道（有应变时加第四条）各出变点，取最大一致簇的中位数为 b2，
#          并回传置信级别；b3 = 块能量峰值位置，由假设 99.0 改为可测。
NB_BLOCK = 200      # 块数 → 0.5% 分辨率
SM_BLOCK = 21       # 块级平滑窗
TOL_BLOCK = 5.0     # 通道一致容差（life%）

CH_SPEC = [
    ('energy', 'MeanSquare', ['Peak']),                    # MeanSquare 已是幅度平方
    ('freq', 'RMSFrequency', ['MeanSquaredFrequency']),    # 频谱质心
    ('impuls', 'Kurtosis', ['PeakFactor', 'ImpulseFactor']),
]


def _pick_col(df, col, fallback):
    if col in df.columns:
        return col
    for c in fallback:
        if c in df.columns:
            return c
    return None


def _onset_blocks(y, lo=0.10, hi=0.95):
    """块级序列的 argmax|d²| 位置（life%）及其符号；含反向漂移检测。"""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return None, 0
    if not ok.all():
        y = np.interp(np.arange(y.size), np.where(ok)[0], y[ok])
    if y.std() < 1e-12:
        return None, 0
    ys = smooth(y - y.mean(), SM_BLOCK)
    d2 = np.gradient(np.gradient(ys))
    c = (np.arange(y.size) + 0.5) / y.size
    m = (c > lo) & (c < hi)
    if not m.any():
        return None, 0
    ii = np.where(m)[0]
    j = int(ii[int(np.argmax(np.abs(d2[ii])))])
    return float(c[j] * 100.0), int(np.sign(d2[j]))


def _consensus(vals, tol=TOL_BLOCK):
    """取成员最多、极差不超过 tol 的一致簇；返回 (簇中位, 成员数)。"""
    vs = sorted(v for v in vals if v is not None)
    if not vs:
        return None, 0
    best = []
    for v0 in vs:
        grp = [v for v in vs if v0 <= v <= v0 + tol]
        if len(grp) > len(best):
            best = grp
    if len(best) >= 2:
        return float(np.median(best)), len(best)
    return float(np.median(vs)), 1


def _load_ae_strain(gid):
    """弱标签只需声发射与应变，不读光纤（可省一半 I/O）。"""
    dl = DataLoader(gid)
    dl.load_strain()
    dl.load_ae()
    return dl


def run_weaklabel_multi(gid):
    """多通道共识弱标签；返回 dict（含置信标记）或 None。"""
    dl = _load_ae_strain(gid)
    ae = dl.ae
    if ae is None or len(ae) < 400:
        return None
    n = len(ae)
    idx = np.array_split(np.arange(n), NB_BLOCK)

    pos, sign, carrier = {}, {}, {}
    for name, col, fallback in CH_SPEC:
        use = _pick_col(ae, col, fallback)
        if use is None:
            continue
        if name == 'energy':
            ev = np.maximum(ae[use].to_numpy(float), 0.0)
            if use == 'Peak':
                ev = ev ** 2
            s = np.array([float(np.sum(ev[i])) if i.size else np.nan for i in idx])
            s = np.log10(np.maximum(s, 1e-12))
        else:
            v = ae[use].to_numpy(float)
            s = np.array([float(np.median(v[i][np.isfinite(v[i])]))
                          if i.size and np.isfinite(v[i]).any() else np.nan for i in idx])
        p, sg = _onset_blocks(s)
        pos[name], sign[name], carrier[name] = p, sg, use

    # 应变发散作为可选第四通道（与 v1 口径一致）
    strain_diverge = None
    if dl.strain is not None and 'strain' in dl.strain.columns and len(dl.strain) > 200:
        strain = dl.strain['strain'].values
        ns = len(strain)
        edges = np.linspace(0, ns, 200 + 1).astype(int)
        blk_std = np.array([float(np.std(strain[edges[i]:edges[i + 1]]))
                            if edges[i + 1] > edges[i] else 0.0 for i in range(200)])
        bpct = np.linspace(0, 100, 200)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk_std[mid])) if mid.any() else 0.0
        if base > 1e-9:
            over = np.where((bpct > 50) & (blk_std > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])
                pos['strain'], sign['strain'], carrier['strain'] = strain_diverge, 1, 'strain'

    vals = [v for v in pos.values() if v is not None]
    if not vals:
        return None
    b2, n_agree = _consensus(vals)
    n_ch = len(vals)
    conf = 'high' if n_agree == n_ch else ('mid' if n_agree >= 2 else 'low')

    # b3 = 块能量峰值位置（可测），不再固定为 99.0
    ecol = carrier.get('energy', 'MeanSquare')
    if ecol != 'strain' and ecol in ae.columns:
        ev = np.maximum(ae[ecol].to_numpy(float), 0.0)
        if ecol == 'Peak':
            ev = ev ** 2
    else:
        ev = np.maximum(ae['Peak'].to_numpy(float), 0.0) ** 2
    Eraw = np.array([float(np.sum(ev[i])) if i.size else 0.0 for i in idx])
    b3 = float((int(np.argmax(Eraw)) + 0.5) / NB_BLOCK * 100.0)
    b3_clamped = False
    if b3 <= b2 + 0.5:
        b3 = min(99.0, b2 + 0.5)
        b3_clamped = True

    # b1 沿用持久抬升判据，载体换成块级 log 能量
    logc = smooth(np.log10(np.maximum(Eraw, 1e-12)), SM_BLOCK)
    grid_b = (np.arange(NB_BLOCK) + 0.5) / NB_BLOCK
    b1 = find_b1(logc, grid_b, b2)

    n_grid = 1000
    life = np.arange(n_grid) * 0.1
    stage = np.zeros(n_grid, dtype=int)
    b1v = b1 if b1 is not None else (b2 if b2 is not None else 60.0)
    b2v = b2 if b2 is not None else (b3 * 0.8)
    stage[life >= b1v] = 1
    stage[life >= b2v] = 2
    stage[life >= b3] = 3
    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2v), 1),
                b3=round(float(b3), 1), n_ae=n, strain_diverge=strain_diverge,
                b1_none=bool(b1 is None), b3_clamped=bool(b3_clamped),
                conf=conf, n_agree=n_agree, n_ch=n_ch,
                o_energy=pos.get('energy'), o_freq=pos.get('freq'),
                o_impuls=pos.get('impuls'), o_strain=pos.get('strain'),
                col_energy=carrier.get('energy'), col_freq=carrier.get('freq'),
                col_impuls=carrier.get('impuls'), life=life, stage=stage)


# ------------------------------------------------------------
# v3：台阶口径
# ------------------------------------------------------------
# v2 用 argmax|d²| 找的是能量曲线**最大**的那一步，实际落在崩塌处，
# 于是 b2 与 b3 重合。诊断 main/probe_knee.py 显示曲线多为双台阶：
# 第一台阶在 66~77（高度 1.3~1.5 倍能量），崩塌在 90~94（1.6~2.9 倍）。
# v3 改成：
#   b3 = 台阶高度最大的位置（崩塌）
#   b2 = 崩塌之前台阶高度最大的位置，且高度不低于崩塌台阶的 STEP_REL
# 台阶高度用两侧各 STEP_W 宽的滑动中位差衡量，与 d² 无关，抗单点尖峰。
STEP_W = 0.05      # 台阶测量窗（占全程比例）
STEP_REL = 0.25    # 早台阶相对崩塌台阶的最低高度比
STEP_GAP = 5.0     # b2 与 b3 的最小间隔（life%）


def _smooth_edge(x, w):
    """边缘保持的滑动平均。smooth() 用 mode='same' 零填充，会在序列末段
    造出虚假的下降（台阶检测会把这个当成台阶），台阶口径改用它。"""
    x = np.asarray(x, float)
    if w < 3 or x.size < w:
        return x
    h = w // 2
    xp = np.pad(x, h, mode='edge')
    return np.convolve(xp, np.ones(w) / w, mode='valid')[:x.size]


def _sustained_shift(s, lo=3.0, hi=90.0, thresh=0.10, w_frac=0.05):
    """首个「持续电平抬升」位置（life%），找不到返回 None。

    判据：step(j) = median(s[j:j+w]) − median(s[j−w:j]) ≥ thresh，
    且抬升**保持住**——其后段电平相对前段仍高出 thresh。
    「保持住」这一条把一闪而过的瞬变排除掉，同时不必依赖高能命中的比例截断，
    因此像 016 那种稀疏早期事件（只占全部命中 0.06%）也能检出。
    """
    s = np.asarray(s, float)
    n = s.size
    w = max(int(round(w_frac * n)), 3)
    span = max(int(round(0.10 * n)), 3)
    c = (np.arange(n) + 0.5) / n * 100.0
    for j in range(w, n - w):
        if not (lo <= c[j] <= hi):
            continue
        if np.median(s[j:j + w]) - np.median(s[j - w:j]) < thresh:
            continue
        pre = s[max(0, j - span):j]
        tail = s[j:]
        if pre.size >= 3 and tail.size >= 3 and \
                np.median(tail) - np.median(pre) >= thresh:
            return float(c[j])
    return None


def _step_series(s):
    """step(j) = median(s[j:j+w]) - median(s[j-w:j])，w = STEP_W 段长。"""
    n = s.size
    w = max(int(round(STEP_W * n)), 3)
    st = np.full(n, np.nan)
    for j in range(w, n - w):
        st[j] = float(np.median(s[j:j + w]) - np.median(s[j - w:j]))
    return st


def _step_labels(s):
    """返回 (b2, b3)：b3 为最大台阶处，b2 为崩塌前最大台阶处（够高才取）。"""
    s = np.asarray(s, float)
    if not np.all(np.isfinite(s)):
        s = np.nan_to_num(s, nan=float(np.nanmedian(s)))
    if s.std() < 1e-12:
        return None, None
    st = np.abs(_step_series(s))
    c = (np.arange(s.size) + 0.5) / s.size * 100.0
    m = (c > 10.0) & (c < 95.0) & np.isfinite(st)
    if not m.any():
        return None, None
    ii = np.where(m)[0]
    j3 = int(ii[int(np.argmax(st[ii]))])
    b3 = float(c[j3])
    early = ii[c[ii] <= b3 - STEP_GAP]
    if early.size == 0 or st[j3] <= 0:
        return None, b3
    j2 = int(early[int(np.argmax(st[early]))])
    if st[j2] < STEP_REL * st[j3]:
        return None, b3
    return float(c[j2]), b3


def run_weaklabel_step(gid):
    """v3 台阶口径弱标签；返回 dict 或 None。"""
    dl = _load_ae_strain(gid)
    ae = dl.ae
    if ae is None or len(ae) < 400:
        return None
    n = len(ae)
    idx = np.array_split(np.arange(n), NB_BLOCK)

    b2s, b3s, carrier = {}, {}, {}
    for name, col, fallback in CH_SPEC:
        use = _pick_col(ae, col, fallback)
        if use is None:
            continue
        if name == 'energy':
            ev = np.maximum(ae[use].to_numpy(float), 0.0)
            if use == 'Peak':
                ev = ev ** 2
            s = np.array([float(np.sum(ev[i])) if i.size else np.nan for i in idx])
            s = np.log10(np.maximum(s, 1e-12))
        else:
            v = ae[use].to_numpy(float)
            s = np.array([float(np.median(v[i][np.isfinite(v[i])]))
                          if i.size and np.isfinite(v[i]).any() else np.nan for i in idx])
        s = _smooth_edge(s - float(np.nanmean(s)), SM_BLOCK)
        p2, p3 = _step_labels(s)
        b2s[name], b3s[name], carrier[name] = p2, p3, use

    strain_diverge = None
    if dl.strain is not None and 'strain' in dl.strain.columns and len(dl.strain) > 200:
        strain = dl.strain['strain'].values
        ns = len(strain)
        edges = np.linspace(0, ns, 200 + 1).astype(int)
        blk_std = np.array([float(np.std(strain[edges[i]:edges[i + 1]]))
                            if edges[i + 1] > edges[i] else 0.0 for i in range(200)])
        bpct = np.linspace(0, 100, 200)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk_std[mid])) if mid.any() else 0.0
        if base > 1e-9:
            over = np.where((bpct > 50) & (blk_std > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])

    v3 = [v for v in b3s.values() if v is not None]
    b3_none = not v3
    n_ch = len(v3)
    if v3:
        b3 = float(np.median(v3))
        n_agree = int(sum(abs(v - b3) <= TOL_BLOCK for v in v3))
    else:
        b3, n_agree = 99.0, 0
    conf = 'high' if (n_agree >= 3 and n_agree == n_ch) else ('mid' if n_agree >= 2 else 'low')
    # b2 只由能量通道定：能量是直接的损伤量，频谱与冲激性的最大台阶
    # 常落在崩塌处，混进来会把 b2 拉回 b3。
    b2 = b2s.get('energy')
    b2_none = b2 is None
    if b2_none:
        b2 = b3 * 0.8
    if b3 <= b2 + 0.5:
        b3 = min(99.0, b2 + 0.5)

    ecol = carrier.get('energy')
    if ecol is not None:
        ev = np.maximum(ae[ecol].to_numpy(float), 0.0)
        if ecol == 'Peak':
            ev = ev ** 2
    else:
        ev = np.maximum(ae['Peak'].to_numpy(float), 0.0) ** 2
    Eraw = np.array([float(np.sum(ev[i])) if i.size else 0.0 for i in idx])
    logc = _smooth_edge(np.log10(np.maximum(Eraw, 1e-12)), SM_BLOCK)
    grid_b = (np.arange(NB_BLOCK) + 0.5) / NB_BLOCK
    b1 = find_b1(logc, grid_b, b2)

    life = np.arange(1000) * 0.1
    stage = np.zeros(1000, dtype=int)
    b1v = b1 if b1 is not None else b2
    stage[life >= b1v] = 1
    stage[life >= b2] = 2
    stage[life >= b3] = 3
    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2), 1),
                b3=round(float(b3), 1), n_ae=n, strain_diverge=strain_diverge,
                conf=conf, n_agree=n_agree, n_ch=n_ch,
                b1_none=bool(b1 is None), b2_none=b2_none, b3_clamped=b3_none,
                o_energy=b2s.get('energy'), o_freq=b2s.get('freq'),
                o_impuls=b2s.get('impuls'), o_strain=strain_diverge,
                e3_energy=b3s.get('energy'), e3_freq=b3s.get('freq'),
                e3_impuls=b3s.get('impuls'),
                life=life, stage=stage)


# ------------------------------------------------------------
# v4：双簇登记
# ------------------------------------------------------------
# 早期 AE 事件有三种机制，处理方向互相矛盾，必须分开：
#   (a) 检测器劫持 —— 最大台阶选中早期簇；
#   (b) life 轴失真 —— 记录远长于破坏，100% 不是破坏点；
#   (c) 真实早期损伤 —— 断了又跑一段，该簇不能剔。
# v4 不再让簇去竞争一个 b2，而是**按簇登记阶段**：
#   b1 = 第一个高能簇（首次损伤）
#   b2 = 最后一个高能簇的起点（末端事件启动）
#   b3 = 最后一个高能簇的中心（破坏）
# 只有一个簇且位于记录前半段时，判定为**结构性不可标定**：破坏发生在
# 记录前三分之一，后面没有记录到退化段，三阶段标签不存在，只保留 b3
# 供「破坏前是否报警」使用。
# 簇是否算状态改变，用块**中位**是否同步抬升判定；只抬总和不抬中位的
# 判为瞬变，不参与登记。
CLUSTER_TOP = 0.01        # 取能量前 1% 的命中定位事件
CLUSTER_GAP = 5.0         # 相邻居间隔超过此值即分簇（life%）
CLUSTER_MIN_HITS = 20     # 少于该数的簇视为瞬变
CLUSTER_START_MIN = 2.0   # 中心早于此值的簇视为启振瞬变
CLUSTER_B1_TOP = 0.05     # b1 检测用的能量截断（比 b3 的 1% 宽，捕稀疏早期事件）
CLUSTER_STATE_RATIO = 1.3  # 块中位抬升倍数（仅作诊断输出，不参与筛选）


def _energy_clusters(e, top=CLUSTER_TOP, nb=100, sw=5, frac=0.20):
    """按命中能量前 top 比例的位置密度分簇。

    直接按相邻位置间隔分簇会被中段的零散命中“桥接”成一整块，
    改为先做 1% 分箱的密度曲线，再取高于阈值(峰值 frac 倍)的连续段。
    """
    n = e.size
    k = max(int(top * n), CLUSTER_MIN_HITS)
    if k >= n:
        return []
    life = (np.arange(n) + 0.5) / n * 100.0
    pos = life[np.argsort(e)[-k:]]
    h, _ = np.histogram(pos, bins=nb, range=(0.0, 100.0))
    hs = np.convolve(np.pad(h.astype(float), sw // 2, mode='edge'),
                     np.ones(sw) / sw, mode='valid')[:nb]
    if hs.max() <= 0:
        return []
    above = hs >= frac * hs.max()
    segs, i = [], 0
    while i < nb:
        if above[i]:
            j = i
            while j + 1 < nb and above[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    out = []
    for i, j in segs:
        sel = pos[(pos >= i) & (pos < j + 1)]
        if sel.size < CLUSTER_MIN_HITS:
            continue
        lo, hi = np.percentile(sel, 5), np.percentile(sel, 95)
        out.append(dict(n=int(sel.size), start=round(float(lo), 1),
                        end=round(float(hi), 1),
                        center=round(float(np.median(sel)), 1),
                        span=round(float(hi - lo), 1)))
    return out


def run_weaklabel_cluster(gid):
    """v4 双簇登记弱标签；返回 dict 或 None。"""
    dl = _load_ae_strain(gid)
    ae = dl.ae
    if ae is None or len(ae) < 400:
        return None
    n = len(ae)
    ecol = _pick_col(ae, 'MeanSquare', ['Peak'])
    if ecol is None:
        return None
    e = np.maximum(ae[ecol].to_numpy(float), 0.0)
    if ecol == 'Peak':
        e = e ** 2

    cl = _energy_clusters(e)
    if not cl:
        return None

    idx = np.array_split(np.arange(n), NB_BLOCK)
    med = np.array([float(np.median(e[i])) if i.size else np.nan for i in idx])
    med = _smooth_edge(np.log10(np.maximum(med, 1e-12)), SM_BLOCK)
    cgrid = (np.arange(NB_BLOCK) + 0.5) / NB_BLOCK * 100.0

    def _ratio(c):
        a = med[(cgrid >= c) & (cgrid < c + 5.0)]
        b = med[(cgrid >= c - 5.0) & (cgrid < c)]
        if a.size < 2 or b.size < 2:
            return 1.0
        return float(10 ** (np.median(a) - np.median(b)))

    for c in cl:
        c['ratio'] = round(_ratio(c['center']), 2)
        c['state'] = bool(c['ratio'] >= CLUSTER_STATE_RATIO)

    # 筛选：绝对命中数太少 → 孤立瞬变；中心落在启振段 → 加载初期瞬变。
    # 不用块中位作判据：AE 的损伤本身就是稀疏大事件，中位被噪声底线占住，
    # 真实破坏处的中位往往不抬反降。
    kept = [c for c in cl if c['n'] >= CLUSTER_MIN_HITS
            and c['center'] >= CLUSTER_START_MIN]
    if not kept:
        kept = [max(cl, key=lambda c: c['n'])]
    kept.sort(key=lambda c: c['center'])

    last = kept[-1]
    b3 = last['center']
    b2 = last['start']

    # b1 = 最早的高能事件簇。只在**确实有两个簇**时给值；单簇组的首次损伤
    # 由 cmd_weaklabels 用跨方法中位补，避免这里凭一个簇硬造一个锚点。
    # 不用「更宽的能量截断」：5% 截断会把启振瞬变（020 的 0.5%、018 的 10%）
    # 抬成伪事件；也不用「持续电平抬升」：这批数据损伤是离散大事件，
    # 电平抬起来又落回去，判据会整体失效。
    b1 = kept[0]['center'] if len(kept) >= 2 else None
    unlabelable = b1 is None
    b1v = b1 if b1 is not None else float('nan')
    if b2 > b3:
        b2, b3 = b3, max(b2, b3)

    strain_diverge = None
    if dl.strain is not None and 'strain' in dl.strain.columns and len(dl.strain) > 200:
        strain = dl.strain['strain'].values
        ns = len(strain)
        edges = np.linspace(0, ns, 200 + 1).astype(int)
        blk = np.array([float(np.std(strain[edges[i]:edges[i + 1]]))
                        if edges[i + 1] > edges[i] else 0.0 for i in range(200)])
        bpct = np.linspace(0, 100, 200)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk[mid])) if mid.any() else 0.0
        if base > 1e-9:
            over = np.where((bpct > 50) & (blk > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])

    life = np.arange(1000) * 0.1
    stage = np.zeros(1000, dtype=int)
    if b1 is not None:
        stage[life >= b1] = 1
    stage[life >= b2] = 2
    stage[life >= b3] = 3
    conf = 'na' if unlabelable else ('high' if len(kept) >= 2 else 'mid')
    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2), 1),
                b3=round(float(b3), 1), n_ae=n, strain_diverge=strain_diverge,
                conf=conf, n_agree=len(kept), n_ch=len(cl),
                b1_none=bool(b1 is None), b2_none=unlabelable,
                b3_clamped=False, unlabelable=unlabelable,
                n_clusters=len(kept),
                clusters=' | '.join('%d@%.1f(%.1f-%.1f,r%.2f%s)'
                                    % (c['n'], c['center'], c['start'], c['end'],
                                       c['ratio'], '' if c['state'] else ',瞬变')
                                    for c in cl),
                life=life, stage=stage)


def run_weaklabel_peak(gid):
    """v1 原口径（单列 Peak），逐字保留，仅供消融对比。"""
    dl = _load_ae_strain(gid)
    peak = dl.ae['Peak'].values if dl.ae is not None and 'Peak' in dl.ae.columns else None
    strain = dl.strain['strain'].values if dl.strain is not None else None
    n_ae = len(peak) if peak is not None else 0
    if n_ae == 0:
        return None
    grid = np.linspace(0, 1, 2001)
    e2 = np.maximum(peak, 0.0) ** 2
    cum = np.cumsum(e2)
    logc = np.log10(np.maximum(np.interp(grid, np.linspace(0, 1, n_ae), cum), 1e-12))
    logc = smooth(logc, 21)                       # ~1% 平滑
    b2 = find_b2(logc, grid)
    b1 = find_b1(logc, grid, b2) if b2 is not None else None
    b3 = 99.0                                     # 失效锚点 = 末端
    # 应变末期发散点（b2 取应变发散更早者 → 应变主导）
    strain_diverge = None
    if strain is not None and len(strain) > 200:
        n = len(strain)
        nb = 200
        edges = np.linspace(0, n, nb + 1).astype(int)
        blk_std = np.array([float(np.std(strain[edges[i]:edges[i+1]])) if edges[i+1] > edges[i] else 0.0 for i in range(nb)])
        bpct = np.linspace(0, 100, nb)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk_std[mid])) if mid.any() else 0.0
        tail = bpct > 50
        if base > 1e-9:
            over = np.where(tail & (blk_std > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])
    if strain_diverge is not None and 0 < strain_diverge < 95:
        if b2 is None or strain_diverge < b2:
            b2 = strain_diverge
    n_grid = 1000                                  # 阶梯组装 0.1% 步长
    life = np.arange(n_grid) * 0.1
    stage = np.zeros(n_grid, dtype=int)
    b1v = b1 if b1 is not None else (b2 if b2 is not None else 60.0)
    b2v = b2 if b2 is not None else (b3 * 0.8)
    stage[life >= b1v] = 1
    stage[life >= b2v] = 2
    stage[life >= b3] = 3
    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2v), 1), b3=b3,
                n_ae=n_ae, strain_diverge=strain_diverge,
                life=life, stage=stage)


MODES = [
    ('peak', run_weaklabel_peak, 'labels_summary_peak.csv', '_peak'),
    ('multi', run_weaklabel_multi, 'labels_summary_multi.csv', '_multi'),
    ('step', run_weaklabel_step, 'labels_summary_step.csv', '_step'),
    ('cluster', run_weaklabel_cluster, 'labels_summary.csv', '_cluster'),
]


def cmd_weaklabels():
    """四套口径同时产出，并把跨方法中位的 b1 锚写进规范标签。

    peak  = v1 单列 Peak（b3 假设 99.0）
    multi = v2 三通道 argmax|d²| 共识（b2 落在崩塌处）
    step  = v3 台阶口径（b2 取崩塌前第一个显著台阶）
    cluster = v4 双簇登记（b3 = 最后一个高能簇）

    规范集 = cluster 的 b3 + **三法中位的 b1**。单法 b2 都不稳，但三法
    中位加一个显式的不确定度（极差）是可复算的；极差大的组照样出结论，
    只是把不确定度一并写出来，而不是把它们丢掉。
    """
    outdir = os.path.join(ROOT, 'weak_labels')
    os.makedirs(outdir, exist_ok=True)
    hist = {m: [] for m, _, _, _ in MODES}
    per_gid = {}

    for gid in AE_GROUPS:
        got = {}
        for mode, fn, _, _ in MODES:
            try:
                r = fn(gid)
            except Exception as e:
                print(f'{gid} {mode} 失败 {type(e).__name__}: {e}')
                r = None
            if r is None:
                continue
            hist[mode].append(r)
            got[mode] = r
        per_gid[gid] = got

    # ---------- 跨方法 b1 锚 ----------
    cons = {}
    for gid, got in per_gid.items():
        cand = []
        if 'peak' in got and got['peak'].get('b2') is not None:
            cand.append(float(got['peak']['b2']))
        if 'step' in got and not got['step'].get('b2_none'):
            cand.append(float(got['step']['b2']))
        if 'cluster' in got and not got['cluster'].get('b1_none'):
            cand.append(float(got['cluster']['b1']))
        if cand:
            cons[gid] = (float(np.median(cand)),
                         float(max(cand) - min(cand)) if len(cand) > 1 else float('nan'),
                         len(cand))

    def _stage_of(b1, b2, b3):
        life = np.arange(1000) * 0.1
        st = np.zeros(1000, dtype=int)
        if b1 is not None and not np.isnan(b1):
            st[life >= b1] = 1
        if b2 is not None and not np.isnan(b2):
            st[life >= b2] = 2
        st[life >= b3] = 3
        return life, st

    # ---------- 写规范集（cluster 基底 + 中位 b1） ----------
    for r in hist['cluster']:
        gid = r['gid']
        k = cons.get(gid)
        if k is None:
            continue
        b3 = float(r['b3'])
        if k[0] >= b3 - STEP_GAP:
            # 与崩塌分不开：单阶段试验，b1 不存在
            r['b1'], r['b1_none'], r['unlabelable'] = float('nan'), True, True
        else:
            r['b1'], r['b1_none'], r['unlabelable'] = round(k[0], 1), False, False
        r['b1_spread'] = None if np.isnan(k[1]) else round(k[1], 1)
        r['b1_n'] = k[2]
        r['life'], r['stage'] = _stage_of(r['b1'], r.get('b2'), b3)
        df = pd.DataFrame({'life_pct': r['life'], 'ref_stage': r['stage']})
        df.loc[df['life_pct'] >= b3, 'note'] = 'failure(b3)'
        df.loc[(df['life_pct'] >= r['b2']) & (df['life_pct'] < b3), 'note'] = 'phase2(扩展)'
        if not r['b1_none']:
            df.loc[(df['life_pct'] >= r['b1']) & (df['life_pct'] < r['b2']), 'note'] = 'phase1(微损伤)'
            df.loc[df['life_pct'] < r['b1'], 'note'] = 'phase0(健康/加载)'
        else:
            df.loc[df['life_pct'] < r['b2'], 'note'] = 'phase0/1(无独立萌生段)'
        df.iloc[::2].to_csv(os.path.join(outdir, f'{gid}_label_cluster.csv'),
                            index=False, encoding='utf-8-sig')

    # ---------- 其余口径的逐组文件 ----------
    for mode, _, _, suffix in MODES:
        if mode == 'cluster':
            continue
        for r in hist[mode]:
            df = pd.DataFrame({'life_pct': r['life'], 'ref_stage': r['stage']})
            df.loc[df['life_pct'] >= r['b3'], 'note'] = 'failure(b3)'
            df.loc[(df['life_pct'] >= r['b2']) & (df['life_pct'] < r['b3']), 'note'] = 'phase2(扩展)'
            df.loc[(df['life_pct'] >= r['b1']) & (df['life_pct'] < r['b2']), 'note'] = 'phase1(微损伤)'
            df.loc[df['life_pct'] < r['b1'], 'note'] = 'phase0(健康/加载)'
            df.iloc[::2].to_csv(os.path.join(outdir, f"{r['gid']}_label{suffix}.csv"),
                                index=False, encoding='utf-8-sig')

    # ---------- 汇总表 ----------
    for mode, _, name, _ in MODES:
        rows = hist[mode]
        cols = ['gid', 'b1', 'b2', 'b3', 'n_ae', 'strain_diverge', 'conf', 'n_agree', 'n_ch']
        for extra in ('b1_none', 'b2_none', 'b3_clamped', 'o_energy', 'o_freq',
                      'o_impuls', 'o_strain', 'e3_energy', 'e3_freq', 'e3_impuls',
                      'unlabelable', 'n_clusters', 'clusters', 'b1_spread', 'b1_n'):
            if rows and extra in rows[0]:
                cols.append(extra)
        pd.DataFrame([{k: r.get(k) for k in cols} for r in rows]).to_csv(
            os.path.join(outdir, name), index=False, encoding='utf-8-sig')

    print('gid      b1(中位)  极差  候选数      b3   判定')
    for gid in AE_GROUPS:
        k = cons.get(gid)
        got = per_gid.get(gid, {})
        b3 = got.get('cluster', {}).get('b3')
        if k is None:
            print('%s      %8s  %5s  %5s  %8s   无候选' % (gid, '-', '-', 0,
                                                        '-' if b3 is None else '%.1f' % b3))
            continue
        sp = '  -  ' if np.isnan(k[1]) else '%5.1f' % k[1]
        tag = '单阶段' if (b3 is not None and k[0] >= b3 - STEP_GAP) else (
            '锚点不稳' if (not np.isnan(k[1]) and k[1] > 10) else 'ok')
        print('%s      %8.1f  %5s  %5d  %8s   %s'
              % (gid, k[0], sp, k[2], '-' if b3 is None else '%.1f' % b3, tag))
    print('\n' + ' ; '.join('%s %d 组 -> %s' % (m, len(hist[m]), nm)
                            for m, _, nm, _ in MODES))


# ============================================================
# 分发
# ============================================================
PREP_TASKS = {'align': cmd_align, 'weaklabels': cmd_weaklabels}


def prep_main():
    ap = argparse.ArgumentParser(description='阶段① 数据准备：align / weaklabels')
    ap.add_argument('tasks', nargs='+', choices=list(PREP_TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    a = ap.parse_args()
    todo = list(PREP_TASKS) if 'all' in a.tasks else a.tasks
    for t in todo:
        print(f'\n########## [{t}] ##########', flush=True)
        PREP_TASKS[t]()
        print(f'########## [{t}] 完成 ##########', flush=True)


# ==========================================================================
# 段 2/3  原 main/evaluate.py（355 行，正文逐字保留；改名 main→eval_main, TASKS→EVAL_TASKS, WORKERS→EVAL_WORKERS, _stream_d→eval_stream_d）
#==========================================================================

# -*- coding: utf-8 -*-
"""阶段② 主样本评估与出图（5 组 016-020）

子命令（可组合，如 `python main/pipeline.py evaluate degree warning` 或 `all`）：
  degree   评估损伤度 D 达阈/单调 → results/damage_degree_metrics.csv（缓存 cache/_hi_cache）
  warning  A-预警 onset 分级 → results/warning_onset.csv（读 cache/_hi_cache，先跑 degree）
  curves   D(t) 曲线总览 → figures/damage_degree_curves.png（读 cache/_hi_cache）
  paper    论文级四联图 + 方法流程图 → figures/paper_*.png（逐点重算，较慢）
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from concurrent.futures import ProcessPoolExecutor
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, GROUPS_EVAL, ref_map, ref_stage, onset_of, LOW, DROP, HOLD_FRAC

ROOT = os.path.dirname(os.path.abspath(__file__))
DNCACHE = os.path.join(ROOT, 'cache', '_hi_cache')      # 逐点 D 缓存
RESULTS = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(DNCACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# 公共：逐点跑 D（事件门控 AE 能量）
# ============================================================
def eval_stream_d(gid, di):
    """StreamSimulator 逐点喂 OnlineDamageIndex → 返回逐点 D 数组。"""
    sim = StreamSimulator(gid)
    sim.load_data()
    dlist = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        dlist.append(di.update(strain, float(pk) if pk is not None else None,
                               None, di.shape_value(p.get('ae'))))
    sim.cleanup()
    return np.array(dlist, dtype=np.float32)


def _first_ge(d, th):
    idx = np.where(d >= th)[0]
    return float(idx[0]) / len(d) * 100.0 if len(idx) else np.nan


# ============================================================
# 子命令 degree —— 评估损伤度 D（达阈/单调）
# ============================================================
DEG_OUT = os.path.join(RESULTS, 'damage_degree_metrics.csv')
EVAL_WORKERS = 4


def degree_run_one(gid):
    dpath = os.path.join(DNCACHE, f'{gid}.npy')
    if os.path.exists(dpath):
        return gid, np.load(dpath)
    try:
        d = eval_stream_d(gid, OnlineDamageIndex())
    except Exception as e:
        print(f'{gid}: D 计算失败 {type(e).__name__}: {e}', flush=True)
        return gid, None
    np.save(dpath, d)
    return gid, d


def cmd_degree():
    if EVAL_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=EVAL_WORKERS) as ex:
            res = list(ex.map(degree_run_one, GROUPS_EVAL))
    else:
        res = [degree_run_one(g) for g in GROUPS_EVAL]
    rows = []
    refs = ref_stage()
    for gid, d in res:
        if d is None:
            print(f'{gid}: 跳过（D 不可计算）')
            continue
        n = len(d)
        t25, t55, t85 = _first_ge(d, .25), _first_ge(d, .55), _first_ge(d, .85)
        tail = d[int(n * .80):]                       # 断裂前 20% 单调性
        mono = float(np.mean(np.diff(tail) >= 0)) if len(tail) > 2 else np.nan
        b1, b3 = refs.get(gid, (np.nan, np.nan))      # 两阶段锚：首次损伤 / 崩塌
        rows.append(dict(gid=gid, n=n, D_end=round(float(d[-1]), 2),
                         t25=round(t25, 1) if not np.isnan(t25) else np.nan,
                         t55=round(t55, 1) if not np.isnan(t55) else np.nan,
                         t85=round(t85, 1) if not np.isnan(t85) else np.nan,
                         b1=round(b1, 1) if not np.isnan(b1) else np.nan,
                         b3=round(b3, 1) if not np.isnan(b3) else np.nan,
                         err85=(round(t85 - b1, 1)
                                if not (np.isnan(t85) or np.isnan(b1)) else np.nan),
                         lead85=round(b3 - t85, 1)
                         if not (np.isnan(t85) or np.isnan(b3)) else np.nan,
                         tail_mono=round(mono * 100, 1)))
    df = pd.DataFrame(rows).sort_values('gid')
    df.to_csv(DEG_OUT, index=False, encoding='utf-8-sig')
    print('=== D(t) 达阈统计 ===')
    print('  D_end>=0.85 组数:', int((df['D_end'] >= .85).sum()), '/', len(df))
    print('  D_end>=0.55 组数:', int((df['D_end'] >= .55).sum()))
    l85 = df['lead85'].dropna()
    print(f'  达0.85组 平均断裂前提前量={l85.mean():.1f}%  (n={len(l85)})')
    print(df.to_string(index=False))
    print('结果已存:', DEG_OUT)


# ============================================================
# 子命令 warning —— A-预警 onset 分级
# ============================================================
WARN_OUT = os.path.join(RESULTS, 'warning_onset.csv')


def cmd_warning():
    rows = []
    for gid in GROUPS_EVAL:
        dpath = os.path.join(DNCACHE, f'{gid}.npy')
        if not os.path.exists(dpath):
            print(f'{gid}: 无 D 缓存，跳过（先跑 degree）', flush=True)
            continue
        d = np.load(dpath)
        n = len(d)
        hold = max(2000, int(n * HOLD_FRAC))
        tw = onset_of(d, hold)
        refs = ref_stage()
        b1, b3 = refs.get(gid, (np.nan, np.nan))
        err = round(tw - b1, 1) if (tw is not None and not np.isnan(b1)) else None
        lead = round(b3 - tw, 1) if (tw is not None and not np.isnan(b3)) else None
        if tw is None:
            grade = 'C漏报'
        elif err is None:
            grade = 'N锚点缺'
        elif err < -15:
            grade = 'E过早'
        elif err > 15:
            grade = 'D偏晚'
        else:
            grade = 'A合理'
        rows.append(dict(gid=gid, t_warn=tw, b1=b1, b3=b3,
                         err=err, lead=lead, grade=grade))
        print(f'{gid}: t_warn={tw}  err={err}  lead={lead}  [{grade}]', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WARN_OUT, index=False, encoding='utf-8-sig')
    cnt = df['grade'].value_counts()
    print('=== 分级计数 ===')
    for k in ['A合理', 'D偏晚', 'E过早', 'C漏报', 'N锚点缺']:
        print(f'  {k}: {int(cnt.get(k, 0))} 组')
    ok = df[df['grade'] == 'A合理']
    if len(ok):
        print(f'A 组平均 lead(断裂前提前)={ok["lead"].mean():.1f}%')
    print('结果已存:', WARN_OUT)


# ============================================================
# 子命令 curves —— D(t) 曲线总览
# ============================================================
def cmd_curves():
    refs = ref_map()
    n = len(GROUPS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.4 * nrows))
    axes = axes.ravel()
    for ax, g in zip(axes, GROUPS):
        d = np.load(os.path.join(DNCACHE, f'{g}.npy'))
        x = np.linspace(0, 100, len(d))
        ax.plot(x, d, lw=0.8, color='tab:blue')
        for th, c in [(0.25, 'tab:orange'), (0.55, 'tab:red'), (0.85, 'darkred')]:
            ax.axhline(th, color=c, lw=0.6, ls='--', alpha=0.6)
        b2, b3 = refs.get(g, (np.nan, 99.0))
        ax.axvline(b2, color='tab:green', lw=0.8, ls='-.', alpha=0.8)
        ax.axvline(b3, color='k', lw=0.8, ls=':', alpha=0.8)
        ax.set_title(f'{g}  D_end={d[-1]:.2f}', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 99])
        ax.grid(alpha=0.3)
    for ax in axes[n:]:                       # 多余子图隐藏
        ax.axis('off')
    fig.suptitle('主样本 5 组 D(t) 曲线（默认 v6+latch；分级 0.25/0.55/0.85；绿虚线 b2，黑点线 b3=断裂）',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(FIGDIR, 'damage_degree_curves.png')
    fig.savefig(out, dpi=110)
    print('已存:', out)


# ============================================================
# 子命令 paper —— 论文级四联图 + 方法流程图
# ============================================================
BLK = 500
LV_COLORS = ['#d9ead3', '#fff2cc', '#fce5cd', '#f4cccc']   # 0正常..3临危


def paper_collect(gid):
    """逐点跑 full，每块记 {pct, strain, ae_en, e_ae, e_strain, D, level}"""
    di = OnlineDamageIndex()
    sim = StreamSimulator(gid)
    sim.load_data()
    pts = 0
    blk_st, blk_en = [], 0.0
    rows = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        if strain is not None and not np.isnan(strain):
            blk_st.append(float(strain))
        if pk is not None:
            blk_en += float(pk) ** 2
        d = di.update(strain, float(pk) if pk is not None else None,
                      None, di.shape_value(p.get('ae')))
        pts += 1
        if pts % BLK == 0:
            rows.append(dict(
                pct=pts / sim.total_points * 100.0,
                strain=float(np.mean(blk_st)) if blk_st else np.nan,
                ae_en=np.log10(blk_en + 1.0),
                e_ae=di._last_e_ae, e_strain=di._last_e_strain,
                D=float(d), level=di.level))
            blk_st, blk_en = [], 0.0
    sim.cleanup()
    return pd.DataFrame(rows)


def paper_draw_one(gid):
    df = paper_collect(gid)
    refs = ref_map()
    b2, b3 = refs.get(gid, (np.nan, 99.0))
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.3, 0.6], hspace=0.32,
                          left=0.09, right=0.97, top=0.94, bottom=0.08)
    ax = [fig.add_subplot(gs[i]) for i in range(4)]
    # A 原始信号
    a = ax[0]
    a.plot(df['pct'], df['strain'], color='#1f77b4', lw=0.9, label='应变(块均值)')
    a2 = a.twinx()
    a2.plot(df['pct'], df['ae_en'], color='#d62728', lw=0.8, alpha=0.75, label='AE 块能量 log')
    a2.set_ylabel('log(能量+1)')
    a.set_ylabel('应变')
    a.set_title(f'{gid}  原始多源信号')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left', framealpha=0.6)
    # B 证据
    b = ax[1]
    b.plot(df['pct'], df['e_ae'], color='#9467bd', lw=1.0, label='e_ae (AE 证据)')
    b.plot(df['pct'], df['e_strain'], color='#2ca02c', lw=1.0, label='e_strain (应变证据)')
    b.set_ylim(-0.02, 1.02)
    b.set_ylabel('证据')
    b.legend(fontsize=8, loc='upper left', framealpha=0.6)
    b.set_title('各源损伤证据')
    # C D + 分级
    c = ax[2]
    c.plot(df['pct'], df['D'], color='#000000', lw=1.4, label='D(t)')
    for th, col, lab in [(0.25, '#e07b00', '注意'), (0.55, '#d62728', '预警'), (0.85, '#8b0000', '临危')]:
        c.axhline(th, color=col, lw=0.9, ls='--', alpha=0.6)
        c.text(1.5, th + 0.01, f'{th:.2f} {lab}', color=col, fontsize=7, va='bottom')
    c.axvline(b2, color='#2e7d32', lw=1.0, ls='-.', alpha=0.9)
    c.text(b2 + 0.5, 0.95, 'b2 扩展', color='#2e7d32', fontsize=7, rotation=90, va='top')
    c.axvline(b3, color='k', lw=1.0, ls=':', alpha=0.8)
    c.text(b3 - 0.5, 0.95, 'b3 断裂', fontsize=7, rotation=90, va='top', ha='right')
    tw = df[df['D'] >= 0.30]
    if len(tw):
        seg = df['D'].to_numpy()
        idx0 = np.where(seg >= 0.30)[0]
        onset = None
        hold = max(1, int(len(seg) * 0.02))
        for i in idx0:
            if seg[i:min(len(seg), i + hold)].min() >= 0.15:
                onset = df['pct'].iloc[i]
                break
        if onset is not None:
            c.plot(onset, 0.30, 'r*', ms=14, zorder=5)
            c.annotate(f'预警 {onset:.0f}%', (onset, 0.30), textcoords='offset points',
                       xytext=(8, 6), fontsize=8, color='red')
    c.set_ylim(-0.02, 1.02)
    c.set_ylabel('损伤度 D')
    c.legend(fontsize=8, loc='lower right', framealpha=0.6)
    c.set_title('损伤度 D(t) 与分级预警')
    # D 预警级别
    dax = ax[3]
    for lv in range(0, 4):
        m = df['level'] == lv
        dax.fill_between(df['pct'], 0, 1, where=m, color=LV_COLORS[lv], step='post',
                         label=f'级别{lv}({"正常/注意/预警/临危".split("/")[lv]})')
    dax.set_ylim(0, 1)
    dax.set_yticks([])
    dax.set_ylabel('预警级别')
    dax.set_title('分级预警输出(0正常-1注意-2预警-3临危)')
    for a2x in ax:
        a2x.set_xlim(0, 99)
        a2x.grid(alpha=0.3)
    ax[-1].set_xlabel('寿命 (%)')
    out = os.path.join(FIGDIR, f'paper_{gid}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('已存:', out)


def paper_flowchart():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis('off')
    boxes = [
        (0.03, '原始多源信号\n光纤 / 声发射(AE) / 应变', '#eef3fb'),
        (0.24, '证据提取(块级,在线自适应)\ne_dmg: 损伤型AE事件能量\ne_full: 全能量累积加速度\ne_strain: 应变std发散', '#fdf2e9'),
        (0.47, '证据融合\nrisk = max(e_ae, e_strain)\n(e_ae = max(e_dmg, e_full))', '#f3f0fb'),
        (0.70, '单调累积损伤度\nD(t): 升快降慢(损伤记忆)\n确认后 latch 快追(0.85可达)', '#e9f7ef'),
        (0.90, '分级预警\n0.25注意 / 0.55预警 / 0.85临危', '#fdecec'),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.16, 0.42, boxstyle='round,pad=0.012',
                                    fc=col, ec='0.55', lw=1.1, transform=ax.transAxes))
        ax.text(x + 0.08, 0.56, txt, ha='center', va='center', fontsize=8.5,
                transform=ax.transAxes)
    for x0, x1 in [(0.19, 0.24), (0.40, 0.47), (0.63, 0.70), (0.86, 0.90)]:
        ax.add_patch(FancyArrowPatch((x0 + 0.01, 0.56), (x1 - 0.005, 0.56),
                                     arrowstyle='-|>', mutation_scale=16, lw=1.3,
                                     color='0.3', transform=ax.transAxes))
    ax.text(0.5, 0.92, '多源连续损伤度 D(t) 方法流程（在线因果、零标签）', ha='center',
            fontsize=12, fontweight='bold')
    ax.text(0.5, 0.08, '在线每点更新：证据(块) → risk → D 单调累积 → 分级输出', ha='center',
            fontsize=9, color='0.35')
    out = os.path.join(FIGDIR, 'method_flowchart.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('已存:', out)


def cmd_paper():
    paper_flowchart()
    for g in GROUPS:
        paper_draw_one(g)


# ============================================================
# 分发
# ============================================================
EVAL_TASKS = {'degree': cmd_degree, 'warning': cmd_warning,
         'curves': cmd_curves, 'paper': cmd_paper}


def eval_main():
    ap = argparse.ArgumentParser(description='阶段② 主样本评估与出图：degree / warning / curves / paper')
    ap.add_argument('tasks', nargs='+', choices=list(EVAL_TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    ap.add_argument('--workers', type=int, default=4,
                    help='多进程 worker 数(默认 4)')
    a = ap.parse_args()
    global EVAL_WORKERS
    EVAL_WORKERS = a.workers
    todo = list(EVAL_TASKS) if 'all' in a.tasks else a.tasks
    for t in todo:
        print(f'\n########## [{t}] ##########', flush=True)
        EVAL_TASKS[t]()
        print(f'########## [{t}] 完成 ##########', flush=True)


# ==========================================================================
# 段 3/3  原 main/robustness.py（361 行，正文逐字保留；改名 main→rob_main, TASKS→ROB_TASKS, WORKERS→ROB_WORKERS, _stream_d→rob_stream_d）
#==========================================================================

# -*- coding: utf-8 -*-
"""阶段③ 方法稳健性研究（5 组主样本）

子命令（可组合，如 `python main/pipeline.py robust sens loso ablation` 或 `all`）：
  sens      参数 ±30% 敏感性 → results/parameter_sensitivity.csv（缓存 cache/_t7_cache）
  loso      留一试件交叉验证选参 → results/leave_one_out_cv.csv（需先跑 sens 生成缓存）
  ablation  证据消融(去各源/结构) → results/ablation.csv + figures/ablation.png
  stats     T9 统计显著性(多源 vs 单源配对 + Bootstrap CI) → results/statistical_test.csv
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from concurrent.futures import ProcessPoolExecutor
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import (PARS, DEFAULT, GROUPS, cfg_id, T7CACHE,
                         per_group_metrics, run_cfg, table_for, summary_row)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
ROB_WORKERS = 8


# ============================================================
# 公共：逐点跑 D（事件门控 AE 能量）
# ============================================================
def rob_stream_d(gid, di):
    sim = StreamSimulator(gid)
    sim.load_data()
    dlist = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        dlist.append(di.update(strain, float(pk) if pk is not None else None,
                               None, di.shape_value(p.get('ae'))))
    sim.cleanup()
    return np.array(dlist, dtype=np.float32)


def _cached_d(gid, di, cdir):
    """按缓存目录计算/读取逐点 D。"""
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, f'{gid}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    d = rob_stream_d(gid, di)
    np.save(fp, d)
    return d


# ============================================================
# 子命令 sens —— 参数敏感性
# ============================================================
SENS_OUT = os.path.join(RESULTS, 'parameter_sensitivity.csv')


def build_configs():
    """默认 + 每参数 ±30%。返回 [(cfg_params, label)]"""
    cfgs = [({}, 'default')]
    for p in PARS:
        for m in (0.7, 1.3):
            cp = dict(DEFAULT)
            cp[p] = DEFAULT[p] * m
            cfgs.append((cp, f'{p}x{m}'))
    return cfgs


def cmd_sens():
    cfgs = build_configs()
    rows = []
    print(f'[T7-sens] 共 {len(cfgs)} 配置 × {len(GROUPS)} 组, workers={ROB_WORKERS}', flush=True)
    t0 = time.time()
    for params, label in cfgs:
        cid = cfg_id(params)
        ts = time.time()
        gid_d = run_cfg(params, workers=ROB_WORKERS)
        df = table_for(cid, gid_d)
        row = summary_row(cid, df)
        row['label'] = label
        rows.append(row)
        print(f'  [{label}] {time.time()-ts:.0f}s  n55={row["n55"]} n85={row["n85"]} '
              f'A={row["nA"]} E={row["nE"]} D={row["nD"]} C={row["nC"]} '
              f'med|err|={row["mean_abs_err"] if row["mean_abs_err"]==row["mean_abs_err"] else "-"}', flush=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(SENS_OUT, index=False, encoding='utf-8-sig')
    print(f'总耗时 {time.time()-t0:.0f}s; 结果已存: {SENS_OUT}', flush=True)


# ============================================================
# 子命令 loso —— 留一试件交叉验证
# ============================================================
LOSO_OUT = os.path.join(RESULTS, 'leave_one_out_cv.csv')


def load_table(params):
    """从缓存读 {gid:d} 并转逐组指标表。缺缓存则报错提示先跑 sens。"""
    cid = cfg_id(params)
    cdir = os.path.join(T7CACHE, cid)
    rows = []
    for g in GROUPS:
        fp = os.path.join(cdir, f'{g}.npy')
        if not os.path.exists(fp):
            sys.exit(f'缺缓存 {fp} —— 请先运行 main/pipeline.py robust sens(或单独补算该配置)')
        d = np.load(fp)
        rows.append(per_group_metrics(g, d))
    return pd.DataFrame(rows)


def cand_cfgs(p):
    """参数 p 的 3 档候选(其余默认)。"""
    out = []
    for m in (0.7, 1.0, 1.3):
        cp = dict(DEFAULT)
        cp[p] = DEFAULT[p] * m
        out.append((m, cp))
    return out


def score(df):
    """聚合目标(越大越好): A 对齐 + 达级 + 漏报重罚。"""
    nA = int((df['grade'] == 'A').sum())
    n55 = int((df['D_end'] >= .55).sum())
    nC = int((df['grade'] == 'C').sum())
    return nA + 0.25 * n55 - 1.5 * nC


def cmd_loso():
    recs = []
    for p in PARS:
        tabs = {m: load_table(cp) for m, cp in cand_cfgs(p)}
        tab0 = tabs[1.0]
        chosen = []
        diff_rows = []
        for g in GROUPS:
            train = [x for x in GROUPS if x != g]
            best_m, best_s = None, -1e18
            for m, cp in cand_cfgs(p):
                s = score(tabs[m][tabs[m]['gid'].isin(train)])
                if s > best_s:
                    best_s, best_m = s, m
            chosen.append(best_m)
            row0 = tab0[tab0['gid'] == g].iloc[0]
            rowc = tabs[best_m][tabs[best_m]['gid'] == g].iloc[0]
            diff_rows.append(dict(gid=g, par=p, chosen_m=best_m,
                                  g0=row0['grade'], gc=rowc['grade'],
                                  err0=row0['err'], errc=rowc['err'],
                                  dend0=row0['D_end'], dendc=rowc['D_end']))
        ch = pd.Series(chosen).value_counts().to_dict()
        diff = pd.DataFrame(diff_rows)
        improved = int(((diff['g0'] == 'E') & (diff['gc'] == 'A')).sum())
        worsened = int(((diff['g0'].isin(['A', 'D'])) & (diff['gc'] == 'E')).sum())
        nA0 = int((tab0['grade'] == 'A').sum())
        recs.append(dict(par=p, desc='', ch_0_7=ch.get(0.7, 0),
                         ch_1_0=ch.get(1.0, 0), ch_1_3=ch.get(1.3, 0),
                         nA_default=nA0, EtoA=improved, AtoE=worsened))
        print(f'[{p}] 选档(0.7/1.0/1.3)={ch.get(0.7,0)}/{ch.get(1.0,0)}/{ch.get(1.3,0)}  '
              f'留出: E→A {improved}, A→E {worsened}  (默认全数据 nA={nA0})', flush=True)
    out = pd.DataFrame(recs)
    out.to_csv(LOSO_OUT, index=False, encoding='utf-8-sig')
    print('结果已存:', LOSO_OUT, flush=True)


# ============================================================
# 子命令 ablation —— 证据消融
# ============================================================
ABL_CACHE = os.path.join(ROOT, 'cache', '_t8_cache')
ABL_MODES = [
    ('full', '完整 v6(基准)'),
    ('no_dmg', '去损伤型事件(仅全能量)'),
    ('no_full', '去全能量加速(仅损伤型)'),
    ('no_strain', '去应变证据(仅 AE)'),
    ('only_strain', '仅应变证据'),
    ('no_accum', '去单调累积(D=risk)'),
]
KEY = ['016', '020', '017', '019']   # 消融展示重点组(均属主样本)


def _abl_run_one(args):
    gid, mode = args
    cdir = os.path.join(ABL_CACHE, mode)
    return gid, _cached_d(gid, OnlineDamageIndex({'abl': mode}), cdir)


def cmd_ablation():
    rows_all = []
    agg = []
    for mode, desc in ABL_MODES:
        with ProcessPoolExecutor(max_workers=ROB_WORKERS) as ex:
            res = list(ex.map(_abl_run_one, [(g, mode) for g in GROUPS]))
        gid_d = dict(res)
        per = {r['gid']: r for r in (per_group_metrics(g, d) for g, d in gid_d.items())}
        df = pd.DataFrame([per[g] for g in GROUPS]).sort_values('gid')
        df.insert(0, 'mode', mode)
        rows_all.append(df)
        n55 = int((df['D_end'] >= .55).sum())
        n85 = int((df['D_end'] >= .85).sum())
        cnt = df['grade'].value_counts()
        agg.append(dict(mode=mode, desc=desc, n55=n55, n85=n85,
                        D_end_med=float(df['D_end'].median()),
                        nA=int(cnt.get('A', 0)), nE=int(cnt.get('E', 0)),
                        nD=int(cnt.get('D', 0)), nC=int(cnt.get('C', 0)),
                        tail_mono=float(df['tail_mono'].mean())))
        print(f'[{desc}] n55={n55} n85={n85} A/E/D/C='
              f'{int(cnt.get("A",0))}/{int(cnt.get("E",0))}/{int(cnt.get("D",0))}/{int(cnt.get("C",0))} '
              f'D_end中位={df["D_end"].median():.2f}', flush=True)
    big = pd.concat(rows_all, ignore_index=True)
    big.to_csv(os.path.join(RESULTS, 'ablation.csv'), index=False, encoding='utf-8-sig')
    key_row = big[big['gid'].isin(KEY)].pivot_table(index='gid', columns='mode',
                                                    values='D_end').reindex(KEY)
    print('\n=== 重点组 D_end 对比 ===')
    print(key_row.round(2).to_string())
    a = pd.DataFrame(agg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(a))
    axes[0].bar(x, a['n55'], color='tab:blue'); axes[0].set_xticks(x)
    axes[0].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[0].set_title('达 0.55 组数')
    axes[1].bar(x, a['nA'], color='tab:green'); axes[1].set_xticks(x)
    axes[1].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[1].set_title('A-预警精准组数')
    axes[2].bar(x, a['nE'], color='tab:red'); axes[2].set_xticks(x)
    axes[2].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[2].set_title('E-过早组数')
    fig.suptitle('T8 消融: 各证据/结构对 D(t) 的影响')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'ablation.png'), dpi=110)
    print('结果已存: results/ablation.csv + figures/ablation.png')


# ============================================================
# 子命令 stats —— T9 统计检验
# ============================================================
STATS_CACHE = os.path.join(ROOT, 'cache', '_stats_cache')
STATS_MODES = {'full': '多源融合D(full)', 'no_strain': '仅AE(no_strain)',
               'only_strain': '仅应变(only_strain)'}
NBOOT = 2000


def _stats_run_one(args):
    gid, mode = args
    cdir = os.path.join(STATS_CACHE, mode)
    return gid, _cached_d(gid, OnlineDamageIndex({'abl': mode}), cdir)


def wilcoxon_onesided(a, b):
    """H1: a < b(多源融合误差更小)。返回 p(单侧) 与 n 配对。"""
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return np.nan, 0
    try:
        p = sp_stats.wilcoxon(diff, alternative='less').pvalue
        return p, n
    except ValueError:
        return np.nan, n


def bootstrap_ci(x, stat=np.mean, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boot = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def cmd_stats():
    modes = list(STATS_MODES.keys())
    jobs = [(g, m) for g in GROUPS for m in modes]
    with ProcessPoolExecutor(max_workers=ROB_WORKERS) as ex:
        res = list(ex.map(_stats_run_one, jobs))
    per = {m: {} for m in modes}
    for (g, m), (gid, d) in zip(jobs, res):
        per[m][gid] = per_group_metrics(gid, d)
    rows = []
    for g in GROUPS:
        base = per['full'][g]
        row = {'gid': g, 'b2': base['b2'], 'b3': base['b3']}
        for m in modes:
            mm = per[m][g]
            row[f'{m}_t_warn'] = mm['t_warn']
            row[f'{m}_err'] = mm['err']
            row[f'{m}_lead'] = mm['lead']
            row[f'{m}_grade'] = mm['grade']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'statistical_test.csv'), index=False, encoding='utf-8-sig')

    def colerr(m):
        return df[f'{m}_err'].to_numpy(float)

    print('=== 逐组 |预警-扩展| 误差(小=好) ===')
    print(df[['gid', 'full_err', 'no_strain_err', 'only_strain_err']]
          .assign(**{'|full|': df.full_err.abs(), '|AE|': df.no_strain_err.abs(),
                     '|strain|': df.only_strain_err.abs()})[['gid', '|full|', '|AE|', '|strain|']]
          .round(1).to_string(index=False))

    a_full = np.abs(colerr('full'))
    a_ae = np.abs(colerr('no_strain'))
    a_st = np.abs(colerr('only_strain'))
    print('\n=== Wilcoxon 配对检验(H1: 多源融合 |err| 更小) ===')
    p1, n1 = wilcoxon_onesided(a_full, a_ae)
    p2, n2 = wilcoxon_onesided(a_full, a_st)
    print(f'  D vs 仅AE   : p={p1:.3f} (n={n1})  均值|err| D={np.nanmean(a_full):.1f} vs AE={np.nanmean(a_ae):.1f}')
    print(f'  D vs 仅应变 : p={p2:.3f} (n={n2})  均值|err| D={np.nanmean(a_full):.1f} vs 应变={np.nanmean(a_st):.1f}')

    print('\n=== 漏报(C, 无不可逆onset)组数 ===')
    for m in modes:
        print(f'  {STATS_MODES[m]}: {int((df[f"{m}_grade"]=="C").sum())}')

    print('\n=== Bootstrap 95% CI(主样本 5 组, 试件重采样) ===')
    lo, hi = bootstrap_ci(a_full)
    print(f'  D  |err| 均值 CI: [{lo:.1f}, {hi:.1f}]  样本均值 {np.nanmean(a_full):.1f}')
    lead = df['full_lead'].to_numpy(float)
    lo2, hi2 = bootstrap_ci(lead)
    print(f'  D  断裂前提前量 lead 均值 CI: [{lo2:.1f}, {hi2:.1f}]  样本均值 {np.nanmean(lead):.1f}')
    print('  (小样本 n=5，CI 仅示意；结论需谨慎解读)')
    print('结果已存: results/statistical_test.csv')


# ============================================================
# 分发
# ============================================================
ROB_TASKS = {'sens': cmd_sens, 'loso': cmd_loso,
         'ablation': cmd_ablation, 'stats': cmd_stats}


def rob_main():
    ap = argparse.ArgumentParser(description='阶段③ 稳健性研究：sens / loso / ablation / stats')
    ap.add_argument('tasks', nargs='+', choices=list(ROB_TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    ap.add_argument('--workers', type=int, default=8,
                    help='多进程 worker 数(默认 8)')
    a = ap.parse_args()
    global ROB_WORKERS
    ROB_WORKERS = a.workers
    todo = list(ROB_TASKS) if 'all' in a.tasks else a.tasks
    for t in todo:
        print(f'\n########## [{t}] ##########', flush=True)
        ROB_TASKS[t]()
        print(f'########## [{t}] 完成 ##########', flush=True)


# ============================================================ 段分发 ====
STAGES = {
    'prepare':  ('阶段① 数据准备：align / weaklabels',              prep_main),
    'evaluate': ('阶段② 评估与出图：degree / warning / curves / paper', eval_main),
    'robust':   ('阶段③ 稳健性：sens / loso / ablation / stats',    rob_main),
}


def main():
    ap = argparse.ArgumentParser(
        prog='main/pipeline.py',
        description='main 主样本（016-020）主流程统一入口',
        epilog='例：python main/pipeline.py prepare align   |   '
               'python main/pipeline.py evaluate degree warning --workers 2')
    ap.add_argument('stage', nargs='?', help='段名（省略则列出全部）')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='透传给该段的参数')
    a = ap.parse_args()
    if a.stage is None:
        print('可用段（以及各自的子命令）：')
        for k, (d, _) in STAGES.items():
            print('  %-10s %s' % (k, d))
        print('\n第二级子命令的参数与用法请见各段 docstring；'
              '\n例：python main/pipeline.py evaluate degree --workers 2')
        return 0
    if a.stage not in STAGES:
        print('未知段 %r，可用：%s' % (a.stage, ', '.join(STAGES)))
        return 2
    fn = STAGES[a.stage][1]
    old = sys.argv
    sys.argv = ['main/pipeline.py %s' % a.stage] + list(a.rest)
    try:
        r = fn()
    finally:
        sys.argv = old
    return 0 if r is None else r


if __name__ == '__main__':
    sys.exit(main())
