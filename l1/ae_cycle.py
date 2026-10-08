# -*- coding: utf-8 -*-
"""把 L1 新组的**时间轴映射到循环轴** —— 用 FBG 判「加载 / 停机」，再用 n_f 自洽校验。

为什么需要它
------------
老组 C 批栽在没有 cycle 锚（"AE 时间不能线性映射到 cycle"，组间速率差 3.4 倍）。
新组的 FBG 恰好提供了一个**干净的加载指示**：实测同一帧内 10 个应变通道的峰峰值
    加载态 ≈ 1500 至 3100 µε        停机态 ≈ 3 至 16 µε
两者相差近 100 倍，判「这一帧在不在加载」几乎不会错。

重建加载状态（2026-10-02 改写，重要）
--------------------------------------
FBG 是**突发式**采集（每文件 20 s 数据，文件间隔 420 s 或 240 s），
所以不能直接把加载标签平均到 AE 帧上。旧实现要每帧 ±300 s 内凑到 **>=3 个窗**
才采信，否则赋**全局均值**，后果：
  · `g_load` 退化成少数几个离散值，且 50% 至 73% 的帧拿到同一个常数；
  · 循环轴在这些区段变成**线性时间斜坡**（而真实加载是块状）；
  · 长停录期也照样被赋全局均值 ⇒ 凭空声称停机期间在加载。

现改为 **holdgap**：把突发状态**零阶保持**到网格（最近突发，最近邻），
但离最近突发超过 `MAX_CARRY`（20 min）的帧判为**停机**。物理依据：
  · 加载本身是「几小时连续 + 停机」的分段常值信号；
  · 对分段常值信号的稀疏采样，零阶保持即最优重建；
  · FBG 停录超过 20 min 时 AE 侧也几乎无事件（L1-41 空隙内 AE 事件为 0）
    ⇒ 试验确实在暂停，应判 0 而不是外推。

五种口径的实测对比（`l1/cycle_fill_compare.py`，11 组；2026-10-05 重跑）：

    口径        反解f中位  f组间IQR  空隙内加载h  rho中位(可用组)
    mean3(旧)     1.87      0.90       5.6       0.286 (11/11)
    mean1         1.85      0.73       2.7       0.471 (11/11)
    hold          1.82      0.75       3.1       0.475 ( 8/11)
    holdcut       1.86      0.45       2.6       0.497 (10/11)
    holdgap       1.86      0.75       2.8       0.494 ( 8/11)   <- 采用
    aeact         1.83      0.72       3.4       0.490 ( 7/11)

⚠️ **两处口径说明（2026-10-05 更正，勿照抄旧数）**

1. `rho(g,AE)` 列已改成**中位 + 可用组数**。旧版报的是把**退化组**也平均进去的均值
   （L1-06 的 `g` 有 99.0 % 恒为 1、L1-13 是 99.7 %、L1-14 是 100 %），所以旧列
   系统性偏低（如 holdgap 旧写 0.397、现为 0.494）⇒ 由此得出的
   「`aeact` 略差」**不成立**（§29）。该判据的完整适用边界见 §28。
2. **`holdgap` 并不是四条判据都最优**：`holdcut` 的 IQR（0.45）与空隙内加载（2.6）
   都比它好。选 `holdgap` 的决定性理由是**物理正确** —— 纯时间阈值会在 3/7 组上
   把真实加载段切掉，见 `l1/check_carry.py` 与下表。

**采用 holdgap 最直接的证据：L1-31 的离群消失** ——
旧口径反解 2.72 Hz（偏差 -26%），新口径得 2.12 Hz，与同分组 C4 组
（1.82 至 1.91）齐平；且其加载块平均长度由 **1.0 帧**（等于无结构）
升到 **5.6 帧**，`rho(g, AE事件数)` 由 0.208 升到 0.489。

但**只用固定时间阈值还不够** —— `l1/check_carry.py` 的事后审查发现
「离最近突发 > 20 min 就判停机」在 3/7 组上切错：

    组      被截断   时长     截断帧AE率   其余帧AE率   比值     判读
    L1-13   35.0%   21.0 h    18984.5     34589.0    0.549   切错（试验在跑）
    L1-41    0.3%    0.8 h    29451.8      1523.7   19.329   切错（事件高峰段）
    L1-31    0.8%    1.3 h     7026.0     15757.8    0.446   可疑
    L1-06    0.3%    0.2 h      635.0     20493.7    0.031   确为停机
    L1-24    9.4%    6.5 h       16.3     11174.9    0.001   确为停机

真停机与「试验在跑但 FBG 恰好拦住」的比值相差 1 至 2 个数量级，
因此改为用 **AE 静默** 判停机：没有载荷循环就没有裂纹扩展，
不可能几小时一个事件都没有。加入该守卫后 L1-13 由 1.733 Hz 回到
**1.126 Hz**，分组内一致性显著改善。

自检：反解出的 f 若落在本分组名义值附近，说明 p 估得对。
实测两组各自聚得很紧 —— 分组 A（L1-06/13/14/24）**1.075 至 1.148 Hz**
（极差 0.073），分组 B（L1-25…L1-44）**1.821 至 2.119 Hz**（IQR 0.02）。
分组间差异是真实的设备/程序差异，不是重建误差（PDF 均未给出载荷频率）。

注意区分两种 FBG 记录格式
--------------------------
  · 突发式（C4 及部分分组）：`Sensors.<时间戳>.txt` 每文件 1 个突发（260 行，20 s @10 Hz）
  · 长记录：1 至 2 个连续长记录（数万行 @5 Hz）
本模块统一按「140 数据行为一窗」扫描，两种格式都覆盖。

用法
----
    python l1/ae_cycle.py                       # 全部组
    python l1/ae_cycle.py --groups L1-29        # 指定组
    python l1/ae_cycle.py --f-hz 2.0            # 指定载荷频率（默认按 n_f 反解）
    python l1/ae_cycle.py --fill mean           # 旧口径（仅供复现历史结果）

细帧格（2026-10-08 新增）
------------------------
有些组的总循环数极少（L1-34 只有 1400 cycle，试验仅 0.2 h），在 600 s 帧格上
只落下 **2 帧**，会被下游「至少 3 个有效帧」的门槛静默丢弃。`ae_frames.py` 本来
就能出 60 s 帧（`--frame-s 60 --tag 60s` → `_l1ae_frames60s_{gid}.npz`），本模块
现在也能在 60 s 格上**原生重建**循环轴 —— 不是插值：帧号本身等于 `t_epoch/frame_s`，
所以 `frame * frame_s` 直接就是绝对时刻，换帧长不需要任何重标定。

    python l1/ae_cycle.py --frame-s 60 --tag 60s --groups L1-34

⚠️ `--tag` 必须与 `--frame-s` 配套（'' 对 600、'60s' 对 60），否则读到别的帧表。
⚠️ 帧长一变，`AE_PAUSE_WIN` 需**等比缩放**（见 `_pause_win`），否则静默判据
   会从「局部约 100 min 无事」缩到「局部约 10 min 无事」，明显变严。

输出：results/_l1cyc{tag}_{gid}.npz（frame / cycle / g_load / fill / 自检信息）
"""

import collections
import datetime as _dt
import glob
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

WINDOW_LINES = 140          # 每个统计窗的数据行数（10 Hz → 14 s；5 Hz → 28 s）
P2P_LOAD = 200.0            # 加载判据：帧内应变峰峰值中位 > 该值即视为加载
F_NOMINAL = 2.0             # 名义载荷频率（Hz）
FRAME_S = 600.0

# 加载状态重建（见 _ratio_on_grid）：
BURST_GAP = 60.0            # 相邻窗前隔超过它就分成两个「突发」
MAX_CARRY = 1200.0          # 离最近突发超过它就算「长停录段」，需 AE 静默才判停机
AE_PAUSE_FRAC = 0.05        # 长停录段内局部 AE 率低于组内中位的该比例 -> 判停机
AE_PAUSE_WIN = 10           # 局部中位的半窗（帧，**以 600 s 帧为基准**）；细格按帧长等比缩放
FILL = 'holdgap'            # 默认口径；可改 'mean' 复现旧结果

# 总循环数（失效循环数 n_f）：**唯一来源 = shm.datasets**
# （原来 ae_cycle / ae_hi 各硬编码一份、l1_meta 又靠 PDF 解析，现已合并到一处）
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from shm.datasets import N_F                                        # noqa: E402


def _file_dt(path):
    """从 `Sensors.YYYYMMDDHHMMSS.txt` 取文件创建时刻（作为日期消歧的参考）。"""
    m = re.search(r'(\d{14})', os.path.basename(path))
    if not m:
        return None
    s = m.group(1)
    try:
        return _dt.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                            int(s[8:10]), int(s[10:12]), int(s[12:14]))
    except ValueError:
        return None


def _parse_ts(txt, ref=None):
    """行内时间戳 → epoch 秒。

    两种格式都存在，**必须自动消歧**：
      · L1 谱载+FBG：`03/09/2020 12:08:51.97496`（D/M/YYYY）
      · L1 变幅VA+FBG：`5/28/2020 13:29:02.94485` （M/D/YYYY）
    规则：若某一段 > 12 则它只能是“日”；两段都 ≤ 12 时，用**文件名时刻**
    当参考，两个候选解释里取更接近的那个。
    """
    try:
        d, t = txt.split()[0], txt.split()[1]
        p = d.split('/')
        a, b, yy = int(p[0]), int(p[1]), int(p[2])
        hh, mi, ss = t.split(':')
        sec = int(float(ss))
        micro = int(round((float(ss) - sec) * 1e6))
    except Exception:                                        # noqa: BLE001
        return None
    cands = [(a, b)] if a == b else [(a, b), (b, a)]
    best = None
    for mm, dd in cands:
        try:
            dt = _dt.datetime(yy, mm, dd, int(hh), int(mi), sec, micro)
        except ValueError:
            continue
        gap = abs((dt - ref).total_seconds()) if ref is not None else 0.0
        if best is None or gap < best[0]:
            best = (gap, dt)
    return best[1].timestamp() if best else None


def _iter_rows(path):
    """逐行产出 (epoch 秒, 10 通道数值)。表头行与残留行自动跳过。"""
    ref = _file_dt(path)
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
            if ts is None:
                continue
            yield ts, vals


def scan_fbg(path, window_lines=WINDOW_LINES, channels=False):
    """扫一个 FBG 文件 → [(t_epoch, p2p_median), ...] 逐窗。

    列 = Timestamp + R1..R5 + L1..L5。

    `channels=True` 时返回 [(t_epoch, p2p向量), ...] —— 逐通道峰峰值，
    用于诊断「中位数是否被部分坏通道拉低」（见 `l1/p2p_load_check.py --channels`）。
    📌 **表头结构**（实测 SM130 文件，典型共 260 行）：
      0..36   设备/配置表头
      37..46  通道标定式（17 列，**含数字但不是数据**）
      48..57  FBG 定义（15 列）
      59      列名行 `Timestamp R1 R2 ...`（21 列）
      60..259 数据（200 行 x 0.1 s = 20.0 s）

    旧版硬编码 `i < 60` **恰好正确**（数据确实从第 60 行起）。现改为**按内容
    识别**（首个「前两列能拼出时间戳且第 3 到 12 列可转浮点」的行）——
    已验证与原实现逐位相同，仅对表头长度变化的文件更稳。
    注意：不能用「字段数 >= 12」当数据判据，标定式有 17 列会被误收。
    """
    ref = _file_dt(path)
    out = []
    buf, t0 = [], None
    for ts, vals in _iter_rows(path):
        if t0 is None:
            t0 = ts
        buf.append(vals)
        if len(buf) >= window_lines:
            out.append((_win_stat_ch if channels else _win_stat)(buf, t0))
            buf, t0 = [], None
    if buf:
        out.append((_win_stat_ch if channels else _win_stat)(buf, t0))
    return [o for o in out if o is not None]


def _win_level(buf, t0):
    """返回 (t0, 逐通道**均值**向量)。均值与幅度一起看才能分出「停机」与「保载」"""
    if t0 is None or not buf:
        return None
    a = np.asarray(buf, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(all='ignore'):
            m = np.nanmean(a, axis=0)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return None
    return (t0, m)


def scan_fbg_ex(path, window_lines=WINDOW_LINES):
    """逐窗返回 (t0, p2p向量, 均值向量)。

    用途：`P2P_LOAD` 只看振荡幅度，而**真停机与保载（恒定载荷）都给出 p2p 接近 0**。
    要分开它们必须看**应变绝对水平**：保载时水平仍在载荷对应的量级，停机时趋近 0。
    """
    out = []
    buf, t0 = [], None
    for ts, vals in _iter_rows(path):
        if t0 is None:
            t0 = ts
        buf.append(vals)
        if len(buf) >= window_lines:
            a = _win_stat_ch(buf, t0)
            b = _win_level(buf, t0)
            out.append(None if (a is None or b is None) else (a[0], a[1], b[1]))
            buf, t0 = [], None
    if buf:
        a = _win_stat_ch(buf, t0)
        b = _win_level(buf, t0)
        out.append(None if (a is None or b is None) else (a[0], a[1], b[1]))
    return [o for o in out if o is not None]


def _win_stat_ch(buf, t0):
    """返回 (t0, 逐通道峰峰值向量)。非有限通道已被剔除。

    ⚠️ `np.nanmax` 遇到**整列全 NaN** 时发的是 `RuntimeWarning`（不是浮点 errstate），
    所以必须用 `warnings` 上下文压 —— 否则 PowerShell 会把 stderr 当成错误，
    让退出码变成 1（实测踩过）。
    """
    if t0 is None or not buf:
        return None
    a = np.asarray(buf, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(all='ignore'):
            d = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    return (t0, d)


def _win_stat(buf, t0):
    r = _win_stat_ch(buf, t0)
    return None if r is None else (r[0], float(np.median(r[1])))


def group_load(gid, root=None):
    """该组全部 FBG 窗的 (t, p2p) 序列，按时间排序。"""
    root = root or _data_l1()
    pts = []
    for f in sorted(glob.glob(os.path.join(root, gid, 'FBG', '*.txt'))):
        pts.extend(scan_fbg(f))
    pts.sort(key=lambda x: x[0])
    return pts


def build(gid, root=None, f_hz=None, n_f=None, frame_s=FRAME_S, fill=FILL, tag=''):
    """建循环轴：返回 dict(frame, cycle, g_load, ...)。

    `tag` 与 `frame_s` **必须配套**：'' 对 600 s 帧（`_l1ae_frames_{gid}`），
    '60s' 对 60 s 帧（`_l1ae_frames60s_{gid}`）。见模块 docstring「细帧格」。

    ⚠️ **`load_h` 的定义（2026-10-05 明确，勿误读为日历占空比）**
    ------------------------------------------------------------
        load_h = sum(g) * frame_s / 3600

    其中求和号只跑在 **AE 帧网格**上，而该网格是**稀疏**的：
    `ae_frames.py` 只为**有事件的时段**建帧（某 600 s 内一个 event 都没有就不建帧）。
    实测 L1-25 有 1668 帧，但帧号跨度 5586（缺 3918 帧），
    相邻帧间隔中位 0.17 h、**最大 576.33 h**，超过 12 h 的缺口 3 处。

    ⇒ `load_h` 统计的是「**AE 在录的时段内**的加载时长」，
      **不等于日历时长乘以加载占空比**。
    ⇒ `load_h / span_h` 也**不是**该试验的日历占空比（`span_h` 是帧时刻跨度，
      它跨越了中间那些没有帧的空洞）。
    ⇒ 但用它反解频率 `f = n_f / load_h` 是自洽的：
      分母与「事件在什么时候被记下来」共用同一张网格。
    """
    fr = _load_frames(gid, frame_s, tag)
    if fr is None:
        return None
    frame, t_sec, n_hits = fr
    pts = group_load(gid, root)
    n_f = n_f or N_F.get(gid)
    if not pts:
        g = np.ones_like(t_sec)
        src = '无 FBG，按全程加载'
    else:
        tf = np.asarray([p[0] for p in pts])
        lf = np.asarray([1.0 if p[1] > P2P_LOAD else 0.0 for p in pts])
        g = _ratio_on_grid(t_sec, tf, lf, frame_s, fill=fill, n_hits=n_hits,
                           pause_win=_pause_win(frame_s))
        src = 'FBG %d 窗，全局加载占比 %.3f，fill=%s' % (len(pts), float(lf.mean()), fill)
    span = float(t_sec[-1] - t_sec[0])
    load_s = float(np.sum(g)) * frame_s
    f_use = f_hz
    if f_use is None:
        f_use = (n_f / load_s) if (n_f and load_s > 0) else F_NOMINAL
    cycle = np.cumsum(g * frame_s * f_use)
    res = {'frame': np.asarray(frame), 't_epoch': t_sec, 'g_load': g,
           'cycle': cycle, 'f_hz': float(f_use), 'load_h': load_s / 3600.0,
           'span_h': span / 3600.0, 'n_f': n_f, 'src': src, 'fill': fill,
           'tag': tag, 'frame_s': float(frame_s)}
    return res


def _bursts(tf, lf, gap=BURST_GAP):
    """把 FBG 窗聚成「突发」（同一文件内的连续窗）。"""
    tb, sb = [], []
    i = 0
    while i < len(tf):
        j = i
        while j + 1 < len(tf) and tf[j + 1] - tf[j] <= gap:
            j += 1
        tb.append(float(tf[i:j + 1].mean()))
        sb.append(float(lf[i:j + 1].mean()))
        i = j + 1
    return np.asarray(tb), np.asarray(sb)


def _ae_silent(n_hits, win=AE_PAUSE_WIN, frac=AE_PAUSE_FRAC):
    """局部 AE 事件率是否「基本静默」—— 判定试验暂停的物理依据。

    没有载荷循环就不会有新裂纹扩展，因而不可能几小时一个 AE 事件都没有。
    实测分得很开：真停机的被截断帧事件率比正常帧低 2 至 3 个数量级
    （L1-24 为 0.001），而「试验在跑但 FBG 恰好拦住」的段位是 0.45 至 19
    （L1-13 为 0.549、L1-41 为 19.3）。
    """
    n_hits = np.asarray(n_hits, dtype=float)
    k = 2 * win + 1
    ref = float(np.median(n_hits))
    if len(n_hits) < k:
        local = n_hits
    else:
        from numpy.lib.stride_tricks import sliding_window_view
        local = np.median(sliding_window_view(np.pad(n_hits, win, mode='edge'), k),
                          axis=1)
    return local < frac * max(ref, 1.0)


def _pause_win(frame_s):
    """AE 静默判据的半窗（帧）—— 以 600 s 帧的 10 帧为基准**等比缩放**。

    判据的物理含义是「局部几小时一个事件都没有」，基准窗 = 10 x 600 s ≈ 100 min。
    改帧长时若不缩放，60 s 帧的窗会缩到 10 min，判据明显变严 ——
    会把「试验在跑、只是恰好没事件」的短段误判成停机，从而白丢真实加载。
    """
    return max(1, int(round(AE_PAUSE_WIN * FRAME_S / float(frame_s))))


def _ratio_on_grid(t_sec, tf, lf, frame_s, fill=FILL, n_hits=None, pause_win=None):
    """AE 帧网格上的加载占比。

    fill='holdgap'（默认）：把突发状态**零阶保持**到网格（最近突发，最近邻）。
        对「离最近突发超过 MAX_CARRY」的长停录段，**再看 AE 是否静默**：
          · AE 基本静默  → 试验确实在暂停，判 0；
          · AE 仍在活动  → 维持零阶保持（FBG 只是没录上，试验在跑）。
        物理依据：
          · 加载本身是「几小时连续 + 停机」的分段常值信号；
          · 对分段常值信号的稀疏采样，零阶保持即最优重建；
          · 有无载荷循环可由 AE 独立验证 —— 真停机段的事件率低 2 至 3 个数量级。
        （只用固定时间阈值会切错：L1-13 会白丢 21 h，L1-41 会把事件高峰段判成停机。）

    fill='mean'：旧口径（±frame_s/2 内 >= 3 个窗取局部均值，否则全局均值）。
        保留仅供复现历史结果 —— 它会把长空隙也赋上全局均值，
        等于凭空声称停机期间还在加载（L1-31 的 432 h 空隙被赋 47%）。
    """
    if len(tf) == 0:
        return np.ones_like(t_sec)
    if fill == 'mean':
        g = np.full(len(t_sec), np.nan)
        for i, t in enumerate(t_sec):
            m = (tf >= t - frame_s / 2) & (tf < t + frame_s / 2)
            if m.sum() >= 3:
                g[i] = lf[m].mean()
        g[~np.isfinite(g)] = float(lf.mean())
        return g
    if fill != 'holdgap':
        raise ValueError('未知 fill: %r' % fill)
    if pause_win is None:
        pause_win = AE_PAUSE_WIN
    tb, sb = _bursts(tf, lf)
    idx = np.searchsorted(tb, t_sec)
    left = np.clip(idx - 1, 0, len(tb) - 1)
    right = np.clip(idx, 0, len(tb) - 1)
    dl = np.abs(t_sec - tb[left])
    dr = np.abs(t_sec - tb[right])
    g = np.where(dl <= dr, sb[left], sb[right])
    far = np.minimum(dl, dr) > MAX_CARRY
    if n_hits is not None and len(n_hits) == len(t_sec):
        g = np.where(far & _ae_silent(n_hits, win=pause_win), 0.0, g)
    else:                                        # 无 AE 信息时退回纯时间阈值
        g = np.where(far, 0.0, g)
    return g


def _load_frames(gid, frame_s, tag=''):
    """返回 (帧序号, 帧时刻秒, 每帧 AE 事件数)。

    帧号本身 = int(t_epoch / frame_s)（见 `ae_frames.py::scan_file`），
    所以 `frame * frame_s` 就是绝对时刻，换帧长无需重标定。
    """
    p = os.path.join(RES, '_l1ae_frames%s_%s.npz' % (tag, gid))
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n_hits = np.asarray(z['n'], dtype=float) if 'n' in z.files else None
    return z['frame'], z['frame'].astype(float) * frame_s, n_hits


def _data_l1():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                        # noqa: BLE001
        pass
    return HERE


def groups_with_frames(tag=''):
    """有该帧格产物的组。

    ⚠️ **不能用前缀切片**：`_l1ae_frames60s_L1-06.npz` 切出来是 `60s_L1-06`，
    会被当成一个不存在的组名去 build（静默返回 None）。必须按正则严格匹配。
    """
    pat = re.compile(r'^_l1ae_frames%s_(L1-\d+)\.npz$' % re.escape(tag))
    out = []
    for p in glob.glob(os.path.join(RES, '_l1ae_frames*_L1-*.npz')):
        m = pat.match(os.path.basename(p))
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    gids, f_hz, fill = [], None, FILL
    frame_s, tag = FRAME_S, ''
    i = 0
    while i < len(argv):
        if argv[i] == '--f-hz':
            f_hz = float(argv[i + 1])
            i += 2
        elif argv[i] == '--fill':
            fill = argv[i + 1]
            i += 2
        elif argv[i] == '--frame-s':
            frame_s = float(argv[i + 1])
            i += 2
        elif argv[i] == '--tag':
            tag = argv[i + 1]
            i += 2
        elif argv[i] == '--groups':
            gids.extend(argv[i + 1:])
            break
        else:
            gids.append(argv[i])
            i += 1
    gids = gids or groups_with_frames(tag)
    print('加载状态重建口径 fill=%s（BURST_GAP=%.0fs, MAX_CARRY=%.0fs, 帧长 %.0fs, '
          'tag=%r, 静默半窗 %d 帧）'
          % (fill, BURST_GAP, MAX_CARRY, frame_s, tag, _pause_win(frame_s)))
    print('%-7s %6s %8s %8s %9s %11s %9s %10s' % (
        '组', '帧数', '跨度h', '加载h', '反解f(Hz)', '名义2Hz预测n_f', 'PDF n_f', '偏差@2Hz'))
    for gid in gids:
        r = build(gid, f_hz=f_hz, fill=fill, frame_s=frame_s, tag=tag)
        if r is None:
            print('%-7s 无帧表（tag=%r），跳过' % (gid, tag))
            continue
        nf = r['n_f'] or 0
        pred = r['load_h'] * 3600.0 * (f_hz or F_NOMINAL)
        dev = (pred - nf) / nf * 100 if nf else float('nan')
        np.savez_compressed(os.path.join(RES, '_l1cyc%s_%s.npz' % (tag, gid)), **r)
        print('%-7s %6d %8.1f %8.1f %9.3f %11.0f %9.0f %9.1f%%  %s'
              % (gid, len(r['frame']), r['span_h'], r['load_h'], r['f_hz'],
                 pred, nf, dev, r['src']))


if __name__ == '__main__':
    main(sys.argv[1:])
