# -*- coding: utf-8 -*-
"""L1 原始数据 → CSV（三批统一入口）
=================================================================

把 ReMAP/TU-Delft **L1 公开集**的原始 SHM 数据转成标准 CSV。三批合并在本文件内，
用 `--batch` 选分组：

| --batch | 分组              | 试件                   | 模态与口径                                                                 | 产物 |
| ------- | ----------------- | ---------------------- | -------------------------------------------------------------------------- | ---- |
| `c1`    | L1 恒幅+FBG+DFOS | L1-03/04/05/09/23      | LUNA **逐行**落盘 + FBG（`Sensors.*.txt`）+ AE（`vallenae` 列名，hit 级）  | `{gid}分布式应变.csv` / `{gid}光纤.csv` / `{gid}声发射.csv` |
| `c2`    | L1 恒幅+DFOS | L1-49 至 L1-60（9 组） | LUNA 按**测量段**聚合（段内逐位置中位/峰值/幅值）+ AE 按 **1 s bin** 聚合 | 上述三个 + `{gid}分布式应变_peak.csv` / `_amp.csv` / `{gid}dfos_anchor.csv` |

L1 恒幅+DFOS 组**没有 FBG**，且原始体量比 L1 恒幅+FBG+DFOS 组大两个数量级，所以必须段级聚合
（15 GB/组 → 15 MB/组），否则后续脚本无法加载。段间判定 `dt > --gap`（默认 60 s，
实测段内约 1 s、段间约 373 s）。`{gid}dfos_anchor.csv` 给 `l1_time_align.py`
建立 AE 与 DFOS 段与 cycle 的映射（新组无 FBG 锚）。

⚠️ L1 变幅VA/谱载（L1 变幅VA+FBG / L1 谱载+FBG，PAC/Mistras `.DTA`）不在本文件：AE 走
`l1/ae_dta.py`（自研 `.DTA` 解析器），FBG 走 `l1/fbg_tools.py profile` 系列。

用法
----
    python l1/step0.py                              # L1 恒幅+FBG+DFOS 全部（三类数据）
    python l1/step0.py --only fbg                   # 只重生成 {gid}光纤.csv
    python l1/step0.py --groups L1-05               # 指定试件（逗号分隔）
    python l1/step0.py --batch c2                   # L1 恒幅+DFOS 全部
    python l1/step0.py --batch c2 --groups L1-49 --space 1500 --gap 60
    python l1/step0.py --batch c2 --skip-ae         # 只做 DFOS
    python l1/step0.py --batch c1 --root <dir>      # 输出到别的根（输入仍读本目录）
    python l1/step0.py --batch c2 --dry-run         # 只打印计划，不写文件

⚠️ `--root` 只改**输出**根：C1 的 LUNA/FBG 输入按 `<root>/<gid>/...` 找（要么给真目录，
要么在里面建 junction），AE 与 C2 的原始输入始终读本目录（经 `ae_io` / 本脚本目录）。

说明
----
本文件由 `l1/step0.py`(539 行) 与 `l1/step0_v2.py`(350 行) 合并而来：两条流水线的
**函数体逐字保留**（只把入口改名成 `*_c1` / `*_c2`），公共脚手架（import / 路径常量 /
`print_header` / 统一的 `main`）只写一份；顺带删掉全仓无引用的死函数
`load_ae_vallenae`（`process_ae_c1` 早已改走 `ae_io.read_hits_vallenae`），
并把两组的组名单改由 `shm.datasets.L1_CAMPAIGNS` 派生。见 `docs/details.md` §33。
"""

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                    # 输出根；--root 可在 main() 里改写
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)       # shm
sys.path.insert(0, HERE)       # ae_io / l1_meta 等同级模块
from shm.paths import repo_rel  # noqa: E402  产物里只写相对路径，见该函数

# 原来 step0.py 的全局设置（保留：同一批数据在 pandas 下有大量 NaN 语义警告）
warnings.filterwarnings('ignore')

from shm.datasets import campaign_members                      # noqa: E402

# 两组的组名单：**唯一来源 = shm.datasets**（'C1'/'C2' 是短别名，语义名见那里）
C1_GROUPS = campaign_members('C1')      # L1 恒幅+FBG+DFOS
C2_GROUPS = campaign_members('C2')      # L1 恒幅+DFOS

# ==================================================================
# L1 恒幅+FBG+DFOS（L1 恒幅+FBG+DFOS，L1-03/04/05/09/23）：LUNA 逐行 + FBG + AE（vallenae 列名）
# ==================================================================
# 1st TU Delft Campaign 试件 (Paper 1: Broer et al. 2022)

# AE 传感器配置 (4个 VS900-M 传感器)
AE_SENSOR_POSITIONS = {
    1: (145, 190),   # Ch1: (145, 190) mm
    2: (145, 20),    # Ch2: (145, 20) mm
    3: (20, 50),     # Ch3: (20, 50) mm
    4: (20, 220),    # Ch4: (20, 220) mm
}

AE_WAVE_VELOCITIES = {
    'longitudinal': 5586.59,  # m/s
    'lateral': 4054.05,       # m/s
}


def to_relative_seconds(timestamps):
    """将绝对时间戳序列转换为相对开始的秒数（该路首条记录 = 0）。

    参数:
        timestamps: datetime64 数组 / datetime 序列（需已按时间升序排列）

    返回:
        np.ndarray: 相对开始的秒数 (float)；无法解析的位置为 NaN
    """
    ts = pd.to_datetime(pd.Series(timestamps), errors='coerce')
    t0 = ts.dropna().iloc[0]
    return (ts - t0).dt.total_seconds().values


# ============================================================
# 1. LUNA DFOS → CSV
# ============================================================

def load_luna(filepath):
    """
    加载 LUNA ODiSI-B DFOS 数据。

    参数:
        filepath: .txt 文件路径

    返回:
        timestamps: 时间戳数组 (pd.DatetimeIndex)
        positions:  空间位置数组 (mm)
        strain_map: 应变矩阵 (n_timestamps × n_positions), 单位 με
    """
    if not os.path.exists(filepath):
        print(f'  [警告] 文件不存在: {filepath}')
        return None, None, None

    print(f'  [LUNA] 加载: {os.path.basename(filepath)}')

    # 跳过4行文件头, 读取数据
    df = pd.read_csv(filepath, sep='\t', skiprows=4, header=0,
                     encoding='utf-8', engine='python')

    time_col = df.columns[0]

    # 提取位置信息 (mm)
    positions = df.columns[1:].astype(float).values

    # 提取时间戳
    timestamps = pd.to_datetime(df[time_col], errors='coerce')

    # 提取应变矩阵 (με)
    strain_cols = df.columns[1:]
    strain_map = df[strain_cols].values.astype(float)

    n_times, n_pos = strain_map.shape
    print(f'    时间点: {n_times}, 空间位置: {n_pos}')
    print(f'    时间范围: {timestamps.min()} ~ {timestamps.max()}')

    return timestamps, positions, strain_map


def save_luna_to_csv(timestamps, positions, strain_map, output_path):
    """
    将 LUNA 数据保存为 CSV 文件。

    CSV 格式:
      - 第1列: timestamp (相对开始的秒数, 本文件首条=0)
      - 第2~N列: 各空间位置的应变值 (με), 列名为位置值 (mm)
    """
    if timestamps is None or strain_map is None:
        print(f'  [跳过] 无数据可保存')
        return False

    # 构建 DataFrame: 绝对时间戳 → 相对开始的秒数
    df_out = pd.DataFrame(strain_map, columns=[f'{p:.2f}mm' for p in positions])
    df_out.insert(0, 'timestamp', to_relative_seconds(timestamps))

    # 保存 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_out.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def process_luna_c1(specimen_name, folder_path):
    """处理单个试件的所有 LUNA 文件，合并为一个 CSV。

    同一试件可能有多个 LUNA 文件 (如 L1-04-1.txt, L1-04-2.txt)，
    它们代表同一试件全流程的不同时间段，空间位置 (列) 一致，
    因此按时间顺序合并为一个 CSV 文件。
    """
    luna_dir = os.path.join(folder_path, 'LUNA')
    if not os.path.exists(luna_dir):
        print(f'  [跳过] LUNA 目录不存在: {luna_dir}')
        return

    txt_files = sorted([f for f in os.listdir(luna_dir)
                        if f.endswith('.txt') and f != 'LUNASetup.txt'])

    if not txt_files:
        print(f'  [跳过] 无 LUNA 数据文件')
        return

    # 合并所有文件的 DataFrame
    frames = []
    positions_ref = None
    for txt_file in txt_files:
        filepath = os.path.join(luna_dir, txt_file)
        timestamps, positions, strain_map = load_luna(filepath)

        if timestamps is None:
            continue

        # 检查空间位置是否一致
        if positions_ref is None:
            positions_ref = positions
        elif not np.allclose(positions_ref, positions):
            print(f'  [警告] {txt_file} 的空间位置与首个文件不一致，跳过该文件')
            continue

        df = pd.DataFrame(strain_map, columns=[f'{p:.2f}mm' for p in positions])
        df.insert(0, 'timestamp', timestamps)
        frames.append(df)

    if not frames:
        print(f'  [跳过] 无有效的 LUNA 数据')
        return

    # 合并、按时间排序、去重
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # 输出: {specimen}分布式应变.csv (放在试件根目录)
    output_path = os.path.join(folder_path, f'{specimen_name}分布式应变.csv')
    save_luna_to_csv(df_all['timestamp'].values, positions_ref, df_all.iloc[:, 1:].values, output_path)


# ============================================================
# 2. FBG → CSV
# ============================================================

def load_fbg_txt(filepath):
    """
    加载 Micron Optics sm130 导出的 .txt FBG 数据文件。

    参数:
        filepath: .txt 文件路径

    返回:
        timestamps: datetime 数组
        fbg_data: FBG 应变矩阵 (n_samples × 20), 单位 με
        column_names: 传感器名称列表
    """
    if not os.path.exists(filepath):
        print(f'  [警告] 文件不存在: {filepath}')
        return None, None, None

    print(f'  [FBG] 加载: {os.path.basename(filepath)}')

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # 自动查找列名行 (包含 'Timestamp' 的行)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Timestamp\t'):
            header_idx = i
            break

    if header_idx is None:
        print('    [错误] 未找到列名行 (Timestamp)')
        return None, None, None

    # 解析列名
    header_line = lines[header_idx].strip()
    column_names = header_line.split('\t')
    n_channels = len(column_names) - 1

    # 解析数据
    data_lines = lines[header_idx + 1:]
    timestamps = []
    fbg_values = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != n_channels + 1:
            continue

        # 解析时间戳 (荷兰语格式: dd-MM-yyyy HH:mm:ss.ffffff)
        try:
            ts = datetime.strptime(parts[0], '%d-%m-%Y %H:%M:%S.%f')
        except ValueError:
            try:
                ts = datetime.strptime(parts[0], '%d-%m-%Y %H:%M:%S')
            except ValueError:
                continue

        # 解析应变值 (十进制逗号 -> 点)
        try:
            values = [float(v.replace(',', '.')) for v in parts[1:]]
        except ValueError:
            continue

        timestamps.append(ts)
        fbg_values.append(values)

    if not fbg_values:
        print('    [错误] 无法解析任何数据行')
        return None, None, None

    timestamps = np.array(timestamps)
    fbg_data = np.array(fbg_values, dtype=float)

    print(f'    数据行数: {len(timestamps)}')
    print(f'    传感器: {column_names[1:]}')

    return timestamps, fbg_data, column_names[1:]


def save_fbg_to_csv(timestamps, fbg_data, column_names, output_path):
    """
    将 FBG 数据保存为 CSV 文件。

    CSV 格式:
      - 第1列: timestamp (相对开始的秒数, 本文件首条=0)
      - 第2~21列: 20个 FBG 传感器的应变值 (με)
    """
    if timestamps is None or fbg_data is None:
        print(f'  [跳过] 无数据可保存')
        return False

    # 构建 DataFrame: 绝对时间戳 → 相对开始的秒数
    df_out = pd.DataFrame(fbg_data, columns=column_names)
    df_out.insert(0, 'timestamp', to_relative_seconds(timestamps))

    # 保存 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_out.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def process_fbg_c1(specimen_name, folder_path):
    """处理单个试件的所有 FBG 文件，合并为一个 CSV。

    同一试件可能有多个 Sensors.*.txt 文件，它们代表同一试件
    全流程的不同时间段 (可能时间有重叠)，传感器列一致，
    因此合并并按时间戳去重后输出为一个 CSV 文件。
    """
    fbg_dir = os.path.join(folder_path, 'FBG')
    if not os.path.exists(fbg_dir):
        print(f'  [跳过] FBG 目录不存在: {fbg_dir}')
        return

    # 查找 Sensors.*.txt 文件
    txt_files = sorted([f for f in os.listdir(fbg_dir)
                        if f.startswith('Sensors.') and f.endswith('.txt')])

    if not txt_files:
        print(f'  [跳过] 无 FBG 数据文件')
        return

    # 合并所有文件的 DataFrame
    frames = []
    column_names_ref = None
    for fbg_file in txt_files:
        filepath = os.path.join(fbg_dir, fbg_file)
        timestamps, fbg_data, column_names = load_fbg_txt(filepath)

        if timestamps is None:
            continue

        # 检查传感器列是否一致
        if column_names_ref is None:
            column_names_ref = column_names
        elif column_names != column_names_ref:
            print(f'  [警告] {fbg_file} 的传感器列与首个文件不一致，跳过该文件')
            continue

        df = pd.DataFrame(fbg_data, columns=column_names)
        df.insert(0, 'timestamp', pd.to_datetime(timestamps))
        frames.append(df)

    if not frames:
        print(f'  [跳过] 无有效的 FBG 数据')
        return

    # 合并、按时间排序、去重
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # 输出: {specimen}光纤.csv (放在试件根目录)
    output_path = os.path.join(folder_path, f'{specimen_name}光纤.csv')
    save_fbg_to_csv(df_all['timestamp'].values, df_all.iloc[:, 1:].values,
                    column_names_ref, output_path)


# ============================================================
# 3. AE → CSV
# ============================================================



def save_ae_to_csv(df_ae, output_path):
    """
    将 AE hit 参数保存为 CSV 文件。

    CSV 格式:
      - 包含所有 vallenae 返回的 hit 参数字段
      - amplitude 和 threshold 已转换为 dB 单位
    """
    if df_ae is None or len(df_ae) == 0:
        print(f'  [跳过] 无 AE 数据可保存')
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ae.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_ae.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def _v_to_db(df):
    """把 `amplitude` / `threshold` 从 V 换成 dB（20·log10(V/1µV)）。就地改并返回。"""
    V_ref = 1e-6
    for c in ('amplitude', 'threshold'):
        if c in df.columns:
            df[c] = 20 * np.log10(df[c] / V_ref)
    return df


def process_ae_c1(specimen_name, folder_path):
    """处理单个试件的所有 AE 文件，合并为一个 CSV。

    ⚠️ **多段 `.pridb` 的会话缝合**：`.pridb` 的 `time` 是**各次采集自己的起算秒数**
    （采集中断重开会从 ~0 重算），不是墙钟。旧实现 `concat` → `drop_duplicates('time')`
    → `sort_values('time')` 假设两段"时间有重叠⇒有重复"，但实测 L1-04 两段事件
    **零重合**（是两段独立会话），去重无效、排序交错 ⇒ AE 只覆盖 cycle 61.3%、
    且后期高活跃段被错放到 cycle 110k–161k（与论文"前 240k cycles 几乎无 AE"矛盾）。
    现交由 `ae_io.plan()` 判定「按 Time 合并」还是「按会话顺序缝合」，见
    `ae_io` 模块 docstring 与 docs/details.md §17.11。

    ⚠️ **单段走流式**（2026-10-09）：L1-23 有 3261 万条 hit，整体路径的
    `PriDatabase.read_hits()` 要先物化成约 13 GB，在 15.4 GB 机器上直接
    `_ArrayMemoryError`。单段不存在跨段排序/去重需求 ⇒ 交给
    `ae_io.write_hits_vallenae_csv()` 按块落盘，内存恒定、产物逐字节等价
    （已验证：L1-03 两条路径的 CSV 完全一致）。
    """
    ae_dir = os.path.join(folder_path, 'AE')
    if not os.path.exists(ae_dir):
        print(f'  [跳过] AE 目录不存在: {ae_dir}')
        return

    import sys as _sys
    _h = os.path.dirname(os.path.abspath(__file__))
    if _h not in _sys.path:
        _sys.path.insert(0, _h)
    import ae_io

    output_path = os.path.join(folder_path, f'{specimen_name}声发射.csv')

    # 单段 ⇒ 流式落盘
    try:
        _p = ae_io.plan(specimen_name, 'auto', verbose=False)
    except Exception:                                            # noqa: BLE001
        _p = None
    if _p is not None and len(_p.get('parts', [])) == 1:
        n, cols, pl = ae_io.write_hits_vallenae_csv(
            specimen_name, output_path, transform=_v_to_db)
        if n == 0:
            print(f'  [跳过] 无有效的 AE 数据')
            return
        print(f'  [保存] {output_path}')
        print(f'    形状: ({n}, {len(cols)}), '
              f'大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
        return

    df_all, pl = ae_io.read_hits_vallenae(specimen_name)
    if df_all is None or len(df_all) == 0:
        print(f'  [跳过] 无有效的 AE 数据')
        return
    print(f'    mode={pl["mode"]}  hits={len(df_all):,}  '
          f'time={df_all["time"].min():.1f}~{df_all["time"].max():.1f} s  '
          f'通道: {sorted(df_all["channel"].unique())}')

    # 振幅转换: V → dB (20 * log10(V / 1µV))
    df_all = _v_to_db(df_all)

    # 输出: {specimen}声发射.csv (放在试件根目录)
    save_ae_to_csv(df_all, output_path)


# ============================================================
# 4. 主处理流程
# ============================================================

def print_header(title):
    """打印格式化的章节标题。"""
    print()
    print('=' * 60)
    print(title)
    print('=' * 60)
    print()


# ==================================================================
# L1 恒幅+DFOS（L1 恒幅+DFOS，L1-49 至 L1-60，无 FBG）：LUNA 段级聚合 + AE 1 s 分箱
# ==================================================================
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                       # noqa: BLE001
    pass


# L1 恒幅+DFOS试件（2026-09-14 新增）：无 FBG，AE + DFOS(ODiSi-B)

GAP_S = 60.0          # 段间最小间隔（s）；实测段间 ≈373 s、段内 ≈1 s
N_SPACE = 1500        # 空间降采样点数
AE_BIN_S = 1.0        # AE 聚合 bin（s）
AE_CHUNK = 2_000_000  # sqlite 分批读取行数
TIME_BASE = 1e7       # .pridb Time 单位: 100 ns ticks
TS_SKIPROWS = 4       # LUNA 文件头 4 行（第 5 行为位置行）


# ============================================================
# LUNA / DFOS
# ============================================================
def luna_positions(fp):
    """读第 5 行（位置, mm）。返回 np.ndarray。"""
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        for _ in range(TS_SKIPROWS):
            f.readline()
        raw = f.readline().rstrip('\n')
    cells = raw.split('\t')[1:]
    vals = []
    for c in cells:
        try:
            vals.append(float(c))
        except ValueError:
            pass
    return np.asarray(vals, float)


def pick_cols(n_pos, n_keep):
    """等距抽取 n_keep 个空间索引（保持覆盖全长度）。"""
    if n_pos <= n_keep:
        return np.arange(n_pos)
    return np.unique(np.linspace(0, n_pos - 1, n_keep).round().astype(int))


def foot_mask_of(gid, pos_keep):
    """保留列中的「加强筋脚」掩码（用 l1_meta 从 PDF 解析的空间段）。

    用于判定段内**压缩峰值载荷行**：DFOS 行结构为 valley(卸载)/peak(最大压缩) 交替，
    取两脚应变均值**最负**的那一行（同 reproduce_broer_l1.dfos_binned_mean 口径）。
    两脚均缺时回退 None（改用全体位置均值）。
    """
    try:
        from l1_meta import load_meta
        m = load_meta(gid)
    except Exception as e:                                       # noqa: BLE001
        print(f'    [警告] l1_meta 不可用({e})，peak 行改用全体位置均值')
        return None
    fm = np.zeros(pos_keep.size, bool)
    for seg in (m.get('foot_L'), m.get('foot_R')):
        if seg:
            fm |= (pos_keep >= seg[0]) & (pos_keep <= seg[1])
    return fm if fm.any() else None


def _seg_stats(arr, fmask):
    """段内统计 → (中位分布, 压缩峰值行分布, 循环幅值分布)。

    amp = 逐位置 **(p90 − p10)**：ODiSi-B 采样 1 Hz vs 加载 2 Hz → **欠采样**，
    段内 120 行采到不同相位；取“最负单行”会被相位偶然主导
    （实测：peak 口径的寿命相关性 ρ = −0.40~0.33，比中位的 0.46~1.00 差很多）。
    故用**段内分布**的极差作为循环幅值 —— 与 L1 恒幅+FBG+DFOS 的 FBG「块内 std」同语义
    （载荷控制下 幅值 ∝ 1/刚度）。
    """
    med = np.nanmedian(arr, axis=0)
    score = (np.nanmean(arr[:, fmask], axis=1) if fmask is not None
             else np.nanmean(arr, axis=1))
    peak = (med.copy() if not np.isfinite(score).any()
            else arr[int(np.nanargmin(score))].copy())
    hi = np.nanpercentile(arr, 90, axis=0)
    lo = np.nanpercentile(arr, 10, axis=0)
    return med, peak, (hi - lo).astype(np.float32)


def luna_segments(fp, keep_idx, gap_s=GAP_S, fmask=None):
    """流式分段：返回 (t_start, t_end, prof_med, prof_peak, prof_amp, nrow)。

    prof_med  = 段内逐位置**中位数**（抗掉点）→ 空间分布 / DFOS HI 分析
    prof_peak = 段内**压缩峰值载荷行**分布 → 参考（单行口径，实测不如 amp）
    prof_amp  = 段内逐位置 **(p90−p10)** 循环幅值 → D(t)/RUL 应变证据首选
    用 pandas 分块读（C 解析器），避免 Python 逐行 float 解析。
    """
    n_pos = luna_positions(fp).size
    usecols = [0] + [i + 1 for i in keep_idx]
    t_start, t_end, nrow = [], [], []
    pmed, ppeak, pamp = [], [], []
    buf, t0, t1 = [], None, None

    def flush():
        if not buf:
            return
        arr = np.vstack(buf)
        m, p, a = _seg_stats(arr, fmask)
        t_start.append(t0); t_end.append(t1); nrow.append(len(buf))
        pmed.append(m); ppeak.append(p); pamp.append(a)

    reader = pd.read_csv(fp, sep='\t', skiprows=TS_SKIPROWS, header=0,
                         usecols=usecols, chunksize=4000, engine='c',
                         on_bad_lines='skip')
    for chunk in reader:
        ts_col = chunk.columns[0]
        tv = pd.to_datetime(chunk[ts_col], errors='coerce')
        tv = tv.to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9
        vals = chunk.iloc[:, 1:].to_numpy(np.float32)
        for i in range(len(tv)):
            t = tv[i]
            if not np.isfinite(t):
                continue
            if t1 is not None and (t - t1) > gap_s:
                flush()
                buf = []
                t0 = t
            if t0 is None:
                t0 = t
            buf.append(vals[i])
            t1 = t
    flush()
    emp = np.zeros((0, len(keep_idx)), np.float32)
    return (np.asarray(t_start, float), np.asarray(t_end, float),
            np.asarray(pmed, np.float32) if pmed else emp,
            np.asarray(ppeak, np.float32) if ppeak else emp,
            np.asarray(pamp, np.float32) if pamp else emp,
            np.asarray(nrow, int))


def luna_files(gid):
    """返回 {阶段: [文件路径,...]}；阶段 ∈ {'AI','BI','main'}。

    **已核实（2026-09-14，据数据集官方描述 + 实测）**：
      AI = **A**fter **I**mpact（冲击后）、BI = **B**efore **I**mpact（冲击前）
      —— 而不是“同一光纤两端解调”（早期推断，已废止）。
    旁证：L1-49 的 BI 仅 65 min /1,293 行（对应 5000 cycles 预疲劳 + 测量暂停，
    其结束时刻即冲击时刻），AI 则长 21.5 h /11,449 行。
    故两组都缺一不可：BI 提供**真正的健康基线**，AI 提供损伤演化段。
    """
    d = os.path.join(ROOT, gid, 'LUNA')
    if not os.path.isdir(d):
        return {}
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith('.txt') or fn == 'LUNASetup.txt':
            continue
        up = fn.upper()
        if '_AI' in up:
            out.setdefault('AI', []).append(os.path.join(d, fn))
        elif '_BI' in up:
            out.setdefault('BI', []).append(os.path.join(d, fn))
        else:
            out.setdefault('main', []).append(os.path.join(d, fn))
    return out


def process_luna_c2(gid, n_space=N_SPACE, gap_s=GAP_S):
    files = luna_files(gid)
    if not files:
        print(f'  [{gid}] 无 LUNA 数据，跳过')
        return None
    # 主阶段 = AI（冲击后）；**BI（冲击前）也纳入** → 恢复完整时间轴与健康基线
    #   旧版只取 AI 并跳过 BI，丢掉了冲击前基线（2026-09-14 修正）。
    key = 'AI' if 'AI' in files else ('main' if 'main' in files else 'BI')
    tasks = [(key, f) for f in files[key]]
    for k, v in files.items():
        if k != key:
            tasks += [(k, f) for f in v]
    pos = luna_positions(tasks[0][1])
    keep = pick_cols(pos.size, n_space)
    fmask = foot_mask_of(gid, pos[keep])
    phases = '/'.join(sorted({p for p, _ in tasks}))
    print(f'  [{gid}] LUNA 阶段={phases} 文件数={len(tasks)} '
          f'空间 {pos.size}→{keep.size} 点  脚位掩码点={0 if fmask is None else int(fmask.sum())}')

    T, TEND, PM, PP, PA, N, PH = [], [], [], [], [], [], []
    for phase, fp in tasks:
        p2 = luna_positions(fp)
        if p2.size != pos.size or not np.allclose(p2, pos, atol=1e-6):
            print(f'    [警告] {os.path.basename(fp)} 空间网格不一致，跳过')
            continue
        ts, te, pm, pp, pa, nr = luna_segments(fp, keep, gap_s, fmask)
        print(f'    [{phase}] {os.path.basename(fp)}: {nr.size} 段 '
              f'(行数中位={int(np.median(nr)) if nr.size else 0})')
        T.append(ts); TEND.append(te)
        PM.append(pm); PP.append(pp); PA.append(pa); N.append(nr)
        PH.append(np.full(nr.size, phase, dtype=object))
    if not T:
        return None
    ts = np.concatenate(T); te = np.concatenate(TEND); nr = np.concatenate(N)
    pm = np.vstack(PM); pp = np.vstack(PP); pa = np.vstack(PA)
    ph = np.concatenate(PH)
    order = np.argsort(ts, kind='stable')
    ts, te, pm, pp, pa, nr, ph = (ts[order], te[order], pm[order], pp[order],
                                  pa[order], nr[order], ph[order])
    uniq = np.concatenate([[True], np.diff(ts) > 1e-6])      # BI/AI 边界去重叠
    ts, te, pm, pp, pa, nr, ph = (ts[uniq], te[uniq], pm[uniq], pp[uniq],
                                  pa[uniq], nr[uniq], ph[uniq])

    # 输出 3 个口径（列名 = {pos:.2f}mm，与 load_dfos 接口一致；timestamp = 相对首段秒数）:
    #   {gid}分布式应变.csv        段内中位        → 空间分布 / DFOS HI
    #   {gid}分布式应变_peak.csv   段内峰值载荷行  → 参考（单行，实测不如 amp）
    #   {gid}分布式应变_amp.csv    段内(p90−p10)   → 循环幅值（与L1 恒幅+FBG+DFOS FBG 块内 std 同语义）
    cols = [f'{p:.2f}mm' for p in pos[keep]]
    rel = ts - ts[0]
    for tag, mat in (('', pm), ('_peak', pp), ('_amp', pa)):
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, 'timestamp', rel)
        out = os.path.join(ROOT, gid, f'{gid}分布式应变{tag}.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'    → {os.path.basename(out)} ({os.path.getsize(out)/1e6:.2f} MB)')
    # 输出: {gid}dfos_anchor.csv （段序/阶段/起始墙钟/相对秒/段时长/行数）
    anc = pd.DataFrame({
        'seg': np.arange(len(ts)),
        'phase': ph,
        'wall_iso': pd.to_datetime(ts, unit='s').strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'rel_s': rel, 'seg_span_s': te - ts, 'n_rows': nr,
    })
    out2 = os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')
    anc.to_csv(out2, index=False, encoding='utf-8-sig')
    n_bi = int((ph == 'BI').sum()); n_ai = int((ph == 'AI').sum())
    print(f'    段数={len(ts)} (BI={n_bi} / AI={n_ai})  跨度={(ts[-1]-ts[0])/3600:.2f} h  '
          f'有效段(≥30行)={int((nr >= 30).sum())}')
    if n_bi and n_ai:
        i0 = int(np.argmax(ph == 'AI'))
        print(f'    冲击分界(BI→AI) = 段 {i0} @ '
              f'{pd.to_datetime(ts[i0], unit="s").strftime("%m-%d %H:%M:%S")}')
    return dict(gid=gid, seg=len(ts), rows=int(nr.sum()), n_bi=n_bi, n_ai=n_ai)


# ============================================================
# AE (.pridb)
# ============================================================
def process_ae_c2(gid, bin_s=AE_BIN_S):
    d = os.path.join(ROOT, gid, 'AE')
    if not os.path.isdir(d):
        print(f'  [{gid}] 无 AE 目录，跳过')
        return None
    fl = sorted(f for f in os.listdir(d) if f.endswith('.pridb'))
    if not fl:
        return None
    # --- 多段 .pridb 会话缝合（见 ae_io / docs/details.md §17.11）---
    import ae_io
    pl = ae_io.plan(gid)
    print(ae_io.describe(gid))
    segs = [(s['path'], o) for s, o in zip(pl['parts'], pl['offsets'])]
    frames = []
    for fp, off in segs:
        con = sqlite3.connect(f'file:{fp}?mode=ro', uri=True)
        n = 0
        parts = []
        sql = 'SELECT Time, Chan, Eny, Amp FROM ae_data WHERE SetType=2'
        for chunk in pd.read_sql_query(sql, con, chunksize=AE_CHUNK):
            n += len(chunk)
            t = chunk['Time'].to_numpy('float64') / TIME_BASE + off
            e = chunk['Eny'].fillna(0.0).to_numpy('float64')
            a = chunk['Amp'].fillna(0.0).to_numpy('float64')
            bi = np.floor(t / bin_s).astype('int64')
            g = pd.DataFrame({'bi': bi, 'energy': e, 'amplitude': a, 'one': 1}) \
                .groupby('bi', sort=True).agg(
                    energy=('energy', 'max'), amplitude=('amplitude', 'max'),
                    n_hits=('one', 'sum'))
            parts.append(g)
        con.close()
        print(f'    {os.path.basename(fp)}: {n:,} hits (offset {off:.0f}s)')
        if not parts:
            continue
        g = pd.concat(parts).groupby(level=0).agg(
            energy=('energy', 'max'), amplitude=('amplitude', 'max'),
            n_hits=('n_hits', 'sum')).reset_index()
        g = g.rename(columns={'bi': 'time'})
        g['time'] = g['time'] * bin_s
        frames.append(g[['time', 'n_hits', 'energy', 'amplitude']])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    out = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'    hits 总数={int(df["n_hits"].sum()):,}  bin 数={len(df)}  '
          f'time={df["time"].min():.0f}~{df["time"].max():.0f}s '
          f'→ {os.path.basename(out)} ({os.path.getsize(out)/1e6:.2f} MB)')
    return dict(gid=gid, n_bin=len(df), t_max=float(df['time'].max()))

# ==================================================================
# 统一入口
# ==================================================================
def run_c1(groups, only='all', dry=False):
    """L1 恒幅+FBG+DFOS：LUNA + FBG + AE 三类（only='all'/'luna'/'fbg'/'ae'）。"""
    for gid in groups:
        print_header(f'--- 试件: {gid} ---')
        folder = os.path.join(ROOT, gid)
        if only in ('all', 'luna'):
            print('[LUNA DFOS]')
            if not dry:
                process_luna_c1(gid, folder)
        if only in ('all', 'fbg'):
            print('[FBG]')
            if not dry:
                process_fbg_c1(gid, folder)
        if only in ('all', 'ae'):
            print('[AE]')
            if not dry:
                process_ae_c1(gid, folder)


def c1_summary(groups):
    """L1 恒幅+FBG+DFOS处理后的产物小结。"""
    print_header('处理摘要')
    total_size, n_csv = 0.0, 0
    for specimen in groups:
        for suffix in ('声发射.csv', '光纤.csv', '分布式应变.csv'):
            fp = os.path.join(ROOT, specimen, f'{specimen}{suffix}')
            if os.path.exists(fp):
                size_mb = os.path.getsize(fp) / 1e6
                total_size += size_mb
                n_csv += 1
                print(f'  {fp}  ({size_mb:.2f} MB)')
    print()
    print(f'  共生成 {n_csv} 个 CSV 文件')
    print(f'  总大小: {total_size:.2f} MB')


def run_c2(groups, space=None, gap=None, skip_ae=False, skip_dfos=False,
           dry=False):
    """L1 恒幅+DFOS：DFOS 段级聚合 + AE 1 s 分箱。"""
    space = N_SPACE if space is None else space
    gap = GAP_S if gap is None else gap
    for g in groups:
        print(f'--- {g} ---')
        if not skip_dfos:
            print('  [DFOS]')
            if not dry:
                process_luna_c2(g, n_space=space, gap_s=gap)
        if not skip_ae:
            print('  [AE]')
            if not dry:
                process_ae_c2(g)
        print()
    print('完成')


def main():
    global ROOT
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser(
        description='Step0: L1 原始数据 → CSV（三批统一入口）')
    ap.add_argument('--batch', default='c1', choices=['c1', 'c2'],
                    help='c1 = L1 恒幅+FBG+DFOS（L1-03/04/05/09/23）/ '
                         'c2 = L1 恒幅+DFOS（L1-49 至 L1-60，无 FBG）')
    ap.add_argument('--groups', default=None,
                    help='逗号分隔试件号；省略=该批全部')
    ap.add_argument('--only', default='all',
                    choices=['all', 'ae', 'luna', 'fbg'],
                    help='[c1] 只处理某一类数据（重生成 {gid}声发射.csv 用 --only ae）')
    ap.add_argument('--space', type=int, default=None,
                    help=f'[c2] DFOS 空间降采样点数（默认 {N_SPACE}）')
    ap.add_argument('--gap', type=float, default=None,
                    help=f'[c2] LUNA 段间最小间隔(s)（默认 {GAP_S:g}）')
    ap.add_argument('--skip-ae', action='store_true', help='[c2] 跳过 AE')
    ap.add_argument('--skip-dfos', action='store_true', help='[c2] 跳过 DFOS')
    ap.add_argument('--root', default=None,
                    help='输出根（默认本脚本目录）；输入仍读本目录')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不写文件')
    a = ap.parse_args()

    default_groups = C1_GROUPS if a.batch == 'c1' else C2_GROUPS
    groups = ([g.strip() for g in a.groups.split(',') if g.strip()]
              if a.groups else list(default_groups))
    if a.root:
        ROOT = os.path.abspath(a.root)

    print('=' * 60)
    print('Step 0: 原始数据 → CSV')
    print('=' * 60)
    print(f'分组      : {a.batch}  ({SECTION_TITLE[a.batch]})')
    print(f'试件      : {" ".join(groups)}')
    print(f'输出根    : {repo_rel(ROOT)}')
    print(f'输入根    : {repo_rel(HERE)}')
    if a.dry_run:
        print('模式      : dry-run（不写文件）')
    print()

    if a.batch == 'c1':
        run_c1(groups, a.only, a.dry_run)
        if not a.dry_run:
            c1_summary(groups)
    else:
        run_c2(groups, a.space, a.gap, a.skip_ae, a.skip_dfos, a.dry_run)

    print()
    print('=' * 60)
    print('数据预处理完成!' if not a.dry_run else '（dry-run 结束，未写文件）')
    print('=' * 60)


SECTION_TITLE = {
    'c1': 'L1 恒幅+FBG+DFOS（LUNA 逐行 + FBG + AE）',
    'c2': 'L1 恒幅+DFOS（DFOS 段级 + AE 1 s 分箱，无 FBG）',
}

if __name__ == '__main__':
    main()
