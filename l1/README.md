# l1：ReMAP / TU-Delft L1 公开集

公开数据，用来回答「损伤指标能不能跨试件、跨机构归一化到同一根轴上」。
29 组试件按**数据本身**（AE 采集格式、有无 FBG/DFOS、加载谱、AE 时钟差）分四批，**C1–C4 只是短别名**，
分组真源是仓库根的 [`datasets.py`](../datasets.py) 的 `L1_CAMPAIGNS`。

L1 **只交付离线复评与论文复现口径**：不支持跨试件统一阈值的在线预警（相关脚本与看板包已于 2026-10-07 下线）。

## 1. 数据

| 批次 | 试验类型 | 组数 | 传感 | 试件 |
| --- | --- | --- | --- | --- |
| **C1** 恒幅+FBG+DFOS | 恒幅 | 5 | AE + FBG + DFOS + PZT | L1-03 / 04 / 05 / 09 / 23 |
| **C2** 恒幅+DFOS | 恒幅 | 9 | AE + DFOS | L1-49 / 50 / 51 / 52 / 54 / 55 / 56 / 59 / 60 |
| **C3** 变幅VA+FBG | 变幅 VA | 5 | AE + FBG + PZT | L1-06 / 13 / 14 / 24 / 22 |
| **C4** 谱载+FBG | 谱载 | 10 | AE + FBG + PZT | L1-25 / 27 / 29 / 30 / 31 / 34 / 35 / 36 / 41 / 44 |

合计 **29 组**；导波 PZT 另有 20 组有数据。规模约 39 GB，数据在外接盘，仓库内是目录联接。

几条必须先知道的：

- **L1-22 / L1-23** 由数据方补交，签名分别与 C3 / C1 一致；L1-23 不是单级恒幅（10 万循环后提载，438k 循环）。
- **C4 的 AE 时钟比 C3 晚 8 天**（`clock_shift_days`），跨批比较前必须按 `datasets.py` 平移。
- **L1-34 极短**（1400 cycle）在 600 s 帧网格上只落 2 帧，指标自动回落到 60 s 细格（见 `ae_hi.py`）。

## 2. 方法：HI 与 t85

**主指标** `HI = unity01(累积 AE hits)`：C1/C3/C4 记作 `HI_hit`，C2 记作 `HI_AE`。
自归一化曲线，`t85` = 曲线达到终值 85% 时的寿命分数。

⚠️ **`HI` 是离线复评**：自归一化用到全寿命终值，**非因果**；`t85` 是曲线形状指标，**不是预测时刻**。

## 3. 结果

### 3.1 四批 t85

> 表中四批统一用 `cum_hits` 口径（循环轴、29 组全量、不剔除）；C2 的交付口径仍是 `HI_AE`。

| 批次 | 试验类型 | 组数 | 传感 | t85 中位 |
| --- | --- | --- | --- | --- |
| C1 恒幅+FBG+DFOS | 恒幅 | 5 | AE + FBG + DFOS + PZT | **89.2%** |
| C2 恒幅+DFOS | 恒幅 | 9 | AE + DFOS | **92.1%** |
| C3 变幅VA+FBG | 变幅 VA | 5 | AE + FBG + PZT | **94.0%** |
| C4 谱载+FBG | 谱载 | 10 | AE + FBG + PZT | **94.1%** |

### 3.2 论文复现（Broer 2021）

Level 1 / 3a / 4 **均复现**：

- `HI_F` 5 组全部 0 → 1；
- L1-04 的 `HI_F` t85 为 **94.3%**，与论文正文可推出的 > 92.8% 自洽；
- Level-1 检测点 **240.0k / 260.5k** 对论文 **239.5k / 260.0k**（各差 500 cycle）。

Level 3b 脱粘**未复现**。

### 3.3 形状类证据

唯一对载荷级缩放不变、可跨批比较的量：

- **FBG 剖面 `shape_rob25`**：暖机后 **11 正 / 0 负 / 2 弱**，|ρ| 中位 **0.70**；
- **DFOS 块级 HI**：C1 五组末值全为 **1.000**、尾段单调 **100%**。

### 3.4 跨机构可比性

两个机构（TU-Delft / UPatras）、两套采集系统：

- **横轴** = 状态序号 × 文档记载的采集间隔 ÷ `n_f`，与试验机循环记录秩相关 **13/14 组 ≥ 0.993**；
- **纵轴**用**无标度**口径 SNHI：绝对阈值 **6/15 漏报**、组内极差中位 **60.7 pp**；
  SNHI **0/15 漏报**、**18.5 pp**；
- 置换检验下的方差占比 **η²**：机构对 AE t50 ≈ **0**（p=0.027）、实验室对 PZT 斜率 ≈ **0**（p=0.004），
  AE t85 残留 0.057（p=0.78，不显著）——归一化后无系统性机构偏差。

> *原「批间/组内比值 AE 0.33、PZT 0.29」经置换检验不具判别力（零分布中位 0.44，p=0.18/0.21），
> 已撤下，见 `crosslab_variance.py`。*

### 3.5 载荷可迁移性

变幅 / 谱载下**不存在可迁移的归一化**——敏感性本身跨试件不可迁移
（拟合指数 `m_eff` 从 −0.20 到 +9.05）。
因此变幅下交付的是**形状量**，而不是「把幅值归一化掉」。

### 3.6 AE 事件定位

- **X 向可用**（跨组 σ ≈ 14 mm）；
- **Y 向不可用**（σ ≈ 46 mm，加筋条把波导成非直线）；
- 残余误差主因是**探头通道延迟**而非声速或几何：把 δ 显式建进走时模型后，9 组走时残差收敛到 **0.45–2.37 µs**。

## 4. 边界

- **`HI` 是离线复评**（自归一化用到全寿命终值，非因果），`t85` 是形状指标，不是预测时刻。
- 阈值 0.25 / 0.55 / 0.85 **只是分档刻度**，两级之间不承诺触发间隔，也不代表物理裕度。
- **不做的事**：Y 向分区、把形状配准当 RUL 预测器（已证伪）、声速反演（拿错了自由度）、
  导波在线因果化（不可行）、分布式应变空间定位（已证伪）。理由与实测见 `docs/details.md` 附录 A。
- **`l1/results/l1_fusion*.csv` 是孤儿产物**：`l1/` 已无对应生成脚本（全仓只有 `main/attic/verify.py`
  的同名 `main/` 产物），**不可复现**，引用时必须声明为历史产物。

## 5. 怎么跑

```bat
cd l1
python run.py --list                            rem 任务矩阵（按主题分组）
python run.py --task hi-ae                      rem 离线复评 HI_AE（C2 主指标）
python run.py --task hi-hit                     rem 离线复评 HI_hit（C3/C4 主指标）
python run.py --task paper                      rem 论文 Level 1 / 4 复现
python run.py --task paper-l23                  rem 论文 Level 2 / 3 复现
python run.py --task crosslab-var               rem 方差分解 η² + 置换零（Table 1/2 唯一来源）
python run.py --task fig-routes --dry-run       rem 只打印将执行的命令
python run.py --task paper -- --groups L1-03    rem '--' 之后透传给底层脚本
```

- `--task all` 会把**重的任务**（训练、全量重建缓存）一起跑，先 `--dry-run` 看清。
- 定位相关脚本**先跑一次 `ae-cache`**：它把 9 组 `.pridb`（9.54 GB、1.42 亿命中）固化成二进制缓存，
  之后读取从数百秒降到 0.02 秒；缓存可删可重建。
- 首次准备数据：`--task prepare`（C1）与 `--task prepare-c2`（C2）把原始 `.pridb`/`.txt` 转成 CSV。

## 6. 产物位置

| 产物 | 位置 |
| --- | --- |
| 结果表 | `l1/results/*.csv`、`l1/*.csv`（如 `AE特征概览.csv`） |
| 中间缓存 | `l1/results/*.npz`（帧、循环轴、FBG 剖面、AE 逐 hit 缓存） |
| 论文图 | `l1/figures/fig_*.png` / `.pdf`（300 dpi） |
| 看板数据包 | `../dashboard/data/`（由 `--task dashboard` 生成，不入库） |

## 7. 脚本索引

| 主题 | 脚本 |
| --- | --- |
| 读入 | `step0.py`（原始 → CSV）、`ae_io.py` / `ae_dta.py` / `ae_cycle.py` / `ae_frames.py`（AE 三种格式）、`l1_meta.py` / `pdf_specimen_meta.py` / `survey_groups.py`（元信息） |
| 指标 | `ae_hi.py`（HI_hit）、`evaluate_l1_hi_ae.py`（HI_AE）、`evaluate_l1_fbg.py`、`evaluate_l1_dfos.py`、`fbg_tools.py` |
| 复现与对照 | `reproduce_broer_l1.py`、`reproduce_broer_l23.py`、`hi_crosslab.py`、`hi_l123.py`、`crosslab_variance.py`、`level_vs_shape.py`、`threshold_sweep.py`、`four_modal.py`、`snhi_shape.py`、`snhi_rul.py`、`paper_transferability.py` |
| 导波 PZT | `pzt_crosslab.py`、`pzt_sensor_fail.py`、`pzt_causal.py`、`pzt_locadiff_probe.py`、`pzt_index.py` |
| 定位 | `ae_cache.py`、`loc_bias_probe.py`、`loc_vel_calib.py`、`loc_debias.py`、`loc_debias_drift.py`、`loc_early.py`、`ae_locate.py` |
| 图与表 | `figs.py`、`four_modal.py`、`make_tables_45.py`、`paper_style.py` |
| 看板重建 | `restore_dashboard_packages.py`、`build_campaign_index.py` |
