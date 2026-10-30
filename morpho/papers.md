# MORPHO 相关论文调研

> 目的：搞清楚官方在这批数据上到底做了什么，我们哪些能复现、哪些复现不了、为什么。
> 所有条目都经过 Zenodo/Crossref/HAL 的实际检索核对，检索日期 2026-10-30。

## 1. 论文清单

### P1（核心，必读）实验与数据分析

**Galanopoulos G., Paunikar S., Stamatelatos G., Loutas T., Mechbal N., Rébillat M., Zarouchas D.**
*SHM for Complex Composite Aerospace Structures: A Case Study on Engine Fan Blades.*
**Aerospace 2025, 12(11), 963.** DOI `10.3390/aerospace12110963`｜HAL `hal-05554207`｜CC-BY 4.0

这是这批 FOD 板的**主论文**：描述了五块板、四点弯曲多级加载、以及三类传感数据的分析。
我们复现的对象就是这篇（见 §3）。

### P2 数据集说明

**Paunikar S., Galanopoulos G., Rébillat M., et al.**
*An experimental data set for the SHM of a substructure of an engine fan blade from the MORPHO project.*
Zenodo `10.5281/zenodo.14627730`（2025-01-10，CC-BY 4.0）

仓里 `morpho/data/ReadMe.pdf` 就是这份的说明文档。**只含 `MORPHO_FOD7.h5` 一块板。**

### P3 50 kHz 派生集与主动学习

**Stamatelatos G., Dimitriou D. K., Loutas T.**
*Active deep learning with operator self-correction for guided-wave SHM dataset curation.*
**Measurement, 2026**（卷期待定）。数据 Zenodo `10.5281/zenodo.20120070`；
代码 `signal-curator`，DOI `10.5281/zenodo.20119847`，MIT。

这份直接对应我们手上的 `morpho_50khz_full.h5`（FOD3–FOD7、12 504 条、含 1 011 条人工 GOOD/BAD 标签）。
**它是唯一带"可核对的标注"的论文**，因此也是最容易严格复现的一篇。

### P4 相关但非同一数据集

| 条目 | 说明 |
| --- | --- |
| *Printed PZT Transducers Network for the SHM of Foreign Object Damage Composite…*（Zenodo 15050547 论文 / 11501569 报告） | 同组的印刷 PZT 会议论文 |
| `GNN-Spatial-Strain-SHM`（Zenodo 22207949，GitHub johnstamly） | 复材板应变 SHM 的图神经网络，非 FOD 板 |
| MORPHO 项目交付物（DMP 等） | `morpho-h2020.eu/dissemination/deliverables/`；CORDIS `101006854` |

> ⚠️ `ReadMe.pdf` 参考文献里的 `zenodo.org/records/14193336` 不是本数据集，那是对照组的
> COQTEL 腐蚀 Lamb 波数据集（`10.5281/zenodo.14193336`）。引用应改用 `14627730`。

## 2. P1 做了什么（逐节）

| 节 | 内容 | 关键结论 |
| --- | --- | --- |
| §2.2 实验 | MTS 810 + 四点弯曲，11 级载荷（4→28 kN），每级若干块；**导波采集在 0.5 kN 极小载荷下进行**，5 个频率 × 4 个陶瓷作动器 × 10 次重复；FBG 在每个块内**连续**记录 | 这是最关键的一句：导波测量条件在段与段之间是恒定的 |
| §3.2 刚度 | 每个循环**加载段初始线性部分**（屈曲前）拟合 load–displacement 斜率 | 早期基本恒定、8 kN 后略升（曲面压平）；20 kN 起快速下降；拐点约 **2000–2500 圈** |
| §3.3 FBG | 原始应变随载荷变化，不能直接当退化指标；先 200 点窗均值降采样 + 10 点滑动平均，再取 s_i = ∂ε_i/∂t，**HI = Σ\|s_i − s_j\|**（成对差分累积） | 近损伤的配对"缓慢上升"；远离的只在末段突升 |
| §3.4 导波 | Symlet-8 小波降噪；比较健康 / 16 kN 后 / 失效三状态 | 50 kHz：首波包基本不变，**后续波包**先降后与前波混叠；150 kHz：响应幅值随退化明显下降 |

## 3. 我们的复现结果（`reproduce.py`）

产物在 `processed/12_reproduce/`，SUMMARY 见该目录下 `00_reproduce.md`。

| 项 | 复现结论 |
| --- | --- |
| **A 刚度** | ✅ **完全复现**。4 kN 638 → 8 kN 663（升）→ 12–16 kN 平台 684 → 18 kN 675 → 20 kN 664→633 → 22 kN 595→523 → 24 kN 461 N/mm。早期上升、20 kN 后陡降、拐点落在 2400–2500 圈，与论文逐条对上 |
| **B FBG HI** | ✅ **趋势复现**。120 个配对的 HI **速率** 随累计循环上升：Spearman 中位 **+0.83**（73/120 对 ρ>0.8），首末比中位 **51 倍**。加载速率同期只涨 7 倍，所以增量大于加载速率可解释的部分 —— 方向与论文一致。⚠️ 但两者与载荷级完全同步、残差不可分，**"这是损伤"仍未独立证明** |
| **C 导波三状态** | ⚠️ **50 kHz 对上，150 kHz 编号不符**。50 kHz/PZT_3：总峰值 0.234→0.031→0.024 V，后续/首波比 11.5→0.97→0.41，与论文描述一致。150 kHz 论文说"传感器 2 幅值下降"，但在 FOD7 上 **PZT_2 是上升**（0.317→0.616→0.776 V），**PZT_3 才是下降**（0.200→0.102→0.057 V） |
| **D 导波 vs 刚度** | 论文未做。`xcorr_基线` 与刚度 **几乎无关**（ρ=+0.03）；只有包络重心有中等相关（ρ=−0.57）。前者在第 200 圈就掉到 0.57，而刚度到 2400 圈才明显下降，**不是同一根轴** |

## 4. 三点必须写进结论的方法学提醒

1. **"载荷级"不是测量条件。** 我一度以为导波随载荷级变化是测量假象，但 P1 §2.2 明确写了
   导波采集在 0.5 kN 下进行、条件恒定。因此导波指标的台阶式变化更可能是**损伤在载荷级提高时
   成批累积**（块加载的典型行为）。仍排除不掉的是夹具重装/粘接老化这类装置漂移。
2. **FBG 的 HI 天然混入加载速率。** 论文的加载速率随载荷级从 15 提到 105.5 mm/min（7 倍），
   而 s_i = ∂ε/∂t 正比于加载速率。实测 HI 速率涨了 51 倍，**大于**加载速率能解释的部分，
   所以方向上是支持的；但两者与载荷级完全同步、残差不可分。要定量必须按加载速率归一。

3. **PZT 编号在论文与 HDF5 之间可能差一位。** 引用具体通道号时必须写明是哪个编号体系。
   在 FOD7 上，论文所说的"150 kHz 传感器 2 下降"实际对应 PZT_3。

## 5. 还能做什么

| 优先级 | 事项 | 依据 |
| --- | --- | --- |
| 高 | 复现 P3 的主动学习收敛曲线（用 `paper_labels.json` 的 1 011 条标签，核对其 ~96% 准确率） | 有标注、有官方代码，最容易严格核对 |
| 中 | 用 P1 的刚度曲线作为**唯一退化参照轴**，重新评估所有 SHM 指标（含 50 kHz 导波） | 现在有了可信的参照，之前的"寿命分数"可以用刚度替代 |
| 中 | 按加载速率归一后再算 FBG HI，看能否救活这个指标 | 上面第 2 点 |
| 低 | 拿 FOD3–FOD6 的 50 kHz 数据做跨试件一致性（已有 `analyze.py`） | 已做，结论见 `11_wave50/00_analysis.md` |
