# 多源结构健康监测

用四路传感——**声发射 AE**、**光纤光栅 FBG**、**分布式应变 DFOS**、**导波 PZT**——在三个数据平台上回答两个问题：

1. **一个试件的寿命进度，能不能只用在线可得的信号估出来？**
2. **跨试件、跨机构地，损伤指标能不能归一化到同一根轴上？**

三个平台的答案不同，本项目只交付在数据上站得住的那部分。

## 1. 数据平台

| 平台 | 是什么 | 规模 | 提供了什么 |
| --- | --- | --- | --- |
| **main** | 内部电液伺服疲劳机，5 Hz 压-压疲劳 | 27 组（看板 24 组） | AE 25 列逐 hit 特征 + 应变 + 5 路 FBG，10 Hz 统一时间轴 |
| **l1** | ReMAP / TU-Delft + UPatras 公开集（冲击后压-压疲劳） | 29 组，按批次分 **C1–C4** | C1 恒幅+FBG+DFOS（5，AE+FBG+DFOS+PZT）、C2 恒幅+DFOS（9，AE+DFOS）、C3 变幅VA+FBG（5，AE+FBG+PZT）、C4 谱载+FBG（10，AE+FBG+PZT）；导波 PZT 另有 20 组有数据 |
| **phmdc** | PHM2019 铝搭接件（外部平台） | 8 件 | 20 MHz 原始波形 + 光学实测**裂纹长度真值** |

三者的关系：`main` 提供在线场景，`l1` 提供跨机构/跨模态的公开数据，`phmdc` 是唯一**有定量真值**的平台，用来回答「这个指标到底对不对」。

## 2. 方法与结果

### 2.1 main：在线寿命进度（累积能量口径）

**方法**：AE 累积对数谱能量作为时钟（CElog）+ 组成指纹 KNN + 库容量贝叶斯，只用截至当前窗口的信息，逐窗输出寿命百分比与 90% 区间。

| 指标 | 结果（24 组留一） |
| --- | --- |
| 寿命百分比 MAE | 均值 **9.6** / 中位 **8.8** 个百分点 |
| 末期点估计均值 | **78.9**（真值 100）⇒ 末期偏低，应以区间读，不以点读 |

**必须知道的口径**：这个量估计的是**试验记录进度**，不是损伤。AE 累积能量与寿命近似线性（寿命最后 10% 只贡献 10% 能量），因此它能排时间、不能分辨损伤程度；配套证据层（AE 能量 / 应变增量归一化曲线）也只作辅证。

### 2.2 l1：离线损伤复评与跨机构归一化

**主指标**：`HI = unity01(累积 AE hits)`（C1/C3/C4 记作 `HI_hit`，C2 记作 `HI_AE`），自归一化曲线，`t85` = 达终值 85% 时的寿命分数。

> 表中四批统一用 `cum_hits` 口径（循环轴、29 组全量、不剔除）；C2 的交付口径仍是 `HI_AE`。

| 批次 | 试验类型 | 组数 | 传感 | t85 中位 |
| --- | --- | --- | --- | --- |
| C1 恒幅+FBG+DFOS | 恒幅 | 5 | AE + FBG + DFOS + PZT | **89.2%** |
| C2 恒幅+DFOS | 恒幅 | 9 | AE + DFOS | **92.1%** |
| C3 变幅VA+FBG | 变幅 VA | 5 | AE + FBG + PZT | **94.0%** |
| C4 谱载+FBG | 谱载 | 10 | AE + FBG + PZT | **94.1%** |

**论文复现（Broer 2021）**：Level 1 / 3a / 4 均复现——`HI_F` 5 组全部 0 → 1；L1-04 的 `HI_F` t85 为 94.3%，与论文正文可推出的 > 92.8% 自洽；Level-1 检测点 240.0k / 260.5k 对论文 239.5k / 260.0k（各差 500 cycle）。Level 3b 脱粘未复现。

**形状类证据**（唯一对载荷级缩放不变、可跨批比较的量）：

- FBG 剖面 `shape_rob25`：暖机后 **11 正 / 0 负 / 2 弱**，|ρ| 中位 **0.70**；
- DFOS 块级 HI：C1 五组末值全为 1.000、尾段单调 100%。

**跨机构可比性**（两所大学、两套采集系统）：

- 横轴 = 状态序号 × 文档记载的采集间隔 ÷ `n_f`，与试验机循环记录秩相关 **13/14 组 ≥ 0.993**；
- 纵轴用**无标度**口径 SNHI：绝对阈值 **6/15 漏报**、组内极差中位 60.7 pp；SNHI **0/15 漏报**、18.5 pp；
- 置换检验下的方差占比 **η²**：机构对 AE t50 ≈ **0**（p=0.027）、实验室对 PZT 斜率 ≈ **0**（p=0.004），AE t85 残留 0.057（p=0.78，不显著）——归一化后无系统性机构偏差。*原「批间/组内比值 AE 0.33、PZT 0.29」经置换检验不具判别力（零分布中位 0.44，p=0.18/0.21），已撤下，见 `l1/crosslab_variance.py`。*

**载荷可迁移性**：变幅/谱载下**不存在可迁移的归一化**——敏感性本身跨试件不可迁移（拟合指数 `m_eff` 从 −0.20 到 +9.05）。因此变幅下交付的是形状量，不是「把幅值归一化掉」。

**定位**：X 向可用（跨组 σ ≈ 14 mm），Y 向不可用（σ ≈ 46 mm，加筋条把波导成非直线）。残余误差主因是**探头通道延迟**而非声速或几何：把 δ 显式建进走时模型后，9 组走时残差收敛到 0.45–2.37 µs。

### 2.3 phmdc：指标对不对

| 方法 | RMSE | 说明 |
| --- | --- | --- |
| 32 特征 + 单调约束 | **1.501 mm** | 折内 ρ 0.928（量程 0–7.46 mm） |
| 波形级 CNN（双通道）+ 约束 | **1.692 mm** | 零裂纹误报 24% |

**统计验证**：试件内随机置换标签后重跑同一套 LOSO，真实指标的**全部低于置换分布最小值**（z = 4.21 / 4.78）⇒ 学到的是真信号；**第二通道消融**（shuffle / dup / zero / rand 四个无信息对照）把零裂纹误报推回 81–88%（真实 24%，零重叠）⇒ 双通道增益来自**信息**而非容量。

**口径**：phmdc 是铝 / 铆接 / Lamb 波，与复材加筋板物理不同，**不改变 l1 的任何结论**；它只用于标定与统计验证。

## 3. 数字大屏

纯前端（`dashboard/index.html`，双击即开），一套代码容纳两种口径：

| 数据集 | 内容 |
| --- | --- |
| 疲劳机主样本 003–014 / 016–027 | 24 组**在线**进度回放（留一预测） |
| ReMAP C1 / C2 / C3 / C4 | 5 / 9 / 5 / 10 组**离线复评**（界面标注「非在线」） |

- 在线语义：选一组数据后看不到任何未来信息，曲线从起点随回放生长；拖进度条 = 在该时刻重新观察。
- 读数口径：主样本「进度偏差」= 预测进度 − 真实寿命进度（百分点）；L1 显示 `1 − HI`。两者都**不是**剩余寿命裕度。
- 数据包由脚本生成、**不入库**（见 §4.5）。

## 4. 怎么跑

### 4.1 装依赖

Windows + Anaconda，Python 3.12：

```bat
conda create -n shm python=3.12
conda activate shm
pip install -r requirements.txt
```

`torch` 只有 `phmdc` 的波形级模型需要；不跑它就把 `requirements.txt` 里那行注释掉。

### 4.2 数据在外接盘

原始数据不放进仓库，唯一配置入口是 `paths.json`（`data_root` 指向外接盘）。仓库里的
`l1/L1-03` 这类目录是**目录联接（junction）**，只是访问入口，不占磁盘。

```bat
python shm/paths.py          rem 人读：每条映射 → 外接盘目标，全 OK 说明数据在位
python shm/paths.py --json   rem 机器可读
```

新增数据集：数据放外接盘，在 `paths.json` 的 `items` 里加一条即可。

### 4.3 五分钟自检

```bat
python -m shm.datasets --cross-nf                     rem 数据定义自洽（应打印「不一致 0 条」）
python run.py --list                                  rem 任务矩阵
python run.py --dataset main --task cdio --dry-run    rem 只打印要执行的命令
```

三条都 `exit 0` 就说明环境 + 数据 + 入口都通了。

### 4.4 任务

**所有入口只有 `run.py`**；每个任务都能加 `--dry-run` 先看命令，`--` 之后的参数透传给底层脚本。

```bat
set PY=python            rem 换成你自己的解释器（或直接写绝对路径）

%PY% run.py --list                                rem 任务矩阵（按数据集分组）

rem main：只有在线进度预测
%PY% run.py --dataset main --task cdio            rem 预测示例流
%PY% run.py --dataset main --task dashboard       rem 看板数据（24 组）

rem l1：常用
%PY% run.py --dataset l1 --task hi-ae             rem 离线复评 HI_AE（C2 主指标）
%PY% run.py --dataset l1 --task hi-hit            rem 离线复评 HI_hit（C3/C4 主指标）
%PY% run.py --dataset l1 --task paper             rem 论文 Level 1 / 4 复现
%PY% run.py --dataset l1 --task paper-l23         rem 论文 Level 2 / 3 复现
%PY% run.py --dataset l1 --task crosslab-ae       rem AE 跨批次一致性（28 组）
%PY% run.py --dataset l1 --task crosslab-ae23     rem 补第 29 组（重，流式读 4.1 GB）
%PY% run.py --dataset l1 --task crosslab-pzt      rem 导波 PZT 跨批次（17 组，重）
%PY% run.py --dataset l1 --task level-vs-shape    rem 绝对阈值 vs SNHI 对照（15 组）
%PY% run.py --dataset l1 --task ae-cache          rem 首次必跑：逐 hit 二进制缓存
%PY% run.py --dataset l1 --task loc-debias        rem 通道延迟 δ 标定（需 ae-cache）

rem phmdc
%PY% run.py --dataset phmdc --task step0          rem 数据规整
%PY% run.py --dataset phmdc --task step1          rem 特征 + LOSO
%PY% run.py --dataset phmdc --task step4 step5    rem 置换检验 / 不确定性
```

> `--task all` 会把**重的任务**（训练、全量重建缓存）一起跑，先 `--dry-run` 看清。
> L1 的定位相关脚本**先跑一次 `ae-cache`**：它把 9 组 `.pridb`（9.54 GB、1.42 亿命中）固化成二进制缓存，
> 之后读取从数百秒降到 0.02 秒；缓存可删可重建。

### 4.5 看板

浏览器打开 `dashboard/index.html`。顶栏选数据集（`疲劳机主样本` + `ReMAP C1–C4` 共 5 个）再选试件。
数据包不在时先导出：

```bat
%PY% run.py --dataset main --task dashboard     rem 24 组主样本
%PY% l1\restore_dashboard_packages.py           rem 29 组 L1（C1–C4）+ 四个清单
```

字段与读数口径见 [`dashboard/README.md`](dashboard/README.md)。

### 4.6 产物位置

| 产物 | 位置 |
| --- | --- |
| 主样本结果 | `main/cdio_baseline/results/cdio_summary.csv`（24 行） |
| L1 结果表 / 缓存 | `l1/results/*.csv`、`l1/results/*.npz`、`l1/*.csv` |
| L1 论文图 | `l1/figures/fig_*.png` / `.pdf`（300 dpi） |
| phmdc 结果 | `phmdc/results/*` |
| 看板数据包 | `dashboard/data/*.js`（53 个包 + 5 个清单，**不入库**） |

`results/`、`figures/`、`dashboard/data/` 都在 `.gitignore` 里，可随时删掉重跑。

### 4.7 常见坑

1. **`-h` 不是所有脚本都有**：`phmdc/step0.py`、`l1/ae_frames.py`、`l1/ae_cycle.py` 没有 argparse，传 `-h` 会直接开跑（后者还会跑全部默认组）。
2. **脚本导入时会 `chdir` 到自己所在目录**：相对路径参数（如 `--out`）是相对**脚本目录**，不是你的当前目录。
3. **PowerShell 会吃掉组号前导零**：`--groups "L1-03,L1-04"` 一定要加引号。
4. **别用 `| Out-Null` / `Select-Object -First N` 截断长命令**：管道提前关闭会让命令报失败退出码，真错误反而看不到。
5. **拷到别处运行的脚本要设 `PYTHONPATH`**：`set PYTHONPATH=%CD%\l1`。
6. **图里的中文**：代码已设 `Microsoft YaHei / SimHei`。

## 5. 读这个项目需要知道的边界

- **主样本不声称损伤度、分级与在线预警**。它的交付是「记录进度」的基线，不是剩余寿命。
- **L1 的 `HI` 是离线复评**（自归一化用到全寿命终值，非因果），`t85` 是曲线形状指标，不是预测时刻。
- **阈值 0.25 / 0.55 / 0.85 只是分档刻度**，两级之间不承诺触发间隔，也不代表物理裕度。
- **不做的事**：主样本定位、`l1` 的 Y 向分区、把形状配准当 RUL 预测器、声速反演、导波在线因果化。理由与实测见 `docs/details.md` 附录 A 与 `当前结论与口径.md`。

## 6. 文档与目录

| 想知道什么 | 看哪里 |
| --- | --- |
| 每个文件干什么 | [`仓库地图.md`](仓库地图.md) |
| 结论与口径为什么是这样 | [`当前结论与口径.md`](当前结论与口径.md) |
| 还没闭合的问题 | [`待办与未决问题.md`](待办与未决问题.md) |
| 实现细节与全部实测记录 | `docs/details.md`（本地维护，不入库） |
| 看板字段与重建 | [`dashboard/README.md`](dashboard/README.md) |
| 主样本方法与限制 | [`main/README.md`](main/README.md) |
| phmdc 全部结果 | [`phmdc/README.md`](phmdc/README.md) |

```
main/      主样本：cdio_baseline（唯一在线方法）、attic/（归档）、results/（诊断记录）
l1/        公开集：读入（ae_io / ae_dta / ae_cycle）、指标（ae_hi / fbg_tools）、
           复现与跨机构（reproduce_broer_* / hi_crosslab / pzt_* / level_vs_shape）、看板重建
shm/       公共库：数据定义（datasets）、路径（paths）、编码（pkgcodec）等
phmdc/     PHM2019 外部验证平台（step0–step6）
dashboard/ 纯前端看板：index.html + assets/
```
