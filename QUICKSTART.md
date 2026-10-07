# QUICKSTART —— 从零把它跑起来

---

## 0. 这是什么

四类数据集，同一套「**在线损伤度 D(t) + 三级预警**（0.25 注意 / 0.55 预警 / 0.85 临危）」方法：

| 数据集 | 是什么                                                  | 命令里叫  |
| ------ | ------------------------------------------------------- | --------- |
| 主样本 | 内部疲劳机，5 Hz 正弦载荷疲劳，016–020 + 023–026      | `main`  |
| L1     | ReMAP / TU-Delft 公开集（同一数据集分 4 组模态），27 组 | `l1`    |
| phmdc  | PHM 2019 铝搭接件，外部方法验证平台                     | `phmdc` |

**所有入口只有一个**：仓库根的 `run.py`。

---

## 1. 装依赖

开发环境是 **Windows + Anaconda + Python 3.12**（实测 3.12.13）。

```bat
conda create -n shm python=3.12
conda activate shm
pip install -r requirements.txt
```

`torch` 只有 `phmdc` 的 Step 2（波形级 DL）需要 —— 不跑它就把 `requirements.txt` 里那行注释掉。

---

## 2. 数据：**只在外接盘上**，仓库里不放数据

原始数据约 **239 GB**，**全部放在外接盘**（当前 `E:\`）；仓库（`D:\`）里**不保存任何数据**。

> 🔴 **规矩（2026-10-07 起）**：以后**新增数据一律直接放到外接盘**（如 `E:\l1\L1-xx`、`E:\main\0xx`），
> **不允许从 D 盘进入** —— 不要先把数据拷进仓库再搬迁，也不要在仓库里保留数据副本。
> D 盘只放代码与产物。

**唯一配置入口是 `paths.json`**（换盘 / 换机器只改这一个文件，**不要改代码**）：

```json
{ "data_root": "E:\\", ... }
```

**查看数据路径状态（只读，不改任何东西）：**

```bat
python shm/paths.py            rem 人读表格：每条映射 → 外接盘目标
python shm/paths.py --json     rem 机器可读
```

输出形如 `l1\L1-03   已联接    OK    E:\l1\L1-03`，全 `OK` 就说明数据都在位。

> 为什么仓库里还能用 `l1/L1-03/...` 这种相对路径：这些目录是**目录联接（junction）**，
> **不占 D 盘空间、数据本体在 E 盘**，它只是「访问入口」。新增数据集：数据放外接盘，
> 再在 `paths.json` 的 `items` 里加一条（规则见该文件内注释与 `shm/paths.py`）。

---

## 3. 五分钟自检

```bat
python -m shm.datasets --cross-nf                     rem 四类数据定义自洽（应打印「不一致 0 条」）
python run.py --list                                  rem 任务矩阵（按数据集分组）
python run.py --dataset main --task degree --dry-run   rem 只打印要执行的命令，不真跑
```

三条都 `exit 0` 就说明环境 + 数据 + 入口都通了。

---

## 4. 想干什么 → 跑什么

先看矩阵：`python run.py --list`（**每个任务都能加 `--dry-run` 只打印命令**）。

### 主样本 `main`（016–020 有标签 + 023–026 无标签）

```bat
python run.py --dataset main --task prepare       rem ① 多源对齐       → main/aligned/
python run.py --dataset main --task labels        rem   弱标签 b2/b3    → main/weak_labels/
python run.py --dataset main --task degree         rem ② D(t) 达阈/单调 → main/results/
python run.py --dataset main --task warning        rem   预警 onset 分级 → main/results/warning_onset.csv
python run.py --dataset main --task curves paper   rem   出图 → main/figures/
python run.py --dataset main --task robust         rem ③ 稳健性（灵敏度/LOSO/消融/统计）
python run.py --dataset main --task verify         rem 验证层 7 件事一把跑完（最耗时）
python run.py --dataset main --task dashboard      rem 看板数据 → dashboard/data/
```

也可直接调两个合并入口（`-h` 看子命令）：

```bat
python main/pipeline.py            rem prepare / evaluate / robust
python main/verify.py              rem baselines / loso-cv / labels / ae-cols / grade / fusion / candidates
```

### 公开集 `l1`（27 组，按模态分 4 组）

```bat
python run.py --dataset l1 --task prepare          rem 原始 .pridb → CSV（耗时长、占空间）
python run.py --dataset l1 --task prepare-c2       rem 另一批：LUNA 段级 + AE 1s 分箱
python run.py --dataset l1 --task paper            rem 论文 Level 1 / 4 复现
python run.py --dataset l1 --task paper-l23        rem 论文 Level 2 / 3 复现
python run.py --dataset l1 --task hi-ae            rem 离线复评 HI_AE（一批主指标）
python run.py --dataset l1 --task fiber-hi         rem FBG 块级 HI（离线）
python run.py --dataset l1 --task dfos             rem 分布式应变逐块（离线）
python run.py --dataset l1 --task meta             rem 试件元信息 + L1数据记录.xlsx
```

> ⚠️ **L1 只保留离线口径** —— 它不支持跨试件统一阈值的**在线预警**（四条独立路径一致失败），
> 因此在线的 D(t) 与 L1 看板包已于 2026-10-07 整体移除。依据见 `docs/details.md`〈l1 恒幅+DFOS 组 —— 不能做什么〉。

### 外部平台 `phmdc`

```bat
python run.py --dataset phmdc --task step0         rem 数据规整 → labels.csv / index.csv
python run.py --dataset phmdc --task step1         rem 特征提取 + LOSO 8 折
python run.py --dataset phmdc --task step2         rem 波形级 DL（需要 torch，重）
python run.py --dataset phmdc --task step3         rem 外推 / 预后评测
python run.py --dataset phmdc --task step4 step5   rem 置换检验 / 不确定性量化
```

> `--task all` 会把**重的任务**（训练、全量重建缓存）也一起跑 —— 先 `--dry-run` 看清楚。
> `--` 之后的参数会透传给底层脚本，例：
> `python run.py --dataset l1 --task paper -- --groups L1-03`

---

## 5. 产物在哪

| 产物              | 位置                                                     |
| ----------------- | -------------------------------------------------------- |
| 主样本结果表 / 图 | `main/results/*.csv`、`main/figures/*.png`           |
| L1 结果表 / 缓存  | `l1/results/*.csv`、`l1/results/*.npz`、`l1/*.csv` |
| phmdc 结果与报告  | `phmdc/results/*`                                      |
| 看板数据包        | `dashboard/data/*.js`（39 个）                         |
| 中间缓存          | `main/cache/`、`l1/results/_*_cache/`                |

绝大多数产物**可以删了重跑**（`results/`、`figures/`、`dashboard/data/` 都在 `.gitignore` 里）。

---

## 6. 看板

直接用浏览器打开仓库根的 **`dashboard/index.html`**。
顶栏可切换试件（当前只有**主样本 11 组**），逐帧回放 D(t)、多源信号与三级预警。
（L1 的看板包已于 2026-10-07 随在线口径移除，见 `docs/details.md` §38。）

数据包不在时要先导出：

```bat
python run.py --dataset main --task dashboard
python run.py --dataset l1   --task dashboard       rem 还有 dashboard-v2 / dashboard-v3
```

出包后建议校验一次：

```bat
python l1/check_dashboard_pkg.py --ds l1
```

细节见 [`dashboard/README.md`](dashboard/README.md)。

---

## 7. Agent

**默认走模板渲染 —— 零依赖、不联网、不需要 key**：

```bat
python agent/cli.py --demo                             rem 跑内置示例（约 0.5 s）
python agent/cli.py "016 什么时候预警的" --quiet        rem 直接提问
python agent/cli.py                                   rem 交互模式
```

要接真·LLM 才需要 key：复制 `.env.example` 为 `.env` 并填入（`.env` 已在 `.gitignore` 里）：

```bat
copy .env.example .env
python agent/cli.py --llm qwen2.5:3b                   rem 或 --llm http://...
```

细节见 [`agent/README.md`](agent/README.md)。

---

## 8. 常见坑

**① `-h` 不能用来试探所有脚本。** 有些脚本**没有 argparse**，传 `-h` 会**直接开跑**而不是打用法：

- `main/verify.py` 的 `baselines` / `loso-cv` / `labels` / `grade` / `fusion`
- `phmdc/step0.py`

⇒ 想确认某个脚本能不能跑，先看它 `import argparse` 没有。

**② 脚本会在导入时 `chdir` 到自己所在目录。** 所以相对路径参数（如 `--out`）是相对**脚本目录**
解释的，不是相对你敲命令的目录。要看"调用时目录"的脚本自己留了 `CWD0`。

**③ PowerShell 会把 `021,022` 当成数字、吃掉前导零。** 传组号一定要加引号：

```bat
python main/verify.py candidates --groups "021,022"
```

**④ 别用 `| Out-Null` 或 `Select-Object -First N` 截断长命令。** 管道提前关闭会让命令报
失败退出码（`exit=-1`），但你其实没看到真错误。

**⑤ 移动过的脚本要设 `PYTHONPATH`。** 例如 `l1/attic/` 下的归档脚本：

```bat
set PYTHONPATH=%CD%\l1
python l1\attic\p2p_load_check.py
```

**⑥ 图片里的中文。** 已在代码里设 `Microsoft YaHei / SimHei`；在 Linux 上跑要先装中文字体。

---

## 9. 接下来读什么

| 想知道                             | 看                                                      |
| ---------------------------------- | ------------------------------------------------------- |
| 每个文件干什么、哪些是入口/库/归档 | [`docs/仓库地图.md`](docs/仓库地图.md)                 |
| 项目做了什么、结论是什么           | [`README.md`](README.md)                               |
| 工程细节与全部实测记录（很长）     | [`docs/details.md`](docs/details.md)                   |
| 数据在哪、怎么查状态               | 本文 §2 + `paths.json` + `python shm/paths.py`         |
| 还没解决的问题                     | [`docs/待办与未决问题.md`](docs/待办与未决问题.md)     |
