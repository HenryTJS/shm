# 多源结构健康监测

用四路传感——**声发射 AE**、**光纤光栅 FBG**、**分布式应变 DFOS**、**导波 PZT**——在三个数据平台上回答两个问题：

1. **一个试件的寿命进度，能不能只用在线可得的信号估出来？**
2. **跨试件、跨机构地，损伤指标能不能归一化到同一根轴上？**

三个平台的答案不同，本项目只交付在数据上站得住的那部分。

> **本文只做导览与跳转。** 每个平台的数字、口径与边界都写在它自己的文件夹里（见下表）；
> 想知道"结果是多少"，点进对应文件夹的 README。

## 1. 文件夹导览

| 文件夹 | 是什么 | 交付什么 | 一键入口 |
| --- | --- | --- | --- |
| [`main/`](main/README.md) | 内部电液伺服疲劳机主样本，27 组（看板 24 组），AE + 应变 + 5 路 FBG | 在线寿命进度基线（累积能量口径） | `cd main` → `python run.py --list` |
| [`l1/`](l1/README.md) | ReMAP / TU-Delft + UPatras 公开集，29 组，分 C1–C4 四批 | 离线损伤复评、跨机构归一化、论文复现 | `cd l1` → `python run.py --list` |
| [`phmdc/`](phmdc/README.md) | PHM2019 铝搭接件（外部平台），8 件，20 MHz Lamb 波 | 有**定量真值**的方法标定与统计验证 | `cd phmdc` → `python run.py --list` |
| [`l2/`](l2/README.md) | Delft Level 2 多筋板（外部结构验证，进行中） | AE t50/t85 与 PZT 对照 | 见 l2 README |
| [`llm/`](llm/README.md) | 证据约束的 SHM 助手（skill + 工具 + RAG + claim–evidence 校验） | 可追溯、可审计的结论问答 | 见 llm README |
| [`dashboard/`](dashboard/README.md) | 纯前端数字大屏（`index.html`，双击即开） | 24 组在线回放 + 29 组离线复评 | 双击 `dashboard/index.html` |
| `docs/` | 论文、申报书、文献与投稿材料 | — | 本地保留，不入库 |
| `figures_ppt/` | 汇报与论文用图 | — | 本地保留，不入库 |

数据集之间的关系：`main` 提供在线场景，`l1` 提供跨机构/跨模态的公开数据，
`phmdc` 是唯一**有定量真值**的平台，用来回答「这个指标到底对不对」。
`l2` 是 L1 结论在更高结构层级上的一次外部检验。

## 2. 仓库根的公共文件

| 文件 | 作用 |
| --- | --- |
| [`paths.json`](paths.json) | **数据集路径的唯一入口**（`data_root` 指向外接盘）。换盘只改这里 |
| [`paths.py`](paths.py) | 读 `paths.json`：`python paths.py` 逐条列出仓库侧/数据侧状态，`--json` 机器可读 |
| [`datasets.py`](datasets.py) | 三套数据的组、批次、`n_f`、时钟平移的**单一定义源**；`python datasets.py --cross-nf` 自检 |
| [`mono_ablation.py`](mono_ablation.py) | 跨 `main` / `l1` 的一次性消融对照（物理约束层）→ `results/mono_ablation_*.csv` |
| [`requirements.txt`](requirements.txt) | 依赖（`torch` 只有 `phmdc` 的波形级模型需要） |

`paths.py` 与 `datasets.py` 是跨数据集的基础设施，所以留在根目录；**其余代码都在各自的文件夹里**。

## 3. 怎么跑

### 3.1 装依赖

Windows + Anaconda，Python 3.12：

```bat
conda create -n shm python=3.12
conda activate shm
pip install -r requirements.txt
```

### 3.2 数据在外接盘

原始数据不放进仓库，唯一配置入口是 `paths.json`（`data_root` 指向外接盘）。仓库里的
`l1/L1-03`、`main/001`、`phmdc/T1` 这类目录是**目录联接（junction）**，只是访问入口，不占磁盘。

```bat
python paths.py          rem 人读：每条映射 → 外接盘目标，全 OK 说明数据在位
python paths.py --json   rem 机器可读
```

新增数据集：数据放外接盘，在 `paths.json` 的 `items` 里加一条即可。

### 3.3 五分钟自检

```bat
rem 仓库根
python datasets.py --cross-nf                   rem 数据定义自洽（应打印「不一致 0 条」）

rem 进到任一数据集文件夹，它有自己的任务矩阵
cd l1
python run.py --list                            rem 任务矩阵
python run.py --task hi-ae --dry-run            rem 只打印要执行的命令
```

都 `exit 0` 就说明环境 + 数据 + 入口都通了。

### 3.4 每个数据集一个一键脚本

没有全局入口：**进哪个文件夹，就用哪个文件夹的 `run.py`**。三份脚本界面一致：

| 命令 | 作用 |
| --- | --- |
| `python run.py --list` | 打印本文件夹的任务矩阵 |
| `python run.py --task <任务> [<任务> …]` | 按顺序跑；`--task all` 跑全部 |
| `python run.py --task <任务> --dry-run` | 只打印将执行的命令 |
| `python run.py --paths` | 打印数据路径映射并自检 |
| `python run.py --task x -- <参数…>` | `--` 之后的参数透传给底层脚本 |

子进程的工作目录 = 脚本所在目录，所以相对路径参数以**脚本目录**为准。

```bat
cd main
python run.py --task cdio                      rem 在线进度预测（示例流）
python run.py --task dashboard                 rem 看板数据包（24 组）

cd ..\l1
python run.py --task paper                     rem 论文 Level 1 / 4 复现

cd ..\phmdc
python run.py --task step3 step4               rem 外推 / 置换检验
```

各文件夹的任务清单、结果与边界见 [`main/README.md`](main/README.md)、[`l1/README.md`](l1/README.md)、
[`phmdc/README.md`](phmdc/README.md)。

### 3.5 看板

浏览器打开 `dashboard/index.html`。顶栏选数据集（`疲劳机主样本` + `ReMAP C1–C4` 共 5 个）再选试件。
数据包不在时先导出：

```bat
cd main
python run.py --task dashboard                 rem 24 组主样本

cd ..\l1
python run.py --task dashboard                 rem 29 组 L1（C1–C4）+ 四个清单
```

字段与读数口径见 [`dashboard/README.md`](dashboard/README.md)。

## 4. 产物位置

| 产物 | 位置 |
| --- | --- |
| 主样本结果 | `main/cdio_baseline/results/` |
| L1 结果表 / 缓存 / 图 | `l1/results/`、`l1/figures/` |
| phmdc 结果 | `phmdc/results/` |
| 看板数据包 | `dashboard/data/`（不入库） |

`results/`、`figures/`、`dashboard/data/` 都在 `.gitignore` 里，可随时删掉重跑。

## 5. 常见坑

1. **`-h` 不是所有脚本都有**：`phmdc/step0.py`、`l1/ae_frames.py`、`l1/ae_cycle.py` 没有 argparse，传 `-h` 会直接开跑（后者还会跑全部默认组）。
2. **部分脚本导入时会 `chdir` 到自己所在目录**：相对路径参数以**脚本目录**为准，不是你的当前目录。
3. **PowerShell 会吃掉组号前导零**：`--groups "L1-03,L1-04"` 一定要加引号。
4. **别用 `| Out-Null` / `Select-Object -First N` 截断长命令**：管道提前关闭会让命令报失败退出码，真错误反而看不到。
5. **拷到别处运行的脚本要设 `PYTHONPATH`**：例如 `set PYTHONPATH=D:\shm;D:\shm\l1`（仓库根 + 脚本目录）。
6. **图里的中文**：代码已设 `Microsoft YaHei / SimHei`。

## 6. 文档索引

| 想知道什么 | 看哪里 |
| --- | --- |
| 每个文件干什么 | [`仓库地图.md`](仓库地图.md) |
| main / l1 / l2 / phmdc / llm / dashboard 的方法、结果与能力边界 | 各自文件夹的 `README.md` |
| 实现细节、作废清单与全部实测记录 | `docs/details.md`（本地维护，不入库） |
