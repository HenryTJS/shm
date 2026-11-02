# SHM Skill Toolkit：证据约束的结构健康监测助手

这个目录不是“通用大模型”，也不是“聊天框”。

它的定位是：

> **一组面向结构健康监测（SHM）的任务型 skill：每个 skill 只做一件明确的事，调用确定性工具算数，检索文献/规范作依据，输出可核验的 claim–evidence 结构；LLM 只负责理解问题、选择 skill、解释结果，不负责算数和编结论。**

换句话说：

```text
人类/Agent 下命令
  -> skill 路由
  -> 工具计算 / RAG 检索 / Adapter 取数
  -> claim–evidence 图
  -> validator + critic
  -> 通过则输出报告；不通过则拒答
```

---

## 0. 当前方案（2026-11）

### 0.1 目标

不是“一个 metric 吃所有数据集”，也不是“一个模型做所有任务”。

目标是：

> **有限原语 + LLM 组合 + 确定性执行 + 验证器**：  
> 给一个没见过的任务、数据字典和原语文档，LLM 生成 workflow；executor 执行；verifier 决定能不能交。

任务分三类：

| 类型 | 例子 | 适配方式 |
|---|---|---|
| 同任务族、新数据 | L2 / C1-C2 的 t85 | reader + profile，core 不改 |
| 新任务族、同原语 | MORPHO 载荷混杂、PHM 裂纹回归 | LLM 组合原语，新增 workflow |
| 新原语/新物理量 | 图像、振动、视频 | 先补原语，不硬套 |

**没有数据字典或没有 cycle 轴时，不猜，直接 `unsupported`。**

### 0.2 总体架构

```text
数据字典 + 原语文档 + 任务
        |
        v
  Planner（LLM）
        |
        v
  workflow JSON（DAG）
        |
        v
  Executor（确定性 Python）
        |
        v
  Verifier（schema / 数值 / 证据 / 拒答）
        |
        v
  artifact.md + claims.json + evidence.json
```

### 0.3 原语库（计划）

第一版只做表格/时序数据：

```text
load_table, select_columns, filter_rows, groupby_agg,
join_tables, sort, window, cumsum, normalize,
interpolate_at, corr, partial_corr, regress,
compare_groups, refuse
```

每个原语必须有：

- 输入/输出 schema；
- 参数和单位；
- 失败条件；
- evidence 输出；
- golden case。

### 0.4 API 教师 + 本地学生

```text
API planner
  -> 生成 workflow 候选
  -> executor 执行
  -> verifier 过滤
  -> 形成训练集
  -> 训练本地 3B QLoRA planner
  -> 本地推理
```

- API 只接收 **schema / 任务 / 原语文档**，不接收原始信号；
- API 是教师和上限 baseline，不是最终唯一大脑；
- 本地 3B 是最终可离线、可训练、可 ablation 的 planner；
- 8 GB 显存优先 3B；7B 4-bit 很紧，不保证；
- 不训练原始波形编码器，先训练 **workflow planner**。

### 0.5 留出任务

不拿已知任务自欺。第一轮 held-out：

| 任务 | 给什么 | 不给什么 |
|---|---|---|
| L2 test-stop t85 | L2 原始 AE/cycle 字典 | L2 专用 skill |
| MORPHO 载荷混杂 | active_features / mts 字典 | load_confound skill |
| PHM2019 裂纹回归 | 特征表 + crack_mm | PHM 专用 skill |

成功标准：

- workflow 合法；
- 能执行；
- 数值在容差内；
- 证据完整；
- 该拒答就拒答；
- **0 次“算错但不说”**。

### 0.6 当前状态

| 模块 | 状态 |
|---|---|
| `adapters/raw.py` | 已完成：只读原始目录，不读 results |
| `skills/check` | 已完成：原始模态覆盖，不报 t85 |
| `skills/metric` | 部分完成：L1 DTA（C3/C4）可算；C1/C2、L2 未接 |
| `skills/compare` | 暂停：没有 raw campaign metric，返回 unsupported |
| `skills/audit` | 已完成：claim–evidence 校验 |
| workflow schema / executor / verifier | **未开始** |
| API planner | **未开始** |
| 本地 3B QLoRA | **未开始** |
| case 级评测 | **未开始**；旧 40/11/16 问答评测已删除 |

当前能诚实的说法：

> **这是一个 raw-only 的 SHM 工具层，加一个 L1-DTA 的 metric 原型；还不是 workflow 合成系统。**

### 0.7 实施顺序

1. 原语 + executor + verifier（不碰模型）；
2. 3 个 held-out workflow，人工跑通；
3. API 教师生成候选 workflow，verifier 过滤；
4. 训练 3B QLoRA planner；
5. 比较规则 / API / 本地学生；
6. 通过后再接 L2、C1/C2、MORPHO、PHM。

---

## 1. 为什么要做成 skill，而不是一个“万能问答”

上一版的问题是：

- 问题太简单：单跳查表；
- 评测单位是“一句话/一个数”；
- 和 RAG-only 比，证明不了真实价值；
- LLM 容易被当成知识库。

skill 化的目标是：

1. **每个 skill 有明确输入/输出**，可以单独测试；
2. **数字来自工具**，不是 LLM 记忆；
3. **每个结论有 evidence**，可以审计；
4. **可以组合成工作流**，而不是堆 prompt；
5. **同任务族换数据只加 reader + profile；新任务族必须新原语或新 workflow**，不能靠加 Adapter 冒充通用。

---

## 2. 目标 skill 清单

当前可用 3 个（`check` / `metric` / `audit`），`compare` 暂停。

### 当前

| skill | 命令示例 | 输出 |
|---|---|---|
| `check` | `python llm\skills\check\run.py --dataset L1 --gid L1-03` | 原始模态覆盖、缺失；t85 交给 metric |
| `metric` | `python llm\skills\metric\run.py --dataset L1 --gid L1-06` | 从 raw DTA/FBG 算 t50/t85；v0 只支持 C3/C4 |
| `audit` | `python llm\skills\audit\run.py --graph llm\examples\L1-03_t85.json` | 每个 claim 的出处、无证据/不一致项 |
| `compare` | 暂停到 raw metric 扩展到 campaign | 当前返回 `unsupported` |

### 后续（workflow 层）

| 能力 | 用途 |
|---|---|
| `diagnose` | 项目/数据/证据缺口诊断 |
| `recommend` | 下一步读什么、补什么、跑什么 |
| `probe` | 生成验证 case、边界测试、反驳问题 |
---

## 3. 当前已实现的代码

当前还不是完整的 `shm` CLI，而是可运行的原型：

```text
llm/
├─ adapters/
│  ├─ base.py        ← DatasetAdapter 接口
│  └─ raw.py         ← 只读原始数据的 RawL1/L2/PHM2019 adapter
├─ rag.py            ← 文献 RAG（TF-IDF，本地 PDF）
├─ rag_index.pkl     ← 已构建索引（可用 rag.py 重建）
├─ validate.py       ← claim–evidence 结构/语义校验
├─ schema/
│  └─ claim_evidence.schema.json
├─ examples/
│  ├─ L1-03_t85.json
│  ├─ invalid_missing_evidence.json
│  └─ invalid_temporal_leakage.json
├─ tests/
│  └─ test_validate.py
└─ skills/
   ├─ README.md
   ├─ common.py
   ├─ registry.json
   ├─ check/
   ├─ metric/
   ├─ compare/
   ├─ audit/
   └─ tests/
```

---

## 4. 证据契约

核心不是 prompt，而是 `schema/claim_evidence.schema.json`：

- **Claim**：结论，带 `predicate/value/unit/scope/allowed_lifetime_max`；
- **Evidence**：`tool_result / literature / standard / dataset_metadata`；
- **Edge**：`supports / contradicts / context`；
- **ToolCall**：工具名、参数、结果；
- **Verification**：validator 自动生成。

硬规则：

1. `supported` claim 必须有 evidence；
2. numeric/comparative claim 必须有匹配的 `tool_result`，数值在容差内；
3. literature/standard 必须有 `quote + locator`；
4. 非数值 claim 必须有 `support_quote`，且能在引用原文中找到；
5. evidence 的时间范围不能超过 claim 的 `allowed_lifetime_max`；
6. 被 Critic 判为 rejected 的数值 claim 不允许被“旁支剔除”放过；
7. 全部失败则拒答，不输出未验证数字。

示例：

```powershell
cd D:\shm
python llm\validate.py llm\examples\L1-03_t85.json
```

---

## 5. 时间门：这个项目最值得保留的技术点

SHM/PHM 最危险的不是“答错”，而是“用未来数据回答早期问题”。

当前实现：

- 每个 claim 可带 `allowed_lifetime_max`；
- 每条 tool_result evidence 带 `observed_lifetime_fraction`；
- validator 检查 evidence 是否越过 claim 的时间上限；
- 如果 `allowed_lifetime_max < 1` 且没有 supported 的 numeric claim，直接 fail。

这条规则比“RAG + 提示词”具体，也是专利交底书最值得写的部分。

---

## 6. 评测现状

旧评测（`llm/eval/`：40 条 L1/L2、11 条 PHM、16 条 held-out、所有 baseline 与结果）已全部删除，不再保留。

删除原因：

- 这些题和答案来自结果表，不能证明 `metric` 从原始曲线计算；
- 部分题过易，存在 ceiling effect；
- 继续保留会诱导 skill 读结果表。

当前只有：

- `skills/tests/` 的 10 条契约测试：输入输出、缺模态、结果目录禁令、computation 字段、metric raw；
- 核心测试 `llm/tests/` 5 条（validator）。

新的评测分两层：

1. **契约测试**：现在已有 10 条；
2. **case 级评测**：等 workflow/executor/verifier 建好后，用 L2、MORPHO、PHM 三个 held-out case，不用问答集。

---

## 7. 专利思路

“skill 命令集合”本身专利机会低；有机会的是背后的**具体技术方法**。

优先写：

1. **一种 SHM 结论的证据约束生成方法**  
   采集多模态信号 → 窗口特征 → 工具计算 → claim → claim–evidence 图 → 数值/scope/引用校验 → 拒绝/重算 → 带证据输出。

2. **一种 SHM 信息时限控制方法**  
   数据点记录寿命/循环 → 查询设定 allowed horizon → 工具调用裁剪未来数据 → 证据时间校验 → 违规拒绝 + 审计日志。

3. **一种跨机构可比性审计方法**  
   读取新 campaign 协议/元数据 → 映射统一 schema → 计算可比性维度 → 生成不可比证据 → 推荐补充对照。

注意：

- 必须绑定传感器数据、处理步骤、数据结构、技术效果；
- 不能只写“用 LLM 生成答案”；
- 先检索现有技术，再找学校知识产权办/代理所；
- **先申请，再公开论文/GitHub/答辩视频**；
- 软件著作权可同时申请，成本低。

---

## 8. 数据边界

当前主数据：

- L1：ReMAP/Delft+Patras 复合材料；
- L2：Delft Level-2 多筋板；
- PHM2019：铝搭接件，外部验证。

已知边界：

- 四模态齐全的试件很少（L1 中主要是 5 个）；
- L2 同实验室，不是外部机构；
- PHM2019 物理不同，不能改 L1 结论；
- 公开的 OGW/外翼/MORPHO 等可做外部可移植性，但不是同群体新 campaign；
- 没有新 campaign 时，不能声称“跨机构泛化”。

---

## 9. 快速开始

```powershell
cd D:\shm

# 1. 校验证据图
python llm\validate.py llm\examples\L1-03_t85.json

# 2. 单元测试
python -m unittest llm.tests.test_validate -v

# 3. 重建 RAG 索引（可选）
python llm\rag.py build
python llm\rag.py query "t85 definition" -k 5

# 4. skill 契约
python llm\skills\check\run.py --dataset L1 --gid L1-03
python llm\skills\metric\run.py --dataset L1 --gid L1-06
python llm\skills\audit\run.py --graph llm\examples\L1-03_t85.json
python -m unittest discover -s llm\skills\tests -v

---

## 10. 清理说明

已删除：

- `llm/eval/`：旧 40 条、11 条、16 条任务及全部 baseline/结果；
- `llm/archive/`：旧 adapter v0 与草稿题；
- `out/`、`__pycache__/`：生成文件。

保留：

- `skills/`：check / metric / audit 契约与测试；compare 暂停；
- `adapters/raw.py`：只读原始数据的 adapter；
- `schema/`、`examples/`、`validate.py`、`rag.py`；
- `rag_index.pkl`（可重建）。

---

## 11. 下一步

1. `check` / `metric` / `audit` 已落地；`metric` 先补 C1/C2 `.pridb` 和 L2 test-stop；
2. 建 `workflow/`：原语、executor、verifier，不碰模型；
3. 用 API 教师生成候选 workflow，verifier 过滤；
4. 训练 3B QLoRA planner，做规则 / API / 本地 ablation；
5. 用 L2、MORPHO、PHM 三个 held-out case 评测；
6. 恢复 `compare` 只在 raw campaign metric 可用后；
7. 专利交底书先写“时间门 + 证据校验”。

一句话：

> **这不是“大模型回答 SHM 问题”，而是“用 LLM 组合原语、用工具算数、用 verifier 决定能不能交”。**
