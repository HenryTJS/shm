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
5. **换数据集只加 Adapter**，不改主逻辑。

---

## 2. 目标 skill 清单

先做 3 个 MVP，再做后面 3 个。

### MVP

| skill | 命令示例 | 输出 |
|---|---|---|
| `check` | `shm check --dataset L1 --specimen L1-03` | 模态覆盖、缺失、寿命终点、数据质量告警 |
| `compare` | `shm compare --a C1 --b C3 --metric t85` | 差值、范围、可比性限制、证据 |
| `audit` | `shm audit --report draft.md --data L1` | 每个 claim 的出处、无证据/不一致项 |

### 后续

| skill | 用途 |
|---|---|
| `metric` | t50/t85/SNHI/阈值等单指标查询与计算 |
| `diagnose` | AE/FBG/DFOS/PZT 多模态不一致诊断 |
| `plan` | 下一步试验/检查/补充对照建议 |

---

## 3. 当前已实现的代码

当前还不是完整的 `shm` CLI，而是可运行的原型：

```text
llm/
├─ agent.py          ← DeepSeek planner + writer + critic + validator
├─ tools.py          ← L1/L2 指标、批次、跨模态工具
├─ adapters/
│  ├─ base.py        ← DatasetAdapter 接口
│  ├─ l1.py          ← L1/L2 适配器
│  ├─ phm2019.py     ← PHM2019 外部适配器
│  └─ registry.py    ← 数据集注册表
├─ rag.py            ← 文献 RAG（TF-IDF，本地 PDF）
├─ rag_index.pkl     ← 已构建索引（可用 rag.py 重建）
├─ validate.py       ← claim–evidence 结构/语义校验
├─ schema/
│  └─ claim_evidence.schema.json
├─ examples/
│  ├─ L1-03_t85.json
│  ├─ invalid_missing_evidence.json
│  └─ invalid_temporal_leakage.json
├─ eval/
│  ├─ tasks.jsonl              ← 40 条 L1/L2 开发集
│  ├─ tasks_phm.jsonl          ← 11 条 PHM 开发集
│  ├─ tasks_phm_heldout.jsonl  ← 16 条半独立 held-out
│  ├─ run_eval.py
│  ├─ run_baseline.py
│  ├─ results.csv
│  ├─ results_phm.csv
│  ├─ results_phm_heldout.csv
│  ├─ baselines.csv
│  ├─ baselines_phm.csv
│  ├─ HELDOUT_GUIDE.md
│  └─ freeze_run.py
├─ tests/
│  ├─ test_validate.py
│  └─ test_tools.py
└─ archive/          ← 不参与主链路的实验与草稿
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

## 6. 评测现状（必须带 caveat）

| 数据 | 结果 | 性质 |
|---|---:|---|
| L1/L2 40 条 | 40/40 | 内部开发集，偏简单 |
| PHM2019 11 条 | 11/11 | 内部开发集 |
| PHM2019 held-out 16 条 | 16/16 | 用户改写后的半独立集 |
| KB-only 40 条 | 5/40 | 弱基线 |
| RAG-only 40 条 | 6/40 | 弱基线 |
| KB-only PHM 11 条 | 3/11 | 弱基线 |
| RAG-only PHM 11 条 | 3/11 | 弱基线 |

这些数字只能说明：

- 工具链路能跑；
- 证据门能拦住缺证据/时间泄漏；
- Agent 在**当前任务集**上比弱 KB/RAG 基线好。

不能说明：

- 跨机构泛化；
- 通用 SHM 智能；
- 方法创新；
- 在未见 campaign 上有效。

PHM held-out 是用户改写后的 16 条，仍不是完全独立第三方评测。

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
python -m unittest llm.tests.test_validate llm.tests.test_tools -v

# 3. 重建 RAG 索引（可选）
python llm\rag.py build
python llm\rag.py query "t85 definition" -k 5

# 4. Agent（需要 DEEPSEEK_API_KEY）
$env:DEEPSEEK_API_KEY="sk-..."
python llm\agent.py "L1-03 的 t85 是多少？比 C1 中位低多少？"

# 5. 评测
python llm\eval\run_eval.py
python llm\eval\run_baseline.py
```

---

## 10. 清理说明

已归档/删除：

- `archive/adapter_v0/`：MLP/RF adapter 实验（MLP 负结果、RF 1.416 mm），不参与主链路；
- `archive/drafts/`：机械草稿题与生成脚本，不参与正式评测；
- `eval/out/`、`out/`、`__pycache__/`：生成文件，已删除。

保留：

- 主链路：`agent.py`、`tools.py`、`rag.py`、`validate.py`、`adapters/`；
- schema、examples、tests；
- 评测集与结果；
- `rag_index.pkl`（可重建）。

---

## 11. 下一步

1. 把 `check` / `compare` / `audit` 做成真正的 skill 目录（`SKILL.md + run.py + schema + tests`）；
2. 把 16 条 PHM held-out 扩成 case 级审计，不再做单跳问答；
3. 加人工 checklist 基线；
4. 专利交底书先写“时间门 + 证据校验”；
5. 不再扩 B（跨模态大模型），除非拿到更多四模态试件。

一句话：

> **这不是“大模型回答 SHM 问题”，而是“用 LLM 编排一组可审计的 SHM skill”。**
