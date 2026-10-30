# PHM2019 独立 held-out 任务编写指南

目标：让**没有参与 Agent 开发的人**，只依据数据字典和 `labels.csv`，独立写一批题；  
在**冻结代码和 prompt** 后跑一次，不针对结果调参。这样得到的数字才算 held-out 评估。

---

## 1. 先冻结，再出题

出题前记录：

```powershell
git rev-parse HEAD
git status --short
python -c "import hashlib; print(hashlib.sha256(open(r'llm/agent.py','rb').read()).hexdigest())"
```

记录：

- 代码 commit / 工作区是否干净；
- `llm/agent.py`、`llm/adapters/phm2019.py` 的 SHA256；
- 模型：`deepseek-chat`；
- 日期；
- 运行命令。

**出题人不要看 Agent 的报错和输出。**  
如果看了再改题，这批题就不是 held-out。

---

## 2. 只使用这些数据事实

数据文件：

- `phmdc/labels.csv`：`specimen, cycle, crack_mm`
- `phmdc/index.csv`：测量文件索引

可用指标：

| 指标 | 含义 | 单位 |
|---|---|---|
| `crack_mm_final` | 该试件最后一次测量的裂纹长度 | mm |
| `crack_mm_at_cycle` | 指定 cycle 的裂纹长度；必须用表中存在的 cycle | mm |
| `crack_delta_mm` | `crack(cycle_b) - crack(cycle_a)` | mm |
| `n_measurements` | 该试件的裂纹测量次数 | count |
| `n_specimens` | PHM2019 试件总数 | count |

不要出：

- `t85`、`t50`、寿命分数 —— PHM2019 没有这个指标；
- 没有测量记录的 cycle；
- 需要外推的裂纹值；
- “哪个试件最好”这类没有明确标准的题。

---

## 3. 建议题量与覆盖

15–20 条，至少覆盖：

| 类别 | 建议数量 | 说明 |
|---|---:|---|
| 最终裂纹 | 4–5 | 包含当前 11 条里没出现过的 T5/T6 |
| 指定 cycle 裂纹 | 3–4 | 用表里真实存在的 cycle |
| 裂纹增长量 | 2–3 | 两个真实 cycle 的差 |
| 测量次数 | 2 | 不同试件 |
| 试件总数 | 1 | 8 |
| 拒答 | 3–4 | 不存在试件、没有 t85、没有 PZT |

**不要只用 T1。**  
当前开发集已经大量使用 T1；held-out 应优先 T5、T6，以及 T2/T3/T8 的其他 cycle。

---

## 4. 任务 JSONL 格式

每行一个 JSON：

```json
{
  "task_id": "H01",
  "category": "external",
  "question": "T5 的最终裂纹长度是多少？",
  "expect": {
    "type": "numeric",
    "predicate": "crack_mm_final",
    "value": 3.64,
    "tol": 0.01
  },
  "should_refuse": false
}
```

拒答题：

```json
{
  "task_id": "H12",
  "category": "refusal",
  "question": "T5 的 t85 是多少？",
  "expect": { "type": "refusal" },
  "should_refuse": true
}
```

字段说明：

- `task_id`：`H01` 开始，不要和已有 `P01` 重复；
- `category`：`external` 或 `refusal`；
- `predicate` 必须精确等于上表指标名；
- `value` 用 `labels.csv` 里的数值，保留两位小数；
- `tol`：裂纹 0.01，计数 0；
- `should_refuse`：拒答题为 `true`。

---

## 5. 出题规则

1. **答案必须能从 `labels.csv` 直接查到**，不能靠推算。
2. 指定 cycle 题必须写表中真实存在的 cycle；不要用“大约 60000”。
3. 增长量题要写清两个 cycle，例如“从 50000 到 70766”。
4. 单位写清：mm 或“次”。
5. 拒答题要明确“为什么没有”：
   - 试件不存在（T9）；
   - 指标不存在（t85）；
   - 模态不存在（PHM2019 没有 PZT）。
6. 同一题不要同时问两个数值；拆成两题。
7. 不要写“根据论文”或“根据模型”；只问数据事实。

---

## 6. 运行方式

把文件命名为：

```text
llm/eval/tasks_phm_heldout.jsonl
```

运行一次：

```powershell
python llm\eval\run_eval.py `
  --tasks llm\eval\tasks_phm_heldout.jsonl `
  --results llm\eval\results_phm_heldout.csv
```

不要改：

- `agent.py`
- `adapters/phm2019.py`
- prompt
- `run_eval.py`

---

## 7. 评分与报告

至少报告：

- 总通过率；
- 按类别：最终裂纹 / 指定 cycle / 增长量 / 计数 / 拒答；
- 失败题的错误类型：
  - 工具路由错误；
  - 数值错误；
  - 单位错误；
  - 该拒答但回答；
  - 不该拒答但拒答。

如果失败：

- 先记录失败，不要立刻改 prompt；
- 判断是 **Adapter 缺能力**，还是 **Agent 路由/表达问题**；
- 修复后必须用**新的 held-out 集**重测，不能在同一批题上反复调。

---

## 8. 最小交付

- `tasks_phm_heldout.jsonl`（15–20 条）
- `results_phm_heldout.csv`
- 一段 200 字结论：
  - 通过率；
  - 哪类题失败；
  - 是 Adapter 问题还是 Agent 问题；
  - 下一批要补什么。

---

## 9. 一句话原则

> **出题人只看数据，不看 Agent；跑之前冻结代码，跑之后不回头改题。**
