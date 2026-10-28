# SHM Claim–Evidence Graph (v1.0)

这是“证据约束”的最小数据契约。它不是 prompt，也不是 UI，而是 **Agent 输出结论时必须遵守的结构**。

## 1. 为什么需要它

普通 RAG 的输出是“一段话”，无法判断哪一句有依据。  
这里把回答拆成 **claim（结论）** 和 **evidence（证据）** 两类对象，再用 `edges` 连接：

```
Question
  ├─ Claim C001  "L1-03 的 AE t85 = 86.4%"
  │    ├─ E001 tool_result  (l1_hi_ae.csv: t85_pct=86.4)
  │    └─ E002 literature   (t85 的定义)
  ├─ Claim C002  "比 C1 中位低 2.8 pp"
  │    └─ E003 tool_result  (C1 median=89.2, delta=-2.8)
  └─ Claim C003  "这是离线指标，不能当在线报警"
       └─ E004 literature   (使用最终值 C(1))
```

规则：

1. **没有 evidence 的 claim 不能是 `supported`**；
2. **numeric claim 必须有 `tool_result` evidence**，且数值在容差内；
3. **literature/standard evidence 必须有 quote + locator**；
4. **evidence 的时间范围不能超过 claim 允许的寿命范围**（防止偷看未来）；
5. **矛盾证据**必须让 claim 变成 `rejected` 或 `insufficient`；
6. `insufficient` 必须写清楚缺什么。

## 2. 文件

```
llm/
├─ README.md
├─ schema/
│  └─ claim_evidence.schema.json
├─ examples/
│  ├─ L1-03_t85.json
│  ├─ invalid_missing_evidence.json
│  └─ invalid_temporal_leakage.json
├─ validate.py
└─ tests/
   └─ test_validate.py
```

## 3. 用法

```powershell
cd D:\shm
python llm\validate.py llm\examples\L1-03_t85.json
python -m unittest llm.tests.test_validate -v
```

退出码：

- `0`：结构合法 + 语义检查通过；
- `1`：结构或语义失败（会打印 fail 项）。

## 4. 最小示例

见 `examples/L1-03_t85.json`。它包含：

- 1 个 numeric claim（t85=86.4）；
- 1 个 comparative claim（比 C1 中位低 2.8 pp）；
- 1 个 method claim（t85 的定义）；
- 1 个 limitation claim（离线、使用最终值）；
- 4 条 evidence（2 条 tool_result + 2 条 literature）。

## 5. 与后续模块的关系

- **工具层**：每个 Python 工具返回 `tool_result` evidence；
- **RAG**：每个检索片段返回 `literature` / `standard` evidence；
- **Adapter**：输出向量或摘要，作为 `tool_result` evidence 的一部分；
- **Writer**：只能写 claim，不能直接写无证据段落；
- **Critic**：调用 `validate.py` 的同一套规则，失败就打回；
- **Dashboard**：点击 claim 可展开 evidence 和 locator。

## 6. 设计边界

- 这是 **v1.0 最小契约**，不是最终 JSON Schema 标准；
- 先覆盖数值、比较、方法、局限四类 claim；
- 规范条文用 `standard` evidence，必须带 clause 编号；
- 多模态 scope 先支持 `AE / FBG / DFOS / PZT / MULTI`；
- 不把原始波形塞进 evidence；只放窗口级摘要或文件 locator。

## 7. 工具层（llm/tools.py）

每个工具返回三件事：

```python
{"tool_call": T001, "evidence": E001, "result": {...}}
```

当前工具：

- `lookup(gid, metric="t85")`：查单个试件的 t50/t85；
- `campaign_stats(campaign, metric)`：算批次中位/最小/最大；
- `compare_to_campaign(gid, metric)`：算试件相对批次中位的差值；
- `dataset_overview()`：返回 29 组/批次/变体清单。

示例：

```powershell
python llm\tools.py
```

## 8. RAG v0（llm/rag.py）

不需要 API key，纯本地 TF-IDF：

```powershell
python llm\rag.py build
python llm\rag.py query "t85 definition" -k 5
python llm\rag.py query "absolute threshold" --source literature
```

索引来自：

- `docs/refs/*.pdf`（9 篇）；
- `docs/iwshm2025/pdfs/*.pdf`（39 篇）；
- 本项目的 IWSHM/ICRE 稿。

每个片段带 `doc/page/char_start/char_end`，可直接转成 `literature` evidence。  
`docs/standards/` 目前为空；放入真实规范 PDF 后重新 `build`，才会生成 `standard` evidence。

## 9. 当前边界

- 工具层只覆盖 L1 AE 表；FBG/DFOS/PZT 和 L2 还没有接；
- RAG 是 TF-IDF v0，不是语义 embedding；后面可换 BGE；
- 还没有接 DeepSeek API；下一步是 `llm/agent.py` 的 planner/writer；
- 证据约束器已经能阻止无证据、数值不匹配、时间泄漏三类错误。
