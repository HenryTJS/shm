# Skill contracts

这些目录是 SHM 助手的任务单元。每个 skill 只做一件事，输入输出固定，数据通过 adapter 取得，LLM 不参与计算。

> 总体方案（原语 + workflow + API 教师 + 本地 planner + verifier）见 [`../README.md`](../README.md) 的“0. 当前方案”。

## 结果目录禁令

skill 层禁止直接打开 `*/results/*` 或 `*/out/*`：

- `skills/*/run.py` 不出现结果文件路径；
- `common.read_csv()` 对包含 `results` / `out` 的路径直接抛 `PermissionError`；
- `check` 通过 raw adapter 取数；`compare` 暂停，只返回 `unsupported`；
- `audit` 只读 claim–evidence 图和 schema；
- adapter 内部怎么缓存是数据层的事，不是 skill 层直读结果。

这条规则的目的，是防止 skill 把 `l1_hi_crosslab.csv` 里的 t85 当成“自己算的”。

## 当前范围

| skill | 用途 | 状态 |
|---|---|---|
| `check` | 看一个数据集/试件有什么、缺什么、能不能分析 | MVP |
| `metric` | 从原始 DTA/FBG 计算 t50/t85 | MVP |
| `compare` | 比较两个对象的一个指标，并给可比性边界 | 暂停（等 metric 扩展到 campaign） |
| `audit` | 校验一份 claim–evidence 图，列出失败项 | MVP |

`diagnose` 暂不在本轮契约里。

## 统一 envelope

所有 skill 返回同一个外层结构：

```json
{
  "skill": "check",
  "version": "0.1.0",
  "status": "ok",
  "data": {},
  "warnings": [],
  "evidence": [],
  "computation": {
    "kind": "read",
    "formula": null,
    "inputs": ["L1Adapter"]
  },
  "report": "Markdown 文本"
}
```

`computation.kind`：

- `read`：从 adapter/数据表读取已有值，没有重新计算；
- `derived`：由已有值做统计（如 median、delta）；
- `validated`：只校验证据，不产生新数字；
- `computed`：从原始曲线重新计算；`metric` 使用这一类。

`status`：

- `ok`：任务完成；
- `insufficient`：数据不够，明确说缺什么；
- `unsupported`：这个 skill 不支持该请求；
- `error`：输入或执行错误；
- `not_found`：对象不存在。

## 目录

```text
skills/
├─ common.py
├─ registry.json
├─ skill_result.schema.json
├─ check/
│  ├─ SKILL.md
│  ├─ schema.json
│  └─ run.py
├─ metric/
│  ├─ SKILL.md
│  ├─ schema.json
│  └─ run.py
├─ compare/
│  ├─ SKILL.md
│  ├─ schema.json
│  └─ run.py
└─ audit/
   ├─ SKILL.md
   ├─ schema.json
   └─ run.py
```

## 运行

```powershell
cd D:\shm
python llm\skills\check\run.py --dataset L1 --gid L1-03
python llm\skills\metric\run.py --dataset L1 --gid L1-06
python llm\skills\compare\run.py --dataset L1 --a C1 --b C3 --metric t85
python llm\skills\audit\run.py --graph llm\examples\L1-03_t85.json
```

每个脚本都支持 `--json`。

## 边界

- `check` 不判断“数据好坏”，只报告覆盖与可用性；
- `compare` 不做显著性检验，只给差值与可比性限制；
- `audit` 不抽取自由文本，只校验结构化 claim–evidence 图；
- 三个 skill 都不训练模型，不调用 LLM。
