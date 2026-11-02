# audit

## 目的

校验一份已经结构化的 claim–evidence 图。  
它不抽取自由文本，不调用 LLM，只回答：

- 哪些 claim 没有证据；
- 哪些数值与工具结果不一致；
- 哪些引用没有原文片段；
- 哪些时间范围越界。

## 输入

```json
{"graph": "llm/examples/L1-03_t85.json"}
```

`graph` 符合 `llm/schema/claim_evidence.schema.json`。  
如果只有 Markdown 报告，需要先由上游抽成 claim–evidence 图，本 skill 不做抽取。

## 输出

```json
{
  "graph": "llm/examples/L1-03_t85.json",
  "n_claims": 4,
  "n_evidence": 4,
  "n_failed": 0,
  "failed_checks": []
}
```

`status`：

- `ok`：全部通过；
- `insufficient`：有失败项，报告里逐条列出；
- `error`：文件读不了/JSON 不合法。

## 规则

1. 复用 `llm/validate.py`，不另写一套规则；
2. 失败项必须保留 `claim_id` 和 `evidence_id`，方便回查；
3. 不把“校验失败”改写成“模型不确定”；
4. 不在这里改 claim，只报告问题。

## 命令

```powershell
python llm\skills\audit\run.py --graph llm\examples\L1-03_t85.json
python llm\skills\audit\run.py --graph llm\examples\invalid_missing_evidence.json --json
```
