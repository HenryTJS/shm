# compare

## 状态

**暂停。** 旧实现通过 `L1Adapter` 读 `l1/results`，违反 skill 数据政策，已移除。  
在 `metric` 从原始 `.pridb`/`.DTA` 算出 t85 之前，`compare` 不返回数值。

## 目的（恢复后）

比较 L1 benchmark 内两个对象的一个指标，给差值和可比性边界。  
不做显著性检验，不做跨物理比较。

## 输入

```json
{"dataset": "L1", "a": "C1", "b": "C3", "metric": "t85"}
```

## 输出（恢复后）

```json
{
  "a": "C1", "b": "C3", "metric": "t85",
  "comparison_type": "campaign_vs_campaign",
  "delta": 4.8,
  "unit": "percentage points",
  "direction": "b - a",
  "comparable": true
}
```

## 当前行为

```json
{
  "skill": "compare",
  "status": "unsupported",
  "data": {
    "reason": "raw metric not implemented; no result-table fallback"
  }
}
```

## 规则

1. 只支持 L1；
2. 试件和 campaign 不匹配时返回 `unsupported`；
3. 结果必须带 raw-derived evidence；
4. 不写“显著”“导致”这类词；
5. 没有 raw metric，不得用旧 t85 代替。

## 命令

```powershell
python llm\skills\compare\run.py --dataset L1 --a C1 --b C3 --metric t85
```

当前会返回 `unsupported`，这是预期行为。
