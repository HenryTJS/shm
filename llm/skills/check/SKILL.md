# check

## 目的

回答“这个数据集/试件有什么、缺什么、能不能继续分析”。  
它不判断数据质量好坏，也不计算新指标。

## 输入

```json
{"dataset": "L1", "gid": "L1-03"}
```

- `dataset`：`L1` / `L2` / `PHM2019`
- `gid`：试件编号（L1-xx、L2-0x、T1–T8）

## 输出

统一 envelope，`data` 包含：

```json
{
  "dataset": "L1",
  "gid": "L1-03",
  "exists": true,
  "campaign": "C1",
  "fatigue_end": 152458,
  "modalities": [
    {"name": "AE", "available": true, "n": 152458, "t85_pct": null, "note": "cum_hits"}
  ],
  "capabilities": [
    {"name": "t85", "available": false, "reason": "metric skill not implemented"}
  ]
}
```

`status`：

- `ok`：至少一个模态可用；
- `insufficient`：对象存在，但没有可用模态；
- `not_found`：表里没有这个 gid。

## 规则

1. L1 的 AE 通过 `L1Adapter.lookup` 取得；
2. L1 的 FBG/DFOS/PZT 通过 `L1Adapter.modality_lookup` 取得，缺行就报不可用；
3. L2 的 PZT 覆盖率固定写死为 0 / 20.2 / 100，不允许把 L2-05 当全寿命；
4. PHM2019 只有 `crack_mm`，不提供 t85；
5. 每条可用性判断带 adapter 产出的 evidence；skill 层不直接读 results 文件。

## 失败模式

- `L2-04`：无 PZT → 只能 AE；
- `L2-05`：PZT 覆盖 20.2% → 不能算全寿命 t85；
- `L1-xx` 不在 `four_modal` 表：不推断有 FBG/DFOS/PZT；
- `PHM2019`：不要问 t85。

## 命令

```powershell
python llm\skills\check\run.py --dataset L1 --gid L1-03
python llm\skills\check\run.py --dataset L2 --gid L2-05 --json
```
