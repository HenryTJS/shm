# metric

## 目的

从**原始数据**计算 t50/t85，不读结果表。  
当前 v0 只支持 L1 的 `.DTA` 组（C3/C4）。

## 输入

```json
{"dataset": "L1", "gid": "L1-06"}
```

## 输出

```json
{
  "gid": "L1-06",
  "n_f": 202300,
  "n_frames": 303,
  "hits": 6189719,
  "f_hz": 1.117,
  "t50_pct": 68.02,
  "t85_pct": 93.76,
  "frame_s": 600.0,
  "raw_ae": "E:/l1/L1-06/AE",
  "raw_fbg": "E:/l1/L1-06/FBG"
}
```

`computation.kind = "computed"`。

## 算法

1. `ae_frames.group_frames(gid, root=E:/l1, frame_s=600)` 从原始 `.DTA` 建帧，得到每帧 hit 数；
2. `ae_cycle.group_load(gid, root=E:/l1)` 扫原始 FBG，得到加载/停机标签；
3. `ae_cycle._ratio_on_grid` 把加载标签映射到 AE 帧，得到 `g_load`；
4. `n_f` 用项目静态兜底表（`datasets.N_F_FALLBACK`），不读 results；
5. `cycle = cumsum(g_load × 600 × f)`，`HI = cumsum(hits) / total`；
6. 在 `HI=0.50/0.85` 处插值，得到 t50/t85。

## 规则

1. skill 层不读 `results/`；本 skill 只读 `E:/l1/<gid>/AE` 和 `FBG`；
2. 没有 raw FBG、没有 DTA 帧、没有 `n_f` → `unsupported`，不猜；
3. 结果缓存到 `skills/cache/metric_<gid>.json`，缓存是 raw-derived，不是 results；
4. 和旧结果表的差异必须保留，不强行对齐；旧表已删除。

## 当前边界

- C1/C2 是 `.pridb`，还没有 raw frame builder；
- L2/PHM2019 不支持；
- C4 文件大，单组计算慢，先缓存、不批量跑。

## 命令

```powershell
python llm\skills\metric\run.py --dataset L1 --gid L1-06
python llm\skills\metric\run.py --dataset L1 --gid L1-06 --json
```
