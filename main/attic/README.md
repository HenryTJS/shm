# main/attic —— 已下线的评估链路

2026-10-20 口径收缩时把下列内容移出主路径，**保留备查，不在任何入口被调用**。

## 为什么下线

主样本的累积 AE 能量与寿命近似线性（`u@90% = 0.90`、`R² = 0.9999`、
曲线偏离不超过总容量 6%，而库容量离散度 6.1 倍），它是时钟而不是损伤计。
在此基础上建立的弱标签 `b1/b2/b3`、由它算出的偏差与 A/E/D 分级、
以及依赖这些的基线对照与 LOSO，全部失去可辨识性。

详见 `docs/当前结论与口径.md`。

## 移入的文件

| 文件 | 原用途 |
| --- | --- |
| `verify.py` | 验证层：基线对照、LOSO、标签审计、AE 逐列对照、分级闸门、源融合、候选体检 |
| `merge_into_dashboard.py`、`probe_*.py`（9 个） | 旧看板打补丁链 + 2026-10-20 的一次性诊断；**已并入本节后于 2026-10-21 彻底删除**（用户确认；结论与数值已记入 `main/results/` 与 `docs/当前结论与口径.md`，证据表里的路径不再指向脚本） |

`main/probe_ae_label_uncertainty.py` 早已在此（旧位置即 `main/attic/probes/`）。

仍在原处但**已不可从 `run.py` 调用**：

- `main/pipeline.py` 的 `evaluate` 段：`degree` / `warning` / `curves` / `paper`
- `main/pipeline.py` 的 `prepare weaklabels` 段：弱标签生成
- `main/pipeline.py` 的 `robust` 段：参数灵敏度、LOSO、消融、统计检验

这些代码保留是为了让上面那些结论仍可复算，不构成交付物。

## 已删除的产物

`main/results/` 下的 `damage_degree_metrics.csv`、`warning_onset*.csv`、
`grade_*.csv`、`baseline_*.csv`、`fusion_*.csv`、`loso_*.csv`、
`leave_one_out_cv.csv`、`label_audit.csv`、`candidate_check.csv`、
`inclusion_criteria.csv`、`parameter_sensitivity.csv`、`statistical_test.csv`、
`ablation.csv`、`per_specimen_stats.csv`、`ae_column_eval*.csv` 及 `_abl_ext*.csv`；
`main/weak_labels/*.csv`；`main/cache/_hi_cache/*.npy`。

均可由被保留的代码重跑，因此不留档。

## 保留在 main/results 的诊断记录

这些是支撑「收缩」结论本身的数据，**不是结论性交付**：

`ae_channel_diag.*`（25 列有效维度与逐通道变点）、`ae_label_uncertainty.csv`
（b1/b3 的跨方法跨度）、`early_event.*`（早期事件分类）、`knee_peaks.csv`
（台阶位置与高度）、`label_sensitivity.*`（口径灵敏度）、`label_v1_v2.md`、
`early_alarm_023_024.md`（023/024 过早报警归因）。
