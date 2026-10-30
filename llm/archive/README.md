# Archive

本目录不参与主链路，保留原因如下。

## adapter_v0/

PHM2019 波形特征 → 裂纹长度的小型 adapter 实验。

- `adapter_v0.py`：MLP（39→64→16→1），LOSO MAE ≈ 4.0 mm；
- `adapter_v0_rf.py`：RandomForest，LOSO MAE ≈ 1.416 mm；
- 全局均值基线：≈ 1.872 mm；
- 结论：小样本下 MLP 过拟合，RF 仅略优于均值；不作为主线。

## drafts/

PHM2019 机械草稿题与生成脚本，供人工改写使用；不是正式评测集。

正式评测集在：

- `../eval/tasks.jsonl`
- `../eval/tasks_phm.jsonl`
- `../eval/tasks_phm_heldout.jsonl`

## 注意

这些文件不要被主链路 import。  
如果要复用，先确认数据和协议版本。
