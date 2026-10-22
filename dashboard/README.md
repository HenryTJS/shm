# 数字大屏

纯前端回放看板：双击 `index.html` 即可用，不需要后端、不需要构建。一套代码容纳两种口径——
主样本的**在线进度**与 L1 公开集的**离线损伤复评**——顶栏切换即可。

## 1. 打开

```bat
rem 直接双击
dashboard\index.html

rem 或用本地 http 服务（需要按目录加载时更稳）
python -m http.server 8080 --directory dashboard
```

## 2. 数据集

| 数据集 | 组数 | 传感 | 时间基 | 主指标 |
| --- | --- | --- | --- | --- |
| 疲劳机主样本 003–014 / 016–027 | 24 | AE + 应变 + FBG | 秒（帧 0.5 s） | 在线进度 `D`（留一预测） |
| ReMAP C1 恒幅+FBG+DFOS | 5 | AE + FBG + DFOS + PZT | cycle（50 cycle/帧） | 离线 `HI_hit` |
| ReMAP C2 恒幅+DFOS | 9 | AE + DFOS | cycle（50 cycle/帧） | 离线 `HI_AE` |
| ReMAP C3 变幅VA+FBG | 5 | AE + FBG + PZT | cycle（600 s 帧） | 离线 `HI_hit` |
| ReMAP C4 谱载+FBG | 10 | AE + FBG + PZT | cycle（600 s 帧） | 离线 `HI_hit` |

## 3. 页面与操作

| 区块 | 内容 |
| --- | --- |
| 顶栏 | 数据集 / 试件 / 试验台架 / 采集模式 / 状态灯 / 时钟 |
| 控制条 | 开始 · 暂停 · 复位 · 声音 · 回放倍率 30×–600× · 进度拖拽 |
| 主指标 | 当前值、分级色环、峰值、寿命进度、偏差/剩余量 |
| 趋势图 | 主指标 + 证据曲线、阈值带、锚点、90% 区间（主样本） |
| 分级预警 | 三级灯、各级触发时刻、证据条（e_ae / e_st / risk） |
| 多源面板 | ① AE 事件率 + 峰值 ② FBG 通道 ③ 应变或 DFOS ④ 证据层分解 |
| 空间分布 | 仅 L1：应变相对基线的空间偏离热图 + 当前分布 |
| 底栏 | 事件日志、通道健康表 |

快捷键：`空格` 播放/暂停，`←` `→` 前后 200 帧。

**在线语义**：新选一组时看不到任何未来信息，曲线从起点随回放生长，锚点/预警灯只在真实发生后才出现；拖进度条等于在该时刻重新观察。

## 4. 读数口径

- **主样本**：`D` 是留一预测的寿命百分比。「进度偏差」= `D × 100 − 真实寿命百分比`（百分点，负值标橙）。
- **L1**：主指标是离线复评 `HI = unity01(累积 AE hits)`，副读数显示 `1 − HI`；顶部标注「离线复评（非在线）」。
- **阈值 0.25 / 0.55 / 0.85 只是分档刻度**，两级之间不承诺触发间隔，也不代表剩余寿命裕度。
- **预测中止**：个别试件末期超出建库容量，预测不再出数（不是外推）；此时曲线为末值保持，并标出中止点。
- **90% 区间**：主样本带贝叶斯 p05–p95 半透明带；末期点估计偏低，应看区间上界，不宜单读点。

## 5. 数据包字段（要点）

```js
window.SHM_DATA["<gid>"] = {
  gid, n, nfr, step, dt, dur,        // 点数 / 帧数 / 降采样步长 / 帧间隔 / 时长
  labels, chans, meta, warn,         // 文案覆盖 / 通道表 / 统计锚点 / 分级事件
  D, risk, eae, est, lv, st, ael, aen, t, fo
}
```

| 字段 | 说明 |
| --- | --- |
| 编码 | `D/risk/eae/est` ×1000，`st/fo` ×100，`lv` 0–3，`ael` = log10(峰值)×1000（无事件 −99999），`aen` = 帧内事件数 |
| `meta` | `t25/t55/t85`、锚点 `b2/b3`、`c0Pct`、`aeEvents`、`nFo`、`truncPct`（主样本预测中止点） |
| 主样本特有 | `lo`/`hi`（90% 区间 ×1000）、`spec.excluded` |
| L1 特有 | `unit:'cycle'`、`frameDt`、`chans`（AE/FBG/DFOS/引擎）、`dfos`（空间分布与块级序列）、`ax`（面板纵轴） |
| 二进制 | 数值数组写成 `{"i16"|"i32"|"f32": "<base64>"}`（`shm/pkgcodec.py`），前端 `hydrateAll()` 解码；旧纯数组包仍可加载 |

## 6. 重建数据包

`dashboard/data/` 由脚本生成、**不入库**（`.gitignore` 已忽略），换机器后按下面重建：

```bat
set PY=C:\Users\ASUS\.conda\envs\xjtushm\python.exe

%PY% main\cdio_baseline\build_dashboard_all.py --write   rem 主样本 24 组
%PY% l1\restore_dashboard_packages.py                    rem L1 29 组（C1–C4）
%PY% l1\build_campaign_index.py                          rem C1–C4 四个清单
```

主样本链路是两步：`export_dashboard.py` 逐组留一预测 → `build_dashboard_all.py` 合并原始信号出包。
L1 链路是回放归档导出器，分组真源在 `shm/datasets.L1_CAMPAIGNS`。

## 7. 使用边界

- L1 的主曲线是**离线量**（自归一化用到全寿命终值），界面据此标注，不当在线能力用。
- C4 的 L1-30 / L1-34 / L1-36 的 AE 只覆盖寿命的一小段（帧长 60 s / 30 s），包内脚注已声明，**不要与全寿命曲线并列比较**。
- 看板只做回放与读数，不重新训练、不引入未来信息。
