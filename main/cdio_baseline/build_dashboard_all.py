"""从对齐后的原始数据构建看板数据包（展示 001-027；**001/002 不进建库与统计**）。

与 merge_into_dashboard.py 的区别：那个只能在既有数据包上打补丁（于是只覆盖了
016-020/022-027 共 11 组）；本脚本直接吃 *_aligned.csv，因此 001-015 与 021
也能生成完整数据包，且 27 组通道口径完全一致。

⚠️ **001/002 为问题数据**（用户裁定 2026-10-21）：
  不进建库（见 `export_dashboard.py --exclude 001,002`）、不进任何误差统计；
  看板仍可浏览，但 `spec.kind` 标「已排除」、`meta.excluded=true`。
  实测：从库里剔除后，其余 25 组的预测曲线逐组不变（MAE 完全一致）。

主指标：
  D    = CDIO 在线预测的记录进度（×1000 整数，沿用前端 dec(x,1000) 约定）
       —— 已做**单调化**（cummax）：进度不应倒退。
证据层（与 D 同尺度，均为 0–1000 的归一化累计量，来自原始信号自身）：
  eae  = 累计对数谱能量（AE 单源）
  est  = 累计应变绝对增量（应变单源）
  risk = 两者均值

数值数组写包时用 base64(Int16/Int32/Float32) 编码（`shm/pkgcodec.py`），
前端 `dashboard.js hydrateAll()` 解码；旧包（纯数组）仍兼容。

输入  D:\\同步\\5-2 CDIO项目实践（2）\\final_code\\_work\\processed\\main\\*_aligned.csv
      dashboard/data/cdio/<gid>.js（预测曲线，由 export_dashboard.py 产出）
输出  dashboard/data/<gid>.js + dashboard/data/index.js
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from shm.pkgcodec import encode_package  # noqa: E402

DASH = os.path.abspath(os.path.join(HERE, "..", "..", "dashboard", "data"))
SRC = r"D:\同步\5-2 CDIO项目实践（2）\final_code\_work\processed\main"
WINDOW_S = 0.1594
TARGET_PTS = 8000
STEP = 5
DT_REAL = 0.1

# 明确排除的试件（用户裁定，2026-10-21）：**不进建库、不进统计、不进看板**。
# 001/002 末期偏差过大、015 时间列/采样不可识别。
# 与 main/cdio_baseline/export_dashboard.py 的默认工件 model_bundle_24.pkl 严格一致。
EXCLUDE = {"001", "002", "015"}

LABELS = {
    "indexName": "累计能量进度",
    "trendName": "预测进度 vs 证据层",
    "labels": {
        "unit": "%",
        "legend": "预测进度",
        "margin": "进度偏差",
        "marginMode": "dev",      # dev = 预测进度 − 真实进度（百分点）；看板据此计算
        "thrName": "进度",
        "peak": "末值进度",
        "foot": "口径：累计对数谱能量的回归（时钟），非损伤力学量。"
                "{T} ≥ 0.25 / 0.55 / 0.85 为记录完成度分档，不代表剩余寿命的物理裕度。"
                "证据层 = AE 累计能量 / 应变累计增量各自的归一化曲线。",
    },
}


def load_pred(path):
    s = open(path, encoding="utf-8").read()
    head = "window.SHM_LIFE["
    i = s.index(head) + len(head)
    j = s.index("]=", i) + 2
    return json.loads(s[j:].rstrip().rstrip(";"))


def build(gid, csv_path, pred):
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in ("ae_SpectralEnergy", "ae_Peak",
                                "strain", "fiber_1", "fiber_2", "fiber_3",
                                "fiber_4", "fiber_5"),
    )
    n = len(df)
    if n == 0:
        return None

    # 横轴用事件序号 × window_s，不用 ae_time_s：
    # 016 的 31442 × 0.1594 = 5012 s 与真实时长吻合，但 001/002/015 的时间列失真
    # （015 的 54128 个事件全挤在 1.0 s 内）。模型只依赖累计能量、不依赖时间，
    # 因此统一以事件序号为栅格最自洽。
    step = max(1, int(np.ceil(n / TARGET_PTS)))
    idx = np.arange(0, n, step)
    nfr = int(idx.size)
    dt = round(WINDOW_S * step, 4)
    dur = round(n * WINDOW_S, 1)
    t = np.round((idx + 1) * WINDOW_S, 2)

    E = pd.to_numeric(df["ae_SpectralEnergy"], errors="coerce").fillna(0).values
    P = pd.to_numeric(df["ae_Peak"], errors="coerce").fillna(0).values
    le = np.log1p(np.clip(E, 0, None))

    # 区间内的事件数（每行即一个 AE 事件，故约为常数 step）
    edges = np.append(idx, n)
    aen = np.diff(edges).astype(int).tolist()

    pmax = np.maximum.reduceat(np.clip(P, 0, None), idx)
    with np.errstate(divide="ignore"):
        ael = np.where(pmax > 0, 2000.0 * np.log10(np.maximum(pmax, 1e-12)), -3000.0)
    ael = np.clip(ael, -3000, 1000).round().astype(int).tolist()

    def chan(col, scale=1000.0):
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce").values
        s = pd.Series(v).ffill().bfill().fillna(0.0).values[idx]
        return np.round(s * scale).astype(int).tolist()

    st = chan("strain")
    fo = {}
    for k in range(1, 6):
        c = chan(f"fiber_{k}")
        if c is not None:
            fo[f"s{k}"] = c

    # 证据层：归一化累计量（0-1000），分别来自 AE 与应变自身
    cum_e = np.cumsum(le)
    tot_e = cum_e[-1]
    eae = (cum_e[idx] / tot_e * 1000.0) if tot_e > 0 else np.zeros(nfr)
    if st is not None:
        d_st = np.abs(np.diff(np.asarray(st, dtype=float), prepend=0.0))
        cum_s = np.cumsum(d_st)
        est = (cum_s / cum_s[-1] * 1000.0) if cum_s[-1] > 0 else np.zeros(nfr)
    else:
        est = np.zeros(nfr)
    risk = (eae + est) / 2.0

    # 预测曲线按 window_index 对齐（彻底绕开时间列）。
    # 注意：**不能把最后一个预测窗强行拉到记录末端**——个别组（005/009）的预测流会在
    # 中途中止（容量超出库覆盖 → predict() 返回 life_pct=None），硬拉会把“无数据”
    # 画成一条停在 83%/87% 的平线，看上去像“末期严重低估”。
    # 这里如实按真实窗口号对齐，中止点之后由 np.interp 保持末值，并在图上标出中止位置。
    stride = int(pred.get("stride") or 1)
    lp = np.asarray(pred["life"], dtype=float)
    wi = np.asarray(pred.get("wi") or (np.arange(1, lp.size + 1) * stride), dtype=float)
    d01 = np.clip(np.interp(idx + 1, wi, lp), 0, 100) / 100.0
    d01[idx + 1 < wi[0]] = lp[0] / 100.0
    # 进度不应倒退：模型偶有回落（如 012 末段 78 -> 69），取累计最大。
    # 实测（25 组，已排除 001/002）总体 MAE 9.07 -> 8.85，末期 13.4 -> 12.1。
    d01 = np.maximum.accumulate(d01)

    # 90% 区间（贝叶斯 p05/p95）：末期点估计不可信，区间才是可采信的
    lo_w = np.asarray(pred.get("lo") or [], dtype=float)
    hi_w = np.asarray(pred.get("hi") or [], dtype=float)
    if lo_w.size == lp.size and hi_w.size == lp.size:
        lo01 = np.clip(np.interp(idx + 1, wi, lo_w), 0, 100) / 100.0
        hi01 = np.clip(np.interp(idx + 1, wi, hi_w), 0, 100) / 100.0
        lo01 = np.minimum(np.maximum.accumulate(lo01), d01)   # 保持 lo <= D <= hi
        hi01 = np.maximum(np.maximum.accumulate(hi01), d01)
    else:
        lo01 = hi01 = None

    # 预测中止点：u_end 是“最后一个出数窗口”对应的真实进度（%）。
    # 它 < 99.5 就说明预测流没走完，后面看到的是末值保持，不是预测。
    u_end = pred.get("meta", {}).get("u_end")
    trunc = round(float(u_end), 1) if (u_end is not None and u_end < 99.5) else None
    d_int = np.round(d01 * 1000).astype(int)

    lv = np.where(d01 >= 0.85, 3, np.where(d01 >= 0.55, 2, np.where(d01 >= 0.25, 1, 0)))
    warn = []
    for k, lev in enumerate((0.25, 0.55, 0.85)):
        idx = np.where(d01 >= lev)[0]
        if idx.size:
            i0 = int(idx[0])
            warn.append({"f": i0, "t": round(float(t[i0]), 1), "lv": k + 1})

    def pct(lev):
        idx = np.where(d01 >= lev)[0]
        return round(float(t[int(idx[0])]) / dur * 100.0, 1) if idx.size else None

    obj = {
        "gid": gid,
        "n": int(n),
        "nfr": nfr,
        "step": int(step),
        "dt": dt,
        "dur": dur,
        "dtReal": DT_REAL,
        "foCols": list(fo.keys()),
        "spec": ({"kind": "已排除", "load": "—",
                  "note": "数据问题：未参与建库与统计，仅供浏览"}
                 if gid in EXCLUDE else {"kind": "—", "load": "—", "note": ""}),
        "meta": {
            "D_end": round(float(d01[-1]), 3),
            "t25": pct(0.25), "t55": pct(0.55), "t85": pct(0.85),
            "b2": None, "b3": None,
            "aeEvents": int(n), "nFo": len(fo),
            "truncPct": trunc,
            "life_end": pred["meta"]["life_end"],
            "u_end": pred["meta"]["u_end"],
            "mae": pred["meta"]["mae"],
            "at_u25": pred["meta"]["at_u25"],
            "at_u55": pred["meta"]["at_u55"],
            "at_u85": pred["meta"]["at_u85"],
            "src": "CDIO 在线预测（留一）",
            "excluded": gid in EXCLUDE,
        },
        "warn": warn,
        "t": [round(float(x), 1) for x in t],
        "D": d_int.tolist(),
        "lv": [int(x) for x in lv],
        "lo": (np.round(lo01 * 1000).astype(int).tolist() if lo01 is not None else []),
        "hi": (np.round(hi01 * 1000).astype(int).tolist() if hi01 is not None else []),
        "risk": np.round(risk).astype(int).tolist(),
        "eae": np.round(eae).astype(int).tolist(),
        "est": np.round(est).astype(int).tolist(),
        "aen": aen,
        "ael": ael,
        "st": st if st is not None else [0] * nfr,
        "fo": fo,
    }
    obj.update(copy.deepcopy(LABELS))
    if lo01 is not None:
        obj["meta"]["band"] = {"lo_end": round(float(lo01[-1]), 3),
                                "hi_end": round(float(hi01[-1]), 3)}
    if trunc is not None:
        # 中止点：复用 refs 竖线 + 图例（refsName），并写进 footer
        obj["meta"]["refs"] = [{"pct": trunc, "label": "预测中止"}]
        obj["labels"]["refsName"] = "预测中止点"
        obj["labels"]["foot"] += (
            f"⚠ 本组预测在寿命 {trunc}% 后中止（容量超出建库范围，模型不再出数），"
            "其后曲线为末值保持，不是预测。")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--data", default=DASH)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    life_dir = os.path.join(a.data, "cdio")
    index, n = [], 0
    files = sorted(f for f in os.listdir(a.src) if f.endswith("_aligned.csv"))
    print(f"待构建 {len(files)} 组")
    for f in files:
        gid = f.split("_")[0]
        if gid in EXCLUDE:
            print(f"  [排除] {gid}: 在 EXCLUDE 内（不建库、不进统计、不出包）")
            continue
        pp = os.path.join(life_dir, f"{gid}.js")
        if not os.path.exists(pp):
            print(f"  [跳过] {gid}: 缺预测曲线")
            continue
        obj = build(gid, os.path.join(a.src, f), load_pred(pp))
        if obj is None:
            print(f"  [跳过] {gid}: 无有效时间")
            continue
        if a.write:
            # 数值数组 → base64 二进制（Int16/Int32/Float32），前端 hydrateAll() 解码
            payload = copy.deepcopy(obj)
            encode_package(payload)
            js = ('window.SHM_DATA=window.SHM_DATA||{};'
                  f'window.SHM_DATA[{json.dumps(gid)}]={json.dumps(payload, ensure_ascii=False)};')
            with open(os.path.join(a.data, f"{gid}.js"), "w", encoding="utf-8") as fh:
                fh.write(js)
            kb = len(js.encode("utf-8")) / 1024
            obj["_kb"] = round(kb, 1)
        index.append({k: obj[k] for k in
                      ("gid", "nfr", "dur", "n", "meta", "foCols", "spec", "warn")})
        index[-1]["indexName"] = LABELS["indexName"]
        n += 1
        print(f"  {gid}  nfr={obj['nfr']:6d} dur={obj['dur']:8.1f} "
              f"进度末值 {obj['meta']['D_end']*100:5.1f}%  "
              f"AE事件 {obj['n']:6d}  光纤 {len(obj['fo'])}ch"
              + (f"  包 {obj['_kb']:.0f} KB" if "_kb" in obj else "")
              + ("  [已排除]" if gid in EXCLUDE else ""))

    if a.write and index:
        # 清掉被排除组的历史包，避免“库里没有、看板上还在”
        for g in sorted(EXCLUDE):
            fp = os.path.join(a.data, f"{g}.js")
            if os.path.exists(fp):
                os.remove(fp)
                print(f"  [清理] 已删除 {g}.js")
        with open(os.path.join(a.data, "index.js"), "w", encoding="utf-8") as fh:
            fh.write("window.SHM_INDEX=" + json.dumps(index, ensure_ascii=False) + ";")
        print(f"索引已重建：{len(index)} 组")
    print(f"{'已写回' if a.write else '试运行'}：{n} 组")


if __name__ == "__main__":
    main()
