"""把 CDIO 在线寿命预测批量跑到 27 个主样本上，导出大屏可读曲线。

替代已下线的 D(t) 交付链：大屏从此显示"累计能量进度预测"，不再显示损伤度。

输入：D:\\同步\\5-2 CDIO项目实践（2）\\final_code\\_work\\processed\\main\\*_aligned.csv
      每份含 u（真实记录进度 %）、ae_time_s、ae_SpectralEnergy、ae_Peak。
输出：dashboard/data/cdio/<gid>.js  （window.SHM_LIFE[gid]）
      main/cdio_baseline/results/cdio_summary.csv

留一（exclude_ids=[gid]）保证不是"自己预测自己"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_predictor import LiveLifePredictor  # noqa: E402

SRC = r"D:\同步\5-2 CDIO项目实践（2）\final_code\_work\processed\main"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "dashboard", "data", "cdio"))
RES = os.path.join(HERE, "results")
COLS = ["u", "life_pct", "ae_time_s", "ae_SpectralEnergy", "ae_Peak"]
TARGET_PTS = 1200


def _cross(t, p, level):
    """预测序列首次达到 level(%) 的时刻与窗口号。"""
    idx = np.where(np.asarray(p) >= level)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    return {"t": float(np.asarray(t)[i]), "f": int(i), "lv": level}


def _at_true_u(u, pred, level):
    """真实进度到达 level% 时的预测值（标定检查）。"""
    idx = np.where(np.asarray(u) >= level)[0]
    if len(idx) == 0:
        return None
    return round(float(np.asarray(pred)[int(idx[0])]), 2)


def run_one(gid, path, bundle, every_override=None, exclude=()):
    df = pd.read_csv(path, usecols=lambda c: c in COLS)
    n = len(df)
    every = every_override or max(10, n // TARGET_PTS)
    excl = sorted({gid} | {x for x in exclude if x != gid})
    p = LiveLifePredictor(bundle, exclude_ids=excl)
    t0 = time.time()
    res = p.run_dataframe(df, every=every)
    if res.empty:
        return None, None
    wi = res["window_index"].astype(int).values
    u_true = pd.to_numeric(df["u"], errors="coerce").values
    u = np.array([float(u_true[i - 1]) if i - 1 < len(u_true) else np.nan for i in wi])
    u = u * 100.0

    life = res["life_pct"].astype(float).values
    ok = ~np.isnan(u)
    # 预热段（前 5% 窗）不计入误差
    warm = wi > max(30, int(0.05 * n))
    m = ok & warm
    mae = float(np.nanmean(np.abs(life[m] - u[m]))) if m.any() else float("nan")

    obj = {
        "gid": gid,
        "n": int(n),
        "stride": int(every),
        "window_s": float(p.b["window_s"]),
        "dur": float(res["time_s"].astype(float).max()),
        "t": [round(float(x), 2) for x in res["time_s"].astype(float).values],
        "wi": [int(x) for x in wi],
        "life": [round(float(x), 2) for x in life],
        "knn": [round(float(x), 2) for x in res["knn_pct"].astype(float).values],
        "bayes": [round(float(x), 2) for x in res["bayes_pct"].astype(float).values],
        "lo": [round(float(x), 2) for x in res["bayes_p05"].astype(float).values],
        "hi": [round(float(x), 2) for x in res["bayes_p95"].astype(float).values],
        "u": [None if np.isnan(x) else round(float(x), 2) for x in u],
        "warn": [w for w in (_cross(res["time_s"].astype(float).values, life, lv)
                             for lv in (25, 55, 85)) if w],
        "meta": {
            "life_end": round(float(life[-1]), 2),
            "u_end": round(float(u[-1]), 2) if not np.isnan(u[-1]) else None,
            "mae": round(mae, 2),
            "at_u25": _at_true_u(u, life, 25),
            "at_u55": _at_true_u(u, life, 55),
            "at_u85": _at_true_u(u, life, 85),
            "excluded_lib": excl,
            "alive_end": round(float(res["alive_frac"].astype(float).values[-1]), 3),
            "sec_used": round(time.time() - t0, 1),
        },
    }
    return obj, {"gid": gid, "n": n, "stride": every, "mae": obj["meta"]["mae"],
                 "life_end": obj["meta"]["life_end"], "u_end": obj["meta"]["u_end"],
                 "at_u25": obj["meta"]["at_u25"], "at_u55": obj["meta"]["at_u55"],
                 "at_u85": obj["meta"]["at_u85"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default=None, help="逗号分隔的组号，如 016,018")
    ap.add_argument("--exclude", default="",
                    help="从建库中剔除的组号（逗号分隔），如 001,002；被剔除组本身仍可被预测")
    ap.add_argument("--skip", default="",
                    help="不生成数据包的组号（逗号分隔）")
    ap.add_argument("--every", type=int, default=None)
    ap.add_argument("--bundle", default=os.path.join(HERE, "model_bundle_24.pkl"),
                    help="模型工件路径（默认 model_bundle_24.pkl：24 组库，001/002/015 已剔除；"
                         "原始 27 组工件见 model_bundle_final.pkl）")
    ap.add_argument("--res", default=None,
                    help="汇总 CSV 目录（默认 main/cdio_baseline/results；A/B 对比时用独立目录）")
    a = ap.parse_args()

    bundle = a.bundle
    res_dir = a.res or RES
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(a.src) if f.endswith("_aligned.csv"))
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        files = [f for f in files if f.split("_")[0] in want]
    print(f"共 {len(files)} 组，输出到 {a.out}")

    rows, idx = [], []
    excl = tuple(x.strip() for x in a.exclude.split(",") if x.strip())
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    for f in files:
        gid = f.split("_")[0]
        if gid in skip:
            print(f"[跳过] {gid}: 在 --skip 列表")
            continue
        try:
            obj, row = run_one(gid, os.path.join(a.src, f), bundle, a.every, exclude=excl)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {gid}: {e}")
            continue
        if obj is None:
            print(f"[跳过] {gid}: 无有效预测")
            continue
        js = ('window.SHM_LIFE=window.SHM_LIFE||{};'
              f'window.SHM_LIFE[{json.dumps(gid)}]={json.dumps(obj, ensure_ascii=False)};')
        with open(os.path.join(a.out, f"{gid}.js"), "w", encoding="utf-8") as fh:
            fh.write(js)
        rows.append(row)
        idx.append({"gid": gid, "life_end": obj["meta"]["life_end"],
                    "u_end": obj["meta"]["u_end"], "mae": obj["meta"]["mae"]})
        print(f"  {gid}  n={row['n']:6d} stride={row['stride']:4d} "
              f"MAE={row['mae']:6.2f}  末值预测={row['life_end']:6.2f}  "
              f"真实={row['u_end'] if row['u_end'] is None else round(row['u_end'],1)}  "
              f"({obj['meta']['sec_used']}s)")

    summ = pd.DataFrame(rows)
    if not summ.empty:
        summ.to_csv(os.path.join(res_dir, "cdio_summary.csv"), index=False, encoding="utf-8-sig")
        print("\n=== 汇总 ===")
        print(f"组数 {len(summ)}   MAE 中位 {summ['mae'].median():.2f}   均值 {summ['mae'].mean():.2f}")
    with open(os.path.join(a.out, "_index.js"), "w", encoding="utf-8") as fh:
        fh.write("window.SHM_LIFE_INDEX=" + json.dumps(idx, ensure_ascii=False) + ";")
    print(f"索引写入 {os.path.join(a.out, '_index.js')}")


if __name__ == "__main__":
    main()
