# -*- coding: utf-8 -*-
"""按指定试件集合重建模型工件（**只重拟与“哪些试件算数”直接相关的两项**）。

背景
----
`model_bundle_final.pkl` 由学生的训练侧生成（仓库里没有训练代码）。当用户裁定
「001/002/015 是问题数据，当它们不存在」时，需要把工件里**依赖这 27 组**的部分换掉：

  1. `library`  —— 近邻库成员（去掉被剔除的试件）
  2. `bayes`    —— 容量先验 `mu`/`sig`（= 库内 log 容量 的均值 / 标准差）

**不重训**：门控 Ridge 权重（`gate`）、特征归一化（`gate.mu/sd`）、报警阈值（`alarm`）。
这些需要训练侧的数据与流程，本仓库不具备；因此本脚本会把 `build_note` 写成
「仅在 N 组库上重拟库与容量先验，门控权重沿用原工件」，避免以后误以为是整包重训。

用法
----
    python main/cdio_baseline/refit_bundle.py --drop 001,002,015 \
        --out main/cdio_baseline/model_bundle_24.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "model_bundle_final.pkl"))
    ap.add_argument("--out", default=os.path.join(HERE, "model_bundle_24.pkl"))
    ap.add_argument("--drop", default="001,002,015", help="要当“不存在”的组号（逗号分隔）")
    a = ap.parse_args()

    b = pickle.load(open(a.src, "rb"))
    drop = {x.strip() for x in a.drop.split(",") if x.strip()}
    keep = [d for d in b["library"] if d["id"] not in drop]
    dropped = sorted({d["id"] for d in b["library"]} & drop)
    if not keep:
        raise SystemExit("剔除后库为空，检查 --drop")

    logs = np.array([np.log(float(np.max(d["CElog"]))) for d in keep], float)
    b["library"] = keep
    b["specimens"] = [d["id"] for d in keep]
    b["n_specimens"] = len(keep)
    b["bayes"] = {"mu": float(logs.mean()), "sig": float(logs.std(ddof=1))}
    b["build_note"] = (
        "飞行器寿命监测 final；CElog容量-组成指纹KNN + 库容量对数正态贝叶斯 + Ridge在线动态门控。"
        f"【派生件】仅按 {len(keep)} 组重建了 library 与 bayes(mu/sig)；"
        f"剔除 {','.join(dropped) if dropped else '无'}；"
        "门控权重/归一化/阈值沿用原工件（训练侧代码不在本仓库，未重训）。"
    )

    with open(a.out, "wb") as fh:
        pickle.dump(b, fh)
    print(f"写出 {a.out}")
    print(f"  库 {len(keep)} 组（剔除 {','.join(dropped) if dropped else '无'}）")
    print(f"  bayes mu={b['bayes']['mu']:.4f}  sig={b['bayes']['sig']:.4f}")


if __name__ == "__main__":
    main()
