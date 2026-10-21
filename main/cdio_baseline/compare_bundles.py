# -*- coding: utf-8 -*-
"""对比两套模型工件/预测目录的误差（同一组集，逐阶段）。

用途：判断“把某几组当不存在”是否真的更好，而不是拍脑袋。
两个输入目录都应含 `window.SHM_LIFE["gid"]` 形式的数据包（export_dashboard.py 产出）。

用法
----
    python main/cdio_baseline/compare_bundles.py \
        --a dashboard/data/cdio --b dashboard/data/cdio_24 --drop 001,002,015
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

STAGES = [(0, 25), (25, 50), (50, 75), (75, 100)]


def load_dir(d, drop=()):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.js"))):
        s = open(f, encoding="utf-8").read()
        m = re.search(r'window\.SHM_LIFE\["([0-9]{3})"\]\s*=\s*', s)
        if not m:
            continue
        gid = m.group(1)
        if gid in drop:
            continue
        d_ = json.loads(s[m.end():].rstrip().rstrip(";"))
        u = np.array([np.nan if x is None else x for x in d_["u"]], float)
        p = np.array(d_["life"], float)
        ok = np.isfinite(u) & np.isfinite(p)
        out[gid] = (u[ok], p[ok])
    return out


def metrics(pairs):
    res = {}
    for lo, hi in STAGES:
        e = [np.abs(p[(u >= lo) & (u < hi)] - u[(u >= lo) & (u < hi)])
             for u, p in pairs.values() if ((u >= lo) & (u < hi)).sum()]
        res[f"{lo}-{hi}"] = float(np.concatenate(e).mean()) if e else float("nan")
    res["all"] = float(np.concatenate([np.abs(p - u) for u, p in pairs.values()]).mean())
    res["end_pred"] = float(np.mean([p[-1] for _, p in pairs.values()]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线目录")
    ap.add_argument("--b", required=True, help="候选目录")
    ap.add_argument("--drop", default="", help="两组都不参与统计的组号")
    a = ap.parse_args()
    drop = {x.strip() for x in a.drop.split(",") if x.strip()}

    A, B = load_dir(a.a, drop), load_dir(a.b, drop)
    common = sorted(set(A) & set(B))
    print(f"公共组 {len(common)} 组（排除 {','.join(sorted(drop)) or '无'}）")
    ma, mb = metrics({g: A[g] for g in common}), metrics({g: B[g] for g in common})
    print("\n阶段       基线 A   候选 B     Δ(B−A)")
    for k in [f"{lo}-{hi}" for lo, hi in STAGES] + ["all", "end_pred"]:
        print(f"{k:10s} {ma[k]:7.2f} {mb[k]:8.2f} {mb[k]-ma[k]:+8.2f}")

    print("\n逐组总体 MAE（Δ>0 表示候选更差）")
    better = worse = 0
    for g in common:
        ea = float(np.mean(np.abs(A[g][1] - A[g][0])))
        eb = float(np.mean(np.abs(B[g][1] - B[g][0])))
        if eb < ea - 0.05:
            better += 1
        elif eb > ea + 0.05:
            worse += 1
        print(f"  {g}  A {ea:6.2f}   B {eb:6.2f}   {eb-ea:+6.2f}")
    print(f"\n候选更好 {better} 组 / 更差 {worse} 组")


if __name__ == "__main__":
    main()
