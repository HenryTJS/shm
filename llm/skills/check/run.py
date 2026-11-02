# -*- coding: utf-8 -*-
"""check: report raw-data coverage only; no result tables, no t85 read."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # D:\shm

from llm.adapters import RawL1Adapter, RawL2Adapter, RawPHM2019Adapter
from llm.skills.common import emit, envelope, evidence

PZT_COVERAGE = {"L2-04": 0.0, "L2-05": 20.2, "L2-07": 100.0}


def _mods(raw: dict) -> list[dict]:
    out = []
    for m in raw["modalities"]:
        out.append({"name": m["name"], "available": bool(m["available"]),
                    "n": m.get("files"), "t85_pct": None,
                    "note": "raw files only; t85 needs metric skill"})
    return out


def _evidence(gid: str, raw: dict, dataset: str) -> list[dict]:
    path = raw.get("root", "")
    return [evidence("E-CHECK-RAW", path, f"{dataset} {gid} raw",
                     {"gid": gid, "raw_files": raw.get("raw_files"),
                      "modalities": [m["name"] for m in raw["modalities"] if m["available"]]},
                     {"gid": gid, "campaign": None, "modality": "MULTI",
                      "cycle_min": None, "cycle_max": None,
                      "lifetime_min": None, "lifetime_max": None})]


def check_l1(gid: str):
    raw = RawL1Adapter().describe(gid)
    mods = _mods(raw)
    warns = ["t85 is not computed here; metric skill is not implemented yet"]
    data = {"dataset": "L1", "gid": gid, "exists": True,
            "campaign": None, "fatigue_end": None, "modalities": mods,
            "capabilities": [
                {"name": "raw_coverage", "available": any(m["available"] for m in mods),
                 "reason": "raw directories"},
                {"name": "t85", "available": False,
                 "reason": "metric skill not implemented; no result-table access"},
            ]}
    return data, mods, _evidence(gid, raw, "L1"), warns


def check_l2(gid: str):
    raw = RawL2Adapter().describe(gid)
    mods = _mods(raw)
    cov = PZT_COVERAGE.get(gid, 0.0)
    for m in mods:
        if m["name"] == "PZT":
            m["available"] = bool(m["available"] and cov >= 100)
            m["note"] = (f"PZT covers ~{cov:.1f}% of fatigue life"
                         if cov < 100 else "raw PZT complete; t85 needs metric")
    warns = []
    if cov < 100:
        warns.append(f"{gid}: PZT coverage {cov:.1f}%; full-life t85 not defined")
    data = {"dataset": "L2", "gid": gid, "exists": True,
            "campaign": None, "fatigue_end": None, "modalities": mods,
            "pzt_coverage_pct": cov,
            "capabilities": [
                {"name": "raw_coverage", "available": any(m["available"] for m in mods),
                 "reason": "raw directories"},
                {"name": "t85", "available": False,
                 "reason": "metric skill not implemented; no result-table access"},
                {"name": "PZT_t85", "available": False,
                 "reason": "raw metric not implemented"},
            ]}
    return data, mods, _evidence(gid, raw, "L2"), warns


def check_phm(gid: str):
    raw = RawPHM2019Adapter().describe(gid)
    mods = _mods(raw)
    data = {"dataset": "PHM2019", "gid": gid, "exists": True,
            "campaign": "PHM2019", "fatigue_end": raw.get("cycle_max"),
            "modalities": mods,
            "capabilities": [
                {"name": "crack_mm", "available": False,
                 "reason": "raw labels available; metric skill not implemented"},
                {"name": "t85", "available": False, "reason": "PHM2019 has no t85"},
            ]}
    return data, mods, _evidence(gid, {"root": "phmdc/labels.csv", "raw_files": raw["n_measurements"],
                                        "modalities": [{"name": "PZT", "available": True}]}, "PHM2019"), [
        f"{gid}: PHM2019 has no t85"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["L1", "L2", "PHM2019"])
    ap.add_argument("--gid", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.dataset == "L1":
            data, mods, ev, warns = check_l1(args.gid)
        elif args.dataset == "L2":
            data, mods, ev, warns = check_l2(args.gid)
        else:
            data, mods, ev, warns = check_phm(args.gid)
    except KeyError:
        obj = envelope("check", "not_found", {"dataset": args.dataset, "gid": args.gid},
                       computation={"kind": "read", "inputs": ["raw adapter"]},
                       report=f"# check\n未找到 {args.dataset} / {args.gid}。")
        emit(obj, args.json)
        return 1

    status = "ok" if any(m["available"] for m in mods) else "insufficient"
    lines = [f"# check: {args.dataset} / {args.gid}", "",
             "| 模态 | 原始文件 | t85 | 说明 |", "|---|---:|---:|---|"]
    for m in mods:
        lines.append(f"| {m['name']} | {'有' if m['available'] else '无'} | — | {m['note']} |")
    if warns:
        lines += ["", "## 警告", *[f"- {w}" for w in warns]]
    obj = envelope("check", status, data, warnings=warns, evidence=ev,
                   computation={"kind": "read", "inputs": ["RawL1/L2/PHM adapter"]},
                   report="\n".join(lines))
    emit(obj, args.json)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
