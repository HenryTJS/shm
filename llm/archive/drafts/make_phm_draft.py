# -*- coding: utf-8 -*-
"""Generate a mechanical PHM2019 draft task set (for human rewriting).

Rules (fixed, no cherry-picking):
  - every specimen T1..T8: final crack
  - every specimen: number of measurements
  - every specimen: first-to-last crack growth
  - every specimen: crack at its second measurement cycle
  - one dataset count
  - four refusals

The output is a DRAFT. Rewrite the question wording, keep the answer key, then
freeze and run once. This file does not call the Agent or the API.
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LABELS = os.path.join(ROOT, "phmdc", "labels.csv")
OUT = os.path.join(HERE, "tasks_phm_draft.jsonl")
REVIEW = os.path.join(HERE, "phm_draft_review.csv")


def main() -> int:
    with open(LABELS, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["cycle"] = int(float(r["cycle"]))
        r["crack_mm"] = float(r["crack_mm"])
    by = {}
    for r in rows:
        by.setdefault(r["specimen"], []).append(r)
    for k in by:
        by[k].sort(key=lambda x: x["cycle"])

    tasks = []
    review = []

    def add(tid, question, predicate, value, tol, source, category="external"):
        tasks.append({
            "task_id": tid,
            "category": category,
            "question": question,
            "expect": {"type": "numeric", "predicate": predicate,
                       "value": round(value, 2), "tol": tol},
            "should_refuse": False,
            "source": source,
            "draft_note": "mechanical draft; rewrite the question, keep the key",
        })
        review.append({"task_id": tid, "category": category,
                       "question": question, "predicate": predicate,
                       "value": round(value, 2), "tol": tol, "source": json.dumps(source)})

    n = 0
    # 1) final crack, all 8
    for gid in sorted(by):
        n += 1
        r = by[gid][-1]
        add(f"D{n:02d}", f"{gid} 的最终裂纹长度是多少？",
            "crack_mm_final", r["crack_mm"], 0.01,
            {"specimen": gid, "cycle": r["cycle"], "crack_mm": r["crack_mm"]})

    # 2) measurement count, all 8
    for gid in sorted(by):
        n += 1
        add(f"D{n:02d}", f"{gid} 有多少次裂纹测量？",
            "n_measurements", len(by[gid]), 0,
            {"specimen": gid, "n_measurements": len(by[gid])})

    # 3) first-to-last growth, all 8
    for gid in sorted(by):
        n += 1
        a, b = by[gid][0], by[gid][-1]
        add(f"D{n:02d}",
            f"{gid} 从 {a['cycle']} 到 {b['cycle']} 次循环，裂纹增长了多少？",
            "crack_delta_mm", b["crack_mm"] - a["crack_mm"], 0.01,
            {"specimen": gid, "cycle_a": a["cycle"], "cycle_b": b["cycle"],
             "crack_a": a["crack_mm"], "crack_b": b["crack_mm"]})

    # 4) crack at the second measurement cycle, all 8
    for gid in sorted(by):
        n += 1
        if len(by[gid]) < 2:
            continue
        r = by[gid][1]
        add(f"D{n:02d}", f"{gid} 在 {r['cycle']} 次循环时的裂纹长度是多少？",
            "crack_mm_at_cycle", r["crack_mm"], 0.01,
            {"specimen": gid, "cycle": r["cycle"], "crack_mm": r["crack_mm"]})

    # 5) dataset count
    n += 1
    add(f"D{n:02d}", "PHM2019 一共有多少件试件？",
        "n_specimens", len(by), 0,
        {"n_specimens": len(by)})

    # 6) refusals (clearly unsupported)
    refusals = [
        ("T9 的最终裂纹长度是多少？", "unknown specimen T9"),
        ("T1 的 t85 是多少？", "PHM2019 has no t85 metric"),
        ("T1 的 PZT t85 是多少？", "PHM2019 has no PZT t85"),
        ("T9 有多少次裂纹测量？", "unknown specimen T9"),
    ]
    for q, why in refusals:
        n += 1
        tasks.append({
            "task_id": f"D{n:02d}",
            "category": "refusal",
            "question": q,
            "expect": {"type": "refusal"},
            "should_refuse": True,
            "source": {"reason": why},
            "draft_note": "mechanical draft; rewrite the question, keep the key",
        })
        review.append({"task_id": f"D{n:02d}", "category": "refusal",
                       "question": q, "predicate": "", "value": "",
                       "tol": "", "source": why})

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(REVIEW, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review[0]))
        w.writeheader()
        w.writerows(review)
    print(f"wrote {len(tasks)} tasks -> {OUT}")
    print(f"wrote review -> {REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
