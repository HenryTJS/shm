# -*- coding: utf-8 -*-
"""Run the SHM agent on the task set and score simple, auditable criteria.

This is deliberately not an LLM judge. It checks:
  - numeric: a supported claim with the expected predicate/value;
  - method: verification passes and a supported claim has a literature quote;
  - refusal: the graph is rejected, or no supported claim is emitted;
  - citation coverage: every supported claim has evidence.

Usage:
    python llm/eval/run_eval.py --limit 3
    python llm/eval/run_eval.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AGENT = os.path.join(ROOT, "llm", "agent.py")
TASKS = os.path.join(HERE, "tasks.jsonl")
OUT = os.path.join(HERE, "out")
RESULTS = os.path.join(HERE, "results.csv")


def load_tasks(path: str = TASKS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_one(task: dict) -> dict:
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, task["task_id"] + ".json")
    cmd = [sys.executable, AGENT, task["question"], "--json", "--k", "3",
           "--out", out_path]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    elapsed = time.time() - t0
    if not os.path.exists(out_path):
        return {"task_id": task["task_id"], "status": "error",
                "detail": (proc.stderr or proc.stdout)[-300:], "seconds": elapsed}
    with open(out_path, encoding="utf-8") as fh:
        graph = json.load(fh)

    claims = graph.get("claims", [])
    supported = [c for c in claims if c.get("status") == "supported"]
    expectations = task.get("expect", {})
    kind = expectations.get("type")

    if task.get("should_refuse"):
        text = " ".join(c.get("text", "") for c in supported).lower()
        markers = ("cannot", "cannot be determined", "no pzt", "not available",
                   "无法", "不足", "没有", "不能")
        ok = (graph["verification"]["status"] != "pass"
              or not supported
              or any(m in text for m in markers))
        detail = "refused" if ok else "answered despite insufficient evidence"
    elif kind == "numeric":
        pred = expectations["predicate"]
        target = float(expectations["value"])
        tol = float(expectations.get("tol", 0.1))
        hit = [c for c in supported
               if c.get("predicate") == pred
               and isinstance(c.get("value"), (int, float))
               and abs(float(c["value"]) - target) <= tol]
        ok = bool(hit) and graph["verification"]["status"] == "pass"
        detail = f"{len(hit)} matching numeric claim(s)"
    elif kind == "nonnumeric":
        ok = (graph["verification"]["status"] == "pass"
              and any(c.get("support_quote") for c in supported))
        detail = f"{len(supported)} supported claim(s)"
    else:
        ok, detail = False, "unknown expectation"

    coverage = (all(c.get("evidence_ids") for c in supported)
                if supported else False)
    return {"task_id": task["task_id"], "status": "pass" if ok else "fail",
            "detail": detail, "seconds": round(elapsed, 2),
            "verification": graph["verification"]["status"],
            "n_claims": len(claims), "n_supported": len(supported),
            "citation_coverage": coverage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    rows = []
    for task in tasks:
        print(f"[run] {task['task_id']} {task['question']}", flush=True)
        row = run_one(task)
        print(f"      {row['status']} ({row['detail']}, {row.get('seconds')}s)", flush=True)
        rows.append(row)

    os.makedirs(HERE, exist_ok=True)
    with open(args.results, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    print(f"\n{n_pass}/{len(rows)} pass -> {args.results}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
