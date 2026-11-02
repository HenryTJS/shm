# -*- coding: utf-8 -*-
"""compare: contract placeholder until the raw metric skill exists.

The old implementation read precomputed t85 from l1/results via L1Adapter. That
violates the skill data policy, so it is removed rather than silently kept.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # D:\shm

from llm.skills.common import emit, envelope


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="L1", choices=["L1"])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", default="t85", choices=["t85", "t50"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    obj = envelope(
        "compare", "unsupported",
        {"a": args.a, "b": args.b, "metric": args.metric,
         "reason": "raw metric not implemented; no result-table fallback"},
        computation={"kind": "derived", "formula": None,
                     "inputs": ["metric skill (pending)"]},
        report=("# compare\n"
                "当前没有可用的 raw metric，不能拿 results 里的 t85 代替。\n"
                "等 `metric` 从原始曲线算完 t85 后，再启用 compare。"))
    emit(obj, args.json)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
