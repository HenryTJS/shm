# -*- coding: utf-8 -*-
"""audit: validate a structured claim–evidence graph; no text extraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # D:\shm

from llm.validate import validate_graph
from llm.skills.common import emit, envelope

ROOT = HERE.parents[2]
SCHEMA = ROOT / "llm" / "schema" / "claim_evidence.schema.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    path = Path(args.graph)
    if not path.is_absolute():
        path = ROOT / path

    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    except Exception as exc:
        obj = envelope("audit", "error", {"graph": str(path), "reason": str(exc)},
                       report=f"# audit\n无法读取或解析：{exc}")
        emit(obj, args.json)
        return 2

    rep = validate_graph(graph, schema)
    failed = rep.failed
    data = {
        "graph": str(path),
        "n_claims": len(graph.get("claims", [])),
        "n_evidence": len(graph.get("evidence", [])),
        "n_failed": len(failed),
        "failed_checks": failed,
    }
    lines = [f"# audit: {path.name}", "",
             f"- claims: {data['n_claims']}",
             f"- evidence: {data['n_evidence']}",
             f"- failed checks: {data['n_failed']}", ""]
    if failed:
        lines += ["## 失败项", *[f"- {x.get('name')} {x.get('claim_id', '')}: {x.get('detail')}"
                                 for x in failed]]
    else:
        lines.append("全部通过。")

    status = "ok" if not failed else "insufficient"
    obj = envelope("audit", status, data, report="\n".join(lines),
                   computation={"kind": "validated", "formula": None,
                                "inputs": ["llm/validate.py"]})
    emit(obj, args.json)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
