# -*- coding: utf-8 -*-
"""Freeze code/prompt/task hashes before a held-out run.

Usage:
    python llm/eval/freeze_run.py --tag phm_heldout --tasks llm/eval/tasks_phm_heldout.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

FILES = [
    "llm/agent.py",
    "llm/tools.py",
    "llm/rag.py",
    "llm/adapters/base.py",
    "llm/adapters/l1.py",
    "llm/adapters/phm2019.py",
    "llm/adapters/registry.py",
    "llm/eval/run_eval.py",
]


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.run(["git"] + args, cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=20).stdout.strip()
    except Exception as exc:
        return f"<git unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    manifest = {
        "tag": args.tag,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "files": {},
        "tasks": args.tasks,
        "tasks_sha256": None,
    }
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            manifest["files"][rel] = sha(path)
    if args.tasks:
        path = os.path.join(ROOT, args.tasks)
        if os.path.exists(path):
            manifest["tasks_sha256"] = sha(path)

    out = os.path.join(HERE, f"freeze_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
