# -*- coding: utf-8 -*-
"""Cheap KB-only and RAG-only baselines for the starter task set.

This is a smoke test, not a full LLM judge:
  - KB: return the top retrieved chunk, no model.
  - RAG: one DeepSeek call with the same chunks, no tools.
  - Agent: the evidence-gated tool agent.

Numeric tasks are scored by whether the expected value appears in the answer;
method tasks by keywords; refusal tasks by whether the system refuses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm.agent import DeepSeek
from llm.rag import retrieve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TASKS = os.path.join(HERE, "tasks.jsonl")
RESULTS = os.path.join(HERE, "baselines.csv")


def load_tasks(path: str = TASKS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def snippets(question: str, k: int = 3) -> str:
    hits = retrieve(question, k)
    return "\n\n".join(
        f"[{i}] {h['title']} p.{h['page']}\n{h['text'][:700]}"
        for i, h in enumerate(hits, 1)
    )


def ask_rag(question: str, context: str, llm: DeepSeek) -> str:
    prompt = f"""Answer the question using only the excerpts below.
If the excerpts do not contain enough evidence, say "证据不足" and explain what
is missing. Do not invent numbers. Keep the answer under 120 words.

Question: {question}

Excerpts:
{context}
"""
    return llm.chat([{"role": "user", "content": prompt}], json_mode=False)


def score(task: dict, answer: str) -> bool:
    exp = task.get("expect", {})
    kind = exp.get("type")
    low = answer.lower()
    if task.get("should_refuse"):
        return any(x in low for x in ("证据不足", "insufficient", "cannot", "无法", "not enough"))
    if kind == "numeric":
        val = str(exp["value"])
        return val in answer or val.replace("-", "−") in answer
    if kind == "nonnumeric":
        return any(k.lower() in low for k in exp.get("keywords", []))
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]

    llm = DeepSeek()
    rows = []
    for t in tasks:
        ctx = snippets(t["question"])
        kb_answer = ctx
        rag_answer = ask_rag(t["question"], ctx, llm)
        rows.append({
            "task_id": t["task_id"],
            "category": t["category"],
            "kb_pass": score(t, kb_answer),
            "rag_pass": score(t, rag_answer),
            "rag_answer": rag_answer.replace("\n", " ")[:300],
        })
        print(f"{t['task_id']}: KB={rows[-1]['kb_pass']} RAG={rows[-1]['rag_pass']}",
              flush=True)

    with open(args.results, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kb = sum(r["kb_pass"] for r in rows)
    rag = sum(r["rag_pass"] for r in rows)
    print(f"\nKB {kb}/{len(rows)}  RAG {rag}/{len(rows)} -> {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
