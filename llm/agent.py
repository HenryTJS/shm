# -*- coding: utf-8 -*-
"""Evidence-gated SHM agent (L3 prototype).

Flow:
  question
    -> DeepSeek planner (tool calls + allowed lifetime horizon)
    -> ToolBox (numeric evidence)
    -> RAG (literature evidence)
    -> DeepSeek writer (claims only, using supplied evidence IDs)
    -> validate_graph (reject unsupported/leaky claims)
    -> deterministic answer rendered from the verified claims

Usage:
    python llm/agent.py "L1-03 的 t85 是多少？比 C1 中位低多少？"
    python llm/agent.py "..." --json
    python llm/agent.py "..." --out llm/out/example.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm.rag import retrieve, to_evidence
from llm.adapters import L1Adapter, PHM2019Adapter
from llm.tools import ToolBox
from llm.validate import validate_graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(HERE, "schema", "claim_evidence.schema.json")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OUT_DIR = os.path.join(HERE, "out")


def _json_from_text(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


class DeepSeek:
    def __init__(self, model: str = DEFAULT_MODEL, base: str = DEFAULT_BASE) -> None:
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise SystemExit("DEEPSEEK_API_KEY is not set")
        self.model = model
        self.base = base.rstrip("/")

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            self.base + "/chat/completions",
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if r.status_code == 400 and json_mode:
            # Some model aliases do not accept response_format; retry once.
            payload.pop("response_format", None)
            r = requests.post(
                self.base + "/chat/completions",
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek {r.status_code}: {r.text[:500]}")
        return r.json()["choices"][0]["message"]["content"]


PLANNER_SYS = """You are the planning module of a structural health monitoring (SHM) assistant.
You may call only these tools:
- lookup(gid, metric): one specimen's metric. metric is t50 or t85.
- campaign_stats(campaign, metric): campaign median/min/max. campaign is C1..C4.
- compare_to_campaign(gid, metric): specimen value minus its campaign median.
- compare_campaigns(campaign_a, campaign_b, metric): median of B minus median of A.
- modality_lookup(gid, modality, metric): one modality t50/t85; modality is AE/FBG/DFOS/PZT.
- modality_compare(gid, modality_a, modality_b, metric): B minus A, in percentage points.
- l2_lookup(gid, metric): Level-2 AE t85/t50; gid is L2-04/L2-05/L2-07.
- l2_pzt_lookup(gid): Level-2 PZT t85, only L2-07.
- Level-2 gids: L2-04, L2-05, L2-07. Use lookup for L1 AE; use modality_lookup
  for L1 FBG/DFOS/PZT; use l2_lookup for Level-2 AE.
- dataset_overview(): counts and available specimens.
- phm_lookup(gid, metric, cycle?): PHM2019 aluminium lap-joint; gid T1..T8;
  metric is crack_mm_final, crack_mm_at_cycle (needs cycle), n_measurements.
- phm_crack_delta(gid, cycle_a, cycle_b): crack(B) - crack(A) in mm.
- phm_overview(): PHM2019 specimen count.
- If gid starts with T1..T8, use phm_* tools, not lookup.

Return JSON only:
{
  "tool_calls": [{"tool": "lookup", "arguments": {"gid": "L1-03", "metric": "t85"}}],
  "allowed_lifetime_max": 1.0,
  "retrieval_query": "short English technical query for literature search",
  "reason": "short"
}

Rules:
- Never calculate numbers yourself; call tools.
- For a definition/method question, tool_calls may be empty, but retrieval_query
  should include the technical term and its context (e.g. "t85 self-normalised
  cumulative hit curve lifetime fraction definition").
- If the question asks about what is available at an early lifetime (e.g. 40 percent),
  set allowed_lifetime_max to that fraction; otherwise 1.0 for a completed record.
- Do not invent specimen IDs or campaigns.
"""

WRITER_SYS = """You are the claim writer of an SHM evidence graph.
You may use ONLY the evidence IDs supplied by the system. Never invent an ID,
a number, a DOI, or a quote.

Return JSON only:
{
  "claims": [
    {
      "text": "natural-language claim",
      "claim_type": "numeric|comparative|categorical|method|recommendation|limitation",
      "subject": "L1-03",
      "predicate": "t85_pct",
      "value": 86.4,
      "unit": "%",
      "comparator": "eq|lt|le|gt|ge|approx|in|null",
      "tolerance": 0.05,
      "evidence_ids": ["E001"],
      "support_quote": "exact phrase copied from the cited quote, or null",
      "missing_evidence": null,
      "confidence": 0.95
    }
  ]
}

Rules:
- For numeric/comparative claims, predicate MUST equal the `metric` field of a
  tool_result evidence, and value MUST equal its `value` within tolerance.
- A numeric/comparative claim without a matching tool_result is not allowed.
- For method/limitation/recommendation claims, cite literature evidence and
  copy an exact phrase from its quote into support_quote. A bare citation is rejected.
- If the evidence is insufficient, use evidence_ids=[] and put the missing
  item in missing_evidence; status will be set to insufficient by the system.
- Use the minimum number of claims needed to answer. If the question asks for a
  difference/comparison, do NOT separately restate each input value; make one
  claim whose predicate is the delta metric returned by the tool.
- Do not write an answer outside the JSON.
"""


def _tool_summary(results: list[dict]) -> str:
    rows = []
    for r in results:
        ev = r["evidence"]
        rows.append({
            "evidence_id": ev["evidence_id"],
            "tool_call_id": r["tool_call"]["tool_call_id"],
            "tool": r["tool_call"]["tool_name"],
            "metric": ev["content"].get("metric"),
            "value": ev["content"].get("value"),
            "content": ev["content"],
            "scope": ev["scope"],
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _rag_summary(evs: list[dict]) -> str:
    rows = []
    for ev in evs:
        rows.append({
            "evidence_id": ev["evidence_id"],
            "title": ev["source"]["title"],
            "locator": ev["locator"],
            "score": ev["content"].get("score"),
            "quote": ev["quote"][:900],
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def plan(question: str, llm: DeepSeek) -> dict:
    prompt = f"""Question:
{question}

Return the plan JSON. Use at most two tool calls."""
    txt = llm.chat([
        {"role": "system", "content": PLANNER_SYS},
        {"role": "user", "content": prompt},
    ], json_mode=True)
    return _json_from_text(txt)


def execute_plan(plan_obj: dict, tb: ToolBox, phm=None) -> tuple[list[dict], list[str]]:
    results, errors = [], []
    allowed = {"lookup", "campaign_stats", "compare_to_campaign", "compare_campaigns",
               "modality_lookup", "modality_compare", "l2_lookup", "l2_pzt_lookup",
               "dataset_overview", "phm_lookup", "phm_crack_delta", "phm_overview"}
    for call in plan_obj.get("tool_calls", [])[:2]:
        name = call.get("tool")
        args = call.get("arguments") or {}
        if name not in allowed:
            errors.append(f"unknown tool {name}")
            continue
        try:
            if name == "lookup":
                r = tb.lookup(args["gid"], args.get("metric", "t85"))
            elif name == "campaign_stats":
                r = tb.campaign_stats(args["campaign"], args.get("metric", "t85"))
            elif name == "compare_to_campaign":
                r = tb.compare_to_campaign(args["gid"], args.get("metric", "t85"))
            elif name == "compare_campaigns":
                r = tb.compare_campaigns(args["campaign_a"], args["campaign_b"], args.get("metric", "t85"))
            elif name == "modality_lookup":
                r = tb.modality_lookup(args["gid"], args["modality"], args.get("metric", "t85"))
            elif name == "modality_compare":
                r = tb.modality_compare(args["gid"], args["modality_a"], args["modality_b"], args.get("metric", "t85"))
            elif name == "l2_lookup":
                r = tb.l2_lookup(args["gid"], args.get("metric", "t85_all"))
            elif name == "l2_pzt_lookup":
                r = tb.l2_pzt_lookup(args["gid"])
            elif name == "phm_lookup":
                r = phm.lookup(args["gid"], args.get("metric", "crack_mm_final"),
                               cycle=args.get("cycle"))
            elif name == "phm_crack_delta":
                r = phm.compare(args["gid"], cycle_a=args["cycle_a"],
                                cycle_b=args["cycle_b"])
            elif name == "phm_overview":
                r = phm.overview()
            else:
                r = tb.dataset_overview()
            results.append(r)
        except Exception as exc:
            errors.append(f"{name}{args}: {exc}")
    return results, errors


def _scope_default() -> dict:
    return {
        "gid": "ALL", "campaign": None, "modality": "MULTI",
        "cycle_min": None, "cycle_max": None,
        "lifetime_min": None, "lifetime_max": None,
    }


def build_graph(question: str, plan_obj: dict, tool_results: list[dict],
                rag_evidence: list[dict], writer_obj: dict,
                error_notes: list[str] | None = None) -> dict:
    evidence = [r["evidence"] for r in tool_results] + rag_evidence
    e_by_id = {e["evidence_id"]: e for e in evidence}
    t_by_e = {r["evidence"]["evidence_id"]: r["tool_call"]["tool_call_id"]
              for r in tool_results}
    allowed = plan_obj.get("allowed_lifetime_max", 1.0)
    claims = []
    for i, seed in enumerate(writer_obj.get("claims", [])[:8], start=1):
        eids = [e for e in (seed.get("evidence_ids") or []) if e in e_by_id]
        first = e_by_id[eids[0]] if eids else None
        value = seed.get("value")
        is_numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        claim = {
            "claim_id": f"C{i:03d}",
            "text": str(seed.get("text") or "").strip() or "(empty claim)",
            "claim_type": seed.get("claim_type") or ("numeric" if is_numeric else "categorical"),
            "subject": seed.get("subject"),
            "predicate": seed.get("predicate"),
            "value": value,
            "unit": seed.get("unit"),
            "comparator": seed.get("comparator"),
            "tolerance": seed.get("tolerance", 0.1 if is_numeric else None),
            "status": "supported" if eids else "insufficient",
            "evidence_ids": eids,
            "tool_call_ids": [t_by_e[e] for e in eids if e in t_by_e],
            "scope": first["scope"] if first else _scope_default(),
            "allowed_lifetime_max": allowed,
            "support_quote": seed.get("support_quote"),
            "missing_evidence": seed.get("missing_evidence") or (None if eids else "no supporting evidence"),
            "confidence": seed.get("confidence"),
            "created_by": "Writer",
        }
        if not eids:
            claim["value"] = None
        claims.append(claim)

    if error_notes:
        claims.append({
            "claim_id": f"C{len(claims) + 1:03d}",
            "text": "The planner could not execute one or more tool calls.",
            "claim_type": "limitation",
            "subject": None, "predicate": None, "value": "tool_error",
            "unit": None, "comparator": None, "tolerance": None,
            "status": "supported" if "evidence" else "insufficient",
            "evidence_ids": [], "tool_call_ids": [],
            "scope": _scope_default(), "allowed_lifetime_max": None,
            "missing_evidence": "; ".join(error_notes),
            "confidence": 0.0, "created_by": "Critic",
        })

    edges = []
    for c in claims:
        for eid in c["evidence_ids"]:
            edges.append({"claim_id": c["claim_id"], "evidence_id": eid,
                          "relation": "supports"})

    return {
        "schema_version": "1.0",
        "graph_id": "G" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "question": question,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {
            "require_evidence": True,
            "allow_unsupported_claims": False,
            "max_evidence_per_claim": 8,
            "temporal_leakage_check": True,
            "require_numeric_tool_evidence": True,
        },
        "claims": claims,
        "evidence": evidence,
        "edges": edges,
        "tool_calls": [r["tool_call"] for r in tool_results],
        "verification": {"status": "needs_review", "summary": "", "checks": []},
    }


CRITIC_SYS = """You are a strict evidence critic for a structural health monitoring
assistant. For each non-numeric claim, decide whether the cited quote actually
entails the claim as written. Paraphrase is allowed. Reject if the claim adds a
fact, number, causal relation, or scope not present in the quote. Return JSON
only:
{"verdicts":[{"claim_id":"C001","entailed":true,"reason":"..."}]}
"""


def critique_non_numeric(graph: dict, llm: DeepSeek) -> dict:
    evidence = {e["evidence_id"]: e for e in graph["evidence"]}
    items = []
    for c in graph["claims"]:
        if c["status"] != "supported":
            continue
        if c["claim_type"] in ("numeric", "comparative"):
            continue
        quote = ""
        for eid in c["evidence_ids"]:
            e = evidence.get(eid)
            if e and e["evidence_type"] in ("literature", "standard"):
                quote = e.get("quote") or ""
                break
        if quote:
            items.append({
                "claim_id": c["claim_id"],
                "claim": c["text"],
                "support_quote": c.get("support_quote"),
                "quote": quote[:1200],
            })
    if not items:
        return graph
    try:
        obj = _json_from_text(llm.chat([
            {"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False, indent=2)},
        ], json_mode=True))
    except Exception as exc:
        print(f"critic failed: {exc}", file=sys.stderr)
        for c in graph["claims"]:
            if c["claim_id"] in {x["claim_id"] for x in items}:
                c["status"] = "rejected"
                c["missing_evidence"] = "critic unavailable"
        return graph

    verdicts = {v["claim_id"]: v for v in obj.get("verdicts", [])}
    for c in graph["claims"]:
        v = verdicts.get(c["claim_id"])
        if v is not None and not v.get("entailed", False):
            c["status"] = "rejected"
            c["missing_evidence"] = "critic: " + str(v.get("reason", "quote does not entail claim"))
    return graph


def render_answer(graph: dict, verification_status: str) -> str:
    if verification_status != "pass":
        return "证据不足或校验未通过，不能给出确定结论。"
    lines = []
    for c in graph["claims"]:
        if c["status"] == "supported":
            ev = ", ".join(c["evidence_ids"])
            lines.append(f"- {c['text']} [{ev}]")
        elif c["status"] == "insufficient":
            lines.append(f"- 证据不足：{c['missing_evidence']}")
    return "\n".join(lines) if lines else "没有生成可核验的结论。"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--repair", type=int, default=1)
    args = ap.parse_args(argv)

    llm = DeepSeek()
    tb = L1Adapter()
    try:
        phm = PHM2019Adapter()
    except Exception:
        phm = None

    # 1. planner first, so it can produce a technical retrieval query
    try:
        p = plan(args.question, llm)
    except Exception as exc:
        print(f"planner failed: {exc}", file=sys.stderr)
        p = {"tool_calls": [], "allowed_lifetime_max": 1.0, "reason": "planner fallback"}

    # 2. retrieve literature evidence (IDs E101+)
    rag_evs = []
    retrieval_query = p.get("retrieval_query") or args.question
    if not args.no_rag:
        for i, ch in enumerate(retrieve(retrieval_query, args.k), start=101):
            ch = dict(ch)
            ch["chunk_id"] = f"R{i - 100:03d}"
            rag_evs.append(to_evidence(ch, f"E{i:03d}"))

    tool_results, tool_errors = execute_plan(p, tb, phm)

    # 3. writer
    user = f"""Question:
{args.question}

Plan:
{json.dumps(p, ensure_ascii=False, indent=2)}

Tool evidence:
{_tool_summary(tool_results) if tool_results else "none"}

Literature evidence:
{_rag_summary(rag_evs) if rag_evs else "none"}

Write the claims JSON. allowed_lifetime_max is {p.get("allowed_lifetime_max", 1.0)}."""
    try:
        w = _json_from_text(llm.chat([
            {"role": "system", "content": WRITER_SYS},
            {"role": "user", "content": user},
        ], json_mode=True))
    except Exception as exc:
        print(f"writer failed: {exc}", file=sys.stderr)
        w = {"claims": []}

    graph = build_graph(args.question, p, tool_results, rag_evs, w, tool_errors)
    graph = critique_non_numeric(graph, llm)

    # 4. validator
    with open(SCHEMA, encoding="utf-8") as fh:
        schema = json.load(fh)
    rep = validate_graph(graph, schema)

    # 4. one evidence-aware repair: quote the exact failing phrase, or drop
    # the claim. This is still gated by the same validator and critic.
    if rep.failed and args.repair > 0:
        errs = [f"{x['name']}: {x['detail']} ({x.get('claim_id', '')})"
                for x in rep.failed]
        repair_user = f"""Question:
{args.question}

Your previous claims:
{json.dumps(w.get("claims", []), ensure_ascii=False, indent=2)}

Validation failures:
{json.dumps(errs, ensure_ascii=False, indent=2)}

Tool evidence:
{_tool_summary(tool_results) if tool_results else "none"}

Literature evidence:
{_rag_summary(rag_evs) if rag_evs else "none"}

Fix the claims. For support_quote, copy an EXACT contiguous phrase from the
cited quote, not a paraphrase. If no exact phrase supports the claim, drop the
claim or mark it as insufficient. Return the full claims JSON only."""
        try:
            w2 = _json_from_text(llm.chat([
                {"role": "system", "content": WRITER_SYS},
                {"role": "user", "content": repair_user},
            ], json_mode=True))
            if w2.get("claims"):
                w = w2
                graph = build_graph(args.question, p, tool_results, rag_evs, w, tool_errors)
                graph = critique_non_numeric(graph, llm)
                rep = validate_graph(graph, schema)
        except Exception as exc:
            print(f"repair failed: {exc}", file=sys.stderr)

    failed_ids = {c["claim_id"] for c in rep.failed if c.get("claim_id")}
    pruned = 0
    for claim in graph["claims"]:
        if claim["claim_id"] in failed_ids:
            claim["status"] = "rejected"
            claim["missing_evidence"] = "validation failed for this claim"

    # Drop rejected side-claims when at least one claim survives. This keeps a
    # correct answer (e.g. a definition) from being lost because the Writer
    # added an extra unsupported sentence. If all claims are rejected, the
    # graph still fails and the answer is a refusal.
    kept = [c for c in graph["claims"] if c["status"] != "rejected"]
    rejected = [c for c in graph["claims"] if c["status"] == "rejected"]
    numeric_rejected = any(c["claim_type"] in ("numeric", "comparative")
                           for c in rejected)
    if failed_ids and kept and not numeric_rejected:
        kept_ids = {c["claim_id"] for c in kept}
        graph["claims"] = kept
        graph["edges"] = [e for e in graph["edges"] if e["claim_id"] in kept_ids]
        pruned = len(failed_ids)
        rep = validate_graph(graph, schema)

    allowed = float(p.get("allowed_lifetime_max", 1.0) or 1.0)
    numeric_supported = any(
        c.get("claim_type") in ("numeric", "comparative")
        and c.get("status") == "supported"
        for c in graph["claims"]
    )
    if allowed < 1.0 and not numeric_supported:
        rep.add("temporal_scope", "fail",
                f"no supported numeric claim within allowed lifetime {allowed}")

    graph["verification"] = {
        "status": "pass" if not rep.failed else "fail",
        "summary": f"{len(rep.failed)} failed / {len(rep.checks)} checks"
                   + (f"; {pruned} rejected side-claim(s) pruned" if pruned else ""),
        "checks": rep.checks,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or OUT_DIR, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, ensure_ascii=False, indent=2)
        print("wrote", args.out)

    if args.json:
        print(json.dumps(graph, ensure_ascii=False, indent=2))
    else:
        print("PLAN:", json.dumps(p, ensure_ascii=False))
        print("VERIFICATION:", graph["verification"]["status"],
              f"({graph['verification']['summary']})")
        print("ANSWER:")
        print(render_answer(graph, graph["verification"]["status"]))

    return 0 if graph["verification"]["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
