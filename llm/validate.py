# -*- coding: utf-8 -*-
"""Validate a SHM claim–evidence graph.

Usage:
    python llm/validate.py llm/examples/L1-03_t85.json
    python llm/validate.py graph.json --json
    python llm/validate.py graph.json --out graph.verified.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMA = os.path.join(HERE, "schema", "claim_evidence.schema.json")

KIND_BY_TYPE = {
    "tool_result": "tool",
    "literature": "paper",
    "standard": "standard",
    "dataset_metadata": "dataset",
    "human_note": "human",
}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str,
            claim_id: str | None = None, evidence_id: str | None = None) -> None:
        item = {"name": name, "status": status, "detail": detail}
        if claim_id:
            item["claim_id"] = claim_id
        if evidence_id:
            item["evidence_id"] = evidence_id
        self.checks.append(item)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "fail"]


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _scope_ok(claim_scope: dict, ev_scope: dict) -> tuple[bool, str]:
    cg, eg = claim_scope.get("gid"), ev_scope.get("gid")
    if cg != "ALL" and eg != "ALL" and cg != eg:
        return False, f"gid {cg} != {eg}"
    cc, ec = claim_scope.get("campaign"), ev_scope.get("campaign")
    if cc and ec and cc != ec:
        return False, f"campaign {cc} != {ec}"
    cm, em = claim_scope.get("modality"), ev_scope.get("modality")
    if cm and em and cm != "MULTI" and em != "MULTI" and cm != em:
        return False, f"modality {cm} != {em}"
    return True, "scope compatible"


def _locator_ok(ev: dict) -> tuple[bool, str]:
    loc = ev.get("locator") or {}
    has_anchor = any(loc.get(k) not in (None, "") for k in
                     ("page", "section", "clause", "table", "figure", "row"))
    if not ev.get("quote"):
        return False, "missing quote"
    if not has_anchor:
        return False, "missing page/section/clause/table/figure/row locator"
    return True, "quote + locator present"


def _numeric_evidence(claim: dict, evidence: dict) -> tuple[bool, str]:
    content = evidence.get("content") or {}
    metric = content.get("metric")
    value = _num(content.get("value"))
    target = _num(claim.get("value"))
    if metric != claim.get("predicate"):
        return False, f"metric {metric!r} != predicate {claim.get('predicate')!r}"
    if value is None or target is None:
        return False, "non-numeric content.value or claim.value"
    tol = _num(claim.get("tolerance")) or 1e-6
    if abs(value - target) > tol:
        return False, f"|{value} - {target}| > tolerance {tol}"
    return True, f"{metric}={value} (tolerance {tol})"


def validate_graph(graph: dict, schema: dict) -> Report:
    rep = Report()

    # 1. structural schema
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(graph), key=lambda e: list(e.path))
    if errors:
        for e in errors[:20]:
            path = ".".join(str(x) for x in e.path) or "<root>"
            rep.add("schema", "fail", f"{path}: {e.message}")
        return rep
    rep.add("schema", "pass", "JSON Schema valid")

    claims = graph["claims"]
    evidence = graph["evidence"]
    edges = graph["edges"]
    tool_calls = graph.get("tool_calls", [])
    policy = graph.get("policy", {})

    # 2. unique ids
    for name, items, key in (
        ("claim_id", claims, "claim_id"),
        ("evidence_id", evidence, "evidence_id"),
        ("tool_call_id", tool_calls, "tool_call_id"),
    ):
        ids = [x.get(key) for x in items]
        dup = sorted({x for x in ids if ids.count(x) > 1})
        rep.add("unique_" + key, "fail" if dup else "pass",
                "duplicates: " + ", ".join(dup) if dup else "unique")

    c_by_id = {c["claim_id"]: c for c in claims}
    e_by_id = {e["evidence_id"]: e for e in evidence}
    t_ids = {t["tool_call_id"] for t in tool_calls}

    # 3. edge references
    bad_edges = [x for x in edges
                 if x["claim_id"] not in c_by_id or x["evidence_id"] not in e_by_id]
    rep.add("edge_references", "fail" if bad_edges else "pass",
            f"{len(bad_edges)} dangling edges" if bad_edges else "all edges resolve")

    # 4. per-claim semantic checks
    for c in claims:
        cid = c["claim_id"]
        eids = c.get("evidence_ids") or []

        if c["status"] == "rejected":
            rep.add("claim_status", "fail", "claim rejected by critic", cid)

        if c["status"] == "supported":
            if not eids:
                rep.add("claim_evidence", "fail", "supported claim has no evidence", cid)
            else:
                rep.add("claim_evidence", "pass", f"{len(eids)} evidence item(s)", cid)
        elif c["status"] == "insufficient":
            if not c.get("missing_evidence"):
                rep.add("insufficient_reason", "fail",
                        "insufficient claim needs missing_evidence", cid)
            else:
                rep.add("insufficient_reason", "pass", c["missing_evidence"], cid)

        if len(eids) > int(policy.get("max_evidence_per_claim", 8)):
            rep.add("evidence_budget", "fail", f"{len(eids)} > max", cid)

        supports = {x["evidence_id"] for x in edges
                    if x["claim_id"] == cid and x["relation"] == "supports"}
        for eid in eids:
            if eid not in supports:
                rep.add("claim_support_edge", "fail",
                        f"{eid} has no supports edge", cid, eid)

        for eid in eids:
            ev = e_by_id.get(eid)
            if not ev:
                continue
            expected_kind = KIND_BY_TYPE.get(ev["evidence_type"])
            actual_kind = ev["source"].get("kind")
            if expected_kind and actual_kind != expected_kind:
                rep.add("evidence_source_kind", "fail",
                        f"{ev['evidence_type']} expects {expected_kind}, got {actual_kind}",
                        cid, eid)

            ok, why = _scope_ok(c["scope"], ev["scope"])
            if not ok:
                rep.add("scope", "fail", why, cid, eid)

            if ev["evidence_type"] in ("literature", "standard"):
                ok, why = _locator_ok(ev)
                rep.add("citation_locator", "pass" if ok else "fail", why, cid, eid)

            # temporal leakage
            if policy.get("temporal_leakage_check", True) and c.get("allowed_lifetime_max") is not None:
                allowed = float(c["allowed_lifetime_max"])
                observed = ev.get("observed_lifetime_fraction")
                ev_max = (ev.get("scope") or {}).get("lifetime_max")
                if observed is not None and float(observed) > allowed + 1e-9:
                    rep.add("temporal_leakage", "fail",
                            f"observed {observed} > allowed {allowed}", cid, eid)
                if ev_max is not None and float(ev_max) > allowed + 1e-9:
                    rep.add("temporal_leakage", "fail",
                            f"evidence lifetime_max {ev_max} > allowed {allowed}", cid, eid)

        # non-numeric claims must quote a phrase that actually appears in a
        # literature/standard evidence item; a bare citation is not enough.
        if c["status"] == "supported" and c["claim_type"] in (
                "method", "limitation", "recommendation", "categorical"):
            sq = _norm(c.get("support_quote"))
            if not sq:
                rep.add("support_quote", "fail",
                        "non-numeric claim needs support_quote", cid)
            else:
                candidates = [e_by_id[e] for e in eids if e in e_by_id]
                ok = any(
                    ev["evidence_type"] in ("literature", "standard")
                    and sq in _norm(ev.get("quote"))
                    for ev in candidates
                )
                rep.add("support_quote", "pass" if ok else "fail",
                        "exact phrase found" if ok else "phrase not found in cited quote", cid)

        # numeric claims (including numeric comparative claims) require a
        # tool result with matching metric/value.
        if c["status"] == "supported" and _num(c.get("value")) is not None:
            matches = []
            for eid in eids:
                ev = e_by_id.get(eid)
                if ev and ev["evidence_type"] == "tool_result":
                    ok, why = _numeric_evidence(c, ev)
                    if ok:
                        matches.append((eid, why))
                    else:
                        rep.add("numeric_value", "fail", why, cid, eid)
            if not matches:
                rep.add("numeric_value", "fail",
                        "no matching tool_result evidence", cid)
            else:
                rep.add("numeric_value", "pass", matches[0][1], cid, matches[0][0])

        # tool call references
        for tid in c.get("tool_call_ids") or []:
            if tid not in t_ids:
                rep.add("tool_call_reference", "fail", f"{tid} not found", cid)

    # 5. verification block is computed, not trusted
    rep.add("verification_status",
            "fail" if rep.failed else "pass",
            f"{len(rep.failed)} failed check(s)")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", help="path to claim-evidence JSON")
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--json", action="store_true", help="print machine-readable report")
    ap.add_argument("--out", default=None, help="write graph with computed verification block")
    args = ap.parse_args(argv)

    with open(args.schema, encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(args.graph, encoding="utf-8") as fh:
        graph = json.load(fh)

    rep = validate_graph(graph, schema)
    status = "pass" if not rep.failed else "fail"

    if args.json:
        print(json.dumps({"status": status, "checks": rep.checks}, ensure_ascii=False, indent=2))
    else:
        for c in rep.checks:
            loc = ""
            if c.get("claim_id"):
                loc += " " + c["claim_id"]
            if c.get("evidence_id"):
                loc += " " + c["evidence_id"]
            print(f"[{c['status'].upper():4s}] {c['name']:<24s}{loc}: {c['detail']}")
        print(f"\nVERIFICATION: {status} ({len(rep.failed)} failed / {len(rep.checks)} checks)")

    if args.out:
        graph["verification"] = {
            "status": status,
            "summary": f"{len(rep.failed)} failed / {len(rep.checks)} checks",
            "checks": rep.checks,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, ensure_ascii=False, indent=2)
        print("wrote", args.out)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
