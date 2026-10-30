# -*- coding: utf-8 -*-
"""PHM2019 aluminium lap-joint adapter.

This is the first external dataset adapter. It does NOT train the LLM and does
NOT store answers in a prompt: it reads `labels.csv` at tool-call time and emits
the same claim–evidence `tool_result` object as the L1 adapter.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

from llm.adapters.base import DatasetAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(ROOT, "phmdc", "labels.csv")
INDEX = os.path.join(ROOT, "phmdc", "index.csv")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class PHM2019Adapter(DatasetAdapter):
    name = "PHM2019"

    def __init__(self, labels: str = LABELS, index: str = INDEX) -> None:
        self.labels_path = labels
        self.index_path = index
        if not os.path.exists(labels):
            raise FileNotFoundError(labels)
        with open(labels, encoding="utf-8-sig", newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        for r in self.rows:
            r["cycle"] = int(float(r["cycle"]))
            r["crack_mm"] = float(r["crack_mm"])
        self._t = 200
        self._e = 200

    @staticmethod
    def capabilities() -> dict:
        return {
            "dataset": "PHM2019",
            "entities": "T1..T8",
            "metrics": {
                "crack_mm_final": {"unit": "mm", "definition": "last measured crack length"},
                "crack_mm_at_cycle": {"unit": "mm", "definition": "crack length nearest to a cycle"},
                "crack_delta_mm": {"unit": "mm", "definition": "crack(B) - crack(A)"},
                "n_measurements": {"unit": "count", "definition": "number of crack measurements"},
                "n_specimens": {"unit": "count", "definition": "number of specimens"},
            },
            "validity": "8 aluminium lap-joints; single piezo pair; crack length is ground truth; not the same physics as L1 composites.",
        }

    def list_entities(self) -> list[str]:
        return sorted({r["specimen"] for r in self.rows})

    def _next(self) -> tuple[str, str]:
        self._t += 1
        self._e += 1
        return f"T{self._t:03d}", f"E{self._e:03d}"

    def _rows_for(self, gid: str) -> list[dict]:
        rows = [r for r in self.rows if r["specimen"] == gid.upper()]
        if not rows:
            raise KeyError(f"unknown PHM2019 specimen {gid}")
        return sorted(rows, key=lambda r: r["cycle"])

    def _nearest(self, gid: str, cycle: int) -> tuple[dict, bool]:
        rows = self._rows_for(gid)
        exact = [r for r in rows if r["cycle"] == cycle]
        if exact:
            return exact[0], True
        return min(rows, key=lambda r: abs(r["cycle"] - cycle)), False

    def _evidence(self, eid: str, title: str, tool: str, content: dict,
                  scope: dict, observed: float) -> dict:
        return {
            "evidence_id": eid,
            "evidence_type": "tool_result",
            "source": {
                "kind": "tool", "title": title, "tool_name": tool,
                "version": "0.1.0", "authors": [], "year": None,
                "venue": None, "doi": None, "url": None, "clause": None,
            },
            "content": content,
            "quote": None,
            "locator": {"file": self.labels_path, "page": None, "section": None,
                        "table": None, "figure": None, "row": None, "clause": None,
                        "char_start": None, "char_end": None},
            "scope": scope,
            "observed_lifetime_fraction": observed,
            "retrieved_at": _now(),
            "content_hash": None,
            "reliability": "primary",
        }

    def lookup(self, entity: str, metric: str = "crack_mm_final", **kwargs) -> dict:
        gid = entity.upper()
        if metric == "n_specimens":
            return self.overview()

        rows = self._rows_for(gid)
        max_cycle = max(r["cycle"] for r in rows)
        if metric == "n_measurements":
            value = len(rows)
            cycle = max_cycle
            observed = 1.0
            title = f"{gid} PHM2019 measurement count"
        elif metric == "crack_mm_final":
            r = max(rows, key=lambda x: x["cycle"])
            value = r["crack_mm"]
            cycle = r["cycle"]
            observed = 1.0
            title = f"{gid} final crack length"
        elif metric == "crack_mm_at_cycle":
            cycle = int(kwargs.get("cycle"))
            r, exact = self._nearest(gid, cycle)
            value = r["crack_mm"]
            cycle = r["cycle"]
            observed = cycle / max_cycle if max_cycle else 1.0
            title = f"{gid} crack at cycle {cycle}" + ("" if exact else " (nearest)")
        elif metric == "crack_mm_first":
            r = min(rows, key=lambda x: x["cycle"])
            value = r["crack_mm"]
            cycle = r["cycle"]
            observed = cycle / max_cycle if max_cycle else 0.0
            title = f"{gid} first crack length"
        else:
            raise ValueError(f"unsupported metric {metric}")

        tid, eid = self._next()
        result = {"gid": gid, "metric": metric, "value": value,
                  "cycle": cycle, "max_cycle": max_cycle}
        ev = self._evidence(
            eid, title, "phm2019.lookup",
            {"metric": metric, "value": value, "gid": gid, "cycle": cycle,
             "max_cycle": max_cycle},
            {"gid": gid, "campaign": "PHM2019", "modality": "PZT",
             "cycle_min": 0, "cycle_max": cycle,
             "lifetime_min": 0, "lifetime_max": observed},
            observed,
        )
        call = {"tool_call_id": tid, "tool_name": "phm_lookup",
                "arguments": {"gid": gid, "metric": metric, **kwargs},
                "result": result, "created_at": _now(), "elapsed_ms": 0.0}
        return {"tool_call": call, "evidence": ev, "result": result}

    def compare(self, entity: str, **kwargs) -> dict:
        gid = entity.upper()
        a = int(kwargs["cycle_a"])
        b = int(kwargs["cycle_b"])
        ra, ea = self._nearest(gid, a)
        rb, eb = self._nearest(gid, b)
        delta = rb["crack_mm"] - ra["crack_mm"]
        rows = self._rows_for(gid)
        max_cycle = max(r["cycle"] for r in rows)
        observed = max(ra["cycle"], rb["cycle"]) / max_cycle if max_cycle else 1.0
        tid, eid = self._next()
        result = {"gid": gid, "metric": "crack_delta_mm", "value": delta,
                  "cycle_a": ra["cycle"], "cycle_b": rb["cycle"],
                  "crack_a": ra["crack_mm"], "crack_b": rb["crack_mm"]}
        ev = self._evidence(
            eid, f"{gid} crack growth", "phm2019.compare",
            {"metric": "crack_delta_mm", "value": delta, "gid": gid,
             "cycle_a": ra["cycle"], "cycle_b": rb["cycle"],
             "crack_a": ra["crack_mm"], "crack_b": rb["crack_mm"]},
            {"gid": gid, "campaign": "PHM2019", "modality": "PZT",
             "cycle_min": min(ra["cycle"], rb["cycle"]),
             "cycle_max": max(ra["cycle"], rb["cycle"]),
             "lifetime_min": 0, "lifetime_max": observed},
            observed,
        )
        call = {"tool_call_id": tid, "tool_name": "phm_crack_delta",
                "arguments": {"gid": gid, "cycle_a": a, "cycle_b": b},
                "result": result, "created_at": _now(), "elapsed_ms": 0.0}
        return {"tool_call": call, "evidence": ev, "result": result}

    def overview(self) -> dict:
        entities = self.list_entities()
        tid, eid = self._next()
        result = {"dataset": "PHM2019", "n_specimens": len(entities),
                  "entities": entities, "n_measurements": len(self.rows)}
        ev = self._evidence(
            eid, "PHM2019 overview", "phm2019.overview",
            {"metric": "n_specimens", "value": len(entities),
             "n_measurements": len(self.rows)},
            {"gid": "ALL", "campaign": "PHM2019", "modality": "MULTI",
             "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            1.0,
        )
        call = {"tool_call_id": tid, "tool_name": "phm_overview",
                "arguments": {}, "result": result, "created_at": _now(),
                "elapsed_ms": 0.0}
        return {"tool_call": call, "evidence": ev, "result": result}
