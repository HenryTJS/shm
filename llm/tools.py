# -*- coding: utf-8 -*-
"""SHM tool layer that emits claim–evidence `tool_result` objects.

The point is not to compute numbers (Python does that), but to make every
number carry a machine-checkable evidence record:

    ToolBox.lookup_t85("L1-03")
      -> {"tool_call": T001, "evidence": E001}

The evidence is intentionally conservative: it records the CSV file, row,
scope, and observed lifetime fraction, so validate.py can reject a claim that
uses future data or the wrong metric.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_TABLE = os.path.join(ROOT, "l1", "results", "l1_hi_crosslab.csv")

METRICS = {
    "t50": "t50_pct",
    "t50_pct": "t50_pct",
    "t85": "t85_pct",
    "t85_pct": "t85_pct",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class ToolBox:
    """Read-only tool facade over the L1 HI table.

    Each public method returns a dict with:
      - tool_call: schema-compatible ToolCall
      - evidence: schema-compatible Evidence
      - result: plain Python result for the Agent
    """

    def __init__(self, table: str = DEFAULT_TABLE) -> None:
        self.table = table
        if not os.path.exists(table):
            raise FileNotFoundError(table)
        self.rows: list[dict[str, str]] = []
        with open(table, encoding="utf-8-sig", newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        self._t = 0
        self._e = 0

    def _next(self) -> tuple[str, str]:
        self._t += 1
        self._e += 1
        return f"T{self._t:03d}", f"E{self._e:03d}"

    def _record(self, gid: str, variant: str = "cum_hits") -> dict[str, str]:
        for r in self.rows:
            if r["gid"] == gid and r["variant"] == variant:
                return r
        raise KeyError(f"{gid}/{variant} not found")

    @staticmethod
    def _metric(metric: str) -> str:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")
        return METRICS[metric]

    def _evidence(self, eid: str, title: str, tool: str, content: dict,
                  scope: dict, locator: dict, observed: float | None) -> dict:
        return {
            "evidence_id": eid,
            "evidence_type": "tool_result",
            "source": {
                "kind": "tool",
                "title": title,
                "tool_name": tool,
                "version": "0.1.0",
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "url": None,
                "clause": None,
            },
            "content": content,
            "quote": None,
            "locator": locator,
            "scope": scope,
            "observed_lifetime_fraction": observed,
            "retrieved_at": _now(),
            "content_hash": None,
            "reliability": "primary",
        }

    def lookup(self, gid: str, metric: str = "t85", variant: str = "cum_hits") -> dict:
        """One specimen, one metric. Returns numeric evidence."""
        col = self._metric(metric)
        r = self._record(gid, variant)
        value = _float(r[col])
        if value is None:
            raise ValueError(f"{gid} has no {col}")
        n_f = _float(r.get("n_f"))
        tid, eid = self._next()
        result = {
            "gid": gid,
            "campaign": r["batch"],
            "lab": r["lab"],
            "variant": variant,
            "metric": col,
            "value": value,
            "n_f": int(n_f) if n_f is not None else None,
        }
        scope = {
            "gid": gid,
            "campaign": r["batch"],
            "modality": "AE",
            "cycle_min": 0,
            "cycle_max": int(n_f) if n_f is not None else None,
            "lifetime_min": 0,
            "lifetime_max": 1,
        }
        ev = self._evidence(
            eid, f"{gid} {variant} {col}", f"{os.path.basename(__file__)}:lookup",
            {"metric": col, "value": value, "gid": gid, "variant": variant,
             "n_f": int(n_f) if n_f is not None else None},
            scope,
            {"file": self.table, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {
            "tool_call_id": tid, "tool_name": "lookup",
            "arguments": {"gid": gid, "metric": metric, "variant": variant},
            "result": result, "created_at": _now(), "elapsed_ms": 0.0,
        }
        return {"tool_call": call, "evidence": ev, "result": result}

    def campaign_stats(self, campaign: str, metric: str = "t85",
                       variant: str = "cum_hits") -> dict:
        """Campaign-level median/min/max for one metric."""
        col = self._metric(metric)
        vals = []
        for r in self.rows:
            if r["batch"] == campaign and r["variant"] == variant:
                v = _float(r[col])
                if v is not None:
                    vals.append(v)
        if not vals:
            raise KeyError(f"{campaign}/{variant} has no {col}")
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        tid, eid = self._next()
        result = {"campaign": campaign, "variant": variant, "metric": col,
                  "n": n, "median": median, "min": vals[0], "max": vals[-1]}
        ev = self._evidence(
            eid, f"{campaign} {variant} {col} summary",
            f"{os.path.basename(__file__)}:campaign_stats",
            {"metric": f"campaign_median_{col}", "value": median,
             "campaign": campaign, "n": n, "variant": variant},
            {"gid": "ALL", "campaign": campaign, "modality": "AE",
             "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": self.table, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {
            "tool_call_id": tid, "tool_name": "campaign_stats",
            "arguments": {"campaign": campaign, "metric": metric, "variant": variant},
            "result": result, "created_at": _now(), "elapsed_ms": 0.0,
        }
        return {"tool_call": call, "evidence": ev, "result": result}

    def compare_to_campaign(self, gid: str, metric: str = "t85",
                            variant: str = "cum_hits") -> dict:
        """Specimen metric minus its campaign median."""
        col = self._metric(metric)
        r = self._record(gid, variant)
        v = _float(r[col])
        if v is None:
            raise ValueError(f"{gid} has no {col}")
        camp = r["batch"]
        vals = []
        for x in self.rows:
            if x["batch"] == camp and x["variant"] == variant:
                z = _float(x[col])
                if z is not None:
                    vals.append(z)
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        delta = v - median
        tid, eid = self._next()
        result = {"gid": gid, "campaign": camp, "variant": variant,
                  "metric": col, "specimen": v, "campaign_median": median,
                  "delta": delta, "n": n}
        ev = self._evidence(
            eid, f"{gid} vs {camp} {col}", f"{os.path.basename(__file__)}:compare",
            {"metric": f"delta_{col}", "value": delta, "gid": gid,
             "campaign": camp, "campaign_median": median, "specimen": v,
             "n": n, "variant": variant},
            {"gid": gid, "campaign": camp, "modality": "AE",
             "cycle_min": 0, "cycle_max": int(_float(r.get("n_f")) or 0),
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": self.table, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {
            "tool_call_id": tid, "tool_name": "compare_to_campaign",
            "arguments": {"gid": gid, "metric": metric, "variant": variant},
            "result": result, "created_at": _now(), "elapsed_ms": 0.0,
        }
        return {"tool_call": call, "evidence": ev, "result": result}

    def dataset_overview(self) -> dict:
        """Counts used by a DataAuditor."""
        gids = sorted({r["gid"] for r in self.rows})
        campaigns = sorted({r["batch"] for r in self.rows})
        tid, eid = self._next()
        result = {"n_specimens": len(gids), "campaigns": campaigns,
                  "gids": gids, "variants": sorted({r["variant"] for r in self.rows})}
        ev = self._evidence(
            eid, "L1 HI table overview", f"{os.path.basename(__file__)}:overview",
            {"metric": "n_specimens", "value": len(gids),
             "campaigns": campaigns},
            {"gid": "ALL", "campaign": None, "modality": "AE",
             "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": self.table, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {
            "tool_call_id": tid, "tool_name": "dataset_overview",
            "arguments": {}, "result": result, "created_at": _now(),
            "elapsed_ms": 0.0,
        }
        return {"tool_call": call, "evidence": ev, "result": result}

    def list_specimens(self) -> list[str]:
        return sorted({r["gid"] for r in self.rows})


if __name__ == "__main__":
    tb = ToolBox()
    print("specimens", len(tb.list_specimens()))
    import json
    print(json.dumps(tb.lookup("L1-03", "t85"), ensure_ascii=False, indent=2))
    print(json.dumps(tb.compare_to_campaign("L1-03", "t85"), ensure_ascii=False, indent=2))
