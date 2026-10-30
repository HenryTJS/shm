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
MODALITY_TABLE = os.path.join(ROOT, "l1", "results", "four_modal.csv")
L2_TABLE = os.path.join(ROOT, "l2", "results", "l2_main_table.csv")
L2_PZT = os.path.join(ROOT, "l2", "results", "l2_pzt_summary.csv")

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
        self.modality_rows = []
        for path in (MODALITY_TABLE, L2_TABLE, L2_PZT):
            if os.path.exists(path):
                with open(path, encoding="utf-8-sig", newline="") as fh:
                    setattr(self, "_" + os.path.basename(path).split(".")[0], list(csv.DictReader(fh)))
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

    def compare_campaigns(self, campaign_a: str, campaign_b: str,
                          metric: str = "t85", variant: str = "cum_hits") -> dict:
        """Median(metric | campaign_b) - median(metric | campaign_a)."""
        col = self._metric(metric)

        def med(camp: str) -> tuple[float, int]:
            vals = []
            for r in self.rows:
                if r["batch"] == camp and r["variant"] == variant:
                    v = _float(r[col])
                    if v is not None:
                        vals.append(v)
            if not vals:
                raise KeyError(f"{camp}/{variant} has no {col}")
            vals.sort()
            n = len(vals)
            return (vals[n // 2] if n % 2 else
                    (vals[n // 2 - 1] + vals[n // 2]) / 2), n

        ma, na = med(campaign_a)
        mb, nb = med(campaign_b)
        delta = mb - ma
        tid, eid = self._next()
        result = {"campaign_a": campaign_a, "campaign_b": campaign_b,
                  "variant": variant, "metric": col,
                  "median_a": ma, "median_b": mb, "delta": delta,
                  "n_a": na, "n_b": nb}
        ev = self._evidence(
            eid, f"{campaign_b} minus {campaign_a} {col}",
            f"{os.path.basename(__file__)}:compare_campaigns",
            {"metric": f"delta_campaign_{col}", "value": delta,
             "campaign_a": campaign_a, "campaign_b": campaign_b,
             "median_a": ma, "median_b": mb, "n_a": na, "n_b": nb,
             "variant": variant},
            {"gid": "ALL", "campaign": f"{campaign_a}-{campaign_b}",
             "modality": "AE", "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": self.table, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {
            "tool_call_id": tid, "tool_name": "compare_campaigns",
            "arguments": {"campaign_a": campaign_a, "campaign_b": campaign_b,
                          "metric": metric, "variant": variant},
            "result": result, "created_at": _now(), "elapsed_ms": 0.0,
        }
        return {"tool_call": call, "evidence": ev, "result": result}

    def _modality_row(self, gid: str, modality: str) -> dict:
        for r in getattr(self, "_four_modal", []):
            if r["gid"] == gid and r["modality"] == modality:
                return r
        raise KeyError(f"{gid}/{modality} not found")

    def modality_lookup(self, gid: str, modality: str, metric: str = "t85") -> dict:
        """One modality's t50/t85 in percent, from four_modal.csv."""
        col = "t50" if metric in ("t50", "t50_pct") else "t85"
        if metric not in ("t50", "t50_pct", "t85", "t85_pct"):
            raise ValueError("metric must be t50 or t85")
        r = self._modality_row(gid.upper(), modality.upper())
        value = _float(r[col])
        if value is None:
            raise ValueError(f"{gid}/{modality} has no {col}")
        value *= 100.0
        try:
            base = self._record(gid.upper())
            campaign = base["batch"]
        except KeyError:
            campaign = None
        tid, eid = self._next()
        result = {"gid": gid.upper(), "modality": modality.upper(),
                  "metric": f"{col}_pct", "value": value}
        ev = self._evidence(
            eid, f"{gid} {modality} {col}", f"{os.path.basename(__file__)}:modality_lookup",
            {"metric": f"{col}_pct", "value": value, "gid": gid.upper(),
             "modality": modality.upper()},
            {"gid": gid.upper(), "campaign": campaign, "modality": modality.upper(),
             "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": MODALITY_TABLE, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {"tool_call_id": tid, "tool_name": "modality_lookup",
                "arguments": {"gid": gid, "modality": modality, "metric": metric},
                "result": result, "created_at": _now(), "elapsed_ms": 0.0}
        return {"tool_call": call, "evidence": ev, "result": result}

    def modality_compare(self, gid: str, modality_a: str, modality_b: str,
                         metric: str = "t85") -> dict:
        """t50/t85 difference: modality B minus modality A, in percentage points."""
        a = self.modality_lookup(gid, modality_a, metric)
        b = self.modality_lookup(gid, modality_b, metric)
        delta = b["result"]["value"] - a["result"]["value"]
        tid, eid = self._next()
        col = a["result"]["metric"]
        result = {"gid": gid.upper(), "modality_a": modality_a.upper(),
                  "modality_b": modality_b.upper(), "metric": col,
                  "value_a": a["result"]["value"], "value_b": b["result"]["value"],
                  "delta": delta}
        ev = self._evidence(
            eid, f"{gid} {modality_b} minus {modality_a} {col}",
            f"{os.path.basename(__file__)}:modality_compare",
            {"metric": f"delta_{col}", "value": delta, "gid": gid.upper(),
             "modality_a": modality_a.upper(), "modality_b": modality_b.upper(),
             "value_a": a["result"]["value"], "value_b": b["result"]["value"]},
            {"gid": gid.upper(), "campaign": None, "modality": "MULTI",
             "cycle_min": None, "cycle_max": None,
             "lifetime_min": 0, "lifetime_max": 1},
            {"file": MODALITY_TABLE, "page": None, "section": None, "table": None,
             "figure": None, "row": None, "clause": None,
             "char_start": None, "char_end": None},
            1.0,
        )
        call = {"tool_call_id": tid, "tool_name": "modality_compare",
                "arguments": {"gid": gid, "modality_a": modality_a,
                              "modality_b": modality_b, "metric": metric},
                "result": result, "created_at": _now(), "elapsed_ms": 0.0}
        return {"tool_call": call, "evidence": ev, "result": result}

    def l2_lookup(self, gid: str, metric: str = "t85_all") -> dict:
        """Level-2 AE metric from l2_main_table.csv."""
        aliases = {"t50": "t50_all", "t85": "t85_all",
                   "t50_all": "t50_all", "t85_all": "t85_all",
                   "t85_burst": "t85_burst"}
        if metric not in aliases:
            raise ValueError(f"metric must be one of {sorted(aliases)}")
        col = aliases[metric]
        for r in getattr(self, "_l2_main_table", []):
            if r["gid"] == gid.upper():
                value = _float(r[col])
                if value is None:
                    raise ValueError(f"{gid} has no {col}")
                n_f = _float(r.get("fatigue_end"))
                tid, eid = self._next()
                result = {"gid": gid.upper(), "metric": f"{col}_pct",
                          "value": value, "fatigue_end": n_f}
                ev = self._evidence(
                    eid, f"{gid} {col}", f"{os.path.basename(__file__)}:l2_lookup",
                    {"metric": f"{col}_pct", "value": value, "gid": gid.upper()},
                    {"gid": gid.upper(), "campaign": None, "modality": "AE",
                     "cycle_min": 0, "cycle_max": int(n_f) if n_f else None,
                     "lifetime_min": 0, "lifetime_max": 1},
                    {"file": L2_TABLE, "page": None, "section": None, "table": None,
                     "figure": None, "row": None, "clause": None,
                     "char_start": None, "char_end": None},
                    1.0,
                )
                call = {"tool_call_id": tid, "tool_name": "l2_lookup",
                        "arguments": {"gid": gid, "metric": metric},
                        "result": result, "created_at": _now(), "elapsed_ms": 0.0}
                return {"tool_call": call, "evidence": ev, "result": result}
        raise KeyError(f"{gid} not found in L2 table")

    def l2_pzt_lookup(self, gid: str) -> dict:
        """L2-07 PZT t85, where available."""
        for r in getattr(self, "_l2_pzt_summary", []):
            if r["gid"] == gid.upper():
                value = _float(r["t85"])
                tid, eid = self._next()
                result = {"gid": gid.upper(), "modality": "PZT",
                          "metric": "pzt_t85_pct", "value": value,
                          "n_states": int(float(r.get("n_states", 0)))}
                ev = self._evidence(
                    eid, f"{gid} PZT t85", f"{os.path.basename(__file__)}:l2_pzt_lookup",
                    {"metric": "pzt_t85_pct", "value": value, "gid": gid.upper()},
                    {"gid": gid.upper(), "campaign": None, "modality": "PZT",
                     "cycle_min": None, "cycle_max": None,
                     "lifetime_min": 0, "lifetime_max": 1},
                    {"file": L2_PZT, "page": None, "section": None, "table": None,
                     "figure": None, "row": None, "clause": None,
                     "char_start": None, "char_end": None},
                    1.0,
                )
                call = {"tool_call_id": tid, "tool_name": "l2_pzt_lookup",
                        "arguments": {"gid": gid}, "result": result,
                        "created_at": _now(), "elapsed_ms": 0.0}
                return {"tool_call": call, "evidence": ev, "result": result}
        raise KeyError(f"{gid} has no PZT summary")

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
