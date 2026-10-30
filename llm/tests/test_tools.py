# -*- coding: utf-8 -*-
"""Tests for the SHM tool layer and its evidence contract."""
import json
import os
import unittest

from llm.tools import ToolBox
from llm.validate import validate_graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "claim_evidence.schema.json")


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tb = ToolBox()

    def test_lookup_t85(self):
        r = self.tb.lookup("L1-03", "t85")
        self.assertAlmostEqual(r["result"]["value"], 86.4, places=3)
        self.assertEqual(r["evidence"]["evidence_type"], "tool_result")
        self.assertEqual(r["evidence"]["content"]["metric"], "t85_pct")

    def test_campaign_delta(self):
        r = self.tb.compare_to_campaign("L1-03", "t85")
        self.assertAlmostEqual(r["result"]["campaign_median"], 89.2, places=3)
        self.assertAlmostEqual(r["result"]["delta"], -2.8, places=3)

    def test_overview(self):
        r = self.tb.dataset_overview()
        self.assertEqual(r["result"]["n_specimens"], 29)

    def test_compare_campaigns(self):
        r = self.tb.compare_campaigns("C1", "C3", "t85")
        self.assertAlmostEqual(r["result"]["delta"], 4.8, places=3)
        self.assertEqual(r["evidence"]["content"]["metric"], "delta_campaign_t85_pct")

    def test_tool_evidence_passes_validator(self):
        look = self.tb.lookup("L1-03", "t85")
        comp = self.tb.compare_to_campaign("L1-03", "t85")
        graph = {
            "schema_version": "1.0",
            "graph_id": "G100",
            "question": "L1-03 t85 and C1 delta",
            "created_at": look["tool_call"]["created_at"],
            "policy": {
                "require_evidence": True,
                "allow_unsupported_claims": False,
                "max_evidence_per_claim": 8,
                "temporal_leakage_check": True,
                "require_numeric_tool_evidence": True,
            },
            "claims": [
                {
                    "claim_id": "C001",
                    "text": "L1-03 t85 is 86.4%.",
                    "claim_type": "numeric",
                    "subject": "L1-03",
                    "predicate": "t85_pct",
                    "value": 86.4,
                    "unit": "%",
                    "comparator": "eq",
                    "tolerance": 0.05,
                    "status": "supported",
                    "evidence_ids": [look["evidence"]["evidence_id"]],
                    "tool_call_ids": [look["tool_call"]["tool_call_id"]],
                    "scope": look["evidence"]["scope"],
                    "allowed_lifetime_max": 1.0,
                    "missing_evidence": None,
                    "confidence": 0.99,
                    "created_by": "Analyst",
                },
                {
                    "claim_id": "C002",
                    "text": "L1-03 is 2.8 pp below the C1 median.",
                    "claim_type": "comparative",
                    "subject": "L1-03 vs C1",
                    "predicate": "delta_t85_pct",
                    "value": -2.8,
                    "unit": "pp",
                    "comparator": "lt",
                    "tolerance": 0.1,
                    "status": "supported",
                    "evidence_ids": [comp["evidence"]["evidence_id"]],
                    "tool_call_ids": [comp["tool_call"]["tool_call_id"]],
                    "scope": comp["evidence"]["scope"],
                    "allowed_lifetime_max": 1.0,
                    "missing_evidence": None,
                    "confidence": 0.95,
                    "created_by": "Analyst",
                },
            ],
            "evidence": [look["evidence"], comp["evidence"]],
            "edges": [
                {"claim_id": "C001", "evidence_id": look["evidence"]["evidence_id"], "relation": "supports"},
                {"claim_id": "C002", "evidence_id": comp["evidence"]["evidence_id"], "relation": "supports"},
            ],
            "tool_calls": [look["tool_call"], comp["tool_call"]],
            "verification": {"status": "needs_review", "summary": "", "checks": []},
        }
        with open(SCHEMA, encoding="utf-8") as fh:
            schema = json.load(fh)
        rep = validate_graph(graph, schema)
        self.assertEqual(rep.failed, [], rep.failed)


if __name__ == "__main__":
    unittest.main()
