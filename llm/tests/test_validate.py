# -*- coding: utf-8 -*-
"""Tests for the claim–evidence contract."""
import copy
import json
import os
import unittest

from llm.validate import validate_graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "claim_evidence.schema.json")
EXAMPLE = os.path.join(ROOT, "examples", "L1-03_t85.json")


def load():
    with open(SCHEMA, encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(EXAMPLE, encoding="utf-8") as fh:
        graph = json.load(fh)
    return schema, graph


class TestClaimEvidence(unittest.TestCase):
    def test_valid_example_passes(self):
        schema, graph = load()
        rep = validate_graph(graph, schema)
        self.assertEqual(rep.failed, [], rep.failed)

    def test_missing_evidence_fails(self):
        schema, graph = load()
        g = copy.deepcopy(graph)
        g["claims"][0]["evidence_ids"] = []
        g["edges"] = [e for e in g["edges"] if e["claim_id"] != "C001"]
        rep = validate_graph(g, schema)
        self.assertTrue(rep.failed)
        self.assertTrue(any(c["name"] == "claim_evidence" for c in rep.failed))

    def test_numeric_mismatch_fails(self):
        schema, graph = load()
        g = copy.deepcopy(graph)
        g["claims"][0]["value"] = 90.0
        rep = validate_graph(g, schema)
        self.assertTrue(any(c["name"] == "numeric_value" for c in rep.failed))

    def test_temporal_leakage_fails(self):
        schema, graph = load()
        g = copy.deepcopy(graph)
        g["claims"][0]["allowed_lifetime_max"] = 0.5
        rep = validate_graph(g, schema)
        self.assertTrue(any(c["name"] == "temporal_leakage" for c in rep.failed))

    def test_checked_in_invalid_examples_fail(self):
        schema, _ = load()
        for name in ("invalid_missing_evidence.json", "invalid_temporal_leakage.json"):
            with open(os.path.join(ROOT, "examples", name), encoding="utf-8") as fh:
                g = json.load(fh)
            rep = validate_graph(g, schema)
            self.assertTrue(rep.failed, name)


if __name__ == "__main__":
    unittest.main()
