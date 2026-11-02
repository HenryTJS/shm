# -*- coding: utf-8 -*-
"""Contract smoke tests for check / compare / audit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # D:\shm
PY = sys.executable


def run(script: str, *args: str) -> dict:
    cmd = [PY, str(ROOT / "llm" / "skills" / script / "run.py"), *args, "--json"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"non-JSON output: {p.stdout[:500]} / {p.stderr[:500]}") from exc


class TestSkills(unittest.TestCase):
    def test_check_l1(self):
        r = run("check", "--dataset", "L1", "--gid", "L1-03")
        self.assertEqual(r["status"], "ok")
        ae = [m for m in r["data"]["modalities"] if m["name"] == "AE"][0]
        self.assertTrue(ae["available"])
        self.assertIsNone(ae["t85_pct"])
        self.assertTrue(r["evidence"])

    def test_check_l2_partial_pzt(self):
        r = run("check", "--dataset", "L2", "--gid", "L2-05")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["data"]["pzt_coverage_pct"], 20.2)
        pzt = [m for m in r["data"]["modalities"] if m["name"] == "PZT"][0]
        self.assertFalse(pzt["available"])
        self.assertTrue(any("20.2" in w for w in r["warnings"]))

    def test_check_l2_no_pzt(self):
        r = run("check", "--dataset", "L2", "--gid", "L2-04")
        pzt = [m for m in r["data"]["modalities"] if m["name"] == "PZT"][0]
        self.assertFalse(pzt["available"])

    def test_compare_is_disabled_without_raw_metric(self):
        r = run("compare", "--dataset", "L1", "--a", "C1", "--b", "C3", "--metric", "t85")
        self.assertEqual(r["status"], "unsupported")
        self.assertIn("raw metric", r["data"]["reason"])

    def test_metric_l1_06_raw(self):
        r = run("metric", "--dataset", "L1", "--gid", "L1-06")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["computation"]["kind"], "computed")
        self.assertGreater(r["data"]["t85_pct"], 90.0)
        self.assertLess(r["data"]["t85_pct"], 96.0)
        self.assertTrue(r["data"]["raw_ae"].startswith("E:"))

    def test_audit_pass(self):
        r = run("audit", "--graph", "llm/examples/L1-03_t85.json")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["data"]["n_failed"], 0)

    def test_audit_fail(self):
        r = run("audit", "--graph", "llm/examples/invalid_missing_evidence.json")
        self.assertEqual(r["status"], "insufficient")
        self.assertGreater(r["data"]["n_failed"], 0)

    def test_skill_sources_do_not_open_results_folder(self):
        for name in ("check", "metric", "compare", "audit"):
            text = (ROOT / "llm" / "skills" / name / "run.py").read_text(encoding="utf-8")
            self.assertNotRegex(text, r"[/\\\\]results[/\\\\]")
            self.assertNotRegex(text, r"[/\\\\]out[/\\\\]")

    def test_computation_field(self):
        r = run("compare", "--dataset", "L1", "--a", "C1", "--b", "C3", "--metric", "t85")
        self.assertIn("computation", r)
        self.assertEqual(r["computation"]["kind"], "derived")

    def test_common_blocks_results_path(self):
        sys.path.insert(0, str(ROOT))
        from llm.skills.common import read_csv
        with self.assertRaises(PermissionError):
            read_csv("l1/results/l1_hi_crosslab.csv")


if __name__ == "__main__":
    unittest.main()
