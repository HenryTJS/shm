# -*- coding: utf-8 -*-
"""Shared helpers for the SHM skill contracts."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if any(part in {"results", "out"} for part in path.parts):
        raise PermissionError(
            "skill layer may not read results/ or out/ directly; use an adapter")
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def envelope(skill: str, status: str, data: dict, warnings: list[str] | None = None,
             evidence: list[dict] | None = None, report: str = "",
             computation: dict | None = None) -> dict:
    return {
        "skill": skill,
        "version": "0.1.0",
        "status": status,
        "data": data,
        "warnings": warnings or [],
        "evidence": evidence or [],
        "computation": computation or {"kind": "read", "formula": None, "inputs": []},
        "report": report,
    }


def evidence(evidence_id: str, file: str, title: str, content: dict,
             scope: dict, evidence_type: str = "dataset_metadata") -> dict:
    kind = "dataset" if evidence_type == "dataset_metadata" else "tool"
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source": {
            "kind": kind,
            "title": title,
            "tool_name": None,
            "version": None,
            "authors": [],
            "year": None,
            "venue": None,
            "doi": None,
            "url": None,
            "clause": None,
        },
        "content": content,
        "quote": None,
        "locator": {
            "file": file,
            "page": None,
            "section": None,
            "table": None,
            "figure": None,
            "row": None,
            "clause": None,
            "char_start": None,
            "char_end": None,
        },
        "scope": scope,
        "observed_lifetime_fraction": None,
        "retrieved_at": now(),
        "content_hash": None,
        "reliability": "primary",
    }


def emit(obj: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj.get("report") or json.dumps(obj, ensure_ascii=False, indent=2))
