# -*- coding: utf-8 -*-
"""Raw-data adapters for the skill layer.

These adapters only inspect original acquisition directories. They never read
`*/results/*` or `*/out/*`; derived metrics belong to the `metric` skill.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

L1_ROOT = Path(os.environ.get("SHM_L1_ROOT", "E:/l1"))
L2_ROOT = Path(os.environ.get("SHM_L2_ROOT", "E:/l2"))
ROOT = Path(__file__).resolve().parents[2]
PHM_LABELS = ROOT / "phmdc" / "labels.csv"


def _dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def _files(path: Path, pattern: str = "*") -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.glob(pattern) if p.is_file())


class RawL1Adapter:
    name = "L1-raw"

    def __init__(self, root: Path = L1_ROOT):
        self.root = root

    def list_entities(self) -> list[str]:
        return [d for d in _dirs(self.root) if d.startswith("L1-")]

    def describe(self, gid: str) -> dict:
        path = self.root / gid
        if not path.exists():
            raise KeyError(gid)
        mods = []
        ae = path / "AE"
        mods.append({"name": "AE", "available": ae.exists(),
                     "files": len(_files(ae)) if ae.exists() else 0})
        fbg = path / "FBG"
        mods.append({"name": "FBG", "available": fbg.exists(),
                     "files": len(_files(fbg)) if fbg.exists() else 0})
        luna = path / "LUNA"
        mods.append({"name": "DFOS/LUNA", "available": luna.exists(),
                     "files": len(_files(luna)) if luna.exists() else 0})
        pzt = [p for p in path.iterdir() if p.is_dir() and p.name.upper().startswith("PZT")] if path.exists() else []
        mods.append({"name": "PZT", "available": bool(pzt),
                     "files": sum(len(_files(p)) for p in pzt)})
        return {"dataset": "L1", "gid": gid, "root": str(self.root),
                "exists": True, "modalities": mods,
                "raw_files": len(_files(path)), "has_pdf": (path / f"{gid}.pdf").exists()}


class RawL2Adapter:
    name = "L2-raw"

    def __init__(self, root: Path = L2_ROOT):
        self.root = root

    def list_entities(self) -> list[str]:
        return [d for d in _dirs(self.root) if d.startswith("L2-")]

    def describe(self, gid: str) -> dict:
        path = self.root / gid
        if not path.exists():
            raise KeyError(gid)
        mods = []
        ae = path / "AE"
        mods.append({"name": "AE", "available": ae.exists(),
                     "files": len(_files(ae)) if ae.exists() else 0})
        pzt = path / "PZT"
        mods.append({"name": "PZT", "available": pzt.exists(),
                     "files": len(_files(pzt)) if pzt.exists() else 0})
        luna = path / "LUNA"
        mods.append({"name": "DFOS/LUNA", "available": luna.exists(),
                     "files": len(_files(luna)) if luna.exists() else 0})
        return {"dataset": "L2", "gid": gid, "root": str(self.root),
                "exists": True, "modalities": mods, "raw_files": len(_files(path))}


class RawPHM2019Adapter:
    name = "PHM2019-raw"

    def describe(self, gid: str) -> dict:
        if not PHM_LABELS.exists():
            raise KeyError("PHM labels missing")
        with open(PHM_LABELS, encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("specimen") == gid]
        if not rows:
            raise KeyError(gid)
        cycles = [int(float(r["cycle"])) for r in rows]
        cracks = [float(r["crack_mm"]) for r in rows]
        return {"dataset": "PHM2019", "gid": gid, "exists": True,
                "modalities": [{"name": "PZT", "available": True, "files": len(rows)}],
                "n_measurements": len(rows), "cycle_max": max(cycles),
                "crack_mm_final": cracks[cycles.index(max(cycles))]}
