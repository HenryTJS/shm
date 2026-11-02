# -*- coding: utf-8 -*-
"""metric: compute t50/t85 from raw AE frames and raw FBG load windows.

Scope v0: L1 `.DTA` groups (C3/C4) with raw `AE/` and `FBG/`. No results access.
C1/C2 `.pridb` and L2 are reported as unsupported until their raw frame builder
is added.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # D:\shm
sys.path.insert(0, str(HERE.parents[2] / "l1"))

import numpy as np

from llm.adapters import RawL1Adapter
from llm.skills.common import emit, envelope, evidence

FRAME_S = 600.0
CACHE = HERE.parent / "cache"


def _t_at(hi: np.ndarray, cycle: np.ndarray, level: float, n_f: float) -> float:
    target = level * hi[-1]
    j = int(np.searchsorted(hi, target))
    if j <= 0:
        cyc = cycle[0]
    elif j >= len(cycle):
        cyc = cycle[-1]
    else:
        h0, h1 = hi[j - 1], hi[j]
        f = 0.0 if h1 == h0 else (target - h0) / (h1 - h0)
        cyc = cycle[j - 1] + f * (cycle[j] - cycle[j - 1])
    return float(cyc / n_f * 100.0) if n_f else float("nan")


def compute_l1(gid: str) -> dict:
    import ae_cycle
    import ae_frames
    from datasets import N_F_FALLBACK

    root = str(RawL1Adapter().root)
    fr = ae_frames.group_frames(gid, root=root, frame_s=FRAME_S, verbose=False)
    if fr is None or "n" not in fr:
        raise ValueError("no raw DTA frames")
    t_sec = np.asarray(fr["t_day"], dtype=float) * 86400.0
    n = np.asarray(fr["n"], dtype=float)
    pts = ae_cycle.group_load(gid, root=root)
    if not pts:
        raise ValueError("no raw FBG load windows")
    tf = np.asarray([p[0] for p in pts], dtype=float)
    lf = np.asarray([1.0 if p[1] > ae_cycle.P2P_LOAD else 0.0 for p in pts], dtype=float)
    g = ae_cycle._ratio_on_grid(t_sec, tf, lf, FRAME_S,
                                fill=ae_cycle.FILL, n_hits=n,
                                pause_win=ae_cycle._pause_win(FRAME_S))
    n_f = N_F_FALLBACK.get(gid)
    if not n_f:
        raise ValueError("no raw n_f for this gid")
    load_s = float(np.sum(g)) * FRAME_S
    f_hz = n_f / load_s if load_s > 0 else float("nan")
    cycle = np.cumsum(g * FRAME_S * f_hz)
    hi = np.cumsum(n) / np.sum(n)
    t50 = _t_at(hi, cycle, 0.50, n_f)
    t85 = _t_at(hi, cycle, 0.85, n_f)
    return {"gid": gid, "n_f": int(n_f), "n_frames": int(len(n)),
            "hits": int(np.sum(n)), "f_hz": float(f_hz),
            "t50_pct": t50, "t85_pct": t85,
            "frame_s": FRAME_S,
            "raw_ae": f"{root}/{gid}/AE",
            "raw_fbg": f"{root}/{gid}/FBG"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="L1", choices=["L1", "L2", "PHM2019"])
    ap.add_argument("--gid", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.dataset != "L1":
        obj = envelope("metric", "unsupported",
                       {"dataset": args.dataset, "gid": args.gid,
                        "reason": "metric v0 only supports L1 DTA groups"},
                       computation={"kind": "computed", "formula": "raw HI -> t85",
                                    "inputs": ["raw AE", "raw FBG"]},
                       report="# metric\n当前只支持 L1 的 `.DTA` 组。")
        emit(obj, args.json)
        return 1

    try:
        data = compute_l1(args.gid)
    except Exception as exc:
        obj = envelope("metric", "unsupported",
                       {"dataset": "L1", "gid": args.gid, "reason": str(exc)},
                       computation={"kind": "computed", "formula": "raw HI -> t85",
                                    "inputs": [f"raw {args.gid}"]},
                       report=f"# metric\n无法从 raw 计算：{exc}")
        emit(obj, args.json)
        return 1

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"metric_{args.gid}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = [evidence("E-METRIC-RAW", data["raw_ae"], f"{args.gid} raw AE+FBG",
                   {"gid": args.gid, "n_f": data["n_f"], "hits": data["hits"],
                    "t50_pct": data["t50_pct"], "t85_pct": data["t85_pct"]},
                   {"gid": args.gid, "campaign": None, "modality": "AE",
                    "cycle_min": 0, "cycle_max": data["n_f"],
                    "lifetime_min": 0, "lifetime_max": 1})]
    report = (f"# metric: {args.gid}\n\n"
              f"- raw frames: {data['n_frames']}\n"
              f"- hits: {data['hits']}\n"
              f"- t50: {data['t50_pct']:.2f}%\n"
              f"- t85: {data['t85_pct']:.2f}%\n")
    obj = envelope("metric", "ok", data, evidence=ev,
                   computation={"kind": "computed",
                                "formula": "cumsum(raw hits)/total -> t85 on raw cycle axis",
                                "inputs": [data["raw_ae"], data["raw_fbg"]]},
                   report=report)
    emit(obj, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
