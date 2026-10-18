"""生成论文 Table 1 / Table 2 的 markdown 草稿（含 eta2 与 p 列）。

只读 l1/results 下的既有 CSV，不重算原始数据。
用法：python l1/make_tables_45.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RES = Path(__file__).resolve().parent / "results"
BATCH_LAB = {"C1": "Delft", "C2": "Delft", "C3": "Patras", "C4": "Patras"}


def read(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def spread(values: list[float]) -> float:
    return float(max(values) - min(values))


def variance_rows() -> dict[tuple[str, str], dict]:
    out = {}
    for r in read("crosslab_variance.csv"):
        out[(r["quantity"], r["scheme"])] = r
    return out


def fmt_var(vr: dict | None, key: str, nd: int = 3) -> str:
    if vr is None:
        return "n/a"
    return fmt(fnum(vr, key), nd)


def table1() -> str:
    rows = [r for r in read("l1_hi_crosslab.csv") if r["variant"] == "cum_hits"]
    var = variance_rows()
    L: list[str] = []
    L.append("| campaign | laboratory | n | median $t_{50}$ (%) | median $t_{85}$ (%) "
             "| range of $t_{85}$ within a campaign (pp) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for c in ("C1", "C2", "C3", "C4"):
        sub = [r for r in rows if r["batch"] == c]
        t50 = [fnum(r, "t50_pct") for r in sub]
        t85 = [fnum(r, "t85_pct") for r in sub]
        t50 = [v for v in t50 if v is not None]
        t85 = [v for v in t85 if v is not None]
        L.append(
            f"| {c} | {BATCH_LAB[c]} | {len(sub)} | {np.median(t50):.1f} | "
            f"{np.median(t85):.1f} | {spread(t85):.2f} |"
        )
    # 不再加汇总行：4.85（四个中位之间的极差）与 14.75（组内极差的中位数）
    # 正文 §3.3 已逐字给出，表里再列一遍是纯重复。表只归档正文没有的逐活动明细。
    L.append("")
    L.append("| quantity | grouping | $\\eta^2$ | null median | $p$ (lower tail) |")
    L.append("| --- | --- | --- | --- | --- |")
    specs = [
        ("AE cum_hits t85_pct", "batch", "campaign", "p_le"),
        ("AE cum_hits t85_pct", "university", "university", "p_le"),
        ("AE cum_hits t50_pct", "university", "university", "p_le"),
        ("AE cum_energy t50_pct", "batch", "campaign", "p_ge"),
    ]
    for q, sch, label, pk in specs:
        vr = var.get((q, sch))
        star = " \\*" if pk == "p_ge" else ""
        L.append(
            f"| {q.split(' ', 1)[1]} | {label} | {fmt_var(vr, 'eta2')} | "
            f"{fmt_var(vr, 'null_median')} | {fmt_var(vr, pk)}{star} |"
        )
    L.append("")
    L.append("\\* control row, upper tail: the share is larger than under shuffling.")
    return "\n".join(L)


def table2() -> str:
    rows = read("pzt_crosslab_summary.csv")
    var = variance_rows()
    L: list[str] = []
    L.append("| campaign | laboratory | n | median slope (microsecond / 100% life) | Kendall $\\tau$ > 0 |")
    L.append("| --- | --- | --- | --- | --- |")
    for c in ("C1", "C3", "C4"):
        sub = [r for r in rows if r["batch"] == c]
        sl = [fnum(r, "tof_slope_ts") for r in sub]
        sl = [v for v in sl if v is not None]
        pos = sum(1 for v in sl if v > 0)
        L.append(
            f"| {c} | {BATCH_LAB[c]} | {len(sub)} | +{np.median(sl):.4f} | {pos}/{len(sub)} |"
        )
    L.append("")
    L.append("| quantity | grouping | $\\eta^2$ | null median | $p$ |")
    L.append("| --- | --- | --- | --- | --- |")
    for q, sch, label in [
        ("PZT tof_slope_ts", "batch", "campaign"),
        ("PZT tof_slope_ts", "university", "university"),
    ]:
        vr = var.get((q, sch))
        L.append(
            f"| TOF drift slope | {label} | {fmt_var(vr, 'eta2')} | {fmt_var(vr, 'null_median')} | "
            f"{fmt_var(vr, 'p_le')} |"
        )
    return "\n".join(L)


def main() -> None:
    print("### Table 1　AE 跨批次（29 组，x = 85 于累积撞击数）\n")
    print(table1())
    print("\n\n### Table 2　PZT 跨批次（17 组）\n")
    print(table2())


if __name__ == "__main__":
    main()
