# -*- coding: utf-8 -*-
"""Write a minimal L2 manifest for the three main panels (L2-04/05/07)."""
import csv
import os

OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
os.makedirs(OUT, exist_ok=True)

rows = [
    # gid, lab, level, fatigue_end_cycle, impact_cycles, modalities, analysis_set, note
    dict(gid="L2-04", lab="Delft", level="L2", fatigue_end=1046000,
         impacts="42000 (4 impacts)", modalities="AE, LUNA, Dolphicam",
         analysis="main", note="PZT absent; AE/LUNA to 1046k; BI_2k covers 2,076–42,076"),
    dict(gid="L2-05", lab="Delft", level="L2", fatigue_end=1140000,
         impacts="5000 (1 impact)", modalities="AE, LUNA, PZT, Dolphicam",
         analysis="main", note="fatigue sum 5k+5k+670k+120k+110k+230k=1,140k"),
    dict(gid="L2-07", lab="Delft", level="L2", fatigue_end=540000,
         impacts="before fatigue (4 impacts)", modalities="AE, LUNA, PZT, Dolphicam",
         analysis="main", note="post-impact only; no pre-impact fatigue baseline"),
    dict(gid="L2-03", lab="Delft", level="L2", fatigue_end=2710000,
         impacts="5000, 610k, 720k, 1130k, ...", modalities="AE (partial), LUNA, PZT, Dolphicam",
         analysis="appendix", note="AE Part2 not uploaded; AE 2370k may end ~2550k"),
    dict(gid="L2-06", lab="Delft", level="L2", fatigue_end=0,
         impacts="before QS (4 impacts)", modalities="AE (QS), LUNA (QS), PZT, Dolphicam",
         analysis="exclude", note="QS only; no fatigue"),
    dict(gid="L2-QS", lab="Delft", level="L2", fatigue_end=0,
         impacts="before QS (1 impact)", modalities="AE (QS), PZT",
         analysis="exclude", note="static baseline; failed 310 kN"),
]

with open(os.path.join(OUT, "l2_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("saved", os.path.join(OUT, "l2_manifest.csv"))
