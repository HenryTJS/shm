# -*- coding: utf-8 -*-
"""Summarise L2 AE SNHI for all scanned panels."""
import glob
import os

import numpy as np
import pandas as pd

OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"


def snhi(cycle, cum, frac):
    x = frac * cum[-1]
    j = int(np.searchsorted(cum, x))
    if j <= 0:
        return 0.0
    if j >= len(cum):
        return 100.0
    c0, c1 = cum[j - 1], cum[j]
    f = 0.0 if c1 == c0 else (x - c0) / (c1 - c0)
    val = cycle[j - 1] + f * (cycle[j] - cycle[j - 1])
    return val / cycle[-1] * 100.0


rows = []
for fp in sorted(glob.glob(os.path.join(OUT, "l2_ae_bins_L2-*.csv"))):
    gid = os.path.basename(fp)[len("l2_ae_bins_"):-4]
    d = pd.read_csv(fp).sort_values("cycle")
    cyc = np.concatenate([[0.0], d.cycle.to_numpy(float)])
    row = dict(gid=gid, n_bins=len(d), cycle_end=float(d.cycle.max()))
    for key in ("hits_all", "hits_masked"):
        cum = np.cumsum(np.concatenate([[0.0], d[key].to_numpy(float)]))
        row[key] = int(cum[-1])
        row[key.replace("hits", "t50")] = snhi(cyc, cum, .50)
        row[key.replace("hits", "t85")] = snhi(cyc, cum, .85)
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "l2_snhi_summary.csv"), index=False)
print("saved")
