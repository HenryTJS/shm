# -*- coding: utf-8 -*-
"""Freeze the L2 main table: AE t85 (all / impact-masked / burst-masked) + PZT L2-07."""
import os

import numpy as np
import pandas as pd

OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
BURST = {
    "L2-04": [(0, 10000), (40000, 44000)],
    "L2-05": [(0, 10000), (75000, 90000)],
    "L2-07": [],
}


def snhi(cyc, cum, f):
    x = f * cum[-1]
    j = int(np.searchsorted(cum, x))
    if j <= 0:
        return 0.0
    if j >= len(cum):
        return 100.0
    c0, c1 = cum[j - 1], cum[j]
    q = 0 if c1 == c0 else (x - c0) / (c1 - c0)
    return (cyc[j - 1] + q * (cyc[j] - cyc[j - 1])) / cyc[-1] * 100


rows = []
for gid in ("L2-04", "L2-05", "L2-07"):
    d = pd.read_csv(os.path.join(OUT, "l2_ae_bins_%s.csv" % gid)).sort_values("cycle")
    cyc = np.r_[0.0, d.cycle.to_numpy(float)]
    out = dict(gid=gid, fatigue_end=float(d.cycle.max()))
    for key, label in (("hits_all", "all"), ("hits_masked", "impact")):
        cum = np.cumsum(np.r_[0.0, d[key].to_numpy(float)])
        out["t50_" + label] = snhi(cyc, cum, .50)
        out["t85_" + label] = snhi(cyc, cum, .85)
    keep = np.ones(len(d), bool)
    for a, b in BURST[gid]:
        keep &= ~((d.cycle >= a) & (d.cycle <= b))
    cum = np.cumsum(np.r_[0.0, d.hits_masked.to_numpy(float) * keep])
    out["t85_burst"] = snhi(cyc, cum, .85)
    out["hits_total"] = int(d.hits_all.sum())
    rows.append(out)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "l2_main_table.csv"), index=False)

pzt = pd.read_csv(os.path.join(OUT, "l2_pzt_L2-07.csv"))
tot = pzt.D_cum.iloc[-1]


def t_at(frac):
    x = frac * tot
    j = int(np.searchsorted(pzt.D_cum.to_numpy(), x))
    if j <= 0:
        return 0.0
    if j >= len(pzt):
        return 100.0
    c0, c1 = pzt.D_cum.iloc[j - 1], pzt.D_cum.iloc[j]
    f = 0 if c1 == c0 else (x - c0) / (c1 - c0)
    cyc = pzt.cycle.iloc[j - 1] + f * (pzt.cycle.iloc[j] - pzt.cycle.iloc[j - 1])
    return cyc / 540000 * 100


pz = pd.DataFrame([dict(gid="L2-07", modality="PZT", t50=t_at(.50), t85=t_at(.85),
                        n_states=len(pzt), note="state index mapped linearly to 0-540k")])
print(pz.to_string(index=False))
pz.to_csv(os.path.join(OUT, "l2_pzt_summary.csv"), index=False)
print("saved")
