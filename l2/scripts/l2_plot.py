# -*- coding: utf-8 -*-
"""Plot cumulative AE hits and SNHI for one L2 panel."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


ap = argparse.ArgumentParser()
ap.add_argument("--gid", default="L2-05")
a = ap.parse_args()
gid = a.gid
d = pd.read_csv(os.path.join(OUT, "l2_ae_bins_%s.csv" % gid)).sort_values("cycle")
cyc = np.concatenate([[0.0], d.cycle.to_numpy(float)])
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
for key, col in (("hits_all", "#4c8dff"), ("hits_masked", "#ff8a1f")):
    cum = np.cumsum(np.concatenate([[0.0], d[key].to_numpy(float)]))
    cum = cum / cum[-1] * 100
    ax.plot(cyc / cyc[-1] * 100, cum, lw=1.6, color=col, label=key)
    t50, t85 = snhi(cyc, cum, .5), snhi(cyc, cum, .85)
    ax.plot([t50, t85], [50, 85], "o", ms=4, color=col)
    ax.text(t85, 86, "%.1f%%" % t85, color=col, fontsize=8, ha="center")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel("lifetime fraction (%)")
ax.set_ylabel("cumulative hits (% of final)")
ax.set_title("%s  AE cumulative hits  (t85 all/masked)" % gid)
ax.grid(alpha=.2)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
p = os.path.join(OUT, "l2_curve_%s.png" % gid)
fig.savefig(p)
print("saved", p)
