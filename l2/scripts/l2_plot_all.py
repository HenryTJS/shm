# -*- coding: utf-8 -*-
"""Combined L2 AE cumulative curves for the three main panels."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
GIDS = ["L2-04", "L2-05", "L2-07"]
COL = {"L2-04": "#4c8dff", "L2-05": "#ff8a1f", "L2-07": "#2e8b57"}


def snhi(cyc, cum, f):
    x = f * cum[-1]
    j = int(np.searchsorted(cum, x))
    if j <= 0:
        return 0
    if j >= len(cum):
        return 100
    c0, c1 = cum[j - 1], cum[j]
    q = 0 if c1 == c0 else (x - c0) / (c1 - c0)
    return (cyc[j - 1] + q * (cyc[j] - cyc[j - 1])) / cyc[-1] * 100


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), dpi=200, sharey=True)
for ax, gid in zip(axes, GIDS):
    d = pd.read_csv(os.path.join(OUT, "l2_ae_bins_%s.csv" % gid)).sort_values("cycle")
    cyc = np.r_[0.0, d.cycle.to_numpy(float)]
    cum = np.cumsum(np.r_[0.0, d.hits_masked.to_numpy(float)])
    cum = cum / cum[-1] * 100
    ax.plot(cyc / cyc[-1] * 100, cum, color=COL[gid], lw=1.5)
    t85 = snhi(cyc, cum, .85)
    ax.axvline(t85, color=COL[gid], ls=":", lw=.9)
    ax.text(t85 + 1.5, 10, "t85 %.1f%%" % t85, color=COL[gid], fontsize=7.5)
    ax.set_title(gid, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.2)
    ax.set_xlabel("lifetime fraction (%)", fontsize=7.5)
axes[0].set_ylabel("cumulative hits (% of final)", fontsize=7.5)
for ax in axes:
    ax.tick_params(labelsize=7)
fig.tight_layout()
p = os.path.join(OUT, "l2_curves_all.png")
fig.savefig(p)
print("saved", p)
