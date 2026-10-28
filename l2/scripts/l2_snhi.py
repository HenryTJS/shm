# -*- coding: utf-8 -*-
"""Compute SNHI50/85 from L2 AE bins."""
import argparse
import os

import numpy as np
import pandas as pd

OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"


def snhi(cycle, cum, frac):
    cum = np.asarray(cum, float)
    cycle = np.asarray(cycle, float)
    if cum.size == 0 or cum[-1] <= 0:
        return np.nan
    x = frac * cum[-1]
    j = int(np.searchsorted(cum, x))
    if j <= 0:
        return 0.0
    if j >= len(cum):
        return 100.0
    c0, c1 = cum[j - 1], cum[j]
    f = 0.0 if c1 == c0 else (x - c0) / (c1 - c0)
    val = cycle[j - 1] + f * (cycle[j] - cycle[j - 1])
    return float(val / cycle[-1] * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", default="L2-05")
    a = ap.parse_args()
    gid = a.gid
    df = pd.read_csv(os.path.join(OUT, "l2_ae_bins_%s.csv" % gid))
    df = df.sort_values("cycle").reset_index(drop=True)
    # prepend zero
    cyc = np.concatenate([[0.0], df["cycle"].to_numpy(float)])
    for key in ("hits_all", "hits_masked"):
        h = np.concatenate([[0.0], df[key].to_numpy(float)])
        cum = np.cumsum(h)
        print("%s %-12s total hits %10.0f  SNHI50 %7.2f%%  SNHI85 %7.2f%%"
              % (gid, key, cum[-1], snhi(cyc, cum, 0.50), snhi(cyc, cum, 0.85)))
    print("cycle range", cyc[1], "->", cyc[-1], "bins", len(df))


if __name__ == "__main__":
    main()
