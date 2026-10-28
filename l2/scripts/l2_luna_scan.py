# -*- coding: utf-8 -*-
"""Prototype LUNA/DFOS scan for L2: spatial deviation from a healthy baseline."""
import argparse
import csv
import os

import numpy as np

ROOT = r"E:\l2"
OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
os.makedirs(OUT, exist_ok=True)


def read_profile(path, row=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for _ in range(4):
            fh.readline()
        fh.readline()  # length row
        for i in range(row + 1):
            line = fh.readline()
        k = line.find("\t")
        return np.fromstring(line[k + 1:], sep="\t")


def count_rows(path):
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            n += b.count(b"\n")
    return max(0, n - 5)  # 4 header + length row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", default="L2-04")
    ap.add_argument("--file", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--c0", type=float, required=True)
    ap.add_argument("--c1", type=float, required=True)
    ap.add_argument("--step", type=int, default=10)
    a = ap.parse_args()
    base = read_profile(a.base)
    print("baseline points", len(base), "file", a.file, flush=True)
    nrow = count_rows(a.file)
    print("rows", nrow, flush=True)
    rows = []
    with open(a.file, "r", encoding="utf-8", errors="ignore") as fh:
        for _ in range(5):
            fh.readline()
        m = min(len(base), 200000)
        for i in range(nrow):
            line = fh.readline()
            if not line:
                break
            if i % a.step:
                continue
            k = line.find("\t")
            x = np.fromstring(line[k + 1:], sep="\t")
            if x.size < 100:
                continue
            n = min(len(base), x.size)
            dev = x[:n] - base[:n]
            lo, hi = 80, n - 80
            if hi <= lo:
                continue
            seg = np.abs(dev[lo:hi])
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            val = float(np.percentile(seg, 99))
            cyc = a.c0 + (i / max(1, nrow - 1)) * (a.c1 - a.c0)
            rows.append(dict(gid=a.gid, cycle=cyc, local=val))
    p = os.path.join(OUT, "l2_luna_%s.csv" % a.gid)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gid", "cycle", "local"])
        w.writeheader()
        w.writerows(rows)
    print("saved", p, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
