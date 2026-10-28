# -*- coding: utf-8 -*-
"""L2-04/05/07 AE streaming scan: per-file summary + 1,000-cycle hit bins."""
import argparse
import csv
import os
import sqlite3

import numpy as np

ROOT = r"E:\l2"
OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
os.makedirs(OUT, exist_ok=True)

# gid -> list of (relative path, cycle_start, cycle_end, phase, impact_window or None)
SPECS = {
    "L2-04": [
        (r"BI\BI_0k\L204.pridb", 0, 2076, "BI_before_impact", None),
        (r"BI\BI_2k\L204-BI-2k.pridb", 2076, 42076, "BI_before_impact", (40000, 44000)),
        (r"AI\AI_42k\L204_42K.pridb", 42076, 162000, "AI_after_impact", (40000, 44000)),
        (r"AI\AI_162k\L204_162K.pridb", 162000, 352000, "AI_after_impact", None),
        (r"AI\AI_352k\L204_352K.pridb", 352000, 365000, "AI_after_impact", None),
        (r"AI\AI_365k\L204_365K.pridb", 365000, 415000, "AI_after_impact", None),
        (r"AI\AI_415k\L204_415K.pridb", 415000, 416000, "AI_after_impact", None),
        (r"AI\AI_416k\L204_416K.pridb", 416000, 506000, "AI_after_impact", None),
        (r"AI\AI_506k\L204_506K.pridb", 506000, 826000, "AI_after_impact", None),
        (r"AI\AI_826k\L204_826K.pridb", 826000, 1046000, "AI_after_impact", None),
    ],
    "L2-05": [
        (r"Fatigue\BI\L205_BI.pridb", 0, 5000, "BI_before_impact", (3000, 7000)),
        (r"Fatigue\AI\AI_5K\L205_AI_5000.pridb", 5000, 10000, "AI_after_impact", (3000, 7000)),
        (r"Fatigue\AI\AI_10K\L205_AI_10K.pridb", 10000, 680000, "AI_after_impact", None),
        (r"Fatigue\AI\AI_680K\L205_AI_680.pridb", 680000, 800000, "AI_after_impact", None),
        (r"Fatigue\AI\AI_800K\L205_AI_800.pridb", 800000, 850000, "AI_after_impact", None),
        (r"Fatigue\AI\AI_850K\L205_AI_850.pridb", 850000, 910000, "AI_after_impact", None),
        (r"Fatigue\AI\AI_910K\L205_AI_910.pridb", 910000, 1140000, "AI_after_impact", None),
    ],
    "L2-07": [
        (r"Fatigue\AI\0K\L2_07.pridb", 0, 160000, "AI_after_impact", None),
        (r"Fatigue\AI\160K\L207_160K.pridb", 160000, 540000, "AI_after_impact", None),
    ],
}


def scan_file(gid, rel, c0, c1, phase, impact, binw, chunk=300000):
    path = os.path.join(ROOT, gid, "AE", rel)
    if not os.path.exists(path):
        return None, []
    con = sqlite3.connect(path)
    con.execute("PRAGMA cache_size=-131072")
    r0 = con.execute("SELECT Time FROM ae_data WHERE SetType=2 ORDER BY SetID ASC LIMIT 1").fetchone()
    r1 = con.execute("SELECT Time FROM ae_data WHERE SetType=2 ORDER BY SetID DESC LIMIT 1").fetchone()
    if not r0 or not r1:
        con.close()
        return None, []
    t0, t1 = float(r0[0] or 0), float(r1[0] or 0)
    nbin = max(1, int(np.ceil((c1 - c0) / binw)))
    hits_all = np.zeros(nbin, dtype=np.int64)
    hits_msk = np.zeros(nbin, dtype=np.int64)
    en_all = np.zeros(nbin, dtype=np.float64)
    n_hit = 0
    cur = con.execute("SELECT Time,SetType,COALESCE(Eny,0) FROM ae_data WHERE SetType=2")
    while True:
        rows = cur.fetchmany(chunk)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.float64)
        if a.size == 0:
            continue
        tt = a[:, 0]
        en = a[:, 2]
        if t1 > t0:
            cyc = c0 + (tt - t0) / (t1 - t0) * (c1 - c0)
        else:
            cyc = np.full(tt.shape, float(c0))
        np.clip(cyc, c0, c1 - 1e-9, out=cyc)
        i = ((cyc - c0) / binw).astype(np.int64)
        np.clip(i, 0, nbin - 1, out=i)
        n_hit += int(i.size)
        hits_all += np.bincount(i, minlength=nbin)
        en_all += np.bincount(i, weights=en, minlength=nbin)
        if impact is not None:
            keep = ~((cyc >= impact[0]) & (cyc <= impact[1]))
            i2 = i[keep]
        else:
            i2 = i
        hits_msk += np.bincount(i2, minlength=nbin)
    con.close()
    summ = dict(gid=gid, rel=rel, phase=phase, c0=c0, c1=c1, n_hit=n_hit,
                t0_s=t0 / 1e7, t1_s=t1 / 1e7, nbin=nbin)
    bins = []
    for i in range(nbin):
        if hits_all[i] == 0:
            continue
        ctr = c0 + (i + 0.5) * binw
        bins.append(dict(gid=gid, cycle=ctr, hits_all=int(hits_all[i]),
                         hits_masked=int(hits_msk[i]), en_all=float(en_all[i])))
    return summ, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", default="L2-05")
    ap.add_argument("--bin", type=float, default=1000.0)
    a = ap.parse_args()
    gid = a.gid
    if gid not in SPECS:
        raise SystemExit("unknown gid")
    all_bins = []
    rows = []
    for rel, c0, c1, phase, impact in SPECS[gid]:
        print("scan", gid, rel, flush=True)
        summ, bins = scan_file(gid, rel, c0, c1, phase, impact, a.bin)
        if summ is None:
            print("  no hits", flush=True)
            continue
        print("  hits {:,}  t {:.0f}->{:.0f} s".format(summ["n_hit"], summ["t0_s"], summ["t1_s"]), flush=True)
        rows.append(summ)
        all_bins.extend(bins)
    with open(os.path.join(OUT, "l2_ae_summary_%s.csv" % gid), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gid", "rel", "phase", "c0", "c1", "n_hit", "t0_s", "t1_s", "nbin"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "l2_ae_bins_%s.csv" % gid), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gid", "cycle", "hits_all", "hits_masked", "en_all"])
        w.writeheader()
        w.writerows(all_bins)
    print("saved", gid, "bins", len(all_bins), flush=True)


if __name__ == "__main__":
    main()
