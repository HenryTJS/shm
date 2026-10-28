# -*- coding: utf-8 -*-
"""PZT state-level D for L2-07 (Fatigue1 + Fatigue2)."""
import importlib.util
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = r"D:\shm\l1"
OUT = r"C:\Users\ASUS\Documents\deepseek-harness\default-workspace\l2_work"
FREQ = "100kHz_5cycles"
REP = 1
ROOT = r"E:\l2\L2-07\PZT"

spec = importlib.util.spec_from_file_location("pzt_index", os.path.join(HERE, "pzt_index.py"))
PI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PI)


def state_no(name):
    m = re.match(r"State_(\d+)_", name)
    return int(m.group(1)) if m else 10 ** 9


def signals(session, state, ba):
    out = {}
    for k in range(1, 9):
        p = os.path.join(ROOT, session, state, FREQ, "Actionneur%d" % k,
                         "measured_data_rep_%d.mat" % REP)
        if not os.path.exists(p):
            return None
        tr = sio.loadmat(p)["Time_Response"]
        for j, m in enumerate(PI.rx_of(k)):
            x = PI.prep(tr[:, 2 + j], ba)
            out[(k, m)] = PI.rn(x)
    return out


def main():
    ba = PI.bandpass(float(FREQ.split("kHz")[0]))
    sessions = ["L207_PZT1_Fatigue1", "L207_PZT1_Fatigue2"]
    states = []
    for s in sessions:
        d = os.path.join(ROOT, s)
        for x in os.listdir(d):
            if x.startswith("State_"):
                states.append((s, x))
    states.sort(key=lambda t: (sessions.index(t[0]), state_no(t[1])))
    print("states", len(states))
    rows, prev = [], None
    for i, (sess, st) in enumerate(states, 1):
        sig = signals(sess, st, ba)
        if sig is None:
            print("skip", sess, st)
            continue
        row = dict(gid="L2-07", state=st, idx=i, n=len(states))
        if prev is not None:
            d = [1.0 - float(np.dot(sig[k], prev[k])) for k in sig if k in prev]
            row["D"] = float(np.median(d)) if d else np.nan
        rows.append(row)
        prev = sig
    df = pd.DataFrame(rows)
    df["cycle"] = (df.idx - 1) / max(1, int(df.n.iloc[0]) - 1) * 540000
    df["D_cum"] = df.D.fillna(0).cumsum()
    tot = df.D_cum.iloc[-1]
    df["D_frac"] = df.D_cum / tot * 100 if tot > 0 else np.nan

    def t_at(frac):
        x = frac * tot
        j = int(np.searchsorted(df.D_cum.to_numpy(), x))
        if j <= 0:
            return 0.0
        if j >= len(df):
            return 100.0
        c0, c1 = df.D_cum.iloc[j - 1], df.D_cum.iloc[j]
        f = 0 if c1 == c0 else (x - c0) / (c1 - c0)
        cyc = df.cycle.iloc[j - 1] + f * (df.cycle.iloc[j] - df.cycle.iloc[j - 1])
        return cyc / 540000 * 100

    print("t50 %.2f  t85 %.2f  states %d  D_total %.3f"
          % (t_at(.50), t_at(.85), len(df), tot))
    df.to_csv(os.path.join(OUT, "l2_pzt_L2-07.csv"), index=False)
    print("saved")


if __name__ == "__main__":
    main()
