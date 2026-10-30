# -*- coding: utf-8 -*-
"""Adapter v0 (selected): RandomForest on PHM2019 waveform features.

Why not the MLP? `adapter_v0.py` is kept as a negative result: the small MLP
overfits and its LOSO MAE (~4.0 mm) is worse than the global-mean baseline.
A shallow RandomForest is the honest v0 choice for 92 rows.

Split: leave-one-specimen-out (8 folds), no test specimen in training.
Features: 36 waveform features; target: crack_mm.

Usage:
    python llm/adapter_v0_rf.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "phmdc", "results", "step1_features.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

DROP = {"specimen", "cycle", "rep", "crack_mm"}
CLIP = (0.0, 8.0)  # physical range of crack length in this dataset


def main() -> int:
    df = pd.read_csv(DATA).dropna(subset=["crack_mm"]).reset_index(drop=True)
    feats = [c for c in df.columns if c not in DROP]
    rows = []
    for test in sorted(df["specimen"].unique()):
        tr = df[df["specimen"] != test]
        te = df[df["specimen"] == test]
        sc = StandardScaler().fit(tr[feats])
        x_tr, x_te = sc.transform(tr[feats]), sc.transform(te[feats])
        model = RandomForestRegressor(
            n_estimators=400, max_depth=4, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        model.fit(x_tr, tr["crack_mm"].to_numpy())
        pred = np.clip(model.predict(x_te), *CLIP)
        mae = float(np.abs(pred - te["crack_mm"].to_numpy()).mean())
        print(f"  held-out {test}: n={len(te)} MAE={mae:.3f}")
        for i, (_, row) in enumerate(te.iterrows()):
            rows.append({
                "specimen": test,
                "cycle": int(row["cycle"]),
                "y_true": float(row["crack_mm"]),
                "y_pred": float(pred[i]),
                "abs_err": float(abs(pred[i] - row["crack_mm"])),
            })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "adapter_v0_rf_predictions.csv"), index=False)
    mae = float(res["abs_err"].mean())
    rmse = float(np.sqrt(((res["y_pred"] - res["y_true"]) ** 2).mean()))
    baseline = float(np.abs(res["y_true"] - res["y_true"].mean()).mean())
    summary = {
        "dataset": "PHM2019 step1_features",
        "model": "RandomForest(max_depth=4, n=400)",
        "split": "leave-one-specimen-out",
        "rows": int(len(res)),
        "features": len(feats),
        "mae_mm": mae,
        "rmse_mm": rmse,
        "global_mean_mae_mm": baseline,
        "n_specimens": int(res["specimen"].nunique()),
    }
    with open(os.path.join(OUT, "adapter_v0_rf_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
