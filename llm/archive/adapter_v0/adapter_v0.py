# -*- coding: utf-8 -*-
"""Adapter encoder v0 on PHM2019 waveform features.

Input : 39 hand-crafted waveform features per measurement.
Target: crack_mm.
Split : leave-one-specimen-out (8 folds); no test specimen in training.

This is a small encoder experiment, not a foundation-model fine-tune. On an
8 GB laptop GPU it should finish in under a minute.

Usage:
    python llm/adapter_v0.py
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "phmdc", "results", "step1_features.csv")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

DROP = {"specimen", "cycle", "rep", "crack_mm"}
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Encoder(nn.Module):
    """39 -> 64 -> 16 (embedding) -> 1; the 16-D vector is the adapter output."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 16), nn.ReLU(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.head(z).squeeze(-1), z


def train_fold(x_tr, y_tr, x_te, y_te, device, epochs: int = 350):
    model = Encoder(x_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    xb = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yb = torch.tensor(y_tr, dtype=torch.float32, device=device)
    xt = torch.tensor(x_te, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred, z = model(xt)
    return pred.cpu().numpy(), z.cpu().numpy(), model


def main() -> int:
    set_seed()
    df = pd.read_csv(DATA)
    feats = [c for c in df.columns if c not in DROP]
    df = df.dropna(subset=["crack_mm"]).reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"rows={len(df)} features={len(feats)} device={device}")

    specimens = sorted(df["specimen"].unique())
    all_rows = []
    for test in specimens:
        tr = df[df["specimen"] != test]
        te = df[df["specimen"] == test]
        x_tr = tr[feats].to_numpy(dtype=np.float32)
        x_te = te[feats].to_numpy(dtype=np.float32)
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
        x_tr = (x_tr - mu) / sd
        x_te = (x_te - mu) / sd
        y_tr = tr["crack_mm"].to_numpy(dtype=np.float32)
        y_te = te["crack_mm"].to_numpy(dtype=np.float32)
        pred, z, _ = train_fold(x_tr, y_tr, x_te, y_te, device)
        for i, (_, row) in enumerate(te.iterrows()):
            all_rows.append({
                "specimen": test,
                "cycle": int(row["cycle"]),
                "y_true": float(y_te[i]),
                "y_pred": float(pred[i]),
                "abs_err": float(abs(pred[i] - y_te[i])),
                "z0": float(z[i, 0]),
            })
        print(f"  held-out {test}: n={len(te)} MAE={np.abs(pred-y_te).mean():.3f}")

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT, "adapter_v0_predictions.csv"), index=False)
    mae = float(res["abs_err"].mean())
    rmse = float(np.sqrt(((res["y_pred"] - res["y_true"]) ** 2).mean()))
    # baseline: global training mean cannot be computed per fold here exactly;
    # use overall mean as a transparent, slightly optimistic reference.
    base = float(np.abs(res["y_true"] - res["y_true"].mean()).mean())
    summary = {
        "dataset": "PHM2019 step1_features",
        "rows": int(len(res)),
        "features": len(feats),
        "device": device,
        "split": "leave-one-specimen-out",
        "mae_mm": mae,
        "rmse_mm": rmse,
        "global_mean_mae_mm": base,
        "n_specimens": len(specimens),
    }
    with open(os.path.join(OUT, "adapter_v0_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
