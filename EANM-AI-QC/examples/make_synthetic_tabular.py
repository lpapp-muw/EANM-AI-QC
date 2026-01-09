#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)

out_dir = Path(__file__).resolve().parents[1] / "demo_data" / "tabular"
out_dir.mkdir(parents=True, exist_ok=True)

n_train = 80
n_infer = 10
n_features = 8

X = rng.normal(size=(n_train, n_features))
w = rng.normal(size=(n_features,))
logits = X @ w
p = 1 / (1 + np.exp(-logits))
y = (p > 0.5).astype(int)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
df["label"] = y
df.to_csv(out_dir / "synth_train.csv", index=False)

X2 = rng.normal(size=(n_infer, n_features))
df2 = pd.DataFrame(X2, columns=[f"f{i}" for i in range(n_features)])
df2.to_csv(out_dir / "synth_infer.csv", index=False)

print(f"Wrote: {out_dir/'synth_train.csv'}")
print(f"Wrote: {out_dir/'synth_infer.csv'}")
