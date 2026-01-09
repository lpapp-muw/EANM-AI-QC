#!/usr/bin/env python3
"""Build demo_data/tabular/real_train.csv and real_infer.csv from FDB.csv + LDB.csv.

Files:
  demo_data/tabular/raw/FDB.csv  (features; must include 'Key' column)
  demo_data/tabular/raw/LDB.csv  (labels; must include 'Key' and one label column)

Output:
  demo_data/tabular/real_train.csv
  demo_data/tabular/real_infer.csv
  demo_data/tabular/real_feature_map.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def read_csv_sniff(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdb", default="demo_data/tabular/raw/FDB.csv")
    ap.add_argument("--ldb", default="demo_data/tabular/raw/LDB.csv")
    ap.add_argument("--key", default="Key")
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()

    fdb_path = Path(args.fdb)
    ldb_path = Path(args.ldb)
    if not fdb_path.exists():
        raise SystemExit(f"Missing {fdb_path}")
    if not ldb_path.exists():
        raise SystemExit(f"Missing {ldb_path}")

    fdb = read_csv_sniff(fdb_path)
    ldb = read_csv_sniff(ldb_path)

    if args.key not in fdb.columns or args.key not in ldb.columns:
        raise SystemExit(f"Key column '{args.key}' must exist in both files.")

    # pick label column
    label_col = args.label_col
    if label_col is None:
        # choose the non-key column
        non_key = [c for c in ldb.columns if c != args.key]
        if len(non_key) != 1:
            raise SystemExit("LDB must have exactly one non-key label column or pass --label-col.")
        label_col = non_key[0]

    merged = fdb.merge(ldb[[args.key, label_col]], on=args.key, how="inner")
    if merged.shape[0] == 0:
        raise SystemExit("Merge produced 0 rows; Key mismatch.")

    y = merged[label_col]
    X = merged.drop(columns=[label_col, args.key])

    Xn = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    feature_cols = list(Xn.columns)
    fcols = [f"f{i}" for i in range(len(feature_cols))]
    Xn.columns = fcols

    out_dir = Path("demo_data/tabular")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = Xn.copy()
    train_df["label"] = y.to_numpy()
    train_df.to_csv(out_dir/"real_train.csv", index=False)

    Xn.to_csv(out_dir/"real_infer.csv", index=False)

    pd.DataFrame({"f_col": fcols, "original_feature": feature_cols}).to_csv(out_dir/"real_feature_map.csv", index=False)

    print(f"Wrote {out_dir/'real_train.csv'} rows={train_df.shape[0]} features={Xn.shape[1]}")
    print(f"Wrote {out_dir/'real_infer.csv'} rows={Xn.shape[0]} features={Xn.shape[1]}")
    print(f"Wrote {out_dir/'real_feature_map.csv'}")

if __name__ == "__main__":
    main()
