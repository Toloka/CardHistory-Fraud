"""Score the benchmark's rows with the XGBoost baseline and write a file that evaluate.py reads.

    python baselines/xgboost/predict.py --arrays arrays/ --model weights/xgboost-geo.json --split test --out scores/xgboost.parquet

The score is XGBoost's margin (log-odds; higher = more suspicious). Features: baselines/xgboost/features.py.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models")); sys.path.insert(0, HERE)   # this folder first (its own train.py)
from data import TxnData  # noqa: E402
from features import features, fit_error_none_token  # noqa: E402

SPLITS = {"train": 0, "validation": 1, "test": 2}


def predict(model, X):
    return np.asarray(model.predict(X, output_margin=True), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=list(SPLITS)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xgboost as xgb
    d = TxnData(a.arrays, drop_fields=[], num_groups=())
    rows = np.flatnonzero(d.split == SPLITS[a.split])
    model = xgb.XGBClassifier(); model.load_model(a.model)
    s = predict(model, features(d, rows, error_none=fit_error_none_token(d)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.table({"row_id": pa.array(rows.astype(np.int64)), "score": pa.array(s)}), a.out)
    print(f"scored {len(rows):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
