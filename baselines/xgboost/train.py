"""Train the XGBoost baseline the way the released model was trained.

    python baselines/xgboost/train.py --arrays arrays/ --out weights/xgboost-geo-retrained.json [--device cuda]

Training sample: every training fraud + 400,000 training legit transactions (rng 0), the legit ones weighted back to their true
count. Early stopping (50 rounds, area under the precision-recall curve) on the validation sample of models/data.py
eval_rows(1, 20000, seed=1, n_neg_foreign=10000): every validation fraud + 20,000 legit at home + 10,000 legit abroad, weighted.
At most 800 trees, depth 6, learning rate 0.05, 80% of rows and of columns per tree, histogram method.
The released model (weights/xgboost-geo.json) was trained on a GPU; XGBoost's CPU and GPU versions sample rows and columns
differently, so a CPU retrain gives a similar, not an identical, model.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models")); sys.path.insert(0, HERE)   # this folder first (its own train.py)
from data import TxnData  # noqa: E402
from features import features, fit_error_none_token  # noqa: E402

PARAMS = dict(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              eval_metric="aucpr", early_stopping_rounds=50)


def training_rows(d):
    """(rows, weights) of the training sample and (rows, weights) of the early-stopping sample."""
    rng = np.random.default_rng(0)
    train = np.flatnonzero(d.split == 0)
    pos, neg = train[d.label[train] == 1], train[d.label[train] == 0]
    negs = rng.choice(neg, min(400000, len(neg)), replace=False)
    rows = np.r_[pos, negs]
    weight = np.where(d.label[rows] == 1, 1.0, len(neg) / len(negs))
    vr, vw = d.eval_rows(1, 20000, seed=1, n_neg_foreign=10000)
    return rows, weight, vr, vw


def fit(xt, yt, wt, xv, yv, wv, device="cpu", n_jobs=8):
    import xgboost as xgb
    model = xgb.XGBClassifier(**PARAMS, n_jobs=n_jobs, device=device)
    model.fit(xt, yt, sample_weight=wt, eval_set=[(xv, yv)], sample_weight_eval_set=[wv], verbose=False)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    t0 = time.time()
    d = TxnData(a.arrays, drop_fields=[], num_groups=())
    rows, weight, vr, vw = training_rows(d)
    need = np.unique(np.r_[rows, vr])
    X = features(d, need, error_none=fit_error_none_token(d))
    at = lambda r: X.iloc[np.searchsorted(need, r)].reset_index(drop=True)
    print(f"features of {len(need):,} rows in {time.time() - t0:.0f}s", flush=True)
    model = fit(at(rows), d.label[rows], weight, at(vr), d.label[vr], vw, a.device, a.n_jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    model.save_model(a.out)
    print(f"{model.best_iteration + 1} trees -> {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
