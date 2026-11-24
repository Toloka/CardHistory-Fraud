"""Score a fraud model on the benchmark's test period, exactly as in our report.

The metric is the share of fraud caught at 5 false alarms per caught fraud ("recall at 5:1"). Each model gets its own alert line:
sort the test transactions by the model's score, highest first, and walk down the list for as long as the alerts contain at most
5 legit transactions for every caught fraud. Transactions with equal scores are never split: the alert line takes all of them or
none of them, so the result does not depend on the order of ties.

Views (the same as in the report):
  headline      the fraud we added, at 5 false alarms per caught fraud. IBM's original fraud is left out (one field gives it away)
                and so are the attempts on a blocked card (the bank has already stopped them).
  no_probes     the headline without the fraudster's small test payments of distributed testing and network waves.
  after_block   the headline plus the attempts on a blocked card.
  rarer         the headline at 2 false alarms per caught fraud: what 5 would mean if fraud were 2.5x rarer.
  fixed_line    optional (needs --dev-scores): the alert line is fixed on the validation set at 5 false alarms per caught fraud
                and applied unchanged to the test period, as in deployment.
  by pattern    at the headline alert line: the share of each fraud pattern caught and of each legit look-alike flagged.

Legit transactions are every row with is_fraud = 0 in the test period (look-alikes included).

Usage:
  python evaluate.py --data <dataset folder or .parquet> --scores run1.parquet [run2.parquet ...] [--dev-scores dev1.parquet ...]
A scores file has one row per transaction: row_id and score (higher = more suspicious). Parquet, CSV or NPZ (arrays rows, scores).
It must cover every test-period row (and, for --dev-scores, every validation row). With several files (e.g. three training runs)
each is reported, plus their mean, which is how the report states every transformer result.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

# fraud patterns (scenario_id) in each view; see the dataset card for the names
HEADLINE = (3, 4, 5, 8, 9, 10, 11, 12, 14)
PROBES = (11, 12)          # small test payments of distributed testing and network waves
BLOCKED = (13,)            # the fraudster's attempts on a blocked card
VIEWS = {
    "headline": dict(frauds=HEADLINE, ratio=5),
    "no_probes": dict(frauds=tuple(s for s in HEADLINE if s not in PROBES), ratio=5),
    "after_block": dict(frauds=HEADLINE + BLOCKED, ratio=5),
    "rarer": dict(frauds=HEADLINE, ratio=2),
}
NAMES = {0: "legit", 1: "IBM's original fraud", 2: "legit trip abroad", 3: "card-testing burst", 4: "stolen card",
         5: "breached merchant", 6: "legit domestic trip", 7: "legit new subscription", 8: "counterfeit card",
         9: "distributed testing: purchase", 10: "network testing wave: purchase", 11: "distributed testing: probe",
         12: "network testing wave: probe", 13: "attempt on a blocked card", 14: "fraudulent recurring charge"}


def recall_at_ratio(scores, labels, ratio=5):
    """Fraud caught (recall) at `ratio` false alarms per caught fraud, and the alert line (threshold).

    The alert line is the lowest score s such that flagging every transaction with score >= s gives at most `ratio` legit
    transactions per caught fraud. Equal scores are never split. Returns (recall, threshold); recall is 0 if no line works."""
    s, y = np.asarray(scores), np.asarray(labels)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be aligned 1-d arrays")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinite values")
    order = np.argsort(-s, kind="stable")
    so, yo = s[order], y[order].astype(np.float64)
    ends = np.flatnonzero(np.r_[so[1:] != so[:-1], True])     # last position of each group of equal scores
    tp = np.cumsum(yo)[ends]
    fp = np.cumsum(1.0 - yo)[ends]
    ok = np.flatnonzero((tp > 0) & (fp <= ratio * tp))
    if not len(ok):
        return 0.0, float(np.nextafter(s.max(), np.asarray(np.inf, dtype=s.dtype)))
    i = ok[-1]
    return float(tp[i] / tp[-1]), float(so[ends[i]])


def load_benchmark(path):
    """row_id, is_fraud, scenario_id of the test period and of the validation set, from the released parquet files."""
    if os.path.isdir(path):   # the dataset folder: its data/ parts (or parquet files directly in it)
        files = sorted(glob.glob(os.path.join(path, "data", "*.parquet"))) or sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]
    d = ds.dataset(files, format="parquet")
    cols = ["row_id", "is_fraud", "scenario_id", "split"]
    out = {}
    for split in ("test", "validation"):
        t = d.to_table(columns=cols, filter=pc.field("split") == split)
        o = np.argsort(t["row_id"].to_numpy(), kind="stable")
        out[split] = {k: t[k].to_numpy()[o] for k in ("row_id", "is_fraud", "scenario_id")}
    return out


def load_scores(path, rows):
    """Scores aligned to `rows` (sorted row ids); every row must be present exactly once."""
    if path.endswith(".npz"):
        z = np.load(path)
        r, s = np.asarray(z["rows"]), np.asarray(z["scores"])
    elif path.endswith(".csv"):
        a = np.genfromtxt(path, delimiter=",", names=True)
        r, s = a["row_id"].astype(np.int64), a["score"]
    else:
        t = ds.dataset(path, format="parquet").to_table(columns=["row_id", "score"])
        r, s = t["row_id"].to_numpy(), t["score"].to_numpy()
    o = np.argsort(r, kind="stable")
    r, s = r[o], s[o]
    if (np.diff(r) == 0).any():
        raise ValueError(f"{path}: duplicate row_id")
    pos = np.searchsorted(r, rows)
    if (pos >= len(r)).any() or not np.array_equal(r[np.minimum(pos, len(r) - 1)], rows):
        raise ValueError(f"{path}: does not score every row of the population ({len(rows):,} rows needed)")
    return s[pos]


def score_views(s, y, sc, dev=None):
    res = {}
    for name, v in VIEWS.items():
        keep = (y == 0) | ((y == 1) & np.isin(sc, v["frauds"]))
        rec, thr = recall_at_ratio(s[keep], y[keep], v["ratio"])
        res[name] = dict(recall=rec, threshold=thr, frauds=int(y[keep].sum()), rows=int(keep.sum()))
    keep = (y == 0) | ((y == 1) & np.isin(sc, HEADLINE))
    thr = res["headline"]["threshold"]
    flag = s >= thr
    res["by_pattern"] = {NAMES[int(k)]: dict(n=int((sc == k).sum()),
                                            share_flagged=float(flag[(sc == k) & keep].mean()) if ((sc == k) & keep).any() else None)
                         for k in np.unique(sc) if int(k) in NAMES and ((sc == k) & keep).any()}
    if dev is not None:   # alert line fixed on the validation set, applied to the test period
        ds_, dy, dsc = dev
        dkeep = (dy == 0) | ((dy == 1) & np.isin(dsc, HEADLINE))
        _, dthr = recall_at_ratio(ds_[dkeep], dy[dkeep], 5)
        f = s[keep] >= dthr
        tp, fp = float((f & (y[keep] == 1)).sum()), float((f & (y[keep] == 0)).sum())
        res["fixed_line"] = dict(threshold=dthr, recall=tp / float(y[keep].sum()), false_alarms_per_catch=fp / max(tp, 1.0))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the dataset folder (or its .parquet file)")
    ap.add_argument("--scores", nargs="+", required=True, help="test-period scores, one file per training run")
    ap.add_argument("--dev-scores", nargs="*", default=[], help="validation-set scores, same order as --scores (fixed alert line)")
    ap.add_argument("--json", default="", help="also write all numbers to this JSON file")
    a = ap.parse_args()
    if a.dev_scores and len(a.dev_scores) != len(a.scores):
        sys.exit("--dev-scores needs one file per --scores file")
    B = load_benchmark(a.data)
    T, V = B["test"], B["validation"]
    runs = []
    for i, f in enumerate(a.scores):
        s = load_scores(f, T["row_id"])
        dev = (load_scores(a.dev_scores[i], V["row_id"]), V["is_fraud"], V["scenario_id"]) if a.dev_scores else None
        runs.append(dict(file=f, **score_views(s, T["is_fraud"], T["scenario_id"], dev)))
    keys = ["headline", "no_probes", "after_block", "rarer"] + (["fixed_line"] if a.dev_scores else [])
    mean = {k: float(np.mean([r[k]["recall"] for r in runs])) for k in keys}
    print(f"Test period: {len(T['row_id']):,} transactions; frauds in the headline view: {runs[0]['headline']['frauds']:,}")
    print("fraud caught (%)      " + "  ".join(f"{k:>11s}" for k in keys))
    for r in runs:
        print(f"{os.path.basename(r['file'])[:20]:20s}  " + "  ".join(f"{100 * r[k]['recall']:11.1f}" for k in keys))
    if len(runs) > 1:
        print(f"{'mean of ' + str(len(runs)) + ' runs':20s}  " + "  ".join(f"{100 * mean[k]:11.1f}" for k in keys))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(runs=runs, mean=mean), fh, indent=1)


if __name__ == "__main__":
    main()
