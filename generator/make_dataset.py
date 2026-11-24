"""Build the benchmark's files from IBM's TabFormer CSV.

    python generator/make_dataset.py --ibm-csv card_transaction.v1.csv --out dataset/

IBM's file: https://github.com/IBM/TabFormer (data/credit_card/transactions.tgz, which holds card_transaction.v1.csv).
The build takes about 10 minutes and under 8 GB of memory. It writes:
  dataset/data/part-NN.parquet   every transaction, in the order the models read them (card, then time), split into parts
  dataset/episodes.parquet       one row per added fraud episode: its card, kind, and when the card was compromised / stolen
  dataset/reissues.parquet       cards the bank reissued after a breach, before any misuse
  dataset/FINGERPRINT.json       row counts per split and scenario, library versions, sha256 of the core arrays and of every file
A rebuild is the same dataset if its sha256_core_arrays equal the published ones. They depend on the library versions
(numpy's random streams can change between versions): the published build used the versions listed in FINGERPRINT.json.
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build as B  # noqa: E402

ROWS_PER_PART = 2_000_000
EPISODE_KINDS = {"burst": "card-testing burst", "stolen": "stolen card", "slow": "counterfeit card", "compromised": "breached merchant",
                 "distributed": "distributed testing", "wave": "network testing wave", "ibm": "IBM's original fraud"}


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.view(np.uint8).reshape(-1) if a.size else b"").hexdigest() + f":{a.dtype}:{a.shape}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ibm-csv", required=True, help="IBM's card_transaction.v1.csv")
    ap.add_argument("--out", required=True, help="folder for the dataset files")
    ap.add_argument("--work", default="", help="folder for a cached copy of IBM's data (default: <out>_work, next to <out>)")
    a = ap.parse_args()
    out = os.path.abspath(a.out); work = os.path.abspath(a.work or out.rstrip("/") + "_work")
    csv = os.path.abspath(a.ibm_csv)
    os.makedirs(os.path.join(out, "data"), exist_ok=True); os.makedirs(os.path.join(work, B.RAW), exist_ok=True)
    link = os.path.join(work, B.RAW, "card_transaction.v1.csv")
    if not os.path.exists(link):
        os.symlink(csv, link)
    os.chdir(work)                                     # the generator reads IBM's file from data/raw/ under the working folder

    raw = B.load_benchmark()
    n = len(raw["card"]); cat, dec = raw["cat"], raw["decode"]
    origin = raw["origin"].astype(np.int8)
    o = np.lexsort((raw["tie"], raw["ts"], raw["card"]))   # card, time, then the fixed order of same-minute rows
    card, ts, amount, label, sc = (raw[k][o] for k in ("card", "ts", "amount", "label", "scenario"))
    t_val, t_test = raw["info"]["t_val"], raw["info"]["t_test"]
    period = (ts >= t_val).astype(np.int8) + (ts >= t_test).astype(np.int8)
    # split: before 2018 training; January-June 2018 training for 80% of the cards and validation for the other 20%; from July 2018 test
    cards = np.unique(card)
    vc = np.random.default_rng(123).choice(cards, int(len(cards) * 0.2), replace=False)
    split = period.copy(); split[(split == 1) & ~np.isin(card, vc)] = 0
    names = [B.SCENARIOS[k] for k in sorted(B.SCENARIOS)]
    dict_col = lambda codes, values: pa.DictionaryArray.from_arrays(pa.array(np.asarray(codes, np.int32)), pa.array([str(x) for x in values]))
    tbl = pa.table({
        "row_id": pa.array(np.arange(n, dtype=np.int64)),
        "user": pa.array(card // 10), "card_index": pa.array((card % 10).astype(np.int8)), "card": pa.array(card),
        "timestamp": pa.array(ts.astype("datetime64[s]")), "amount": pa.array(amount.astype(np.float64).round(2)),
        "use_chip": dict_col(cat["chip"][o], dec["chip"]),
        "mcc": pa.array(np.asarray(dec["mcc"], np.int32)[cat["mcc"][o]]),
        "merchant_id": pa.array(np.asarray(dec["merchant"], np.int64)[cat["merchant"][o]]),
        "merchant_city": dict_col(cat["city"][o], dec["city"]), "merchant_state": dict_col(cat["state"][o], dec["state"]),
        "errors": dict_col(cat["errors"][o], dec["errors"]),
        "is_fraud": pa.array(label.astype(np.int8)), "scenario_id": pa.array(sc.astype(np.int8)),
        "scenario": dict_col(sc.astype(np.int32), names),
        "origin": dict_col(origin[o], B.ORIGINS),
        "split": dict_col(split, ["train", "validation", "test"]),
        "episode": pa.array(raw["episode"][o].astype(np.int32)),
        "is_probe": pa.array(raw["probe"][o].astype(bool)),
    })
    parts = []
    for i, s in enumerate(range(0, n, ROWS_PER_PART)):
        p = os.path.join(out, "data", f"part-{i:02d}.parquet")
        pq.write_table(tbl.slice(s, ROWS_PER_PART), p, compression="zstd", row_group_size=500_000)
        parts.append(p)
    E, R = raw["episodes"], raw["reissues"]
    p_ep, p_re = os.path.join(out, "episodes.parquet"), os.path.join(out, "reissues.parquet")
    pq.write_table(pa.table({"episode": pa.array(np.arange(len(E["card"]), dtype=np.int32)), "card": pa.array(E["card"]),
                             "kind": pa.array([EPISODE_KINDS[B.EP_KINDS[int(k)]] for k in E["kind"]]),
                             "compromised_at": pa.array(np.asarray(E["compromise"]).astype("datetime64[s]")),
                             "stolen_at": pa.array(np.where(np.asarray(E["theft"]) >= 0, np.asarray(E["theft"]), np.iinfo(np.int64).min)
                                                   .astype("datetime64[s]"), mask=np.asarray(E["theft"]) < 0)}), p_ep)
    pq.write_table(pa.table({"card": pa.array(R["card"]), "exposed_at": pa.array(R["exposure"].astype("datetime64[s]")),
                             "new_card_active_at": pa.array(R["activation"].astype("datetime64[s]"))}), p_re)
    oo = origin[o]
    per_split = {nm: dict(rows=int((split == i).sum()), frauds=int(label[split == i].sum()),
                          frauds_added=int(label[(split == i) & (oo == 2)].sum()), cards=int(len(np.unique(card[split == i]))))
                 for i, nm in enumerate(["train", "validation", "test"])}
    per_scenario = {B.SCENARIOS[int(k)]: dict(rows=int((sc == k).sum()), frauds=int(label[sc == k].sum()),
                                              test_rows=int(((sc == k) & (split == 2)).sum())) for k in np.unique(sc)}
    import polars, pyarrow
    fp = dict(rows=int(n), frauds=int(label.sum()), cards=int(len(cards)), per_split=per_split, per_scenario=per_scenario,
              origins={B.ORIGINS[g]: int((oo == g).sum()) for g in range(len(B.ORIGINS))},
              sha256_ibm_csv=sha_file(csv), sha256_city_geo=sha_file(B.CITY_GEO),
              sha256_generator=sha_file(os.path.join(HERE, "build.py")),
              sha256_core_arrays={k: sha_arr(v) for k, v in (("card", card), ("ts", ts), ("amount", amount), ("label", label), ("scenario", sc))},
              sha256_files={os.path.relpath(p, out): sha_file(p) for p in parts + [p_ep, p_re]},
              versions=dict(python=platform.python_version(), numpy=np.__version__, polars=polars.__version__, pyarrow=pyarrow.__version__),
              params=B.PARAMS, bank=B.BANK)
    with open(os.path.join(out, "FINGERPRINT.json"), "w") as f:
        json.dump(fp, f, indent=1, default=str)
    print(json.dumps({k: fp[k] for k in ("rows", "frauds", "cards", "per_split", "sha256_core_arrays")}, indent=1))


if __name__ == "__main__":
    main()
