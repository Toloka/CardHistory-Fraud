"""Turn the benchmark's parquet files into the arrays the transformer models and the XGBoost baseline read.

    python models/prepare.py --data dataset/ --out arrays/

Writes, in the dataset's row order (row_id = position: cards one after another, each card's transactions in time order):
  tokens.npy       [N, 14] int16: one id per field (amount bin, chip, MCC, merchant, city, state, errors, hour, day of week, gap
                   since the card's previous transaction, and 4 "first time on this card" flags) from vocab.json
  numf.npy         [N, 4] float16: signed log amount, log gap, hour sine and cosine, standardized on the rows before 2018
  card.npy, ts.npy, amount.npy, label.npy, scenario.npy, foreign.npy, card_starts.npy, meta.json
The vocabulary (vocab.json) was fitted on the rows before 2018 when the models were trained. The released weights need exactly
these ids, so the vocabulary is shipped rather than refitted:
  categorical fields: values seen at least 20 times before 2018 get their own id (2 + position in the list); every other value
                      shares one of a few hashed ids (crc32 of the value); 0 = padding
  amount, gap:        bins of log(1 + |amount|) (negative amounts get their own bins) and of log(1 + seconds since the card's
                      previous transaction) (1 = the card's first transaction)
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_COLUMN = {"chip": "use_chip", "mcc": "mcc", "merchant": "merchant_id", "city": "merchant_city", "state": "merchant_state",
                "errors": "errors"}


def hash_bucket(value: str, n: int) -> int:
    return zlib.crc32(value.encode()) % n


def dataset_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "data", "*.parquet"))) or sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {folder}/data or {folder}")
    return files


def as_codes(col):
    """A column as (integer codes, the distinct values as strings)."""
    if pa.types.is_dictionary(col.type):
        col = col.unify_dictionaries() if isinstance(col, pa.ChunkedArray) else col
        values = [str(x) for x in col.chunk(0).dictionary.to_pylist()] if col.num_chunks else []
        codes = np.concatenate([c.indices.to_numpy(zero_copy_only=False) for c in col.chunks]).astype(np.int64)
        return codes, values
    arr = col.to_numpy()
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64), [str(x) for x in uniq.tolist()]


def first_seen(card, codes, n_values):
    """1 where the card sees this value for the first time (rows are in card, time order)."""
    key = card * (n_values + 2) + codes + 1
    _, fi = np.unique(key, return_index=True)
    flag = np.zeros(len(card), np.int32); flag[fi] = 1
    return flag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the dataset folder (with data/part-*.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=os.path.join(HERE, "vocab.json"))
    a = ap.parse_args()
    V = json.load(open(a.vocab))
    os.makedirs(a.out, exist_ok=True)
    t = ds.dataset(dataset_files(a.data), format="parquet").to_table(
        columns=["row_id", "card", "timestamp", "amount", "is_fraud", "scenario_id"] + list(FIELD_COLUMN.values()))
    for name in t.column_names:
        assert t[name].null_count == 0, f"{name} has missing values"
    assert np.array_equal(t["row_id"].to_numpy(), np.arange(t.num_rows)), "rows must be in the dataset's order (row_id 0..N-1)"
    card = t["card"].to_numpy().astype(np.int64)
    ts = pc.cast(pc.cast(t["timestamp"], pa.timestamp("s")), pa.int64()).to_numpy().astype(np.int64)   # seconds since 1970, UTC
    amount = t["amount"].to_numpy().astype(np.float32)            # the generator's float32 amounts (the dollars-and-cents values round trip)
    label = t["is_fraud"].to_numpy().astype(np.int8); scenario = t["scenario_id"].to_numpy().astype(np.int8)
    codes, values = {}, {}
    for f, c in FIELD_COLUMN.items():
        codes[f], values[f] = as_codes(t[c])
    del t
    n = len(card)
    assert (np.diff(card) >= 0).all() and (np.diff(ts)[np.diff(card) == 0] >= 0).all(), "rows must be sorted by card, then time"
    tr = ts < V["t_val"]
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]])
    first = np.zeros(n, bool); first[starts] = True

    tokens = np.zeros((n, len(V["fields"])), np.int16)
    for j, (f, kind) in enumerate(zip(V["fields"], V["kinds"])):
        if kind == "amount":
            ae = np.asarray(V["decode"]["amt"]); nb = len(ae) + 1
            tok = 2 + np.digitize(np.log1p(np.abs(amount)), ae) + nb * (amount < 0)
        elif kind == "cat":
            kept = {v: i for i, v in enumerate(V["decode"][f])}; nh = V["hashed_buckets"][f]
            unknown = [v for v in values[f] if v not in kept]
            assert nh or not unknown, f"{f}: values outside the vocabulary and no hashed ids: {unknown[:5]}"
            per_value = np.array([2 + kept[v] if v in kept else 2 + len(kept) + hash_bucket(v, nh) for v in values[f]], np.int64)
            tok = per_value[codes[f]]
        elif f == "hour":
            tok = 2 + (ts // 3600) % 24
        elif f == "dow":
            tok = 2 + ((ts // 86400) + 3) % 7                    # 1 January 1970 was a Thursday: Monday = 2
        elif f == "gap":
            gap = np.r_[0, np.diff(ts)].astype(np.float64); gap[first] = -1
            ge = np.asarray(V["decode"]["gap"])
            tok = np.where(gap < 0, 1, 2 + np.digitize(np.log1p(np.clip(gap, 0, None)), ge))
        elif kind == "flag":
            src = f[len("new_"):]
            tok = 2 + first_seen(card, codes[src], len(values[src]))
        else:
            raise ValueError(f"unknown field {f}")
        assert tok.min() >= 1 and tok.max() < V["vocab_sizes"][j], f"{f}: ids outside the vocabulary"
        tokens[:, j] = tok
    np.save(os.path.join(a.out, "tokens.npy"), tokens); del tokens

    # numeric side inputs: signed log amount, log gap, hour of day as sine / cosine, standardized on the rows before 2018
    gap = np.r_[0, np.diff(ts)].astype(np.float64); gap[first] = 0
    hr = (ts % 86400) / 86400 * 2 * np.pi
    numf = np.zeros((n, 4), np.float16)
    columns = (np.sign(amount) * np.log1p(np.abs(amount)), np.log1p(np.clip(gap, 0, None)), np.sin(hr), np.cos(hr))
    for i, v in enumerate(columns):
        v = np.asarray(v, np.float64)
        mu, sd = v[tr].mean(), v[tr].std() + 1e-6
        numf[:, i] = np.clip((v - mu) / sd, -20, 20)
    np.save(os.path.join(a.out, "numf.npy"), numf); del numf, columns, gap, hr

    foreign = np.array([len(s) > 2 for s in values["state"]])[codes["state"]].astype(np.int8)   # a country name instead of a US state code
    for k, v in dict(card=card, ts=ts, amount=amount, label=label, scenario=scenario, foreign=foreign,
                     card_starts=starts.astype(np.int64)).items():
        np.save(os.path.join(a.out, f"{k}.npy"), v)
    meta = dict(fields=V["fields"], kinds=V["kinds"], vocab_sizes=V["vocab_sizes"], decode=V["decode"], t_val=V["t_val"],
                t_test=V["t_test"], next_field=V["next_field"], amt_field="amt", id_fields=V["id_fields"],
                flag_fields=V["flag_fields"], num_features=V["num_features"], num_groups={"time": V["num_features"]},
                n=int(n), n_cards=int(len(starts)))
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {n:,} rows of {len(starts):,} cards to {a.out}")


if __name__ == "__main__":
    main()
