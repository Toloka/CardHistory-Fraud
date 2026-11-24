"""The 46 features of the XGBoost baseline ("XGBoost + geo"), computed per card from the arrays of models/prepare.py.

Every feature of a transaction uses only the card's earlier transactions plus the transaction itself:
  the transaction:  the 10 field ids the transformers read (amount bin, chip, MCC, merchant, city, state, errors, hour, day of
                    week, gap bin), the amount, signed log amount
  card history:     mean / std of the last 10 log amounts and the amount's z-score against them, log gap to the previous
                    transaction, first time on this card for the merchant / city / MCC, same MCC as the previous transaction,
                    transactions in the last 1 h / 24 h
  windows:          declines, small card-not-present payments (online, under $5), new merchants, new cities in the last 7 / 14 /
                    30 days; transactions in the last 7 days and the share at new merchants; days since the last decline and the
                    last small online payment; amount against the largest of the last 20
  places:           the card's home cities (at least 10% of its earlier transactions, once it has 20); transactions at home /
                    elsewhere / both and city switches in the last 24 h / 7 days; whether this transaction is away from home;
                    hours since the previous transaction when the city changed; away within 24 h of a transaction at home
"No decline" is the most frequent errors id among the training rows (fitted once, on training rows only).
"""
import numpy as np
import pandas as pd


def fit_error_none_token(d):
    column = d.meta["fields"].index("errors")
    tokens = np.asarray(d.tokens_all[d.split == 0, column], np.int64)
    if not len(tokens):
        raise ValueError("training rows are needed to fit the no-decline id")
    return int(np.bincount(tokens).argmax())


def features(d, rows=None, error_none=None, chunk_rows=1_500_000):
    """A float32 DataFrame with one row per entry of `rows` (default: every row). `d` is a models/data.py TxnData."""
    if error_none is None:
        error_none = fit_error_none_token(d)
    rows = np.arange(len(d.card)) if rows is None else np.asarray(rows)
    starts = np.r_[np.unique(d.card_start), len(d.card)]
    out, lo = [], 0
    while lo < len(d.card):                                    # chunks of whole cards (every feature is per card)
        hi = starts[np.searchsorted(starts, lo + chunk_rows, side="left")] if lo + chunk_rows < len(d.card) else len(d.card)
        sel = rows[(rows >= lo) & (rows < hi)]
        if len(sel):
            f = _chunk(d, lo, hi, error_none)
            out.append((sel, f.iloc[sel - lo]))
        lo = hi
    order = np.concatenate([o[0] for o in out])
    X = pd.concat([o[1] for o in out]); X.index = order
    return X.loc[rows].reset_index(drop=True)


def _chunk(d, lo, hi, error_none):
    m = d.meta
    t = np.asarray(d.tokens_all[lo:hi])
    df = pd.DataFrame({f: t[:, i] for i, (f, k) in enumerate(zip(m["fields"], m["kinds"])) if k != "flag"})
    df["amount"] = d.amount[lo:hi]
    df["card"] = d.card[lo:hi]; ts = d.ts[lo:hi]
    df["lamt"] = np.log1p(np.abs(df.amount)) * np.sign(df.amount)
    g = df.groupby("card", sort=False)
    prev = g["lamt"].shift(1)
    df["roll_mean10"] = prev.groupby(df.card).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    df["roll_std10"] = prev.groupby(df.card).rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
    df["amt_z"] = (df.lamt - df.roll_mean10) / (df.roll_std10 + 1e-3)
    df["gap_s"] = np.log1p(pd.Series(ts).groupby(df.card.values).diff().fillna(-1).clip(lower=0).values)
    for c in list(m["id_fields"]) + [m["next_field"]]:            # merchant, city, MCC
        df[f"new_{c}"] = (df.groupby(["card", c]).cumcount() == 0).astype(np.int8)
    df[f"same_{m['next_field']}_as_prev"] = (df[m["next_field"]] == g[m["next_field"]].shift(1)).astype(np.int8)
    key = d._key[lo:hi]
    df["n_24h"] = np.arange(len(df)) - np.searchsorted(key, key - 86400, side="left")
    df["n_1h"] = np.arange(len(df)) - np.searchsorted(key, key - 3600, side="left")
    _windowed(d, df, lo, hi, ts, key, error_none)
    _places(d, df, lo, hi, ts, key)
    return df.drop(columns=["card"]).astype(np.float32)


def _window_count(flag, W, key, cs, ar):
    """For each row: rows of the same card in the W seconds before it that have `flag` (the row itself excluded)."""
    C = np.r_[0, np.cumsum(flag.astype(np.int64))]
    j = np.maximum(np.searchsorted(key, key - W, side="left"), cs)
    return C[ar] - C[j]


def _windowed(d, df, lo, hi, ts, key, error_none):
    m = d.meta; n = hi - lo; ar = np.arange(n)
    cs = d.card_start[lo:hi] - lo                                # the card's first row, local index
    err = df["errors"].values
    online_tok = 2 + m["decode"]["chip"].index("Online Transaction") if "Online Transaction" in m["decode"]["chip"] else -1
    flags = {"decl": (err != error_none), "tiny_cnp": (df["chip"].values == online_tok) & (np.abs(df["amount"].values) < 5),
             "new_merch": df["new_merchant"].values > 0, "new_city": df["new_city"].values > 0}
    D = 86400
    for nm, wins in (("decl", (7, 30)), ("tiny_cnp", (7, 30)), ("new_merch", (7, 14)), ("new_city", (7, 14))):
        for w in wins:
            df[f"{nm}_{w}d"] = _window_count(flags[nm], w * D, key, cs, ar)
    n7 = _window_count(np.ones(n, bool), 7 * D, key, cs, ar)
    df["n_7d"] = n7; df["share_new_merch_7d"] = df["new_merch_7d"] / np.maximum(n7, 1)
    for nm in ("decl", "tiny_cnp"):
        last = np.maximum.accumulate(np.where(flags[nm], ar, -1)); prev = np.r_[-1, last[:-1]]
        ok = prev >= cs
        df[f"days_since_{nm}"] = np.where(ok, (ts - ts[np.maximum(prev, 0)]) / D, 999.0)
    a = np.abs(df["amount"].values); mx = np.zeros(n)
    for k in range(1, 21):
        sh = np.r_[np.zeros(k), a[:-k]]
        mx = np.where(ar - k >= cs, np.maximum(mx, sh), mx)
    df["amt_vs_max20"] = a / np.maximum(mx, 1.0)


def _places(d, df, lo, hi, ts, key):
    n = hi - lo; ar = np.arange(n); cs = d.card_start[lo:hi] - lo
    city = df["city"].values.astype(np.int64)
    prior_city = df.groupby(["card", "city"]).cumcount().values      # earlier transactions of this card in this city
    prior_all = ar - cs                                              # earlier transactions of this card
    home = (prior_all >= 20) & (prior_city >= 0.10 * prior_all)      # the transaction's city is one of the card's home cities
    known = prior_all >= 20
    nonhome = known & ~home
    prev_city = np.r_[-1, city[:-1]]; first = ar == cs
    switch = (~first) & (city != prev_city)
    D = 86400
    for W, nm in ((D, "24h"), (7 * D, "7d")):
        h_, nh_ = _window_count(home, W, key, cs, ar), _window_count(nonhome, W, key, cs, ar)
        df[f"home_{nm}"] = h_; df[f"nonhome_{nm}"] = nh_
        df[f"home_and_nonhome_{nm}"] = np.minimum(h_, nh_)              # > 0: at home and elsewhere in the same window
        df[f"city_switches_{nm}"] = _window_count(switch, W, key, cs, ar)
    df["cur_nonhome"] = nonhome.astype(np.int8)
    gap_h = np.r_[0.0, np.diff(ts.astype(np.float64))] / 3600.0
    df["hours_since_prev_if_city_changed"] = np.where(switch, gap_h, 999.0)
    prev_home = np.r_[False, home[:-1]] & ~first                     # the previous transaction (within 24 h) was at home
    df["away_after_home_24h"] = (nonhome & prev_home & (gap_h < 24)).astype(np.int8)
