"""Build the card-fraud benchmark from IBM's TabFormer card transactions.

IBM's file (card_transaction.v1.csv, 24.4M transactions of 2,000 simulated US cardholders, 1991-2020) is kept as it is, its own
fraud included. On top of it this module adds fraud patterns that only a card's history reveals, legit look-alikes that resemble
fraud, and a bank that reacts to fraud. Two steps:
  inject_scenarios  fraud patterns and legit look-alikes; every added fraud row carries its episode id;
  bank_reaction     the bank notices each fraud episode, blocks the card, declines the fraudster's next attempts and issues a new
                    card; also card-on-file merchants, cards reissued after a breach, and rare false declines of legit purchases.
Every setting is in PARAMS and BANK below, with what it means. All randomness comes from fixed seeds, so a build is the same
dataset when the library versions match (numpy's random streams can change between versions; see the README).

Raw format of a build (dict of numpy arrays, one entry per transaction):
  card int64 (user * 10 + card index), ts int64 (local time as seconds since 1970, no time zone), label int8 (1 = fraud),
  amount float32 ($, negative = refund), cat {field: int32 codes into decode[field]}, decode {field: list of str},
  scenario int8 (see SCENARIOS), foreign int8 (1 = outside the US), origin int8 (see ORIGINS), episode int32 (-1 = none),
  probe bool, info dict.
"""
import json, os, zlib
import numpy as np
import polars as pl

SCENARIOS = {0: "legit", 1: "IBM's original fraud", 2: "legit trip abroad", 3: "card-testing burst", 4: "stolen card",
             5: "breached merchant", 6: "legit domestic trip", 7: "legit new subscription", 8: "counterfeit card",
             9: "distributed testing: purchase", 10: "network testing wave: purchase", 11: "distributed testing: probe",
             12: "network testing wave: probe", 13: "attempt on a blocked card", 14: "fraudulent recurring charge"}
RAW = "data/raw"


def _codes(s: pl.Series):
    """Factorize a column -> (int32 codes, decode list). Nulls -> -1."""
    u = s.drop_nulls().unique().sort()
    mp = {v: i for i, v in enumerate(u.to_list())}
    codes = s.replace_strict(mp, default=-1, return_dtype=pl.Int32).to_numpy()
    return codes.astype(np.int32), [str(x) for x in u.to_list()]


def _epoch(date):
    return int(np.datetime64(date, "s").astype(np.int64))


# ---------------------------------------------------------------- IBM's TabFormer card transactions (24.4M)
TAB_STR = ["Use Chip", "Merchant City", "Merchant State", "Errors?"]


def _tabformer_std(csv=f"{RAW}/card_transaction.v1.csv", out=f"{RAW}/tabformer_std.parquet"):
    """CSV -> compact parquet (int codes instead of strings), one streaming pass with bounded memory."""
    if os.path.exists(out):
        return out
    import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.compute as pc, pyarrow.parquet as pq
    conv = pacsv.ConvertOptions(column_types={"Merchant Name": pa.int64(), "Zip": pa.float64(), "Errors?": pa.string(),
                                              "Merchant State": pa.string(), "Merchant City": pa.string()},
                                include_columns=["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
                                                 "Merchant Name", "Merchant City", "Merchant State", "MCC", "Errors?", "Is Fraud?"])
    rd = pacsv.open_csv(csv, read_options=pacsv.ReadOptions(block_size=64 << 20), convert_options=conv)
    dec = {c: [] for c in TAB_STR}; idx = {c: {} for c in TAB_STR}
    writer = None
    for batch in rd:
        cols = {n: batch.column(n) for n in batch.schema.names}
        tm = cols["Time"].to_numpy(zero_copy_only=False).astype("U5")
        hh = np.array([int(x[:2]) for x in tm], np.int64); mm = np.array([int(x[3:5]) for x in tm], np.int64)
        day = (np.array([np.datetime64(f"{y:04d}-{m:02d}-{dd:02d}") for y, m, dd in zip(
            cols["Year"].to_numpy(), cols["Month"].to_numpy(), cols["Day"].to_numpy())]).astype("datetime64[s]").astype(np.int64))
        out_cols = dict(
            card=cols["User"].to_numpy().astype(np.int64) * 10 + cols["Card"].to_numpy().astype(np.int64),
            ts=day + hh * 3600 + mm * 60,
            amount=np.array([float(x[1:]) if x[0] == "$" else float(x) for x in cols["Amount"].to_numpy(zero_copy_only=False)], np.float32),
            label=(cols["Is Fraud?"].to_numpy(zero_copy_only=False) == "Yes").astype(np.int8),
            mcc=cols["MCC"].to_numpy().astype(np.int32),
            merchant=cols["Merchant Name"].to_numpy().astype(np.int64))
        for c, nm in zip(TAB_STR, ("chip", "city", "state", "errors")):
            v = cols[c].fill_null("NA" if c != "Errors?" else "none").to_numpy(zero_copy_only=False)
            uniq, inv = np.unique(v, return_inverse=True)
            for u in uniq:
                if u not in idx[c]:
                    idx[c][u] = len(dec[c]); dec[c].append(str(u))
            out_cols[nm] = np.array([idx[c][u] for u in uniq], np.int32)[inv]
        tbl = pa.table(out_cols)
        if writer is None:
            writer = pq.ParquetWriter(out + ".tmp", tbl.schema)
        writer.write_table(tbl)
    writer.close()
    os.replace(out + ".tmp", out)
    json.dump(dec, open(out.replace(".parquet", "_decode.json"), "w"))
    return out


def load_tabformer():
    """IBM's transactions as they are (labels included), in the raw format."""
    p = _tabformer_std()
    dec = json.load(open(p.replace(".parquet", "_decode.json")))
    mx = int(os.environ.get("MAX_CARDS", "0"))   # quick software tests only: about the first mx cards (not the benchmark)
    df = pl.scan_parquet(p).filter(pl.col("card") < 10 * (mx // 3 + 1)).collect() if mx else pl.read_parquet(p)
    card = df["card"].to_numpy(); ts = df["ts"].to_numpy(); amount = df["amount"].to_numpy()
    label = df["label"].to_numpy().astype(np.int8)
    mcc_c, mcc_d = _codes(df["mcc"]); merch_c, merch_d = _codes(df["merchant"])
    cat = dict(chip=df["chip"].to_numpy().astype(np.int32), mcc=mcc_c, merchant=merch_c,
               city=df["city"].to_numpy().astype(np.int32), state=df["state"].to_numpy().astype(np.int32),
               errors=df["errors"].to_numpy().astype(np.int32))
    decode = dict(chip=dec["Use Chip"], mcc=mcc_d, merchant=merch_d, city=dec["Merchant City"],
                  state=dec["Merchant State"], errors=dec["Errors?"])
    del df
    st_foreign = np.array([len(s) > 2 for s in decode["state"]])
    foreign = st_foreign[cat["state"]].astype(np.int8)
    raw = dict(card=card, ts=ts, label=label, amount=amount, cat=cat, decode=decode, num={},
               scenario=label.copy(), foreign=foreign,
               info=dict(name="ibm", t_val=_epoch("2018-01-01"), t_test=_epoch("2018-07-01"),
                         next_field="mcc", id_fields=["merchant", "city"], novelty_fields=["merchant", "city", "state", "mcc"],
                         notes="IBM TabFormer synthetic card transactions, 2,000 users / 6,139 cards, 1991-2020."))
    return raw


def _mix64(*cols):
    """Deterministic 64-bit hash of a row's contents (splitmix64 chained over the columns). Used to order same-minute ties
    without looking at the label or at whether a row was injected."""
    h = np.zeros(len(cols[0]), np.uint64)
    with np.errstate(over="ignore"):
        for c in cols:
            x = (h ^ np.asarray(c).astype(np.int64).view(np.uint64)) + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = x ^ (x >> np.uint64(31))
    return h


def tie_rank(raw):
    """Order of same-card same-minute ties: a hash of the row's contents (never the label or injected / organic), then a fixed-seed
    random key for exact duplicates. Returned as one integer rank per row (the export sorts by card, time, rank)."""
    cat = raw["cat"]
    tie = _mix64(raw["card"], raw["ts"], np.round(raw["amount"].astype(np.float64) * 100), cat["merchant"], cat["mcc"], cat["city"],
                 cat["chip"], cat["errors"])
    rnd = np.random.default_rng(12345).permutation(len(raw["card"]))
    return np.lexsort((rnd, tie)).argsort().astype(np.int64)




# ---------------------------------------------------------------------------------------------------------------------------------
# merchant categories the added rows use
SUB_MCCS = (4899, 5815, 5816)              # subscriptions: cable / streaming, digital media, digital games
TEST_MCCS = (5192, 5045, 5193, 5815, 5816)  # test probes: small online shops (books and magazines, software, florists, digital media, games)



# ---------------------------------------------------------------------------------------------------------------------------------
# The settings of the added fraud, the look-alikes and the bank. Card-fraud practitioners reviewed an earlier build; the numbers
# they gave from their own cases are the basis of most settings below. Known simplifications (see the dataset card): fraud is more
# frequent than in a real portfolio, one purchase per breached card, testing waves of 20-60 cards, no cash-out categories IBM does
# not have (cash machines, crypto, online gift cards), no account events (contact changes), and IBM's base-data quirks are kept.
ADDED_FRAUD = (3, 4, 5, 8, 9, 10, 11, 12, 13, 14)   # every fraud row this module adds (IBM's own fraud, scenario 1, is not among them)
PROBES = (11, 12)          # the fraudster's test payments of distributed testing and of waves
AFTER_BLOCK = (13,)        # attempts on a blocked card number: a decline code gives most of them away (scored apart)
LOOK_ALIKES = (2, 6, 7)            # legit look-alikes: trips abroad, domestic trips, new subscriptions
BLOCK_CODES = ("Card Blocked,", "Do Not Honor,", "Suspected Fraud,", "Stolen Card,")   # the decline codes after a block
EP_KINDS = ("burst", "stolen", "slow", "compromised", "distributed", "wave", "ibm")
ACTIVE_KINDS = ("burst", "stolen", "slow")   # the fraudster is using the card when the bank blocks it
# stolen card: a small first purchase at a gas station or a small shop, then larger ones where goods resell
# (IBM has no jewellery shops; gift cards are sold at groceries, drugstores and department stores)
SMALL_SHOP_MCCS = (5541, 5499, 5411, 5912)
STOLEN_BIG = (("electronics", (5732, 5045, 5733, 5722), 0.30), ("stores", (5311, 5310, 5300), 0.30),
              ("gift_cards", (5411, 5912), 0.20), ("fuel", (5541,), 0.10), ("other", (5941, 5651, 5621, 5655, 5661, 5712, 5947), 0.10))
# online cash-out (practitioners: electronics, gift cards, digital goods and games, airline and travel, ride-hailing, money transfer;
# IBM has no online gift cards or crypto and almost no online food delivery: documented)
CASH_OUT_GROUPS = (("electronics", (5732, 5045, 5733), 0.20), ("digital", (5815, 5816), 0.15), ("travel", (4511, 4722), 0.15),
                   ("ride_hailing", (4121,), 0.08), ("transfer", (4829,), 0.07), ("gambling", (7801,), 0.05),
                   ("department", (5311,), 0.20), ("clothing", (5651, 5661), 0.10))
PARAMS = dict(
    # ---- door-to-door travel time (h) from the great-circle distance: by car below 400 km (0.5 h + km / 80), else by air (2 h
    # before + km / 800 + 1 h after, + 2 h for a connection above 3,000 km), times U(0.9, 1.15); a road trip always drives
    drive_below_km=400, drive_kmh=80, airport_before_h=2.0, flight_kmh=800, connection_km=3000, connection_h=2.0, airport_after_h=1.0,
    # ---- legit look-alikes: domestic trips and new subscriptions; road trips are booked ahead less often than flights; trips often
    # stay in one city and have days without a purchase; hotel and transport charges on trips; subscription amounts that change
    trip_abroad_rate=0.25, trip_domestic_rate=0.5, road_trip_share=0.4, road_trip_km=(150, 800), one_city_share=0.7,
    active_day_p=0.8, booking_fly_p=0.75, booking_road_p=0.25, hotel_p=0.65, hotel_night=(80.0, 220.0), hotel_night_abroad=(60.0, 200.0),
    taxi_p=0.5, taxi_rides=(1, 3), subscription_rate=0.3, sub_price=(0.60, 0.25, 0.15), sub_step=(1.05, 1.25), sub_vary=(0.85, 1.15),
    zero_auth_p=0.4,
    # ---- uneven fraud (some customers get fraud again and again): a card's fraud rates x Gamma(1, 1)
    risk_shape=1.0,
    # ---- fraud rates per card-year (times the card's risk); breaches and waves per year in the whole data
    burst_rate=0.02, stolen_rate=0.02, slow_rate=0.015, distributed_rate=0.02, waves_per_year=8, breaches_per_year=72,
    # ---- chip vs swipe in the years the card uses the chip (practitioners: counterfeit cards 60-90% swipe or fallback, as a cloned
    # card cannot produce the chip cryptogram; stolen cards 70-80% chip), scaled down while the card's own chip share is below 80%
    counterfeit_swipe=(0.60, 0.90), stolen_chip=(0.70, 0.80),
    # ---- test probes ($0.10-3, minutes apart, several merchants, ~60% declined; some are $0 authorizations; "do not honour" and
    # insufficient funds among the declines)
    probe_zero_p=0.15,
    probe_codes=(("Bad CVV,", .30), ("Bad Card Number,", .20), ("Bad Expiration,", .20), ("Insufficient Balance,", .15), ("Do Not Honor,", .15)),
    burst_probe_amount=(0.10, 3.00), burst_probe_gap_s=180, burst_decline=0.60,
    distributed_probe_amount=(0.30, 5.00), distributed_decline=0.5, distributed_use_p=0.6, distributed_delay=(6.0, 1.0, 0.5, 120.0),
    # waves (many tested cards are sold on or never used)
    wave_cards=(20, 60), wave_hours=6, wave_probe_amount=(0.50, 3.00), wave_decline=0.60, wave_use_p=0.35, wave_delay=(7.0, 1.1, 1.0, 180.0),
    # ---- online cash-out (amounts often higher than normal; fraud clusters in risky shops, the same shops for many victims;
    # fraudulent recurring charges that follow the card to its replacement)
    cash_out_q=(0.75, 0.99), cash_out_hi_mult=1.5, risky_share=0.65, risky_per_mcc=5, recurring_fraud_p=0.12, recurring_fraud_months=(2, 6),
    # ---- stolen card: $5-30 first, then 3-10x larger ($100-500), 3-10 purchases over hours to 2 days, near home
    stolen_k=(3, 10), stolen_test_p=0.75, stolen_test_amount=(5.0, 30.0), stolen_big_amount=(100.0, 500.0),
    stolen_fuel_amount=(40.0, 100.0), stolen_gap_min=45.0, stolen_span_h=48.0, theft_before_first_h=(0.5, 6.0), burst_near_km=50,
    # ---- counterfeit card used slowly, kept subtle on purpose: irregular days; unfamiliar places get noticed within days, so most
    # episodes are short; some fraudsters convert funds quickly
    slow_short_p=0.55, slow_short_days=(2, 6), slow_long_days=(7, 21), slow_gap_poisson=0.8, slow_stay_p=0.7, slow_big_p=0.15,
    # ---- breached (compromised) merchant, one purchase per card: the misuse rate varies a lot, issuers reissue exposed cards, and
    # use after a breach can run to months, with skewed delays
    breach_misuse=(0.02, 0.10), breach_delay=(14.0, 1.1, 1.0, 180.0), reissue_p=0.10, reissue_discovery_days=(20, 90),
    reissue_activation_days=(5, 14),
    # ---- time of day (practitioners: 5-10% of in-person and 20-30% of online fraud between midnight and 6 am)
    online_fraud_night=0.25, inperson_fraud_late=0.2,
    # ---- copied amounts vary around the source row's amount
    amount_jitter_sigma=0.15,
)
BANK = dict(
    # when the bank notices an episode (until fraud is noticed it just continues; once caught, the card is blocked within minutes to
    # a few hours; customers notice the rest on their statement)
    detect=dict(burst=("first_purchase_h", 0.0, 24.0), stolen=("last_h", 0.1, 4.0), slow=("last_h", 0.1, 24.0),
                compromised=("first_lognormal_d", 10.0, 0.9, 1.0, 120.0), distributed=("first_lognormal_d", 10.0, 0.9, 1.0, 120.0),
                wave=("first_lognormal_d", 10.0, 0.9, 1.0, 120.0), ibm=("last_d", 0.0, 7.0)),
    new_card=((0.85, 2.0, 7.0), (0.15, 7.0, 14.0)),   # new card: 2-7 days, sometimes up to 2 weeks
    tries_active=(2, 10), tries_statement_extra=(0.3, 1, 3), tries_ibm=(1, 5),   # the fraudster tries 2-10 more times
    codes_stolen=(("Stolen Card,", .50), ("Card Blocked,", .25), ("Do Not Honor,", .25)),
    codes_other=(("Card Blocked,", .35), ("Do Not Honor,", .35), ("Suspected Fraud,", .30)),
    owner_try_p=dict(rule=0.5, stolen=0.0), owner_tries=(1, 2),
    card_on_file_days=60, card_on_file_not_updated=0.10,
    false_decline_rate=0.001, false_decline_retry=0.7, false_decline_weights=dict(trip=10.0, large=5.0, new_merchant=3.0, night=2.0),
    ibm_cluster_days=7,
)
CITY_GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_geo.json")   # made by build_city_geo.py
_ROW_KEYS = ("card", "ts", "label", "amount", "scenario", "foreign", "origin", "retimed", "episode", "probe")
ORIGINS = ("ibm", "ibm_moved_by_trip", "added", "ibm_retimed_after_trip", "ibm_declined_by_bank")   # where a row comes from


def _reorder_rows(raw, idx):
    """Apply one row selection or permutation to every per-row array of raw (categorical fields and bookkeeping arrays)."""
    for k in _ROW_KEYS:
        if k in raw: raw[k] = raw[k][idx]
    for k in raw["cat"]:
        raw["cat"][k] = raw["cat"][k][idx]
    return raw


def _code(dec, field, value):
    """Index of a decode value; appended if IBM's data does not have it (the bank's decline codes)."""
    v = str(value)
    if v not in dec[field]: dec[field].append(v)
    return dec[field].index(v)


def _pick(rng, pairs, k):
    """k draws from ((value, weight), ...)."""
    vals = [v for v, _ in pairs]; w = np.array([p for _, p in pairs], float)
    return [vals[i] for i in rng.choice(len(vals), k, p=w / w.sum())]


def _lognormal_days(rng, spec, k):
    """Skewed delays in days: lognormal with the given median and sigma, clipped to [lo, hi]."""
    med, sig, lo, hi = spec
    return np.clip(np.exp(np.log(med) + sig * rng.standard_normal(k)), lo, hi)


def inject_scenarios(raw, seed=0):
    """Step 1: fraud patterns and legit look-alikes on IBM's data (IBM's rows, labels and quirks as they are).
      Legit look-alikes: trips abroad and domestic trips take travel time and time zones into account (leg by leg; local times;
         nobody shops in person 23:00-07:00 after a move); trips mostly stay in one city (70%; else 2-3 cities of >= 2 days each);
         20% of trip days add no purchase; a hotel charge at check-out per city (65% of trips, only where IBM has a hotel there),
         1-3 taxi rides on 50% of trips by plane; domestic trips 0.5 per card-year, 40% of them road trips 150-800 km from home
         (booked ahead 25% of the time, flights 75%); new subscriptions 0.3 per card-year, 25% with one price rise, 15% with an
         amount that varies every month, 40% with a $0 card check first.
      One labelling rule: every transaction the fraudster makes is fraud, test payments (probes) included; the owner's stay legit.
      Counterfeit card: swiped in 60-90% of an episode's in-person purchases in the years the card uses its chip; stolen card:
         chip in 70-80%.
      Card-testing probes: $0.10-3, ~3 min apart, each at a small online shop, 60% declined; 15% of all probes are $0 checks.
      Hours: in-person fraud at shop hours (20% of it 18:00-02:00), online fraud 25% between 00:00 and 06:00, probes at any hour.
      Stolen card: near home (<= 50 km); 75% start with a $5-30 purchase at a gas station or small shop; then purchases of
         $100-500 that grow, at electronics, department / discount / wholesale stores, gift-card sellers (round amounts + fee),
         fuel ($40-100); 3-10 purchases over hours to 2 days, at shop hours.
      Counterfeit card used slowly: irregular days (gaps of 1 + Poisson(0.8) days), 55% last 2-6 days and 45% 7-21 days, 15% of
         its purchases larger (card's 90th-99th percentile x 1-1.5) at shops where goods resell.
      Online cash-outs at the card's 75th-99th percentile (x 1.5 at the top), 65% at the 5 riskiest shops of their category (most
         of IBM's fraud), and 12% of online episodes add a fraudulent monthly charge for 2-6 months (scenario 14).
      A card's fraud rates scale with its own risk ~ Gamma(1, 1): fraud is not evenly spread over cards.
      Distributed testing is used later 60% of the time and waves 35%, after skewed delays (lognormal, up to 4-6 months).
      Breached merchant: a misuse rate of 2-10%, the purchase after a skewed delay (median 14 days, up to 6 months); 10% of exposed
         cards get a new number from the bank (applied in bank_reaction). Copied amounts vary by x exp(N(0, 0.15)).
    Episode ids: raw["episode"] per row, raw["episodes"], raw["reissues"]; raw["probe"] marks probes."""
    rng = np.random.default_rng(seed + 33)
    raw.setdefault("origin", np.zeros(len(raw["card"]), np.int8))
    _reorder_rows(raw, np.lexsort((raw["ts"], raw["card"])))
    cat, dec = raw["cat"], raw["decode"]
    for c_ in BLOCK_CODES:
        _code(dec, "errors", c_)                       # fixed indices for the decline codes of the probes and of the bank
    cat0 = {k: v.copy() for k, v in cat.items()}; foreign0 = raw["foreign"].copy()
    card, ts, amount, label = raw["card"], raw["ts"], raw["amount"], raw["label"]
    n = len(card)
    retimed = np.zeros(n, bool)
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]]); ends = np.r_[starts[1:], n]
    YEAR, DAY, HOUR = 365.25 * 86400, 86400, 3600
    CHIP_ONLINE = dec["chip"].index("Online Transaction")
    CHIP_SWIPE = dec["chip"].index("Swipe Transaction")
    CHIP_CHIP = dec["chip"].index("Chip Transaction")
    ERR_NONE = int(np.bincount(cat["errors"]).argmax())
    probe_err = [(dec["errors"].index(e), w) for e, w in PARAMS["probe_codes"] if e in dec["errors"]]
    legit = (label == 0) & (foreign0 == 0)
    on = legit & (cat["chip"] == CHIP_ONLINE)
    on_any = np.flatnonzero(on & (amount > 0))
    mcc_code = {int(v): i for i, v in enumerate(dec["mcc"])}
    codes = lambda ms: [mcc_code[m] for m in ms if m in mcc_code]
    sub_on = on_any[np.isin(cat["mcc"][on_any], codes(SUB_MCCS)) & (amount[on_any] >= 3) & (amount[on_any] <= 30)]
    sub_on = sub_on if len(sub_on) else on_any
    test_on = on_any[np.isin(cat["mcc"][on_any], codes(TEST_MCCS))]
    test_on = test_on if len(test_on) else on_any
    tm_ = cat["merchant"][test_on]; tord = np.argsort(tm_, kind="stable"); test_on, tm_ = test_on[tord], tm_[tord]
    test_merchants, tm0 = np.unique(tm_, return_index=True); tm1 = np.r_[tm0[1:], len(test_on)]
    lg_all = np.flatnonzero(legit & (amount > 0))
    lg_sample = rng.choice(lg_all, min(4_000_000, len(lg_all)), replace=False)
    ss = lg_sample[np.lexsort((cat["city"][lg_sample], cat["state"][lg_sample]))]
    st_s, ci_s = cat["state"][ss], cat["city"][ss]
    by_state = {int(s): ss[st_s == s] for s in np.unique(st_s)}
    sc_key = st_s.astype(np.int64) * 100000 + ci_s
    u_sc, i0, cnt_sc = np.unique(sc_key, return_index=True, return_counts=True); i1 = np.r_[i0[1:], len(ss)]
    by_state_city = {int(k): ss[a:b] for k, a, b in zip(u_sc, i0, i1)}
    state_cities = {}
    for k, c_ in zip(u_sc, cnt_sc):
        state_cities.setdefault(int(k // 100000), []).append((int(k), int(c_)))
    dom_states = [s for s in by_state if len(dec["state"][s]) == 2 and len(by_state[s]) > 2000]
    dsw = np.array([len(by_state[s]) for s in dom_states], float); dsw /= dsw.sum()
    foreign_rows = np.flatnonzero((foreign0 == 1) & (label == 0) & (cat["chip"] != CHIP_ONLINE))
    fr_key = cat["state"][foreign_rows].astype(np.int64) * 100000 + cat["city"][foreign_rows]
    fo = np.argsort(fr_key, kind="stable"); foreign_rows, fr_key = foreign_rows[fo], fr_key[fo]
    u_fc, f0, cnt_fc = np.unique(fr_key, return_index=True, return_counts=True); f1 = np.r_[f0[1:], len(foreign_rows)]
    by_country_city = {int(k): foreign_rows[a:b] for k, a, b in zip(u_fc, f0, f1)}
    country_cities = {}
    for k, c_ in zip(u_fc, cnt_fc):
        country_cities.setdefault(int(k // 100000), []).append((int(k), int(c_)))
    countries = np.array(sorted(country_cities)); cw = np.array([sum(c for _, c in country_cities[c_]) for c_ in countries], float); cw /= cw.sum()
    travel_pre = codes((4511, 4722, 7011))
    online_travel = on_any[np.isin(cat["mcc"][on_any], travel_pre)] if travel_pre else on_any
    online_travel = online_travel if len(online_travel) else on_any
    road_booking = on_any[np.isin(cat["mcc"][on_any], codes((4722,)))]   # a hotel booked online before a road trip
    road_booking = road_booking if len(road_booking) else online_travel
    ip_legit = np.flatnonzero(legit & (amount > 0) & (cat["chip"] != CHIP_ONLINE))

    def rows_by_city(mccs):
        """Legit in-person rows of these MCCs, by city key (hotels and taxis of a trip's city)."""
        r = ip_legit[np.isin(cat["mcc"][ip_legit], codes(mccs))]
        k_ = cat["state"][r].astype(np.int64) * 100000 + cat["city"][r]
        oo = np.argsort(k_, kind="stable"); r, k_ = r[oo], k_[oo]
        u, a_ = np.unique(k_, return_index=True); b_ = np.r_[a_[1:], len(r)]
        return {int(x): r[p:q] for x, p, q in zip(u, a_, b_)}
    hotels, taxis = rows_by_city((7011,)), rows_by_city((4121,))   # generic hotels only (IBM's brand codes sit in odd cities)
    new = {k: [] for k in ("card", "ts", "amount", "label", "scenario", "foreign", "chip", "mcc", "merchant", "city", "state", "errors",
                           "episode", "probe")}
    cur = {"C": None}
    EP = {k: [] for k in ("card", "kind", "compromise", "theft")}
    RE = {k: [] for k in ("card", "exposure", "activation")}

    def new_ep(c, kind, compromise, theft=-1):
        EP["card"].append(int(c)); EP["kind"].append(EP_KINDS.index(kind)); EP["compromise"].append(int(compromise))
        EP["theft"].append(int(theft))
        return len(EP["card"]) - 1
    # online cash-out: per category the online legit rows by merchant and its 5 riskiest shops (most IBM fraud)
    ibm_fraud = np.flatnonzero(label == 1)
    co_groups, co_mcc = [], {}
    for name, ms, w in CASH_OUT_GROUPS:
        avail = []
        for m in codes(ms):
            r = on_any[cat0["mcc"][on_any] == m]
            if len(r) < 50: continue
            r = r[np.argsort(cat0["merchant"][r], kind="stable")]
            mu, m0, cnt = np.unique(cat0["merchant"][r], return_index=True, return_counts=True)
            fm, fc = np.unique(cat0["merchant"][ibm_fraud[cat0["mcc"][ibm_fraud] == m]], return_counts=True)
            fcount = np.zeros(len(mu)); ii = np.searchsorted(mu, fm); okk = ii < len(mu)
            okk[okk] = mu[ii[okk]] == fm[okk]; fcount[ii[okk]] = fc[okk]
            co_mcc[m] = dict(rows=r, m0=m0, m1=m0 + cnt, p=cnt / cnt.sum(), risky=np.lexsort((-cnt, -fcount))[:PARAMS["risky_per_mcc"]],
                             risky_fraud=int(fcount[np.lexsort((-cnt, -fcount))[:PARAMS["risky_per_mcc"]]].sum()))
            avail.append((m, len(r)))
        if avail: co_groups.append((name, avail, w))
    co_w = np.array([w for _, _, w in co_groups], float); co_w /= co_w.sum()
    pool_mcc_cache = {}
    # the hours at which people shop, in person and online (legit rows of all cards), as samples of the time of day
    tod_inperson = (ts[rng.choice(lg_all[cat["chip"][lg_all] != CHIP_ONLINE], 200_000)] % DAY).astype(np.int64)
    tod_online = (ts[rng.choice(on_any, 200_000)] % DAY).astype(np.int64)
    night_online_legit = float((tod_online < 6 * HOUR).mean())
    p_night = max(0.0, (PARAMS["online_fraud_night"] - night_online_legit) / (1 - night_online_legit))
    # coordinates and standard UTC offset of every city (city_geo.json, made by build_city_geo.py), by city key state * 100000 + city
    geo_json = json.load(open(CITY_GEO))
    geo = {}
    for k in np.unique(cat0["state"].astype(np.int64) * 100000 + cat0["city"]):
        st_, ci_ = int(k // 100000), int(k % 100000)
        g = geo_json.get(f"{dec['city'][ci_]}|{dec['state'][st_]}")
        if g: geo[int(k)] = (float(g[0]), float(g[1]), float(g[2]))
    st_geo = {}   # the cities of each state with coordinates and row counts
    for s_, lst in state_cities.items():
        kk = [(k, c_) for k, c_ in lst if k in geo]
        if kk:
            ks = np.array([k for k, _ in kk], np.int64)
            st_geo[s_] = (ks, np.array([c_ for _, c_ in kk], float), np.array([geo[int(k)][0] for k in ks]), np.array([geo[int(k)][1] for k in ks]))
    us = [(k, c_) for s_, lst in state_cities.items() if len(dec["state"][s_]) == 2 for k, c_ in lst if k in geo and c_ >= 50]
    us_keys = np.array([k for k, _ in us], np.int64); us_cnt = np.array([c_ for _, c_ in us], float)
    us_lat = np.array([geo[int(k)][0] for k in us_keys]); us_lon = np.array([geo[int(k)][1] for k in us_keys])

    def km_between(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - lon1)
        h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    def travel_h(a, b, drive=False):
        """Door-to-door hours from city key a to city key b (by car below 400 km or on a road trip, else by air), and b's UTC
        offset minus a's (h)."""
        if int(a) == int(b): return 0.0, 0.0
        ga, gb = geo.get(int(a)), geo.get(int(b))
        if ga is None or gb is None: return 5.0 * rng.uniform(0.9, 1.15), 0.0
        km = float(km_between(ga[0], ga[1], gb[0], gb[1]))
        if drive or km < PARAMS["drive_below_km"]:
            h = 0.5 + km / PARAMS["drive_kmh"]
        else:
            h = PARAMS["airport_before_h"] + km / PARAMS["flight_kmh"] + PARAMS["airport_after_h"] + (PARAMS["connection_h"] if km > PARAMS["connection_km"] else 0.0)
        return h * rng.uniform(0.9, 1.15), gb[2] - ga[2]

    def emit(c, t, src, lab, sc, amt=None, errors=None, chip=None, swipe_share=None, chip_p=None, ep=-1, probe=False):
        """One added row per time in t, copying merchant, MCC, city and state from source row src (IBM's original values).
        amt None: the source amount x exp(N(0, 0.15)), whole dollars stay whole. An in-person row takes the card's own chip / swipe
        mix of that year, or, with swipe_share (counterfeit card) / chip_p (stolen card), the chip with that probability scaled by
        min(1, card's chip share / 0.8), else swipe."""
        k = len(t)
        if k == 0: return
        t = (np.asarray(t, np.int64) // 60) * 60
        src = np.asarray(src, np.int64)
        if amt is None:
            a0 = amount[src].astype(np.float64); j_ = a0 * np.exp(PARAMS["amount_jitter_sigma"] * rng.standard_normal(k))
            a = np.where(np.round(a0 * 100) % 100 == 0, np.round(j_), np.round(j_, 2))
        else:
            a = np.round(np.broadcast_to(np.asarray(amt, np.float64), (k,)), 2)
        ch = (cat0["chip"][src] if chip is None else np.broadcast_to(chip, (k,))).astype(np.int32).copy()
        C = cur["C"]
        if C is not None:
            ip = ch != CHIP_ONLINE
            if ip.any():
                if swipe_share is not None or chip_p is not None:
                    p = (1.0 - swipe_share) if swipe_share is not None else chip_p
                    p = p * np.minimum(1.0, C.chip_share(t[ip]) / 0.8)
                    ch[ip] = np.where(rng.random(int(ip.sum())) < p, CHIP_CHIP, CHIP_SWIPE)
                else:
                    ch[ip] = C.chip_at(t[ip])
        new["card"].append(np.full(k, c)); new["ts"].append(t); new["amount"].append(a.astype(np.float32))
        new["label"].append(np.full(k, lab, np.int8)); new["scenario"].append(np.full(k, sc, np.int8))
        new["foreign"].append(foreign0[src].astype(np.int8)); new["chip"].append(ch)
        for f in ("mcc", "merchant", "city", "state"):
            new[f].append(cat0[f][src].astype(np.int32))
        new["errors"].append((cat0["errors"][src] if errors is None else np.broadcast_to(np.asarray(errors), (k,))).astype(np.int32))
        new["episode"].append(np.full(k, ep, np.int32)); new["probe"].append(np.full(k, probe, bool))

    def in_range(pool, k, lo, hi, extra=None, t=None):
        if len(pool) == 0: pool = on_any
        cand = rng.choice(pool, min(len(pool), (256 if t is not None else 64) * k + 64))
        ok = (amount[cand] >= lo) & (amount[cand] <= hi)
        if extra is not None: ok2 = ok & extra(cand); ok = ok2 if ok2.sum() >= k else ok
        if t is not None: ok3 = ok & (np.abs(ts[cand] - t) <= YEAR); ok = ok3 if ok3.sum() >= k else ok
        good = cand[ok] if ok.sum() >= k else cand
        return rng.choice(good, k, replace=len(good) < k)

    def era(pool, k, t):
        if len(pool) == 0: pool = on_any
        cand = rng.choice(pool, min(len(pool), 256 * k + 256))
        ok = np.abs(ts[cand] - t) <= YEAR
        good = cand[ok] if ok.sum() >= k else cand
        return rng.choice(good, k, replace=len(good) < k)

    def cash_out(k, lo, hi, t):
        """k online cash-out source rows: a category group (CASH_OUT_GROUPS weights), an MCC of it (by IBM's online volume), 65% one
        of its 5 riskiest shops, else any shop by volume; a row of that shop with an amount in [lo, hi] where it has one."""
        src = np.empty(k, np.int64)
        for i in range(k):
            _, avail, _ = co_groups[rng.choice(len(co_groups), p=co_w)]
            mw = np.array([c_ for _, c_ in avail], float)
            P = co_mcc[avail[rng.choice(len(avail), p=mw / mw.sum())][0]]
            j = int(rng.choice(P["risky"])) if rng.random() < PARAMS["risky_share"] else int(rng.choice(len(P["p"]), p=P["p"]))
            src[i] = in_range(P["rows"][P["m0"][j]:P["m1"][j]], 1, lo, hi, t=t)[0]
        return src

    def probe_src(k, t):
        js = rng.integers(0, len(test_merchants), k)
        return np.array([era(test_on[tm0[j]:tm1[j]], 1, t)[0] for j in js], np.int64)

    def city_pool(key, pools):
        return pools.get(key, np.zeros(0, np.int64))

    def mcc_rows(key, pools):
        cache = pool_mcc_cache.get(key)
        if cache is None:
            P = city_pool(key, pools); mm = cat0["mcc"][P]; oo = np.argsort(mm, kind="stable"); P, mm = P[oo], mm[oo]
            u, a_ = np.unique(mm, return_index=True); b_ = np.r_[a_[1:], len(P)]
            cache = pool_mcc_cache[key] = (P, {int(x): P[p:q] for x, p, q in zip(u, a_, b_)})
        return cache

    def same_mcc(key, pools, m, t):
        P, by_m = mcc_rows(key, pools)
        return int(era(by_m.get(int(m), P), 1, t)[0])

    def shop_row(key, pools, mccs, t, amt=None):
        """A row of city key at one of these MCCs (by IBM's volume; with an amount within 3x of amt where possible); None if the
        city has none."""
        P, by_m = mcc_rows(key, pools)
        r = [by_m[m] for m in codes(mccs) if m in by_m]
        if not r: return None
        r = np.concatenate(r)
        return int(in_range(r, 1, amt / 3, amt * 3, t=t)[0]) if amt is not None else int(era(r, 1, t)[0])

    def itinerary(dur, cities):
        """Cities of a trip in turn: one city in 70% of trips (and every trip under 4 days), else 2 (under 7 days) or 2-3 cities,
        each for at least 2 days."""
        keys = np.array([k for k, _ in cities]); w = np.array([c for _, c in cities], float); w /= w.sum()
        n_c = 1 if (dur < 4 or rng.random() < PARAMS["one_city_share"]) else (2 if dur < 7 else int(rng.integers(2, 4)))
        n_c = int(min(n_c, len(keys)))
        chosen = rng.choice(keys, n_c, replace=False, p=w)
        lengths = 2 + rng.multinomial(dur - 2 * n_c, np.ones(n_c) / n_c) if n_c > 1 else np.array([dur])
        return np.repeat(chosen, lengths)
    # the times of fraud
    tod_inperson_late = tod_inperson[(tod_inperson >= 18 * HOUR) | (tod_inperson < 2 * HOUR)]

    def times_inperson(days):
        """The hours people shop in person (legit rows of all cards, +-20 min within the day), 20% of them from 18:00-02:00."""
        days = np.asarray(days, np.int64); k = len(days)
        tod = np.where(rng.random(k) < PARAMS["inperson_fraud_late"], rng.choice(tod_inperson_late, k), rng.choice(tod_inperson, k))
        return np.sort(days * DAY + np.clip(tod + rng.integers(-1200, 1200, k), 0, DAY - 60))

    def times_online_fraud(days):
        """25% between 00:00 and 06:00 (practitioners: 20-30% of online fraud), the rest at the hours people shop online."""
        days = np.asarray(days, np.int64); k = len(days)
        tod = np.where(rng.random(k) < p_night, rng.integers(0, 6 * HOUR, k),
                       np.clip(rng.choice(tod_online, k) + rng.integers(-1200, 1200, k), 0, DAY - 60))
        return np.sort(days * DAY + tod)

    def times_any_hour(days):
        days = np.asarray(days, np.int64)
        return np.sort(days * DAY + rng.integers(0, DAY, len(days)))

    def push_after(times, t_min, keys=None, drive=False):
        """Sorted in-person times moved forward so that the first is not before t_min and each later one not before
        the previous one (plus the travel time when its city differs, keys = city keys; the local time never goes back, so the
        rows keep their order); a moved time comes 10-60 min after its earliest possible time, and one that lands between 23:00
        and 07:00 goes to 07:30-10:00 the next morning. Stops at the first time that need not move. Returns (times, how many moved)."""
        out = np.array(times, np.int64); moved = 0
        for i in range(len(out)):
            req = int(t_min)
            if i:
                req = int(out[i - 1])
                if keys is not None and keys[i] != keys[i - 1]:
                    h, dtz = travel_h(keys[i - 1], keys[i], drive); req = max(req, req + int((h + dtz) * HOUR))
            if out[i] >= req: break
            t = (req // 60 + int(rng.integers(10, 61))) * 60
            tod = t % DAY
            if tod >= 23 * HOUR or tod < 7 * HOUR:
                t = (t // DAY + (1 if tod >= 23 * HOUR else 0)) * DAY + int(rng.integers(450, 600)) * 60
            out[i] = t; moved += 1
        return out, moved

    def city_of(r):
        return int(cat["state"][r]) * 100000 + int(cat["city"][r])

    def trip(c, s, e, d, dur, day_city, pools, sc, per_day, lo, hi, booking_p, booking_pool, booking_hi, drive, night_rate):
        """A trip whose legs take time. Its purchases are walked leg by leg: home -> city 1 -> ... -> home; on each leg the
        traveller leaves 1.5-4 h after his last purchase where he is and shops 0.5-1.5 h after the travel time (local times);
        earlier purchases of the next place move later (push_after). The purchases: the cardholder's own in-person rows on the
        trip days (moved to that day's city, each merchant to one merchant of the same category there), added purchases on the
        trip's active days (80% of days), a hotel charge at check-out per city (65% of trips), 1-3 taxi rides (50% of trips by
        plane). A booking before the trip (booking_p). Back home, his first in-person purchases wait the same way (raw["retimed"])."""
        C = cur["C"]
        own = s + np.flatnonzero((label[s:e] == 0) & (raw["scenario"][s:e] == 0) & (cat["chip"][s:e] != CHIP_ONLINE))
        own = own[np.argsort(ts[own], kind="stable")]
        before = own[ts[own] < d * DAY]; after = own[ts[own] >= (d + dur) * DAY]
        inside = own[(ts[own] >= d * DAY) & (ts[own] < (d + dur) * DAY)]
        seg0 = np.flatnonzero(np.r_[True, day_city[1:] != day_city[:-1]])            # first trip day of each city
        seg_last = np.r_[seg0[1:], dur] - 1                                           # last trip day of each city
        active = rng.random(dur) < PARAMS["active_day_p"]
        cnt = rng.poisson(per_day / PARAMS["active_day_p"], dur) * active
        add_d = d + np.repeat(np.arange(dur), cnt)
        add_t = add_d * DAY + np.clip(rng.choice(C.tod, len(add_d)) + rng.integers(-1200, 1200, len(add_d)), 0, DAY - 60)
        kind = [np.zeros(len(add_t), np.int8)]; extra_t = [add_t.astype(np.int64)]; extra_a = [np.full(len(add_t), np.nan)]
        if rng.random() < PARAMS["hotel_p"]:                                            # check-out, 07:00-11:00
            for i, a_ in enumerate(seg0):
                key = int(day_city[a_])
                if key not in hotels: continue
                nights = max(1, int(seg_last[i] - a_ + 1) - (1 if i == len(seg0) - 1 else 0))
                extra_t.append(np.array([(d + seg_last[i]) * DAY + int(rng.uniform(7, 11) * HOUR)], np.int64))
                extra_a.append(np.array([round(nights * rng.uniform(*night_rate), 2)])); kind.append(np.ones(1, np.int8))
                stats["trip_hotels"] += 1
        if not drive and rng.random() < PARAMS["taxi_p"]:
            kt = int(rng.integers(PARAMS["taxi_rides"][0], PARAMS["taxi_rides"][1] + 1)); td = d + rng.integers(0, dur, kt)
            extra_t.append((td * DAY + np.clip(rng.choice(C.tod, kt) + rng.integers(-1200, 1200, kt), 0, DAY - 60)).astype(np.int64))
            extra_a.append(np.full(kt, np.nan)); kind.append(np.full(kt, 2, np.int8))
        add_t, add_a, add_kind = np.concatenate(extra_t), np.concatenate(extra_a), np.concatenate(kind)
        all_t = np.r_[ts[inside].astype(np.int64), add_t]
        who = np.r_[inside, np.full(len(add_t), -1, np.int64)]                        # own row index, or -1 = added
        aid = np.r_[np.full(len(inside), -1, np.int64), np.arange(len(add_t))]        # index into the added rows
        oo = np.argsort(all_t, kind="stable"); all_t, who, aid = all_t[oo], who[oo], aid[oo]
        seg = np.clip(np.searchsorted(seg0, all_t // DAY - d, side="right") - 1, 0, len(seg0) - 1)
        new_t = all_t.copy()
        if len(before):
            prev, t_prev = city_of(before[-1]), int(ts[before[-1]])
        else:
            prev, t_prev = C.home_key, d * DAY - int(rng.uniform(12, 36) * HOUR)
        for i, a_ in enumerate(seg0):
            key = int(day_city[a_])
            h, dtz = travel_h(prev, key, drive)
            arrive = t_prev + int(rng.uniform(1.5, 4.0) * HOUR) + int((h + dtz) * HOUR)   # local time of the place he arrives at
            arrive = max(arrive, t_prev + HOUR)            # the data is in local time: it never goes back, even flying west
            m = np.flatnonzero(seg == i)
            if len(m):
                new_t[m], mv = push_after(new_t[m], arrive + int(rng.uniform(0.5, 1.5) * HOUR))
                stats["trip_rows_moved_later"] += mv; t_prev = int(new_t[m].max())
            else:
                t_prev = arrive
            if i == 0: travel_out.append(h)
            prev = key
        mapping, add_by = {}, {}
        for t_, r, j, i in zip(new_t, who, aid, seg):
            key = int(day_city[seg0[i]])
            if r >= 0:
                mk = (int(cat0["merchant"][r]), key)
                if mk not in mapping: mapping[mk] = same_mcc(key, pools, cat0["mcc"][r], d * DAY)
                src = mapping[mk]
                for f in ("city", "state", "merchant", "mcc"):
                    cat[f][r] = cat0[f][src]
                raw["foreign"][r] = foreign0[src]; raw["scenario"][r] = sc; ts[r] = int(t_) // 60 * 60
            else:
                add_by.setdefault((key, int(add_kind[j])), []).append((int(t_), add_a[j]))
        for (key, kd), lst in add_by.items():
            tl = np.array([x for x, _ in lst], np.int64); al = np.array([y for _, y in lst])
            if kd == 0:
                emit(c, tl, in_range(city_pool(key, pools), len(tl), lo, hi, t=int(tl.min())), 0, sc)
            elif kd == 1:
                emit(c, tl, era(hotels[key], len(tl), int(tl.min())), 0, sc, amt=al)
            elif key in taxis:
                emit(c, tl, era(taxis[key], len(tl), int(tl.min())), 0, sc); stats["trip_taxi_rides"] += len(tl)
        if rng.random() < booking_p:
            emit(c, C.times([d - int(rng.integers(1, 31))]), in_range(booking_pool, 1, 100, booking_hi, t=d * DAY), 0, sc)
            stats["trip_bookings"] += 1
        # the way back: he leaves on the last trip day (08:00-20:00) at the earliest
        dep = max(t_prev + int(rng.uniform(1.5, 4.0) * HOUR), (d + dur - 1) * DAY + int(rng.uniform(8, 20) * HOUR))
        if len(after):
            h, dtz = travel_h(prev, city_of(after[0]), drive)
            keys = cat["state"][after].astype(np.int64) * 100000 + cat["city"][after]
            back = max(dep + int((h + dtz) * HOUR), t_prev + HOUR)    # local time at home when he is back (never before he left)
            nt, mv = push_after(ts[after].astype(np.int64), back + int(rng.uniform(0.5, 1.5) * HOUR), keys)
            if mv:
                ts[after] = nt; retimed[after[:mv]] = True; stats["home_rows_moved_later"] += mv
        stats["moved_rows"] += len(inside)

    class Card:
        def __init__(self, s, e):
            lg = s + np.flatnonzero(label[s:e] == 0)
            self.tod = ts[lg] % DAY if len(lg) else np.array([12 * 3600])
            a = amount[lg]; a = np.sort(a[a > 0]) if (a > 0).any() else np.array([20.0, 50.0])
            self.q = lambda p: float(np.quantile(a, p))
            self.state = int(np.bincount(cat0["state"][s:e]).argmax())
            mc = np.bincount(cat0["mcc"][lg], minlength=1) if len(lg) else np.zeros(1)
            self.mcc_top = np.argsort(mc)[::-1][:5]
            self.cities = set(np.unique(cat0["city"][s:e]).tolist())
            self.t0, self.t1 = ts[s], ts[e - 1]
            ip = lg[cat["chip"][lg] != CHIP_ONLINE]
            nch = len(dec["chip"])
            if len(ip):
                yr = (ts[ip] // YEAR).astype(np.int64); self.chip_years = np.unique(yr)
                self.chip_cdf = np.cumsum([np.bincount(cat["chip"][ip][yr == y], minlength=nch) / (yr == y).sum() for y in self.chip_years], 1)
                hk = cat0["state"][ip].astype(np.int64) * 100000 + cat0["city"][ip]
                u_, c_ = np.unique(hk, return_counts=True); self.home_key = int(u_[c_.argmax()])   # the home city
            else:
                self.chip_years = np.array([0], np.int64); self.chip_cdf = np.cumsum(np.eye(nch)[[CHIP_SWIPE]], 1)
                self.home_key = int(self.state) * 100000 + int(np.bincount(cat0["city"][s:e]).argmax())
            self.home_state = self.home_key // 100000
            self.busy = []

        def year_index(self, t):
            y = (np.asarray(t) // YEAR).astype(np.int64)
            i = np.clip(np.searchsorted(self.chip_years, y), 0, len(self.chip_years) - 1); j = np.clip(i - 1, 0, len(self.chip_years) - 1)
            return np.where(np.abs(self.chip_years[j] - y) < np.abs(self.chip_years[i] - y), j, i)

        def chip_at(self, t):
            yi = self.year_index(t)
            u = rng.random(len(yi))
            return np.minimum((u[:, None] > self.chip_cdf[yi]).sum(1), self.chip_cdf.shape[1] - 1).astype(np.int32)

        def chip_share(self, t):
            """The card's share of chip among its in-person rows in the year nearest to t."""
            cdf = self.chip_cdf[self.year_index(t)]
            return cdf[:, CHIP_CHIP] - (cdf[:, CHIP_CHIP - 1] if CHIP_CHIP else 0)

        def times(self, days):
            days = np.asarray(days, np.int64)
            t = days * DAY + rng.choice(self.tod, len(days)) + rng.integers(-1200, 1200, len(days))
            return np.sort(t)

        def free(self, d0, d1):
            if any(d0 < b1 and a1 < d1 for a1, b1 in self.busy): return False
            self.busy.append((d0, d1)); return True

    def near_home_city(C):
        """The home city or a city of its state within 50 km, as often as IBM's data has them; None if unknown."""
        g, hg = st_geo.get(C.home_state), geo.get(C.home_key)
        if g is not None and hg is not None:
            ks, cnt, la, lo_ = g
            ok = km_between(hg[0], hg[1], la, lo_) <= PARAMS["burst_near_km"]
            if ok.any():
                return int(rng.choice(ks[ok], p=cnt[ok] / cnt[ok].sum()))
        return C.home_key if C.home_key in by_state_city else None

    def road_cities(C):
        """A road trip's destination: a US city 150-800 km from home (by IBM's row counts), plus up to 5 cities within 100 km of it
        for a trip that visits more than one; None if the home city has no coordinates."""
        hg = geo.get(C.home_key)
        if hg is None or not len(us_keys): return None
        dk = km_between(hg[0], hg[1], us_lat, us_lon)
        ok = np.flatnonzero((dk >= PARAMS["road_trip_km"][0]) & (dk <= PARAMS["road_trip_km"][1]))
        if not len(ok): return None
        i = int(rng.choice(ok, p=us_cnt[ok] / us_cnt[ok].sum()))
        near = np.flatnonzero(km_between(us_lat[i], us_lon[i], us_lat, us_lon) <= 100)
        near = near[near != i][:5]
        return [(int(us_keys[i]), float(us_cnt[i]) * 3)] + [(int(us_keys[j]), float(us_cnt[j])) for j in near]

    def probes(c, tt, sc, ep, amount_range, decline_p, src=None):
        """Card-testing probes at small online shops (one shop per probe unless src is given): 15% are $0 checks; declined ones
        carry a decline code (bad CVV / card number / expiry, insufficient funds, do not honour). Returns which were declined."""
        k = len(tt)
        if src is None: src = probe_src(k, int(np.min(tt)))
        a = rng.uniform(*amount_range, k); a[rng.random(k) < PARAMS["probe_zero_p"]] = 0.0
        declined = rng.random(k) < decline_p
        errs = np.where(declined, np.array(_pick(rng, probe_err, k)), ERR_NONE)
        emit(c, tt, src, 1, sc, amt=a, errors=errs, ep=ep, probe=True)
        stats["probes"] += k; stats["probes_zero"] += int((a == 0).sum()); stats["probes_declined"] += int(declined.sum())
        return declined

    def cash_emit(c, times, sc, ep):
        """Online cash-out purchases of an episode (card's 75th-99th percentile, x 1.5 at the top), and with probability 0.12 a
        fraudulent monthly charge for 2-6 months from the first one (scenario 14; it follows the card to its replacement)."""
        C = cur["C"]; times = np.asarray(times, np.int64); k = len(times)
        if k == 0: return
        lo = C.q(PARAMS["cash_out_q"][0]); hi = max(C.q(PARAMS["cash_out_q"][1]) * PARAMS["cash_out_hi_mult"], lo + 20)
        emit(c, times, cash_out(k, lo, hi, int(times[0])), 1, sc, ep=ep)
        stats["cash_outs"] += k
        if rng.random() < PARAMS["recurring_fraud_p"]:
            months = int(rng.integers(PARAMS["recurring_fraud_months"][0], PARAMS["recurring_fraud_months"][1] + 1))
            src = era(sub_on, 1, int(times[0]))
            days = int(times[0] // DAY) + int(rng.integers(0, 4)) + 30 * np.arange(months) + rng.integers(-1, 2, months)
            days = days[days <= C.t1 // DAY]
            if len(days):
                emit(c, times_any_hour(days), np.repeat(src, len(days)), 1, 14, amt=round(float(amount[src[0]]), 2), errors=ERR_NONE, ep=ep)
                stats["recurring_fraud_series"] += 1; stats["recurring_fraud_rows"] += len(days)

    stats = {k: 0 for k in ("trips", "dom_trips", "road_trips", "trips_skipped_overlap", "trip_hotels", "trip_taxi_rides", "trip_bookings",
                            "subscriptions", "subscription_zero_checks", "subscription_price_rises", "subscription_varying",
                            "card_testing", "card_testing_no_purchase", "stolen", "stolen_with_test", "stolen_no_shop", "slow_ato",
                            "slow_big_rows", "dist_testing", "dist_testing_used", "waves", "wave_cards", "wave_cashouts", "compromised_frauds",
                            "reissues_planned", "probes", "probes_zero", "probes_declined", "cash_outs", "recurring_fraud_series",
                            "recurring_fraud_rows", "moved_rows", "trip_rows_moved_later", "home_rows_moved_later", "ato_not_near_home")}
    travel_out = []
    risk = rng.gamma(PARAMS["risk_shape"], 1.0 / PARAMS["risk_shape"], len(starts))
    for ci, (s, e) in enumerate(zip(starts, ends)):
        c = card[s]; C = Card(s, e); cur["C"] = C
        years = max((C.t1 - C.t0) / YEAR, 1e-3)
        day0, day1 = int(C.t0 // DAY) + 1, int(C.t1 // DAY) - 1
        if day1 <= day0: continue
        rday = lambda: int(rng.integers(day0, day1))
        r_ = float(risk[ci])
        # (2) legit trip abroad: one country, mostly one city; (6) legit domestic trip: 40% road trips 150-800 km from home (driving,
        #     rarely booked), else by plane to another state. Legs take time. All of the card's trips are planned first and
        #     then made in date order, so home rows that a trip moved later can still belong to the next trip.
        plans = []
        for _ in range(rng.poisson(PARAMS["trip_abroad_rate"] * years)):
            d = rday(); dur = int(rng.integers(3, 11))
            country = int(rng.choice(countries, p=cw))
            if not C.free(d - 2, d + dur + 2): stats["trips_skipped_overlap"] += 1; continue
            plans.append((d, dur, itinerary(dur, country_cities[country]), by_country_city, 2, 2.0, PARAMS["booking_fly_p"], online_travel,
                          2000, False, PARAMS["hotel_night_abroad"]))
        for _ in range(rng.poisson(PARAMS["trip_domestic_rate"] * years)):
            road = rng.random() < PARAMS["road_trip_share"]
            cities = road_cities(C) if road else None
            if road and cities is None: road = False
            if not road:
                st2 = int(rng.choice(dom_states, p=dsw))
                if st2 == C.home_state: continue
                cities = state_cities[st2]
            d = rday(); dur = int(rng.integers(3, 11))
            if not C.free(d - 2, d + dur + 2): stats["trips_skipped_overlap"] += 1; continue
            plans.append((d, dur, itinerary(dur, cities), by_state_city, 6, 1.5, PARAMS["booking_road_p"] if road else PARAMS["booking_fly_p"],
                          road_booking if road else online_travel, 600 if road else 1500, road, PARAMS["hotel_night"]))
        for d, dur, day_city, pools, sc_, per_day, bp, bpool, bhi, road, night in sorted(plans, key=lambda x: x[0]):
            trip(c, s, e, d, dur, day_city, pools, sc_, per_day, C.q(0.1), C.q(0.9), bp, bpool, bhi, road, night)
            if sc_ == 2: stats["trips"] += 1
            else: stats["dom_trips"] += 1; stats["road_trips"] += int(road)
        # (7) legit new subscription: monthly at a streaming, digital media or games merchant; the price may rise or vary
        for _ in range(rng.poisson(PARAMS["subscription_rate"] * years)):
            d = rday(); months = int(rng.integers(3, 13)); src = era(sub_on, 1, d * DAY)
            days = [x for x in d + 30 * np.arange(months) + rng.integers(-1, 2, months) if x <= day1]
            if not days: continue
            k = len(days); base = round(float(amount[src[0]]), 2); amts = np.full(k, base); u = rng.random()
            if u < PARAMS["sub_price"][0]:
                pass
            elif u < PARAMS["sub_price"][0] + PARAMS["sub_price"][1]:
                if k > 3:
                    amts[int(rng.integers(3, k)):] = round(base * rng.uniform(*PARAMS["sub_step"]), 2); stats["subscription_price_rises"] += 1
            else:
                amts = np.round(base * rng.uniform(*PARAMS["sub_vary"], k), 2); stats["subscription_varying"] += 1
            tt = C.times(days)
            if rng.random() < PARAMS["zero_auth_p"]:                      # a $0 check of the card when it is first saved
                emit(c, [tt[0] - int(rng.integers(60, 900))], src, 0, 7, amt=0.0, errors=ERR_NONE); stats["subscription_zero_checks"] += 1
            emit(c, tt, np.repeat(src, k), 0, 7, amt=amts, errors=np.full(k, ERR_NONE))
            stats["subscriptions"] += 1
        # (3) card-testing burst: probes of $0.10-3 about 3 min apart, each at a small online shop, 60% declined, at any
        #     hour; the 1-2 online purchases 0.5-6 h later only if a probe went through
        for _ in range(rng.poisson(PARAMS["burst_rate"] * years * r_)):
            t_start = rday() * DAY + int(rng.integers(0, DAY)); k1 = int(rng.integers(3, 8))
            tt = t_start + np.cumsum(rng.exponential(PARAMS["burst_probe_gap_s"], k1)).astype(np.int64)
            ep = new_ep(c, "burst", t_start)
            declined = probes(c, tt, 3, ep, PARAMS["burst_probe_amount"], PARAMS["burst_decline"])
            if (~declined).any():
                k2 = int(rng.integers(1, 3)); tb = tt[-1] + (np.sort(rng.uniform(0.5, 6, k2)) * HOUR).astype(np.int64)
                cash_emit(c, tb, 3, ep)
            else:
                stats["card_testing_no_purchase"] += 1
            stats["card_testing"] += 1
        # (4) stolen card: near home, a small first purchase, then larger ones that grow
        for _ in range(rng.poisson(PARAMS["stolen_rate"] * years * r_)):
            d = rday(); key = near_home_city(C)
            if key is None:                                   # no coordinates: a city of the home state
                stats["ato_not_near_home"] += 1
                cl = state_cities.get(C.home_state)
                if cl:
                    w = np.array([x for _, x in cl], float); key = int(np.array([x for x, _ in cl])[rng.choice(len(cl), p=w / w.sum())])
            if key is None or key not in by_state_city: continue
            k = int(rng.integers(PARAMS["stolen_k"][0], PARAMS["stolen_k"][1] + 1))
            t_theft = int(times_inperson([d])[0])               # when people are out (20% of them in the evening or at night)
            t1 = t_theft + int(rng.uniform(*PARAMS["theft_before_first_h"]) * HOUR)
            gaps = np.clip(np.exp(np.log(PARAMS["stolen_gap_min"]) + rng.standard_normal(k - 1)), 5, 480) * 60
            tt = t1 + np.r_[0, np.cumsum(gaps)].astype(np.int64)
            span = PARAMS["stolen_span_h"] * HOUR
            if tt[-1] - t1 > span: tt = t1 + ((tt - t1) * (span / (tt[-1] - t1))).astype(np.int64)
            big = np.sort(np.exp(rng.uniform(np.log(PARAMS["stolen_big_amount"][0]), np.log(PARAMS["stolen_big_amount"][1]), k)))
            gi = rng.choice(len(STOLEN_BIG), k, p=np.array([w for _, _, w in STOLEN_BIG]) / sum(w for _, _, w in STOLEN_BIG))
            tod = tt % DAY; night = (tod >= 23 * HOUR) | (tod < 7 * HOUR)
            # at night only gas stations and 24-hour shops are open: a night purchase is fuel or gift cards there, else the next morning
            open_24h = np.array([STOLEN_BIG[g][0] in ("fuel", "gift_cards") for g in gi])
            switch = night & ~open_24h & (rng.random(k) < 0.5)
            gi[switch] = [i for i, (nm, _, _) in enumerate(STOLEN_BIG) if nm == "gift_cards"][0]
            open_24h = np.array([STOLEN_BIG[g][0] in ("fuel", "gift_cards") for g in gi])
            prev = None
            for i in range(k):                                  # in order; a closed shop's purchase waits for 07:30-10:00
                t_ = int(tt[i]) if prev is None else max(int(tt[i]), prev + int(rng.integers(5, 21)) * 60)
                tod_ = t_ % DAY
                if (tod_ >= 23 * HOUR or tod_ < 7 * HOUR) and not open_24h[i]:
                    t_ = (t_ // DAY + (1 if tod_ >= 23 * HOUR else 0)) * DAY + int(rng.integers(450, 600)) * 60
                tt[i] = prev = t_
            amts, mccs = big.copy(), [STOLEN_BIG[g][1] for g in gi]
            for i, g in enumerate(gi):
                if STOLEN_BIG[g][0] == "gift_cards":          # gift cards: round amounts plus a fee
                    amts[i] = max(100.0, round(big[i] / 50) * 50) + float(rng.choice([0.0, 3.95, 4.95, 5.95]))
                elif STOLEN_BIG[g][0] == "fuel":
                    amts[i] = rng.uniform(*PARAMS["stolen_fuel_amount"])
            if rng.random() < PARAMS["stolen_test_p"]:           # the card is checked with a small purchase first
                amts[0] = rng.uniform(*PARAMS["stolen_test_amount"]); mccs[0] = SMALL_SHOP_MCCS; stats["stolen_with_test"] += 1
            src = []
            for i in range(k):
                r = shop_row(key, by_state_city, mccs[i], int(tt[i]), amts[i])
                if r is None: r = shop_row(key, by_state_city, tuple(m for _, ms, _ in STOLEN_BIG for m in ms) + SMALL_SHOP_MCCS, int(tt[i]), amts[i])
                if r is None: r = int(era(by_state_city[key], 1, int(tt[i]))[0]); stats["stolen_no_shop"] += 1
                src.append(r)
            ep = new_ep(c, "stolen", t_theft, theft=t_theft)
            emit(c, tt, np.array(src, np.int64), 1, 4, amt=np.round(amts, 2), chip_p=rng.uniform(*PARAMS["stolen_chip"]), ep=ep)
            stats["stolen"] += 1
        # (8) counterfeit card used slowly: 1-2 new cities of the home state, irregular days, the card's usual
        #     categories and amounts (15% larger, where goods resell), mostly swiped
        for _ in range(rng.poisson(PARAMS["slow_rate"] * years * r_)):
            d = rday()
            dur = int(rng.integers(PARAMS["slow_short_days"][0], PARAMS["slow_short_days"][1] + 1)) if rng.random() < PARAMS["slow_short_p"] else \
                int(rng.integers(PARAMS["slow_long_days"][0], PARAMS["slow_long_days"][1] + 1))
            pool = by_state.get(C.home_state, lg_sample)
            seeds = rng.choice(pool, 50)
            new_city = list(dict.fromkeys(int(x) for x in cat0["city"][seeds] if int(x) not in C.cities))[:2]
            if not new_city: continue
            act, x = [], d
            while x < d + dur:
                act.append(x); x += 1 + int(rng.poisson(PARAMS["slow_gap_poisson"]))
            per = rng.choice([1, 2, 3], len(act), p=[0.5, 0.35, 0.15])
            ci_ = [int(rng.integers(len(new_city)))]
            for _ in act[1:]:
                ci_.append(ci_[-1] if (len(new_city) == 1 or rng.random() < PARAMS["slow_stay_p"]) else 1 - ci_[-1])
            days = np.repeat(np.array(act, np.int64), per)
            dcity = np.repeat(np.array([C.home_state * 100000 + new_city[i] for i in ci_], np.int64), per)
            tt = times_inperson(days)                         # sorted; rows of one day stay together, so dcity still matches
            big = rng.random(len(days)) < PARAMS["slow_big_p"]
            swipe = rng.uniform(*PARAMS["counterfeit_swipe"])
            ep = new_ep(c, "slow", int(tt.min()) - int(rng.uniform(1, 30) * DAY))
            mt = set(C.mcc_top.tolist())
            for key in np.unique(dcity):
                mn, mb = (dcity == key) & ~big, (dcity == key) & big
                if mn.any():
                    emit(c, tt[mn], in_range(city_pool(int(key), by_state_city), int(mn.sum()), C.q(0.25), C.q(0.9),
                                             extra=lambda r: np.isin(cat0["mcc"][r], list(mt)), t=int(tt[mn].min())),
                         1, 8, swipe_share=swipe, ep=ep)
                if mb.any():
                    amts = np.array([C.q(rng.uniform(0.9, 0.99)) * rng.uniform(1.0, 1.5) for _ in range(int(mb.sum()))])
                    src = []
                    for t_, a_ in zip(tt[mb], amts):
                        r = shop_row(int(key), by_state_city, tuple(m for _, ms, _ in STOLEN_BIG for m in ms), int(t_), a_)
                        src.append(r if r is not None else int(era(city_pool(int(key), by_state_city), 1, int(t_))[0]))
                    emit(c, tt[mb], np.array(src, np.int64), 1, 8, amt=np.round(amts, 2), swipe_share=swipe, ep=ep)
                    stats["slow_big_rows"] += int(mb.sum())
            stats["slow_ato"] += 1
        # (9) distributed card testing: the probes are fraud (scenario 11) at any hour; 60% of the cards with an approved
        #     probe are used later (scenario 9) after a skewed delay, at the hours of online fraud
        for _ in range(rng.poisson(PARAMS["distributed_rate"] * years * r_)):
            d = rday(); k1 = int(rng.integers(2, 5))
            pd = d + rng.integers(0, int(rng.integers(1, 4)), k1)
            tt = times_any_hour(pd)
            ep = new_ep(c, "distributed", int(tt.min()))
            declined = probes(c, tt, 11, ep, PARAMS["distributed_probe_amount"], PARAMS["distributed_decline"])
            if (~declined).any() and rng.random() < PARAMS["distributed_use_p"]:
                cd = int((tt.max() + _lognormal_days(rng, PARAMS["distributed_delay"], 1)[0] * DAY) // DAY)
                k2 = int(rng.integers(1, 3))
                if cd + 1 <= day1:
                    cash_emit(c, times_online_fraud(cd + rng.integers(0, 2, k2)), 9, ep); stats["dist_testing_used"] += 1
            stats["dist_testing"] += 1
    # (10) network testing waves: 20-60 cards probed at one small online shop within 6 hours (cards with a higher risk
    #      more often); 35% of the cards with an approved probe are used later, after a skewed delay
    first_day, last_day = int(ts.min() // DAY) + 30, int(ts.max() // DAY) - 1
    card_first = ts[starts] // DAY; card_last = ts[ends - 1] // DAY
    cards_obj = {}
    for w in range(int(PARAMS["waves_per_year"] * (last_day - first_day) * DAY / YEAR)):
        d = int(rng.integers(first_day, last_day))
        active = np.flatnonzero((card_first < d - 30) & (card_last > d + 30))
        if len(active) < PARAMS["wave_cards"][0]: continue
        src_m = probe_src(1, d * DAY)
        size = int(rng.integers(PARAMS["wave_cards"][0], PARAMS["wave_cards"][1] + 1))
        chosen = rng.choice(active, min(len(active), size), replace=False, p=risk[active] / risk[active].sum())
        t_wave = d * DAY + int(rng.integers(0, DAY))
        for ci in chosen:
            s, e = starts[ci], ends[ci]
            if ci not in cards_obj: cards_obj[ci] = Card(s, e)
            C = cards_obj[ci]; c = card[s]; cur["C"] = C
            k1 = int(rng.integers(1, 3))
            tp = np.sort(t_wave + rng.integers(0, PARAMS["wave_hours"] * HOUR, k1))
            ep = new_ep(c, "wave", int(tp[0]))
            declined = probes(c, tp, 12, ep, PARAMS["wave_probe_amount"], PARAMS["wave_decline"], src=np.repeat(src_m, k1))
            stats["wave_cards"] += 1
            if (~declined).any() and rng.random() < PARAMS["wave_use_p"]:
                cd = int(d + _lognormal_days(rng, PARAMS["wave_delay"], 1)[0]); k2 = int(rng.integers(1, 3))
                if cd + 1 <= card_last[ci]:
                    cash_emit(c, times_online_fraud(cd + rng.integers(0, 2, k2)), 10, ep); stats["wave_cashouts"] += 1
        stats["waves"] += 1
    # (5) compromised merchant: 72 breaches a year at large merchants; each exposed card (used there in the 30 days before) gets one
    #     online purchase with the breach's misuse rate (2-10%, x the card's risk) after a skewed delay; 10% of exposed cards get a
    #     new number from the bank 20-90 days after the breach, active 5-14 days later (applied in bank_reaction)
    first, last = ts.min(), ts.max()
    merch_count = np.bincount(cat0["merchant"], minlength=len(dec["merchant"]))
    n_breach = int(PARAMS["breaches_per_year"] * (last - first) / YEAR)
    popular = np.flatnonzero(merch_count >= 2000)
    if len(popular):
        bm = rng.choice(popular, n_breach); bt = rng.uniform(first + 30 * DAY, last, n_breach)
        rows = np.flatnonzero(np.isin(cat0["merchant"], np.unique(bm)))
        rm = cat0["merchant"][rows]; oo = np.lexsort((ts[rows], rm)); rows, rm = rows[oo], rm[oo]
        card_pos = {int(card[st]): i for i, st in enumerate(starts)}
        for m, d in zip(bm, bt):
            seg = rows[np.searchsorted(rm, m, "left"):np.searchsorted(rm, m, "right")]; tseg = ts[seg]
            r = seg[np.searchsorted(tseg, d - 30 * DAY, "left"):np.searchsorted(tseg, d, "left")]
            p_b = rng.uniform(*PARAMS["breach_misuse"])
            for cc in np.unique(card[r]):
                ci = card_pos[int(cc)]
                if ci not in cards_obj: cards_obj[ci] = Card(starts[ci], ends[ci])
                C = cards_obj[ci]; cur["C"] = C
                if rng.random() < PARAMS["reissue_p"]:
                    act = int(d + rng.uniform(*PARAMS["reissue_discovery_days"]) * DAY + rng.uniform(*PARAMS["reissue_activation_days"]) * DAY)
                    if act < C.t1:
                        RE["card"].append(int(cc)); RE["exposure"].append(int(ts[r[card[r] == cc]].max())); RE["activation"].append(act)
                        stats["reissues_planned"] += 1
                if rng.random() < min(0.5, p_b * float(risk[ci])):
                    dd = int(d // DAY + _lognormal_days(rng, PARAMS["breach_delay"], 1)[0])
                    if dd * DAY > C.t1: continue
                    ep = new_ep(cc, "compromised", int(d))
                    cash_emit(cc, times_online_fraud([dd]), 5, ep)
                    stats["compromised_frauds"] += 1
    stats["breaches"] = n_breach
    n_new = sum(len(x) for x in new["card"])
    for f in cat:
        cat[f] = np.r_[cat[f], np.concatenate(new[f]).astype(np.int32)]
    origin = raw["origin"].copy()
    origin[(origin == 0) & np.isin(raw["scenario"], (2, 6))] = 1          # IBM rows moved to a trip's city
    origin[(origin == 0) & retimed] = 3                                    # IBM rows at home moved later (not back from a trip yet)
    for k, dt in (("card", np.int64), ("ts", np.int64), ("amount", np.float32), ("label", np.int8), ("scenario", np.int8), ("foreign", np.int8)):
        raw[k] = np.r_[raw[k], np.concatenate(new[k]).astype(dt)]
    raw["origin"] = np.r_[origin, np.full(n_new, 2, np.int8)]
    raw["retimed"] = np.r_[retimed, np.zeros(n_new, bool)]
    raw["episode"] = np.r_[np.full(n, -1, np.int32), np.concatenate(new["episode"]).astype(np.int32)]
    raw["probe"] = np.r_[np.zeros(n, bool), np.concatenate(new["probe"]).astype(bool)]
    raw["episodes"] = {k: np.array(v, np.int64) for k, v in EP.items()}
    raw["reissues"] = {k: np.array(v, np.int64) for k, v in RE.items()}
    info = {k: int(v) for k, v in stats.items()}
    if travel_out:
        info["travel_hours_out_median_x10"] = int(round(10 * float(np.median(travel_out))))
        info["travel_hours_out_p90_x10"] = int(round(10 * float(np.quantile(travel_out, 0.9))))
    info["cities_with_coordinates"] = len(geo); info["episodes"] = len(EP["card"])
    info["online_legit_night_share_x1000"] = int(round(1000 * night_online_legit))
    info["cash_out_groups"] = {name: [int(dec["mcc"][m]) for m, _ in avail] for name, avail, _ in co_groups}
    info["cash_out_risky_shops_ibm_fraud_rows"] = {int(dec["mcc"][m]): P["risky_fraud"] for m, P in co_mcc.items()}
    raw["info"]["injection"] = info
    raw["info"]["name"] = "added_scenarios"
    raw["info"]["history_fraud_scenarios"] = list(ADDED_FRAUD); raw["info"]["confuser_scenarios"] = list(LOOK_ALIKES)
    raw["info"]["probe_scenarios"] = list(PROBES); raw["info"]["after_block_scenarios"] = list(AFTER_BLOCK)
    raw["info"]["notes"] += " + added scenarios: " + json.dumps(info)
    print("added scenarios:", info, flush=True)
    return raw


def bank_reaction(raw, seed=0):
    """Step 2: the bank reacts. raw = the output of inject_scenarios. A card has a sequence of numbers (the card id stays: one
    account). Each fraud episode (added, or IBM's own fraud cut at 7-day gaps) uses the number that was valid when the card was
    compromised. When the bank notices it:
      card-testing burst  0-24 h after its first purchase (no purchase: not noticed);
      stolen card         reported 0.1-4 h after the thief's last purchase;
      counterfeit card    0.1-24 h after its last purchase (the episode lasts until it is noticed);
      breach, distributed testing, wave: from the statement, lognormal median 10 days after the (first) purchase;
      IBM's own fraud     0-7 days after the episode's last row.
    Episodes are handled in the order they are noticed. The number an episode uses dies at the first block or reissue after its
    compromise. If it died before the episode's first purchase, or before the episode was noticed, the episode raises no block
    (an earlier block covers it). Otherwise the bank blocks the number when it notices it, and the new card arrives 2-7 days later
    (15%: 7-14 days):
      - the fraudster's attempts after the block, declined ("Do Not Honor", "Card Blocked", "Suspected Fraud", "Stolen Card"):
        stolen and counterfeit cards and testing bursts 2-10 tries, the others their planned attempts plus, 30% of the time,
        1-3 more, IBM's fraud 1-5; planned rows are declined first, missing tries are added near his last ones; all of them are
        fraud, scenario 13 (IBM's own declined rows keep scenario 1); planned rows beyond the tries never happen;
      - fraudulent recurring charges (scenario 14) continue on the new card (recurring merchants get the new card details,
        fraudulent ones too);
      - the owner has no card from the block until the new card (a stolen card: no in-person purchases from the theft): his
        subscriptions and the online merchants he used in the last 60 days continue (card on file; 10% of these merchants do not get
        the new details and decline), after a bank rule's block he tries 1-2 times with the old card in half the cases (declined,
        legit), all his other transactions never happen.
    Reissues after a breach (10% of exposed cards): the old number dies when the new card is active; nothing else changes for the
    owner. False declines: 0.1% of the other legit purchases are declined by the bank's rules ("suspected fraud" 60%, "do not honour"
    40%), more often on trips (x10), above the card's 95th percentile (x5), at a new merchant (x3) and at night (x2); 70% of them
    are tried again 2-30 min later and approved. raw["origin"] marks declined IBM rows (4)."""
    import bisect
    rng = np.random.default_rng(seed + 34)
    _reorder_rows(raw, np.lexsort((raw["ts"], raw["card"])))
    cat, dec = raw["cat"], raw["decode"]
    card, ts, label, sc, amount = raw["card"], raw["ts"], raw["label"], raw["scenario"], raw["amount"]
    ep_row, probe, origin = raw["episode"], raw["probe"], raw["origin"]
    E, R = raw["episodes"], raw["reissues"]
    n = len(card)
    code = {c_: _code(dec, "errors", c_) for c_ in BLOCK_CODES}
    ERR_NONE = int(np.bincount(cat["errors"]).argmax())
    CHIP_ONLINE = dec["chip"].index("Online Transaction")
    H, DAY = 3600, 86400
    drop = np.zeros(n, bool); decl = np.full(n, -1, np.int32); to13 = np.zeros(n, bool)
    new_rows = []                       # (template row, time, amount, label, scenario, decline code, episode)
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]]); ends = np.r_[starts[1:], n]
    fr = np.flatnonzero(ep_row >= 0)
    fr = fr[np.lexsort((ts[fr], ep_row[fr]))]
    ids, e0, ecnt = np.unique(ep_row[fr], return_index=True, return_counts=True)
    ep_rows = {int(i): fr[a:a + c_] for i, a, c_ in zip(ids, e0, ecnt)}
    by_card_ep, by_card_re = {}, {}
    for i, c_ in enumerate(E["card"]):
        by_card_ep.setdefault(int(c_), []).append(i)
    for i, c_ in enumerate(R["card"]):
        by_card_re.setdefault(int(c_), []).append(i)
    st = {k: 0 for k in ("blocks", "episodes", "episodes_dead_before_purchase", "episodes_covered", "fraud_rows_declined", "fraud_rows_dropped",
                         "tries_added", "owner_rows_dropped", "owner_rows_dropped_stolen_card", "owner_tries_declined", "card_on_file_kept",
                         "card_on_file_declined", "subscription_rows_kept", "reissues_applied", "reissues_moot", "false_declines",
                         "false_decline_retries", "recurring_fraud_rows_after_block")}
    st.update({f"blocks_{k}": 0 for k in EP_KINDS})
    delays, gaps, tries_per_block = {k: [] for k in EP_KINDS}, [], []
    U = lambda lo, hi: float(rng.uniform(lo, hi))

    def new_card_days():
        (p1, a1, b1), (_, a2, b2) = BANK["new_card"]
        return U(a1, b1) if rng.random() < p1 else U(a2, b2)

    def detect(kind, purch):
        if not len(purch): return None
        rule = BANK["detect"][kind]
        if rule[0] == "first_purchase_h": return int(ts[purch[0]] + U(rule[1], rule[2]) * H)
        if rule[0] == "last_h": return int(ts[purch[-1]] + U(rule[1], rule[2]) * H)
        if rule[0] == "last_d": return int(ts[purch[-1]] + U(rule[1], rule[2]) * DAY)
        return int(ts[purch[0]] + _lognormal_days(rng, rule[1:], 1)[0] * DAY)

    def codes_for(kind, k):
        return [code[x] for x in _pick(rng, BANK["codes_stolen"] if kind == "stolen" else BANK["codes_other"], k)]

    def after_death(ep, x):
        """The episode's attempts at or after x, when its number is dead: declined tries first (codes), the rest never happen;
        missing tries are added after his last attempt (active episodes: 2-10 in all)."""
        att = ep["att"]
        late = att[(ts[att] >= x) & ~drop[att] & (decl[att] < 0)]
        kind = ep["kind"]
        if kind in ACTIVE_KINDS:
            n_try = int(rng.integers(BANK["tries_active"][0], BANK["tries_active"][1] + 1))
        elif kind == "ibm":
            n_try = int(rng.integers(BANK["tries_ibm"][0], BANK["tries_ibm"][1] + 1))
        else:
            p_, a_, b_ = BANK["tries_statement_extra"]
            n_try = len(late) + (int(rng.integers(a_, b_ + 1)) if rng.random() < p_ else 0)
        dec_rows, gone = late[:n_try], late[n_try:]
        decl[dec_rows] = codes_for(kind, len(dec_rows)); drop[gone] = True
        ours = dec_rows[sc[dec_rows] != 1]; to13[ours] = True
        st["fraud_rows_declined"] += len(dec_rows); st["fraud_rows_dropped"] += len(gone)
        extra = n_try - len(dec_rows)
        if extra > 0 and len(att):
            tmpl = att[-min(3, len(att)):]
            t_last = int(max(x, ts[dec_rows].max() if len(dec_rows) else x))
            t_ = t_last + int((U(5 / 60, 1.0) if kind in ACTIVE_KINDS else U(1.0, 72.0)) * H)
            for j in range(extra):
                r = int(tmpl[j % len(tmpl)])
                if cat["chip"][r] != CHIP_ONLINE:                     # in person: shop hours
                    tod = t_ % DAY
                    if tod >= 23 * H or tod < 7 * H: t_ = (t_ // DAY + (1 if tod >= 23 * H else 0)) * DAY + int(U(7.5, 10) * H)
                new_rows.append((r, t_, round(float(amount[r]) * U(0.8, 1.2), 2), 1, 13, codes_for(kind, 1)[0], ep["id"]))
                t_ += int(np.clip(np.exp(np.log(1.5) + rng.standard_normal()), 5 / 60, 24) * H)
            st["tries_added"] += extra
        return n_try

    def owner_gap(s, e, ep, T, NC):
        kind = ep["kind"]
        start = ep["theft"] if kind == "stolen" and ep["theft"] >= 0 else T
        own = s + np.flatnonzero((label[s:e] == 0) & (ts[s:e] >= start) & (ts[s:e] < NC))
        own = own[~drop[own] & (decl[own] < 0)]
        if not len(own): return
        online = cat["chip"][own] == CHIP_ONLINE
        if kind == "stolen":                                  # he has no card from the theft: no in-person purchases
            pre = (ts[own] < T) & ~online
            drop[own[pre]] = True; st["owner_rows_dropped_stolen_card"] += int(pre.sum())
            keep = ts[own] >= T; own, online = own[keep], online[keep]
        hist = s + np.flatnonzero((label[s:e] == 0) & (ts[s:e] >= T - BANK["card_on_file_days"] * DAY) & (ts[s:e] < T)
                                  & (cat["chip"][s:e] == CHIP_ONLINE))
        subs = sc[own] == 7
        cof = online & np.isin(cat["merchant"][own], cat["merchant"][hist]) & ~subs
        fail = cof & (rng.random(len(own)) < BANK["card_on_file_not_updated"])
        if fail.any():
            decl[own[fail]] = codes_for(kind if kind == "stolen" else "other", int(fail.sum())) if kind == "stolen" else \
                [code[x] for x in _pick(rng, (("Card Blocked,", .5), ("Do Not Honor,", .5)), int(fail.sum()))]
        st["subscription_rows_kept"] += int(subs.sum()); st["card_on_file_kept"] += int((cof & ~fail).sum())
        st["card_on_file_declined"] += int(fail.sum())
        rest = own[~subs & ~cof]
        p = BANK["owner_try_p"]["stolen" if kind == "stolen" else "rule"]
        n_t = int(rng.integers(BANK["owner_tries"][0], BANK["owner_tries"][1] + 1)) if rng.random() < p else 0
        tries = rest[:n_t]
        if len(tries): decl[tries] = [code[x] for x in _pick(rng, (("Card Blocked,", .5), ("Do Not Honor,", .5)), len(tries))]
        drop[rest[n_t:]] = True
        st["owner_tries_declined"] += len(tries); st["owner_rows_dropped"] += max(0, len(rest) - n_t)

    for s, e in zip(starts, ends):
        c = int(card[s])
        eps = []
        for i in by_card_ep.get(c, []):
            rows = ep_rows.get(i, np.zeros(0, np.int64))
            att = rows[sc[rows] != 14]                        # the fraudster's attempts (recurring charges stay on file)
            kind = EP_KINDS[int(E["kind"][i])]
            purch = att[~probe[att]]
            eps.append(dict(id=i, kind=kind, comp=int(E["compromise"][i]), theft=int(E["theft"][i]), att=att, purch=purch,
                            T=detect(kind, purch), rec=rows[sc[rows] == 14]))
        ibm = s + np.flatnonzero((label[s:e] == 1) & (sc[s:e] == 1))
        if len(ibm):
            for g in np.split(ibm, np.flatnonzero(np.diff(ts[ibm]) > BANK["ibm_cluster_days"] * DAY) + 1):
                eps.append(dict(id=-1, kind="ibm", comp=int(ts[g[0]]), theft=-1, att=g, purch=g, T=detect("ibm", g), rec=g[:0]))
        res = [(int(R["activation"][i]), int(R["exposure"][i])) for i in by_card_re.get(c, [])]
        if not eps and not res: continue
        events = [((ep["T"] if ep["T"] is not None else (int(ts[ep["att"][-1]]) if len(ep["att"]) else ep["comp"])), 0, j)
                  for j, ep in enumerate(eps)] + [(a_, 1, j) for j, (a_, _) in enumerate(res)]
        events.sort()
        deaths, gaps_c = [], []                               # death times of the card's numbers; owner gaps [T, NC)
        for t_ev, typ, j in events:
            if typ == 1:
                act, exp_ = res[j]
                k_ = bisect.bisect_right(deaths, exp_)
                if k_ < len(deaths) and deaths[k_] <= act: st["reissues_moot"] += 1; continue
                bisect.insort(deaths, act); st["reissues_applied"] += 1
                continue
            ep = eps[j]; st["episodes"] += 1
            comp = ep["comp"]
            for a_, b_ in gaps_c:                             # compromised while the card was being replaced: the old number
                if a_ <= comp < b_: comp = a_ - 1
            k_ = bisect.bisect_right(deaths, comp)
            d_ = deaths[k_] if k_ < len(deaths) else None
            first_p = int(ts[ep["purch"][0]]) if len(ep["purch"]) else None
            if d_ is not None and (first_p is None or first_p >= d_):
                after_death(ep, d_); st["episodes_dead_before_purchase"] += 1; continue
            if ep["T"] is None: continue
            if d_ is not None and d_ < ep["T"]:
                after_death(ep, d_); st["episodes_covered"] += 1; continue
            T = ep["T"]; NC = T + int(new_card_days() * DAY)
            bisect.insort(deaths, T); gaps_c.append((T, NC))
            tries_per_block.append(after_death(ep, T))
            owner_gap(s, e, ep, T, NC)
            st["recurring_fraud_rows_after_block"] += int((ts[ep["rec"]] >= T).sum()) if len(ep["rec"]) else 0
            st["blocks"] += 1; st[f"blocks_{ep['kind']}"] += 1
            first_row = int(ts[ep["att"][0]]) if len(ep["att"]) else T
            delays[ep["kind"]].append((T - (ep["theft"] if ep["kind"] == "stolen" and ep["theft"] >= 0 else first_row)) / H)
            gaps.append((NC - T) / DAY)
    # false declines by the bank's rules (legit), weighted towards unusual purchases
    cand = np.flatnonzero((label == 0) & ~drop & (decl < 0) & (amount > 0) & (cat["errors"] == ERR_NONE))
    if len(cand):
        W_ = BANK["false_decline_weights"]
        lg = np.flatnonzero((label == 0) & (amount > 0))
        oq = lg[np.lexsort((amount[lg], card[lg]))]; cq = card[oq]
        a0 = np.searchsorted(cq, card[starts], "left"); a1 = np.searchsorted(cq, card[starts], "right")
        q95 = np.where(a1 > a0, amount[oq[np.clip(a0 + ((a1 - a0 - 1) * 0.95).astype(np.int64), 0, len(oq) - 1)]], np.inf)
        card_q95 = np.repeat(q95, ends - starts)
        mk = card.astype(np.int64) * (int(cat["merchant"].max()) + 1) + cat["merchant"]
        newm = np.zeros(n, bool); newm[np.unique(mk, return_index=True)[1]] = True
        w = (np.where(np.isin(sc[cand], (2, 6)), W_["trip"], 1.0) * np.where(amount[cand] > card_q95[cand], W_["large"], 1.0)
             * np.where(newm[cand], W_["new_merchant"], 1.0) * np.where(ts[cand] % DAY < 6 * H, W_["night"], 1.0))
        k = int(BANK["false_decline_rate"] * len(cand))
        pick = rng.choice(cand, k, replace=False, p=w / w.sum())
        retry = rng.random(k) < BANK["false_decline_retry"]
        cds = [code[x] for x in _pick(rng, (("Suspected Fraud,", .6), ("Do Not Honor,", .4)), k)]
        for r, rt, cd in zip(pick, retry, cds):
            if rt: new_rows.append((int(r), int(ts[r] - int(U(2, 30) * 60)), float(amount[r]), 0, int(sc[r]), cd, -1))
            else: decl[r] = cd
        st["false_declines"] = k; st["false_decline_retries"] = int(retry.sum())
    # apply: declined rows get their code, our declined fraud rows become scenario 13, IBM legit rows declined are origin 4
    bl = decl >= 0
    cat["errors"][bl] = decl[bl]
    sc[to13] = 13
    origin[bl & (label == 0) & np.isin(origin, (0, 1, 3))] = 4
    st["scenario13_from_planned"] = int(to13.sum())
    add, add_cat = None, None
    if new_rows:                  # added tries and false-decline records, built from their template rows before anything is dropped
        src = np.array([r for r, *_ in new_rows], np.int64); k = len(src)
        add = dict(card=card[src].copy(), ts=np.array([t for _, t, *_ in new_rows], np.int64) // 60 * 60,
                   amount=np.array([a for _, _, a, *_ in new_rows], np.float32), label=np.array([x[3] for x in new_rows], np.int8),
                   scenario=np.array([x[4] for x in new_rows], np.int8), foreign=raw["foreign"][src].copy(),
                   origin=np.full(k, 2, np.int8), retimed=np.zeros(k, bool), episode=np.array([x[6] for x in new_rows], np.int32),
                   probe=np.zeros(k, bool))
        add_cat = {f: v[src].copy() for f, v in cat.items()}
        add_cat["errors"] = np.array([x[5] for x in new_rows], np.int32)
    st["rows_dropped"] = int(drop.sum()); st["ibm_rows_dropped"] = int((drop & np.isin(origin, (0, 1, 3, 4))).sum())
    _reorder_rows(raw, np.flatnonzero(~drop))
    if add is not None:
        for k_, v in add.items():
            if k_ in raw: raw[k_] = np.r_[raw[k_], v.astype(raw[k_].dtype)]
        for f in raw["cat"]:
            raw["cat"][f] = np.r_[raw["cat"][f], add_cat[f].astype(raw["cat"][f].dtype)]
    st["rows_added"] = len(new_rows)
    for k_, v in delays.items():
        if v: st[f"detect_hours_median_{k_}"] = int(round(float(np.median(v))))
    if gaps: st["new_card_days_median_x10"] = int(round(10 * float(np.median(gaps)))); st["new_card_days_p90_x10"] = int(round(10 * float(np.quantile(gaps, 0.9))))
    if tries_per_block: st["tries_per_block_quartiles"] = [int(x) for x in np.quantile(tries_per_block, [0.25, 0.5, 0.75])]
    st["decline_codes"] = {c_: dict(fraud=int(((raw["cat"]["errors"] == code[c_]) & (raw["label"] == 1)).sum()),
                                    legit=int(((raw["cat"]["errors"] == code[c_]) & (raw["label"] == 0)).sum())) for c_ in BLOCK_CODES}
    raw["info"]["bank_reaction"] = st
    raw["info"]["name"] = "benchmark"
    raw["info"]["notes"] += " + the bank's reaction (blocks, declined tries, new cards, card on file, reissues, false declines): " + json.dumps(st)
    print("bank reaction:", st, flush=True)
    return raw


def load_benchmark():
    """The benchmark: IBM's data + the added scenarios + the bank's reaction, and the order of same-minute rows (tie_rank)."""
    raw = bank_reaction(inject_scenarios(load_tabformer()))
    raw["tie"] = tie_rank(raw)
    return raw

