"""Windows over each card's transactions, read from the arrays written by models/prepare.py (memory-mapped).

A window ending at row e holds the card's L transactions up to and including e (left-padded with -1 for a card's first rows).
The split is the benchmark's: before 2018 = training; January-June 2018 = training for 80% of the cards and validation for the
other 20% (rng 123, the same cards as the dataset's `split` column); from July 2018 = test.
"""
import json
import os

import numpy as np
import torch


class TxnData:
    def __init__(self, path, drop_fields=("flags",), num_groups=("time",), val_card_frac=0.2):
        """drop_fields: input fields the model does not read ("flags" = the four "first time on this card" flags, which every
        released model drops). num_groups: numeric input groups added to each transaction ("time" = the 4 numeric columns)."""
        self.path = path
        self.meta = m = json.load(open(os.path.join(path, "meta.json")))
        ld = lambda k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r")
        self.tokens_all = ld("tokens")                                   # [N, 14] int16, memory-mapped
        self.card = np.asarray(ld("card")); self.ts = np.asarray(ld("ts")); self.amount = np.asarray(ld("amount"))
        self.label = np.asarray(ld("label")).astype(np.int8)
        self.scenario = np.asarray(ld("scenario")); self.foreign = np.asarray(ld("foreign"))
        starts = np.asarray(ld("card_starts"))
        n = len(self.card)
        lens = np.diff(np.r_[starts, n])
        self.card_start = np.repeat(starts, lens)
        self.card_end = np.repeat(np.r_[starts[1:], n], lens)          # exclusive
        drop = set(drop_fields or [])
        if "flags" in drop:
            drop |= set(m["flag_fields"])
        self.keep = [i for i, f in enumerate(m["fields"]) if f not in drop]
        self.fields = [m["fields"][i] for i in self.keep]
        self.vocab_sizes = [m["vocab_sizes"][i] for i in self.keep]
        self.fidx = {f: i for i, f in enumerate(self.fields)}
        self.id_field_idx = [self.fidx[f] for f in m["id_fields"] if f in self.fidx]   # merchant, city: dropped out in training
        all_idx = {f: i for i, f in enumerate(m["fields"])}
        self.n_next = m["vocab_sizes"][all_idx[m["next_field"]]]
        self.n_amt = m["vocab_sizes"][all_idx["amt"]]
        # split: 0 training, 1 validation, 2 test
        self.split = (self.ts >= m["t_val"]).astype(np.int8) + (self.ts >= m["t_test"]).astype(np.int8)
        cards = np.unique(self.card)
        vc = np.random.default_rng(123).choice(cards, int(len(cards) * val_card_frac), replace=False)
        self.split[(self.split == 1) & ~np.isin(self.card, vc)] = 0
        last = np.r_[self.card[1:] != self.card[:-1], True]
        nf = np.asarray(self.tokens_all[:, all_idx[m["next_field"]]])
        na = np.asarray(self.tokens_all[:, all_idx["amt"]])
        self.next_cat = np.where(last, -1, np.r_[nf[1:], 0]).astype(np.int16)   # the card's next MCC (training target)
        self.next_amt = np.where(last, -1, np.r_[na[1:], 0]).astype(np.int16)   # the card's next amount bin (training target)
        self._key = self.card.astype(np.int64) * 10**10 + self.ts
        self.num_names = m.get("num_features", []); self.num_groups_all = m.get("num_groups", {})
        p = os.path.join(path, "numf.npy")
        self.numf = np.load(p, mmap_mode="r") if (self.num_names and os.path.exists(p)) else None
        self.num_cols = []
        for g in num_groups or []:
            if g not in self.num_groups_all or self.numf is None:
                raise ValueError(f"numeric group {g!r} not in {path}")
            self.num_cols += [self.num_names.index(x) for x in self.num_groups_all[g]]
        self.n_num = len(self.num_cols)

    # ---------- windows
    def window_idx(self, ends, L):
        """[B, L] row indices of the L transactions ending at `ends` (inclusive), -1 = left padding."""
        ends = np.asarray(ends)
        idx = ends[:, None] - np.arange(L - 1, -1, -1)[None, :]
        return np.where(idx >= self.card_start[ends][:, None], idx, -1)

    def gather(self, idx):
        pad = idx < 0
        safe = np.where(pad, 0, idx)
        tok = np.asarray(self.tokens_all[safe.ravel()])[:, self.keep].reshape(*idx.shape, len(self.keep))
        return np.where(pad[..., None], 0, tok).astype(np.int64), pad

    def batch(self, ends, L):
        idx = self.window_idx(ends, L)
        tok, pad = self.gather(idx)
        safe = np.where(pad, 0, idx)
        f = lambda arr: torch.from_numpy(np.where(pad, -1, arr[safe]).astype(np.int64))
        out = dict(tokens=torch.from_numpy(tok), pad=torch.from_numpy(pad), label=f(self.label),
                   next_cat=f(self.next_cat), next_amt=f(self.next_amt), idx=idx)
        if self.n_num:
            x = np.asarray(self.numf[safe.ravel()], np.float32).reshape(*idx.shape, -1)
            x[pad] = 0
            out["num"] = torch.from_numpy(np.ascontiguousarray(x[..., self.num_cols]))
        return out

    # ---------- samplers (training)
    def train_sampler(self, L, batch_size, pos_frac=0.3, split=0, seed=0):
        """Window end rows. A share `pos_frac` of the windows has a fraud in its supervised (second) half."""
        rng = np.random.default_rng(seed)
        rows = np.flatnonzero(self.split == split)
        frauds = rows[self.label[rows] == 1]
        N = len(self.card)

        def near(f):   # window ends that put row f in the supervised half (same card and split)
            e = np.minimum(f + rng.integers(0, max(L // 2, 1), size=len(f)), N - 1)
            ok = (self.card[e] == self.card[f]) & (self.split[e] == split)
            return np.where(ok, e, f)
        while True:
            n_pos = rng.binomial(batch_size, pos_frac) if len(frauds) else 0
            e_neg = rng.choice(rows, batch_size - n_pos)
            f = rng.choice(frauds, n_pos) if n_pos else np.zeros(0, dtype=np.int64)
            yield np.r_[e_neg, near(f)]

    def same_card_pair(self, ends, rng, min_days=30, split=0, tries=8):
        """For each end row, another row of the same card (same split), preferably at least `min_days` away."""
        ends = np.asarray(ends)
        lo, hi = self.card_start[ends], self.card_end[ends]
        best = ends.copy(); found = np.zeros(len(ends), bool)
        for _ in range(tries):
            c = lo + (rng.random(len(ends)) * (hi - lo)).astype(np.int64)
            ok = (self.split[c] == split) & (np.abs(self.ts[c] - self.ts[ends]) >= min_days * 86400) & ~found
            best[ok] = c[ok]; found |= ok
        return best, found

    def eval_rows(self, split, n_neg, seed=0, n_neg_foreign=20000):
        """All frauds of the split + sampled legit rows, stratified (at home / abroad) so that legit transactions abroad are
        well represented. Weights restore the true class balance."""
        rng = np.random.default_rng(seed)
        rows = np.flatnonzero(self.split == split)
        pos = rows[self.label[rows] == 1]
        out_r, out_w = [pos], [np.ones(len(pos))]
        negs = rows[self.label[rows] == 0]
        for stratum, k in ((0, n_neg), (1, n_neg_foreign)):
            s = negs[self.foreign[negs] == stratum]
            if len(s) == 0:
                continue
            pick = rng.choice(s, min(k, len(s)), replace=False)
            out_r.append(pick); out_w.append(np.full(len(pick), len(s) / len(pick)))
        r = np.concatenate(out_r); w = np.concatenate(out_w)
        o = np.argsort(r)
        return r[o], w[o]
