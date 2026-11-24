"""Score the benchmark's rows with a released model and write a file that evaluate.py reads.

    python models/score.py --arrays arrays/ --model weights/student-0.4m-distilled-seed0.safetensors --split test --out scores/s0.parquet

Every row is scored once, in the first window that covers it in the window's second half: windows of L = 128 transactions
end at the card's rows L/2 - 1, L - 1, 3L/2 - 1, ... (and at the card's last row), and a row takes the score of the first such
window in which it is one of the last L/2 transactions. So every score sees at least 64 earlier transactions of the card (or all
of them, for a card's first rows), and no score depends on how rows were batched.
Writes a parquet file with columns row_id and score (the fraud logit; higher = more suspicious).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import TxnData  # noqa: E402
from weights import load_model  # noqa: E402

SPLITS = {"train": 0, "validation": 1, "test": 2}


def window_ends(card_start, card_end, rows, length):
    """For each row, the end of the first stride-L/2 window that has the row in its second half."""
    h = length // 2
    if length < 2 or length % 2:
        raise ValueError("an even window length >= 2 is required")
    return np.minimum(card_start[rows] + ((rows - card_start[rows]) // h + 1) * h - 1, card_end[rows] - 1)


@torch.no_grad()
def first_window_scores(model, data, rows, length=128, batch_size=128, device="cpu", embeddings=False):
    """Fraud logit (and optionally the 64-number embedding) of every row in `rows`, each from its first window."""
    rows = np.asarray(rows)
    ends = window_ends(data.card_start, data.card_end, rows, length)
    unique, inverse = np.unique(ends, return_inverse=True)
    order = np.argsort(inverse, kind="stable"); sorted_inverse = inverse[order]
    score = np.empty(len(rows), np.float32); emb = None
    use_bf16 = str(device).startswith("cuda")             # the study scored on a GPU with bfloat16 autocast
    for lo in range(0, len(unique), batch_size):
        hi = min(lo + batch_size, len(unique))
        b = data.batch(unique[lo:hi], length)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            o = model(b["tokens"].to(device), b["pad"].to(device), **({"num": b["num"].to(device)} if model.n_num else {}))
        ids = order[np.searchsorted(sorted_inverse, lo):np.searchsorted(sorted_inverse, hi)]
        local = inverse[ids] - lo; pos = rows[ids] - ends[ids] + length - 1
        assert ((pos >= length // 2) & (pos < length)).all() or (pos[rows[ids] - data.card_start[rows[ids]] < length // 2] >= 0).all()
        score[ids] = o["logit"][local, pos].float().cpu().numpy()
        if embeddings:
            e = o["emb"][local, pos].float().cpu().numpy()
            if emb is None:
                emb = np.empty((len(rows), e.shape[1]), np.float32)
            emb[ids] = e
    assert np.isfinite(score).all()
    return (score, emb) if embeddings else score


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", required=True, help="the folder written by models/prepare.py")
    ap.add_argument("--model", required=True, help="a .safetensors file from weights/")
    ap.add_argument("--split", default="test", choices=list(SPLITS) + ["all"])
    ap.add_argument("--out", required=True, help="output .parquet (row_id, score)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model, cfg = load_model(a.model, a.device)
    data = TxnData(a.arrays, drop_fields=cfg["inputs"]["drop_fields"], num_groups=cfg["inputs"]["num_groups"])
    assert data.fields == cfg["inputs"]["fields"] and data.vocab_sizes == cfg["model"]["vocab_sizes"], "arrays do not match the model"
    rows = np.arange(len(data.card)) if a.split == "all" else np.flatnonzero(data.split == SPLITS[a.split])
    s = first_window_scores(model, data, rows, cfg["inputs"]["window"], a.batch_size, a.device)
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.table({"row_id": pa.array(rows.astype(np.int64)), "score": pa.array(s)}), a.out)
    print(f"scored {len(rows):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
