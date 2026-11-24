"""The sequence model behind every released transformer (Teacher 5M, Student 2.6M, Student 0.4M).

One transaction = one vector: every field has its own embedding table, a one-layer transformer mixes the fields of the
transaction, and an MLP projects them to d_model (the field encoder). The four numeric inputs (signed log amount, log gap,
hour sine and cosine) are projected and added. A causal transformer then reads the card's last L transactions, so the output at
position i only sees transactions 0..i of the window. Heads: the fraud logit (the score), a 64-number embedding of the card's
history, and two training-only heads that predict the next transaction's MCC and amount bin.
"""
import torch
import torch.nn as nn


def causal_pad_mask(pad, n_heads):
    """Float attention mask [B*H, L, L]: query i may attend key j if j <= i and (j is a real transaction or j == i)."""
    B, L = pad.shape
    causal = torch.ones(L, L, dtype=torch.bool, device=pad.device).tril()
    allow = causal[None] & (~pad)[:, None, :]
    allow = allow | torch.eye(L, dtype=torch.bool, device=pad.device)[None]
    m = torch.zeros(B, L, L, device=pad.device).masked_fill(~allow, float("-inf"))
    return m.repeat_interleave(n_heads, dim=0)


class FieldEncoder(nn.Module):
    """One vector per transaction from its field ids."""

    def __init__(self, vocab_sizes, d_field, d_model, field_layers=1, n_heads=2, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(v, d_field, padding_idx=0) for v in vocab_sizes)
        nf = len(vocab_sizes)
        self.field_pos = nn.Parameter(torch.zeros(nf, d_field))
        self.field_tf = None
        if field_layers > 0:
            layer = nn.TransformerEncoderLayer(d_field, n_heads, 2 * d_field, dropout, batch_first=True, norm_first=True)
            self.field_tf = nn.TransformerEncoder(layer, field_layers, enable_nested_tensor=False)
        self.proj = nn.Sequential(nn.Linear(nf * d_field, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, tokens):  # [B, L, F]
        B, L, Fn = tokens.shape
        x = torch.stack([e(tokens[..., i]) for i, e in enumerate(self.embs)], dim=2) + self.field_pos  # [B, L, F, d_field]
        if self.field_tf is not None:
            x = self.field_tf(x.reshape(B * L, Fn, -1)).reshape(B, L, Fn, -1)
        return self.proj(x.reshape(B, L, -1))


class SeqModel(nn.Module):
    """Field encoder -> causal transformer over the card's transactions -> fraud logit, embedding, next-transaction heads."""

    def __init__(self, vocab_sizes, d_model=128, n_layers=4, n_heads=4, d_field=32, field_layers=1, L_max=128, emb_dim=64,
                 n_mcc=111, dropout=0.1, next_from_emb=True, n_amt=0, n_num=0, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cfg = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_field=d_field, field_layers=field_layers,
                        L_max=L_max, emb_dim=emb_dim, n_mcc=n_mcc, dropout=dropout, next_from_emb=next_from_emb, n_amt=n_amt,
                        n_num=n_num, d_ff=d_ff)
        self.n_num = n_num
        self.n_heads = n_heads
        self.num_proj = nn.Sequential(nn.Linear(n_num, d_model), nn.GELU(), nn.Linear(d_model, d_model)) if n_num else None
        self.fe = FieldEncoder(vocab_sizes, d_field, d_model, field_layers, dropout=dropout)
        self.pos = nn.Embedding(L_max + 1, d_model)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
            for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.fraud_head = nn.Linear(d_model, 1)
        self.emb_head = nn.Linear(d_model, emb_dim)
        self.next_from_emb = next_from_emb          # True: the next-transaction heads read the embedding, so it is trained too
        self.next_head = nn.Linear(emb_dim if next_from_emb else d_model, n_mcc)
        self.amt_head = nn.Linear(emb_dim if next_from_emb else d_model, n_amt) if n_amt else None

    def forward(self, tokens, pad, num=None):
        """tokens [B, L, F] field ids (0 = padding), pad [B, L] True at padding, num [B, L, n_num] numeric inputs."""
        x = self.fe(tokens)
        if self.num_proj is not None:
            x = x + self.num_proj(num)
        L = x.shape[1]
        x = x + self.pos(torch.arange(L, device=x.device))[None]
        mask = causal_pad_mask(pad, self.n_heads)
        hidden = []
        for layer in self.layers:
            x = layer(x, src_mask=mask)
            hidden.append(x)
        h = self.norm(x)
        emb = self.emb_head(h)
        src = emb if self.next_from_emb else h
        return dict(logit=self.fraud_head(h).squeeze(-1), emb=emb, next_logits=self.next_head(src),
                    amt_logits=self.amt_head(src) if self.amt_head is not None else None, hidden=hidden, h=h)


def keep_layers(model: SeqModel, keep):
    """A copy of `model` with only the listed transformer layers (in that order); every other weight is copied unchanged.
    Student 2.6M starts as Teacher 5M with layers 0, 1 and 5 of its 6."""
    cfg = dict(model.cfg); cfg["n_layers"] = len(keep)
    out = SeqModel([e.num_embeddings for e in model.fe.embs], **cfg)
    new = {}
    for k, v in model.state_dict().items():
        if k.startswith("layers."):
            i = int(k.split(".")[1])
            if i in keep:
                new[k.replace(f"layers.{i}.", f"layers.{keep.index(i)}.", 1)] = v
        else:
            new[k] = v
    out.load_state_dict(new)
    return out


def n_params(model, non_embedding=False):
    total = sum(p.numel() for p in model.parameters())
    if non_embedding:
        total -= sum(p.numel() for mod in model.modules() if isinstance(mod, nn.Embedding) for p in mod.parameters())
    return total
