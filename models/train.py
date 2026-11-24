"""Train the released transformers: Teacher 5M, and the students distilled from it (or trained from the labels only).

    python models/train.py --arrays arrays/ --recipe teacher-5m --seed 0 --out runs/teacher-5m-seed0
    python models/train.py --arrays arrays/ --recipe student-0.4m-distilled --seed 0 --teacher runs/teacher-5m-seed0/model.safetensors \
        --out runs/student-0.4m-distilled-seed0

The recipes (models/recipes.json) are the settings of the released models. Every option can be overridden on the command line.
What the losses are:
  every model   fraud: binary cross-entropy on the rows in the second half of each window (at least 64 earlier transactions of the
                card, or all of them); next MCC and next amount bin of the card (cross-entropy, from the embedding)
  teacher       + same-card contrastive loss on the embedding (two windows of the same card at least 30 days apart; trains the
                embedding head only)
  students      + distillation from the teacher (frozen): its fraud logit (temperature-softened cross-entropy, plus a small
                mean-squared error on the logits); Student 0.4M sees the same id-dropped input as its teacher ("consistent");
                Student 2.6M starts as the teacher with layers 0, 1 and 5 of 6, and its embedding head also learns the teacher's
                embedding (cosine, pairwise distances, contrastive, and the teacher's frozen next-MCC head on it)
  labels-only   the 0.4M student with every distillation weight at 0 (the teacher is loaded, not used: the student then starts
                from the same random weights as in our run)
Merchant and city ids are replaced by "missing" with probability 0.1 during training. The checkpoint kept is the one with the
best area under the precision-recall curve on a validation sample (every validation fraud except IBM's own + 4,000 legit at home +
2,000 abroad, weighted), checked every eval_every steps. On a GPU the forward passes run in bfloat16 autocast, as in our run.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import TxnData  # noqa: E402
from model import SeqModel, keep_layers, n_params  # noqa: E402
from weights import load_model, save_model  # noqa: E402

LOSSES = ("label", "next", "amt", "nce", "kd", "logit", "emb", "rkd", "crd", "bct")


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", required=True, help="the folder written by models/prepare.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="", help="a recipe of models/recipes.json: teacher-5m, student-2.6m-distilled, "
                    "student-0.4m-distilled, student-0.4m-labels-only")
    ap.add_argument("--teacher", default="", help="the teacher's .safetensors file (students)")
    ap.add_argument("--role", choices=["teacher", "student"]); ap.add_argument("--arch", choices=["seq", "keep_layers"])
    ap.add_argument("--keep_layers", help="students cut from the teacher (--arch keep_layers): the teacher layers kept, e.g. 0,1,5")
    for k, t in (("d_model", int), ("n_layers", int), ("n_heads", int), ("d_field", int), ("field_layers", int), ("emb_dim", int),
                 ("L", int), ("next_from_emb", int), ("steps", int), ("bs", int), ("lr", float), ("wd", float), ("warmup", int),
                 ("pos_frac", float), ("label_w", float), ("next_w", float), ("amt_w", float), ("nce_w", float), ("nce_tau", float),
                 ("nce_pairs", int), ("nce_detach", int), ("kd_w", float), ("T", float), ("logit_w", float), ("emb_w", float),
                 ("rkd_w", float), ("kd_consistent", int), ("bct_w", float), ("crd_w", float), ("crd_tau", float), ("emb_detach", int),
                 ("id_dropout", float), ("eval_every", int), ("val_neg", int), ("seed", int)):
        ap.add_argument(f"--{k}", type=t)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=500, help="resumable checkpoint every N steps (0 = off)")
    a = ap.parse_args(argv)
    recipe = {}
    if a.recipe:
        recipes = json.load(open(os.path.join(HERE, "recipes.json")))
        if a.recipe not in recipes:
            raise SystemExit(f"unknown recipe {a.recipe}; choose from {sorted(recipes)}")
        recipe = recipes[a.recipe]
    for k, v in recipe.items():
        if k in ("teacher", "note"):
            continue
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    missing = [k for k, v in vars(a).items() if v is None and k not in ("keep_layers",)]
    if missing:
        raise SystemExit(f"set --recipe or every option; missing: {missing}")
    if a.role == "student" and not a.teacher:
        raise SystemExit("students need --teacher (the recipe's teacher: " + recipe.get("teacher", "?") + ")")
    return a


def id_dropout(tok, p, gen, fields):
    if p <= 0 or not fields:
        return tok
    m = torch.rand(tok.shape[:-1], generator=gen) < p
    tok = tok.clone()
    for f in fields:
        tok[..., f] = torch.where(m & (tok[..., f] > 0), torch.ones_like(tok[..., f]), tok[..., f])
    return tok


def info_nce(z1, z2, tau):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    y = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def kd_binary(s_logit, t_logit, T=2.0):
    """Binary cross-entropy between the temperature-softened teacher and student probabilities."""
    return F.binary_cross_entropy_with_logits(s_logit / T, torch.sigmoid(t_logit / T)) * T * T


def logit_mse(s_logit, t_logit, clip=12.0):
    return F.mse_loss(s_logit.clamp(-clip, clip), t_logit.clamp(-clip, clip))


def emb_cosine(s_emb, t_emb):
    return (1 - F.cosine_similarity(s_emb, t_emb, dim=-1)).mean()


def rkd_distance(s, t, eps=1e-8):
    """Relational distillation (Park et al. 2019): match the normalized pairwise distances within the batch."""
    ds = torch.cdist(s, s); dt = torch.cdist(t, t)
    ds = ds / (ds[ds > 0].mean() + eps); dt = dt / (dt[dt > 0].mean() + eps)
    return F.smooth_l1_loss(ds, dt)


def training_masks(split, card, idx):
    """For the window rows `idx`: which may be supervised (training rows), and which have a next transaction that is a training row
    of the same card (the next-MCC and next-amount targets never come from validation or test rows)."""
    idx = np.asarray(idx)
    valid = (idx >= 0) & (idx < len(split))
    safe = np.clip(idx, 0, len(split) - 1)
    allowed = valid & (split[safe] == 0)
    following = np.minimum(safe + 1, len(split) - 1)
    aux = allowed & (safe + 1 < len(split)) & (split[following] == 0) & (card[following] == card[safe])
    return allowed, aux


def selection_mask(label, scenario):
    """Rows used to pick the checkpoint: everything except IBM's own fraud (scenario 1)."""
    return ~((label == 1) & (scenario == 1)) & ~((label == 0) & np.isin(scenario, (9, 10)))   # (no legit row has scenario 9 or 10)


def mfwd(model, b, device, tokens=None):
    return model((b["tokens"] if tokens is None else tokens).to(device), b["pad"].to(device),
                 **({"num": b["num"].to(device)} if model.n_num else {}))


@torch.no_grad()
def quick_val(model, data, rows, wts, L, dev):
    """Area under the precision-recall curve of the score at each row (window ending at the row), weighted."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    sc = []
    for i in range(0, len(rows), 128):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            o = mfwd(model, data.batch(rows[i:i + 128], L), dev)
        sc.append(o["logit"][:, -1].float().cpu().numpy())
    model.train()
    s, y = np.concatenate(sc), data.label[rows].astype(int)
    return float(average_precision_score(y, s, sample_weight=wts)), float(roc_auc_score(y, s, sample_weight=wts))


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    data = TxnData(a.arrays, drop_fields=["flags"], num_groups=["time"])
    teacher = None
    if a.teacher:                                  # loaded before the student is built (the student's random start depends on it)
        teacher, tcfg = load_model(a.teacher, dev)
        for p in teacher.parameters():
            p.requires_grad = False
        assert tcfg["inputs"]["fields"] == data.fields, "the teacher reads other fields"
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    if a.arch == "keep_layers":
        model = keep_layers(teacher, [int(x) for x in a.keep_layers.split(",")]).to(dev)
    else:
        model = SeqModel(data.vocab_sizes, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_field=a.d_field,
                         field_layers=a.field_layers, L_max=a.L, emb_dim=a.emb_dim, n_mcc=data.n_next, n_amt=data.n_amt,
                         next_from_emb=bool(a.next_from_emb), n_num=data.n_num).to(dev)
    distill = a.role == "student" and (a.kd_w + a.logit_w + a.emb_w + a.rkd_w + a.bct_w + a.crd_w) > 0
    params = list(model.parameters())
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    L_ = a.L
    sampler = data.train_sampler(L_, a.bs, a.pos_frac, split=0, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    best, best_sd, best_step = -1, None, None
    ck = os.path.join(a.out, "ckpt.pt"); start = 0
    if os.path.exists(ck):                         # resume after a restart
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, best, best_sd, best_step = c["step"], c["best"], c["best_sd"], c.get("best_step")
        rng = np.random.default_rng(a.seed + start)
        sampler = data.train_sampler(L_, a.bs, a.pos_frac, split=0, seed=a.seed + start)
        print(f"resumed from step {start}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a" if start else "w")
    if a.eval_every:
        q_rows, q_w = data.eval_rows(1, 4000, seed=7, n_neg_foreign=2000)
        k_ = selection_mask(data.label[q_rows], data.scenario[q_rows]); q_rows, q_w = q_rows[k_], q_w[k_]
    t0 = time.time(); skipped = 0
    weight = dict(label=a.label_w, next=a.next_w, amt=a.amt_w, nce=a.nce_w, kd=a.kd_w, logit=a.logit_w, emb=a.emb_w, rkd=a.rkd_w,
                  crd=a.crd_w, bct=a.bct_w)
    n_pairs = a.nce_pairs or a.bs
    model.train()
    for step in range(start, a.steps):
        Ls = {}
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda"))
        amp.__enter__()
        ends = next(sampler)
        b = data.batch(ends, L_)
        tok_clean = b["tokens"]
        tok = id_dropout(tok_clean, a.id_dropout, gen, data.id_field_idx).to(dev)
        lab, nxt, nam = (b[k].to(dev) for k in ("label", "next_cat", "next_amt"))
        o = mfwd(model, b, dev, tokens=tok)
        sel = lab >= 0
        sel[:, : L_ // 2] = False                  # supervise the second half of the window
        allowed, auxiliary = training_masks(data.split, data.card, b["idx"])
        sel &= torch.from_numpy(allowed).to(dev)
        aux_sel = sel & torch.from_numpy(auxiliary).to(dev)
        if a.label_w:
            Ls["label"] = F.binary_cross_entropy_with_logits(o["logit"][sel], lab[sel].float())
        if a.next_w:
            m2 = aux_sel & (nxt >= 0)
            Ls["next"] = F.cross_entropy(o["next_logits"][m2], nxt[m2])
        if a.amt_w and o.get("amt_logits") is not None:
            m3 = aux_sel & (nam >= 0)
            Ls["amt"] = F.cross_entropy(o["amt_logits"][m3], nam[m3])
        if a.nce_w:
            k = min(n_pairs, len(ends))
            other, found = data.same_card_pair(ends[:k], rng)
            if found.sum() >= 2:
                b2 = data.batch(other[found], L_)
                t2 = id_dropout(b2["tokens"], a.id_dropout, gen, data.id_field_idx).to(dev)
                fm = torch.from_numpy(found).to(dev)
                if a.nce_detach:
                    with torch.no_grad():
                        h2 = mfwd(model, b2, dev, tokens=t2)["h"][:, -1]
                    Ls["nce"] = info_nce(model.emb_head(o["h"][:k][fm, -1].detach()), model.emb_head(h2), a.nce_tau)
                else:
                    o2 = mfwd(model, b2, dev, tokens=t2)
                    Ls["nce"] = info_nce(o["emb"][:k][fm, -1], o2["emb"][:, -1], a.nce_tau)
        if distill:
            with torch.no_grad():
                to = mfwd(teacher, b, dev, tokens=tok if a.kd_consistent else tok_clean)
                tl, te = to["logit"], to["emb"]
            sl = o["logit"]
            if a.kd_w: Ls["kd"] = kd_binary(sl[sel], tl[sel], a.T)
            if a.logit_w: Ls["logit"] = logit_mse(sl[sel], tl[sel])
            es = model.emb_head(o["h"].detach()) if a.emb_detach else o["emb"]
            if a.emb_w: Ls["emb"] = emb_cosine(es[sel], te[sel])
            if a.rkd_w or a.crd_w:
                se, tee = es[sel], te[sel]
                kk = torch.randperm(se.shape[0], device=dev)[:256]
                if a.rkd_w: Ls["rkd"] = rkd_distance(se[kk].float(), tee[kk].float())
                if a.crd_w: Ls["crd"] = info_nce(se[kk].float(), tee[kk].float(), a.crd_tau)
            if a.bct_w and teacher.next_from_emb:
                mb = aux_sel & (nxt >= 0)
                Ls["bct"] = F.cross_entropy(teacher.next_head(es[mb]).float(), nxt[mb])
        amp.__exit__(None, None, None)
        fin = [weight[k] * v for k, v in Ls.items() if torch.isfinite(v)]
        opt.zero_grad(set_to_none=True)
        if not fin:                                # every loss term NaN / inf: skip the batch
            skipped += 1; sched.step(); continue
        loss = sum(fin)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):                 # NaN / inf gradient: do not let it reach the weights
            opt.zero_grad(set_to_none=True); skipped += 1; sched.step(); continue
        opt.step(); sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            rec = dict(step=step, loss=loss.item(), sec=round(time.time() - t0, 1), skipped=skipped, **{k: round(v.item(), 4) for k, v in Ls.items()})
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 250 == 0:
                print(rec, flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            ap_, auc = quick_val(model, data, q_rows, q_w, L_, dev)
            log.write(json.dumps(dict(step=step, val_auprc=ap_, val_rocauc=auc)) + "\n"); log.flush()
            print("validation", step, round(ap_, 4), round(auc, 4), flush=True)
            if ap_ > best:
                best, best_step = ap_, step; best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, best=best,
                            best_sd=best_sd, best_step=best_step), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    recipe = {k: getattr(a, k) for k in ("role", "arch", "keep_layers", "d_model", "n_layers", "n_heads", "steps", "bs", "lr", "wd", "warmup",
                                         "pos_frac", "id_dropout", "eval_every", "seed") + tuple(f"{x}_w" for x in LOSSES)}
    cfg = dict(name=os.path.basename(os.path.normpath(a.out)), model_family=a.recipe or a.role,
               model=dict(vocab_sizes=data.vocab_sizes, **{k: v for k, v in model.cfg.items()}),
               inputs=dict(fields=data.fields, drop_fields=["flags"], num_groups=["time"], num_features=data.num_names, window=a.L),
               training=recipe, parameters=n_params(model), parameters_without_embeddings=n_params(model, True))
    save_model(model, cfg, os.path.join(a.out, "model.safetensors"))
    json.dump(dict(best_validation_auprc=best, best_step=best_step, train_seconds=round(time.time() - t0, 1), skipped_steps=skipped),
              open(os.path.join(a.out, "val_metrics.json"), "w"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)
    kept = f"best validation AUPRC {best:.4f} at step {best_step}" if best_sd is not None else "the last step (no validation checks)"
    print(f"done: {a.out}/model.safetensors, weights of {kept}", flush=True)


if __name__ == "__main__":
    main()
