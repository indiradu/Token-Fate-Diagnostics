"""CPU. Train the retrained Learn2PD filter on EGP traces of TRAINING-split questions, freeze tau, and register the official released weights.

Usage: <abs venv python> fit_l2pd.py [--check]      (--check: report only, write nothing)
Recipe (paper v2 Sec. 4 / Appendix; the official notebook train.ipynb differs, see README): Linear(32,32)-ReLU-Linear(32,32) on the 32 block confidences, one logit per slot,
  BCEWithLogitsLoss over all 32 slots, AdamW lr 1e-3 (default weight decay), full batch, 5,000 epochs. Rows = EGP steps of the 'fit' questions (MATH train + GSM8K train).
  The only deviation chosen by us: the number of epochs is picked from {500, 1000, 2000, 5000} by validation BCE on the held-out 'thr' training questions (our data set is ~30x smaller
  than the paper's 2,640 questions, so 5,000 full-batch epochs may overfit; the rule is fixed here, before any test decode).
tau: 'published' = 0.96 (applied to the sigmoid, as in the official code). 'matched' = the lowest tau on a 0.001 grid whose pooled precision on the held-out B0-state blank rows
  (label = proposal equals the one-word-per-call final token, i.e. the Lctx label) is at least the precision of 'confidence >= 0.8' on the same rows (the Lctx matching rule).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
OFFICIAL = Path("/home/rui.gao/Desktop/l2pd_untrusted/repo/llada/layer_2_flan.pth")
EPOCH_GRID = (500, 1000, 2000, 5000)
SEED = 23


def load_all():
    data = {"fit": [], "thr": []}
    for sub in ("train_math", "train_gsm8k"):
        for p in sorted((OUT / "training_data" / sub).glob("*.npz")):
            if p.name.endswith(".tmp.npz"):
                continue
            z = np.load(p)
            data[str(z["role"])].append({"qid": p.stem, "set": sub, **{k: z[k] for k in z.files if k not in ("role",)}})
    return data


def egp_xy(items):
    X = np.concatenate([d["egp_vec"] for d in items]).astype(np.float32)
    Y = np.concatenate([d["egp_ok"] for d in items]).astype(np.float32)
    M = np.concatenate([d["egp_mask"] for d in items])
    return X, Y, M


def b0_rows(items):
    """Blank-slot rows of the one-word-per-call (B0) states: confidence vector, label (proposal == reference), slot, question."""
    V, L, S, Q = [], [], [], []
    for qi, d in enumerate(items):
        ref, bb = d["ref"], d["b0_block"].astype(int)
        y = np.stack([ref[b * 32:(b + 1) * 32] for b in bb])
        lab = d["b0_prop"] == y
        n = len(bb)
        V.append(np.repeat(d["b0_vec"][:, None, :], 32, axis=1)[d["b0_mask"]])        # (rows, 32): the vector each blank slot's filter output reads
        L.append(lab[d["b0_mask"]])
        S.append(np.tile(np.arange(32), (n, 1))[d["b0_mask"]])
        Q.append(np.full(int(d["b0_mask"].sum()), qi))
    return np.concatenate(V), np.concatenate(L), np.concatenate(S), np.concatenate(Q)


def mlp(state=None):
    m = torch.nn.Sequential()
    m.linear1, m.relu, m.linear2 = torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32)
    if state is not None:
        m.linear1.load_state_dict({"weight": state["linear1.weight"], "bias": state["linear1.bias"]})
        m.linear2.load_state_dict({"weight": state["linear2.weight"], "bias": state["linear2.bias"]})
    return m


def fwd(m, x):
    return m.linear2(m.relu(m.linear1(x)))


def state_of(m):
    return {"linear1.weight": m.linear1.weight.detach().clone(), "linear1.bias": m.linear1.bias.detach().clone(),
            "linear2.weight": m.linear2.weight.detach().clone(), "linear2.bias": m.linear2.bias.detach().clone()}


def probs_on_rows(state, vec_rows, slots):
    with torch.no_grad():
        m = mlp(state)
        p = torch.sigmoid(fwd(m, torch.as_tensor(vec_rows)))
    return p.numpy()[np.arange(len(slots)), slots]


def evaluate(state, X, Y, M):
    m = mlp(state)
    with torch.no_grad():
        logit = fwd(m, torch.as_tensor(X))
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.as_tensor(Y)))
        p = torch.sigmoid(logit).numpy()
    blank = M.astype(bool)
    auc = float(roc_auc_score(Y[blank], p[blank])) if 0 < Y[blank].mean() < 1 else float("nan")
    return {"bce_all_slots": bce, "auroc_blank_slots": auc, "blank_label_rate": float(Y[blank].mean())}


def match_tau(p, lab, conf, c=0.8):
    target = float(lab[conf >= c].mean())
    best = None
    for tau in np.arange(0.0, 1.0, 0.001):
        sel = p > tau
        if sel.sum() >= 100 and lab[sel].mean() >= target:
            best = float(round(tau, 3))
            break
    return target, best


def rate(p, lab, tau):
    sel = p > tau
    return {"accepted_share": float(sel.mean()), "precision": float(lab[sel].mean()) if sel.any() else None}


def main():
    check = "--check" in sys.argv
    data = load_all()
    nfit, nthr = len(data["fit"]), len(data["thr"])
    X, Y, M = egp_xy(data["fit"])
    Xv, Yv, Mv = egp_xy(data["thr"])
    print(f"fit questions {nfit} ({sum(d['set'] == 'train_math' for d in data['fit'])} MATH), EGP rows {len(X)}; thr questions {nthr}, EGP rows {len(Xv)}", flush=True)
    torch.manual_seed(SEED)
    m = mlp()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, Yt, Xvt, Yvt = map(torch.as_tensor, (X, Y, Xv, Yv))
    snaps, curve = {}, []
    for ep in range(1, max(EPOCH_GRID) + 1):
        opt.zero_grad()
        loss = loss_fn(fwd(m, Xt), Yt)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep in EPOCH_GRID:
            with torch.no_grad():
                vl = float(loss_fn(fwd(m, Xvt), Yvt))
            curve.append((ep, float(loss), vl))
        if ep in EPOCH_GRID:
            snaps[ep] = state_of(m)
    val = {ep: [c for c in curve if c[0] == ep][0][2] for ep in EPOCH_GRID}
    chosen = min(EPOCH_GRID, key=lambda e: val[e])
    state = snaps[chosen]
    print("val BCE by epochs", {e: round(v, 4) for e, v in val.items()}, "-> chosen", chosen, flush=True)
    official = torch.load(str(OFFICIAL), map_location="cpu", weights_only=True)
    # tau on B0-state blank rows of the held-out questions
    V, L, S, Q = b0_rows(data["thr"])
    conf = V[np.arange(len(S)), S]
    is_math = np.array([data["thr"][q]["set"] == "train_math" for q in Q])
    rep = {"fit_questions": nfit, "thr_questions": nthr, "fit_rows": int(len(X)), "thr_rows": int(len(Xv)), "epochs_chosen": chosen, "val_bce_by_epochs": val,
           "train_bce_final_chosen": [c for c in curve if c[0] == chosen][0][1], "eval_thr_egp": {"retrained": evaluate(state, Xv, Yv, Mv), "official": evaluate(official, Xv, Yv, Mv)},
           "eval_fit_egp": {"retrained": evaluate(state, X, Y, M)}, "b0_thr_blank_rows": int(len(L)), "b0_thr_blank_rows_math": int(is_math.sum())}
    for name, st in (("retrained", state), ("official", official)):
        p = probs_on_rows(st, V, S)
        target, matched = match_tau(p, L, conf)
        _, matched_math = match_tau(p[is_math], L[is_math], conf[is_math])
        rep[f"b0_thr_{name}"] = {"c08_precision_target": target, "c08_accepted_share": float((conf >= 0.8).mean()), "tau_matched": matched, "tau_matched_math_only": matched_math,
                                 "at_0.96": rate(p, L, 0.96), "at_matched": rate(p, L, matched) if matched is not None else None,
                                 "conf_ge_0.9": {"accepted_share": float((conf >= 0.9).mean()), "precision": float(L[conf >= 0.9].mean())},
                                 "conf_ge_0.8": {"accepted_share": float((conf >= 0.8).mean()), "precision": float(L[conf >= 0.8].mean())}}
    print(json.dumps(rep, indent=1), flush=True)
    if check:
        return
    F = OUT / "filters"
    F.mkdir(exist_ok=True)
    torch.save(state, F / "l2pd_retrained.pth")
    torch.save(official, F / "l2pd_official.pth")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = {"retrained": {"file": "l2pd_retrained.pth", "file_sha256": sha(F / "l2pd_retrained.pth"), "tau_published": 0.96, "tau_matched": rep["b0_thr_retrained"]["tau_matched"],
                          "epochs": chosen, "fit_questions": nfit, "fit_rows": int(len(X)), "n_parameters": 2112, "seed": SEED, "source_official_sha256_note": "trained here, not the official file"},
            "official": {"file": "l2pd_official.pth", "file_sha256": sha(F / "l2pd_official.pth"), "tau_published": 0.96, "n_parameters": 2112,
                         "origin": "github.com/ims-kdks/Learning-to-Parallel-Decoding llada/layer_2_flan.pth at commit 06bfd7fa3c2634d334f307916a555c58d5f97e45",
                         "origin_file_sha256": sha(OFFICIAL), "note": "re-saved with torch.save(state_dict) after torch.load(weights_only=True); tensors identical"}}
    (F / "rules.json").write_text(json.dumps(meta, indent=1) + "\n")
    (OUT / "validation/fit_report.json").write_text(json.dumps({**rep, "curve_epoch_train_val_bce": curve}, indent=1) + "\n")
    print("wrote filters/rules.json")


if __name__ == "__main__":
    main()
