"""CPU helpers for the Lctx refit (PREREGISTERED.md): trace rows -> labelled frame, the pooled-precision lambda, the HistGradientBoosting fit, feature-range summaries.

RECONSTRUCTION NOTICE. The original fitter (the earlier session's analyze_context.py / ctxlib.py, in a worktree that no longer exists) is not in the repository. What is pinned here comes from
what survives: (a) the learner and its hyperparameters are read off the original pickle (`sklearn.base.clone` of `ContextRule.load(Lctx_c0.8.pkl).payload`, incl. early_stopping=False, seed 23);
(b) lambda follows the earlier session's description ("the lowest threshold whose pooled precision on the threshold set equals the precision of the confidence >= c rule there") and
`rules.json` of the original (every `precision_on_threshold_set` is >= its `target_precision` by a hair): lambda = the lowest score threshold at which the precision of everything admitted
(mean label over ALL admitted threshold-question rows, pooled, not averaged per question) is still >= the target; target = mean label over the threshold rows with confidence >= c.
(c) the row filter is the one recorded in `fit_feature_ranges.json`: non-EOS/EOT proposals (and, for the variant "orig_filter", only calls with at least one word already written in the block).
The reconstruction cannot be shown to reproduce the original pickle (the original training rows are gone).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EOS_ID, EOT_ID = 126081, 126348
FEATURES = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]
BLOCK = 32


def labelled_rows(collected: list[dict], tokens: list[int], prompt_length: int, refresh_kind: list[str], example_id: str, block_length: int = BLOCK) -> dict[str, np.ndarray]:
    """Flatten SelectiveResult.collected into column arrays and fill the label once the decode is finished: y = (proposal at that call == the final token at that position),
    exactly as in the original fit (final token = res.tokens, with never-written blanks already filled with EOS by the decoder)."""
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ["call", "position", "token", "n_first_block", "dense"] + FEATURES}
    for c in collected:
        n = len(c["position"])
        cols["call"].append(np.full(n, c["call"], dtype=np.int32))
        cols["position"].append(np.asarray(c["position"], dtype=np.int32))
        cols["token"].append(np.asarray(c["token"], dtype=np.int64))
        cols["n_first_block"].append(np.full(n, c["n_first_block"], dtype=np.int16))
        cols["dense"].append(np.full(n, refresh_kind[c["call"]] in ("first", "dense"), dtype=bool))
        for f in FEATURES:
            cols[f].append(np.asarray(c[f], dtype=np.float32))
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    final = np.asarray(tokens, dtype=np.int64)[out["position"].astype(np.int64) - prompt_length] if len(out["position"]) else np.zeros(0, dtype=np.int64)
    out["final_token"] = final
    out["y"] = (out["token"] == final).astype(np.int8)
    out["first_call_in_block"] = (out["n_first_block"] == block_length)
    out["example_id"] = np.full(len(out["call"]), example_id)
    return out


def load_traces(trace_dir: Path, ids: list[str]) -> pd.DataFrame:
    frames = []
    for qid in ids:
        z = np.load(Path(trace_dir) / f"{qid}.npz", allow_pickle=False)
        frames.append(pd.DataFrame({k: z[k] for k in z.files if z[k].ndim == 1}))
    return pd.concat(frames, ignore_index=True)


def select_rows(df: pd.DataFrame, variant: str = "all") -> pd.DataFrame:
    """variant 'all' (PRIMARY): every logged row whose proposal is not EOS/EOT. 'orig_filter': additionally only calls with >= 1 word already written in the block (step_in_block >= 1),
    the filter recorded for the original fit."""
    keep = ~df["token"].isin([EOS_ID, EOT_ID])
    if variant == "orig_filter":
        keep &= ~df["first_call_in_block"]
    elif variant != "all":
        raise ValueError(variant)
    return df[keep]


def pooled_lambda(scores: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """Lowest threshold lam such that mean(y[scores >= lam]) >= target (pooled over all rows); None if even the single best score fails. Ties are never split."""
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order].astype(np.float64)
    cum = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    last_of_tie = np.append(s[1:] != s[:-1], True)                 # admitted sets are prefixes that end at a tie boundary
    ok = np.nonzero(last_of_tie & (cum >= target - 1e-15))[0]
    if len(ok) == 0:
        return None
    return float(s[ok[-1]])


def fit_gbm(fit: pd.DataFrame, template, seed: int = 23):
    """Same learner and hyperparameters as the original: a clone of the original pickle's estimator (HistGradientBoostingClassifier, 200 iterations, lr 0.1, early_stopping False)."""
    from sklearn.base import clone
    m = clone(template)
    m.set_params(random_state=seed)
    return m.fit(fit[FEATURES].to_numpy("float32"), fit["y"].to_numpy())


def feature_ranges(df: pd.DataFrame) -> dict:
    rng = {}
    for f in FEATURES:
        v = df[f].to_numpy(dtype=np.float64)
        rng[f] = {"n": int(len(v)), "min": float(v.min()), "p0.1": float(np.percentile(v, 0.1)), "p50": float(np.percentile(v, 50)), "p99.9": float(np.percentile(v, 99.9)), "max": float(v.max())}
    return rng


def share_outside(df: pd.DataFrame, ranges: dict) -> dict:
    return {f: float(((df[f] < ranges[f]["min"]) | (df[f] > ranges[f]["max"])).mean()) for f in FEATURES}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def dump(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
