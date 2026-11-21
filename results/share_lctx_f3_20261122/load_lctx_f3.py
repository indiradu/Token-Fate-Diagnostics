"""Load the Lctx-F3 word lock (hash-checked) and the extra SelectiveConfig fields it needs.

Import AFTER the engine is on sys.path (FAMILY_SRC=<this dir>/source_m1_f3, or any src/ that carries engine_f3.patch): the pickle is a context_rules.ContextRule whose
payload is a sklearn HistGradientBoostingClassifier, and its feature list names lock_features3 inputs.

    from load_lctx_f3 import load_lctx_f3
    rule, sha, token_info = load_lctx_f3()
    ctx["rule"] = rule
    cfg = C.cached_cfg("Lctx-F3", ctx, token_info=token_info, **cache_kwargs)     # lock="rule", threshold 0.9 (only used by the best-blank fallback), rule.lam is the cut

Usage as a check: <python> load_lctx_f3.py   -> prints the hashes, lambda, features and the scores of one synthetic call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RULE = HERE / "rules/Lctx-F3.pkl"
RULES_JSON = HERE / "rules/rules.json"
TOKEN_INFO = HERE / "assets/token_info.npz"
ASSETS_JSON = HERE / "assets/assets.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_token_info() -> dict:
    want = json.loads(ASSETS_JSON.read_text())["token_info_sha256"]
    assert _sha(TOKEN_INFO) == want, "token_info.npz does not match assets.json"
    z = np.load(TOKEN_INFO)
    return {k: z[k] for k in z.files}


def load_lctx_f3():
    """(ContextRule, sha256, token_info dict). Asserts the pickle's hash, lambda and feature list against rules.json."""
    from regret_remasking.context_rules import ContextRule
    from regret_remasking import lock_features3  # noqa: F401  (fails here, not mid-decode, if the engine lacks engine_f3.patch)
    meta = json.loads(RULES_JSON.read_text())["rules"]["Lctx-F3"]
    sha = _sha(RULE)
    assert sha == meta["sha256"], f"Lctx-F3.pkl does not match rules.json: {sha}"
    rule = ContextRule.load(RULE)
    assert abs(rule.lam - meta["lambda"]) < 1e-12 and list(rule.features) == meta["features"]
    return rule, sha, load_token_info()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE / "source_m1_f3/src"))
    rule, sha, info = load_lctx_f3()
    print("Lctx-F3", sha, "lambda", rule.lam, "inputs", len(rule.features))
    print("token_info", _sha(TOKEN_INFO), {k: int(v.sum()) for k, v in info.items()})
    rng = np.random.default_rng(0)
    feats = {f: rng.random(8).astype(np.float32) for f in rule.features}
    print("scores of 8 random rows", np.round(rule.scores(feats), 4))
