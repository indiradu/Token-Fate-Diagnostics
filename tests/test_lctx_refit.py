"""Trace-collection hook (SelectiveConfig.collect_rows) and the CPU helpers of results/lctx_refit_20261110."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from regret_remasking.context_rules import ContextRule
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

_ORCH = Path(__file__).resolve().parents[1] / "results/lctx_refit_20261110/orchestration"      # results/** is gitignored: skip the helper tests on a checkout without it
sys.path.insert(0, str(_ORCH))
try:
    import refit_lib as RL  # noqa: E402
except ImportError:
    RL = None
needs_lib = pytest.mark.skipif(RL is None, reason="results/lctx_refit_20261110 not present")
FEATS = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]

MASK, EOS, V, P, G = 3, 2, 40, 6, 24


def _model(seed=0):
    return ToyLLaDA(vocab=V, d=32, n_heads=4, layers=4, ffn=48, seed=seed)


def _prompt(seed=0):
    return torch.randint(4, V, (1, P), generator=torch.Generator().manual_seed(seed))


def _cfg(**kw):
    base = dict(gen_length=G, window=6, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="block", block_length=8, attn_impl="sdpa", lock="topk", topk=1, far_mask_cache=True)
    base.update(kw)
    return SelectiveConfig(**base)


GEOMS = {"window": dict(shape="window"), "block_p": dict(shape="block", partial_every_k=2, partial_layer=2, partial_join="blank")}


@pytest.mark.parametrize("geom", list(GEOMS))
def test_collect_rows_is_default_off_and_does_not_change_the_decode(geom):
    model, prompt = _model(), _prompt()
    off = decode_selective(model, prompt, _cfg(**GEOMS[geom]))
    on = decode_selective(model, prompt, _cfg(collect_rows=True, **GEOMS[geom]))
    assert off.collected == [] and len(on.collected) == on.nfe
    assert on.tokens == off.tokens and on.nfe == off.nfe and on.refresh_kind == off.refresh_kind and on.written == off.written


@dataclass
class _Recorder(ContextRule):
    """A 'rule' that never admits anything above lam=1: the decoder falls back to the best-scoring blank (= argmax confidence = topk 1) and we record what it was shown."""
    seen: list = None

    def scores(self, feats):
        self.seen.append({k: np.array(v, copy=True) for k, v in feats.items()})
        return feats["confidence"].astype(np.float64) * 0.5


@pytest.mark.parametrize("geom", list(GEOMS))
def test_logged_features_equal_what_the_rule_path_sees(geom):
    model, prompt = _model(1), _prompt(1)
    on = decode_selective(model, prompt, _cfg(collect_rows=True, **GEOMS[geom]))
    rec = _Recorder("conf", 1.0, features=[], seen=[])
    via_rule = decode_selective(model, prompt, _cfg(lock="rule", rule=rec, **GEOMS[geom]))
    assert via_rule.tokens == on.tokens and via_rule.nfe == on.nfe          # same one-word-per-call trajectory, so the same calls are compared
    assert len(rec.seen) == len(on.collected) == on.nfe
    for seen, logged in zip(rec.seen, on.collected):
        for f in FEATS:
            assert np.array_equal(seen[f], logged[f]), f
        assert len(logged["position"]) == len(logged["token"]) == len(logged["confidence"])


def test_logged_rows_are_the_eligible_blanks_and_one_word_is_written_per_call():
    model, prompt = _model(2), _prompt(2)
    w = decode_selective(model, prompt, _cfg(shape="window", collect_rows=True, log_detail=True))
    assert all(len(c["position"]) <= 6 for c in w.collected) and all(len(x) == 1 for x in w.written)
    b = decode_selective(model, prompt, _cfg(shape="block", collect_rows=True, log_detail=True))
    assert all(len(c["position"]) <= 8 for c in b.collected)
    for c, written in zip(b.collected, b.written):
        assert written[0] in c["position"].tolist() and c["n_first_block"] == len(c["position"])     # block shape: eligible = the block's blanks
    first = b.collected[0]
    assert first["n_first_block"] == 8 and np.all(first["pos_in_block"] == 1 / 8)                  # nothing written yet in the block: clamped to 1/block_length


@needs_lib
def test_labels_are_filled_after_the_decode_from_the_final_tokens():
    model, prompt = _model(3), _prompt(3)
    res = decode_selective(model, prompt, _cfg(shape="window", collect_rows=True))
    rows = RL.labelled_rows(res.collected, res.tokens, P, res.refresh_kind, "q", block_length=8)
    assert len(rows["y"]) == sum(len(c["position"]) for c in res.collected)
    assert np.array_equal(rows["final_token"], np.asarray(res.tokens)[rows["position"] - P])
    assert np.array_equal(rows["y"], (rows["token"] == rows["final_token"]).astype(np.int8))
    # one-word-per-call: the row chosen at its own call (the last row in which a position appears) always has y = 1
    last = {}
    for i, pos in enumerate(rows["position"]):
        last[int(pos)] = i
    assert all(rows["y"][i] == 1 for i in last.values())
    # dense flag follows the refresh kinds, first_call flag follows the block-start rule
    first_calls = {i for i, k in enumerate(res.refresh_kind) if k in ("first", "dense")}
    assert set(np.unique(rows["call"][rows["dense"]])) == first_calls


@needs_lib
def test_pooled_lambda_is_the_lowest_threshold_meeting_the_pooled_precision_target():
    s = np.array([0.99, 0.98, 0.97, 0.96, 0.9, 0.5, 0.4, 0.3])
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    # prefixes: precision 1, 1, 1, .75, .8, .667, .571, .5
    assert RL.pooled_lambda(s, y, 1.0) == 0.97
    assert RL.pooled_lambda(s, y, 0.8) == 0.9            # the largest admitted set that still meets the target (0.8 at 5 rows)
    assert RL.pooled_lambda(s, y, 0.75) == 0.9
    assert RL.pooled_lambda(np.array([0.9, 0.8]), np.array([0, 0]), 0.5) is None
    # ties are never split: the two 0.5 scores are admitted together, so precision after the tie is 0.5 < 0.75
    s2 = np.array([0.9, 0.5, 0.5]); y2 = np.array([1, 1, 0])
    assert RL.pooled_lambda(s2, y2, 0.75) == 0.9 and RL.pooled_lambda(s2, y2, 0.6) == 0.5


@needs_lib
def test_row_filters_and_rule_roundtrip(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"token": [5, RL.EOS_ID, 7, RL.EOT_ID], "first_call_in_block": [False, False, True, False], **{f: np.arange(4, dtype=np.float32) for f in RL.FEATURES}, "y": [1, 1, 0, 1]})
    assert len(RL.select_rows(df, "all")) == 2 and len(RL.select_rows(df, "orig_filter")) == 1
    with pytest.raises(ValueError):
        RL.select_rows(df, "nope")
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.random((300, 6)).astype("float32")
    d = pd.DataFrame(X, columns=RL.FEATURES); d["y"] = (X[:, 0] > 0.5).astype(int)
    tpl = HistGradientBoostingClassifier(max_iter=20, learning_rate=0.1, early_stopping=False, random_state=1)
    m = RL.fit_gbm(d, tpl)
    assert m.get_params()["random_state"] == 23 and tpl.get_params()["random_state"] == 1       # the template is cloned, not modified
    rule = ContextRule("gbm", 0.7, RL.FEATURES, m, "t")
    rule.save(tmp_path / "r.pkl")
    again = ContextRule.load(tmp_path / "r.pkl")
    feats = {f: X[:5, i] for i, f in enumerate(RL.FEATURES)}
    assert np.array_equal(rule.scores(feats), again.scores(feats))
