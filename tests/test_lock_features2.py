"""Model-internal word-lock features (src/regret_remasking/lock_features2.py, results/lctx_features2_20261111): default-off invariance, value ranges, hand-checked embedding / attention /
logit-lens values, the previous-call velocities, derivability of n_cand from the log, and the rule path (a ContextRule naming the features is shown what collection logs)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from regret_remasking import lock_features as LF
from regret_remasking import lock_features2 as LF2
from regret_remasking.context_rules import FEATURES as RULE_FEATURES, ContextRule
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

_ORCH = Path(__file__).resolve().parents[1] / "results/lctx_features2_20261111/orchestration"      # results/** is gitignored: skip the helper tests on a checkout without it
try:                                                                                                 # loaded by path under its own name: the sibling directories' modules share names (check_gate, ...)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("features2_lib_f2", _ORCH / "features2_lib.py")
    FL2 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(FL2)
except (ImportError, FileNotFoundError, AttributeError):
    FL2 = None
needs_lib = pytest.mark.skipif(FL2 is None, reason="results/lctx_features2_20261111 not present")

MASK, EOS, V, P, G = 3, 2, 40, 6, 24
SIX = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]
TOKEN_CLASS = np.random.default_rng(0).integers(0, 8, size=V).astype(np.int8)
BASE = {"call", "position", "token", "n_first_block"}


def _model(seed=0):
    return ToyLLaDA(vocab=V, d=32, n_heads=4, layers=4, ffn=48, seed=seed)


def _prompt(seed=0):
    return torch.randint(4, V, (1, P), generator=torch.Generator().manual_seed(seed))


def _cfg(**kw):
    base = dict(gen_length=G, window=6, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="block", block_length=8, attn_impl="sdpa", lock="topk", topk=1, far_mask_cache=True,
                collect_rows=True, log_detail=True)
    base.update(kw)
    return SelectiveConfig(**base)


GEOMS = {"window": dict(shape="window"), "block_p": dict(shape="block", partial_every_k=2, partial_layer=2, partial_join="blank")}


def test_name_lists_agree():
    assert set(LF2.FEATURES2) <= RULE_FEATURES and len(LF2.FEATURES2) == len(set(LF2.FEATURES2)) == 9
    assert [len(g) for g in LF2.GROUPS2.values()] == [3, 3, 3] and not set(LF2.FEATURES2) & set(LF.EXTRA_FEATURES)
    assert LF2.CANDIDATE_CONFIDENCE == LF.CANDIDATE_CONFIDENCE


@needs_lib
def test_features2_lib_groups_equal_the_engine_groups():
    assert FL2.GROUPS2 == LF2.GROUPS2 and FL2.NEW == LF2.FEATURES2


@pytest.mark.parametrize("geom", list(GEOMS))
def test_probes_are_default_off_and_never_change_the_trajectory(geom):
    model, prompt = _model(), _prompt()
    off = decode_selective(model, prompt, _cfg(**GEOMS[geom]))
    assert set(off.collected[0]) == set(SIX) | BASE and off.probe_state is None and off.probe_time_s == 0.0
    on = decode_selective(model, prompt, _cfg(collect_probe=True, **GEOMS[geom]))
    assert set(on.collected[0]) == set(SIX) | set(LF2.FEATURES2) | BASE and on.probe_state is not None
    assert on.tokens == off.tokens and on.nfe == off.nfe and on.refresh_kind == off.refresh_kind and on.written == off.written
    for a, b in zip(off.collected, on.collected):
        for f in SIX:
            assert np.array_equal(a[f], b[f])
    both = decode_selective(model, prompt, _cfg(collect_probe=True, collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    assert set(both.collected[0]) == set(SIX) | set(LF2.FEATURES2) | set(LF.EXTRA_FEATURES) | BASE and both.tokens == off.tokens


def test_default_layers_and_range_errors():
    assert LF2.probe_layers(SelectiveConfig(), 32) == (24, 16) and LF2.probe_layers(SelectiveConfig(probe_mid_layer=4, probe_attn_layer=1), 4) == (4, 1)
    with pytest.raises(ValueError):
        LF2.probe_layers(SelectiveConfig(probe_attn_layer=4), 4)
    with pytest.raises(ValueError):
        LF2.probe_layers(SelectiveConfig(probe_mid_layer=5), 4)
    with pytest.raises(ValueError, match="probe"):
        decode_selective(_model(), _prompt(), _cfg(shape="block", collect_probe=True, probe_attn_layer=9))
    with pytest.raises(ValueError, match="probe"):
        decode_selective(_model(), _prompt(), _cfg(shape="block", collect_probe=True, lock="elastic"))


@pytest.mark.parametrize("geom", list(GEOMS))
def test_value_ranges_and_n_cand_derivable_from_the_log(geom):
    res = decode_selective(_model(1), _prompt(1), _cfg(collect_probe=True, **GEOMS[geom]))
    for c in res.collected:
        assert np.all((c["att_masked"] >= 0) & (c["att_masked"] <= 1 + 1e-5)) and np.all(c["att_cand"] <= c["att_masked"] + 1e-5) and np.all(c["att_cand"] >= 0)
        assert np.all((c["emb_spread"] >= 0) & (c["emb_spread"] <= 2)) and np.all(c["ll_kl"] >= 0) and np.all((c["ll_p"] >= 0) & (c["ll_p"] <= 1))
        assert set(np.unique(c["ll_agree"])) <= {0.0, 1.0}
        cand = c["confidence"] >= 0.5
        assert np.array_equal(c["n_cand"], (cand.sum() - cand).astype(np.float32))        # the number of OTHER candidates, from the logged confidences
        fin = [f for f in LF2.FEATURES2 if f not in ("emb_velocity", "hid_velocity")]
        assert all(np.isfinite(c[f]).all() for f in fin)
    assert any((c["att_masked"] > 0).any() for c in res.collected)                        # blanks do attend to masked rows on the toy model


def test_logit_lens_at_the_final_layer_reproduces_the_proposal():
    """mid layer == L: the lens reads the final residual, so it must agree with the proposal, give it the confidence, and have zero KL."""
    res = decode_selective(_model(2), _prompt(2), _cfg(shape="block", collect_probe=True, probe_mid_layer=4, probe_attn_layer=1))
    for c in res.collected:
        assert np.all(c["ll_agree"] == 1.0) and np.allclose(c["ll_p"], c["confidence"], atol=1e-5) and np.all(c["ll_kl"] < 1e-5)
    mid = decode_selective(_model(2), _prompt(2), _cfg(shape="block", collect_probe=True, probe_mid_layer=2, probe_attn_layer=1))
    assert mid.tokens == res.tokens and any((c["ll_kl"] > 1e-3).any() for c in mid.collected)      # a real middle layer disagrees somewhere


def test_embedding_spread_and_velocities_by_hand():
    model = _model(3)
    eng = SelectiveEngine(model, 12, _cfg(shape="block"))
    E = model.model.transformer.wte.weight.detach()
    d = E.shape[1]
    el = torch.tensor([6, 7])
    blank = torch.zeros(12, dtype=torch.bool)
    blank[6:] = True
    h0 = torch.randn(1, 2, d)
    A = torch.full((2, 12), 1.0 / 12)
    one_hot = torch.full((2, V), -30.0)
    one_hot[0, 10], one_hot[1, 11] = 30.0, 30.0                                                     # all mass on one token
    lp = torch.log_softmax(one_hot, -1)
    tok = lp.argmax(-1)
    st = LF2.ProbeState()
    f0 = LF2.probe_features(st, eng, {"probe_mid": h0, "probe_attn": A, "probe_final": h0}, el, lp.exp().max(-1).values, lp, tok, blank, 0, MASK)
    assert np.allclose(f0["emb_spread"], 0.0, atol=1e-5) and np.isnan(f0["emb_velocity"]).all() and np.isnan(f0["hid_velocity"]).all()
    assert np.allclose(f0["att_masked"], 6 / 12, atol=1e-6) and np.allclose(f0["att_cand"], 1 / 12, atol=1e-6) and f0["n_cand"].tolist() == [1.0, 1.0]
    # next call, same blanks, identical distributions and hidden: both velocities 0
    f1 = LF2.probe_features(st, eng, {"probe_mid": h0, "probe_attn": A, "probe_final": h0}, el, lp.exp().max(-1).values, lp, tok, blank, 1, MASK)
    assert np.allclose(f1["emb_velocity"], 0.0, atol=1e-6) and np.allclose(f1["hid_velocity"], 0.0, atol=1e-6)
    # next call: a different token and a doubled hidden -> emb_velocity > 0, hid_velocity == 1
    two = torch.full((2, V), -30.0)
    two[0, 20], two[1, 11] = 30.0, 30.0
    lp2 = torch.log_softmax(two, -1)
    f2 = LF2.probe_features(st, eng, {"probe_mid": h0, "probe_attn": A, "probe_final": 2 * h0}, el, lp2.exp().max(-1).values, lp2, lp2.argmax(-1), blank, 2, MASK)
    assert f2["emb_velocity"][0] > 1e-3 and abs(f2["emb_velocity"][1]) < 1e-6 and np.allclose(f2["hid_velocity"], 1.0, atol=1e-5)
    # a split distribution has positive spread, equal to the weighted cosine distance of the alternatives
    split = torch.full((2, V), -30.0)
    split[:, 10], split[:, 20] = 0.0, 0.0
    lps = torch.log_softmax(split, -1)
    fs = LF2.probe_features(LF2.ProbeState(), eng, {"probe_mid": h0, "probe_attn": A, "probe_final": h0}, el, lps.exp().max(-1).values, lps, lps.argmax(-1), blank, 0, MASK)
    want = 0.5 * (1.0 - float(F.cosine_similarity(E[10], E[20], dim=0)))
    assert np.allclose(fs["emb_spread"], want, atol=1e-5) and fs["emb_spread"][0] > 0.01
    # a call that does not follow the previous one (resume): no previous observation
    f9 = LF2.probe_features(st, eng, {"probe_mid": h0, "probe_attn": A, "probe_final": h0}, el, lp.exp().max(-1).values, lp, tok, blank, 7, MASK)
    assert np.isnan(f9["emb_velocity"]).all() and np.isnan(f9["hid_velocity"]).all()


def test_velocities_are_nan_on_the_first_call_of_a_block_and_finite_after():
    res = decode_selective(_model(4), _prompt(4), _cfg(shape="block", collect_probe=True, probe_mid_layer=3, probe_attn_layer=1))
    first = [c for c in res.collected if c["n_first_block"] == 8]
    assert first and all(np.isnan(c["emb_velocity"]).all() and np.isnan(c["hid_velocity"]).all() for c in first)
    later = [c for c in res.collected if c["n_first_block"] < 8]
    assert later and all(np.isfinite(c["hid_velocity"]).all() and np.isfinite(c["emb_velocity"]).all() for c in later)
    assert any((c["hid_velocity"] > 0).any() for c in later)


@dataclass
class _Recorder(ContextRule):
    seen: list = None

    def scores(self, feats):
        self.seen.append({k: np.array(v, copy=True) for k, v in feats.items()})
        return feats["confidence"].astype(np.float64) * 0.5            # nothing reaches lam = 1: the best-confidence blank is written (= topk 1)


@pytest.mark.parametrize("geom", list(GEOMS))
def test_a_rule_naming_probe_features_is_shown_exactly_what_collection_logs(geom):
    model, prompt = _model(6), _prompt(6)
    on = decode_selective(model, prompt, _cfg(collect_probe=True, **GEOMS[geom]))
    rec = _Recorder("conf", 1.0, features=list(LF2.FEATURES2), seen=[])
    via = decode_selective(model, prompt, _cfg(lock="rule", rule=rec, collect_rows=False, **GEOMS[geom]))
    assert via.tokens == on.tokens and len(rec.seen) == len(on.collected) and via.probe_time_s > 0
    for seen, logged in zip(rec.seen, on.collected):
        for f in SIX + LF2.FEATURES2:
            assert np.array_equal(seen[f], logged[f], equal_nan=True), f


def test_a_six_input_rule_does_not_trigger_the_probes():
    rec = _Recorder("conf", 1.0, features=list(SIX), seen=[])
    res = decode_selective(_model(7), _prompt(7), _cfg(shape="block", lock="rule", rule=rec, collect_rows=False))
    assert res.probe_state is None and all(set(s) == set(SIX) for s in rec.seen)
    assert not LF2.wants_probe(SelectiveConfig(lock="rule", rule=rec)) and not LF2.wants_probe(SelectiveConfig()) and LF2.wants_probe(SelectiveConfig(collect_probe=True))
    assert LF2.wants_probe(SelectiveConfig(lock="rule", rule=_Recorder("conf", 1.0, features=["confidence", "ll_kl"], seen=[])))


@pytest.mark.parametrize("geom", list(GEOMS))
def test_an_extended_gbm_rule_round_trips_and_decodes_with_multi_write(geom, tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model, prompt = _model(8), _prompt(8)
    col = decode_selective(model, prompt, _cfg(collect_probe=True, **GEOMS[geom]))
    feats = SIX + LF2.FEATURES2
    X = np.concatenate([np.stack([c[f] for f in feats], 1) for c in col.collected]).astype("float32")
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    gbm = HistGradientBoostingClassifier(max_iter=10, early_stopping=False, random_state=23).fit(X, y)
    rule = ContextRule("gbm", 0.5, feats, gbm, "toy-Y")
    rule.save(tmp_path / "r.pkl")
    again = ContextRule.load(tmp_path / "r.pkl")
    d = {f: X[:, i] for i, f in enumerate(feats)}
    assert np.array_equal(rule.scores(d), again.scores(d))
    out = decode_selective(model, prompt, _cfg(lock="rule", rule=again, collect_rows=False, **GEOMS[geom]))
    assert len(out.tokens) == G and out.probe_state is not None and out.written_total >= G


@needs_lib
@pytest.mark.parametrize("geom", list(GEOMS))
def test_features2_lib_rows_and_checks(geom):
    res = decode_selective(_model(9), _prompt(9), _cfg(collect_probe=True, collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    rows = FL2.labelled_rows(res.collected, res.tokens, P, res.refresh_kind, "q", block_length=8)
    assert set(FL2.FEATURES + FL2.EXTRA + FL2.NEW) <= set(rows) and len(rows["y"]) == sum(len(c["position"]) for c in res.collected)
    chk = FL2.derive_check(rows, G, 8)
    assert chk["ok"], chk
    assert FL2.range_check(rows)["ok"]
    rows["n_cand"] = rows["n_cand"] + 1
    assert not FL2.derive_check(rows, G, 8)["ok"]
    rows["ll_kl"] = rows["ll_kl"] - 1.0
    assert not FL2.range_check(rows)["ok"]
