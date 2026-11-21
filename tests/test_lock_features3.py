"""lock_features3 (the features3 inputs h/i/k/j computed live in the decoder) against features3_lib.compute (the offline computation the ablation used)."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from regret_remasking import lock_features3 as LF3
from regret_remasking.context_rules import FEATURES as RULE_FEATURES, ContextRule
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

_P = Path(__file__).resolve().parents[1] / "results/lctx_features3_20261115/orchestration/features3_lib.py"
if _P.exists():
    _spec = importlib.util.spec_from_file_location("features3_lib_lf3", _P)
    F3 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(F3)
else:
    F3 = None
needs_lib = pytest.mark.skipif(F3 is None, reason="results/lctx_features3_20261115 not present")

MASK, EOS, V, P, G, B = 3, 2, 40, 6, 24, 8
_rng = np.random.default_rng(5)
INFO = {k: (_rng.random(V) < 0.25).astype(np.int8) for k in LF3.INFO_KEYS}
INFO["ws"][MASK] = 0


def _model(seed=0):
    return ToyLLaDA(vocab=V, d=32, n_heads=4, layers=4, ffn=48, seed=seed)


def _prompt(seed=0):
    return torch.randint(4, V, (1, P), generator=torch.Generator().manual_seed(seed))


def _cfg(**kw):
    base = dict(gen_length=G, window=6, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="block", block_length=B, attn_impl="sdpa", lock="topk", topk=1, far_mask_cache=True,
                log_detail=True, token_info=INFO)
    base.update(kw)
    return SelectiveConfig(**base)


GEOMS = {"block": dict(shape="block"), "block_p": dict(shape="block", partial_every_k=2, partial_layer=2, partial_join="blank")}


@dataclass
class _Recorder(ContextRule):
    seen: list = None

    def scores(self, feats):
        self.seen.append({k: np.array(v, copy=True) for k, v in feats.items()})
        return feats["confidence"].astype(np.float64) * 0.5            # nothing reaches lam = 1: the best-confidence blank is written (= topk 1)


def test_name_lists_agree():
    assert set(LF3.FEATURES3) <= RULE_FEATURES and len(LF3.FEATURES3) == len(set(LF3.FEATURES3)) == 17
    assert [len(g) for g in LF3.GROUPS3.values()] == [5, 5, 4, 3]


@needs_lib
def test_lists_equal_the_offline_library():
    assert LF3.FEATURES3 == F3.NEW3 and LF3.BACK == F3.BACK
    assert {k: v for k, v in F3.GROUPS3.items() if k != "O"} == LF3.GROUPS3


def _drive(rows, prompt, info, gen_length, block):
    """Feed a trace to F3State call by call (state = writes of earlier calls), as the decoder does."""
    call = rows["call"].astype(np.int64)
    r = rows["position"].astype(np.int64) - len(prompt)
    st = LF3.F3State(prompt, info, gen_length, block)
    out = {f: np.full(len(call), np.nan, np.float32) for f in LF3.FEATURES3}
    for c in np.unique(call):
        idx = np.nonzero(call == c)[0]
        idx = idx[np.argsort(r[idx], kind="stable")]
        f = st.features(r[idx], rows["token"][idx], rows["confidence"][idx])
        for k in LF3.FEATURES3:
            out[k][idx] = f[k]
        w = idx[rows["written_now"][idx].astype(bool)]
        st.update(r[w], rows["token"][w])
    return out


@needs_lib
def test_driving_the_state_reproduces_compute_on_the_hand_built_trace():
    spec = importlib.util.spec_from_file_location("t3", Path(__file__).resolve().parent / "test_features3_lib.py")
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    rows = t3._trace()
    ref, _ = F3.compute(rows, t3.PROMPT, t3.INFO, gen_length=t3.G, block=t3.B)
    got = _drive(rows, t3.PROMPT, t3.INFO, t3.G, t3.B)
    for f in LF3.FEATURES3:
        assert np.array_equal(got[f], ref[f], equal_nan=True), f


@needs_lib
@pytest.mark.parametrize("geom", list(GEOMS))
def test_the_decoder_shows_a_rule_exactly_what_compute_gives_on_its_own_trace(geom):
    """End to end: a rule naming every FEATURES3 input decodes the toy model (writing the best blank, = topk 1); the trajectory's rows (call, position, proposal,
    confidence, written_now) go through the offline compute(); the features the rule was shown must be identical."""
    model, prompt = _model(11), _prompt(11)
    col = decode_selective(model, prompt, _cfg(collect_rows=True, **GEOMS[geom]))
    rec = _Recorder("conf", 1.0, features=list(LF3.FEATURES3), seen=[])
    via = decode_selective(model, prompt, _cfg(lock="rule", rule=rec, **GEOMS[geom]))
    assert via.tokens == col.tokens and len(rec.seen) == len(col.collected) and via.f3_state is not None
    calls, pos, tok, conf, wn = [], [], [], [], []
    for c in col.collected:
        n = len(c["position"])
        best = int(np.argmax(c["confidence"]))                                 # topk 1 writes the first most confident blank (stable argmax)
        calls.append(np.full(n, c["call"]))
        pos.append(c["position"])
        tok.append(c["token"])
        conf.append(c["confidence"])
        wn.append(np.arange(n) == best)
    rows = {"call": np.concatenate(calls), "position": np.concatenate(pos), "token": np.concatenate(tok), "confidence": np.concatenate(conf).astype(np.float32),
            "written_now": np.concatenate(wn)}
    final = np.array(col.tokens)
    rows["final_token"] = final[rows["position"] - P]
    ref, meta = F3.compute(rows, prompt[0].numpy(), INFO, gen_length=G, block=B)
    assert meta["v1"]["written_equals_final"] and meta["v1"]["positions_written"] == G
    k = 0
    for seen, c in zip(rec.seen, col.collected):
        n = len(c["position"])
        for f in LF3.FEATURES3:
            assert np.array_equal(seen[f], ref[f][k:k + n], equal_nan=True), (f, c["call"])
        k += n


@pytest.mark.parametrize("geom", list(GEOMS))
def test_a_gbm_rule_on_six_plus_f3_decodes_with_multi_write_and_stays_in_sync(geom, tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model, prompt = _model(12), _prompt(12)
    rec = _Recorder("conf", 1.0, features=["confidence"] + list(LF3.FEATURES3), seen=[])
    decode_selective(model, prompt, _cfg(lock="rule", rule=rec, **GEOMS[geom]))
    feats = ["confidence", "entropy", "margin"] + LF3.FEATURES3
    X = np.concatenate([np.stack([s[f] for f in feats], 1) for s in rec.seen]).astype("float32")
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    rule = ContextRule("gbm", 0.4, feats, HistGradientBoostingClassifier(max_iter=10, early_stopping=False, random_state=23).fit(X, y), "toy-F3")
    rule.save(tmp_path / "r.pkl")
    out = decode_selective(model, prompt, _cfg(lock="rule", rule=ContextRule.load(tmp_path / "r.pkl"), **GEOMS[geom]))
    assert len(out.tokens) == G and out.nfe < G and out.f3_state is not None                    # several words per call on some calls; check_state passed on every call
    assert np.array_equal(out.f3_state.state, np.array(out.tokens))


def test_default_off_and_missing_table():
    model, prompt = _model(13), _prompt(13)
    six = _Recorder("conf", 1.0, features=["confidence", "entropy"], seen=[])
    res = decode_selective(model, prompt, _cfg(shape="block", lock="rule", rule=six))
    assert res.f3_state is None and all(not set(s) & set(LF3.FEATURES3) for s in six.seen)
    assert not LF3.wants_f3(SelectiveConfig()) and LF3.wants_f3(SelectiveConfig(lock="rule", rule=_Recorder("conf", 1.0, features=["k_run"], seen=[])))
    with pytest.raises(ValueError):
        decode_selective(model, prompt, _cfg(shape="block", lock="rule", rule=_Recorder("conf", 1.0, features=["k_run"], seen=[]), token_info=None))


def test_check_state_catches_a_write_that_bypassed_update():
    st = LF3.F3State(np.array([5, 6, 7]), INFO, gen_length=8, block=4)
    st.update(np.array([0]), np.array([9]))
    st.check_state(np.array([9, MASK, MASK, MASK, MASK, MASK, MASK, MASK]), MASK)
    with pytest.raises(RuntimeError):
        st.check_state(np.array([9, 10, MASK, MASK, MASK, MASK, MASK, MASK]), MASK)


def test_mlp_ensemble_matches_torch_and_round_trips_as_a_rule(tmp_path):
    from regret_remasking.mlp_rule import MLPEnsemble

    class _Prep:
        def __init__(self, X):
            self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
            self.has_nan = np.isnan(X).any(0)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    X[rng.random(200) < 0.2, 2] = np.nan
    prep = _Prep(X)
    nets = []
    for s in (1, 2):
        torch.manual_seed(s)
        nets.append(torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(8, 1)).eval())
    ens = MLPEnsemble.from_torch(prep, nets)
    Z = ens.prep(X)
    with torch.no_grad():
        want = np.mean([torch.sigmoid(n(torch.from_numpy(Z))).squeeze(1).numpy() for n in nets], 0)
    assert np.allclose(ens.predict_proba(X)[:, 1], want, atol=1e-6)
    feats = ["confidence", "entropy", "k_run", "margin"]
    rule = ContextRule("gbm", 0.5, feats, ens, "toy-mlp")
    rule.save(tmp_path / "m.pkl")
    back = ContextRule.load(tmp_path / "m.pkl")
    d = {f: X[:, i] for i, f in enumerate(feats)}
    assert np.array_equal(back.scores(d), rule.scores(d))
