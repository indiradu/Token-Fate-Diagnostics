"""Extra decode-time word-lock features (src/regret_remasking/lock_features.py, results/lctx_features_20261110): default-off invariance, hand-checked stability / neighbour / frontier values,
multi-write behaviour of the word-count persistence, and the rule path (an extended ContextRule scores under lock == 'rule')."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from regret_remasking import lock_features as LF
from regret_remasking.context_rules import FEATURES as RULE_FEATURES, ContextRule
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

_ORCH = Path(__file__).resolve().parents[1] / "results/lctx_features_20261110/orchestration"      # results/** is gitignored: skip the helper tests on a checkout without it
sys.path.insert(0, str(_ORCH))
try:
    import features_lib as FL  # noqa: E402
except ImportError:
    FL = None
needs_lib = pytest.mark.skipif(FL is None, reason="results/lctx_features_20261110 not present")

MASK, EOS, V, P, G = 3, 2, 40, 6, 24
SIX = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]
TOKEN_CLASS = np.random.default_rng(0).integers(0, 8, size=V).astype(np.int8)


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


def test_name_lists_agree_and_eot():
    assert set(LF.EXTRA_FEATURES) <= RULE_FEATURES and len(LF.EXTRA_FEATURES) == len(set(LF.EXTRA_FEATURES)) == 15
    assert [len(g) for g in LF.GROUPS.values()] == [3, 6, 4, 2]
    try:
        from regret_remasking.llada_trace import EOT_ID
    except Exception:       # pragma: no cover - llada_trace needs the full environment
        pytest.skip("llada_trace not importable")
    assert LF.EOT_ID == EOT_ID


@needs_lib
def test_features_lib_groups_equal_the_engine_groups():
    assert FL.GROUPS == LF.GROUPS and FL.EXTRA == LF.EXTRA_FEATURES


@pytest.mark.parametrize("geom", list(GEOMS))
def test_extras_are_default_off_and_never_change_the_trajectory(geom):
    model, prompt = _model(), _prompt()
    off = decode_selective(model, prompt, _cfg(**GEOMS[geom]))
    assert set(off.collected[0]) == set(SIX) | {"call", "position", "token", "n_first_block"} and off.extra_state is None
    on = decode_selective(model, prompt, _cfg(collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    assert set(on.collected[0]) == set(SIX) | set(LF.EXTRA_FEATURES) | {"call", "position", "token", "n_first_block"}
    assert on.tokens == off.tokens and on.nfe == off.nfe and on.refresh_kind == off.refresh_kind and on.written == off.written
    for a, b in zip(off.collected, on.collected):                       # the six inputs are untouched by the extras
        for f in SIX:
            assert np.array_equal(a[f], b[f])


def test_missing_token_class_table_fails_loudly():
    with pytest.raises(ValueError, match="token_class"):
        decode_selective(_model(), _prompt(), _cfg(shape="block", collect_extra=True))


def _logp(rows):
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_stability_features_by_hand_over_three_calls():
    """Three consecutive calls of one block with 3 blanks, one word written per call (W = 0, 1, 2)."""
    V6, dev = 6, "cpu"
    x = torch.full((1, 12), MASK)
    x[0, :4] = torch.tensor([9, 9, 9, 9]) % V6 + 4
    el = torch.tensor([4, 5, 6])
    st = LF.ExtraState()
    tc = np.zeros(V6, dtype=np.int8)
    # call 0: no previous observation
    lp0 = _logp([[4.0, 0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0, 0], [0, 0, 2.0, 0, 0, 0]])
    tok0 = lp0.argmax(-1)
    f0 = LF.extra_features(st, x, el, lp0.exp().max(-1).values, lp0, tok0, 0, 0, MASK, EOS, tc)
    assert np.all(np.isnan(f0["flip"])) and np.all(np.isnan(f0["kl_prev"])) and np.all(f0["run_words"] == 0)
    # call 1: blank 4 was written (W=1), blanks 5, 6 remain; blank 5 keeps its proposal, blank 6 changes
    x[0, 4] = tok0[0]
    el1 = torch.tensor([5, 6])
    lp1 = _logp([[0, 3.0, 0, 0, 0, 0], [0, 0, 0, 0, 2.5, 0]])
    tok1 = lp1.argmax(-1)
    f1 = LF.extra_features(st, x, el1, lp1.exp().max(-1).values, lp1, tok1, 1, 1, MASK, EOS, tc)
    assert f1["flip"].tolist() == [0.0, 1.0] and f1["run_words"].tolist() == [1.0, 0.0]
    kl0 = float((lp1[0].exp() * (lp1[0] - lp0[1])).sum())
    kl1 = float((lp1[1].exp() * (lp1[1] - lp0[2])).sum())
    assert f1["kl_prev"] == pytest.approx([kl0, kl1], abs=1e-6) and kl0 == pytest.approx(0.0, abs=1e-6) and kl1 > 0.1
    # call 2: blank 5 written (W=2); blank 6 keeps its (new) proposal for one word
    x[0, 5] = tok1[0]
    el2 = torch.tensor([6])
    lp2 = _logp([[0, 0, 0, 0, 2.7, 0]])
    f2 = LF.extra_features(st, x, el2, lp2.exp().max(-1).values, lp2, lp2.argmax(-1), 2, 2, MASK, EOS, tc)
    assert f2["flip"].tolist() == [0.0] and f2["run_words"].tolist() == [1.0]
    # a call that does not follow the previous one (resume) has no previous observation
    f9 = LF.extra_features(st, x, el2, lp2.exp().max(-1).values, lp2, lp2.argmax(-1), 7, 5, MASK, EOS, tc)
    assert np.isnan(f9["flip"]).all() and f9["run_words"].tolist() == [0.0]


def test_run_words_counts_words_not_calls_when_several_are_written():
    """The same blank, same proposal on two calls, 3 words written in between: run_words = 3 (calls would say 1); KL is divided by the 3 words."""
    V6 = 6
    x = torch.full((1, 14), MASK)
    x[0, :4] = 5
    el = torch.arange(4, 10)
    lp0 = _logp([[3.0, 1, 0, 0, 0, 0]] * 6)
    st = LF.ExtraState()
    LF.extra_features(st, x, el, lp0.exp().max(-1).values, lp0, lp0.argmax(-1), 0, 0, MASK, EOS, np.zeros(V6, dtype=np.int8))
    x[0, 4:7] = 1                                                       # the previous call wrote 3 words
    el1 = torch.arange(7, 10)
    lp1 = _logp([[2.0, 1.5, 0, 0, 0, 0]] * 3)
    f1 = LF.extra_features(st, x, el1, lp1.exp().max(-1).values, lp1, lp1.argmax(-1), 1, 3, MASK, EOS, np.zeros(V6, dtype=np.int8))
    assert f1["run_words"].tolist() == [3.0] * 3 and f1["flip"].tolist() == [0.0] * 3
    raw = float((lp1[0].exp() * (lp1[0] - lp0[0])).sum())
    assert f1["kl_prev"][0] == pytest.approx(raw / 3, rel=1e-5)


def _simulate(res, prompt_len):
    """Independent python re-derivation of run_words / flip from (call, position, token) and the words written per call."""
    W = 0
    prev = {}
    out = []
    for c, written in zip(res.collected, res.written):
        cur = {}
        rw, fl = [], []
        for pos, tok in zip(c["position"].tolist(), c["token"].tolist()):
            if pos in prev and prev[pos][0] == tok:
                start, f = prev[pos][1], 0.0
            elif pos in prev:
                start, f = W, 1.0
            else:
                start, f = W, math.nan
            cur[pos] = (tok, start)
            rw.append(W - start)
            fl.append(f)
        out.append((np.array(rw, float), np.array(fl, float)))
        prev = cur
        W += len(written)
    return out


@pytest.mark.parametrize("geom", list(GEOMS))
@pytest.mark.parametrize("topk", [1, 3])
def test_logged_stability_matches_an_independent_python_derivation(geom, topk):
    res = decode_selective(_model(4), _prompt(4), _cfg(collect_extra=True, token_class=TOKEN_CLASS, topk=topk, **GEOMS[geom]))
    assert (topk == 1) == all(len(w) == 1 for w in res.written)          # topk=3 really writes several words per call
    sim = _simulate(res, P)
    for c, (rw, fl) in zip(res.collected, sim):
        assert np.array_equal(c["run_words"], rw) and np.array_equal(c["flip"], fl, equal_nan=True)
    if geom == "block_p":
        first = [c for c in res.collected if c["n_first_block"] == 8]       # the first call of a block: nothing was observed before
        assert first and all(np.isnan(c["flip"]).all() and np.isnan(c["kl_prev"]).all() for c in first[:1])
    assert any(np.nanmax(c["run_words"]) >= 2 for c in res.collected[1:])


def test_neighbours_frontier_and_proposal_on_the_first_block_call():
    res = decode_selective(_model(5), _prompt(5), _cfg(shape="block", collect_extra=True, token_class=TOKEN_CLASS, topk=1))
    c = res.collected[0]
    n = len(c["position"])
    assert n == 8 and np.isnan(c["nbr_conf_l"][0]) and np.isnan(c["nbr_conf_r"][-1])                  # edge blanks: no computed neighbour on the outside
    assert np.array_equal(c["nbr_conf_l"][1:], c["confidence"][:-1]) and np.array_equal(c["nbr_conf_r"][:-1], c["confidence"][1:])
    assert np.array_equal(c["nbr_cand_l"][1:], (c["confidence"][:-1] >= 0.5).astype(np.float32)) and np.isnan(c["nbr_cand_l"][0])
    tok = c["token"]
    assert c["same_l"][0] == float(tok[0] == int(_prompt(5)[0, -1])) and np.isnan(c["same_r"][-1])    # left of the first blank: the prompt's last token; right of the last: an uncomputed blank
    assert np.array_equal(c["same_r"][:-1], (tok[:-1] == tok[1:]).astype(np.float32))                # interior: the neighbour's proposal
    assert np.array_equal(c["dist_leftmost"], np.arange(8)) and np.array_equal(c["rank_left"], np.arange(8))
    assert np.all((c["p_eos"] >= 0) & (c["p_eos"] <= 1))
    assert np.array_equal(c["tok_newline"], (TOKEN_CLASS[tok] & 1 != 0).astype(np.float32)) and np.array_equal(c["tok_punct"], (TOKEN_CLASS[tok] & 4 != 0).astype(np.float32))
    # later in the block: written words between blanks make rank_left differ from the distance
    later = [d for d in res.collected if d["n_first_block"] < 8]
    assert any(not np.array_equal(d["dist_leftmost"], d["rank_left"]) for d in later)


@dataclass
class _Recorder(ContextRule):
    seen: list = None

    def scores(self, feats):
        self.seen.append({k: np.array(v, copy=True) for k, v in feats.items()})
        return feats["confidence"].astype(np.float64) * 0.5            # nothing reaches lam = 1: the best-confidence blank is written (= topk 1)


@pytest.mark.parametrize("geom", list(GEOMS))
def test_a_rule_naming_extra_features_is_shown_exactly_what_collection_logs(geom):
    model, prompt = _model(6), _prompt(6)
    on = decode_selective(model, prompt, _cfg(collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    rec = _Recorder("conf", 1.0, features=list(LF.EXTRA_FEATURES), seen=[])
    via = decode_selective(model, prompt, _cfg(lock="rule", rule=rec, collect_rows=False, token_class=TOKEN_CLASS, **GEOMS[geom]))
    assert via.tokens == on.tokens and len(rec.seen) == len(on.collected)
    for seen, logged in zip(rec.seen, on.collected):
        for f in SIX + LF.EXTRA_FEATURES:
            assert np.array_equal(seen[f], logged[f], equal_nan=True), f


def test_a_six_input_rule_does_not_trigger_the_extra_features():
    rec = _Recorder("conf", 1.0, features=list(SIX), seen=[])
    res = decode_selective(_model(7), _prompt(7), _cfg(shape="block", lock="rule", rule=rec, collect_rows=False))
    assert res.extra_state is None and all(set(s) == set(SIX) for s in rec.seen)
    cfg = SelectiveConfig(lock="rule", rule=rec)
    assert not LF.wants_extra(cfg) and not LF.wants_extra(SelectiveConfig()) and LF.wants_extra(SelectiveConfig(collect_extra=True))


@pytest.mark.parametrize("geom", list(GEOMS))
def test_an_extended_gbm_rule_round_trips_and_decodes_with_multi_write(geom, tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model, prompt = _model(8), _prompt(8)
    col = decode_selective(model, prompt, _cfg(collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    feats = SIX + LF.EXTRA_FEATURES
    X = np.concatenate([np.stack([c[f] for f in feats], 1) for c in col.collected]).astype("float32")
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    gbm = HistGradientBoostingClassifier(max_iter=10, early_stopping=False, random_state=23).fit(X, y)
    rule = ContextRule("gbm", 0.5, feats, gbm, "toy-X")
    rule.save(tmp_path / "r.pkl")
    again = ContextRule.load(tmp_path / "r.pkl")
    d = {f: X[:, i] for i, f in enumerate(feats)}
    assert np.array_equal(rule.scores(d), again.scores(d))
    out = decode_selective(model, prompt, _cfg(lock="rule", rule=again, collect_rows=False, token_class=TOKEN_CLASS, **GEOMS[geom]))
    assert len(out.tokens) == G and out.fallback_calls >= 0 and out.extra_state is not None
    assert out.written_total >= G                                        # rule decode writes (possibly) several words per call and terminates
    with pytest.raises(ValueError):
        ContextRule("gbm", 0.5, ["confidence", "not_a_feature"], gbm)


@needs_lib
@pytest.mark.parametrize("geom", list(GEOMS))
def test_features_lib_derivation_matches_the_engine_log(geom):
    res = decode_selective(_model(9), _prompt(9), _cfg(collect_extra=True, token_class=TOKEN_CLASS, **GEOMS[geom]))
    rows = FL.labelled_rows(res.collected, res.tokens, P, res.refresh_kind, "q", block_length=8)
    assert FL.derive_check(rows, G, 8)["ok"], FL.derive_check(rows, G, 8)
    rows["run_words"] = rows["run_words"] + 1                           # a corrupted log must be caught
    assert not FL.derive_check(rows, G, 8)["ok"]


def _fake_result(share=0.06, auroc=0.012, lo=0.002, v2=True, v3=True):
    base = {"point": 0.0, "mean": 0.0, "ci95": [0.0, 0.0], "p_gt0": 0.0}
    sets = {k: {"share_diff_vs_six": dict(base, point=0.01), "auroc_diff_vs_six": dict(base, point=0.001, ci95=[-0.001, 0.003])} for k in ["six+a", "six+b", "six+c", "six+d", "six+all"]}
    sets["six+all"] = {"share_diff_vs_six": dict(base, point=share), "auroc_diff_vs_six": dict(base, point=auroc, ci95=[lo, auroc + 0.01])}
    return {"sets": sets, "validity": {"V2_derived_features_equal": v2, "V3_control_reproduces_refit": v3}}


def test_gate_logic_and_exit_code(tmp_path, monkeypatch):
    try:
        import check_gate as CG
    except ImportError:
        pytest.skip("results/lctx_features_20261110 not present")
    g = CG.evaluate_gate(_fake_result())
    assert g["best_set"] == "six+all" and g["passes"]
    assert not CG.evaluate_gate(_fake_result(share=0.0499))["passes"]               # G1 is a hard 5 points
    assert not CG.evaluate_gate(_fake_result(auroc=0.0099))["passes"]               # G2 point threshold
    assert not CG.evaluate_gate(_fake_result(lo=0.0))["passes"]                      # G2 interval must lie above 0
    assert not CG.evaluate_gate(_fake_result(v2=False))["passes"] and not CG.evaluate_gate(_fake_result(v3=False))["passes"]      # invalid experiment is not a pass
    import json
    monkeypatch.setattr(CG, "OUT", tmp_path)
    (tmp_path / "validation").mkdir()
    assert CG.main() == 1                                                            # no ablation result: fail closed
    (tmp_path / "validation/ablation_S32c.json").write_text(json.dumps(_fake_result()))
    assert CG.main() == 1                                                            # conditions hold but no rule file: fail closed
    (tmp_path / "rules/X").mkdir(parents=True)
    (tmp_path / "rules/X/rules.json").write_text("{}")
    (tmp_path / "rules/X/Lctx_X_c0.8.pkl").write_bytes(b"x")
    assert CG.main() == 0 and json.loads((tmp_path / "validation/gate.json").read_text())["passes"]
    (tmp_path / "validation/ablation_S32c.json").write_text(json.dumps(_fake_result(share=0.01)))
    assert CG.main() == 1


@needs_lib
def test_weighted_bootstrap_statistics_match_the_direct_computation():
    import fit_ablation as FA
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n, nq = 4000, 7
    q = rng.integers(0, nq, n)
    s = np.round(rng.random(n), 2)                                                   # ties on purpose
    y = (rng.random(n) < s).astype(int)
    m = np.array([0, 2, 1, 3, 0, 1, 2])
    rows = np.repeat(np.arange(n), m[q])                                            # the resample materialised
    auc, share, prec = FA.Boot(y, q, s).stats(m, 0.7)
    ys, ss = y[rows], s[rows]
    lam = FL.pooled_lambda(ss, ys, 0.7)
    assert auc == pytest.approx(roc_auc_score(ys, ss), abs=1e-12) and share == pytest.approx((ss >= lam).mean(), abs=1e-12) and prec == pytest.approx(ys[ss >= lam].mean(), abs=1e-12)
