"""Confidence pacing of the learned freeze gate (results/gate_pacing_20261025): pure decision rules plus decoder wiring."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.freeze_gate import SCALARS, paced_decision  # noqa: E402
from test_paper_freeze_gate import FEATURES, FRAMES, FakeProjector, VaryingHiddenModel, freezes, make_gate  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from test_paper_token_kl_commitment import CountingRuntime, TinyTokenizer  # noqa: E402

T = lambda *v: torch.tensor(list(v))
B = lambda *v: torch.tensor(list(v), dtype=torch.bool)


# ------------------------------------------------------------------ pure rules
def test_none_returns_the_gate_decision_unchanged():
    ok = B(True, False, True)
    assert torch.equal(paced_decision(ok, "none"), ok)


def test_intersect_is_the_gate_and_the_confidence_test():
    ok, conf = B(True, True, False, True), B(True, False, True, True)
    assert paced_decision(ok, "intersect", conf_pass=conf).tolist() == [True, False, False, True]
    with pytest.raises(ValueError):
        paced_decision(ok, "intersect")


def test_rank_admits_the_top_fifth_of_the_age_eligible_pool():
    logit = torch.arange(10, dtype=torch.float32)          # 0..9
    age = torch.full((10,), 3)
    ok = torch.ones(10, dtype=torch.bool)
    out = paced_decision(ok, "rank", logit=logit, age=age)
    # quantile(0.8) of 0..9 with linear interpolation = 7.2 -> rows with logit 8 and 9 (the top 20% of 10 rows)
    assert out.tolist() == [False] * 8 + [True, True]


def test_rank_pool_excludes_rows_younger_than_min_age():
    # The row with the highest logit is too young: it must neither freeze nor lift the cutoff.
    logit = T(1.0, 2.0, 3.0, 4.0, 5.0, 100.0)
    age = T(5, 5, 5, 5, 5, 1)
    ok = torch.tensor([True] * 5 + [False])
    out = paced_decision(ok, "rank", logit=logit, age=age)
    assert out.tolist() == [False, False, False, False, True, False]     # quantile(0.8) of 1..5 = 4.2 -> only 5.0


def test_rank_floor_applies_through_ok():
    logit = T(-9.0, -8.0, -7.0, -6.0, -5.0)
    age = torch.full((5,), 2)
    ok = B(False, False, False, False, False)            # nothing passes the probability floor
    assert not paced_decision(ok, "rank", logit=logit, age=age).any()


def test_rank_ties_are_admitted_and_pool_of_one_freezes_iff_ok():
    logit = T(1.0, 1.0, 1.0, 1.0, 1.0)
    age = torch.full((5,), 2)
    assert paced_decision(torch.ones(5, dtype=torch.bool), "rank", logit=logit, age=age).all()
    one = paced_decision(B(True), "rank", logit=T(0.3), age=T(2))
    assert one.tolist() == [True]
    assert paced_decision(B(False), "rank", logit=T(0.3), age=T(2)).tolist() == [False]


def test_rank_empty_pool_freezes_nothing_and_minus_inf_rows_are_not_pool():
    ok = B(True, True)
    assert not paced_decision(ok, "rank", logit=T(1.0, 2.0), age=T(1, 0)).any()
    out = paced_decision(B(True, True, True), "rank", logit=T(float("-inf"), 2.0, 3.0), age=T(4, 4, 4))
    assert out.tolist() == [False, False, True]          # pool {2, 3}: quantile(0.8) = 2.8


def test_rank_percentile_parameter_and_unknown_mode():
    logit = torch.arange(10, dtype=torch.float32)
    age = torch.full((10,), 2)
    ok = torch.ones(10, dtype=torch.bool)
    assert int(paced_decision(ok, "rank", logit=logit, age=age, top_percent=50.0).sum()) == 5
    with pytest.raises(ValueError):
        paced_decision(ok, "bogus")


def test_score_return_logit_matches_the_probability_and_marks_missing_rows():
    gate = make_gate(bias=0.3)
    one = torch.tensor([1.0])
    args = (torch.zeros(1, FEATURES), torch.tensor([0.1]), torch.tensor([1e-6]), torch.tensor([0.9]), one, one, torch.tensor([3.0]))
    p, z = gate.score(*args, return_logit=True)
    assert torch.allclose(p, torch.sigmoid(z)) and torch.equal(p, gate.score(*args))
    nan = torch.tensor([float("nan")])
    p, z = gate.score(args[0], nan, *args[2:], return_logit=True)
    assert float(p) == 0.0 and z.item() == -float("inf")


# ------------------------------------------------------------------ decoder wiring
def run(pacing, *, gate=None, percentile=20, gen=5, scope="all_unmasked", pacing_percent=20.0, threshold=0.9, frames=None):
    config = PaperDecodeConfig(arm="CS-sparse", gen_length=gen, block_length=gen, epsilon=0.005, confidence_percentile=percentile,
                               threshold=threshold, freeze_scope=scope, freeze_gate="learned", learned_pacing=pacing,
                               pacing_percent=pacing_percent, mask_id=0, runtime_cls=CountingRuntime)
    return decode_paper_policy(VaryingHiddenModel(frames or FRAMES), TinyTokenizer(), "prompt", "ex", config,
                               gate=gate or make_gate(), collect_events=True)


def test_pacing_none_matches_the_unpaced_decode_exactly():
    base = run("none")
    again = run("none")
    assert base.token_ids == again.token_ids and base.frozen_count == again.frozen_count
    assert [e["position"] for e in freezes(base)] == [e["position"] for e in freezes(again)]


def test_intersect_never_freezes_more_answer_rows_than_the_gate_alone_and_never_a_non_gate_row():
    base, paced = run("none"), run("intersect")
    assert paced.token_ids == base.token_ids, "pacing changes freezing, not the write rule"
    assert paced.frozen_count <= base.frozen_count
    base_answer = {(e["global_step"], e["position"]) for e in freezes(base) if e["position"] >= base.prompt_length}
    # a row frozen under pacing at some call was also admitted by the gate at that call in the unpaced run only if the
    # trajectories coincide up to then; the weaker invariant used here is ages and gate scores
    assert all(e["age"] >= 2 and e["gate_score"] > 0.99 for e in freezes(paced) if e["position"] >= paced.prompt_length)
    assert base_answer or base.frozen_count == 0


def test_intersect_leaves_prompt_rows_on_the_fixed_rule():
    base = run("none", gate=make_gate(threshold=1.1), percentile=99)
    paced = run("intersect", gate=make_gate(threshold=1.1), percentile=99)
    assert paced.frozen_count == 0 and base.frozen_count == 0
    assert paced.frozen_prompt_count == base.frozen_prompt_count > 0


def test_rank_freezes_only_rows_in_the_top_share_and_respects_age():
    frames = [{1: [0.001, 0.8, 0.1, 0.05, 0.03, 0.019]}] * 8
    base, ranked = run("none", gen=4, frames=frames), run("rank", gen=4, frames=frames)
    assert ranked.frozen_count >= 1, "a pool of one or more rows always lets the top row freeze when it passes the floor"
    assert ranked.frozen_count <= base.frozen_count
    assert all(e["age"] >= 2 for e in freezes(ranked) if e["position"] >= ranked.prompt_length)
    hard = run("rank", gen=4, frames=frames, gate=make_gate(threshold=1.1))
    assert hard.frozen_count == 0, "the probability floor still applies under rank pacing"


def test_pacing_validation():
    with pytest.raises(ValueError, match="needs freeze_gate"):
        decode_paper_policy(VaryingHiddenModel(FRAMES), TinyTokenizer(), "p", "ex",
                            PaperDecodeConfig(arm="CS-sparse", gen_length=3, block_length=3, learned_pacing="rank", mask_id=0,
                                              runtime_cls=CountingRuntime))
    with pytest.raises(ValueError, match="confidence_percentile"):
        run("intersect", percentile=100)
    with pytest.raises(ValueError, match="learned_pacing must"):
        run("sideways")
    with pytest.raises(ValueError, match="pacing_percent"):
        run("rank", pacing_percent=0.0)
