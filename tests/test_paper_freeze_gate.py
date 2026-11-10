from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.freeze_gate import SCALARS, LearnedFreezeGate  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from test_paper_token_kl_commitment import CountingRuntime, ScriptedPosteriorModel, TinyTokenizer  # noqa: E402

D = 4
FEATURES = 3 * D


class FakeProjector:
    """Stands in for HiddenProjector: concatenated hidden states, rounded through float16 like the real one."""

    def encode_hidden(self, last_three):
        return torch.cat([h.float() for h in last_three], dim=-1).to(torch.float16).to(torch.float32)


class VaryingHiddenModel(ScriptedPosteriorModel):
    """Scripted posteriors plus hidden states that move from call to call, so rc_feat is non-zero."""

    def forward(self, input_ids, **kwargs):
        out = super().forward(input_ids, **kwargs)
        n = input_ids.shape[1]
        base = torch.sin(torch.arange(n, dtype=torch.float32)[None, :, None] + torch.arange(D)[None, None, :] + 0.3 * self.calls)
        out.hidden_states = tuple(base * (1 + 0.05 * layer) for layer in range(33))
        return out


def make_gate(*, bias=20.0, threshold=0.9, min_age=2, coef=None):
    n = FEATURES + len(SCALARS)
    return LearnedFreezeGate(FakeProjector(), torch.zeros(n), torch.ones(n), torch.zeros(n) if coef is None else coef, bias,
                             threshold=threshold, min_age=min_age, sha256="test")


HIGH = [0.001, 0.02, 0.95, 0.02, 0.005, 0.004]   # confident in token 2
FRAMES = [{1: HIGH, 2: HIGH, 3: HIGH}] * 6
STATIC = {1: HIGH, 2: HIGH, 3: HIGH}


def run(arm="CS-sparse", gate=None, *, freeze_gate="kl", scope="generated", percentile=100, gen=3, threshold=0.9, frames=None):
    model = VaryingHiddenModel(frames or FRAMES)
    config = PaperDecodeConfig(arm=arm, gen_length=gen, block_length=gen, epsilon=0.005, confidence_percentile=percentile,
                               threshold=threshold, freeze_scope=scope, freeze_gate=freeze_gate, mask_id=0,
                               runtime_cls=CountingRuntime if arm.endswith("sparse") else None)
    result = decode_paper_policy(model, TinyTokenizer(), "prompt", "ex", config, gate=gate, collect_events=True)
    return result, model


def freezes(result):
    return [e for e in result.events if e["type"] == "reference_freeze"]


# Positions 1-3 are confident (written together in call 0); positions 4 and 5 are not (one per later call), so rows 1-3
# can be frozen at call 1 and skipped from call 2.
def test_cs_sparse_writes_by_confidence_threshold_and_freezes_with_the_kl_rule():
    result, _ = run("CS-sparse", gen=5)
    commits = [e for e in result.events if e["type"] == "token_commit"]
    assert result.nfe == 3 < 5, "three blanks are written together in the first call"
    assert len({e["global_step"] for e in commits}) == result.nfe
    assert result.attention_layout == "compact" and result.frozen_count >= 3
    assert result.total_query_rows < result.nfe * (result.prompt_length + 5)
    reference, _ = run("C-block", gen=5)
    assert result.token_ids == reference.token_ids and result.nfe == reference.nfe, "freezing must not change the write rule"


def test_learned_gate_above_one_never_freezes_and_matches_the_dense_decode():
    gate = make_gate(threshold=1.1)
    result, _ = run("CS-sparse", gate, freeze_gate="learned")
    reference, _ = run("C-block")
    assert result.frozen_count == 0 and not freezes(result)
    assert result.token_ids == reference.token_ids and result.nfe == reference.nfe
    assert result.gate_sha256 == "test" and result.gate_threshold == 1.1 and result.gate_min_age == 2


def test_learned_gate_only_freezes_rows_at_least_min_age_old():
    frames = [{1: [0.001, 0.8, 0.1, 0.05, 0.03, 0.019]}] * 8  # one write per call: confidence 0.8 < 0.9 falls back to argmax
    for min_age in (2, 3):
        result, _ = run("CS-sparse", make_gate(min_age=min_age), freeze_gate="learned", gen=4, frames=frames)
        events = freezes(result)
        assert events, "an always-admit gate should freeze some rows"
        assert all(e["age"] >= min_age for e in events), [e["age"] for e in events]
        assert all(e["gate_score"] > 0.99 for e in events)
        committed = {e["position"]: e["global_step"] for e in result.events if e["type"] == "token_commit"}
        assert all(e["global_step"] - committed[e["position"]] >= min_age for e in events)


def test_gate_scores_event_lists_every_scored_candidate_each_call():
    frames = [{1: [0.001, 0.8, 0.1, 0.05, 0.03, 0.019]}] * 8
    result, _ = run("CS-sparse", make_gate(threshold=1.1), freeze_gate="learned", gen=4, frames=frames)
    scored = [e for e in result.events if e["type"] == "gate_scores"]
    assert scored and all(len(e["positions"]) == len(e["scores"]) == len(e["ages"]) for e in scored)
    assert all(a >= 1 for e in scored for a in e["ages"]), "only written rows are candidates"
    assert result.frozen_count == 0


def test_hybrid_prompt_rows_follow_the_fixed_rule_while_answer_rows_follow_the_gate():
    result, _ = run("CS-sparse", make_gate(threshold=1.1), freeze_gate="learned", scope="all_unmasked", gen=5)
    assert result.frozen_count == 0, "answer rows must wait for the gate"
    assert result.frozen_prompt_count > 0, "prompt rows must still use the KL rule"
    assert all(e["position"] < result.prompt_length for e in freezes(result))
    fixed, _ = run("CS-sparse", None, freeze_gate="kl", scope="all_unmasked", gen=5)
    assert result.frozen_prompt_count == fixed.frozen_prompt_count


def test_s_sparse_with_a_learned_gate_still_writes_one_blank_per_call():
    result, _ = run("S-sparse", make_gate(), freeze_gate="learned", gen=3)
    assert result.nfe == 3 and result.frozen_count >= 1
    assert all(e["age"] >= 2 for e in freezes(result))


def test_learned_gate_requires_a_gate_and_a_compatible_arm():
    with pytest.raises(ValueError, match="requires a gate"):
        run("CS-sparse", None, freeze_gate="learned")
    with pytest.raises(ValueError, match="learned freeze gate needs"):
        run("C-block", make_gate(), freeze_gate="learned")


def test_gate_observe_computes_the_relative_change_of_projected_features():
    gate = make_gate()
    gate.reset(5, torch.device("cpu"))
    h1 = tuple(torch.randn(1, 5, D) for _ in range(3))
    h2 = tuple(h + 0.3 * torch.randn(1, 5, D) for h in h1)
    pos = torch.tensor([1, 3, 4])
    f1, rc1 = gate.observe(h1, pos)
    assert torch.isnan(rc1).all(), "no previous call yet"
    f2, rc2 = gate.observe(h2, pos)
    want = (f2 - f1).norm(dim=-1) / f2.norm(dim=-1)
    assert torch.allclose(rc2, want, rtol=1e-5)
    f3, rc3 = gate.observe(h2, torch.tensor([0, 1]))
    assert torch.isnan(rc3[0]) and not torch.isnan(rc3[1]), "position 0 is new, position 1 was seen at the previous call"


def test_gate_scores_use_the_documented_scalar_order_and_transforms():
    n = FEATURES + len(SCALARS)
    for name, value, transform in [("rc_feat", 0.37, lambda v: v), ("log10_vkl", 3e-5, math.log10),
                                   ("log10_1_minus_echo", 0.99, lambda v: math.log10(1 - v)), ("entropy", 1.7, lambda v: v),
                                   ("top1_match", 1.0, lambda v: v), ("log1p_age", 4.0, lambda v: math.log1p(v))]:
        coef = torch.zeros(n)
        coef[FEATURES + SCALARS.index(name)] = 1.0
        gate = make_gate(bias=0.0, coef=coef)
        args = dict(rc_feat=0.0, vkl=1e-6, echo_p=0.5, entropy=0.0, top1_match=0.0, age=1.0)
        key = {"rc_feat": "rc_feat", "log10_vkl": "vkl", "log10_1_minus_echo": "echo_p", "entropy": "entropy",
               "top1_match": "top1_match", "log1p_age": "age"}[name]
        args[key] = value
        score = gate.score(torch.zeros(1, FEATURES), *[torch.tensor([args[k]]) for k in
                                                       ("rc_feat", "vkl", "echo_p", "entropy", "top1_match", "age")])
        # the other scalars contribute zero because their coefficients are zero
        assert float(score) == pytest.approx(1 / (1 + math.exp(-transform(value))), rel=1e-4), name


def test_gate_scores_zero_when_a_signal_is_missing():
    gate = make_gate()
    nan = torch.tensor([float("nan")])
    one = torch.tensor([1.0])
    score = gate.score(torch.zeros(1, FEATURES), nan, torch.tensor([1e-6]), torch.tensor([0.9]), one, one, torch.tensor([3.0]))
    assert float(score) == 0.0
