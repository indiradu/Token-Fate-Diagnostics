"""Learned gates for QUESTION (prompt) rows (results/gate_q_20261026): checkpoint contract and decoder wiring.

Defaults (learned_rows="answer", freeze_answer_rows=True) must reproduce the earlier hybrid exactly; the new scopes only add behaviour.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.freeze_gate import SCALARS, LearnedFreezeGate, check_scalars  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from test_paper_freeze_gate import FEATURES, FRAMES, FakeProjector, VaryingHiddenModel, freezes, make_gate  # noqa: E402
from test_paper_token_kl_commitment import CountingRuntime, TinyTokenizer  # noqa: E402


# ------------------------------------------------------------------ checkpoint contract
def make(scalars, *, bias=20.0, threshold=0.9, coef=None, row_type=None):
    n = FEATURES + len(scalars)
    return LearnedFreezeGate(FakeProjector(), torch.zeros(n), torch.ones(n), torch.zeros(n) if coef is None else coef, bias,
                             threshold=threshold, sha256="test", scalars=scalars, row_type=row_type)


ARGS = dict(rc_feat=0.1, vkl=1e-6, echo_p=0.9, entropy=1.0, top1_match=1.0, age=3.0)


def args_tensors():
    return (torch.zeros(1, FEATURES), *[torch.tensor([ARGS[k]]) for k in ("rc_feat", "vkl", "echo_p", "entropy", "top1_match", "age")])


def test_six_scalar_gate_scores_are_unchanged_and_ignore_progress():
    gate = make_gate(bias=0.3)
    a = gate.score(*args_tensors())
    b = gate.score(*args_tensors(), progress=torch.tensor([0.7]))
    assert torch.equal(a, b)
    assert gate.scalars == SCALARS and gate.row_type is None


def test_progress_scalar_is_appended_after_the_six_and_required():
    n = FEATURES + 7
    coef = torch.zeros(n)
    coef[FEATURES + 6] = 1.0
    gate = make(SCALARS + ["progress"], bias=0.0, coef=coef)
    p = gate.score(*args_tensors(), progress=torch.tensor([0.4]))
    assert float(p) == pytest.approx(1 / (1 + math.exp(-0.4)), rel=1e-5)
    with pytest.raises(ValueError, match="progress"):
        gate.score(*args_tensors())


def test_is_question_is_a_per_gate_constant_and_row_type_is_enforced():
    n = FEATURES + 8
    coef = torch.zeros(n)
    coef[FEATURES + 7] = 2.0
    for row_type in (0.0, 1.0):
        gate = make(SCALARS + ["progress", "is_question"], bias=0.0, coef=coef, row_type=row_type)
        p = gate.score(*args_tensors(), progress=torch.tensor([0.4]))
        assert float(p) == pytest.approx(1 / (1 + math.exp(-2.0 * row_type)), rel=1e-5)
    with pytest.raises(ValueError, match="row_type"):
        make(SCALARS + ["progress", "is_question"])
    with pytest.raises(ValueError, match="row_type"):
        make(SCALARS, row_type=1.0)


def test_scalar_list_validation_and_load_roundtrip():
    with pytest.raises(ValueError):
        check_scalars(SCALARS[:-1])
    with pytest.raises(ValueError):
        check_scalars(SCALARS + ["is_question"])
    with pytest.raises(ValueError):
        check_scalars(["progress"] + SCALARS)
    names = SCALARS + ["progress"]
    n = FEATURES + len(names)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.pt"
        torch.save({"mean": torch.zeros(n), "scale": torch.ones(n), "coef": torch.ones(n), "intercept": 0.1, "scalars": names}, path)
        gate = LearnedFreezeGate.load(path, FakeProjector(), threshold=0.5)
        assert gate.scalars == names and gate.feature_dim == FEATURES and gate.sha256
        torch.save({"mean": torch.zeros(n), "scale": torch.ones(n), "coef": torch.ones(n), "intercept": 0.1, "scalars": names[::-1]}, path)
        with pytest.raises(ValueError):
            LearnedFreezeGate.load(path, FakeProjector())


# ------------------------------------------------------------------ decoder wiring
class Spy(LearnedFreezeGate):
    """Records the positions it observes and the progress values it is scored with."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.observed, self.progress = [], []

    def observe(self, hidden3, positions):
        self.observed.append(positions.tolist())
        return super().observe(hidden3, positions)

    def score(self, *a, progress=None, **k):
        self.progress.append(None if progress is None else progress.tolist())
        return super().score(*a, progress=progress, **k)


def spy(bias=20.0, threshold=0.9, scalars=None):
    scalars = scalars or SCALARS + ["progress"]
    n = FEATURES + len(scalars)
    return Spy(FakeProjector(), torch.zeros(n), torch.ones(n), torch.zeros(n), bias, threshold=threshold, sha256="spy", scalars=scalars)


def run(*, gate=None, prompt_gate=None, freeze_gate="kl", learned_rows="answer", freeze_answer_rows=True, percentile=100, gen=5,
        scope="all_unmasked", frames=None, runtime=CountingRuntime):
    config = PaperDecodeConfig(arm="CS-sparse", gen_length=gen, block_length=gen, epsilon=0.005, confidence_percentile=percentile,
                               threshold=0.9, freeze_scope=scope, freeze_gate=freeze_gate, learned_rows=learned_rows,
                               freeze_answer_rows=freeze_answer_rows, mask_id=0, runtime_cls=runtime)
    return decode_paper_policy(VaryingHiddenModel(frames or FRAMES), TinyTokenizer(), "prompt", "ex", config,
                               gate=gate, prompt_gate=prompt_gate, collect_events=True)


def test_defaults_reproduce_the_hybrid_and_the_prompt_gate_is_ignored_when_learned_rows_is_answer():
    a = run(gate=make_gate(), freeze_gate="learned")
    b = run(gate=make_gate(), freeze_gate="learned", prompt_gate=spy())
    assert a.token_ids == b.token_ids and a.frozen_count == b.frozen_count and a.frozen_prompt_count == b.frozen_prompt_count
    assert [(e["global_step"], e["position"]) for e in freezes(a)] == [(e["global_step"], e["position"]) for e in freezes(b)]
    assert b.prompt_gate_sha256 is None


def test_learned_prompt_gate_decides_prompt_rows_without_kl_or_the_percentile_test():
    fixed = run(gate=make_gate(), freeze_gate="learned")
    assert fixed.frozen_prompt_count > 0
    both = run(gate=make_gate(), prompt_gate=spy(threshold=1.1), freeze_gate="learned", learned_rows="both")
    assert both.frozen_prompt_count == 0, "a prompt gate above 1 must veto the fixed rule's prompt freezes"
    assert both.frozen_count == fixed.frozen_count, "answer rows still follow the answer gate"
    open_ = run(gate=make_gate(), prompt_gate=spy(), freeze_gate="learned", learned_rows="both")
    assert open_.frozen_prompt_count == 1 and open_.prompt_gate_sha256 == "spy" and open_.prompt_gate_threshold == 0.9
    tight = run(gate=make_gate(), prompt_gate=spy(), freeze_gate="learned", learned_rows="both", percentile=1)
    assert tight.frozen_prompt_count == 1, "the percentile test does not apply to a learned prompt decision"


def test_prompt_rows_have_age_step_plus_one_and_need_age_two():
    pg = spy()
    result = run(gate=make_gate(), prompt_gate=pg, freeze_gate="learned", learned_rows="both")
    scored = [e for e in result.events if e["type"] == "gate_scores" and e.get("row_type") == "prompt"]
    assert scored and all(a == e["global_step"] + 1 for e in scored for a in e["ages"])
    pf = [e for e in freezes(result) if e["position"] < result.prompt_length]
    assert pf and all(e["age"] >= 2 and e["global_step"] >= 1 for e in pf)
    min3 = LearnedFreezeGate(FakeProjector(), torch.zeros(FEATURES + 7), torch.ones(FEATURES + 7), torch.zeros(FEATURES + 7), 20.0,
                             threshold=0.9, min_age=3, sha256="m3", scalars=SCALARS + ["progress"])
    late = run(gate=make_gate(), prompt_gate=min3, freeze_gate="learned", learned_rows="both")
    assert all(e["age"] >= 3 and e["global_step"] >= 2 for e in freezes(late) if e["position"] < late.prompt_length)


def test_progress_is_the_share_written_before_the_call_not_after():
    # Positions 1-3 are written together in call 0 (3 of 5 slots): the first scored call (call 1) must see 0.6, not 1.0 or 0.8.
    ag, pg = spy(), spy()
    run(gate=ag, prompt_gate=pg, freeze_gate="learned", learned_rows="both")
    first_prompt = next(p for p in pg.progress if p is not None)
    first_answer = next(p for p in ag.progress if p is not None)
    assert first_prompt == pytest.approx([0.6]) and all(v == pytest.approx(0.6) for v in first_answer)
    seq = [p[0] for p in pg.progress]
    assert seq == sorted(seq) and seq[0] < 1.0


def test_prompt_only_scopes_never_freeze_an_answer_row_and_do_not_observe_answer_rows():
    pg = spy()
    both = run(gate=make_gate(), freeze_gate="learned", prompt_gate=pg, learned_rows="both")
    assert both.frozen_count > 0
    pg2 = spy()
    po = run(prompt_gate=pg2, freeze_gate="kl", learned_rows="prompt", freeze_answer_rows=False)
    assert po.frozen_count == 0 and po.frozen_prompt_count == 1
    assert all(p < po.prompt_length for step in pg2.observed for p in step), "only prompt positions are observed"
    assert po.token_ids == both.token_ids


def test_fixed_rule_prompt_only_vetoes_answer_rows_but_keeps_prompt_decisions_of_the_full_rule():
    full = run(freeze_gate="kl")
    only = run(freeze_gate="kl", freeze_answer_rows=False)
    assert full.frozen_count > 0 and only.frozen_count == 0
    assert only.frozen_prompt_count == full.frozen_prompt_count > 0, "the percentile pool is unchanged, answer rows are only vetoed"
    assert only.token_ids == full.token_ids


def test_learned_answer_with_answer_rows_vetoed_is_rejected_and_other_validation():
    with pytest.raises(ValueError, match="conflicts"):
        run(gate=make_gate(), freeze_gate="learned", freeze_answer_rows=False)
    with pytest.raises(ValueError, match="prompt_gate"):
        run(freeze_gate="kl", learned_rows="prompt", freeze_answer_rows=False)
    with pytest.raises(ValueError, match="freeze_gate='learned'"):
        run(freeze_gate="kl", learned_rows="both", prompt_gate=spy())
    with pytest.raises(ValueError, match="freeze_gate='kl'"):
        run(gate=make_gate(), freeze_gate="learned", learned_rows="prompt", prompt_gate=spy())
    with pytest.raises(ValueError, match="all_unmasked"):
        run(freeze_gate="kl", learned_rows="prompt", prompt_gate=spy(), scope="generated")
    with pytest.raises(ValueError, match="all_unmasked"):
        run(freeze_gate="kl", freeze_answer_rows=False, scope="generated")
    with pytest.raises(ValueError, match="learned_rows must"):
        run(freeze_gate="kl", learned_rows="questions")
