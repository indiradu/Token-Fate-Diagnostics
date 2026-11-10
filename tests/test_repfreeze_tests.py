"""Fixed representation-freeze rules, oracle gate and diagnostics flags (results/repfreeze_tests_20261029).

All new decoder options are default off: `freeze_rule="kl"`, `sync_gate_timer=False`, `force_hidden_output=False` must reproduce today's behaviour exactly.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.oracle_gate import NEVER, OracleGate, settled_ages  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from test_paper_freeze_gate import FRAMES, HIGH, VaryingHiddenModel, freezes, make_gate  # noqa: E402
from test_paper_token_kl_commitment import CountingRuntime, TinyTokenizer  # noqa: E402

LOW = [0.001, 0.60, 0.20, 0.10, 0.05, 0.049]


class ThawRuntime(CountingRuntime):
    def thaw(self) -> None:
        self.locked.clear()


def run(*, gen=4, block=2, frames=None, gate=None, prompt_gate=None, freeze_gate="kl", learned_rows="answer", freeze_answer_rows=True,
        scope="all_unmasked", percentile=20, events=True, runtime=ThawRuntime, **extra):
    config = PaperDecodeConfig(arm="CS-sparse", gen_length=gen, block_length=block, epsilon=0.005, confidence_percentile=percentile, threshold=0.9,
                               freeze_scope=scope, freeze_gate=freeze_gate, learned_rows=learned_rows, freeze_answer_rows=freeze_answer_rows,
                               mask_id=0, runtime_cls=runtime, **extra)
    return decode_paper_policy(VaryingHiddenModel(frames or FRAMES), TinyTokenizer(), "prompt", "ex", config,
                               gate=gate, prompt_gate=prompt_gate, collect_events=events)


# positions: 0 = the one prompt token (id 5, not EOS), 1..4 = answer slots, blocks {1,2} and {3,4}.
# call 0 writes 1 and 2 together (confident); then one slot per call (3 at call 1, 4 at call 2).
BLOCK_FRAMES = [{1: HIGH, 2: HIGH}] * 6


def key(result):
    return sorted((e["global_step"], e["position"]) for e in freezes(result))


# ------------------------------------------------------------------ fixed rules
def test_fix_prompt_freezes_every_prompt_row_after_call_zero_and_nothing_else():
    r = run(frames=BLOCK_FRAMES, freeze_rule="fix_prompt")
    assert key(r) == [(0, 0)]
    assert r.frozen_prompt_count == 1 and r.frozen_count == 0
    assert all(e["effective_from_step"] == 1 for e in freezes(r))


def test_fix_block_freezes_a_block_only_after_its_last_blank_is_written_and_the_row_was_in_the_input():
    r = run(frames=BLOCK_FRAMES, freeze_rule="fix_block")
    # block {1,2} is complete after call 0, but both rows were blanks in call 0's input: they freeze at call 1.
    # block {3,4}: 3 is written at call 1, 4 at call 2 (block complete after call 2); row 3 was in call 2's input and freezes at call 2; row 4
    # was only written at call 2 and the decode ends.
    assert key(r) == [(1, 1), (1, 2), (2, 3)]
    assert r.frozen_prompt_count == 0 and r.frozen_count == 3
    for step, pos in key(r):
        assert pos in (1, 2) and step >= 1 or pos == 3 and step == 2


def test_fix_block_never_freezes_a_row_of_a_block_that_still_has_a_blank():
    r = run(frames=BLOCK_FRAMES, freeze_rule="fix_block")
    committed = {e["position"]: e["global_step"] for e in r.events if e["type"] == "token_commit"}
    last_write = {1: max(committed[1], committed[2]), 2: max(committed[1], committed[2]), 3: max(committed[3], committed[4]), 4: max(committed[3], committed[4])}
    for step, pos in key(r):
        assert step >= last_write[pos] and step > committed[pos]


def test_fix_both_is_the_union():
    both = run(frames=BLOCK_FRAMES, freeze_rule="fix_both")
    assert key(both) == sorted(key(run(frames=BLOCK_FRAMES, freeze_rule="fix_prompt")) + key(run(frames=BLOCK_FRAMES, freeze_rule="fix_block")))


def test_fixed_rules_never_change_the_tokens_and_skip_the_vocabulary_kl():
    base = run(frames=BLOCK_FRAMES)
    for rule in ("fix_prompt", "fix_block", "fix_both"):
        r = run(frames=BLOCK_FRAMES, freeze_rule=rule)
        assert r.token_ids == base.token_ids and r.nfe == base.nfe
        assert r.raw_kl_min is None and r.raw_kl_exact_zero_count == 0


def test_eos_eot_prompt_rows_and_rows_in_frozen_blocks_are_not_candidates_twice():
    # long run: frozen rows leave the observed set, so nothing is frozen twice (commit would raise on duplicates only silently; check via events)
    r = run(gen=6, block=2, frames=[{1: HIGH, 2: HIGH, 3: HIGH, 4: HIGH}] * 6, freeze_rule="fix_both")
    pos = [p for _, p in key(r)]
    assert len(pos) == len(set(pos))


# ------------------------------------------------------------------ defaults reproduce today's behaviour
def test_freeze_rule_kl_is_explicit_default_and_diagnostic_flags_do_not_change_the_decode():
    for gate_kw in ({}, {"freeze_gate": "learned", "gate": make_gate()}):
        a = run(gen=5, block=5, frames=FRAMES, **gate_kw)
        b = run(gen=5, block=5, frames=FRAMES, freeze_rule="kl", sync_gate_timer=True, force_hidden_output=True, **gate_kw)
        assert a.token_ids == b.token_ids and a.nfe == b.nfe
        assert (a.frozen_count, a.frozen_prompt_count) == (b.frozen_count, b.frozen_prompt_count)
        assert [(e["global_step"], e["position"]) for e in freezes(a)] == [(e["global_step"], e["position"]) for e in freezes(b)]
        assert a.raw_kl_min == b.raw_kl_min


def test_fixed_rule_validation():
    with pytest.raises(ValueError, match="freeze_rule must"):
        run(freeze_rule="fix_everything")
    with pytest.raises(ValueError, match="CS-sparse"):
        run(freeze_rule="fix_prompt", scope="generated")
    with pytest.raises(ValueError, match="conflicts"):
        run(freeze_rule="fix_block", freeze_gate="learned", gate=make_gate())
    with pytest.raises(ValueError, match="conflicts"):
        run(freeze_rule="fix_block", freeze_answer_rows=False)


# ------------------------------------------------------------------ oracle gate
def test_settled_ages_takes_the_start_of_the_final_settled_run_and_ignores_unsettled_tails():
    # position 0: unsettled at age 1, settled from age 2 on; position 1: settled, unsettled, settled again (-> age after the last bad row);
    # position 2: unsettled at its last recorded row (never); position 3: no rows (never); position 4: settled from the first row.
    calls = np.array([1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 3, 4])
    pos = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 4])
    age = np.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2])
    dist = np.array([.4, .2, .1, .1, .5, .2, .1, .1, .1, .9, .2, .0])
    a = settled_ages(calls, pos, age, dist, 5)
    assert list(a) == [2, 3, NEVER, NEVER, 1]
    assert list(settled_ages(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0), 3)) == [NEVER] * 3


def test_oracle_gate_scores_by_age_and_respects_min_age_and_never():
    g = OracleGate("answer")
    g.set_example(np.array([2, NEVER, 1]), np.array([4]), prompt_length=1)
    g.reset(4, torch.device("cpu"))
    feats, rc = g.observe(None, torch.tensor([1, 2, 3]))
    assert feats.shape == (3, 1) and rc.shape == (3,)
    z = torch.zeros(3)
    s = g.score(feats, rc, z, z, z, z, torch.tensor([1, 5, 1]), progress=z)
    assert s.tolist() == [0.0, 0.0, 1.0]
    assert g.decide(s, torch.tensor([1, 5, 1])).tolist() == [False, False, True]
    assert g.decide(s, torch.tensor([1, 5, 0])).tolist() == [False, False, False]      # min_age = 1
    nv = OracleGate("answer", never=True)
    nv.set_example(np.array([1, 1, 1]), np.array([1]), prompt_length=1)
    nv.reset(4, torch.device("cpu"))
    assert nv.score(feats, rc, z, z, z, z, torch.tensor([9, 9, 9])).sum() == 0
    with pytest.raises(ValueError):
        g2 = OracleGate("answer")
        g2.set_example(np.array([1, 1]), np.array([1]), prompt_length=1)
        g2.reset(4, torch.device("cpu"))


def test_oracle_in_the_decoder_freezes_at_the_table_age_and_never_changes_tokens():
    base = run(gen=5, block=5, frames=FRAMES, percentile=100)
    # FRAMES: positions 1-3 written together at call 0. Table: answer pos 0,1 (positions 1,2) freeze at age 2; position 3 never; prompt row age 3.
    ans, prm = OracleGate("answer", min_age=1), OracleGate("prompt")
    ans.set_example(np.array([2, 2, NEVER, NEVER, NEVER]), np.array([NEVER]), 1)
    prm.set_example(np.array([2, 2, NEVER, NEVER, NEVER]), np.array([3]), 1)
    r = run(gen=5, block=5, frames=FRAMES, percentile=100, freeze_gate="learned", learned_rows="both", gate=ans, prompt_gate=prm)
    assert r.token_ids == base.token_ids
    assert key(r) == [(2, 0), (2, 1), (2, 2)] or key(r) == [(2, 1), (2, 2), (3, 0)], key(r)
    assert r.frozen_count == 2 and r.frozen_prompt_count == 1
    never_a, never_p = OracleGate("answer", never=True), OracleGate("prompt", never=True)
    never_a.set_example(np.array([1] * 5), np.array([1]), 1)
    never_p.set_example(np.array([1] * 5), np.array([1]), 1)
    n = run(gen=5, block=5, frames=FRAMES, percentile=100, freeze_gate="learned", learned_rows="both", gate=never_a, prompt_gate=never_p)
    assert n.frozen_count == 0 and n.frozen_prompt_count == 0 and n.token_ids == base.token_ids and n.nfe == base.nfe


# ------------------------------------------------------------------ Fast-dLLM prefix cache with refresh, sparse-none control
def test_sparse_none_freezes_nothing_and_keeps_the_tokens():
    base = run(frames=BLOCK_FRAMES)
    r = run(frames=BLOCK_FRAMES, freeze_rule="none")
    assert r.token_ids == base.token_ids and r.nfe == base.nfe
    assert r.frozen_count == 0 and r.frozen_prompt_count == 0 and r.raw_kl_min is None and r.refresh_calls == 0


def test_fast_prefix_refreshes_at_every_block_start_and_freezes_the_whole_prefix_afterwards():
    base = run(frames=BLOCK_FRAMES)
    r = run(frames=BLOCK_FRAMES, freeze_rule="fast_prefix", log_frozen=True)
    assert r.token_ids == base.token_ids and r.nfe == base.nfe == 3
    # call 0 starts block 0 and call 1 starts block 1: both are dense refresh passes; call 2 is inside block 1
    assert r.refresh_calls == 2 and r.refresh_steps == [0, 1]
    assert key(r) == [(0, 0), (1, 0), (1, 1), (1, 2)]      # prompt row after call 0; prompt row and rows 1,2 (block 0) again after the refresh at call 1
    assert r.frozen_by_call == [(0, 0), (0, 0), (2, 1)]    # (answer, prompt) rows frozen at each call's input: the refresh calls see none
    assert (r.frozen_count, r.frozen_prompt_count) == (2, 1)


def test_fast_prefix_block_zero_equals_fix_prompt_and_does_not_freeze_the_current_block():
    one_block = run(gen=2, block=2, frames=BLOCK_FRAMES, freeze_rule="fast_prefix")
    fp = run(gen=2, block=2, frames=BLOCK_FRAMES, freeze_rule="fix_prompt")
    assert key(one_block) == key(fp) == [(0, 0)]
    assert one_block.refresh_calls == 1


def test_fast_prefix_validation_and_log_frozen_default_off():
    with pytest.raises(ValueError, match="conflicts"):
        run(freeze_rule="fast_prefix", freeze_answer_rows=False)
    with pytest.raises(ValueError, match="CS-sparse"):
        run(freeze_rule="fast_prefix", scope="generated")
    assert run(frames=BLOCK_FRAMES).frozen_by_call == [] and run(frames=BLOCK_FRAMES).refresh_calls == 0


def test_runtime_thaw_makes_the_next_forward_a_dense_pass_and_commit_works_again():
    from test_paper_reference import _input, _model
    from regret_remasking.paper_reference import PaperReferenceRuntime
    x0 = _input()
    x1 = x0.clone()
    x1[:, 2, :] = torch.randn_like(x1[:, 2, :]) * 3.0
    native = _model()
    model = copy.deepcopy(native)
    rt = PaperReferenceRuntime(model, mode="sparse", total_len=5)
    rt.install()
    try:
        model(x0)
        rt.commit([2])
        frozen_out = model(x1)
        assert rt.last_head_rows == 4 and frozen_out.shape[1] == 4
        rt.thaw()
        thawed = model(x1)
        assert rt.last_head_rows == 5 and rt.last_attention_query_rows == len(rt.blocks) * 5
        assert torch.allclose(thawed, native(x1), atol=1e-6)
        rt.commit([1])                       # commit after a thaw freezes the fresh pass's state of the new rows only
        again = model(x1)
        assert again.shape[1] == 4 and torch.equal(rt.locked_positions.cpu(), torch.tensor([1]))
        assert torch.allclose(again, thawed.index_select(1, torch.tensor([0, 2, 3, 4])), atol=1e-6)
    finally:
        rt.remove()


# ------------------------------------------------------------------ refresh_every_blocks
THREE = [{1: HIGH, 2: HIGH, 3: HIGH, 4: HIGH, 5: HIGH, 6: HIGH}] * 6     # gen 6, block 2: one block per call


def test_refresh_every_blocks_default_is_k1_and_explicit_k1_is_identical():
    a = run(gen=6, block=2, frames=THREE, freeze_rule="fast_prefix", log_frozen=True)
    b = run(gen=6, block=2, frames=THREE, freeze_rule="fast_prefix", log_frozen=True, refresh_every_blocks=1)
    assert key(a) == key(b) and a.token_ids == b.token_ids and a.refresh_steps == b.refresh_steps == [0, 1, 2]
    assert a.frozen_by_call == b.frozen_by_call == [(0, 0), (0, 0), (0, 0)]


def test_refresh_every_two_blocks_thaws_only_at_even_blocks_and_carries_the_finished_block_otherwise():
    r = run(gen=6, block=2, frames=THREE, freeze_rule="fast_prefix", log_frozen=True, refresh_every_blocks=2)
    # block 0 (call 0, refresh): prompt row 0. block 1 (call 1, non-refresh): nothing thawed, rows 1,2 of block 0 freeze. block 2 (call 2, refresh):
    # everything thawed, then prompt row and rows 1-4 freeze again.
    assert key(r) == [(0, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    assert r.refresh_steps == [0, 2] and r.refresh_calls == 2
    assert r.frozen_by_call == [(0, 0), (0, 1), (0, 0)]


def test_large_k_never_thaws_and_freezes_each_finished_block_one_call_later():
    r = run(gen=6, block=2, frames=THREE, freeze_rule="fast_prefix", log_frozen=True, refresh_every_blocks=3)
    assert key(r) == [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)] and r.refresh_steps == [0]
    assert r.frozen_by_call == [(0, 0), (0, 1), (2, 1)]
    same_tokens = run(gen=6, block=2, frames=THREE)
    assert r.token_ids == same_tokens.token_ids


def test_active_by_call_splits_the_computed_rows_and_refresh_validation():
    r = run(gen=6, block=2, frames=THREE, freeze_rule="fast_prefix", log_frozen=True, refresh_every_blocks=2)
    # length 7 = 1 prompt + 6 answer. call 0: current block 2, later 4, unfrozen earlier 1 (prompt), refresh.
    # call 1: prompt frozen but rows 1,2 (the finished block) still live: earlier 2, current 2, later 2; call 2 refresh after thaw: current 2, later 0, earlier 5.
    assert r.active_by_call == [(2, 4, 1, 1), (2, 2, 2, 0), (2, 0, 5, 1)]
    for bad in (0, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            run(freeze_rule="fast_prefix", refresh_every_blocks=bad)
    with pytest.raises(ValueError, match="fast_prefix"):
        run(freeze_rule="fix_both", refresh_every_blocks=2)
    assert run(gen=6, block=2, frames=THREE).active_by_call == []


# ------------------------------------------------------------------ word lock (context rule, X-block) x representation freeze
def _xrun(rule, **kw):
    from regret_remasking.context_rules import ContextRule  # noqa: F401
    config = PaperDecodeConfig(arm="X-block", block_rule=rule, gen_length=kw.pop("gen", 6), block_length=kw.pop("block", 2), threshold=0.9, epsilon=0.005,
                               confidence_percentile=100, freeze_scope=kw.pop("scope", "all_unmasked"), mask_id=0, runtime_cls=ThawRuntime, **kw)
    return decode_paper_policy(VaryingHiddenModel(THREE), TinyTokenizer(), "prompt", "ex", config, collect_events=True)


def test_x_block_stays_dense_by_default_and_uses_the_sparse_layout_only_with_a_freeze_rule():
    from regret_remasking.context_rules import ContextRule
    rule = ContextRule("conf", 0.9)
    dense = _xrun(rule)
    assert dense.attention_layout == "full" and dense.frozen_count == 0 and dense.frozen_prompt_count == 0
    sparse = _xrun(rule, freeze_rule="none")
    assert sparse.attention_layout == "compact" and sparse.frozen_count == 0 and sparse.token_ids == dense.token_ids and sparse.nfe == dense.nfe
    both = _xrun(rule, freeze_rule="fix_both")
    assert both.attention_layout == "compact" and both.token_ids == dense.token_ids and both.frozen_count > 0 and both.frozen_prompt_count == 1


def test_x_block_with_fix_both_freezes_exactly_like_cs_sparse_with_the_same_word_lock():
    from regret_remasking.context_rules import ContextRule
    x = _xrun(ContextRule("conf", 0.9), freeze_rule="fix_both")
    c = run(gen=6, block=2, frames=THREE, freeze_rule="fix_both")
    assert x.token_ids == c.token_ids and x.nfe == c.nfe and key(x) == key(c) and (x.frozen_count, x.frozen_prompt_count) == (c.frozen_count, c.frozen_prompt_count)


def test_context_features_are_computed_from_blank_rows_and_the_token_state_not_from_frozen_rows(monkeypatch):
    # The rule sees (x, eligible blanks, their logp): blanks are never frozen, and x holds the same written tokens whether or not rows are frozen.
    import regret_remasking.context_rules as cr
    from regret_remasking.context_rules import ContextRule
    seen = {}
    real = cr.block_features

    def spy(*a, **k):
        f = real(*a, **k)
        seen.setdefault(spy.tag, []).append({n: v.copy() for n, v in f.items()})
        return f
    monkeypatch.setattr(cr, "block_features", spy)
    spy.tag = "dense"
    dense = _xrun(ContextRule("conf", 0.9))
    spy.tag = "frozen"
    frozen = _xrun(ContextRule("conf", 0.9), freeze_rule="fix_both")
    assert frozen.frozen_count > 0 and len(seen["dense"]) == len(seen["frozen"]) == dense.nfe
    for a, b in zip(seen["dense"], seen["frozen"]):
        assert a.keys() == b.keys() and all((a[n] == b[n]).all() for n in a)


def test_x_block_freeze_validation():
    from regret_remasking.context_rules import ContextRule
    with pytest.raises(ValueError, match="CS-sparse"):
        _xrun(ContextRule("conf", 0.9), freeze_rule="fix_both", scope="generated")
