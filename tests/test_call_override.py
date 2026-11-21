"""Per-call override of the selective-cache engine (SelectiveConfig.call_override, collect_att_elig; results/lctx_setlock_20261112): default-off invariance, a no-op override reproduces the
reference exactly (the branch rollouts are replays), a threshold override changes only the writes of that call up to that call, a forced dense call keeps the writes of that call and the
refresh schedule of the other calls, and the eligible-row attention log matches the probe's attention."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

MASK, EOS, V, P, G = 3, 2, 40, 6, 32


def _model(seed=0):
    return ToyLLaDA(vocab=V, d=32, n_heads=4, layers=4, ffn=48, seed=seed)


def _prompt(seed=0):
    return torch.randint(4, V, (1, P), generator=torch.Generator().manual_seed(seed))


def _cfg(**kw):
    base = dict(gen_length=G, window=6, mask_id=MASK, eos_id=EOS, eos_truncation=True, refresh="block", block_length=8, attn_impl="sdpa", lock="conf", threshold=0.08, far_mask_cache=True,
                shape="block", partial_every_k=2, partial_layer=2, partial_join="blank", log_calls=True, log_detail=True)
    base.update(kw)
    return SelectiveConfig(**base)


def _ref(seed=0):
    return decode_selective(_model(seed), _prompt(seed), _cfg())


def test_default_off():
    r = _ref()
    assert r.overrides_applied == [] and r.att_elig == [] and r.nfe > 4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noop_override_reproduces_reference(seed):
    r = _ref(seed)
    for t in range(0, r.nfe, 2):
        b = decode_selective(_model(seed), _prompt(seed), _cfg(call_override=((t, 0.08, False),)))
        assert b.tokens == r.tokens and b.written == r.written and b.refresh_kind == r.refresh_kind and b.overrides_applied == [t]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threshold_override_changes_only_from_that_call(seed):
    r = _ref(seed)
    changed = 0
    for t in range(r.nfe - 1):
        b = decode_selective(_model(seed), _prompt(seed), _cfg(call_override=((t, 0.02, False),)))
        assert b.written[:t] == r.written[:t]
        assert set(r.written[t]) <= set(b.written[t])          # a lower threshold writes a superset at that call (same logits up to it)
        changed += b.written[t] != r.written[t]
    assert changed > 0


def test_higher_threshold_writes_subset_or_fallback():
    r = _ref()
    for t in range(r.nfe - 1):
        b = decode_selective(_model(), _prompt(), _cfg(call_override=((t, 0.99, False),)))
        assert b.written[:t] == r.written[:t] and len(b.written[t]) >= 1 and (set(b.written[t]) <= set(r.written[t]) or len(b.written[t]) == 1)


def test_forced_dense_keeps_schedule_and_that_calls_kind():
    r = _ref()
    cached = [t for t, k in enumerate(r.refresh_kind) if k in ("none", "partial")]
    assert cached
    for t in cached:
        b = decode_selective(_model(), _prompt(), _cfg(call_override=((t, -1.0, True),)))
        assert b.written[:t] == r.written[:t] and b.refresh_kind[t] == "dense" and b.overrides_applied == [t]
        # the rest of the schedule is that of the reference wherever the block structure is the same (dense at block starts, partial every 2nd call since the last NATURAL dense call)
        n = min(len(b.refresh_kind), len(r.refresh_kind))
        if b.call_blocks[:n] == r.call_blocks[:n]:
            assert [k for i, k in enumerate(b.refresh_kind[:n]) if i != t] == [k for i, k in enumerate(r.refresh_kind[:n]) if i != t]


def test_dense_override_on_a_dense_call_is_a_noop():
    r = _ref()
    t = next(i for i, k in enumerate(r.refresh_kind) if k == "dense")
    b = decode_selective(_model(), _prompt(), _cfg(call_override=((t, -1.0, True),)))
    assert b.tokens == r.tokens and b.refresh_kind == r.refresh_kind


def test_override_validation():
    with pytest.raises(ValueError):
        decode_selective(_model(), _prompt(), _cfg(call_override=((1, 0.5, False), (1, 0.4, False))))
    with pytest.raises(ValueError):
        decode_selective(_model(), _prompt(), _cfg(lock="topk", topk=1, call_override=((1, 0.5, False),)))
    with pytest.raises(ValueError):
        decode_selective(_model(), _prompt(), _cfg(refresh="every_call", call_override=((1, -1.0, True),)))


def test_att_elig_log():
    r = _ref()
    on = decode_selective(_model(), _prompt(), _cfg(collect_rows=True, collect_probe=True, probe_mid_layer=2, probe_attn_layer=1, collect_att_elig=True))
    assert on.tokens == r.tokens and len(on.att_elig) == on.nfe
    for a, c in zip(on.att_elig, on.collected):
        n = len(c["position"])
        assert a.shape == (n, n) and a.dtype == np.float16 and np.all(a >= 0) and np.all(a.astype(np.float32).sum(1) <= 1.0 + 1e-2)
        # the diagonal-free row sums over candidates reproduce att_cand of lock_features2 (candidates = confidence >= 0.5)
        cand = c["confidence"] >= 0.5
        a32 = a.astype(np.float32)
        expect = (a32 * cand[None, :]).sum(1) - np.diag(a32) * cand
        assert np.allclose(np.maximum(expect, 0), c["att_cand"], atol=2e-3)


_BL_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "results/lctx_setlock_20261112/orchestration/branch_lib.py"
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("branch_lib_setlock", _BL_PATH)
    BL = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(BL)
except (ImportError, FileNotFoundError):
    BL = None
needs_bl = pytest.mark.skipif(BL is None, reason="results/lctx_setlock_20261112 not present")


@needs_bl
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_branch_lib_write_sets_match_the_engine(seed):
    """write_set on the logged confidences reproduces the engine's writes at every call, for the reference threshold and for an overridden one; eligible_calls / choose_points behave."""
    r = decode_selective(_model(seed), _prompt(seed), _cfg(collect_rows=True))
    for t, c in enumerate(r.collected):
        assert BL.write_set(c["position"], c["confidence"], 0.08).tolist() == sorted(r.written[t])
    call = np.concatenate([np.full(len(c["position"]), c["call"]) for c in r.collected])
    conf = np.concatenate([c["confidence"] for c in r.collected])
    elig = BL.eligible_calls(call, conf, r.written, 0.03)
    for t in elig[:4]:
        c = r.collected[t]
        b = decode_selective(_model(seed), _prompt(seed), _cfg(call_override=((t, 0.03, False),)))
        assert sorted(b.written[t]) == BL.write_set(c["position"], c["confidence"], 0.03).tolist() and len(b.written[t]) > len(r.written[t])
    pts = BL.choose_points(list(range(20)), 20, 4, np.random.default_rng([23, 0]))
    assert len(pts) == 8 and sum(p < 10 for p in pts) == 4 and pts == BL.choose_points(list(range(20)), 20, 4, np.random.default_rng([23, 0]))
    assert BL.choose_points([1, 15], 20, 4, np.random.default_rng(0)) == [1, 15]
    flat, off = BL.ragged([np.ones((2, 2)), np.zeros((3, 3))], np.float16)
    assert BL.unragged(flat, off, 1, 3).shape == (3, 3) and off.tolist() == [0, 4, 13]
