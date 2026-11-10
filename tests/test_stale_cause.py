"""CPU tests (toy LLaDA) for the cause-of-damage controls of the selective cache: CacheControl overrides (stale / fresh / noise / compute), resume-by-replay, first divergence."""
from __future__ import annotations

import pytest
import torch

from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, decode_selective, replay_state
from regret_remasking.stale_cause import (BLANKS, CLASSES, REPAIR_CLASSES, CacheControl, CtrlRecorder, KVOverride, dense_calls, events, first_divergence, first_token_diff,
                                          last_dense_before, repair_ctrl, row_classes, seed_for)
from regret_remasking.toy_llada import ToyLLaDA

MASK, EOS, V, P, G, BL = 3, 2, 40, 6, 24, 8


def _setup(seed, eos_boost=1.0):
    model = ToyLLaDA(vocab=V, d=32, n_heads=4, layers=3, ffn=48, seed=seed)
    if eos_boost != 1.0:
        with torch.no_grad():
            model.model.transformer.ff_out.weight[EOS] *= eos_boost
    g = torch.Generator().manual_seed(seed)
    return model, torch.randint(4, V, (1, P), generator=g)


def _cfg(**kw):
    base = dict(gen_length=G, shape="block", block_length=BL, mask_id=MASK, eos_id=EOS, eos_truncation=True, refresh="block", lock="conf", threshold=0.9, window=BL,
                far_mask_cache=True, log_detail=True)
    base.update(kw)
    return SelectiveConfig(**base)


ALL_FRESH = tuple((c, "fresh") for c in CLASSES)
DIVERGING = [2, 3, 4, 7]            # seeds whose S32 decode differs from the dense decode on the toy model


def test_all_stale_with_statistics_is_exactly_s32_and_records_nonzero_distances():
    for seed in (2, 3):
        model, ids = _setup(seed)
        base = decode_selective(model, ids, _cfg())
        ctrl = decode_selective(model, ids, _cfg(ctrl=CacheControl(stats=True)))
        assert ctrl.tokens == base.tokens and ctrl.nfe == base.nfe and ctrl.written == base.written
        st = ctrl.ctrl_stats
        assert st["prompt"]["calls_nonempty"] > 0 and max(st["far_blank"]["rel_k"]) > 0 and max(st["far_blank"]["rel_v"]) > 0
        assert all(r == 0 for r in st["prompt"]["ratio_k"])           # no noise in the stale arm


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 7])
def test_all_classes_fresh_equals_dense_even_where_s32_differs(seed):
    model, ids = _setup(seed)
    dense = decode_selective(model, ids, _cfg(refresh="every_call"))
    fresh = decode_selective(model, ids, _cfg(ctrl=CacheControl(sources=ALL_FRESH)))
    assert fresh.tokens == dense.tokens and fresh.nfe == dense.nfe
    assert [dict(zip(w, t)) for w, t in zip(fresh.written, fresh.written_tokens)] == events(dense.written, dense.written_tokens)
    if seed in DIVERGING:
        assert decode_selective(model, ids, _cfg()).tokens != dense.tokens


def test_toy_seeds_listed_as_diverging_really_diverge():
    for seed in DIVERGING:
        model, ids = _setup(seed)
        assert decode_selective(model, ids, _cfg()).tokens != decode_selective(model, ids, _cfg(refresh="every_call")).tokens


def test_noise_zero_equals_fresh_noise_is_seeded_and_realized_norm_matches_the_scale():
    model, ids = _setup(2)
    fresh = decode_selective(model, ids, _cfg(ctrl=CacheControl(sources=ALL_FRESH)))
    zero = decode_selective(model, ids, _cfg(ctrl=CacheControl(sources=tuple((c, "noise") for c in CLASSES), noise_scale=0.0)))
    assert zero.tokens == fresh.tokens
    for scale in (0.5, 1.0, 2.0):
        mk = lambda seed: _cfg(ctrl=CacheControl(sources=tuple((c, "noise") for c in CLASSES), noise_scale=scale, noise_seed=seed))
        a, a2, b = decode_selective(model, ids, mk(5)), decode_selective(model, ids, mk(5)), decode_selective(model, ids, mk(6))
        assert a.tokens == a2.tokens and a.ctrl_stats == a2.ctrl_stats               # deterministic given the seed
        for c in ("prompt", "far_blank"):
            # layer 0 has no staleness (K/V there depend only on the token): target norm 0, ratio 0 by construction; deeper layers hit the scale
            rk = [r for r in a.ctrl_stats[c]["ratio_k"] if r > 0]
            rv = [r for r in a.ctrl_stats[c]["ratio_v"] if r > 0]
            assert len(rk) >= 2 and len(rv) >= 2 and all(abs(r - scale) < 2e-3 for r in rk + rv), (scale, c, rk, rv)


def test_override_never_writes_the_caches_and_changes_only_the_configured_rows():
    model, ids = _setup(2)
    T = P + G
    cfg = _cfg(log_detail=False)
    eng, shadow = SelectiveEngine(model, T, cfg), SelectiveEngine(model, T, cfg)
    x = torch.full((1, T), MASK, dtype=torch.long)
    x[:, :P] = ids
    allp = torch.arange(T)
    eng.forward(x, allp, 0, allp[:1], allp[:1], x[0] == MASK, fresh=True)
    x[0, P:P + 4] = torch.tensor([9, 10, 11, 12])                                    # a few written rows, so caches go stale
    x[0, P + 4] = 13
    query = torch.tensor([P + 4, P + 5, P + 6, P + 7])
    classes = row_classes(x[0], query, P, 0, BL, MASK)
    ctrl = CacheControl(sources=(("far_blank", "noise"), ("prompt", "fresh")), noise_scale=1.0, noise_seed=1, stats=True)
    rec = CtrlRecorder(ctrl, 3, torch.device("cpu"))
    ov = ctrl.make_override(shadow, x, classes, rec, T)
    assert isinstance(ov, KVOverride)
    for l in range(3):
        kc, vc = eng.k_cache[l], eng.v_cache[l]
        kb, vb = kc.clone(), vc.clone()
        k_full, v_full = ov(l, kc, vc)
        assert torch.equal(kc, kb) and torch.equal(vc, vb)                            # caches untouched
        changed = (k_full != kc).any(dim=-1).any(dim=1)[0]                            # (T,) rows whose K differs from the cache
        want = torch.zeros(T, dtype=torch.bool)
        for c in ("far_blank", "prompt"):
            want[classes[c]] = True
        assert not bool(changed[~want].any())                                         # nothing outside the configured classes moves
        assert torch.equal(k_full[:, :, classes["prompt"]], shadow.k_cache[l][:, :, classes["prompt"]])     # fresh = the dense pass
        if l > 0:                                                                     # layer 0 K/V depend on the token only: nothing is stale there
            assert not torch.equal(k_full[:, :, classes["far_blank"]], shadow.k_cache[l][:, :, classes["far_blank"]])   # noise != fresh
    # a stale-only control without statistics needs no dense pass at all
    assert CacheControl().make_override(shadow, x, classes, rec, T) is None


def test_row_classes_partition_the_rows_outside_the_query_set():
    model, ids = _setup(3)
    T = P + 40                                                # 5 blocks of 8
    x = torch.full((T,), MASK, dtype=torch.long)
    x[:P] = ids[0]
    x[P:P + 11] = 9                                           # block 0 written, block 1 written up to its third slot
    query = torch.tensor([P + 10] + list(range(P + 11, P + 16)))        # written last call + the blanks of block 1 (current block)
    cl = row_classes(x, query, P, 1, BL, MASK)
    union = torch.cat([cl[c] for c in CLASSES])
    assert union.numel() == len(set(union.tolist())) == T - query.numel()
    assert set(cl["prompt"].tolist()) == set(range(P))
    assert set(cl["prior_written"].tolist()) == set(range(P, P + 8))
    assert set(cl["inblock_old"].tolist()) == {P + 8, P + 9}
    assert set(cl["next_blank"].tolist()) == set(range(P + 16, P + 24))
    assert set(cl["far_blank"].tolist()) == set(range(P + 24, P + 40))
    # a blank row left beyond a first-EOS point inside the current block is a far_blank row (still an input row)
    cl2 = row_classes(x, torch.tensor([P + 10, P + 11]), P, 1, BL, MASK)
    assert {P + 12, P + 15} <= set(cl2["far_blank"].tolist())


def test_compute_all_classes_equals_dense_and_an_empty_class_changes_nothing():
    model, ids = _setup(2)
    dense = decode_selective(model, ids, _cfg(refresh="every_call"))
    comp = decode_selective(model, ids, _cfg(ctrl=CacheControl(compute=CLASSES)))
    assert comp.tokens == dense.tokens and comp.nfe == dense.nfe
    # one block only: there is never a next block, so computing it adds no row and the decode is plain S32
    model, ids = _setup(2)
    base = decode_selective(model, ids, _cfg(gen_length=BL))
    one = decode_selective(model, ids, _cfg(gen_length=BL, ctrl=CacheControl(compute=("next_blank",))))
    assert one.tokens == base.tokens and one.rows_per_call == base.rows_per_call and one.written == base.written
    more = decode_selective(model, ids, _cfg(ctrl=CacheControl(compute=("next_blank",))))
    plain = decode_selective(model, ids, _cfg())
    assert sum(more.rows_per_call) > sum(plain.rows_per_call)


@pytest.mark.parametrize("seed,boost", [(2, 1.0), (3, 4.0), (4, 1.0)])
def test_resume_at_every_dense_call_reproduces_the_original_decode(seed, boost):
    model, ids = _setup(seed, boost)
    cfg = _cfg()
    full = decode_selective(model, ids, cfg)
    starts = dense_calls(full.refresh_kind)
    assert len(starts) >= 2
    if seed == 3:
        assert full.eos_position >= 0                                                 # the truncation state is exercised
    for c0 in starts:
        st = replay_state(ids[0].tolist(), G, full.written, full.written_tokens, c0, cfg)
        res = decode_selective(model, ids, cfg, resume=st)
        assert res.start_call == c0 and res.tokens == full.tokens
        assert res.nfe == full.nfe - c0 and res.written == full.written[c0:] and res.refresh_kind == full.refresh_kind[c0:]
        assert res.eos_position == full.eos_position or c0 == 0


def test_resume_is_refused_for_schedules_whose_state_a_dense_call_does_not_rebuild():
    model, ids = _setup(2)
    st = replay_state(ids[0].tolist(), G, [], [], 0, _cfg())
    with pytest.raises(ValueError):
        decode_selective(model, ids, _cfg(refresh="trigger"), resume=st)
    with pytest.raises(ValueError):
        decode_selective(model, ids, _cfg(far_mask_cache=False, ctrl=CacheControl(sources=ALL_FRESH)))


def test_first_divergence_kinds_and_dense_lookup():
    a = [{1: 5}, {2: 6, 3: 7}, {4: 8}]
    assert first_divergence(a, a) is None
    assert first_divergence([{1: 5}, {2: 6, 3: 9}], a) == {"call": 1, "pos": 3, "kind": "token", "tok_a": 9, "tok_b": 7}
    assert first_divergence([{1: 5}, {2: 6}], a)["kind"] == "b_only" and first_divergence([{1: 5}, {2: 6, 3: 7, 5: 1}], a)["kind"] == "a_only"
    assert first_divergence([{1: 5}], a)["kind"] == "length"
    assert first_divergence([{2: 6, 3: 7}, {4: 8}], a, offset_a=1) is None                 # run A resumed at call 1
    assert first_token_diff([1, 2, 3], [1, 2, 4]) == 2 and first_token_diff([1, 2], [1, 2]) is None
    kinds = ["first", "none", "none", "dense", "none", "dense", "none"]
    assert dense_calls(kinds) == [0, 3, 5] and last_dense_before(kinds, 4) == 3 and last_dense_before(kinds, 5) == 5
    assert seed_for("a", 1) == seed_for("a", 1) != seed_for("a", 2)


@pytest.mark.parametrize("seed", DIVERGING)
@pytest.mark.parametrize("mode", ["oracle", "compute"])
def test_oracle_repair_from_the_dense_call_before_the_first_divergence(seed, mode):
    model, ids = _setup(seed)
    cfg = _cfg()
    dense = decode_selective(model, ids, _cfg(refresh="every_call"))
    cached = decode_selective(model, ids, cfg)
    ev_d, ev_c = events(dense.written, dense.written_tokens), events(cached.written, cached.written_tokens)
    div = first_divergence(ev_c, ev_d)
    assert div is not None
    c0 = last_dense_before(cached.refresh_kind, div["call"])
    assert c0 < div["call"] and cached.call_blocks[c0] == cached.call_blocks[div["call"]]        # same block: the dense call that opens it
    st = replay_state(ids[0].tolist(), G, cached.written, cached.written_tokens, c0, cfg)
    blk = cached.call_blocks[c0]
    from dataclasses import replace
    # null repair = the cached decode again
    null = decode_selective(model, ids, cfg, resume=st)
    assert null.tokens == cached.tokens
    # every cached row repaired inside this block: the decode matches dense through the divergence call (and, on the toy, to the end)
    fixed = decode_selective(model, ids, replace(cfg, ctrl=repair_ctrl("all_cached", mode, blk)), resume=st)
    d2 = first_divergence(events(fixed.written, fixed.written_tokens), ev_d, offset_a=c0)
    assert d2 is None or d2["call"] > div["call"]
    # block-limited repair: the control acts only in calls that decode block `blk` (and are not its dense call)
    seg = decode_selective(model, ids, replace(cfg, ctrl=repair_ctrl("all_cached", mode, blk, stats=True)), resume=st)
    in_blk = sum(1 for b, k in zip(seg.call_blocks, seg.refresh_kind) if b == blk and k not in ("first", "dense"))
    assert in_blk > 0
    assert seg.ctrl_stats["prompt"]["calls_nonempty"] == (in_blk if mode == "oracle" else 0)       # compute mode: prompt rows are in the query set, not overridden
    # whole-decode repair with the same class set is dense-equivalent
    full_fix = decode_selective(model, ids, replace(cfg, ctrl=repair_ctrl("all_cached", mode, -1)), resume=st)
    assert full_fix.tokens == dense.tokens


def test_repair_classes_cover_the_partition_and_ctrl_validation():
    assert set(REPAIR_CLASSES["all_cached"]) == set(CLASSES)
    assert REPAIR_CLASSES["all_blank"] == BLANKS
    assert repair_ctrl(None, "oracle", 3) is None
    with pytest.raises(ValueError):
        CacheControl(sources=(("nope", "fresh"),))
    with pytest.raises(ValueError):
        CacheControl(sources=(("prompt", "bad"),))
    with pytest.raises(ValueError):
        repair_ctrl("prompt", "nothing", 0)
