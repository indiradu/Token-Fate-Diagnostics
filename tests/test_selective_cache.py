from __future__ import annotations

import math

import pytest
import torch

from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, _confident_write, decode_selective
from regret_remasking.toy_llada import ToyLLaDA

MASK, EOS, V, P, G = 3, 2, 40, 6, 24


def _setup(seed=0, layers=4, dtype=torch.float32):
    model = ToyLLaDA(vocab=V, d=32, n_heads=4, layers=layers, ffn=48, seed=seed).to(dtype)
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(4, V, (1, P), generator=g)
    return model, prompt


def _cfg(**kw):
    base = dict(gen_length=G, window=G, threshold=0.9, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", attn_impl="sdpa")
    base.update(kw)
    return SelectiveConfig(**base)


def _dense_reference_decode(model, prompt, cfg):
    """Plain dense decode with the same word lock (window over blanks, min(threshold, max) rule)."""
    x = torch.full((1, P + cfg.gen_length), MASK, dtype=torch.long)
    x[:, :P] = prompt
    masked = torch.arange(P, P + cfg.gen_length)
    calls = 0
    while masked.numel():
        win = masked[: cfg.window]
        logits = model(x).logits[:, win]
        pos, tok, eos_pos = _confident_write(logits, win, x, cfg.threshold, EOS)
        masked = masked[~torch.isin(masked, pos)]
        calls += 1
        if cfg.eos_truncation and eos_pos.numel():
            masked = masked[masked <= int(eos_pos.min())]
    x[x == MASK] = EOS
    return x[0, P:].tolist(), calls


def test_every_call_refresh_with_whole_answer_window_equals_dense_forward():
    model, prompt = _setup()
    cfg = _cfg()
    x = torch.full((1, P + G), MASK, dtype=torch.long)
    x[:, :P] = prompt
    eng = SelectiveEngine(model, P + G, cfg)
    rows = torch.arange(P + G)
    blank = x[0] == MASK
    win = torch.arange(P, P + G)
    logits, info = eng.forward(x, rows, 0, win, win, blank, fresh=True)
    want = model(x).logits[:, P:]
    assert torch.allclose(logits, want, atol=1e-5, rtol=1e-4)
    assert info["layer_rows"] == [P + G] * 4
    ids, calls = _dense_reference_decode(model, prompt, cfg)
    res = decode_selective(model, prompt, cfg)
    assert res.tokens == ids and res.nfe == calls


def test_every_call_refresh_bf16_close_to_dense():
    model, prompt = _setup(dtype=torch.bfloat16)
    x = torch.full((1, P + G), MASK, dtype=torch.long)
    x[:, :P] = prompt
    eng = SelectiveEngine(model, P + G, _cfg())
    win = torch.arange(P, P + G)
    logits, _ = eng.forward(x, torch.arange(P + G), 0, win, win, x[0] == MASK, fresh=True)
    assert torch.allclose(logits.float(), model(x).logits[:, P:].float(), atol=0.1, rtol=0.05)


def _write_some(x, positions, seed):
    g = torch.Generator().manual_seed(seed)
    x = x.clone()
    x[0, positions] = torch.randint(4, V, (len(positions),), generator=g)
    return x


def test_refresh_from_layer_zero_equals_dense_and_row_counts():
    model, prompt = _setup()
    cfg = _cfg(window=8, far_mask_cache=True, refresh="trigger")
    T = P + G
    x0 = torch.full((1, T), MASK, dtype=torch.long)
    x0[:, :P] = prompt
    eng = SelectiveEngine(model, T, cfg)
    win0 = torch.arange(P, P + 8)
    eng.forward(x0, torch.arange(T), 0, win0, win0, x0[0] == MASK, fresh=True)
    new = torch.tensor([P + 1, P + 5])
    x1 = _write_some(x0, new, 1)
    window = torch.arange(P + 8, P + 16)
    rows = torch.cat([new, window])
    dec = torch.arange(2, 2 + 8)
    blank = x1[0] == MASK
    # every non-query row joins at layer 0: the computed set is the whole sequence from layer 0 on, identical to a dense forward
    logits, info = eng.forward(x1, rows, 0, dec, dec, blank, refresh_from=0)
    assert info["layer_rows"] == [T] * 4
    assert torch.allclose(logits, model(x1).logits[:, window], atol=1e-5, rtol=1e-4)
    # upstream semantics: a trigger at layer 0 means rows join at layer 1, whose K/V of the cached rows are still the cached ones, so the result is only close
    eng2 = SelectiveEngine(model, T, cfg)
    eng2.forward(x0, torch.arange(T), 0, win0, win0, x0[0] == MASK, fresh=True)
    logits2, info2 = eng2.forward(x1, rows, 0, dec, dec, blank, refresh_from=1)
    assert info2["layer_rows"] == [rows.numel()] + [T] * 3
    want = model(x1).logits[:, window]
    assert float((logits2 - want).abs().max()) > 1e-6          # not exactly dense


def test_no_refresh_keeps_cached_rows_stale_and_counts_rows():
    model, prompt = _setup()
    cfg = _cfg(window=8, refresh="none")
    T = P + G
    x0 = torch.full((1, T), MASK, dtype=torch.long)
    x0[:, :P] = prompt
    eng = SelectiveEngine(model, T, cfg)
    win0 = torch.arange(P, P + 8)
    eng.forward(x0, torch.arange(T), 0, win0, win0, x0[0] == MASK, fresh=True)
    x1 = _write_some(x0, torch.tensor([P + 1]), 1)
    rows = torch.cat([torch.tensor([P + 1]), torch.arange(P + 8, P + 16)])
    dec = torch.arange(1, 9)
    logits, info = eng.forward(x1, rows, 0, dec, dec, x1[0] == MASK)
    assert info["layer_rows"] == [9] * 4 and info["joined"] == 0 and info["refresh_layer"] == -1


@pytest.mark.parametrize("cos", ["elastic_heads", "keys"])
def test_trigger_forced_and_never_fires(cos):
    model, prompt = _setup()
    T = P + G
    out = {}
    for name, gamma in (("always", 2.0), ("never", -2.0)):
        cfg = _cfg(window=6, refresh="trigger", gamma=gamma, trigger_cos=cos, eos_truncation=False)
        res = decode_selective(model, prompt, cfg)
        out[name] = res
    never, always = out["never"], out["always"]
    for lr, kind in zip(never.layer_rows[1:], never.refresh_kind[1:]):
        assert kind == "none" and lr == [lr[0]] * 4
    assert all(r == [r[0]] + [T] * 3 for r in always.layer_rows[1:])      # trigger at layer 0 every call: rows join at layer 1
    assert all(k == "trigger" for k in always.refresh_kind[1:])
    assert all(len(s) >= 1 for s in always.sims[1:]) and all(len(s) == 4 for s in never.sims[1:])
    assert all(-1.0 <= v <= 1.0 + 1e-5 for s in never.sims[1:] for v in s)


def test_heads_and_keys_cosine_differ():
    model, prompt = _setup()
    a = decode_selective(model, prompt, _cfg(window=6, refresh="trigger", gamma=-2.0, trigger_cos="elastic_heads"))
    b = decode_selective(model, prompt, _cfg(window=6, refresh="trigger", gamma=-2.0, trigger_cos="keys"))
    sa = [v for s in a.sims[1:] for v in s]
    sb = [v for s in b.sims[1:] for v in s]
    assert len(sa) == len(sb) > 0 and max(abs(u - v) for u, v in zip(sa, sb)) > 1e-4


def test_tracked_rows_are_written_or_prompt_rows_and_are_disjoint_from_new_and_window():
    model, prompt = _setup()
    res = decode_selective(model, prompt, _cfg(window=6, refresh="trigger", gamma=-2.0))
    seen_written = set()
    for call, (q, w, trk) in enumerate(zip(res.query_sets, res.written, res.tracked)):
        blanks_now = set(range(P, P + G)) - seen_written
        assert len(set(q)) == len(q)
        assert not (set(trk) & blanks_now), "tracked rows must be non-blank rows (written or prompt)"
        seen_written |= set(w)
    assert len(res.tracked[0]) >= 1


def test_block_schedule_dense_calls_at_block_starts():
    model, prompt = _setup()
    cfg = _cfg(window=6, refresh="block", block_length=8)
    res = decode_selective(model, prompt, cfg)
    T = P + G
    dense = [i for i, k in enumerate(res.refresh_kind) if k in ("first", "dense")]
    assert dense[0] == 0 and len(dense) >= 2
    assert all(res.layer_rows[i] == [T] * 4 for i in dense)
    assert all(res.layer_rows[i] != [T] * 4 for i in range(len(res.layer_rows)) if i not in dense)


def _scripted_engine(script, vocab=10):
    """Engine stub: call k returns the logits scripted as {position: (token, confidence)}; every other window row gets a flat distribution."""
    class Stub:
        def __init__(self):
            self.calls = []
            self.token_since = torch.zeros(1000, dtype=torch.long)

        def forward(self, tokens, rows, n_track, dec_local, mass_local, blank_mask, fresh=False):
            k = len(self.calls)
            self.calls.append(dict(rows=rows.tolist(), n_track=n_track, dec=[int(rows[i]) for i in dec_local], blanks=int(blank_mask.sum()), fresh=fresh))
            logits = torch.zeros(1, len(dec_local), vocab)
            for j, i in enumerate(dec_local.tolist()):
                pos = int(rows[i])
                if pos in script[k]:
                    tok, conf = script[k][pos]
                    logits[0, j, tok] = math.log(conf * (vocab - 1) / (1 - conf))
            return logits, {"layer_rows": [len(rows)], "refresh_layer": -1, "sims": [], "joined": 0, "tracked": None}
    return Stub()


def test_eos_truncation_and_final_fill_and_progress_guarantee():
    P_, G_ = 4, 12
    prompt = torch.arange(10, 10 + P_)[None]
    # call 0: window = positions 4..7. Row 5 confident EOS; nothing else passes the threshold -> only the most confident row(s) are written.
    script = [
        {5: (EOS_TOK, 0.95), 4: (7, 0.4), 6: (8, 0.3)},
        {4: (7, 0.99), 6: (8, 0.99)},
    ]
    stub = _scripted_engine(script)
    cfg = SelectiveConfig(gen_length=G_, window=4, threshold=0.9, mask_id=MASK_TOK, eos_id=EOS_TOK, eos_truncation=True, refresh="none")
    res = decode_selective(None, prompt, cfg, engine=stub, sync=False)
    assert res.eos_position == 5
    assert res.nfe == 2                                     # blanks after the EOS are dropped; 4 and 6? 6 > 5 dropped, 4 written on call 1
    assert stub.calls[1]["dec"] == [4]                      # window = remaining blanks <= eos position
    assert res.tokens == [7, EOS_TOK] + [EOS_TOK] * 10      # rows beyond the first EOS (and unwritten) become EOS
    # without truncation the rows beyond are decoded as usual (here: never written by the script -> fallback writes the max each call)
    stub2 = _scripted_engine(script + [{}] * 20)
    cfg2 = SelectiveConfig(gen_length=G_, window=4, threshold=0.9, mask_id=MASK_TOK, eos_id=EOS_TOK, eos_truncation=False, refresh="none")
    res2 = decode_selective(None, prompt, cfg2, engine=stub2, sync=False)
    assert res2.nfe > 2 and all(len(w) >= 1 for w in res2.written)       # at least one write per call
    assert sorted(p for w in res2.written for p in w) == list(range(P_, P_ + G_))   # every answer row written exactly once


def test_row_set_bookkeeping_new_rows_then_window():
    P_, G_ = 4, 12
    prompt = torch.arange(10, 10 + P_)[None]
    script = [{4: (7, 0.95), 5: (8, 0.95)}, {6: (9, 0.95)}] + [{}] * 20
    stub = _scripted_engine(script)
    cfg = SelectiveConfig(gen_length=G_, window=3, threshold=0.9, mask_id=MASK_TOK, eos_id=EOS_TOK, eos_truncation=False, refresh="none")
    res = decode_selective(None, prompt, cfg, engine=stub, sync=False)
    c0, c1, c2 = stub.calls[:3]
    assert c0["rows"] == list(range(P_ + G_)) and c0["fresh"]
    assert res.written[0] == [4, 5]
    assert c1["rows"] == [4, 5, 6, 7, 8] and c1["dec"] == [6, 7, 8] and not c1["fresh"]      # [rows written last call] + next 3 blanks
    assert c2["rows"][:1] == [6] and c2["dec"] == [7, 8, 9]


def test_far_mask_cache_off_computes_every_blank_but_decodes_window():
    P_, G_ = 4, 12
    prompt = torch.arange(10, 10 + P_)[None]
    stub = _scripted_engine([{4: (7, 0.95)}] + [{}] * 30)
    cfg = SelectiveConfig(gen_length=G_, window=3, far_mask_cache=False, threshold=0.9, mask_id=MASK_TOK, eos_id=EOS_TOK, eos_truncation=False, refresh="none")
    decode_selective(None, prompt, cfg, engine=stub, sync=False)
    c1 = stub.calls[1]
    assert c1["rows"] == [4] + list(range(5, P_ + G_)) and c1["dec"] == [5, 6, 7]


def test_selective_exposure_extremes():
    model, prompt = _setup()
    T = P + G
    res_none = decode_selective(model, prompt, _cfg(window=6, refresh="selective", score_layer=1, exposure_theta=1e9, score_heads=0))
    assert all(j == 0 for j in res_none.joined_rows)
    res_all = decode_selective(model, prompt, _cfg(window=6, refresh="selective", score_layer=1, exposure_theta=-1.0, score_heads=0))
    assert all(r[2:] == [T] * (len(r) - 2) for r in res_all.layer_rows[1:])           # every cached row rejoins at layer score_layer + 1
    mid = decode_selective(model, prompt, _cfg(window=6, refresh="selective", score_layer=1, exposure_theta=0.05, score_heads=0))
    assert 0 < sum(mid.joined_rows) < sum(res_all.joined_rows)


EOS_TOK, MASK_TOK = 9, 99


# ---------------------------------------------------------------------------------------------- block-aligned shape (S32) and the paper_decode word locks
class _Tok:
    padding_side = "left"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = torch.tensor([[5 + (ord(c) % 30) for c in text[:8]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x".join(map(str, ids[0]))]


def _dense_paper(model, block=8, **spec):
    from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
    cfg = PaperDecodeConfig(gen_length=G, block_length=block, mask_id=MASK, **spec)
    return decode_paper_policy(model, _Tok(), "abcdefgh", "q", cfg)


def _prompt_ids():
    from regret_remasking.locking_policy import _prepare_prompt
    return _prepare_prompt(_Tok(), "abcdefgh", torch.device("cpu"))[0]


def _s32_cfg(**kw):
    base = dict(gen_length=G, shape="block", block_length=8, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", attn_impl="sdpa", far_mask_cache=True)
    base.update(kw)
    return SelectiveConfig(**base)


@pytest.mark.parametrize("thr", [0.9, 0.05])
def test_s32_every_call_with_conf_lock_equals_dense_c_block(thr):
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    dense = _dense_paper(model, arm="C-block", threshold=thr)
    res = decode_selective(model, ids, _s32_cfg(lock="conf", threshold=thr, window=8))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe


def test_s32_every_call_with_context_rule_equals_dense_x_block():
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    from regret_remasking.context_rules import ContextRule
    rng = np.random.default_rng(0)
    X = rng.random((400, 6)).astype("float32")
    y = (X[:, 0] + 0.3 * X[:, 3] / 8 + 0.2 * rng.random(400)) > 0.6
    feats = ["confidence", "entropy", "margin", "dist_written", "local_mask_ratio", "pos_in_block"]
    rule = ContextRule("gbm", 0.5, feats, GradientBoostingClassifier(n_estimators=15, max_depth=2, random_state=0).fit(X, y), "toy")
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    dense = _dense_paper(model, arm="X-block", block_rule=rule)
    res = decode_selective(model, ids, _s32_cfg(lock="rule", rule=rule))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe
    assert res.rule_time_s > 0 and res.written_total == G


def test_s32_block_schedule_runs_and_rows_are_new_plus_block_blanks():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    res = decode_selective(model, ids, _s32_cfg(lock="conf", threshold=0.05, refresh="block", window=8))
    T = int(ids.shape[1]) + G
    dense = [i for i, k in enumerate(res.refresh_kind) if k in ("first", "dense")]
    assert len(dense) == 3 and all(res.layer_rows[i] == [T] * 3 for i in dense)           # one dense call per block of 8
    for i, (q, w) in enumerate(zip(res.query_sets, res.written)):
        if i in dense:
            continue
        blanks_block = set(q) - set(res.written[i - 1])
        assert set(res.written[i - 1]) <= set(q) and len(blanks_block) <= 8                  # rows written last call + the current block's blanks only


def test_s32_stub_eos_truncation_drops_blanks_beyond_eos_inside_the_block():
    P_, G_ = 4, 16
    prompt = torch.arange(10, 10 + P_)[None]
    script = [{5: (EOS_TOK, 0.95)}, {4: (7, 0.99)}]
    stub = _scripted_engine(script + [{}] * 5)
    cfg = SelectiveConfig(gen_length=G_, shape="block", block_length=8, lock="elastic", threshold=0.9, mask_id=MASK_TOK, eos_id=EOS_TOK, eos_truncation=True, refresh="none")
    res = decode_selective(None, prompt, cfg, engine=stub, sync=False)
    assert res.eos_position == 5 and stub.calls[1]["dec"] == [4] and res.nfe == 2
    assert res.tokens[:2] == [7, EOS_TOK] and set(res.tokens[2:]) == {EOS_TOK}


def test_s16_context_rule_transfer_runs_with_window_eligible_rows():
    from regret_remasking.context_rules import ContextRule
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    rule = ContextRule("conf", 0.05)
    res = decode_selective(model, ids, _s32_cfg(shape="window", window=6, lock="rule", rule=rule, refresh="block"))
    assert res.written_total == G and res.nfe > 0


def test_refresh_every_k_adds_dense_calls_and_k1_equals_every_call():
    model, prompt = _setup(layers=3)
    T = P + G
    base = decode_selective(model, prompt, _cfg(window=6, refresh="block", block_length=8))
    k2 = decode_selective(model, prompt, _cfg(window=6, refresh="block", block_length=8, refresh_every_k=2))
    k1 = decode_selective(model, prompt, _cfg(window=6, refresh="block", block_length=8, refresh_every_k=1))
    every = decode_selective(model, prompt, _cfg(window=6, refresh="every_call"))
    nd = lambda r: sum(1 for k in r.refresh_kind if k in ("first", "dense"))
    assert nd(k2) > nd(base) and nd(k1) == k1.nfe
    assert k1.tokens == every.tokens and k1.nfe == every.nfe
    assert all(lr == [T] * 3 for lr, k in zip(k2.layer_rows, k2.refresh_kind) if k in ("first", "dense"))


def test_s32_without_far_blank_cache_computes_every_blank_and_every_call_equals_dense():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    T = int(ids.shape[1]) + G
    dense = _dense_paper(model, arm="C-block", threshold=0.9)
    res = decode_selective(model, ids, _s32_cfg(lock="conf", threshold=0.9, window=8, refresh="every_call", far_mask_cache=False))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe
    res2 = decode_selective(model, ids, _s32_cfg(lock="conf", threshold=0.05, window=8, refresh="block", far_mask_cache=False))
    # prompt and written rows are cached, but every blank row is computed on every non-dense call
    for i, (lr, k) in enumerate(zip(res2.layer_rows, res2.refresh_kind)):
        if k not in ("first", "dense"):
            assert lr[0] >= 1 and lr[0] < T
    assert res2.written_total == G


# ---------------------------------------------------------------------------------------------- shared staleness state
def _blk(**kw):
    base = dict(lock="conf", threshold=1.5, refresh="block", window=8, log_detail=True)     # 1.5: one word per call, so 24 calls over 3 blocks
    base.update(kw)
    return _s32_cfg(**base)


def test_staleness_options_off_reproduce_the_block_schedule_exactly():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    base = decode_selective(model, ids, _blk())
    for kw in (dict(log_stale=True), dict(stale_budget=0, stale_score="random"), dict(stale_budget=0, lock_stale_beta=0.0, stale_score="age", log_stale=True)):
        r = decode_selective(model, ids, _blk(**kw))
        assert r.tokens == base.tokens and r.nfe == base.nfe and r.rows_per_call == base.rows_per_call and r.query_sets == base.query_sets
    assert all(s == 0.0 for s in decode_selective(model, ids, _blk()).lock_shift)


def test_snap_is_the_last_call_each_position_was_computed_on_every_schedule():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    T = int(ids.shape[1]) + G
    for refresh in ("block", "every_call", "none"):
        cfg = _blk(refresh=refresh)
        eng = SelectiveEngine(model, T, cfg)
        res = decode_selective(model, ids, cfg, engine=eng)
        want = torch.zeros(T, dtype=torch.long)
        for c, q in enumerate(res.query_sets):
            want[torch.tensor(q)] = c
        assert torch.equal(eng.snap.cpu(), want), refresh


def test_stale_scores_rank_changed_context_and_blank_rows_and_pick_excludes_the_query():
    model, _ = _setup(layers=2)
    T = 40
    cfg = _blk(stale_budget=2, gen_length=T - 8)
    eng = SelectiveEngine(model, T, cfg)
    eng.call = 6
    eng.snap[:] = 5                                    # everything was refreshed on call 5 ...
    eng.snap[10] = 0                                   # ... except rows 10 (blank) and 12 (written) and 30 (blank, far from every write)
    eng.snap[12] = 0
    eng.snap[30] = 0
    eng.token_since[:] = 0
    eng.token_since[11] = 3                            # a token appeared at position 11 on call 3: after the refresh of rows 10 and 12, not of the others
    x = torch.full((T,), MASK)
    x[:8] = 7                                          # prompt
    x[12] = 9                                          # written row
    s = eng.stale_scores(x, 8)
    assert s[10] > s[30] > s[12] > 0                   # blank row next to the change > blank row 19 positions away (proximity kernel) > written row next to it (type prior 1.0 vs 0.25)
    assert float(s[[0, 1, 2, 11, 20]].abs().max()) == 0.0
    pick = eng.pick_stale(s, torch.tensor([10, 3]), 2)
    assert 10 not in pick.tolist() and set(pick.tolist()) == {12, 30}      # never an already-computed row; zero-score rows are not picked
    assert eng.pick_stale(s, torch.tensor([10]), 99).tolist() == [12, 30]
    dropped = torch.zeros(T, dtype=torch.bool)
    dropped[30] = True                                                      # a blank beyond the first EOS is never refreshed and does not count toward the level
    assert eng.pick_stale(s, torch.tensor([10]), 99, dropped).tolist() == [12]
    assert eng.stale_level(s, x, torch.tensor([3]), dropped) < eng.stale_level(s, x, torch.tensor([3]))
    age = _blk(stale_score="age")
    assert float(SelectiveEngine(model, T, age).stale_scores(x, 8).max()) == 0.0   # call 0: nobody is stale yet


def test_stale_refresh_adds_budget_rows_before_the_window_and_never_duplicates():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    T = int(ids.shape[1]) + G
    base = decode_selective(model, ids, _blk(window=8))
    for score in ("full", "age", "random"):
        r = decode_selective(model, ids, _blk(stale_budget=5, stale_score=score))
        assert r.written_total == G and max(r.stale_rows_n) <= 5 and max(r.stale_rows_n) > 0
        for i, q in enumerate(r.query_sets):
            assert len(set(q)) == len(q)
            if r.refresh_kind[i] in ("first", "dense"):
                assert r.stale_rows_n[i] == 0 and len(q) == T
                continue
            n_new = len(r.written[i - 1])
            assert set(r.written[i - 1]) <= set(q[: n_new + r.stale_rows_n[i]]) and r.rows_per_call[i] == len(q)
            window = q[n_new + r.stale_rows_n[i]:]
            assert window == sorted(window) and len(window) >= 1 and set(window) <= set(r.written[i]) | {p for p in range(T) if p not in set(sum(r.written[:i + 1], []))}
    refreshed = decode_selective(model, ids, _blk(stale_budget=5))
    assert sum(refreshed.rows_per_call) > sum(base.rows_per_call)


def test_random_stale_rows_are_deterministic_given_the_seed_and_differ_from_scored_rows():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    a = decode_selective(model, ids, _blk(stale_budget=4, stale_score="random", stale_seed=1))
    b = decode_selective(model, ids, _blk(stale_budget=4, stale_score="random", stale_seed=1))
    c = decode_selective(model, ids, _blk(stale_budget=4, stale_score="random", stale_seed=2))
    full = decode_selective(model, ids, _blk(stale_budget=4, stale_score="full"))
    assert a.query_sets == b.query_sets and a.tokens == b.tokens
    assert a.query_sets != c.query_sets and a.query_sets != full.query_sets


def test_lock_reads_the_staleness_level_and_is_stricter_when_stale():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    plain = decode_selective(model, ids, _blk(threshold=0.2, log_stale=True))
    lock = decode_selective(model, ids, _blk(threshold=0.2, lock_stale_beta=0.05))
    assert lock.nfe >= plain.nfe and lock.written_total == G
    for i, k in enumerate(lock.refresh_kind):
        if k in ("first", "dense"):
            assert lock.lock_shift[i] == 0.0 and lock.stale_level[i] == 0.0
        else:
            assert lock.lock_shift[i] == pytest.approx(0.05 * lock.stale_level[i]) and lock.stale_level[i] >= 0.0
    assert max(lock.stale_level) > 0.0 and max(lock.lock_shift) > 0.0
    assert len(lock.stale_scores) == lock.nfe and lock.stale_scores[0] == []


def test_newline_event_forces_dense_calls_and_validation():
    model, _ = _setup(layers=3)
    ids = _prompt_ids()
    base = decode_selective(model, ids, _blk())
    nl = decode_selective(model, ids, _blk(refresh_on_newline=True, newline_token_ids=tuple(range(4, V))))
    nd = lambda r: sum(1 for k in r.refresh_kind if k in ("first", "dense"))
    assert nl.newline_dense_calls > 0 and nd(nl) > nd(base) and nd(nl) == nl.nfe          # every call follows a write that contained a "newline" token
    none = decode_selective(model, ids, _blk(refresh_on_newline=True, newline_token_ids=()))
    assert none.tokens == base.tokens and none.newline_dense_calls == 0
    with pytest.raises(ValueError):
        decode_selective(model, ids, _blk(stale_budget=2, far_mask_cache=False))
    with pytest.raises(ValueError):
        decode_selective(model, ids, _blk(refresh="every_call", refresh_on_newline=True))


def test_stale_refresh_skips_blanks_dropped_by_eos_truncation():
    ids = _prompt_ids()
    for seed in range(40):                                     # find a toy model whose decode writes an EOS before the end
        model, _ = _setup(seed=seed, layers=3)
        r = decode_selective(model, ids, _blk(eos_truncation=True, stale_budget=40, stale_score="age", threshold=1.5))
        if 0 <= r.eos_position < P + G - 6:
            break
    else:
        pytest.skip("no toy decode wrote an early EOS")
    checked = 0
    for i, q in enumerate(r.query_sets):
        if r.refresh_kind[i] in ("first", "dense") or i == 0:
            continue
        written_by_now = set(sum(r.written[:i], []))
        late = [p for p in q if p > r.eos_position]
        assert all(p in written_by_now for p in late), (i, late)      # a computed row beyond the EOS is one that was written, never a dropped blank
        checked += 1
    assert checked > 0
