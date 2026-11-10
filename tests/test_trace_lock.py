"""TraceLock (paper_trace controller + projector) as a word lock inside the selective-cache engine: hidden-state exposure, parity with paper_decode's T-block, stale-feature store under the caches."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from regret_remasking.paper_decode import PAPER_ARMS, TRACE_ARMS, WINDOW_ARMS, PaperDecodeConfig, decode_paper_policy
from regret_remasking.paper_trace import HiddenProjector, PaperTraceController, ProjectorConfig
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, _lock_write, decode_selective
from regret_remasking.toy_llada import ToyLLaDA
from regret_remasking.trace_lock import TraceScorer, bracket, pick_threshold

MASK, EOS, V, P, G, D = 3, 2, 40, 8, 32, 16


class ToyHid(ToyLLaDA):
    """ToyLLaDA with LLaDA's hidden-state convention: hidden_states[i] = input of block i (i < L), hidden_states[L] = final-normalised output (33 entries for 32 layers)."""

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **_unused):
        tr = self.model.transformer
        x = tr.wte(input_ids)
        hs = []
        for block in tr.blocks:
            hs.append(x)
            x, _ = block(x)
        x = tr.ln_f(x)
        hs.append(x)
        return SimpleNamespace(logits=tr.ff_out(x), hidden_states=tuple(hs) if output_hidden_states else None)


class _Tok:
    padding_side = "left"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = torch.tensor([[5 + (ord(c) % 30) for c in text[:8]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x".join(map(str, ids[0]))]


class _FeatureController(torch.nn.Module):
    """A deterministic stand-in with the controller's call signature whose margins depend on the row's own features and on a prefix summary (a random PaperTraceController gives all blanks
    nearly the same score, so no call would ever write more than one blank)."""

    def forward(self, features, valid_mask, position_ids=None):
        own = features[..., :16].mean(-1) * 40.0
        ctx = features.mean(dim=(1, 2), keepdim=False)[:, None] * 5.0
        return {"margins": own + ctx - 0.5}


def _setup(seed=0, fake=False):
    torch.manual_seed(seed)
    model = ToyHid(vocab=V, d=D, n_heads=4, layers=32, ffn=24, seed=seed)
    torch.manual_seed(seed + 1)
    projector = HiddenProjector(ProjectorConfig(hidden_dim=D))
    controller = (_FeatureController() if fake else PaperTraceController()).eval()
    return model, projector, controller


def _ids():
    from regret_remasking.locking_policy import _prepare_prompt
    return _prepare_prompt(_Tok(), "abcdefgh", torch.device("cpu"))[0]


def _cfg(scorer, **kw):
    base = dict(gen_length=G, shape="block", block_length=8, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", attn_impl="sdpa", far_mask_cache=True,
                lock="trace", trace=scorer, threshold=0.5, log_detail=True)
    base.update(kw)
    return SelectiveConfig(**base)


def _paper(model, projector, controller, thr, **kw):
    cfg = PaperDecodeConfig(arm="T-block", gen_length=G, block_length=8, mask_id=MASK, threshold=thr, log_calls="writes", **kw)
    return decode_paper_policy(model, _Tok(), "abcdefgh", "q", cfg, controller=controller, projector=projector)


def test_t_block_is_a_block_geometry_trace_arm():
    assert "T-block" in PAPER_ARMS and "T-block" in TRACE_ARMS and "T-block" not in WINDOW_ARMS
    with pytest.raises(ValueError):
        decode_paper_policy(_setup()[0], _Tok(), "abcdefgh", "q", PaperDecodeConfig(arm="T-block", gen_length=G, block_length=8, mask_id=MASK))


def test_engine_exposes_the_hf_hidden_states_30_31_32_of_every_computed_row():
    model, projector, controller = _setup()
    scorer = TraceScorer(controller, projector)
    ids = _ids()
    T = P + G
    x = torch.full((1, T), MASK, dtype=torch.long)
    x[:, :P] = ids
    eng = SelectiveEngine(model, T, _cfg(scorer))
    rows = torch.arange(T)
    _, info = eng.forward(x, rows, 0, torch.arange(P, P + 8), torch.arange(P, P + 8), x[0] == MASK, fresh=True)
    ref = model(x, output_hidden_states=True).hidden_states
    assert len(ref) == 33
    for (r, h), idx in zip(info["hid"], (30, 31, 32)):
        assert torch.equal(r, rows) and torch.allclose(h, ref[idx], atol=1e-5, rtol=1e-5)
    # a cached call exposes only the computed rows
    q = torch.tensor([P + 1, P + 5, 2])
    _, info = eng.forward(x, q, 0, torch.arange(0, 2), torch.arange(0, 2), x[0] == MASK, fresh=False)
    assert all(torch.equal(r, q) and h.shape[1] == 3 for r, h in info["hid"])


@pytest.mark.parametrize("thr", [0.5, 0.52])
def test_engine_dense_with_trace_lock_equals_paper_decode_t_block(thr):
    model, projector, controller = _setup(fake=True)
    dense = _paper(model, projector, controller, thr)
    res = decode_selective(model, _ids(), _cfg(TraceScorer(controller, projector), threshold=thr))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe
    assert [[p - P for p in w] for w in res.written] == [e["w"] for e in dense.call_log]
    sizes = [len(w) for w in res.written]
    assert max(sizes) > 1 and res.fallback_calls > 0 and res.fallback_calls == dense.fallback_count       # multi-write and fallback calls both occur
    assert res.rule_time_s > 0 and dense.controller_calls == dense.nfe == len(res.ctrl_prefix)
    assert res.ctrl_proj_rows == res.ctrl_prefix                      # dense: every row below the prefix end is projected on every call


def test_trace_scores_equal_the_dense_scores_of_paper_decode_on_the_first_call():
    model, projector, controller = _setup()
    ids = _ids()
    T = P + G
    x = torch.full((1, T), MASK, dtype=torch.long)
    x[:, :P] = ids
    out = model(x, output_hidden_states=True)
    right = P + 8
    feats = projector.encode_hidden(tuple(h[:, :right] for h in (out.hidden_states[30], out.hidden_states[31], out.hidden_states[32])))
    want = torch.sigmoid(controller(feats, valid_mask=torch.ones(feats.shape[:2], dtype=torch.bool), position_ids=torch.arange(right).unsqueeze(0))["margins"])[0, P:right]
    scorer = TraceScorer(controller, projector)
    scorer.reset(T, "cpu")
    eng = SelectiveEngine(model, T, _cfg(scorer))
    rows = torch.arange(T)
    _, info = eng.forward(x, rows, 0, torch.arange(P, right), torch.arange(P, right), x[0] == MASK, fresh=True)
    got = scorer.score(info["hid"], right, torch.arange(P, right))
    assert torch.allclose(got, want.detach(), atol=1e-6, rtol=1e-5)


def test_real_controller_dense_parity_with_paper_decode():
    model, projector, controller = _setup()
    dense = _paper(model, projector, controller, 0.5)
    res = decode_selective(model, _ids(), _cfg(TraceScorer(controller, projector), threshold=0.5))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe
    assert [[p - P for p in w] for w in res.written] == [e["w"] for e in dense.call_log]


def test_cached_calls_keep_stale_features_for_uncomputed_rows_and_project_only_computed_rows():
    model, projector, controller = _setup(fake=True)
    scorer = TraceScorer(controller, projector)
    snaps, seen = [], {"n": 0}
    orig = scorer.observe

    def spy(hid, right):
        n = orig(hid, right)
        snaps.append((n, right, scorer.store[:right].clone()))
        return n

    scorer.observe = spy
    res = decode_selective(model, _ids(), _cfg(scorer, refresh="block", threshold=0.9))
    assert len(snaps) == res.nfe == len(res.ctrl_prefix)
    kinds = res.refresh_kind
    assert kinds[0] == "first" and "none" in kinds                       # at least one cached (non-dense) call
    for i, (n, right, store) in enumerate(snaps):
        if kinds[i] in ("first", "dense"):
            assert n == right                                            # a dense call projects every row of the prefix
        else:
            assert n < right
            prev_right, prev = snaps[i - 1][1], snaps[i - 1][2]
            computed = set(res.query_sets[i])
            for r in range(min(right, prev_right)):
                if r not in computed:
                    assert torch.equal(store[r], prev[r])                # not recomputed: the stored feature is unchanged
            assert all(0 <= r < right for r in computed if r < right)
            assert n == sum(1 for r in computed if r < right)
    # rows per layer are the plain S32 rows (the trace option only exposes states, it does not change the computed rows)
    plain = decode_selective(model, _ids(), SelectiveConfig(**{**_cfg(scorer, refresh="block", threshold=0.9).__dict__, "lock": "conf", "trace": None}))
    assert len(plain.layer_rows[0]) == 32 and plain.layer_rows[0] == res.layer_rows[0]


def test_trace_lock_rejects_unsupported_settings():
    model, projector, controller = _setup()
    scorer = TraceScorer(controller, projector)
    for kw in (dict(shape="window", window=8), dict(refresh="trigger"), dict(refresh="selective"), dict(refresh="none"), dict(trace=None)):
        with pytest.raises(ValueError):
            decode_selective(model, _ids(), _cfg(scorer, **kw))


def test_partial_refresh_and_k4_run_with_trace_lock():
    model, projector, controller = _setup(fake=True)
    for kw in (dict(refresh="block", partial_every_k=2, partial_layer=11, partial_join="blank"), dict(refresh="block", refresh_every_k=3)):
        res = decode_selective(model, _ids(), _cfg(TraceScorer(controller, projector), threshold=0.9, **kw))
        assert res.written_total == G and len(res.ctrl_prefix) == res.nfe


def test_lock_write_trace_is_strictly_above_the_cutoff_with_best_blank_fallback():
    cfg = SelectiveConfig(gen_length=4, lock="trace", threshold=0.5, mask_id=MASK, eos_id=EOS)
    res = SimpleNamespace(nfe=0, written_total=0, collected=[], rule_time_s=0.0, fallback_calls=0, extra_state=None)
    logits = torch.zeros(1, 3, V)
    logits[0, :, 7] = 5.0
    x = torch.full((1, 6), MASK)
    eligible = torch.tensor([3, 4, 5])
    sel, tok, _ = _lock_write(logits, eligible, x, cfg, 3, 3, res, torch.tensor([0.5, 0.7, 0.51]))
    assert sel.tolist() == [4, 5] and res.fallback_calls == 0               # 0.5 is not > 0.5
    x2 = torch.full((1, 6), MASK)
    sel, _, _ = _lock_write(logits, eligible, x2, cfg, 3, 3, res, torch.tensor([0.5, 0.4, 0.3]))
    assert sel.tolist() == [3] and res.fallback_calls == 1                  # nothing above the cutoff: the best-scoring blank


def test_pick_threshold_nearest_on_grid_and_flags_unmatched():
    calls = {0.8: 50.0, 0.9: 60.0, 0.95: 74.0, 0.97: 90.0}
    assert pick_threshold(74.5, calls)[0] == 0.95 and pick_threshold(74.5, calls)[1]["status"] == "bracketed"
    t, info = pick_threshold(30.0, calls)
    assert t == 0.8 and info["status"] == "unmatched_low"
    assert pick_threshold(120.0, calls)[1]["status"] == "unmatched_high"
    assert bracket(74.5, calls) == (0.95, 0.97) and bracket(10.0, calls) is None
