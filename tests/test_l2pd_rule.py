"""Learn2PD word lock (src/regret_remasking/l2pd_rule.py): filter arithmetic, the official acceptance semantics (sigmoid > tau, strict; the most confident blank is
always written), and the two decoders it plugs into (dense X-block path of paper_decode; selective-cache engine with block_rows). CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regret_remasking.context_rules import ContextRule
from regret_remasking.l2pd_rule import Learn2PDRule
from regret_remasking.paper_decode import PaperDecodeConfig, _validate, decode_paper_policy
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.toy_llada import ToyLLaDA
from test_frontier_arms import peaked
from test_paper_token_kl_commitment import _decode

MASK = 0


def threshold_state(n: int, c: float, k: float = 200.0) -> dict[str, torch.Tensor]:
    """A filter that outputs sigmoid(k * (confidence_i - c)) for slot i (identity first layer, scaled identity second)."""
    eye = torch.eye(n)
    return {"linear1.weight": eye, "linear1.bias": torch.zeros(n), "linear2.weight": k * eye, "linear2.bias": torch.full((n,), -k * c)}


def shift_state(n: int, c: float, k: float = 200.0) -> dict[str, torch.Tensor]:
    """Slot i is accepted iff the confidence of slot i-1 exceeds c (slot 0 never): a filter that reads OTHER slots, so it can reject the most confident blank."""
    shift = torch.zeros(n, n)
    for i in range(1, n):
        shift[i, i - 1] = 1.0
    return {"linear1.weight": torch.eye(n), "linear1.bias": torch.zeros(n), "linear2.weight": k * shift, "linear2.bias": torch.full((n,), -k * c)}


def random_state(n: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"linear1.weight": torch.randn(n, n, generator=g), "linear1.bias": torch.randn(n, generator=g) * 0.2, "linear2.weight": torch.randn(n, n, generator=g),
            "linear2.bias": torch.randn(n, generator=g) * 0.5}


# ------------------------------------------------------------------------------------------------ the rule on its own
def test_published_shape_has_2112_parameters_and_validates_state():
    rule = Learn2PDRule(random_state(32, 0), 0.96)
    assert rule.n_parameters == 2112 and rule.kind == "l2pd" and rule.needs_block_vector
    with pytest.raises(ValueError):
        Learn2PDRule({"linear1.weight": torch.zeros(3, 3)}, 0.5)
    with pytest.raises(ValueError):
        Learn2PDRule(random_state(32, 0), 1.5)


def test_filter_equals_manual_two_layer_mlp_and_file_roundtrip(tmp_path):
    st = random_state(32, 1)
    rule = Learn2PDRule(st, 0.5)
    x = torch.rand(32)
    want = torch.sigmoid(torch.relu(x @ st["linear1.weight"].T + st["linear1.bias"]) @ st["linear2.weight"].T + st["linear2.bias"])
    assert torch.allclose(rule.probabilities(x), want, atol=1e-6)
    torch.save(st, tmp_path / "f.pth")
    again = Learn2PDRule.from_file(tmp_path / "f.pth", 0.5)
    assert again.sha256() == rule.sha256() and torch.equal(again.probabilities(x), rule.probabilities(x))
    with pytest.raises(ValueError):
        rule.probabilities(torch.rand(31))


def test_tau_is_strict_and_applied_to_the_sigmoid():
    n = 4
    conf = torch.tensor([0.99, 0.9, 0.5, 0.2])
    blanks = torch.arange(4)
    st = threshold_state(n, 0.7)
    p = Learn2PDRule(st, 0.0, block_length=n).probabilities(conf)
    # strict >: tau equal to a probability does not accept it; tau just below does
    r_eq = Learn2PDRule(st, float(p[1]), always_top1=False, block_length=n)
    assert r_eq.accept_mask(conf, blanks).tolist() == [True, False, False, False]          # slot 1 not accepted at tau == its probability; the forced slot is slot 0
    r_lo = Learn2PDRule(st, float(p[1]) - 1e-6, always_top1=False, block_length=n)
    assert r_lo.accept_mask(conf, blanks).tolist() == [True, True, False, False]
    # tau is a probability threshold, not a logit threshold: tau 0.5 on this filter is confidence > 0.7
    assert Learn2PDRule(st, 0.5, always_top1=False, block_length=n).accept_mask(conf, blanks).tolist() == [True, True, False, False]


def test_only_blank_slots_are_candidates_and_top1_is_forced():
    n = 4
    conf = torch.tensor([0.99, 0.95, 0.4, 0.3])            # slots 0 and 1 already written (high confidence), blanks are slots 2, 3
    rule = Learn2PDRule(threshold_state(n, 0.8), 0.5, block_length=n)
    keep = rule.accept_mask(conf, torch.tensor([2, 3]))
    assert keep.tolist() == [True, False]                  # nothing clears the filter; the most confident blank is written anyway
    assert rule.stats == {"calls": 1, "words": 1, "filter_words": 0, "empty": 1, "top1_added": 1} and rule.last_empty
    # all blanks above c: all written, nothing forced
    keep = rule.accept_mask(torch.tensor([0.1, 0.1, 0.9, 0.85]), torch.tensor([2, 3]))
    assert keep.tolist() == [True, True] and rule.stats["filter_words"] == 2 and not rule.last_empty
    # a filter that rejects the most confident blank: the forced top-1 is added to the filter's set
    shifted = Learn2PDRule(shift_state(n, 0.8), 0.5, block_length=n)
    conf = torch.tensor([0.99, 0.2, 0.3, 0.1])             # slot 1 accepted (reads slot 0); the most confident blank is slot 2
    keep = shifted.accept_mask(conf, torch.tensor([1, 2, 3]))
    assert keep.tolist() == [True, True, False] and shifted.stats["top1_added"] == 1 and shifted.stats["filter_words"] == 1
    shifted.reset_stats()
    assert shifted.stats["calls"] == 0


def test_without_forced_top1_an_empty_filter_still_writes_one_blank():
    rule = Learn2PDRule(threshold_state(4, 0.8), 0.5, always_top1=False, block_length=4)
    assert rule.accept_mask(torch.tensor([0.1, 0.2, 0.4, 0.3]), torch.arange(4)).tolist() == [False, False, True, False]


# ------------------------------------------------------------------------------------------------ dense X-block path
def _frames():
    return [{1: peaked(1, 0.99), 2: peaked(2, 0.9), 3: peaked(3, 0.85), 4: peaked(4, 0.3), 5: peaked(1, 0.999), 6: peaked(2, 0.6), 7: peaked(3, 0.6), 8: peaked(4, 0.6)}]


def _writes(result):
    out: dict[int, list[int]] = {}
    for event in result.events:
        if event["type"] == "token_commit":
            out.setdefault(event["global_step"], []).append(event["position"])
    return out


BASE = dict(gen_length=8, block_length=4, mask_id=MASK)


def test_validation_requires_dense_path():
    rule = Learn2PDRule(threshold_state(4, 0.8), 0.5, block_length=4)
    _validate(PaperDecodeConfig(arm="X-block", block_rule=rule, **BASE))
    with pytest.raises(ValueError):
        _validate(PaperDecodeConfig(arm="X-block", block_rule=rule, freeze_rule="fix_both", freeze_scope="all_unmasked", **BASE))


def test_rejecting_filter_writes_one_most_confident_blank_per_call_like_conf_rule_at_one():
    rule = Learn2PDRule(threshold_state(4, 0.8), 1.0, block_length=4)          # strict: sigmoid never exceeds 1.0
    mine, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=rule, **BASE))
    ref, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=ContextRule("conf", 1.0), **BASE))
    assert _writes(mine) == _writes(ref) and mine.nfe == ref.nfe == 8 and list(mine.token_ids) == list(ref.token_ids)
    assert mine.fallback_count == ref.fallback_count == 8 and rule.stats["empty"] == 8


@pytest.mark.parametrize("c", [0.95, 0.8, 0.5])
def test_threshold_filter_reproduces_c_block(c):
    rule = Learn2PDRule(threshold_state(4, c), 0.5, block_length=4)
    mine, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=rule, **BASE))
    ref, _ = _decode(_frames(), PaperDecodeConfig(arm="C-block", threshold=c, **BASE))
    assert _writes(mine) == _writes(ref) and mine.nfe == ref.nfe and list(mine.token_ids) == list(ref.token_ids)


def test_filter_input_is_the_whole_block_in_order_written_slots_included():
    seen = []

    class Probe(Learn2PDRule):
        def accept_mask(self, slot_conf, blank_slots):
            seen.append((slot_conf.tolist(), blank_slots.tolist()))
            return super().accept_mask(slot_conf, blank_slots)

    rule = Probe(threshold_state(4, 0.999), 0.5, block_length=4)
    _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=rule, **BASE))
    first, second = seen[0], seen[1]
    assert first[1] == [0, 1, 2, 3] and first[0] == pytest.approx([0.99, 0.9, 0.85, 0.3], abs=1e-5)
    assert second[1] == [1, 2, 3] and len(second[0]) == 4 and second[0][0] == pytest.approx(0.99, abs=1e-5)       # slot 0 is written but still in the vector
    # block 2: a different block, slots reindexed from 0
    assert any(blank == [0, 1, 2, 3] and conf[0] == pytest.approx(0.999, abs=1e-4) for conf, blank in seen[4:])


def test_default_off_other_rules_see_the_same_observed_rows():
    # a plain ContextRule X-block decode is unchanged by the Learn2PD code path (no block vector requested)
    a, _ = _decode(_frames(), PaperDecodeConfig(arm="X-block", block_rule=ContextRule("conf", 0.8), **BASE))
    b, _ = _decode(_frames(), PaperDecodeConfig(arm="C-block", threshold=0.8, **BASE))
    assert list(a.token_ids) == list(b.token_ids) and a.nfe == b.nfe


# ------------------------------------------------------------------------------------------------ selective-cache engine
V, G, EOS, MASKT = 40, 24, 2, 3


class _Tok:
    padding_side = "left"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = torch.tensor([[5 + (ord(c) % 30) for c in text[:8]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x".join(map(str, ids[0]))]


def _model(seed=0):
    return ToyLLaDA(vocab=V, d=32, n_heads=4, layers=3, ffn=48, seed=seed)


def _ids():
    from regret_remasking.locking_policy import _prepare_prompt
    return _prepare_prompt(_Tok(), "abcdefgh", torch.device("cpu"))[0]


def _dense(model, rule):
    cfg = PaperDecodeConfig(gen_length=G, block_length=8, mask_id=MASKT, arm="X-block", block_rule=rule)
    return decode_paper_policy(model, _Tok(), "abcdefgh", "q", cfg)


def _sel(**kw):
    base = dict(gen_length=G, shape="block", block_length=8, mask_id=MASKT, eos_id=EOS, eos_truncation=False, refresh="every_call", attn_impl="sdpa", far_mask_cache=True,
                lock="rule", block_rows=True, window=8)
    base.update(kw)
    return SelectiveConfig(**base)


@pytest.mark.parametrize("seed,tau", [(0, 0.5), (1, 0.3), (2, 0.7)])
def test_s32_every_call_with_block_rows_equals_dense_l2pd(seed, tau):
    model = _model()
    rule = Learn2PDRule(random_state(8, seed), tau, block_length=8)
    dense = _dense(model, rule)
    d_stats = dict(rule.stats)
    rule.reset_stats()
    res = decode_selective(model, _ids(), _sel(rule=rule))
    assert res.tokens == dense.token_ids and res.nfe == dense.nfe
    assert rule.stats == d_stats and res.written_total == G and res.rule_time_s > 0
    assert res.fallback_calls == rule.stats["empty"] == dense.fallback_count
    assert 1 <= rule.stats["words"] / res.nfe <= 8


def test_block_rows_cached_calls_compute_the_whole_block_plus_rows_written_outside_it():
    model = _model()
    ids = _ids()
    T = int(ids.shape[1]) + G
    rule = Learn2PDRule(random_state(8, 0), 0.6, block_length=8)
    res = decode_selective(model, ids, _sel(rule=rule, refresh="block", log_detail=True))
    dense = [i for i, k in enumerate(res.refresh_kind) if k in ("first", "dense")]
    assert len(dense) == 3 and all(res.layer_rows[i] == [T] * 3 for i in dense)
    for i, q in enumerate(res.query_sets):
        if i in dense:
            continue
        assert len(q) == 8 and sorted(q) == list(range(min(q), min(q) + 8))                  # exactly the 8 slots of the current block, written or not


def test_block_rule_without_block_rows_is_refused_and_block_rows_needs_block_shape():
    model, ids = _model(), _ids()
    rule = Learn2PDRule(random_state(8, 0), 0.5, block_length=8)
    with pytest.raises(ValueError):
        decode_selective(model, ids, _sel(rule=rule, block_rows=False))
    with pytest.raises(ValueError):
        decode_selective(model, ids, _sel(rule=rule, shape="window"))


def test_block_rows_off_keeps_existing_context_rule_path_identical():
    model, ids = _model(), _ids()
    rule = ContextRule("conf", 0.05)
    a = decode_selective(model, ids, _sel(rule=rule, block_rows=False))
    b = decode_selective(model, ids, _sel(rule=rule, block_rows=True))     # block_rows is ignored for rules that do not need the vector
    assert a.tokens == b.tokens and a.nfe == b.nfe and a.layer_rows == b.layer_rows


def test_eos_truncation_with_block_rows_drops_blanks_beyond_eos():
    model, ids = _model(), _ids()
    rule = Learn2PDRule(random_state(8, 3), 0.5, block_length=8)
    res = decode_selective(model, ids, _sel(rule=rule, eos_truncation=True, refresh="block"))
    assert res.tokens[res.eos_position - int(ids.shape[1]):][0] in (EOS,) if res.eos_position >= 0 else True
    assert all(t == EOS for t in res.tokens[(res.eos_position - int(ids.shape[1])) if res.eos_position >= 0 else len(res.tokens):])
