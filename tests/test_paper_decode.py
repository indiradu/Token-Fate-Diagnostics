from types import SimpleNamespace

import pytest
import torch

from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy, _last_hidden, _stability_logp


class Tokenizer:
    padding_side = "left"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    def batch_decode(self, rows, **kwargs):
        return [" ".join(map(str, rows[0]))]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.inputs = []

    def forward(self, ids, attention_mask=None, output_hidden_states=False):
        self.calls += 1
        self.inputs.append(ids.clone())
        b, n = ids.shape
        logits = torch.full((b, n, 32), -4.0)
        for pos in range(n):
            token = pos + 3 if int(ids[0, pos]) == 31 else int(ids[0, pos])
            logits[0, pos, token] = 5.0 + pos / 100
        hidden = tuple(torch.full((b, n, 4), float(i)) for i in range(33))
        return SimpleNamespace(logits=logits, hidden_states=hidden if output_hidden_states else None)


class Projector(torch.nn.Module):
    def encode_hidden(self, hidden):
        assert len(hidden) == 3
        b, n, _ = hidden[0].shape
        return torch.zeros(b, n, 832)


class Controller(torch.nn.Module):
    def __init__(self, high=True, alternate=False):
        super().__init__()
        self.high, self.alternate = high, alternate
        self.lengths = []

    def forward(self, features, valid_mask, position_ids=None):
        self.lengths.append(features.shape[1])
        logits = torch.full(features.shape[:2], 10.0 if self.high else -10.0)
        if self.alternate:
            logits[:, 3::2] = -10.0
            if features.shape[1] == 6:
                logits[:, 5] = 10.0
        return {"token_logits": logits, "threshold_logit": torch.zeros(features.shape[0]), "margins": logits}


def cfg(arm, **kwargs):
    return PaperDecodeConfig(arm=arm, gen_length=6, block_length=3, window_length=3, mask_id=31, **kwargs)


@pytest.mark.parametrize("arm", ["B0", "B-entropy", "W-confidence", "W-entropy"])
def test_one_token_controls_recompute_every_row(arm):
    model = Model()
    result = decode_paper_policy(model, Tokenizer(), "p", "e", cfg(arm), collect_events=True)
    assert result.nfe == result.forward_calls == model.calls == 6
    assert result.final_mask_count == result.immutable_token_violations == result.frozen_count == 0
    assert result.total_query_rows == result.total_output_head_rows == 8 * 6
    assert all(torch.equal(ids[:, :2], torch.tensor([[1, 2]])) for ids in model.inputs)
    assert len({e["position"] for e in result.events}) == 6


def test_adaptive_trace_accepts_multiple_and_keeps_representations_live():
    model = Model()
    controller = Controller()
    result = decode_paper_policy(model, Tokenizer(), "p", "e", cfg("T-adaptive"), controller=controller, projector=Projector())
    assert result.nfe == 2 and result.adaptive_accept_count == 6
    assert result.total_query_rows == result.total_output_head_rows == 16
    assert result.frozen_count == 0 and result.controller_calls == 2
    assert controller.lengths == [5, 8]


def test_trace_fixed_quota_and_adaptive_fallback_finish_at_baseline_budget():
    for arm, high in [("T-fixed", True), ("T-adaptive", False)]:
        result = decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg(arm), controller=Controller(high), projector=Projector())
        assert result.nfe == 6
        assert result.fallback_count == (6 if arm == "T-adaptive" else 0)


def test_moving_window_crosses_fixed_block_boundary():
    result = decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("T-adaptive"), controller=Controller(alternate=True), projector=Projector(), collect_events=True)
    first = [e["position"] for e in result.events if e["global_step"] == 0]
    second = [e["position"] for e in result.events if e["global_step"] == 1]
    assert first == [2, 4]
    # First unresolved is3, next window[3,6) reaches position5 acrossoldblockedge.
    assert second == [5]
    assert result.nfe <= 6


def test_confidence_threshold_acceptance_and_fallback():
    fast = decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("C-adaptive", threshold=0.5))
    slow = decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("C-adaptive", threshold=1.0))
    assert fast.nfe == 2 and slow.nfe == 6 and slow.fallback_count == 6


def test_observed_input_snapshot_does_not_change_when_future_tokens_commit():
    frames = []
    result = decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("B0"), step_observer=frames.append)
    assert result.nfe == len(frames) == 6
    assert torch.all(frames[0]["input_ids"][0, 2:] == 31)
    assert torch.sum(frames[-1]["input_ids"][0, 2:] == 31) == 1
    assert frames[0]["last_hidden"][0][0, 0, 0] == 30
    assert frames[0]["last_hidden"][2][0, 0, 0] == 32
    assert frames[0]["proposal_positions"].tolist() == list(range(8))


def test_hidden_semantics_and_feature_contract_fail_closed():
    with pytest.raises(ValueError, match="length 33"):
        _last_hidden(SimpleNamespace(hidden_states=[torch.zeros(1, 2, 4)] * 32))
    class WrongProjector(Projector):
        def encode_hidden(self, hidden):
            return super().encode_hidden(hidden)[..., :831]
    with pytest.raises(ValueError, match="832"):
        decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("T-adaptive"), controller=Controller(), projector=WrongProjector())
    with pytest.raises(ValueError, match="trained controller"):
        decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("T-adaptive"))


class ControlledRuntime:
    """Expose controlled hidden/logit histories to test gate causality."""
    def __init__(self, model, mode, total_len, capture_diagnostics=False):
        self.model, self.mode, self.total_len = model, mode, total_len
        self.locked = set()
        self.blocks = [object()]

    def install(self):
        self.model.runtime = self

    def remove(self):
        self.model.runtime = None

    def begin_forward(self):
        self.output_positions = torch.tensor([p for p in range(self.total_len) if p not in self.locked])

    def commit(self, positions):
        self.locked.update(positions)


class ControlledHistories(torch.nn.Module):
    def __init__(self, change_hidden=False, change_vocab=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.change_hidden, self.change_vocab = change_hidden, change_vocab
        self.calls = 0
        self.runtime = None

    def forward(self, ids, attention_mask=None, output_hidden_states=False):
        assert not output_hidden_states  # Final pre-head capture needs no hidden history.
        self.calls += 1
        positions = self.runtime.output_positions
        hidden = torch.zeros(1, positions.numel(), 4)
        hidden[..., 0] = (4 if self.calls % 2 else -4) if self.change_hidden else 1
        logits = torch.zeros(1, positions.numel(), 32)
        logits[..., 7] = (2 if self.calls % 2 else 8) if self.change_vocab else 8
        self.runtime.last_output_hidden = hidden
        self.runtime.last_output_positions = positions.clone()
        rows = self.runtime.total_len if self.runtime.mode == "dense" else positions.numel()
        self.runtime.last_work_counts = {name: rows for name in ("attention_query_rows", "projection_rows", "ffn_rows", "head_rows")}
        return SimpleNamespace(logits=logits)


def controlled_decode(*, change_hidden=False, change_vocab=False, kl_space="representation", events=True):
    model = ControlledHistories(change_hidden, change_vocab)
    result = decode_paper_policy(model, Tokenizer(), "p", "controlled", cfg(
        "S-sparse", kl_space=kl_space, epsilon=0.005, runtime_cls=ControlledRuntime,
    ), collect_events=events)
    assert result.token_ids == [7] * 6
    assert result.final_mask_count == result.immutable_token_violations == 0
    return result


def test_representation_gate_ignores_vocabulary_distribution_changes():
    stable = controlled_decode()
    changed_vocab = controlled_decode(change_vocab=True)
    assert stable.events == changed_vocab.events
    assert stable.frozen_count == changed_vocab.frozen_count == 5
    token_gate = controlled_decode(change_vocab=True, kl_space="token")
    assert token_gate.frozen_count == 0


def test_representation_gate_detects_hidden_changes_with_fixed_vocabulary():
    rep_gate = controlled_decode(change_hidden=True)
    token_gate = controlled_decode(change_hidden=True, kl_space="token")
    assert rep_gate.frozen_count == 0
    assert token_gate.frozen_count == 5
    assert rep_gate.representation_layer == "final_normalized_pre_vocab"
    assert rep_gate.representation_feature_dim == 4


def test_representation_gate_alignment_eligibility_and_diagnostic_parity():
    timed = controlled_decode(events=False)
    diagnostic = controlled_decode()
    assert timed.token_ids == diagnostic.token_ids
    assert timed.nfe == diagnostic.nfe == 6
    assert timed.frozen_count == diagnostic.frozen_count == 5
    assert timed.total_query_rows == diagnostic.total_query_rows < 6 * 8
    freezes = [event for event in diagnostic.events if event["type"] == "reference_freeze"]
    assert all(not event["input_was_masked"] and event["kl_space"] == "representation" for event in freezes)
    assert [event["position"] for event in freezes] == list(range(2, 7))
    assert all(event["effective_from_step"] == event["global_step"] + 1 for event in freezes)


def test_representation_softmax_kl_has_declared_direction_and_fp32_reduction():
    config = cfg("S-sparse", kl_space="representation")
    vocab = torch.zeros(1, 1, 32)
    old = _stability_logp(vocab, torch.zeros(1, 1, 2, dtype=torch.bfloat16), config)
    current = _stability_logp(vocab, torch.tensor([[[torch.log(torch.tensor(3.0)), 0.0]]]), config)
    divergence = (current.exp() * (current - old)).sum(-1)
    assert current.dtype == old.dtype == torch.float32
    expected = 0.75 * torch.log(torch.tensor(1.5)) + 0.25 * torch.log(torch.tensor(0.5))
    torch.testing.assert_close(divergence.squeeze(), expected)
    hotter = _stability_logp(vocab, torch.tensor([[[1.0, -1.0]]]), cfg("S-sparse", kl_space="representation", representation_temperature=2.0))
    torch.testing.assert_close(hotter, torch.log_softmax(torch.tensor([[[0.5, -0.5]]]), -1))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_representation_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="representation_temperature"):
        decode_paper_policy(Model(), Tokenizer(), "p", "e", cfg("B0", representation_temperature=temperature))


def test_representation_gate_rejects_missing_or_nonfinite_capture():
    config = cfg("S-sparse", kl_space="representation")
    with pytest.raises(ValueError, match="aligned pre-vocabulary"):
        _stability_logp(torch.zeros(1, 1, 32), None, config)
    with pytest.raises(ValueError, match="nonfinite representation"):
        _stability_logp(torch.zeros(1, 1, 32), torch.full((1, 1, 4), float("nan")), config)
