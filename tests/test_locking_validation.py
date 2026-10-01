from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from regret_remasking.locking_reference import ReferenceRuntime
from regret_remasking.locking_validation import validate_backends, validate_perturbation_control


class ToyTokenizer:
    mask_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, prompts: list[str], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        rows = []
        max_len = 0
        for prompt in prompts:
            row = [10 + (ord(char) % 20) for char in prompt[:4]]
            rows.append(row)
            max_len = max(max_len, len(row))
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ToyConfig:
    n_heads = 1
    effective_n_kv_heads = 1


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        self.config = ToyConfig()
        self.attn_norm = torch.nn.Identity()
        self.ff_norm = torch.nn.Identity()
        self.dropout = torch.nn.Identity()
        self.act = torch.nn.Tanh()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.attn_out = torch.nn.Linear(dim, dim, bias=False)
        self.ff_proj = torch.nn.Linear(dim, dim, bias=False)
        self.ff_out = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = None
        self.k_norm = None

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            scores = scores + attn_mask
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False):
        if layer_past is not None or use_cache:
            raise RuntimeError("toy cache is unsupported")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        bsz, seq_len, dim = q.shape
        q = q.view(bsz, seq_len, 1, dim).transpose(1, 2)
        k = k.view(bsz, seq_len, 1, dim).transpose(1, 2)
        v = v.view(bsz, seq_len, 1, dim).transpose(1, 2)
        att = self._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        att = att.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = x + self.attn_out(att)
        return x + self.ff_out(self.act(self.ff_proj(x))), None


class ToyModel(torch.nn.Module):
    def __init__(self, *, vocab: int = 128, dim: int = 6, layers: int = 2) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList([ToyBlock(dim) for _ in range(layers)])
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x, _cache = block(x)
        return SimpleNamespace(logits=self.lm_head(x))


def _model(seed: int = 19) -> ToyModel:
    torch.manual_seed(seed)
    model = ToyModel()
    model.eval()
    return model


def test_validate_backends_reports_stable_valid_controls() -> None:
    result = validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5)

    assert result["passed"] is True
    assert result["completed"] is True
    assert result["backend_equivalent"] is True
    assert result["requires_review"] is False
    assert result["within_measured_envelope"] is True
    assert result["observations_after_commit"] == 2
    assert result["no_output_token_replay"] is True
    assert result["tolerances"]["formula"] == "max(minimum, max(no_lock_error, fresh_reference_control_error) * multiplier)"
    assert result["strict_tolerances"]["logit_abs"] == 1e-5
    assert result["fresh_reference_control"]["active_logits"]["argmax_changes"] == 0
    assert result["no_lock"]["dense_vs_sparse"]["argmax_changes"] == 0
    assert result["freeze_trial"]["active_logits"]["argmax_changes"] == 0
    assert result["cleanup"]["dense_removed"] is True
    assert result["cleanup"]["sparse_removed"] is True
    assert result["cleanup"]["post_remove_matches_baseline"]["max_abs"] == 0.0


def test_validate_backends_supports_multiple_frozen_generated_positions() -> None:
    result = validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=6, lock_offset=1, lock_count=3)

    assert result["passed"] is True
    assert result["lock_count"] == 3
    assert result["lock_positions"] == [5, 6, 7]
    assert result["active_generated_count"] == 3
    assert result["fresh_reference_control"]["active_generated_logits"]["argmax_changes"] == 0
    assert result["freeze_trial"]["snapshot_constancy"]["dense"]["max_abs"] == 0.0
    assert result["freeze_trial"]["snapshot_constancy"]["sparse"]["max_abs"] == 0.0


def test_noisy_fresh_control_inside_broad_envelope_requires_review(monkeypatch) -> None:
    original = ReferenceRuntime._sparse_forward
    calls = {"removed": False}

    def bad_sparse(self, *args, **kwargs):
        out, cache = original(self, *args, **kwargs)
        locked_calls = getattr(self, "_test_locked_sparse_calls", 0)
        if bool(self.locked.any()):
            locked_calls += 1
            setattr(self, "_test_locked_sparse_calls", locked_calls)
        if locked_calls >= 1:
            out = out.clone()
            out[:, 1:, :] = out[:, 1:, :] + 0.001
        return out, cache

    original_remove = ReferenceRuntime.remove

    def tracking_remove(self):
        original_remove(self)
        calls["removed"] = True

    monkeypatch.setattr(ReferenceRuntime, "_sparse_forward", bad_sparse)
    monkeypatch.setattr(ReferenceRuntime, "remove", tracking_remove)

    result = validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5)

    assert result["passed"] is False
    assert result["backend_equivalent"] is False
    assert result["requires_review"] is True
    assert result["within_measured_envelope"] is True
    assert result["fresh_reference_control"]["active_logits"]["max_abs"] > result["strict_tolerances"]["logit_abs"]
    assert calls["removed"] is True
    assert result["cleanup"]["post_remove_matches_baseline"]["max_abs"] == 0.0


def test_perturbation_control_can_change_downstream_active_logits() -> None:
    result = validate_perturbation_control(_model(), ToyTokenizer(), "abcd", gen_length=6, lock_count=2)

    assert result["changed"] is True
    assert result["active_logits"]["max_abs"] > 0.0


def test_validate_backends_rejects_invalid_generated_window() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=2)
    with pytest.raises(ValueError, match="active generated"):
        validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5, lock_count=4)
    with pytest.raises(ValueError, match="fit within"):
        validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5, lock_offset=4, lock_count=2)
    with pytest.raises(ValueError, match="positive"):
        validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5, lock_count=0)
    with pytest.raises(ValueError, match="finite and positive"):
        validate_backends(_model(), ToyTokenizer(), "abcd", gen_length=5, strict_logit_abs_tolerance=0.0)
