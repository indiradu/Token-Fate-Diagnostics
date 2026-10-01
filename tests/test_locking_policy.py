from __future__ import annotations

import pathlib
import sys
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking import FEATURE_NAMES
from regret_remasking.llada_trace import MASK_ID, DecodeConfig, decode_batch
import regret_remasking.locking_policy as locking_policy
from regret_remasking.locking_policy import LockingPolicyConfig, decode_locking_policy


def test_baseline_ranks_permitted_proposals_not_excluded_mask_probability():
    class MaskDominatedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask=None):
            logits = torch.full((*input_ids.shape, 5), -10.0)
            logits[:, :2, 1] = 5.0
            logits[0, 2, 4], logits[0, 2, 0] = 10.0, 5.0
            logits[0, 3, 1], logits[0, 3, 2] = 4.0, 3.0
            return SimpleNamespace(logits=logits)

    result = decode_locking_policy(MaskDominatedModel(), ToyTokenizer(), "prompt", "example",
        LockingPolicyConfig(gen_length=2, block_length=2, mask_id=4), collect_events=True)
    assert result.events[0]["position"] == 3
    assert result.token_ids == [0, 1]


class ToyTokenizer:
    padding_side = "left"

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, tokenize: bool = False):
        del add_generation_prompt, tokenize
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens: bool = False, padding: bool = False, return_tensors: str = "pt"):
        del add_special_tokens, padding, return_tensors
        batch = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.tensor([[1, 2]] * batch, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]] * batch, dtype=torch.long),
        }

    def batch_decode(self, rows, skip_special_tokens: bool = True):
        del skip_special_tokens
        return [" ".join(str(token) for token in rows[0])]


class ToyModel(torch.nn.Module):
    def __init__(self, schedules: list[list[int]], scores: list[list[float]] | None = None, vocab_size: int = 160):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.schedules = schedules
        self.scores = scores or [[8.0 - rel for rel in range(len(schedules[0]))] for _ in schedules]
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        step = min(self.calls, len(self.schedules) - 1)
        self.calls += 1
        batch, total_len = input_ids.shape
        logits = torch.full((batch, total_len, 160), -8.0, device=input_ids.device)
        logits[:, :, 0] = -2.0
        prompt_len = 2
        for rel, token in enumerate(self.schedules[step]):
            pos = prompt_len + rel
            if int(input_ids[0, pos]) != MASK_ID:
                token = int(input_ids[0, pos])
            logits[0, pos, int(token)] = float(self.scores[step][rel])
        logits[:, :prompt_len, 1] = 8.0
        return SimpleNamespace(logits=logits)


class HookableBlock(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x + 0.0, None


class HookableToyModel(ToyModel):
    def __init__(self, schedules: list[list[int]], scores: list[list[float]] | None = None):
        super().__init__(schedules=schedules, scores=scores)
        self.blocks = torch.nn.ModuleList([HookableBlock(), HookableBlock()])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 4), device=input_ids.device)
        for block in self.blocks:
            hidden, _cache = block(hidden)
        return super().forward(input_ids, attention_mask=attention_mask)


class RuntimeSpy:
    instances: list["RuntimeSpy"] = []

    def __init__(self, model, mode: str, total_len: int):
        del model
        self.mode = mode
        self.locked = torch.zeros((1, total_len), dtype=torch.bool)
        self.commits: list[list[int]] = []
        self.removed = False
        RuntimeSpy.instances.append(self)

    def install(self):
        return None

    def remove(self):
        self.removed = True

    def commit(self, positions: list[int]):
        self.commits.append(list(positions))
        self.locked[0, positions] = True


class FullAttentionRuntimeSpy(RuntimeSpy):
    attention_layout = "full"


class CompactAttentionRuntimeSpy(RuntimeSpy):
    attention_layout = "compact"


class RecordingScorer:
    def __init__(self, scores: list[float], features=FEATURE_NAMES):
        self.scores = scores
        self.features = tuple(features)
        self.calls: list[np.ndarray] = []

    def predict_stability(self, rows: np.ndarray) -> np.ndarray:
        self.calls.append(rows.copy())
        start = sum(call.shape[0] for call in self.calls[:-1])
        return np.asarray(self.scores[start : start + rows.shape[0]], dtype=np.float32)


def _load_frozen_locking_policy_module():
    path = (
        ROOT
        / "results"
        / "locking_original_plan_20260917_204619"
        / "source_collection"
        / "src"
        / "regret_remasking"
        / "locking_policy.py"
    )
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("frozen_locking_policy_for_parity", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_b0_matches_natural_one_token_per_forward_without_ties():
    model = ToyModel(
        schedules=[
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
        ],
        scores=[
            [9.0, 7.0, 6.0, 5.0],
            [1.0, 9.0, 6.0, 5.0],
            [1.0, 1.0, 9.0, 5.0],
            [1.0, 1.0, 1.0, 9.0],
        ],
    )
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=True,
    )

    assert result.token_ids == [10, 20, 30, 40]
    assert result.nfe == 4
    assert result.forward_calls == 4
    assert result.final_mask_count == 0
    assert result.immutable_token_violations == 0
    assert [row["selected"] for row in result.trace_rows if row["position"] == 2] == [True]


def test_b0_matches_existing_decode_batch_on_no_tie_toy_model():
    schedules = [[10, 20, 30, 40]] * 4
    scores = [
        [9.0, 7.0, 6.0, 5.0],
        [1.0, 9.0, 6.0, 5.0],
        [1.0, 1.0, 9.0, 5.0],
        [1.0, 1.0, 1.0, 9.0],
    ]
    tokenizer = ToyTokenizer()
    locking_result = decode_locking_policy(
        ToyModel(schedules=schedules, scores=scores),
        tokenizer,
        "prompt",
        "ex",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
    )
    batch_result = decode_batch(
        ToyModel(schedules=schedules, scores=scores),
        tokenizer,
        ["prompt"],
        ["ex"],
        DecodeConfig(steps=4, gen_length=4, block_length=4, mask_id=MASK_ID),
    )

    assert locking_result.token_ids == [int(token) for token in batch_result.token_ids[0, 2:].tolist()]
    assert locking_result.nfe == batch_result.nfe == 4


def test_b0_trace_matches_frozen_source_collection_decoder_without_ties():
    frozen = _load_frozen_locking_policy_module()
    if frozen is None:
        pytest.skip("frozen pass-2 source_collection is a local, gitignored results artifact")
    schedules = [[10, 20, 30, 40]] * 4
    scores = [
        [9.0, 7.0, 6.0, 5.0],
        [1.0, 9.0, 6.0, 5.0],
        [1.0, 1.0, 9.0, 5.0],
        [1.0, 1.0, 1.0, 9.0],
    ]
    tokenizer = ToyTokenizer()
    current = decode_locking_policy(
        ToyModel(schedules=schedules, scores=scores),
        tokenizer,
        "prompt",
        "ex",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=True,
    )
    old = frozen.decode_locking_policy(
        ToyModel(schedules=schedules, scores=scores),
        tokenizer,
        "prompt",
        "ex",
        frozen.LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=True,
    )

    assert current.token_ids == old.token_ids
    assert len(current.trace_rows) == len(old.trace_rows)
    for current_row, old_row in zip(current.trace_rows, old.trace_rows):
        for key in ("global_step", "position", "top_token", "selected", "stable", "trace_regret"):
            assert current_row[key] == old_row[key]
        for name in FEATURE_NAMES:
            np.testing.assert_allclose(current_row[name], old_row[name], rtol=1e-6, atol=1e-6)


def test_collecting_events_does_not_change_b0_trace_features_or_tokens():
    schedules = [[10, 20, 30, 40]] * 4
    scores = [
        [9.0, 7.0, 6.0, 5.0],
        [1.0, 9.0, 6.0, 5.0],
        [1.0, 1.0, 9.0, 5.0],
        [1.0, 1.0, 1.0, 9.0],
    ]
    config = LockingPolicyConfig(arm="B0", gen_length=4, block_length=4)
    traced = decode_locking_policy(
        ToyModel(schedules=schedules, scores=scores),
        ToyTokenizer(),
        "prompt",
        "ex",
        config,
        collect_trace=True,
        collect_events=False,
    )
    traced_with_events = decode_locking_policy(
        ToyModel(schedules=schedules, scores=scores),
        ToyTokenizer(),
        "prompt",
        "ex",
        config,
        collect_trace=True,
        collect_events=True,
    )

    assert traced.token_ids == traced_with_events.token_ids
    assert len(traced.events) == 0
    assert len(traced_with_events.events) == 4
    assert len(traced.trace_rows) == len(traced_with_events.trace_rows)
    for left, right in zip(traced.trace_rows, traced_with_events.trace_rows):
        for key in ("global_step", "position", "top_token", "selected", "stable", "trace_regret"):
            assert left[key] == right[key]
        for name in FEATURE_NAMES:
            np.testing.assert_allclose(left[name], right[name], rtol=1e-6, atol=1e-6)
    assert traced.recording_mode == "trace"
    assert traced_with_events.recording_mode == "trace_and_events"


def test_representation_lock_uses_second_post_commit_snapshot_for_next_forward():
    RuntimeSpy.instances.clear()
    model = ToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[
            [9.0, 7.0, 6.0, 5.0],
            [1.0, 9.0, 6.0, 5.0],
            [1.0, 1.0, 9.0, 5.0],
            [1.0, 1.0, 1.0, 9.0],
        ],
    )
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(
            arm="R-dense",
            gen_length=4,
            block_length=4,
            epsilon=0.0,
            reference_runtime_cls=RuntimeSpy,
        ),
        collect_events=True,
    )

    runtime = RuntimeSpy.instances[-1]
    freeze_events = [event for event in result.events if event["type"] == "reference_freeze"]
    assert freeze_events[0]["global_step"] == 2
    assert freeze_events[0]["position"] == 2
    assert freeze_events[0]["effective_from_step"] == 3
    assert runtime.commits[0] == [2]
    assert runtime.removed
    assert result.frozen_count >= 1
    assert result.generated_row_updates_avoided == 1


def test_representation_lock_uses_actual_dense_reference_runtime_by_default():
    model = HookableToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[
            [9.0, 7.0, 6.0, 5.0],
            [1.0, 9.0, 6.0, 5.0],
            [1.0, 1.0, 9.0, 5.0],
            [1.0, 1.0, 1.0, 9.0],
        ],
    )
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(arm="R-dense", gen_length=4, block_length=4, epsilon=0.0),
        collect_events=True,
    )

    assert result.frozen_count >= 1
    assert any(event["type"] == "reference_freeze" for event in result.events)


def test_reference_runtime_diagnostics_default_off_when_supported():
    class RuntimeWithDiagnostics(RuntimeSpy):
        capture_flags: list[bool] = []

        def __init__(self, model, mode: str, total_len: int, capture_diagnostics: bool):
            super().__init__(model, mode=mode, total_len=total_len)
            self.capture_diagnostics = capture_diagnostics
            self.capture_flags.append(capture_diagnostics)

    RuntimeSpy.instances.clear()
    RuntimeWithDiagnostics.capture_flags.clear()
    result = decode_locking_policy(
        ToyModel([[10, 20]] * 2, scores=[[9.0, 8.0]] * 2),
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(
            arm="R-dense",
            gen_length=2,
            block_length=2,
            epsilon=0.0,
            reference_runtime_cls=RuntimeWithDiagnostics,
        ),
        collect_events=False,
    )

    assert result.token_ids == [10, 20]
    assert RuntimeWithDiagnostics.capture_flags == [False]


def test_sparse_full_attention_counts_attention_rows_without_claiming_savings():
    FullAttentionRuntimeSpy.instances.clear()
    result = decode_locking_policy(
        ToyModel(
            schedules=[[10, 20, 30, 40]] * 4,
            scores=[
                [9.0, 7.0, 6.0, 5.0],
                [1.0, 9.0, 6.0, 5.0],
                [1.0, 1.0, 9.0, 5.0],
                [1.0, 1.0, 1.0, 9.0],
            ],
        ),
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(
            arm="R-sparse",
            gen_length=4,
            block_length=4,
            epsilon=0.0,
            reference_runtime_cls=FullAttentionRuntimeSpy,
        ),
        collect_events=True,
    )

    assert result.attention_layout == "full"
    assert result.forward_calls == 4
    assert result.total_query_rows == result.forward_calls * 6
    assert result.total_projection_rows < result.total_query_rows
    assert result.total_ffn_rows == result.total_projection_rows
    assert result.generated_row_updates_avoided == 1
    assert result.generated_row_updates_avoided_basis == "sparse_projection_and_ffn_rows"


def test_sparse_compact_or_legacy_fake_runtime_counts_active_attention_rows():
    result = decode_locking_policy(
        ToyModel(
            schedules=[[10, 20, 30, 40]] * 4,
            scores=[
                [9.0, 7.0, 6.0, 5.0],
                [1.0, 9.0, 6.0, 5.0],
                [1.0, 1.0, 9.0, 5.0],
                [1.0, 1.0, 1.0, 9.0],
            ],
        ),
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(
            arm="R-sparse",
            gen_length=4,
            block_length=4,
            epsilon=0.0,
            reference_runtime_cls=RuntimeSpy,
        ),
        collect_events=True,
    )

    assert result.attention_layout == "compact"
    assert result.forward_calls == 4
    assert result.total_query_rows == result.total_projection_rows
    assert result.total_query_rows == result.total_ffn_rows
    assert result.total_query_rows < result.forward_calls * 6


def test_trace_rows_backfill_final_stability_and_preserve_runlength_history():
    model = ToyModel(
        schedules=[
            [10, 99, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
        ],
        scores=[
            [9.0, 7.0, 6.0, 5.0],
            [1.0, 9.0, 6.0, 5.0],
            [1.0, 1.0, 9.0, 5.0],
            [1.0, 1.0, 1.0, 9.0],
        ],
    )
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=True,
    )

    early_pos1 = [row for row in result.trace_rows if row["global_step"] == 0 and row["relative_position"] == 1][0]
    later_pos2 = [row for row in result.trace_rows if row["global_step"] == 1 and row["relative_position"] == 2][0]
    assert early_pos1["top_token"] == 99
    assert early_pos1["final_token"] == 20
    assert early_pos1["stable"] == 0
    assert early_pos1["trace_regret"] == 1
    assert later_pos2["runlength"] == 2.0


def test_trace_scorer_receives_online_feature_matrix_and_controls_polarity():
    model = ToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )
    scorer = RecordingScorer([0.1, 0.9, 0.2, 0.3, 0.1, 0.8, 0.1, 0.9, 0.99, 0.99])
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "ex",
        LockingPolicyConfig(arm="T-fixed", gen_length=4, block_length=4),
        scorer=scorer,
        collect_trace=True,
    )

    first_trace = result.trace_rows[0]
    np.testing.assert_allclose(
        scorer.calls[0][0],
        np.asarray([first_trace[name] for name in FEATURE_NAMES], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    commits = [event["relative_position"] for event in result.events if event["type"] == "token_commit"]
    assert commits[:2] == [1, 2]
    assert sorted(commits) == [0, 1, 2, 3]
    assert result.frozen_count == 0


def test_adaptive_accepts_all_partial_or_fallback_with_forward_bound():
    all_high = RecordingScorer([0.99, 0.98, 0.97, 0.96])
    all_result = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]]),
        ToyTokenizer(),
        "prompt",
        "all",
        LockingPolicyConfig(arm="T-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=all_high,
    )
    assert all_result.token_ids == [10, 20, 30, 40]
    assert all_result.nfe == 1
    assert all_result.frozen_count == 0

    partial = RecordingScorer([0.99, 0.2, 0.2, 0.2, 0.99, 0.2, 0.2, 0.99, 0.2, 0.99])
    partial_result = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]] * 4),
        ToyTokenizer(),
        "prompt",
        "partial",
        LockingPolicyConfig(arm="T-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=partial,
    )
    assert partial_result.nfe == 4
    assert partial_result.final_mask_count == 0
    assert partial_result.immutable_token_violations == 0

    fallback = RecordingScorer([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
    fallback_result = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]]),
        ToyTokenizer(),
        "prompt",
        "fallback",
        LockingPolicyConfig(arm="T-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=fallback,
        collect_events=True,
    )
    assert fallback_result.nfe <= 4
    assert fallback_result.events[0]["fallback"]
    assert fallback_result.events[0]["relative_position"] == 3


def test_c_adaptive_uses_calibrated_scorer_over_raw_confidence():
    model = ToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[[9.0, 8.0, 7.0, 6.0]] * 4,
    )
    scorer = RecordingScorer([0.1, 0.2, 0.99, 0.1, 0.99, 0.1, 0.99, 0.99])
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "calibrated-confidence",
        LockingPolicyConfig(arm="C-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=scorer,
        collect_events=True,
    )

    commits = [event["relative_position"] for event in result.events if event["type"] == "token_commit"]
    assert commits[0] == 2
    assert result.adaptive_accept_count == 4
    assert result.frozen_count == 0


def test_c_adaptive_sends_only_requested_confidence_feature_without_trace():
    model = ToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[[9.0, 8.0, 7.0, 6.0]] * 4,
    )
    scorer = RecordingScorer([0.1, 0.2, 0.99, 0.1, 0.99, 0.1, 0.99, 0.99], features=("confidence",))
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "calibrated-confidence",
        LockingPolicyConfig(arm="C-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=scorer,
        collect_trace=False,
        collect_events=False,
    )

    assert result.token_ids == [10, 20, 30, 40]
    assert scorer.calls
    assert all(call.shape[1] == 1 for call in scorer.calls)
    assert result.raw_kl_min is None
    assert result.raw_kl_negative_count == 0
    assert result.raw_kl_exact_zero_count == 0


def test_timed_paths_skip_unused_kl_computation(monkeypatch):
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unused KL should not be computed")

    monkeypatch.setattr(locking_policy, "_raw_kl_unclamped", fail_if_called)

    b0 = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]] * 4),
        ToyTokenizer(),
        "prompt",
        "b0-no-kl",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=False,
        collect_events=True,
    )
    scorer = RecordingScorer([0.99, 0.99, 0.99, 0.99], features=("confidence",))
    c_adaptive = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]]),
        ToyTokenizer(),
        "prompt",
        "c-no-kl",
        LockingPolicyConfig(arm="C-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=scorer,
        collect_trace=False,
        collect_events=False,
    )

    assert b0.raw_kl_min is None
    assert b0.raw_kl_negative_count == 0
    assert b0.raw_kl_exact_zero_count == 0
    assert c_adaptive.raw_kl_min is None
    assert c_adaptive.raw_kl_negative_count == 0
    assert c_adaptive.raw_kl_exact_zero_count == 0


def test_trace_and_kl_scorer_paths_still_compute_kl(monkeypatch):
    original = locking_policy._raw_kl_unclamped
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(locking_policy, "_raw_kl_unclamped", counted)

    traced = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]] * 4),
        ToyTokenizer(),
        "prompt",
        "trace-kl",
        LockingPolicyConfig(arm="B0", gen_length=4, block_length=4),
        collect_trace=True,
        collect_events=False,
    )
    trace_calls = calls
    scorer = RecordingScorer([0.1] * 10, features=("kl",))
    scored = decode_locking_policy(
        ToyModel([[10, 20, 30, 40]]),
        ToyTokenizer(),
        "prompt",
        "scorer-kl",
        LockingPolicyConfig(arm="C-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=scorer,
        collect_trace=False,
        collect_events=False,
    )

    assert trace_calls == traced.forward_calls - 1
    assert calls == trace_calls + scored.forward_calls - 1
    assert traced.raw_kl_min is not None
    assert scored.raw_kl_min is not None
    assert scorer.calls
    assert all(call.shape[1] == 1 for call in scorer.calls)


def test_c_adaptive_collect_trace_keeps_full_rows_but_scores_requested_feature_only():
    model = ToyModel(
        schedules=[[10, 20, 30, 40]] * 4,
        scores=[[9.0, 8.0, 7.0, 6.0]] * 4,
    )
    scorer = RecordingScorer([0.1, 0.2, 0.99, 0.1, 0.99, 0.1, 0.99, 0.99], features=("confidence",))
    result = decode_locking_policy(
        model,
        ToyTokenizer(),
        "prompt",
        "calibrated-confidence",
        LockingPolicyConfig(arm="C-adaptive", gen_length=4, block_length=4, threshold=0.95),
        scorer=scorer,
        collect_trace=True,
        collect_events=False,
    )

    assert result.token_ids == [10, 20, 30, 40]
    assert scorer.calls
    assert all(call.shape[1] == 1 for call in scorer.calls)
    assert result.trace_rows
    assert set(FEATURE_NAMES).issubset(result.trace_rows[0])


def test_malformed_scorer_scores_are_rejected():
    class BadScorer:
        features = FEATURE_NAMES

        def predict_stability(self, rows):
            return np.asarray([[0.5] for _ in range(rows.shape[0])], dtype=np.float32)

    try:
        decode_locking_policy(
            ToyModel([[10, 20]] * 2),
            ToyTokenizer(),
            "prompt",
            "bad",
            LockingPolicyConfig(arm="T-adaptive", gen_length=2, block_length=2),
            scorer=BadScorer(),
        )
    except ValueError as error:
        assert "scorer returned shape" in str(error)
    else:
        raise AssertionError("malformed scorer output must fail")


def test_config_and_probability_validation():
    class OutOfRangeScorer:
        features = FEATURE_NAMES

        def predict_stability(self, rows):
            return np.full(rows.shape[0], 1.01, dtype=np.float32)

    try:
        decode_locking_policy(
            ToyModel([[10, 20]] * 2),
            ToyTokenizer(),
            "prompt",
            "bad-threshold",
            LockingPolicyConfig(arm="B0", gen_length=2, block_length=2, threshold=float("nan")),
        )
    except ValueError as error:
        assert "threshold" in str(error)
    else:
        raise AssertionError("non-finite threshold must fail")

    try:
        decode_locking_policy(
            ToyModel([[10, 20]] * 2),
            ToyTokenizer(),
            "prompt",
            "bad-score",
            LockingPolicyConfig(arm="C-adaptive", gen_length=2, block_length=2),
            scorer=OutOfRangeScorer(),
        )
    except ValueError as error:
        assert "probabilities" in str(error)
    else:
        raise AssertionError("out-of-range scorer probabilities must fail")


if __name__ == "__main__":
    test_b0_matches_natural_one_token_per_forward_without_ties()
    test_b0_matches_existing_decode_batch_on_no_tie_toy_model()
    test_b0_trace_matches_frozen_source_collection_decoder_without_ties()
    test_collecting_events_does_not_change_b0_trace_features_or_tokens()
    test_representation_lock_uses_second_post_commit_snapshot_for_next_forward()
    test_representation_lock_uses_actual_dense_reference_runtime_by_default()
    test_reference_runtime_diagnostics_default_off_when_supported()
    test_sparse_full_attention_counts_attention_rows_without_claiming_savings()
    test_sparse_compact_or_legacy_fake_runtime_counts_active_attention_rows()
    test_trace_rows_backfill_final_stability_and_preserve_runlength_history()
    test_trace_scorer_receives_online_feature_matrix_and_controls_polarity()
    test_adaptive_accepts_all_partial_or_fallback_with_forward_bound()
    test_c_adaptive_uses_calibrated_scorer_over_raw_confidence()
    test_c_adaptive_sends_only_requested_confidence_feature_without_trace()
    test_c_adaptive_collect_trace_keeps_full_rows_but_scores_requested_feature_only()
    test_malformed_scorer_scores_are_rejected()
    test_config_and_probability_validation()
    print("locking policy tests passed")
