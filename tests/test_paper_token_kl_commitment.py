from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.paper_experiment import make_paper_policies


class TinyTokenizer:
    padding_side = "left"

    def __call__(self, text: str, *, add_special_tokens: bool, return_tensors: str) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def batch_decode(self, token_ids: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return [" ".join(str(token) for token in token_ids[0])]


def _logits_from_probs(probs: list[float]) -> torch.Tensor:
    values = torch.tensor(probs, dtype=torch.float32)
    values = values / values.sum()
    return values.log()


class ScriptedPosteriorModel(torch.nn.Module):
    def __init__(self, frames: list[dict[int, list[float]]], *, vocab_size: int = 6) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.frames = frames
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -20.0, device=input_ids.device)
        default = _logits_from_probs([0.001, 0.60, 0.20, 0.10, 0.05, 0.049]).to(input_ids.device)
        logits[:] = default
        for position, probs in frame.items():
            logits[0, position] = _logits_from_probs(probs).to(input_ids.device)
        hidden = tuple(torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device) for _ in range(33))
        return SimpleNamespace(logits=logits, hidden_states=hidden)


class CountingRuntime:
    def __init__(self, model: torch.nn.Module, mode: str, total_len: int, capture_diagnostics: bool = False) -> None:
        self.model = model
        self.mode = mode
        self.total_len = total_len
        self.capture_diagnostics = capture_diagnostics
        self.blocks = [object(), object()]
        self.locked: set[int] = set()
        self.output_positions = torch.arange(total_len)
        self.last_output_positions = self.output_positions
        self.last_output_hidden = None
        self._rows = total_len

    def install(self) -> None:
        return None

    def remove(self) -> None:
        return None

    def begin_forward(self) -> None:
        self.output_positions = torch.arange(self.total_len)
        self.last_output_positions = self.output_positions
        self._rows = self.total_len - len(self.locked)

    @property
    def last_work_counts(self) -> dict[str, int]:
        layers = len(self.blocks)
        return {
            "attention_query_rows": self._rows * layers,
            "projection_rows": self._rows * layers,
            "ffn_rows": self._rows * layers,
            "head_rows": self._rows,
        }

    def commit(self, positions: list[int]) -> None:
        self.locked.update(int(position) for position in positions)


def _decode(
    frames: list[dict[int, list[float]]],
    config: PaperDecodeConfig,
    *,
    observer: bool = False,
) -> tuple[object, ScriptedPosteriorModel]:
    model = ScriptedPosteriorModel(frames)
    seen_steps: list[int] = []
    result = decode_paper_policy(
        model,
        TinyTokenizer(),
        "prompt",
        "ex",
        config,
        collect_events=True,
        step_observer=(lambda event: seen_steps.append(int(event["step"]))) if observer else None,
    )
    assert not observer or seen_steps == list(range(result.nfe))
    return result, model


def test_k_adaptive_commits_stable_overlap_but_not_new_window_positions() -> None:
    frames = [
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 2: [0.001, 0.20, 0.70, 0.05, 0.03, 0.019]},
        {2: [0.001, 0.20, 0.70, 0.05, 0.03, 0.019], 3: [0.001, 0.01, 0.01, 0.95, 0.02, 0.009]},
        {3: [0.001, 0.01, 0.01, 0.95, 0.02, 0.009]},
    ]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(arm="K-adaptive", gen_length=3, block_length=3, window_length=2, epsilon=0.005, mask_id=0),
    )

    events = [event for event in result.events if event["type"] == "token_commit"]
    assert [(event["position"], event["fallback"]) for event in events] == [(1, True), (2, False), (3, False)]
    assert events[0]["had_immediate_history"] is False
    assert events[0]["kl"] is None
    assert events[1]["had_immediate_history"] is True
    assert events[1]["kl"] <= 0.005
    assert not any(event["had_immediate_history"] is False and event["fallback"] is False for event in events)
    assert result.frozen_count == 0
    assert result.generated_row_updates_avoided == 0
    assert result.adaptive_accept_count == 3
    assert result.fallback_count == 1


def test_k_adaptive_same_argmax_with_large_kl_uses_fallback() -> None:
    frames = [
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 2: [0.001, 0.10, 0.80, 0.05, 0.03, 0.019]},
        {2: [0.001, 0.30, 0.51, 0.10, 0.05, 0.039]},
    ]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(arm="K-adaptive", gen_length=2, block_length=2, window_length=2, epsilon=0.005, mask_id=0),
    )

    last = result.events[-1]
    assert last["position"] == 2
    assert last["token"] == 2
    assert last["had_immediate_history"] is True
    assert last["kl"] > 0.005
    assert last["fallback"] is True
    assert result.fallback_count == 2


def test_k_adaptive_uses_current_to_previous_kl_direction() -> None:
    previous = [0.001, 0.90, 0.097998, 0.001, 0.000001, 0.000001]
    current = [0.001, 0.99, 0.007998, 0.001, 0.000001, 0.000001]
    previous_t = torch.tensor(previous, dtype=torch.float32)
    previous_t = previous_t / previous_t.sum()
    current_t = torch.tensor(current, dtype=torch.float32)
    current_t = current_t / current_t.sum()
    current_previous = float((current_t * (current_t.log() - previous_t.log())).sum())
    previous_current = float((previous_t * (previous_t.log() - current_t.log())).sum())
    assert current_previous < 0.08
    assert previous_current > 0.08

    frames = [
        {1: [0.001, 0.995, 0.001, 0.001, 0.001, 0.001], 2: previous},
        {2: current},
    ]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(arm="K-adaptive", gen_length=2, block_length=2, window_length=2, epsilon=0.08, mask_id=0),
    )

    last = result.events[-1]
    assert last["position"] == 2
    assert last["fallback"] is False
    assert last["kl"] == pytest.approx(current_previous, abs=1e-6)


def test_k_adaptive_observer_does_not_change_decision_history() -> None:
    frames = [
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 2: [0.001, 0.20, 0.70, 0.05, 0.03, 0.019]},
        {2: [0.001, 0.20, 0.70, 0.05, 0.03, 0.019], 3: [0.001, 0.01, 0.01, 0.95, 0.02, 0.009]},
        {3: [0.001, 0.01, 0.01, 0.95, 0.02, 0.009]},
    ]
    config = PaperDecodeConfig(arm="K-adaptive", gen_length=3, block_length=3, window_length=2, epsilon=0.005, mask_id=0)
    timed, _ = _decode(frames, config, observer=False)
    observed, _ = _decode(frames, config, observer=True)

    keep = ("position", "token", "fallback", "had_immediate_history", "kl")
    assert [{key: event[key] for key in keep} for event in timed.events] == [
        {key: event[key] for key in keep} for event in observed.events
    ]


def test_k_adaptive_rejects_representation_kl() -> None:
    with pytest.raises(ValueError, match="K-adaptive only supports token kl_space"):
        make_paper_policies(["K-adaptive"], [0.005], [0.95], kl_space="representation")
    with pytest.raises(ValueError, match="token-KL commitment arms only support token kl_space"):
        _decode(
            [{1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019]}],
            PaperDecodeConfig(arm="K-adaptive", gen_length=1, block_length=1, kl_space="representation", mask_id=0),
        )


def test_s_window_uses_window_schedule_and_sparse_reference_work() -> None:
    frames = [
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 2: [0.001, 0.01, 0.95, 0.02, 0.01, 0.009]},
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 2: [0.001, 0.01, 0.95, 0.02, 0.01, 0.009]},
        {1: [0.001, 0.80, 0.10, 0.05, 0.03, 0.019], 3: [0.001, 0.01, 0.01, 0.95, 0.02, 0.009]},
    ]
    result, _ = _decode(
        frames,
        PaperDecodeConfig(
            arm="S-window",
            gen_length=3,
            block_length=3,
            window_length=1,
            epsilon=0.005,
            confidence_percentile=100,
            mask_id=0,
            runtime_cls=CountingRuntime,
        ),
    )

    commits = [event for event in result.events if event["type"] == "token_commit"]
    assert commits[0]["position"] == 1
    assert result.attention_layout == "compact"
    assert result.frozen_count >= 1
    assert result.total_query_rows < result.nfe * (result.prompt_length + 3)
    assert result.total_projection_rows == result.total_query_rows
    assert result.total_ffn_rows == result.total_query_rows


def test_policy_factory_ids_for_fixed_four_arm_campaign(tmp_path: Path) -> None:
    controller = tmp_path / "controller.pt"
    projector = tmp_path / "projector.pt"
    controller.write_bytes(b"controller")
    projector.write_bytes(b"projector")

    policies = make_paper_policies(
        ["W-confidence", "S-window", "K-adaptive", "T-adaptive"],
        [0.005],
        [0.95],
        confidence_percentile=20,
        controller_checkpoint=controller,
        projector_checkpoint=projector,
    )

    assert [policy["policy_id"] for policy in policies] == [
        "W-confidence",
        "S-window-eps-0.005-cp-20",
        "K-adaptive-eps-0.005",
        "T-adaptive-tau-0.95",
    ]
    by_id = {policy["policy_id"]: policy for policy in policies}
    assert by_id["S-window-eps-0.005-cp-20"]["confidence_percentile"] == 20
    assert by_id["K-adaptive-eps-0.005"]["confidence_percentile"] == 100
    assert by_id["K-adaptive-eps-0.005"]["epsilon"] == 0.005
    assert by_id["K-adaptive-eps-0.005"]["threshold"] is None
