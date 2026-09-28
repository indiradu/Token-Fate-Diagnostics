from __future__ import annotations

import torch

from regret_remasking.tracelock_baseline import (
    TraceLockConfig,
    TraceLockController,
    pack_left_padded_controller_batch,
    select_tracelock_commits,
    soft_crop_window,
    tracelock_loss,
)


def test_trace_features_include_three_snapshots_and_two_deltas() -> None:
    hidden = torch.tensor([[[[1.0], [3.0], [8.0]]]])
    features = TraceLockController.trace_state_features(hidden)
    assert torch.equal(features, torch.tensor([[[1.0, 3.0, 8.0, 2.0, 5.0]]]))


def test_controller_shapes_and_dynamic_margin() -> None:
    config = TraceLockConfig(
        projected_hidden_dim=2,
        controller_dim=16,
        attention_heads=4,
        ffn_dim=32,
        encoder_layers=2,
        max_sequence_length=8,
    )
    model = TraceLockController(config).eval()
    output = model(
        torch.randn(2, 5, 3, 2),
        torch.ones((2, 5), dtype=torch.bool),
    )
    assert output["margin"].shape == (2, 5)
    assert output["threshold_logit"].shape == (2,)
    assert torch.allclose(
        output["margin"],
        output["raw_logits"] - output["threshold_logit"].unsqueeze(-1),
    )


def test_weighted_loss_penalizes_wrong_sign_more() -> None:
    labels = torch.tensor([[1.0, 0.0]])
    active = torch.ones((1, 2), dtype=torch.bool)
    threshold = torch.zeros(1)
    correct = tracelock_loss(
        torch.tensor([[2.0, -2.0]]), threshold, labels, active
    )
    wrong = tracelock_loss(
        torch.tensor([[-2.0, 2.0]]), threshold, labels, active
    )
    assert wrong > correct * 4.0


def test_soft_crop_begins_at_first_active_position() -> None:
    active = torch.tensor([False, False, False, False, True, False, True, True])
    assert soft_crop_window(active, prompt_length=2, window_width=3) == (4, 7)


def test_selector_uses_threshold_then_fallback() -> None:
    active = torch.tensor([False, False, True, True, True, True])
    probability = torch.tensor([0.0, 0.0, 0.90, 0.96, 0.99, 0.20])
    selected = select_tracelock_commits(
        probability,
        active,
        prompt_length=2,
        window_width=3,
        threshold=0.95,
    )
    assert torch.equal(
        selected, torch.tensor([False, False, False, True, True, False])
    )
    fallback = select_tracelock_commits(
        probability * 0.5,
        active,
        prompt_length=2,
        window_width=3,
        threshold=0.95,
    )
    assert fallback.sum().item() == 1
    assert fallback[4]


def test_controller_batch_repacking_removes_left_padding() -> None:
    # Shared prompt width is three. Sample zero has one left-pad token and two
    # real prompt tokens; both samples then have two generation positions.
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3, 1)
    prompt_attention = torch.tensor([[False, True, True], [True, True, True]])
    active = torch.tensor(
        [
            [False, False, False, True, False],
            [False, False, False, False, True],
        ]
    )
    packed, attention, packed_active, lengths = pack_left_padded_controller_batch(
        hidden,
        prompt_attention,
        active,
        generation_length=2,
    )
    assert torch.equal(lengths, torch.tensor([2, 3]))
    assert torch.equal(packed[0, :4], hidden[0, [1, 2, 3, 4]])
    assert torch.count_nonzero(packed[0, 4]).item() == 0
    assert torch.equal(attention[0], torch.tensor([True, True, True, True, False]))
    assert torch.equal(
        packed_active[0], torch.tensor([False, False, True, False, False])
    )
    assert torch.equal(packed[1], hidden[1])
