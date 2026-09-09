from __future__ import annotations

import torch

from regret_remasking.phase_lock import (
    select_semantic_transfer,
    semantic_priority,
    update_top1_runlength,
)


def _features() -> dict[str, torch.Tensor]:
    return {
        "confidence": torch.tensor([[0.90, 0.80, 0.70, 0.60]]),
        "margin": torch.tensor([[0.30, 0.10, 0.40, 0.20]]),
        "entropy": torch.tensor([[0.10, 0.40, 0.20, 0.30]]),
        "kl": torch.tensor([[0.30, 0.10, 0.20, 0.40]]),
        "runlength": torch.tensor([[1.0, 3.0, 2.0, 1.0]]),
        "top1_flip": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    }


def test_standalone_semantic_priorities() -> None:
    f = _features()
    zeros = torch.zeros_like(f["confidence"])
    entropy = semantic_priority("entropy", predicted_regret=None, **f)
    posterior = semantic_priority("posterior_kl", predicted_regret=None, **f)
    persistence = semantic_priority("persistence_only", predicted_regret=None, **f)
    fate = semantic_priority("token_fate_only", predicted_regret=zeros + torch.tensor([[0.4, 0.1, 0.3, 0.2]]), **f)

    assert entropy.argmax().item() == 0
    assert posterior.argmax().item() == 1
    assert persistence.argmax().item() == 1
    assert fate.argmax().item() == 1


def test_selector_changes_only_accelerated_locks() -> None:
    allowed = torch.ones((1, 4), dtype=torch.bool)
    confidence = torch.tensor([[0.90, 0.80, 0.70, 0.60]])
    alternative = torch.tensor([[0.00, 0.10, 0.20, 1.00]])

    transfer, scheduled, accelerated = select_semantic_transfer(
        allowed,
        confidence,
        alternative,
        base_k=1,
        total_k=2,
    )

    assert torch.equal(scheduled, torch.tensor([[True, False, False, False]]))
    assert torch.equal(accelerated, torch.tensor([[False, False, False, True]]))
    assert torch.equal(transfer, scheduled | accelerated)


def test_no_acceleration_matches_baseline_confidence_selection() -> None:
    allowed = torch.ones((1, 4), dtype=torch.bool)
    confidence = torch.tensor([[0.90, 0.80, 0.70, 0.60]])
    alternative = torch.tensor([[0.00, 0.10, 0.20, 1.00]])

    transfer, scheduled, accelerated = select_semantic_transfer(
        allowed,
        confidence,
        alternative,
        base_k=2,
        total_k=2,
    )

    assert torch.equal(transfer, torch.tensor([[True, True, False, False]]))
    assert torch.equal(transfer, scheduled)
    assert not accelerated.any()


def test_top1_runlength_is_carried_across_steps() -> None:
    previous_top1 = torch.tensor([[-1, 7, 9]])
    current_top1 = torch.tensor([[4, 7, 8]])
    previous_runlength = torch.tensor([[0.0, 3.0, 2.0]])

    updated = update_top1_runlength(previous_top1, current_top1, previous_runlength)

    assert torch.equal(updated, torch.tensor([[1.0, 4.0, 1.0]]))


def test_accelerated_gate_can_abstain_without_changing_scheduled_transfer() -> None:
    allowed = torch.ones((1, 4), dtype=torch.bool)
    confidence = torch.tensor([[0.90, 0.80, 0.70, 0.60]])
    priority = torch.tensor([[0.00, 0.10, 0.20, 1.00]])
    eligible = torch.tensor([[False, False, False, False]])

    transfer, scheduled, accelerated = select_semantic_transfer(
        allowed,
        confidence,
        priority,
        base_k=1,
        total_k=2,
        accelerated_eligible=eligible,
    )

    assert torch.equal(transfer, torch.tensor([[True, False, False, False]]))
    assert torch.equal(transfer, scheduled)
    assert not accelerated.any()


def test_accelerated_gate_excludes_ineligible_high_priority_candidate() -> None:
    allowed = torch.ones((1, 4), dtype=torch.bool)
    confidence = torch.tensor([[0.90, 0.80, 0.70, 0.60]])
    priority = torch.tensor([[0.00, 0.10, 0.20, 1.00]])
    eligible = torch.tensor([[False, False, True, False]])

    transfer, scheduled, accelerated = select_semantic_transfer(
        allowed,
        confidence,
        priority,
        base_k=1,
        total_k=2,
        accelerated_eligible=eligible,
    )

    assert torch.equal(scheduled, torch.tensor([[True, False, False, False]]))
    assert torch.equal(accelerated, torch.tensor([[False, False, True, False]]))
    assert torch.equal(transfer, scheduled | accelerated)
