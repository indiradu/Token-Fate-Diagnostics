from __future__ import annotations

import torch

from regret_remasking.phase_lock import (
    representation_ready,
    select_representation_lock,
    select_semantic_transfer,
    semantic_gate_eligible,
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


def test_capped_rankers_change_ranking_not_admission() -> None:
    f = _features()
    confidence = semantic_priority("capped_confidence", predicted_regret=None, **f)
    margin = semantic_priority("capped_margin", predicted_regret=None, **f)
    entropy = semantic_priority("capped_entropy", predicted_regret=None, **f)
    posterior = semantic_priority("capped_posterior_kl", predicted_regret=None, **f)

    assert confidence.argmax().item() == 0
    assert margin.argmax().item() == 2
    assert entropy.argmax().item() == 0
    assert posterior.argmax().item() == 1


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


def test_semantic_gate_combines_confidence_margin_kl_and_persistence() -> None:
    allowed = torch.ones((1, 4), dtype=torch.bool)
    eligible = semantic_gate_eligible(
        allowed,
        confidence=torch.tensor([[0.96, 0.94, 0.97, 0.98]]),
        margin=torch.tensor([[0.20, 0.20, 0.04, 0.30]]),
        kl=torch.tensor([[0.004, 0.004, 0.004, 0.006]]),
        runlength=torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
        min_confidence=0.95,
        min_margin=0.05,
        max_kl=0.005,
        min_runlength=2,
        step_in_block=1,
        optional_steps_per_block=2,
        require_posterior_history=True,
        posterior_history_available=True,
    )

    assert torch.equal(eligible, torch.tensor([[True, False, False, False]]))


def test_semantic_gate_abstains_without_history_or_outside_step_window() -> None:
    allowed = torch.ones((1, 2), dtype=torch.bool)
    features = {
        "confidence": torch.ones((1, 2)),
        "margin": torch.ones((1, 2)),
        "kl": torch.zeros((1, 2)),
        "runlength": torch.ones((1, 2)),
    }

    no_history = semantic_gate_eligible(
        allowed,
        **features,
        require_posterior_history=True,
        posterior_history_available=False,
    )
    too_late = semantic_gate_eligible(
        allowed,
        **features,
        step_in_block=2,
        optional_steps_per_block=2,
    )
    too_early = semantic_gate_eligible(
        allowed,
        **features,
        min_block_age=1,
        step_in_block=0,
    )

    assert not no_history.any()
    assert not too_late.any()
    assert not too_early.any()


def test_capped_drift_requires_gate_patience_and_age() -> None:
    drift = torch.tensor([[0.001, 0.003, 0.001, 0.001]])
    ready = representation_ready(
        "capped_drift",
        drift,
        confidence=torch.ones_like(drift),
        kl=torch.zeros_like(drift),
        runlength=torch.ones_like(drift),
        age=torch.tensor([[2, 2, 0, 2]]),
        below_threshold_count=torch.tensor([[2, 2, 2, 1]]),
        threshold=0.002,
        patience=2,
        min_age=1,
    )

    assert torch.equal(ready, torch.tensor([[True, False, False, False]]))


def test_capped_drift_ranks_lowest_drift_under_candidate_budget() -> None:
    candidate = torch.tensor([[True, True, True, True, False]])
    ready = torch.tensor([[True, True, False, True, True]])
    drift = torch.tensor([[0.004, 0.001, 0.0001, 0.003, 0.00001]])

    selected = select_representation_lock(
        candidate,
        ready,
        drift,
        lock_fraction=0.5,
    )

    # The candidate budget is ceil(0.5 * 4) = 2. Position 4 is ready but is
    # not semantically committed, and position 2 is a candidate but not ready.
    assert torch.equal(selected, torch.tensor([[False, True, False, True, False]]))


def test_zero_representation_fraction_abstains() -> None:
    candidate = torch.ones((1, 3), dtype=torch.bool)
    ready = torch.ones_like(candidate)
    drift = torch.zeros((1, 3))

    selected = select_representation_lock(candidate, ready, drift, lock_fraction=0.0)

    assert not selected.any()


def test_high_drift_control_reverses_only_ranking() -> None:
    candidate = torch.tensor([[True, True, True, True]])
    ready = torch.tensor([[True, True, True, True]])
    drift = torch.tensor([[0.004, 0.001, 0.009, 0.003]])

    low = select_representation_lock(
        candidate, ready, drift, lock_fraction=1.0, budget_override=2
    )
    high = select_representation_lock(
        candidate,
        ready,
        drift,
        lock_fraction=1.0,
        budget_override=2,
        prefer_high_drift=True,
    )

    assert torch.equal(low, torch.tensor([[False, True, False, True]]))
    assert torch.equal(high, torch.tensor([[True, False, True, False]]))
    assert int(low.sum()) == int(high.sum()) == 2


def test_matched_representation_budget_is_exact_and_validated() -> None:
    candidate = torch.tensor([[True, True, True]])
    ready = torch.tensor([[True, False, True]])
    drift = torch.tensor([[0.01, 0.02, 0.03]])

    selected = select_representation_lock(
        candidate, ready, drift, lock_fraction=0.0, budget_override=2
    )
    assert torch.equal(selected, ready)

    try:
        select_representation_lock(
            candidate, ready, drift, lock_fraction=1.0, budget_override=3
        )
    except ValueError as exc:
        assert "exceeds eligible" in str(exc)
    else:
        raise AssertionError("an impossible matched budget must fail")


def test_high_drift_control_uses_age_for_candidate_admission() -> None:
    drift = torch.tensor([[0.001, 0.200, 0.500]])
    ready = representation_ready(
        "capped_high_drift",
        drift,
        confidence=torch.ones_like(drift),
        kl=torch.zeros_like(drift),
        runlength=torch.ones_like(drift),
        age=torch.tensor([[0, 1, 2]]),
        below_threshold_count=torch.zeros_like(drift, dtype=torch.long),
        threshold=0.002,
        patience=4,
        min_age=1,
    )

    assert torch.equal(ready, torch.tensor([[False, True, True]]))
