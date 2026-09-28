from __future__ import annotations

import torch

from benchmark_phaselock_trace_semantic import POLICIES, phase1_window_mask


def test_phase1_window_mask_matches_packed_soft_crop() -> None:
    # The packed prompts have lengths two and three even though the model batch
    # uses a shared prompt width of four. The second sample's first generation
    # token is already resolved, so its crop begins at relative position one.
    controller_active = torch.tensor(
        [
            [False, False, True, True, True, True, False, False],
            [False, False, False, False, True, True, True, False],
        ]
    )
    prompt_lengths = torch.tensor([2, 3])

    selected, ends = phase1_window_mask(
        controller_active,
        prompt_lengths,
        model_prompt_width=4,
        generation_length=4,
        window_width=2,
    )

    assert selected.tolist() == [
        [False, False, False, False, True, True, False, False],
        [False, False, False, False, False, True, True, False],
    ]
    assert ends == [4, 6]


def test_guarded_surelite_changes_only_surelite_vetoes() -> None:
    expected = {
        "trace_surelite_union_d02_c80_p2": (0.80, 2),
        "trace_surelite_union_d02_c90_p2": (0.90, 2),
        "trace_surelite_union_d02_c80_p3": (0.80, 3),
    }
    for name, (confidence_floor, patience) in expected.items():
        policy = POLICIES[name]
        assert policy["proposal_mode"] == "trace_surelite_union"
        assert policy["probability_threshold"] == 0.95
        assert policy["confidence_threshold"] == 0.0
        assert policy["confidence_delta"] == 0.02
        assert policy["sure_confidence_threshold"] == confidence_floor
        assert policy["minimum_token_patience"] == patience


def test_matched_semantic_policies_are_frozen() -> None:
    capped = POLICIES["capped_confidence"]
    assert capped == {
        "proposal_mode": "capped_confidence",
        "semantic_lock_fraction": 0.06,
        "confidence_threshold": 0.95,
        "optional_steps_per_block": 2,
    }
    assert POLICIES["direct_safety_full"] == {
        "proposal_mode": "direct_safety_full",
        "maximum_extra_per_update": 2,
    }
