from __future__ import annotations

import unittest
import torch

from benchmark_phaselock_sure_hybrid import (
    guard_semantic_candidates, speed_reference_gate, _fallback_progress,
)


class SpeedGateTests(unittest.TestCase):
    def test_delay_retains_prompt_but_never_mask_snapshot(self):
        proposed = torch.ones((1, 5), dtype=torch.bool)
        age = torch.tensor([[0, 0, 0, 1, 2]])
        result = speed_reference_gate(proposed, age, prompt_len=2, policy="delay", min_age=1)
        self.assertEqual(result.tolist(), [[True, True, False, True, True]])
        self.assertTrue(proposed.all())  # no mutation of shared Sure proposals

    def test_delay_two_updates(self):
        result = speed_reference_gate(torch.ones((1, 4), dtype=torch.bool),
                                      torch.tensor([[0, 0, 1, 2]]), prompt_len=1,
                                      policy="delay", min_age=2)
        self.assertEqual(result.tolist(), [[True, False, False, True]])

    def test_drift_requires_two_fresh_observations_and_patience(self):
        result = speed_reference_gate(
            torch.ones((1, 6), dtype=torch.bool), torch.tensor([[0, 1, 2, 3, 3, 3]]),
            prompt_len=1, policy="drift", min_age=1,
            drift=torch.tensor([[float("inf"), 0., 0., .005, .02, float("nan")]]),
            count=torch.tensor([[0, 5, 1, 2, 3, 5]]), threshold=.01, patience=2,
        )
        self.assertEqual(result.tolist(), [[True, False, False, True, False, False]])

    def test_live_and_generated_only_controls(self):
        proposed = torch.ones((1, 4), dtype=torch.bool)
        age = torch.zeros((1, 4), dtype=torch.long)
        self.assertFalse(speed_reference_gate(proposed, age, prompt_len=2, policy="live", min_age=1).any())
        self.assertEqual(speed_reference_gate(proposed, age, prompt_len=2,
                                              policy="generated_only", min_age=1).tolist(),
                         [[False, False, True, True]])
        self.assertTrue(speed_reference_gate(proposed, age, prompt_len=2,
                                            policy="sure", min_age=1).all())

    def test_confidence_guard_cap_and_fallback_are_separate(self):
        allowed = torch.tensor([[True, True, True, True, False]])
        confidence = torch.tensor([[.95, .99, .8, .7, 1.]])
        result = guard_semantic_candidates(torch.ones_like(allowed), allowed, confidence,
                                          min_confidence=.9, cap_fraction=.25)
        self.assertEqual(result.tolist(), [[False, True, False, False, False]])
        rejected = guard_semantic_candidates(allowed, allowed, confidence, min_confidence=1.)
        self.assertFalse(rejected.any())
        self.assertEqual(_fallback_progress(rejected, allowed, confidence).tolist(), result.tolist())

    def test_bad_parameters_fail(self):
        t = torch.ones((1, 2), dtype=torch.bool)
        with self.assertRaises(ValueError):
            guard_semantic_candidates(t, t, t.float(), cap_fraction=0)
        with self.assertRaises(ValueError):
            speed_reference_gate(t, t.long(), prompt_len=1, policy="drift", min_age=1)


if __name__ == "__main__":
    unittest.main()
