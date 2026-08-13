from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import math

import torch

from regret_remasking.features import entropy_from_log_probs, local_window_average, safe_jsd, safe_kl


def test_local_window_average() -> None:
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    got = local_window_average(x, window=1)
    expected = torch.tensor([[0.5, 1.0, 2.0, 2.5]])
    assert torch.allclose(got, expected)


def test_safe_kl_first_step_is_zero() -> None:
    log_probs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    got = safe_kl(log_probs, None)
    assert got.shape == (2, 3)
    assert torch.allclose(got, torch.zeros_like(got))


def test_entropy_from_log_probs_uniform_distribution() -> None:
    log_probs = torch.log_softmax(torch.zeros(1, 2, 4), dim=-1)
    got = entropy_from_log_probs(log_probs)
    expected = torch.full((1, 2), math.log(4.0))
    assert torch.allclose(got, expected)


def test_safe_jsd_zero_for_identical_distributions() -> None:
    log_probs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    got = safe_jsd(log_probs, log_probs)
    assert got.shape == (2, 3)
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)


if __name__ == "__main__":
    test_local_window_average()
    test_safe_kl_first_step_is_zero()
    test_entropy_from_log_probs_uniform_distribution()
    test_safe_jsd_zero_for_identical_distributions()
    print("feature tests passed")
