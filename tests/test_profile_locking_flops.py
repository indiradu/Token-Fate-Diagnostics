import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location("profile_locking_flops", Path(__file__).resolve().parents[1] / "scripts/profile_locking_flops.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_observer_preserves_matrix_output_and_counts_multiply_add_twice():
    a, b = torch.randn(3, 5), torch.randn(5, 7)
    expected = a @ b
    with module.MatrixFlopObserver() as observer:
        actual = a @ b
    assert torch.equal(actual, expected)
    assert sum(observer.flops.values()) == 2 * 3 * 5 * 7


def test_attention_keeps_full_key_length_with_sparse_queries():
    q = torch.empty(1, 2, 3, 4)
    k = v = torch.empty(1, 2, 11, 4)
    assert module.operation_flops("aten._scaled_dot_product_flash_attention.default", (q, k, v)) == 4 * 2 * 3 * 11 * 4


def test_unknown_fused_attention_is_rejected():
    with pytest.raises(ValueError, match="unaccounted"):
        module.operation_flops("aten.unknown_attention.default", ())


def test_fused_encoder_observation_does_not_disable_fastpath():
    layer = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, batch_first=True, dropout=0).eval()
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        expected = layer(x)
        with module.MatrixFlopObserver() as observer:
            actual = layer(x)
    assert torch.equal(actual, expected)
    assert observer.calls["aten._transformer_encoder_layer_fwd.default"] == 1
    assert sum(observer.flops.values()) == 8 * 5 * 8**2 + 4 * 5**2 * 8 + 4 * 5 * 8 * 16
