from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_settlement_fate_audit import CommittedToken, relative_l2, update_committed_drift


def _make_token(position: int, selected_step: int) -> CommittedToken:
    return CommittedToken(
        example_id="a",
        dataset="math500",
        position=position,
        relative_position=position,
        selected_step=selected_step,
        selected_block_t_frac=0.5,
        selected_token=1,
        selected_confidence=0.9,
        selected_entropy=0.1,
        selected_margin=0.8,
        selected_kl=0.0,
    )


def _reference_drift(captured_by_step, layers, tokens):
    """Original per-token loop semantics, using relative_l2 directly."""
    base = {}
    expected = {}
    future_steps = {t.position: 0 for t in tokens}
    for step in sorted(captured_by_step):
        captured = captured_by_step[step]
        for token in tokens:
            if step <= token.selected_step:
                continue
            if token.position not in base:
                base[token.position] = {
                    layer: captured[layer][0, token.position].float() for layer in layers
                }
                expected[token.position] = {layer: (0.0, 0.0) for layer in layers}
                continue
            future_steps[token.position] += 1
            for layer in layers:
                drift = relative_l2(captured[layer][0, token.position], base[token.position][layer])
                prev_max, _ = expected[token.position][layer]
                expected[token.position][layer] = (max(prev_max, drift), drift)
    return expected, future_steps


def test_vectorized_drift_matches_per_token_reference() -> None:
    torch.manual_seed(7)
    layers = [1, 2]
    seq_len, dim, steps = 6, 5, 5
    tokens = [_make_token(0, 0), _make_token(2, 1), _make_token(4, 2)]
    captured_by_step = {
        step: {layer: torch.randn(1, seq_len, dim) for layer in layers} for step in range(steps)
    }

    committed = {t.position: t for t in tokens}
    time_rows: list[dict] = []
    for step in range(steps):
        update_committed_drift(committed, captured_by_step[step], layers, step, time_rows)

    expected, future_steps = _reference_drift(captured_by_step, layers, [_make_token(0, 0), _make_token(2, 1), _make_token(4, 2)])
    for token in committed.values():
        assert token.future_steps_observed == future_steps[token.position]
        for layer in layers:
            exp_max, exp_final = expected[token.position][layer]
            assert abs(token.max_drift_by_layer[layer] - exp_max) < 1e-6
            assert abs(token.final_drift_by_layer[layer] - exp_final) < 1e-6

    # Every tracked (token, step, layer) combination is logged exactly once.
    assert len(time_rows) == sum(future_steps.values()) * len(layers)

    # time_rows=None suppresses logging but still tracks drift.
    committed2 = {t.position: t for t in [_make_token(0, 0)]}
    for step in range(steps):
        update_committed_drift(committed2, captured_by_step[step], layers, step, None)
    assert committed2[0].max_drift_by_layer[1] == committed[0].max_drift_by_layer[1]


if __name__ == "__main__":
    test_vectorized_drift_matches_per_token_reference()
    print("ok")
