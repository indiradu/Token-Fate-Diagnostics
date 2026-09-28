from __future__ import annotations

import numpy as np

from train_commitment_safety_predictor import _selection_metrics


def test_selection_metrics_reports_zero_selected_as_fail_closed() -> None:
    metrics = _selection_metrics(
        np.asarray([1, 0, 1]), np.asarray([0.9, 0.8, 0.7]), 1.000001
    )
    assert metrics["selected"] == 0
    assert metrics["unsafe"] == 0
    assert metrics["coverage"] == 0.0
    assert metrics["unsafe_rate_upper95"] == 1.0

