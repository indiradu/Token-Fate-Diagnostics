from __future__ import annotations

import json

import pytest

from train_tracelock_llada import audit_trace_manifests


def test_trace_manifest_audit_enforces_preregistered_mix(tmp_path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    records = [
        {"sample_id": "gsm8k-1", "source": "gsm8k", "saved_steps": 1},
        {
            "sample_id": "natural_questions-1",
            "source": "natural_questions",
            "saved_steps": 1,
        },
    ]
    (tmp_path / "manifest_part.json").write_text(json.dumps(records))
    (tmp_path / "train" / "gsm8k-1_step_0000.pt").touch()
    (tmp_path / "val" / "natural_questions-1_step_0000.pt").touch()
    audit = audit_trace_manifests(
        tmp_path, "gsm8k=1,natural_questions=1"
    )
    assert audit["unique_examples"] == 2
    assert audit["manifest_step_samples"] == 2
    with pytest.raises(ValueError, match="preregistration"):
        audit_trace_manifests(tmp_path, "gsm8k=2,natural_questions=0")

