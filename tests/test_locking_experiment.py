from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from regret_remasking.data import Example
from regret_remasking.locking_experiment import make_policies, run_experiment
from regret_remasking.locking_manifest import make_manifest_record, write_frozen_jsonl


def inputs(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    entries = [make_manifest_record(Example(f"math-{i}", f"{i} + 2?", str(i+2), "math500"),
               split="smoke_math", seed=23, source_path=source, source_revision="fixture") for i in [2, 3]]
    manifest = tmp_path / "manifest.jsonl"
    write_frozen_jsonl(manifest, entries)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text("{}")
    return manifest, checkpoint


def fake_decode(calls):
    def decode(model, tokenizer, prompt, example_id, config, **kwargs):
        calls.append((example_id, config.arm))
        tokens = [10, 11] if config.arm == "B0" else [10, 12]
        rows = [{"example_id": example_id, "global_step": 0, "position": 1,
                 "top_token": 10, "final_token": 10, "stable": 1}]
        return SimpleNamespace(text="#### 4", token_ids=tokens, trace_rows=rows, events=[], nfe=2,
            forward_calls=2, latency_s=2.0 if config.arm == "B0" else 1.0, prompt_length=3,
            final_mask_count=0, immutable_token_violations=0, frozen_count=0,
            generated_row_updates_avoided=0, total_query_rows=10, gate_time_s=0.0, scorer_time_s=0.0)
    return decode


def test_rotated_runner_records_score_and_resume_without_duplicate_work(tmp_path, monkeypatch):
    from regret_remasking import locking_policy
    manifest, checkpoint = inputs(tmp_path)
    calls = []
    monkeypatch.setattr(locking_policy, "decode_locking_policy", fake_decode(calls))
    policies = make_policies(["B0", "R-dense"], [0.0], [0.95])
    kwargs = dict(checkpoint=checkpoint, gen_length=2, block_length=2, repeats=2, device="cpu",
                  warmup=False, model=object(), tokenizer=SimpleNamespace(mask_token_id=99), purpose="smoke")
    output = tmp_path / "run"
    summary = run_experiment([manifest], output, policies, **kwargs)
    assert summary["records"] == 8
    assert calls[:4] == [("math-2", "B0"), ("math-2", "R-dense"), ("math-2", "R-dense"), ("math-2", "B0")]
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert sum(row["correct"] for row in records) == 4
    with pytest.raises(FileExistsError):
        run_experiment([manifest], output, policies, **kwargs)
    assert run_experiment([manifest], output, policies, resume=True, **kwargs)["records"] == 8
    assert len(calls) == 8
    (checkpoint / "model.safetensors.index.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="provenance"):
        run_experiment([manifest], output, policies, resume=True, **kwargs)


def test_trace_aggregation_can_recover_after_last_record_and_reject_tampering(tmp_path, monkeypatch):
    from regret_remasking import locking_policy
    manifest, checkpoint = inputs(tmp_path)
    calls = []
    monkeypatch.setattr(locking_policy, "decode_locking_policy", fake_decode(calls))
    policies = make_policies(["B0"], [0.0], [0.95])
    kwargs = dict(checkpoint=checkpoint, gen_length=2, block_length=2, repeats=1, device="cpu",
                  warmup=False, collect_trace=True, model=object(), tokenizer=SimpleNamespace(mask_token_id=99))
    output = tmp_path / "trace-run"
    run_experiment([manifest], output, policies, **kwargs)
    (output / "traces.csv").unlink()
    run_experiment([manifest], output, policies, resume=True, **kwargs)
    with (output / "traces.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["example_id"] for r in rows} == {"math-2", "math-3"}
    assert len(rows) == 2 and len(calls) == 2
    trace = next((output / "traces").glob("*.csv"))
    trace.write_text(trace.read_text()+"corrupted\n")
    with pytest.raises(ValueError, match="trace artifact"):
        run_experiment([manifest], output, policies, resume=True, **kwargs)


def test_failed_decode_is_persisted_and_cannot_masquerade_as_complete(tmp_path, monkeypatch):
    from regret_remasking import locking_policy
    manifest, checkpoint = inputs(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("controlled failure")
    monkeypatch.setattr(locking_policy, "decode_locking_policy", broken)
    with pytest.raises(RuntimeError, match="controlled failure"):
        run_experiment([manifest], tmp_path / "run", make_policies(["B0"], [0.0], [0.95]),
            checkpoint=checkpoint, gen_length=2, block_length=2, repeats=1, device="cpu",
            warmup=False, model=object(), tokenizer=SimpleNamespace(mask_token_id=99))
    record = json.loads((tmp_path / "run/records.jsonl").read_text())
    assert record["status"] == "error" and "controlled failure" in record["error"]
    assert not (tmp_path / "run/completion.json").exists()


@pytest.mark.parametrize("arms,eps,tau", [(["B0", "B0"], [0.0], [0.95]),
    (["R-dense"], [float("nan")], [0.95]), (["B0"], [0.0], [1.1])])
def test_invalid_policy_spec_rejected(arms, eps, tau):
    with pytest.raises(ValueError):
        make_policies(arms, eps, tau)


def test_saved_policy_rejects_changed_predictor_before_loading_model(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    predictor = tmp_path / "predictor.joblib"
    predictor.write_bytes(b"first checkpoint")
    policies = make_policies(["T-adaptive"], [0.0], [0.95], str(predictor))
    predictor.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        run_experiment([manifest], tmp_path / "run", policies, checkpoint=checkpoint, device="cpu")
