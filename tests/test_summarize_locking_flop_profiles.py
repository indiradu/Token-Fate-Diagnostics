import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from locking_flop_accounting import ControllerConfig, ProjectorConfig, backbone_flops, controller_flops, load_model_config
from summarize_locking_flop_profiles import summarize


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture_files(tmp_path):
    model_path = tmp_path / "model.json"
    write_json(model_path, dict(d_model=4, n_layers=2, n_heads=2, n_kv_heads=2, mlp_hidden_size=8, vocab_size=12))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(dataset="math", example_id="one")) + "\n")
    row = dict(dataset="math", example_id="one", policy_id="T-adaptive-0.95", arm="T-adaptive",
               source_sha256="source", status="ok", gen_length=256, prompt_length=3,
               nfe=128, forward_calls=128, controller_calls=128, controller_prefix_lengths=[259] * 128,
               total_query_rows=259 * 128, total_projection_rows=259 * 128,
               total_ffn_rows=259 * 128, total_output_head_rows=259 * 128,
               row_count_basis="per_layer_equivalent; output_head_count_is_actual",
               final_mask_count=0, immutable_token_violations=0, parity_checked=True)
    row["projector_config"] = dict(hidden_dim=4, state_hidden_dim=8, state_dim=256, delta_hidden_dim=4, delta_dim=32)
    row["controller_config"] = dict(model_dim=8, layers=1, ff_dim=16)
    row["backbone_counts"] = backbone_flops(row, load_model_config(model_path))
    row["learned_counts"] = controller_flops(row["controller_prefix_lengths"], ProjectorConfig(4, 8, 256, 4, 32), ControllerConfig(8, 1, 16))
    row["dispatch_total_flops"] = row["backbone_counts"]["total_with_lm_head_flops"] + row["learned_counts"]["controller_total_flops"]
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    path = profile_dir / "profiles.jsonl"
    path.write_text(json.dumps(row) + "\n")
    write_json(profile_dir / "completion.json", dict(complete=True, records=1, profiles_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(profile_dir / "provenance.json", dict(policy_ids=["T-adaptive-0.95"], source=dict(sha256="source"), manifests={str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    return profile_dir, model_path


def test_complete_profile_reports_backbone_and_controller_without_fake_runtime(tmp_path):
    profile_dir, model_path = fixture_files(tmp_path)
    report = summarize(profile_dir, model_path, tmp_path / "out")
    overall = report["aggregate"][-1]
    assert report["complete"] and report["cases"] == 1
    assert overall["backbone_reduction_pct"] == 50
    assert overall["extended_reduction_pct"] < overall["backbone_reduction_pct"]
    assert overall["measured_speedup_pct"] is None and overall["correctness"] is None


@pytest.mark.parametrize("corruption", ["checksum", "duplicate", "multiple_policies"])
def test_invalid_profile_or_policy_pooling_is_rejected(tmp_path, corruption):
    profile_dir, model_path = fixture_files(tmp_path)
    if corruption == "multiple_policies":
        p = profile_dir / "provenance.json"
        data = json.loads(p.read_text())
        data["policy_ids"].append("T-adaptive-0.99")
        write_json(p, data)
    else:
        p = profile_dir / "profiles.jsonl"
        p.write_text(p.read_text() * 2)
        if corruption == "duplicate":
            c = profile_dir / "completion.json"
            data = json.loads(c.read_text())
            data.update(records=2, profiles_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            write_json(c, data)
    with pytest.raises(ValueError):
        summarize(profile_dir, model_path, tmp_path / "out")
