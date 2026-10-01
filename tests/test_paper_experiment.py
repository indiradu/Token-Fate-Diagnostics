from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from regret_remasking.data import Example, build_prompt
from regret_remasking.locking_manifest import make_manifest_record, write_frozen_jsonl
from regret_remasking.paper_experiment import file_hash, make_paper_policies, run_paper_experiment, score_paper_generation
from regret_remasking.paper_manifests import prompt_hash as paper_prompt_hash


def inputs(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    entries = [
        make_manifest_record(
            Example(f"math-{i}", f"{i} + 2?", str(i + 2), "math500"),
            split="smoke_math",
            seed=23,
            source_path=source,
            source_revision="fixture",
        )
        for i in [2, 3]
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_frozen_jsonl(manifest, entries)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    return manifest, checkpoint


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def fake_decode(calls):
    def decode(model, tokenizer, prompt, example_id, config, **kwargs):
        assert kwargs["step_observer"] is None
        assert "trace_label" not in kwargs
        calls.append(
            {
                "example_id": example_id,
                "arm": config.arm,
                "prompt": prompt,
                "controller": kwargs.get("controller"),
                "projector": kwargs.get("projector"),
                "collect_events": kwargs.get("collect_events"),
                "allow_newly_unmasked": config.allow_newly_unmasked,
                "kl_space": config.kl_space,
                "representation_temperature": config.representation_temperature,
                "representation_layer": getattr(config, "representation_layer", None),
            }
        )
        token = 10 if config.arm == "B0" else 11
        return SimpleNamespace(
            text="#### 4",
            token_ids=[token, token + 1],
            events=[{"type": "token_commit"}] if kwargs.get("collect_events") else [],
            nfe=2,
            forward_calls=2,
            latency_s=2.0 if config.arm == "B0" else 1.0,
            prompt_length=3,
            final_mask_count=0,
            immutable_token_violations=0,
            frozen_count=0,
            frozen_prompt_count=0,
            generated_row_updates_avoided=0,
            total_query_rows=10,
            total_projection_rows=10,
            total_ffn_rows=10,
            total_output_head_rows=10,
            gate_time_s=0.0,
            scorer_time_s=0.0,
            controller_calls=1 if config.arm.startswith("T-") else 0,
            row_count_basis="fixture",
            kl_space=config.kl_space,
            representation_temperature=config.representation_temperature,
            attention_layout="compact",
            recording_mode="events" if kwargs.get("collect_events") else "timed_minimal",
            generated_row_updates_avoided_basis="fixture",
            policy_subtimers="fixture",
            representation_layer="final_normalized_pre_vocab" if getattr(config, "kl_space", "token") == "representation" else None,
            representation_feature_dim=4096 if getattr(config, "kl_space", "token") == "representation" else None,
        )

    return decode


def test_paper_runner_rotates_repeats_freezes_config_and_completes(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    calls = []
    policies = make_paper_policies(["B0", "S-sparse"], [0.0], [0.95], gen_length=2, block_length=2)
    output = tmp_path / "run"
    summary = run_paper_experiment(
        [manifest],
        output,
        policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=2,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode(calls),
        collect_events=True,
        purpose="smoke",
    )
    assert summary["records"] == 8
    expected_prompt = build_prompt("2 + 2?", "math500")
    assert calls[:4] == [
        {"example_id": "math-2", "arm": "B0", "prompt": expected_prompt, "controller": None, "projector": None, "collect_events": True, "allow_newly_unmasked": False, "kl_space": "token", "representation_temperature": 1.0, "representation_layer": None},
        {"example_id": "math-2", "arm": "S-sparse", "prompt": expected_prompt, "controller": None, "projector": None, "collect_events": True, "allow_newly_unmasked": False, "kl_space": "token", "representation_temperature": 1.0, "representation_layer": None},
        {"example_id": "math-2", "arm": "S-sparse", "prompt": expected_prompt, "controller": None, "projector": None, "collect_events": True, "allow_newly_unmasked": False, "kl_space": "token", "representation_temperature": 1.0, "representation_layer": None},
        {"example_id": "math-2", "arm": "B0", "prompt": expected_prompt, "controller": None, "projector": None, "collect_events": True, "allow_newly_unmasked": False, "kl_space": "token", "representation_temperature": 1.0, "representation_layer": None},
    ]
    records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["status"] == "ok" for row in records)
    assert all(row["total_output_head_rows"] == 10 and row["row_count_basis"] == "fixture" for row in records)
    assert (output / "completion.json").exists()
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert config["collect_events"] is True
    assert config["policies"][1]["epsilon"] == 0.0
    assert config["policies"][1]["allow_newly_unmasked"] is False
    assert config["policies"][1]["kl_space"] == "token"
    assert config["policies"][1]["representation_temperature"] == 1.0
    assert config["policies"][1]["representation_layer"] is None
    ablation_policies = [dict(policy) for policy in policies]
    ablation_policies[1]["allow_newly_unmasked"] = True
    output_ablation = tmp_path / "run-ablation"
    ablation_calls = []
    run_paper_experiment(
        [manifest],
        output_ablation,
        ablation_policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=1,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode(ablation_calls),
        purpose="smoke",
    )
    assert any(call["arm"] == "S-sparse" and call["allow_newly_unmasked"] is True for call in ablation_calls)
    with pytest.raises(FileExistsError):
        run_paper_experiment(
            [manifest],
            output,
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            repeats=2,
            device="cpu",
            warmup=False,
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=fake_decode([]),
        )
    assert run_paper_experiment(
        [manifest],
        output,
        policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=2,
        device="cpu",
        warmup=False,
        resume=True,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode([]),
        collect_events=True,
        purpose="smoke",
    )["records"] == 8
    (checkpoint / "model.safetensors.index.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        run_paper_experiment(
            [manifest],
            output,
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            repeats=2,
            device="cpu",
            warmup=False,
            resume=True,
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=fake_decode([]),
            collect_events=True,
            purpose="smoke",
        )


def test_learned_policy_loads_hashed_artifacts_and_passes_no_labels(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    controller_path = tmp_path / "controller.pt"
    projector_path = tmp_path / "projector.pt"
    projector_path.write_bytes(b"projector")
    torch.save({"projection_sha256": file_hash(projector_path), "config": {}, "state_dict": {}}, controller_path)
    policies = make_paper_policies(
        ["T-fixed", "T-adaptive"],
        [0.0],
        [0.95],
        gen_length=2,
        block_length=2,
        controller_checkpoint=controller_path,
        projector_checkpoint=projector_path,
    )
    calls = []
    loaded = []
    output = tmp_path / "learned-run"
    run_paper_experiment(
        [manifest],
        output,
        policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=1,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode(calls),
        controller_loader=lambda path, device: loaded.append(("controller", path, device)) or "controller-object",
        projector_loader=lambda path, device: loaded.append(("projector", path, device)) or "projector-object",
    )
    assert {tuple(item[:1]) for item in loaded} == {("controller",), ("projector",)}
    assert all(call["controller"] == "controller-object" for call in calls)
    assert all(call["projector"] == "projector-object" for call in calls)
    records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["controller_checkpoint_sha256"] for row in records} == {policies[0]["controller_checkpoint_sha256"]}
    controller_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="controller_checkpoint hash"):
        run_paper_experiment(
            [manifest],
            tmp_path / "changed-policy",
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            device="cpu",
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=fake_decode([]),
        )


def test_controller_payload_must_bind_to_projector_hash(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    controller_path = tmp_path / "controller.pt"
    projector_path = tmp_path / "projector.pt"
    projector_path.write_bytes(b"projector")
    torch.save({"projection_sha256": "wrong", "config": {}, "state_dict": {}}, controller_path)
    policies = make_paper_policies(
        ["T-fixed"],
        [0.0],
        [0.95],
        gen_length=2,
        block_length=2,
        controller_checkpoint=controller_path,
        projector_checkpoint=projector_path,
    )
    with pytest.raises(ValueError, match="projection_sha256"):
        run_paper_experiment(
            [manifest],
            tmp_path / "pair-mismatch",
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            device="cpu",
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=fake_decode([]),
        )


def test_new_paper_manifest_uses_chat_hash_prompt_text_and_kodcode_metadata(tmp_path, monkeypatch):
    source = tmp_path / "kodcode.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    prompt_text = "def add_one(x):\n    \"\"\"Add one.\"\"\"\n"
    row = {
        "record_id": "selection:code:kodcode:task-1",
        "split": "selection",
        "family": "code",
        "dataset_name": "kodcode",
        "dataset": "kodcode",
        "example_id": "task-1",
        "source_id": "kodcode:task-1",
        "source_path": str(source),
        "source_revision": "45d9e3b725e585462fee2bb026e5e32a2eb0a8d7",
        "source_sha256": file_hash(source),
        "prompt_text": prompt_text,
        "prompt_hash": paper_prompt_hash(prompt_text),
        "chat_prompt_hash": "chat-rendered-hash",
        "question": prompt_text,
        "gold_answer": "",
        "metadata": {"test": "def check(fn):\n    assert fn(1) == 2\n", "entry_point": "add_one"},
        "seed": 23,
        "teacher_config": {"gen_length": 256},
    }
    manifest = tmp_path / "paper_manifest.jsonl"
    write_jsonl(manifest, [row])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    calls = []
    seen = []

    def fake_score(generation, example):
        seen.append((generation, example.dataset, example.metadata["entry_point"], example.question))
        return True, "passed"

    monkeypatch.setattr("regret_remasking.paper_experiment.score_generation_with_status", fake_score)
    policies = make_paper_policies(["B0"], [0.0], [0.95], gen_length=2, block_length=2)
    output = tmp_path / "paper-run"
    run_paper_experiment(
        [manifest],
        output,
        policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=1,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode(calls),
    )
    assert calls[0]["prompt"] == prompt_text
    assert seen == [("#### 4", "humaneval", "add_one", prompt_text)]
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert config["examples"][0]["content_hash"] == "chat-rendered-hash"
    record = json.loads((output / "records.jsonl").read_text(encoding="utf-8"))
    assert record["content_hash"] == "chat-rendered-hash"
    assert score_paper_generation("anything", {**row, "metadata": {}}) == (None, "missing_code_tests")


def test_representation_kl_policy_serializes_and_passes_decode_config(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    policies = make_paper_policies(
        ["S-sparse"],
        [0.005],
        [0.95],
        gen_length=2,
        block_length=2,
        kl_space="representation",
        representation_temperature=0.7,
    )
    assert policies[0]["policy_id"] == "S-sparse-eps-0.005-rep-kl"
    assert policies[0]["representation_layer"] == "final_normalized_pre_vocab"
    calls = []
    output = tmp_path / "rep-run"
    run_paper_experiment(
        [manifest],
        output,
        policies,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=1,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode(calls),
    )
    assert calls[0]["kl_space"] == "representation"
    assert calls[0]["representation_temperature"] == 0.7
    assert calls[0]["representation_layer"] is None
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    record = json.loads((output / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert config["policies"][0]["kl_space"] == "representation"
    assert config["policies"][0]["representation_temperature"] == 0.7
    assert config["policies"][0]["representation_layer"] == "final_normalized_pre_vocab"
    assert record["kl_space"] == "representation"
    assert record["representation_temperature"] == 0.7
    assert record["representation_layer"] == "final_normalized_pre_vocab"
    assert record["representation_feature_dim"] == 4096


def test_legacy_policy_config_defaults_kl_fields(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    legacy = make_paper_policies(["B0"], [0.0], [0.95], gen_length=2, block_length=2)
    for field in ("kl_space", "representation_temperature", "representation_layer"):
        legacy[0].pop(field)
    output = tmp_path / "legacy-run"
    run_paper_experiment(
        [manifest],
        output,
        legacy,
        checkpoint=checkpoint,
        gen_length=2,
        repeats=1,
        device="cpu",
        warmup=False,
        model=object(),
        tokenizer=SimpleNamespace(mask_token_id=99),
        decoder=fake_decode([]),
    )
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    record = json.loads((output / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert config["policies"][0]["kl_space"] == "token"
    assert config["policies"][0]["representation_temperature"] == 1.0
    assert config["policies"][0]["representation_layer"] is None
    assert record["kl_space"] == "token"
    assert record["representation_temperature"] == 1.0
    assert record["representation_layer"] is None
    assert record["representation_feature_dim"] is None


def test_decoder_result_kl_metadata_must_match_policy(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    policies = make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2, kl_space="representation")

    def mismatched(model, tokenizer, prompt, example_id, config, **kwargs):
        result = fake_decode([])(model, tokenizer, prompt, example_id, config, **kwargs)
        result.kl_space = "token"
        return result

    with pytest.raises(ValueError, match="kl_space"):
        run_paper_experiment(
            [manifest],
            tmp_path / "mismatch-run",
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            repeats=1,
            device="cpu",
            warmup=False,
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=mismatched,
        )


def test_invalid_kl_policy_fields_rejected(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    policy = make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2)[0]
    policy["kl_space"] = "logits"
    with pytest.raises(ValueError, match="kl_space"):
        run_paper_experiment([manifest], tmp_path / "bad-space", [policy], checkpoint=checkpoint, gen_length=2, device="cpu", model=object(), tokenizer=SimpleNamespace(mask_token_id=99), decoder=fake_decode([]))
    policy = make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2)[0]
    policy["representation_temperature"] = 0.0
    with pytest.raises(ValueError, match="representation_temperature"):
        run_paper_experiment([manifest], tmp_path / "bad-temp", [policy], checkpoint=checkpoint, gen_length=2, device="cpu", model=object(), tokenizer=SimpleNamespace(mask_token_id=99), decoder=fake_decode([]))
    policy = make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2, kl_space="representation")[0]
    policy["representation_layer"] = "hidden_logits"
    with pytest.raises(ValueError, match="representation_layer"):
        run_paper_experiment([manifest], tmp_path / "bad-layer", [policy], checkpoint=checkpoint, gen_length=2, device="cpu", model=object(), tokenizer=SimpleNamespace(mask_token_id=99), decoder=fake_decode([]))

def test_failed_decode_persists_without_completion(tmp_path):
    manifest, checkpoint = inputs(tmp_path)
    policies = make_paper_policies(["B0"], [0.0], [0.95], gen_length=2, block_length=2)

    def broken(*args, **kwargs):
        raise RuntimeError("controlled failure")

    with pytest.raises(RuntimeError, match="controlled failure"):
        run_paper_experiment(
            [manifest],
            tmp_path / "run",
            policies,
            checkpoint=checkpoint,
            gen_length=2,
            repeats=1,
            device="cpu",
            warmup=False,
            model=object(),
            tokenizer=SimpleNamespace(mask_token_id=99),
            decoder=broken,
        )
    record = json.loads((tmp_path / "run/records.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "error" and "controlled failure" in record["error"]
    assert not (tmp_path / "run/completion.json").exists()


@pytest.mark.parametrize(
    "arms,eps,tau",
    [(["B0", "B0"], [0.0], [0.95]), (["S-sparse"], [float("nan")], [0.95]), (["B0"], [0.0], [1.1])],
)
def test_invalid_policy_specs_rejected(arms, eps, tau):
    with pytest.raises(ValueError):
        make_paper_policies(arms, eps, tau, gen_length=2, block_length=2)


def test_invalid_generated_kl_policy_specs_rejected():
    with pytest.raises(ValueError, match="kl_space"):
        make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2, kl_space="logits")
    with pytest.raises(ValueError, match="representation_temperature"):
        make_paper_policies(["S-sparse"], [0.005], [0.95], gen_length=2, block_length=2, representation_temperature=0.0)
    b0 = make_paper_policies(["B0"], [0.0], [0.95], gen_length=2, block_length=2, kl_space="representation")[0]
    assert b0["kl_space"] == "representation"
    assert b0["representation_layer"] is None
