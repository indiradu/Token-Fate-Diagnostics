from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.paper_trace import (
    LLaDA_LAST_THREE_INDICES,
    PAPER_FEATURE_DIM,
    STATE_EOT,
    STATE_GEN,
    STATE_LOCKED,
    STATE_PROMPT,
    ControllerConfig,
    HiddenProjector,
    PaperTraceController,
    PaperTraceExampleDataset,
    ProjectorConfig,
    build_training_frame_index,
    choose_label_blind_crop,
    choose_validation_window_end,
    collate_paper_trace_batch,
    encode_hidden,
    evaluate_controller,
    fbeta_from_counts,
    fit_projector,
    paper_trace_loss,
    read_index_json,
    save_paper_trace_example,
    sha256_file,
    train_controller,
    validate_train_validation_split,
)


def _make_projector(hidden_dim: int = 8) -> HiddenProjector:
    projector = HiddenProjector(ProjectorConfig(hidden_dim=hidden_dim))
    projector.eval()
    return projector


def _write_example(
    root: pathlib.Path,
    projection_sha256: str,
    labels: np.ndarray | None = None,
    example_id: str = "ex",
) -> pathlib.Path:
    features = np.zeros((2, 6, PAPER_FEATURE_DIM), dtype=np.float16)
    features[0, :, 0] = np.linspace(0, 1, 6, dtype=np.float16)
    features[1, :, 1] = np.linspace(1, 0, 6, dtype=np.float16)
    states = np.array(
        [
            [STATE_PROMPT, STATE_PROMPT, STATE_GEN, STATE_GEN, STATE_LOCKED, STATE_EOT],
            [STATE_PROMPT, STATE_PROMPT, STATE_LOCKED, STATE_GEN, STATE_GEN, STATE_EOT],
        ],
        dtype=np.uint8,
    )
    if labels is None:
        labels = np.array([[0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
    return save_paper_trace_example(
        root,
        features,
        states,
        labels,
        {
            "example_id": example_id,
            "prompt_length": 2,
            "gen_length": 4,
            "selected_steps": [0, 1],
            "selected_ids": [0, 1],
            "source": "unit",
            "projection_sha256": projection_sha256,
            "split": "fit",
            "content_hash": f"hash-{example_id}",
        },
    )


def test_projector_roundtrip_polarity_and_train_only_stats(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(0)
    train_hidden = torch.randn(24, 3, 8)
    heldout_hidden = torch.randn(24, 3, 8) + 100.0
    checkpoint = tmp_path / "projector.pt"

    projector = fit_projector({"hidden": train_hidden.to(torch.bfloat16)}, checkpoint, steps=3, seed=3, batch_size=8)
    loaded = HiddenProjector.load(checkpoint)
    assert loaded.config.hidden_indices == LLaDA_LAST_THREE_INDICES
    rounded_train_hidden = train_hidden.to(torch.bfloat16).float()
    assert loaded.state_mean.shape == (3, 8)
    assert torch.allclose(loaded.state_mean, rounded_train_hidden.mean(dim=0), atol=1e-6)
    assert not torch.allclose(loaded.state_mean, heldout_hidden.mean(dim=0), atol=1.0)

    hidden_tuple = tuple(train_hidden[:2, index] for index in range(3))
    features = projector.encode_hidden(hidden_tuple)
    reloaded_features = encode_hidden(hidden_tuple, checkpoint)
    assert features.shape == (2, PAPER_FEATURE_DIM)
    assert features.dtype == torch.float32
    assert torch.allclose(features, features.to(torch.float16).to(torch.float32))
    assert torch.allclose(features, reloaded_features)

    changed = projector.encode_hidden(tuple((train_hidden[:2, index] + (10.0 if index == 2 else 0.0)) for index in range(3)))
    assert not torch.allclose(features, changed)

    bf16_tuple = tuple((train_hidden[:2, index] + 0.003).to(torch.bfloat16) for index in range(3))
    from_bf16 = projector.encode_hidden(bf16_tuple)
    from_rounded_float = projector.encode_hidden(tuple(t.float() for t in bf16_tuple))
    assert torch.allclose(from_bf16, from_rounded_float)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert np.isfinite(payload["initial_reconstruction_loss"])
    assert np.isfinite(payload["final_reconstruction_loss"])


def test_projector_rejects_wrong_hidden_tuple() -> None:
    projector = _make_projector()
    with pytest.raises(ValueError, match="exactly tuple"):
        projector.encode_hidden((torch.zeros(2, 8), torch.zeros(2, 8)))
    with pytest.raises(ValueError, match="identical shapes"):
        projector.encode_hidden((torch.zeros(2, 8), torch.zeros(3, 8), torch.zeros(2, 8)))
    with pytest.raises(ValueError, match="hidden_count"):
        fit_projector({"hidden": torch.zeros(2, 3, 8), "hidden_count": 32}, pathlib.Path("/tmp/unused-paper-projector.pt"))


def test_controller_dynamic_threshold_context_padding_and_loss() -> None:
    torch.manual_seed(5)
    controller = PaperTraceController(ControllerConfig(max_positions=16, dropout=0.0))
    controller.eval()
    features = torch.zeros(2, 4, PAPER_FEATURE_DIM)
    features[:, 1, 0] = 1.0
    features[0, 2, 10] = 2.0
    features[1, 2, 10] = -2.0
    valid_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    position_ids = torch.arange(4).repeat(2, 1)

    first = controller(features, valid_mask, position_ids)
    second = controller(features, valid_mask, position_ids)
    assert first["token_logits"].shape == (2, 4)
    assert first["threshold_logit"].shape == (2,)
    assert first["margins"].shape == (2, 4)
    assert torch.allclose(first["margins"], second["margins"])
    assert first["margins"][0, 1].item() != pytest.approx(first["margins"][1, 1].item())

    labels = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.float32)
    active_mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
    loss = paper_trace_loss(first, labels, active_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in controller.parameters())

    batch = collate_paper_trace_batch(
        [
            {
                "features": features[0, :3],
                "valid_mask": torch.ones(3, dtype=torch.bool),
                "labels": labels[0, :3],
                "active_mask": active_mask[0, :3],
                "position_ids": torch.arange(3),
            },
            {
                "features": features[1, :2],
                "valid_mask": torch.ones(2, dtype=torch.bool),
                "labels": labels[1, :2],
                "active_mask": active_mask[1, :2],
                "position_ids": torch.arange(2),
            },
        ]
    )
    assert batch["features"].shape == (2, 3, PAPER_FEATURE_DIM)
    assert not bool(batch["valid_mask"][1, 2])


def test_paper_trace_loss_weights_only_token_prediction_errors_and_active_count_denominator() -> None:
    margins = torch.tensor([[2.0, -2.0, 2.0, -2.0, 10.0]], requires_grad=True)
    threshold = torch.tensor([3.0], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0]])
    active_mask = torch.tensor([[True, True, True, True, False]])
    outputs = {
        "margins": margins,
        "threshold_logit": threshold,
        "token_logits": margins + threshold[:, None],
    }

    loss = paper_trace_loss(outputs, labels, active_mask, misclassified_weight=4.0, tau_l2=0.001)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(margins[:, :4], labels[:, :4], reduction="none")
    expected = (bce[0, 0] + bce[0, 1] + 4.0 * bce[0, 2] + 4.0 * bce[0, 3]) / 4.0 + 0.001 * threshold.pow(2).mean()
    assert loss.item() == pytest.approx(expected.item())

    loss.backward()
    assert margins.grad is not None
    assert margins.grad[0, 0].abs() > 0
    assert margins.grad[0, 1].abs() > 0
    assert margins.grad[0, 2].abs() > margins.grad[0, 0].abs()
    assert margins.grad[0, 3].abs() > margins.grad[0, 1].abs()
    assert margins.grad[0, 4].item() == pytest.approx(0.0)
    assert threshold.grad is not None
    assert threshold.grad.item() == pytest.approx(2.0 * 0.001 * 3.0)


def test_crop_is_label_blind_and_dataset_masks_active_gen_only(tmp_path: pathlib.Path) -> None:
    projection = "abc123"
    labels_a = np.array([[0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
    labels_b = 1 - labels_a
    labels_b[:, :2] = 0
    path_a = _write_example(tmp_path / "a", projection, labels=labels_a, example_id="a")
    path_b = _write_example(tmp_path / "b", projection, labels=labels_b, example_id="b")
    index_a = build_training_frame_index([path_a], projection_sha256=projection)
    index_b = build_training_frame_index([path_b], projection_sha256=projection)
    index_b[0]["frame"] = index_a[0]["frame"]

    item_a = PaperTraceExampleDataset([index_a[0]], seed=99)[0]
    item_b = PaperTraceExampleDataset([index_b[0]], seed=99)[0]
    assert torch.equal(item_a["features"], item_b["features"])
    assert torch.equal(item_a["valid_mask"], item_b["valid_mask"])
    assert torch.equal(item_a["position_ids"], item_b["position_ids"])
    assert not torch.equal(item_a["labels"], item_b["labels"])
    assert item_a["active_mask"].tolist()[:2] == [False, False]
    assert any(item_a["active_mask"].tolist()[2:])

    dataset = PaperTraceExampleDataset([index_a[0]], seed=99)
    epoch0 = dataset[0]["features"]
    changed_epoch = False
    for epoch in range(1, 8):
        dataset.set_epoch(epoch)
        epoch_item = dataset[0]["features"]
        changed_epoch = changed_epoch or epoch0.shape != epoch_item.shape or not torch.equal(epoch0, epoch_item)
    assert changed_epoch

    lru_dataset = PaperTraceExampleDataset([index_a[0], index_b[0]], seed=1, max_cached_dirs=1)
    _ = lru_dataset[0]
    assert len(lru_dataset._cache) == 1
    _ = lru_dataset[1]
    assert len(lru_dataset._cache) == 1
    assert str(path_b.resolve()) in lru_dataset._cache

    state_row = np.array([STATE_PROMPT, STATE_PROMPT, STATE_LOCKED, STATE_GEN, STATE_EOT], dtype=np.uint8)
    end = choose_label_blind_crop(state_row, prompt_length=2, gen_length=3, rng=__import__("random").Random(1))
    assert end >= 4


def test_validation_window_is_label_blind_and_ignores_future_targets(tmp_path: pathlib.Path) -> None:
    projection = "window-hash"
    features = np.zeros((1, 9, PAPER_FEATURE_DIM), dtype=np.float16)
    states = np.array([[STATE_PROMPT, STATE_PROMPT, STATE_LOCKED, STATE_GEN, STATE_GEN, STATE_GEN, STATE_GEN, STATE_EOT, STATE_EOT]], dtype=np.uint8)
    labels_a = np.array([[0, 0, 0, 1, 1, 0, 1, 0, 0]], dtype=np.uint8)
    labels_b = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    metadata = {
        "example_id": "window",
        "prompt_length": 2,
        "gen_length": 6,
        "selected_steps": [0],
        "selected_ids": [0],
        "source": "unit",
        "projection_sha256": projection,
        "split": "validation",
    }
    path_a = save_paper_trace_example(tmp_path / "window-a", features, states, labels_a, metadata)
    path_b = save_paper_trace_example(tmp_path / "window-b", features, states, labels_b, {**metadata, "example_id": "window-b"})
    index_a = build_training_frame_index([path_a], projection_sha256=projection, split="validation")
    index_b = build_training_frame_index([path_b], projection_sha256=projection, split="validation")

    assert choose_validation_window_end(states[0], prompt_length=2, gen_length=6, validation_window=2) == 5
    item_a = PaperTraceExampleDataset(index_a, crop=False, validation_window=2)[0]
    item_b = PaperTraceExampleDataset(index_b, crop=False, validation_window=2)[0]
    assert item_a["features"].shape[0] == 5
    assert torch.equal(item_a["features"], item_b["features"])
    assert torch.equal(item_a["valid_mask"], item_b["valid_mask"])
    assert torch.equal(item_a["position_ids"], item_b["position_ids"])
    assert item_a["active_mask"].tolist() == [False, False, False, True, True]
    assert item_a["labels"].tolist() != labels_a[0].astype(np.float32).tolist()
    assert item_a["labels"].shape[0] == 5
    assert item_b["labels"].shape[0] == 5


def test_schema_provenance_rejects_mismatches(tmp_path: pathlib.Path) -> None:
    path = _write_example(tmp_path / "ex", "good")
    metadata_path = path / "metadata.json"
    rows = build_training_frame_index([path], projection_sha256="good", split="fit")
    assert len(rows) == 2 and rows[0]["projection_sha256"] == "good"
    with pytest.raises(ValueError, match="projection_sha256 mismatch"):
        build_training_frame_index([path], projection_sha256="bad")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hidden_indices"] = [29, 30, 31]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden_indices"):
        build_training_frame_index([path], projection_sha256="good")

    metadata["hidden_indices"] = list(LLaDA_LAST_THREE_INDICES)
    metadata["hidden_count"] = 32
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden_count"):
        build_training_frame_index([path], projection_sha256="good")


def test_save_rejects_nonfinite_features_after_storage_cast(tmp_path: pathlib.Path) -> None:
    states = np.array([[STATE_PROMPT, STATE_GEN]], dtype=np.uint8)
    labels = np.array([[0, 1]], dtype=np.uint8)
    metadata = {
        "prompt_length": 1,
        "gen_length": 1,
        "selected_steps": [0],
        "selected_ids": [0],
        "source": "unit",
        "projection_sha256": "hash",
    }
    nan_features = np.zeros((1, 2, PAPER_FEATURE_DIM), dtype=np.float32)
    nan_features[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        save_paper_trace_example(tmp_path / "nan", nan_features, states, labels, metadata)
    assert not (tmp_path / "nan" / "features.npy").exists()

    overflow_features = np.zeros((1, 2, PAPER_FEATURE_DIM), dtype=np.float32)
    overflow_features[0, 1, 0] = 1.0e10
    with pytest.raises(ValueError, match="nonfinite"):
        save_paper_trace_example(tmp_path / "overflow", overflow_features, states, labels, metadata)
    assert not (tmp_path / "overflow" / "features.npy").exists()


def test_training_cli_path_runs_real_optimization_and_saves_best(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(0)
    projector_path = tmp_path / "projector.pt"
    fit_projector({"hidden": torch.randn(8, 3, 8)}, projector_path, steps=1, seed=1, batch_size=4)
    projection_hash = sha256_file(projector_path)
    fit_dir = _write_example(tmp_path / "fit", projection_hash, example_id="fit")
    val_dir = _write_example(tmp_path / "val", projection_hash, example_id="val")
    fit_index = build_training_frame_index([fit_dir], projection_sha256=projection_hash, split="fit")
    val_index = build_training_frame_index([val_dir], projection_sha256=projection_hash, split="validation")
    with pytest.raises(ValueError, match="content hash"):
        validate_train_validation_split(fit_index, [{**val_index[0], "content_hash": fit_index[0]["content_hash"]}])

    result = train_controller(
        fit_index=fit_index,
        val_index=val_index,
        output_dir=tmp_path / "out",
        projector_checkpoint=projector_path,
        steps=1,
        batch_size=2,
        validate_every=1,
        seed=4,
    )
    assert result.checkpoint_path.exists()
    assert result.config_path.exists()
    assert result.metrics_path.exists()
    assert result.provenance_path.exists()
    controller = PaperTraceController.load(result.checkpoint_path)
    controller.eval()
    item = PaperTraceExampleDataset(val_index, crop=False)[0]
    batch = collate_paper_trace_batch([item])
    with torch.no_grad():
        outputs = controller(batch["features"], batch["valid_mask"], batch["position_ids"])
    assert outputs["margins"].shape[-1] == batch["features"].shape[1]

    index_json = tmp_path / "fit_index.json"
    index_json.write_text(json.dumps(fit_index), encoding="utf-8")
    loaded_index = read_index_json(index_json, projection_sha256=projection_hash, split="fit")
    assert loaded_index[0]["path"] == str(fit_dir.resolve())

    dirs_json = tmp_path / "fit_dirs.json"
    dirs_json.write_text(json.dumps([str(fit_dir)]), encoding="utf-8")
    loaded_from_dirs = read_index_json(dirs_json, projection_sha256=projection_hash, split="fit")
    assert len(loaded_from_dirs) == 2
    assert loaded_from_dirs[0]["example_id"] == "fit"

    dict_dirs_json = tmp_path / "fit_dict_dirs.json"
    dict_dirs_json.write_text(json.dumps([{"path": str(fit_dir), "split": "fit"}]), encoding="utf-8")
    loaded_from_dict_dirs = read_index_json(dict_dirs_json, projection_sha256=projection_hash, split="fit")
    assert len(loaded_from_dict_dirs) == 2


def test_validation_fbeta_aggregates_token_counts_not_per_frame_mean() -> None:
    class FixedController:
        def eval(self) -> None:
            return None

        def __call__(self, features: torch.Tensor, valid_mask: torch.Tensor, position_ids: torch.Tensor):
            del valid_mask, position_ids
            margins = features[:, :, 0]
            return {
                "token_logits": margins,
                "threshold_logit": torch.zeros(features.shape[0]),
                "margins": margins,
            }

    items = [
        {
            "features": torch.tensor([[10.0] + [0.0] * (PAPER_FEATURE_DIM - 1), [10.0] + [0.0] * (PAPER_FEATURE_DIM - 1)]),
            "valid_mask": torch.ones(2, dtype=torch.bool),
            "labels": torch.tensor([1.0, 0.0]),
            "active_mask": torch.ones(2, dtype=torch.bool),
            "position_ids": torch.arange(2),
        },
        {
            "features": torch.tensor([[10.0] + [0.0] * (PAPER_FEATURE_DIM - 1), [10.0] + [0.0] * (PAPER_FEATURE_DIM - 1)]),
            "valid_mask": torch.ones(2, dtype=torch.bool),
            "labels": torch.tensor([1.0, 1.0]),
            "active_mask": torch.ones(2, dtype=torch.bool),
            "position_ids": torch.arange(2),
        },
    ]

    class ListDataset:
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int):
            return items[index]

    score, loss = evaluate_controller(FixedController(), ListDataset(), batch_size=2)
    assert np.isfinite(loss)
    assert score == pytest.approx(fbeta_from_counts(tp=3, fp=1, fn=0))
