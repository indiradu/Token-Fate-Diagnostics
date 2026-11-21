from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


PAPER_TRACE_SCHEMA_VERSION = 1
PAPER_FEATURE_DIM = 832
STATE_PROMPT = 0
STATE_LOCKED = 1
STATE_GEN = 2
STATE_EOT = 3
LLaDA_HIDDEN_COUNT = 33
LLaDA_LAST_THREE_INDICES = (30, 31, 32)
LLaDA_LAST_THREE_SEMANTICS = (
    "raw_block_30",
    "raw_block_31",
    "final_normalized_block_32",
)
_VALIDATION_CACHE: dict[tuple[str, str | None, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class ProjectorConfig:
    hidden_dim: int
    state_dim: int = 256
    delta_dim: int = 32
    state_hidden_dim: int = 1024
    delta_hidden_dim: int = 256
    expected_hidden_count: int = LLaDA_HIDDEN_COUNT
    hidden_indices: tuple[int, int, int] = LLaDA_LAST_THREE_INDICES
    hidden_semantics: tuple[str, str, str] = LLaDA_LAST_THREE_SEMANTICS
    feature_dim: int = PAPER_FEATURE_DIM


@dataclass(frozen=True)
class ControllerConfig:
    feature_dim: int = PAPER_FEATURE_DIM
    model_dim: int = 384
    layers: int = 3
    heads: int = 8
    ff_dim: int = 768
    dropout: float = 0.1
    max_positions: int = 4096
    cutoff: float = 0.95
    hidden_indices: tuple[int, int, int] = LLaDA_LAST_THREE_INDICES
    hidden_semantics: tuple[str, str, str] = LLaDA_LAST_THREE_SEMANTICS


@dataclass(frozen=True)
class FrameIndexRow:
    path: str
    example_id: str
    frame: int
    prompt_length: int
    gen_length: int
    projection_sha256: str
    split: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    provenance_path: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_native_hidden_metadata(payload: dict[str, Any], source: str) -> None:
    if int(payload.get("expected_hidden_count", payload.get("hidden_count", -1))) != LLaDA_HIDDEN_COUNT:
        raise ValueError(f"{source} hidden_count must be native LLaDA {LLaDA_HIDDEN_COUNT}")
    if list(payload.get("hidden_indices", [])) != list(LLaDA_LAST_THREE_INDICES):
        raise ValueError(f"{source} hidden_indices must be {list(LLaDA_LAST_THREE_INDICES)}")
    if list(payload.get("hidden_semantics", [])) != list(LLaDA_LAST_THREE_SEMANTICS):
        raise ValueError(f"{source} hidden_semantics mismatch")


def _ensure_last_three(last_three: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(last_three) != 3:
        raise ValueError("LLaDA projector expects exactly tuple(out.hidden_states[-3:])")
    h0, h1, h2 = last_three
    if h0.shape != h1.shape or h1.shape != h2.shape:
        raise ValueError(f"last_three hidden states must have identical shapes, got {[tuple(t.shape) for t in last_three]}")
    if h0.ndim not in {2, 3}:
        raise ValueError(f"hidden states must be [M,H] or [B,L,H], got {tuple(h0.shape)}")
    return h0, h1, h2


def _flatten_hidden(last_three: Sequence[torch.Tensor]) -> torch.Tensor:
    h0, h1, h2 = _ensure_last_three(last_three)
    stacked = torch.stack([h0, h1, h2], dim=-2)
    return stacked.reshape(-1, 3, stacked.shape[-1])


def _roundtrip_features(features: torch.Tensor) -> torch.Tensor:
    return features.to(torch.float16).to(torch.float32)


class _AutoEncoderMLP(nn.Module):
    def __init__(self, in_dim: int, encoded_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoded_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def reconstruct(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


class HiddenProjector(nn.Module):
    def __init__(
        self,
        config: ProjectorConfig,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        delta_mean: torch.Tensor | None = None,
        delta_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if config.feature_dim != PAPER_FEATURE_DIM:
            raise ValueError(f"primary paper feature_dim must be {PAPER_FEATURE_DIM}, got {config.feature_dim}")
        self.config = config
        self.state_autoencoder = _AutoEncoderMLP(config.hidden_dim, config.state_dim, config.state_hidden_dim)
        self.delta_autoencoder = _AutoEncoderMLP(config.hidden_dim, config.delta_dim, config.delta_hidden_dim)
        self.register_buffer("state_mean", torch.zeros(3, config.hidden_dim) if state_mean is None else state_mean.float())
        self.register_buffer("state_std", torch.ones(3, config.hidden_dim) if state_std is None else state_std.float())
        self.register_buffer("delta_mean", torch.zeros(2, config.hidden_dim) if delta_mean is None else delta_mean.float())
        self.register_buffer("delta_std", torch.ones(2, config.hidden_dim) if delta_std is None else delta_std.float())

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        return (states.float() - self.state_mean) / self.state_std.clamp_min(1e-6)

    def _normalize_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        return (deltas.float() - self.delta_mean) / self.delta_std.clamp_min(1e-6)

    def encode_hidden(self, last_three: Sequence[torch.Tensor]) -> torch.Tensor:
        h0, h1, h2 = _ensure_last_three(last_three)
        original_shape = h0.shape[:-1]
        flat = _flatten_hidden((h0, h1, h2)).to(device=next(self.parameters()).device, dtype=torch.float32)
        states = self._normalize_states(flat)
        deltas = self._normalize_deltas(torch.stack([flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 1]], dim=1))
        encoded_states = [self.state_autoencoder.encode(states[:, index]) for index in range(3)]
        encoded_deltas = [self.delta_autoencoder.encode(deltas[:, index]) for index in range(2)]
        features = torch.cat([*encoded_states, *encoded_deltas], dim=-1)
        if features.shape[-1] != PAPER_FEATURE_DIM:
            raise RuntimeError(f"projector produced {features.shape[-1]} features, expected {PAPER_FEATURE_DIM}")
        return _roundtrip_features(features.reshape(*original_shape, PAPER_FEATURE_DIM))

    def reconstruction_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != 3:
            raise ValueError(f"reservoir hidden batch must be [M,3,H], got {tuple(hidden.shape)}")
        hidden_float = hidden.float()
        states = self._normalize_states(hidden_float)
        deltas_raw = torch.stack([hidden_float[:, 1] - hidden_float[:, 0], hidden_float[:, 2] - hidden_float[:, 1]], dim=1)
        deltas = self._normalize_deltas(deltas_raw)
        state_loss = sum(torch.mean((self.state_autoencoder.reconstruct(states[:, i]) - states[:, i]) ** 2) for i in range(3))
        delta_loss = sum(torch.mean((self.delta_autoencoder.reconstruct(deltas[:, i]) - deltas[:, i]) ** 2) for i in range(2))
        return state_loss / 3.0 + delta_loss / 2.0

    def checkpoint_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": PAPER_TRACE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "expected_hidden_count": self.config.expected_hidden_count,
            "hidden_indices": list(self.config.hidden_indices),
            "hidden_semantics": list(self.config.hidden_semantics),
            "normalization": "per_slot_states_[3,H]_and_per_slot_deltas_[2,H]_fit_only",
            "feature_dim": self.config.feature_dim,
        }
        if extra:
            payload.update(extra)
        return payload

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra), output)
        return output

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "HiddenProjector":
        payload = torch.load(path, map_location=device, weights_only=True)
        _require_native_hidden_metadata(payload, "projector checkpoint")
        config_payload = dict(payload["config"])
        config_payload["hidden_indices"] = tuple(config_payload["hidden_indices"])
        config_payload["hidden_semantics"] = tuple(config_payload["hidden_semantics"])
        config = ProjectorConfig(**config_payload)
        if config.feature_dim != PAPER_FEATURE_DIM:
            raise ValueError(f"projector checkpoint feature_dim must be {PAPER_FEATURE_DIM}")
        projector = cls(config).to(device)
        projector.load_state_dict(payload["state_dict"])
        projector.eval()
        return projector


def _load_reservoir_hidden(reservoir: Any) -> torch.Tensor:
    if isinstance(reservoir, torch.Tensor):
        hidden = reservoir
    elif isinstance(reservoir, dict):
        if "split" in reservoir and reservoir["split"] != "fit":
            raise ValueError("projector reservoir split must be 'fit'")
        if "hidden_count" in reservoir and int(reservoir["hidden_count"]) != LLaDA_HIDDEN_COUNT:
            raise ValueError(f"reservoir hidden_count must be native LLaDA {LLaDA_HIDDEN_COUNT}")
        if "hidden" not in reservoir:
            raise ValueError("reservoir dict must contain key 'hidden' with shape [M,3,H]")
        hidden = reservoir["hidden"]
    elif isinstance(reservoir, (str, Path)):
        path = Path(reservoir)
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                if payload.get("split") != "fit":
                    raise ValueError("projector reservoir split must be 'fit'")
                if int(payload.get("hidden_count", -1)) != LLaDA_HIDDEN_COUNT:
                    raise ValueError(f"reservoir hidden_count must be native LLaDA {LLaDA_HIDDEN_COUNT}")
            hidden = payload["hidden"] if isinstance(payload, dict) else payload
        elif path.suffix == ".npz":
            payload = np.load(path)
            if "hidden" in payload:
                hidden = torch.from_numpy(payload["hidden"])
            else:
                hidden = torch.stack([torch.from_numpy(payload[f"hidden{i}"]) for i in range(3)], dim=1)
        elif path.suffix == ".npy":
            hidden = torch.from_numpy(np.load(path, mmap_mode="r"))
        else:
            raise ValueError(f"unsupported reservoir file: {path}")
    elif isinstance(reservoir, Iterable):
        chunks = [_flatten_hidden(item if isinstance(item, (list, tuple)) else item["last_three"]) for item in reservoir]
        if not chunks:
            raise ValueError("reservoir iterable is empty")
        hidden = torch.cat(chunks, dim=0)
    else:
        raise TypeError(f"unsupported reservoir type: {type(reservoir)!r}")

    if not isinstance(hidden, torch.Tensor):
        hidden = torch.as_tensor(hidden)
    if hidden.ndim != 3 or hidden.shape[1] != 3:
        raise ValueError(f"reservoir hidden must have shape [M,3,H], got {tuple(hidden.shape)}")
    if hidden.shape[0] == 0:
        raise ValueError("reservoir hidden is empty")
    if not torch.isfinite(hidden.float()).all():
        raise ValueError("reservoir hidden contains nonfinite values")
    return hidden


def fit_projector(
    reservoir: Any,
    path: str | Path,
    device: str | torch.device = "cpu",
    steps: int = 1000,
    seed: int = 42,
    batch_size: int = 256,
) -> HiddenProjector:
    torch.manual_seed(seed)
    hidden_cpu = _load_reservoir_hidden(reservoir).float().cpu()
    hidden_dim = int(hidden_cpu.shape[-1])
    deltas = torch.stack([hidden_cpu[:, 1] - hidden_cpu[:, 0], hidden_cpu[:, 2] - hidden_cpu[:, 1]], dim=1)
    state_mean = hidden_cpu.mean(dim=0)
    state_std = hidden_cpu.std(dim=0, unbiased=False).clamp_min(1e-6)
    delta_mean = deltas.mean(dim=0)
    delta_std = deltas.std(dim=0, unbiased=False).clamp_min(1e-6)
    projector = HiddenProjector(
        ProjectorConfig(hidden_dim=hidden_dim),
        state_mean=state_mean,
        state_std=state_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
    ).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_steps = max(1, int(steps))
    with torch.no_grad():
        initial_reconstruction_loss = float(
            projector.reconstruction_loss(hidden_cpu[: min(batch_size, hidden_cpu.shape[0])].to(device)).detach().cpu()
        )
    for _ in range(train_steps):
        indices = torch.randint(0, hidden_cpu.shape[0], (min(batch_size, hidden_cpu.shape[0]),), generator=generator)
        batch = hidden_cpu.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = projector.reconstruction_loss(batch)
        loss.backward()
        optimizer.step()
    projector.eval()
    with torch.no_grad():
        final_reconstruction_loss = float(
            projector.reconstruction_loss(hidden_cpu[: min(batch_size, hidden_cpu.shape[0])].to(device)).detach().cpu()
        )
    if not math.isfinite(initial_reconstruction_loss) or not math.isfinite(final_reconstruction_loss):
        raise ValueError("projector reconstruction loss is nonfinite")
    output = projector.save(
        path,
        {
            "reservoir_count": int(hidden_cpu.shape[0]),
            "reservoir_hidden_dim": hidden_dim,
            "fit_steps": train_steps,
            "seed": seed,
            "initial_reconstruction_loss": initial_reconstruction_loss,
            "final_reconstruction_loss": final_reconstruction_loss,
            "normalization": "per_slot_states_[3,H]_and_per_slot_deltas_[2,H]_fit_only",
        },
    )
    projection_hash = sha256_file(output)
    sidecar = Path(output).with_suffix(Path(output).suffix + ".json")
    sidecar.write_text(json.dumps({"projection_sha256": projection_hash, "config": asdict(projector.config)}, indent=2), encoding="utf-8")
    return projector


def encode_hidden(
    last_three: Sequence[torch.Tensor],
    projector_checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    projector = HiddenProjector.load(projector_checkpoint, device=device)
    with torch.no_grad():
        return projector.encode_hidden([tensor.to(device) for tensor in last_three]).cpu()


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_positions: int) -> None:
        super().__init__()
        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_positions, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.min().item() < 0 or position_ids.max().item() >= self.pe.shape[0]:
            raise ValueError("position_ids outside configured positional encoding range")
        return self.pe.index_select(0, position_ids.reshape(-1)).reshape(*position_ids.shape, -1)


class PaperTraceController(nn.Module):
    def __init__(self, config: ControllerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ControllerConfig()
        if self.config.feature_dim != PAPER_FEATURE_DIM:
            raise ValueError(f"primary paper controller feature_dim must be {PAPER_FEATURE_DIM}")
        self.input_projection = nn.Linear(self.config.feature_dim, self.config.model_dim)
        self.position_encoding = SinusoidalPositionEncoding(self.config.model_dim, self.config.max_positions)
        self.threshold_token = nn.Parameter(torch.zeros(1, 1, self.config.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.layers)
        self.token_head = nn.Linear(self.config.model_dim, 1)
        self.threshold_head = nn.Linear(self.config.model_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(f"features must be [B,L,{self.config.feature_dim}], got {tuple(features.shape)}")
        if valid_mask.shape != features.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,L]")
        if not torch.isfinite(features.float()).all():
            raise ValueError("features contain nonfinite values")
        batch, length, _ = features.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=features.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        if position_ids.shape != (batch, length) or position_ids.dtype != torch.long:
            raise ValueError("position_ids must be Long [B,L]")
        projected = self.input_projection(features.float()) + self.position_encoding(position_ids)
        threshold = self.threshold_token.expand(batch, -1, -1)
        encoded_input = torch.cat([threshold, projected], dim=1)
        threshold_valid = torch.ones((batch, 1), dtype=torch.bool, device=valid_mask.device)
        attention_mask = torch.cat([threshold_valid, valid_mask], dim=1)
        encoded = self.encoder(encoded_input, src_key_padding_mask=~attention_mask)
        token_logits = self.token_head(encoded[:, 1:]).squeeze(-1)
        threshold_logit = self.threshold_head(encoded[:, 0]).squeeze(-1)
        margins = token_logits - threshold_logit[:, None]
        return {"token_logits": token_logits, "threshold_logit": threshold_logit, "margins": margins}

    def checkpoint_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": PAPER_TRACE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "hidden_count": LLaDA_HIDDEN_COUNT,
            "feature_dim": self.config.feature_dim,
            "hidden_indices": list(self.config.hidden_indices),
            "hidden_semantics": list(self.config.hidden_semantics),
        }
        if extra:
            payload.update(extra)
        return payload

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PaperTraceController":
        payload = torch.load(path, map_location=device, weights_only=True)
        _require_native_hidden_metadata(payload, "controller checkpoint")
        config_payload = dict(payload["config"])
        config_payload["hidden_indices"] = tuple(config_payload["hidden_indices"])
        config_payload["hidden_semantics"] = tuple(config_payload["hidden_semantics"])
        controller = cls(ControllerConfig(**config_payload)).to(device)
        controller.load_state_dict(payload["state_dict"])
        controller.eval()
        return controller


def paper_trace_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    misclassified_weight: float = 4.0,
    tau_l2: float = 0.001,
) -> torch.Tensor:
    margins = outputs["margins"]
    threshold = outputs["threshold_logit"]
    if labels.shape != margins.shape or active_mask.shape != margins.shape:
        raise ValueError("labels and active_mask must match margins")
    targets = labels.float()
    predicted = margins.detach() > 0.0
    target_bool = targets > 0.5
    misclassified = predicted != target_bool
    weights = torch.where(misclassified, torch.full_like(targets, float(misclassified_weight)), torch.ones_like(targets))
    masked_weights = torch.where(active_mask, weights, torch.zeros_like(weights))
    loss = nn.functional.binary_cross_entropy_with_logits(margins, targets, weight=masked_weights, reduction="sum")
    denom = active_mask.sum().to(loss.dtype).clamp_min(1.0)
    return loss / denom + float(tau_l2) * torch.mean(threshold.float() ** 2)


def choose_label_blind_crop(
    states: np.ndarray,
    prompt_length: int,
    gen_length: int,
    rng: random.Random,
    min_fraction: float = 0.25,
) -> int:
    if states.ndim != 1:
        raise ValueError("crop helper expects one frame state row")
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")
    max_end = min(len(states), prompt_length + gen_length)
    min_end = min(max_end, prompt_length + max(1, int(math.ceil(gen_length * min_fraction))))
    candidates = [end for end in range(min_end, max_end + 1) if np.any(states[prompt_length:end] == STATE_GEN)]
    if not candidates:
        raise ValueError("crop has no active generated rows")
    return rng.choice(candidates)


def choose_validation_window_end(states: np.ndarray, prompt_length: int, gen_length: int, validation_window: int = 32) -> int:
    if states.ndim != 1:
        raise ValueError("validation window helper expects one frame state row")
    if validation_window <= 0:
        raise ValueError("validation_window must be positive")
    max_end = min(len(states), prompt_length + gen_length)
    active_offsets = np.flatnonzero(states[prompt_length:max_end] == STATE_GEN)
    if active_offsets.size == 0:
        raise ValueError("validation window has no active generated rows")
    first_active = prompt_length + int(active_offsets[0])
    return min(max_end, first_active + int(validation_window))


def validate_example_dir(example_dir: str | Path, projection_sha256: str | None = None) -> dict[str, Any]:
    root = Path(example_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"missing metadata.json in {root}")
    cache_key = (str(root.resolve()), projection_sha256, sha256_file(metadata_path))
    if cache_key in _VALIDATION_CACHE:
        return dict(_VALIDATION_CACHE[cache_key])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", PAPER_TRACE_SCHEMA_VERSION)) != PAPER_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported paper trace schema_version")
    if int(metadata.get("feature_dim", PAPER_FEATURE_DIM)) != PAPER_FEATURE_DIM:
        raise ValueError("metadata feature_dim mismatch")
    if int(metadata.get("hidden_count", LLaDA_HIDDEN_COUNT)) != LLaDA_HIDDEN_COUNT:
        raise ValueError("metadata hidden_count mismatch for native LLaDA")
    if list(metadata.get("hidden_indices", LLaDA_LAST_THREE_INDICES)) != list(LLaDA_LAST_THREE_INDICES):
        raise ValueError("metadata hidden_indices mismatch for native LLaDA paper projector")
    if list(metadata.get("hidden_semantics", LLaDA_LAST_THREE_SEMANTICS)) != list(LLaDA_LAST_THREE_SEMANTICS):
        raise ValueError("metadata hidden_semantics mismatch")
    metadata_projection_hash = metadata.get("projection_sha256", metadata.get("projection_hash"))
    if not metadata_projection_hash:
        raise ValueError("metadata missing projection_sha256/projection_hash")
    if projection_sha256 is not None and metadata_projection_hash != projection_sha256:
        raise ValueError("projection_sha256 mismatch")
    if not metadata.get("source"):
        raise ValueError("metadata missing nonempty source")
    features_path = root / "features.npy"
    states_path = root / "states.npy"
    labels_path = root / "labels.npy"
    for path in (features_path, states_path, labels_path):
        if not path.exists():
            raise ValueError(f"missing {path.name} in {root}")
    features = np.load(features_path, mmap_mode="r")
    states = np.load(states_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if features.dtype != np.float16 or features.ndim != 3 or features.shape[-1] != PAPER_FEATURE_DIM:
        raise ValueError("features.npy must be float16 [T,N,832]")
    if states.dtype != np.uint8 or labels.dtype != np.uint8:
        raise ValueError("states.npy and labels.npy must be uint8")
    if states.shape != labels.shape or states.shape != features.shape[:2]:
        raise ValueError("features/states/labels shapes disagree")
    if set(np.unique(states).tolist()) - {STATE_PROMPT, STATE_LOCKED, STATE_GEN, STATE_EOT}:
        raise ValueError("states.npy contains unknown state ids")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("labels.npy must be binary uint8 labels")
    expected_shape = metadata.get("shape")
    if expected_shape is not None and list(features.shape) != list(expected_shape):
        raise ValueError("metadata shape disagrees with features.npy")
    selected_steps = metadata.get("selected_steps")
    selected_ids = metadata.get("selected_ids")
    if not isinstance(selected_steps, list) or not isinstance(selected_ids, list):
        raise ValueError("metadata selected_steps and selected_ids must be lists")
    if len(selected_steps) != features.shape[0] or len(selected_ids) != features.shape[0]:
        raise ValueError("metadata selected_steps/selected_ids count must equal retained frame count")
    if "nonfinite_feature_count" in metadata:
        if int(metadata["nonfinite_feature_count"]) != 0:
            raise ValueError("metadata reports nonfinite features")
    elif "finite_feature_count" in metadata:
        if int(metadata["finite_feature_count"]) != int(np.prod(features.shape)):
            raise ValueError("metadata finite_feature_count mismatch")
    elif metadata.get("finite_features") is not None:
        if not bool(metadata["finite_features"]):
            raise ValueError("metadata reports nonfinite features")
    elif not bool(np.isfinite(features).all()):
        raise ValueError("features.npy contains nonfinite values")
    result = dict(metadata)
    result["projection_sha256"] = str(metadata_projection_hash)
    _VALIDATION_CACHE[cache_key] = dict(result)
    return result


def save_paper_trace_example(
    output_dir: str | Path,
    features: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if features.dtype != np.float16:
        features = features.astype(np.float16)
    if not np.isfinite(features).all():
        raise ValueError("features contain nonfinite values after float16 storage cast")
    if states.dtype != np.uint8:
        states = states.astype(np.uint8)
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8)
    if features.ndim != 3 or features.shape[-1] != PAPER_FEATURE_DIM:
        raise ValueError("features must be [T,N,832]")
    if states.shape != labels.shape or states.shape != features.shape[:2]:
        raise ValueError("states/labels must be [T,N] matching features")
    enriched = {
        **metadata,
        "schema_version": PAPER_TRACE_SCHEMA_VERSION,
        "feature_dim": PAPER_FEATURE_DIM,
        "shape": list(features.shape),
        "features_dtype": "float16",
        "states_dtype": "uint8",
        "labels_dtype": "uint8",
        "finite_feature_count": int(np.isfinite(features).sum()),
        "nonfinite_feature_count": 0,
        "hidden_count": LLaDA_HIDDEN_COUNT,
        "hidden_indices": list(LLaDA_LAST_THREE_INDICES),
        "hidden_semantics": list(LLaDA_LAST_THREE_SEMANTICS),
    }
    if "projection_sha256" not in enriched and "projection_hash" in enriched:
        enriched["projection_sha256"] = enriched["projection_hash"]
    required = ["prompt_length", "gen_length", "selected_steps", "selected_ids", "source", "projection_sha256"]
    missing = [name for name in required if name not in enriched]
    if missing:
        raise ValueError(f"metadata missing required keys: {missing}")
    np.save(root / "features.npy", features)
    np.save(root / "states.npy", states)
    np.save(root / "labels.npy", labels)
    (root / "metadata.json").write_text(json.dumps(_jsonable(enriched), indent=2, sort_keys=True), encoding="utf-8")
    validate_example_dir(root, projection_sha256=str(enriched["projection_sha256"]))
    return root


def build_training_frame_index(
    example_dirs: Sequence[str | Path],
    projection_sha256: str | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example_dir in example_dirs:
        metadata = validate_example_dir(example_dir, projection_sha256=projection_sha256)
        features = np.load(Path(example_dir) / "features.npy", mmap_mode="r")
        states = np.load(Path(example_dir) / "states.npy", mmap_mode="r")
        prompt_length = int(metadata["prompt_length"])
        gen_length = int(metadata["gen_length"])
        for frame in range(features.shape[0]):
            if np.any(states[frame, prompt_length : prompt_length + gen_length] == STATE_GEN):
                rows.append(
                    {
                        "path": str(Path(example_dir).resolve()),
                        "example_id": str(metadata.get("example_id", Path(example_dir).name)),
                        "frame": frame,
                        "prompt_length": prompt_length,
                        "gen_length": gen_length,
                        "projection_sha256": str(metadata.get("projection_sha256", metadata.get("projection_hash"))),
                        "content_hash": metadata.get("content_hash"),
                        "source_sha256": metadata.get("source_sha256"),
                        "prompt_hash": metadata.get("prompt_hash"),
                        "normalized_prompt_hash": metadata.get("normalized_prompt_hash"),
                        "split": split or metadata.get("split"),
                    }
                )
    if not rows:
        raise ValueError("no trainable frames found")
    return rows


class PaperTraceExampleDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frame_index: Sequence[dict[str, Any]],
        seed: int = 42,
        crop: bool = True,
        max_cached_dirs: int = 64,
        validation_window: int | None = None,
    ) -> None:
        if not frame_index:
            raise ValueError("frame_index is empty")
        self.frame_index = list(frame_index)
        self.seed = int(seed)
        self.epoch = 0
        self.crop = bool(crop)
        self.validation_window = validation_window
        self.max_cached_dirs = int(max_cached_dirs)
        if self.max_cached_dirs <= 0:
            raise ValueError("max_cached_dirs must be positive")
        self._cache: OrderedDict[str, tuple[np.memmap, np.memmap, np.memmap]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _close_array(array: np.ndarray) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    @classmethod
    def _close_arrays(cls, arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        for array in arrays:
            cls._close_array(array)

    def _arrays(self, path: str) -> tuple[np.memmap, np.memmap, np.memmap]:
        if path not in self._cache:
            root = Path(path)
            self._cache[path] = (
                np.load(root / "features.npy", mmap_mode="r"),
                np.load(root / "states.npy", mmap_mode="r"),
                np.load(root / "labels.npy", mmap_mode="r"),
            )
            while len(self._cache) > self.max_cached_dirs:
                _evicted_path, arrays = self._cache.popitem(last=False)
                self._close_arrays(arrays)
        else:
            self._cache.move_to_end(path)
        return self._cache[path]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame_index[index]
        features, states, labels = self._arrays(row["path"])
        frame = int(row["frame"])
        state_row = np.asarray(states[frame])
        rng = random.Random(self.seed + self.epoch * 9176 + index * 1000003 + frame)
        prompt_length = int(row["prompt_length"])
        gen_length = int(row["gen_length"])
        if self.crop:
            end = choose_label_blind_crop(state_row, prompt_length, gen_length, rng)
        elif self.validation_window is not None:
            end = choose_validation_window_end(state_row, prompt_length, gen_length, self.validation_window)
        else:
            end = min(len(state_row), prompt_length + gen_length)
        feature_row = np.asarray(features[frame, :end], dtype=np.float32)
        label_row = np.asarray(labels[frame, :end], dtype=np.uint8)
        cropped_states = state_row[:end]
        valid_mask = np.ones(end, dtype=np.bool_)
        active_mask = cropped_states == STATE_GEN
        if not np.any(active_mask):
            raise ValueError("cropped frame has no active generated rows")
        return {
            "features": torch.from_numpy(feature_row),
            "valid_mask": torch.from_numpy(valid_mask),
            "labels": torch.from_numpy(label_row.astype(np.float32)),
            "active_mask": torch.from_numpy(active_mask),
            "position_ids": torch.arange(end, dtype=torch.long),
        }


def collate_paper_trace_batch(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["features"].shape[0]) for item in items)
    batch = len(items)
    features = torch.zeros((batch, max_len, PAPER_FEATURE_DIM), dtype=torch.float32)
    valid_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    labels = torch.zeros((batch, max_len), dtype=torch.float32)
    active_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    position_ids = torch.zeros((batch, max_len), dtype=torch.long)
    for index, item in enumerate(items):
        length = int(item["features"].shape[0])
        features[index, :length] = item["features"]
        valid_mask[index, :length] = item["valid_mask"]
        labels[index, :length] = item["labels"]
        active_mask[index, :length] = item["active_mask"]
        position_ids[index, :length] = item["position_ids"]
    return {
        "features": features,
        "valid_mask": valid_mask,
        "labels": labels,
        "active_mask": active_mask,
        "position_ids": position_ids,
    }


def active_token_fbeta(margins: torch.Tensor, labels: torch.Tensor, active_mask: torch.Tensor, beta: float = 0.5, cutoff: float = 0.95) -> float:
    tp, fp, fn = active_token_counts(margins, labels, active_mask, cutoff=cutoff)
    return fbeta_from_counts(tp, fp, fn, beta=beta)


def active_token_counts(margins: torch.Tensor, labels: torch.Tensor, active_mask: torch.Tensor, cutoff: float = 0.95) -> tuple[int, int, int]:
    probs = torch.sigmoid(margins.detach())
    pred = (probs > cutoff) & active_mask.bool()
    target = (labels > 0.5) & active_mask.bool()
    tp = int(torch.sum(pred & target).item())
    fp = int(torch.sum(pred & ~target).item())
    fn = int(torch.sum(~pred & target).item())
    return tp, fp, fn


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    if tp == 0:
        return 0.0
    beta2 = beta * beta
    return float((1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp))


def _content_hashes_for_index(rows: Sequence[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        for key in ("content_hash", "prompt_hash", "normalized_prompt_hash"):
            value = row.get(key)
            if value:
                hashes.add(str(value))
    return hashes


def validate_train_validation_split(fit_index: Sequence[dict[str, Any]], val_index: Sequence[dict[str, Any]]) -> None:
    fit_examples = {str(row["example_id"]) for row in fit_index}
    val_examples = {str(row["example_id"]) for row in val_index}
    example_overlap = sorted(fit_examples & val_examples)
    if example_overlap:
        raise ValueError(f"fit and validation overlap by example_id: {example_overlap[:5]}")
    fit_paths = {str(Path(row["path"]).resolve()) for row in fit_index}
    val_paths = {str(Path(row["path"]).resolve()) for row in val_index}
    path_overlap = sorted(fit_paths & val_paths)
    if path_overlap:
        raise ValueError(f"fit and validation overlap by path: {path_overlap[:5]}")
    hash_overlap = sorted(_content_hashes_for_index(fit_index) & _content_hashes_for_index(val_index))
    if hash_overlap:
        raise ValueError(f"fit and validation overlap by content hash: {hash_overlap[:5]}")
    for name, rows, expected in (("fit", fit_index, "fit"), ("validation", val_index, "validation")):
        bad = [row.get("split") for row in rows if row.get("split") is not None and row.get("split") != expected]
        if bad:
            raise ValueError(f"{name} index contains unexpected split values: {bad[:5]}")


def train_controller(
    fit_index: Sequence[dict[str, Any]],
    val_index: Sequence[dict[str, Any]],
    output_dir: str | Path,
    projector_checkpoint: str | Path,
    steps: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    seed: int = 42,
    validate_every: int = 500,
    device: str | torch.device = "cpu",
    validation_window: int = 32,
) -> TrainingResult:
    torch.manual_seed(seed)
    random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    projection_sha256 = sha256_file(projector_checkpoint)
    for rows, name in ((fit_index, "fit"), (val_index, "validation")):
        for row in rows:
            if str(row.get("projection_sha256")) != projection_sha256:
                raise ValueError(f"{name} index projection_sha256 mismatch")
    validate_train_validation_split(fit_index, val_index)
    controller = PaperTraceController().to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=weight_decay)
    fit_data = PaperTraceExampleDataset(fit_index, seed=seed, crop=True)
    val_data = PaperTraceExampleDataset(val_index, seed=seed + 999, crop=False, validation_window=validation_window)
    generator = torch.Generator().manual_seed(seed)
    best_score = -1.0
    best_path = output / "best_controller.pt"
    metrics: list[dict[str, Any]] = []

    for step in range(1, int(steps) + 1):
        fit_data.set_epoch(step)
        sample_ids = torch.randint(0, len(fit_data), (int(batch_size),), generator=generator).tolist()
        batch = collate_paper_trace_batch([fit_data[int(sample_id)] for sample_id in sample_ids])
        batch = {key: value.to(device) for key, value in batch.items()}
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = controller(batch["features"], batch["valid_mask"], batch["position_ids"])
        loss = paper_trace_loss(outputs, batch["labels"], batch["active_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
        optimizer.step()

        if step % int(validate_every) == 0 or step == int(steps):
            score, val_loss = evaluate_controller(controller, val_data, device=device, batch_size=min(64, max(1, len(val_data))))
            record = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss, "val_active_f0_5": score}
            metrics.append(record)
            if score > best_score:
                best_score = score
                torch.save(
                    controller.checkpoint_payload(
                        {
                            "projection_sha256": projection_sha256,
                            "selected_by": "val_active_token_F0.5_at_sigmoid_gt_0.95",
                            "selected_step": step,
                            "validation_window": int(validation_window),
                            "seed": seed,
                        }
                    ),
                    best_path,
                )

    config_path = output / "config.json"
    metrics_path = output / "metrics.json"
    provenance_path = output / "provenance.json"
    config_path.write_text(json.dumps(_jsonable(asdict(controller.config)), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(_jsonable({"best_val_active_f0_5": best_score, "history": metrics}), indent=2), encoding="utf-8")
    provenance_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "projector_checkpoint": str(Path(projector_checkpoint).resolve()),
                    "projection_sha256": projection_sha256,
                    "fit_examples": len({row["example_id"] for row in fit_index}),
                    "validation_examples": len({row["example_id"] for row in val_index}),
                    "steps": int(steps),
                    "batch_size": int(batch_size),
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "seed": seed,
                    "validation_window": int(validation_window),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingResult(best_path, config_path, metrics_path, provenance_path)


def evaluate_controller(
    controller: PaperTraceController,
    dataset: PaperTraceExampleDataset,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> tuple[float, float]:
    controller.eval()
    losses: list[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collate_paper_trace_batch([dataset[index] for index in range(start, min(len(dataset), start + batch_size))])
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = controller(batch["features"], batch["valid_mask"], batch["position_ids"])
            loss = paper_trace_loss(outputs, batch["labels"], batch["active_mask"])
            losses.append(float(loss.cpu()))
            tp, fp, fn = active_token_counts(outputs["margins"].cpu(), batch["labels"].cpu(), batch["active_mask"].cpu())
            total_tp += tp
            total_fp += fp
            total_fn += fn
    return fbeta_from_counts(total_tp, total_fp, total_fn), float(np.mean(losses)) if losses else math.inf


def read_index_json(path: str | Path, projection_sha256: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    if not isinstance(rows, list):
        raise ValueError("index JSON must be a list or {'rows': [...]}")
    if rows and isinstance(rows[0], str):
        return build_training_frame_index(rows, projection_sha256=projection_sha256, split=split)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("index rows must be objects with path/example metadata or directory strings")
        if "path" not in row:
            raise ValueError("index row missing path")
        metadata = validate_example_dir(row["path"], projection_sha256=projection_sha256)
        if "frame" not in row:
            normalized.extend(build_training_frame_index([row["path"]], projection_sha256=projection_sha256, split=row.get("split", split)))
            continue
        normalized.append(
            {
                "path": str(Path(row["path"]).resolve()),
                "example_id": str(row.get("example_id", metadata.get("example_id", Path(row["path"]).name))),
                "frame": int(row.get("frame", 0)),
                "prompt_length": int(row.get("prompt_length", metadata["prompt_length"])),
                "gen_length": int(row.get("gen_length", metadata["gen_length"])),
                "projection_sha256": str(row.get("projection_sha256", metadata.get("projection_sha256", metadata.get("projection_hash")))),
                "content_hash": row.get("content_hash", metadata.get("content_hash")),
                "source_sha256": row.get("source_sha256", metadata.get("source_sha256")),
                "prompt_hash": row.get("prompt_hash", metadata.get("prompt_hash")),
                "normalized_prompt_hash": row.get("normalized_prompt_hash", metadata.get("normalized_prompt_hash")),
                "split": row.get("split", split),
            }
        )
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the paper-mechanism hidden-state trace controller.")
    parser.add_argument("--fit-index", required=True, help="JSON list of fit frame rows or example dirs.")
    parser.add_argument("--val-index", required=True, help="JSON list of validation frame rows or example dirs.")
    parser.add_argument("--projector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-window", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    projection_sha256 = sha256_file(args.projector_checkpoint)
    fit_index = read_index_json(args.fit_index, projection_sha256=projection_sha256, split="fit")
    val_index = read_index_json(args.val_index, projection_sha256=projection_sha256, split="validation")
    result = train_controller(
        fit_index=fit_index,
        val_index=val_index,
        output_dir=args.output_dir,
        projector_checkpoint=args.projector_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.wd,
        grad_clip=args.grad_clip,
        seed=args.seed,
        validate_every=args.validate_every,
        device=args.device,
        validation_window=args.validation_window,
    )
    print(f"checkpoint={result.checkpoint_path}")
    print(f"config={result.config_path}")
    print(f"metrics={result.metrics_path}")
    print(f"provenance={result.provenance_path}")
    return 0
