#!/usr/bin/env python3
"""Train the paper-specified TraceLock controller on LLaDA traces."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from regret_remasking.tracelock_baseline import (
    STATE_GEN,
    TraceLockConfig,
    TraceLockController,
    tracelock_loss,
)


class TraceStepDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        self.paths = sorted((root / split).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"no TraceLock step samples in {root / split}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


def audit_trace_manifests(
    trace_dir: Path, expected_source_counts: str
) -> dict[str, Any]:
    manifests = sorted(trace_dir.glob("manifest_*.json"))
    if not manifests:
        raise ValueError(f"no trace manifests found in {trace_dir}")
    records: list[dict[str, Any]] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"trace manifest must be a list: {path}")
        records.extend(payload)
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("trace manifests contain duplicate sample ids")
    source_counts = collections.Counter(str(record["source"]) for record in records)
    expected: dict[str, int] = {}
    for raw in expected_source_counts.split(","):
        item = raw.strip()
        if not item:
            continue
        source, count = item.split("=", 1)
        expected[source.strip()] = int(count)
    if expected and dict(source_counts) != expected:
        raise ValueError(
            f"trace source counts do not match preregistration: "
            f"observed={dict(source_counts)}, expected={expected}"
        )
    manifest_steps = sum(int(record["saved_steps"]) for record in records)
    train_files = sum(1 for _ in (trace_dir / "train").glob("*.pt"))
    validation_files = sum(1 for _ in (trace_dir / "val").glob("*.pt"))
    if train_files + validation_files != manifest_steps:
        raise ValueError(
            "trace step-file count does not match manifests: "
            f"files={train_files + validation_files}, manifest={manifest_steps}"
        )
    return {
        "manifest_files": len(manifests),
        "unique_examples": len(sample_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "manifest_step_samples": manifest_steps,
        "train_step_samples": train_files,
        "validation_step_samples": validation_files,
    }


def collate_steps(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["projected_hidden"].shape[0]) for item in batch)
    hidden_dim = int(batch[0]["projected_hidden"].shape[-1])
    projected = torch.zeros(
        (len(batch), max_len, 3, hidden_dim), dtype=torch.float16
    )
    state = torch.zeros((len(batch), max_len), dtype=torch.int8)
    label = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attention = torch.zeros((len(batch), max_len), dtype=torch.bool)
    prompt_lengths = torch.zeros(len(batch), dtype=torch.long)
    generation_lengths = torch.zeros(len(batch), dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = int(item["projected_hidden"].shape[0])
        projected[batch_index, :length] = item["projected_hidden"]
        state[batch_index, :length] = item["state"]
        label[batch_index, :length] = item["label"]
        attention[batch_index, :length] = item["attention_mask"]
        prompt_lengths[batch_index] = int(item["prompt_length"])
        generation_lengths[batch_index] = int(item["generation_length"])
    return {
        "projected_hidden": projected,
        "state": state,
        "label": label,
        "attention_mask": attention,
        "prompt_length": prompt_lengths,
        "generation_length": generation_lengths,
    }


def random_prefix_crop(
    attention_mask: torch.Tensor,
    prompt_length: torch.Tensor,
    generation_length: torch.Tensor,
    *,
    minimum_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    cropped = attention_mask.clone()
    ratios = minimum_ratio + (1.0 - minimum_ratio) * torch.rand(
        attention_mask.shape[0], generator=generator
    )
    for batch_index in range(attention_mask.shape[0]):
        visible = int(round(float(ratios[batch_index]) * int(generation_length[batch_index])))
        end = int(prompt_length[batch_index]) + max(1, visible)
        cropped[batch_index, end:] = False
    return cropped


@torch.inference_mode()
def evaluate(
    model: TraceLockController,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    true_positive = false_positive = false_negative = true_negative = 0
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        projected = batch["projected_hidden"].to(device)
        attention = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        active = (batch["state"].to(device) == STATE_GEN) & attention
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(projected, attention)
            loss = tracelock_loss(
                output["margin"],
                output["threshold_logit"],
                labels,
                active,
                mistake_weight=model.config.mistake_weight,
                threshold_regularizer=model.config.threshold_regularizer,
            )
        losses.append(float(loss.detach().cpu()))
        predictions = output["probability"] > model.config.deployment_probability
        target = labels.bool()
        true_positive += int((predictions & target & active).sum().item())
        false_positive += int((predictions & ~target & active).sum().item())
        false_negative += int((~predictions & target & active).sum().item())
        true_negative += int((~predictions & ~target & active).sum().item())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    beta_sq = 0.25
    f05 = (1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "precision_at_095": precision,
        "recall_at_095": recall,
        "f0_5_at_095": f05,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--random-crop-min-ratio", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-source-counts", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest_audit = audit_trace_manifests(
        args.trace_dir, args.expected_source_counts
    )
    print(json.dumps({"trace_manifest_audit": manifest_audit}), flush=True)
    train_data = TraceStepDataset(args.trace_dir, "train")
    val_data = TraceStepDataset(args.trace_dir, "val")
    first = train_data[0]
    projected_dim = int(first["projected_hidden"].shape[-1])
    max_length = max(
        768,
        int(first["projected_hidden"].shape[0]),
    )
    config = TraceLockConfig(
        projected_hidden_dim=projected_dim,
        max_sequence_length=max_length,
    )
    model = TraceLockController(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_steps,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_steps,
    )
    crop_generator = torch.Generator(device="cpu")
    crop_generator.manual_seed(args.seed + 1000)
    iterator = iter(train_loader)
    best_f05 = -1.0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        attention = random_prefix_crop(
            batch["attention_mask"],
            batch["prompt_length"],
            batch["generation_length"],
            minimum_ratio=args.random_crop_min_ratio,
            generator=crop_generator,
        ).to(device)
        projected = batch["projected_hidden"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        active = (batch["state"].to(device, non_blocking=True) == STATE_GEN) & attention
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(projected, attention)
            loss = tracelock_loss(
                output["margin"],
                output["threshold_logit"],
                labels,
                active,
                mistake_weight=config.mistake_weight,
                threshold_regularizer=config.threshold_regularizer,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"[tracelock-train] step={step} loss={float(loss):.6f}", flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(
                model, val_loader, device, max_batches=args.max_eval_batches
            )
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            checkpoint = {
                "model": model.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "metrics": metrics,
                "trace_dir": str(args.trace_dir),
                "paper_deviation": "seeded Gaussian LLaDA hidden projection",
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if metrics["f0_5_at_095"] > best_f05:
                best_f05 = metrics["f0_5_at_095"]
                torch.save(checkpoint, args.output_dir / "best_f0_5.pt")
    summary = {
        "train_step_samples": len(train_data),
        "validation_step_samples": len(val_data),
        "max_steps": args.max_steps,
        "best_f0_5_at_095": best_f05,
        "elapsed_s": time.perf_counter() - started,
        "controller": config.to_dict(),
        "trace_manifest_audit": manifest_audit,
        "paper_deviation": "authors did not release the LLaDA projection; seeded Gaussian projection used",
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "trace_dir": str(args.trace_dir), "output_dir": str(args.output_dir)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
