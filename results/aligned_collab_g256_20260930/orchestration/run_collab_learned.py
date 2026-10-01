"""Run the collaborator's LEARNED methods (retrained here with her recipe) under the aligned protocol.

Her environment, like run_collab_methods.py, whose prompt patch, guard and example loader are reused.
Methods:
  collab-TraceLock-LLaDA   decode_tracelock, layers 30/31/32, sliding window 32, threshold 0.95 (her online-screen arm)
  collab-safety-full       decode_policy(method="safety_full") with the retrained direct-safety predictor
  collab-open-confidence   her trace_semantic main() at batch 1: Phase-1(+2) planner once per question, then
                           the A3 replay per timed repeat. Time = A3 replay + controller time + extra Phase-2
                           planner time, which is her own additive estimate (never an online measurement).
                           Phase 3 abstains: her branch has no latency calibrator, so an always-abstain table is used.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import run_collab_methods as rcm  # noqa: E402  (installs collab_compat and the prompt patch)
import benchmark_phaselock_trace_semantic as semantic_mod  # noqa: E402
import benchmark_phase_lock_batched_online as batched_mod  # noqa: E402
from benchmark_phaselock_matched_plans import reference_replay_plan, semantic_replay_plan  # noqa: E402
from regret_remasking.commitment_safety import CommitmentSafetyScorer  # noqa: E402
from regret_remasking.llada_trace import DecodeConfig, infer_mask_token_id, load_llada  # noqa: E402
from regret_remasking.phase_lock_sparse import FusedLatencyTable  # noqa: E402
from regret_remasking.tracelock_baseline import TraceLockConfig, TraceLockController  # noqa: E402

for _module in (semantic_mod, batched_mod):
    if hasattr(_module, "build_prompt"):
        _module.build_prompt = rcm.aligned_build_prompt

LAYERS = [30, 31, 32]
# Her trace_semantic main() constants (benchmark_phaselock_trace_semantic.py:846-875) and CLI defaults.
PHASE2 = dict(representation_layers=[8, 16, 24], influence_layer=24, drift_threshold=0.002, influence_threshold=0.5,
              patience=6, min_age=4, lock_fraction=0.04, max_queries=8, window_width=32)
REPLAY_COMMON = dict(semantic_lock_fraction=0.06, semantic_min_confidence=0.95, semantic_optional_steps_per_block=2,
                     representation_policy="capped_drift", representation_delay_updates=1, representation_layer=8,
                     representation_drift_threshold=0.005, representation_patience=4, representation_min_age=1,
                     representation_lock_fraction=0.06, compute_min_locked_fraction=0.20, compute_min_remaining_steps=8,
                     compute_min_projected_saved_rows=1500, compute_min_expected_gain=0.03, bucket_width_multiple=16,
                     max_bucket_size=8, force_semantic_plan_tokens=True)
METHODS = ["collab-TraceLock-LLaDA", "collab-open-confidence", "collab-safety-full"]


def abstain_latency_table() -> FusedLatencyTable:
    """Batch-1 table with no predicted gain anywhere, so Phase 3 never switches (declared adaptation)."""
    rows = []
    for sequence_length in (256, 512, 768, 1024):
        for active_rows in (1, sequence_length // 4, sequence_length // 2, sequence_length):
            rows.append({"sequence_length": sequence_length, "batch_size": 1, "active_rows": active_rows,
                         "median_ms": 1.0, "dense_median_ms": 1.0})
    return FusedLatencyTable.from_rows(rows)


class Runner:
    def __init__(self, model, tokenizer, config, controller, projection, scorer, table):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.controller, self.projection, self.scorer, self.table = controller, projection, scorer, table
        self.planners: dict[str, dict] = {}

    def tracelock(self, example):
        return rcm.tracelock_mod.decode_tracelock(self.model, self.tokenizer, self.controller, self.projection, LAYERS,
                                                  [example], self.config, window_width=32)

    def safety(self, example):
        kwargs = {**rcm.CAPPED, "method": "safety_full", "threshold": self.scorer.threshold, "safety_scorer": self.scorer}
        return rcm.policy_mod.decode_policy(self.model, self.tokenizer, self.controller, self.projection, [example], self.config, **kwargs)

    def plan(self, example):
        if example.example_id not in self.planners:
            common = dict(policy=semantic_mod.POLICIES["open_confidence"], safety_scorer=None, **PHASE2)
            p1 = semantic_mod.decode_trace_semantic_planner(self.model, self.tokenizer, self.controller, self.projection, LAYERS,
                                                            [example], self.config, with_phase2=False, **common)
            p2 = semantic_mod.decode_trace_semantic_planner(self.model, self.tokenizer, self.controller, self.projection, LAYERS,
                                                            [example], self.config, with_phase2=True, **common)
            if p1["token_ids"] != p2["token_ids"]:
                raise RuntimeError("Phase-2 observation changed Phase-1 token output")
            self.planners[example.example_id] = {
                "p1": p1, "p2": p2,
                "replay": {"semantic_plans": [semantic_replay_plan(p) for p in p2["semantic_plans"]],
                           "reference_plans": [reference_replay_plan(p) for p in p2["reference_plans"]]},
                "overhead_s": float(p2["semantic_controller_latency_s"]) + max(0.0, float(p2["latency_s"]) - float(p1["latency_s"])),
            }
        return self.planners[example.example_id]

    def open_confidence(self, example):
        planned = self.plan(example)
        a3 = batched_mod.decode_batch(self.model, self.tokenizer, [example], self.config, "A3", self.table,
                                      **REPLAY_COMMON, **planned["replay"])
        # Her analysis reports A1->A3 exactness rather than failing on it; keep that as data.
        a3["a3_equals_planner_tokens"] = a3["token_ids"] == planned["p2"]["token_ids"]
        a3["estimated_online_latency_s"] = float(a3["latency_s"]) + planned["overhead_s"]
        a3["planner_online_latency_s"] = float(planned["p2"]["latency_s"])
        a3["controller_overhead_s"] = planned["overhead_s"]
        return a3

    def run(self, name, example):
        return {"collab-TraceLock-LLaDA": self.tracelock, "collab-safety-full": self.safety,
                "collab-open-confidence": self.open_confidence}[name](example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    names = args.methods.split(",")
    rows = json.loads(args.prompts.read_text())
    for row in rows:
        rcm.PROMPTS[("math500" if row["dataset"] == "math500" else "humaneval", row["question"])] = row["prompt_text"]
    examples = [rcm.to_example(r) for r in rows]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    preflight = rcm.gpu_guard(initial=True)
    torch.manual_seed(23)
    np.random.seed(23)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    controller = TraceLockController(TraceLockConfig(**checkpoint["config"]))
    controller.load_state_dict(checkpoint["model"])
    model, tokenizer = load_llada("GSAI-ML/LLaDA-8B-Instruct", device="cuda", dtype="bf16")
    controller = controller.to("cuda").eval()
    projection = torch.from_numpy(np.load(args.projection)).float().to("cuda")
    scorer = CommitmentSafetyScorer(args.predictor_dir / "commitment_safety_full.joblib") if "collab-safety-full" in names else None
    config = DecodeConfig(steps=256, gen_length=256, block_length=32, mask_id=infer_mask_token_id(tokenizer), temperature=0.0)
    runner = Runner(model, tokenizer, config, controller, projection, scorer, abstain_latency_table())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "protocol": "docs/aligned_protocol_2026-09-29.md", "methods": names, "repeats": args.repeats, "warmups": 1, "batch_size": 1,
        "decode_config": {f.name: getattr(config, f.name) for f in fields(config)},
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint), "controller_config": checkpoint["config"],
        "projection_sha256": sha(args.projection), "predictor_dir": str(args.predictor_dir) if args.predictor_dir else None,
        "safety_threshold": scorer.threshold if scorer else None, "phase2": PHASE2, "replay_common": REPLAY_COMMON,
        "latency_table": "always-abstain batch-1 table (Phase 3 off)", "prompts_sha256": sha(args.prompts),
        "driver_sha256": sha(__file__), "preflight": preflight, "torch": torch.__version__, "python": platform.python_version(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str) + "\n")
    with (args.output_dir / "records.jsonl").open("x") as out:
        for name in names:
            print(f"[collab-learned] warmup {name}", flush=True)
            runner.run(name, examples[0])
        runner.planners.clear()  # plans are recomputed (and timed) inside the measured loop
        for index, example in enumerate(examples):
            for repeat in range(1, args.repeats + 1):
                shift = (index + repeat - 1) % len(names)
                order = names[shift:] + names[:shift]
                for name in order:
                    rcm.gpu_guard()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    record = runner.run(name, example)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - started
                    tokens = record["token_ids"][0]
                    if config.mask_id in tokens or len(tokens) != 256:
                        raise RuntimeError(f"{name} left masked or wrong-length output")
                    latency = record.get("estimated_online_latency_s", wall) if name == "collab-open-confidence" else wall
                    out.write(json.dumps({
                        "dataset": example.dataset, "example_id": example.example_id, "policy_id": name, "repeat": repeat,
                        "order": order, "token_ids": tokens, "latency_s": latency, "wall_s": wall,
                        "internal_latency_s": record.get("latency_s"), "nfe": record.get("nfe"),
                        "controller_latency_s": record.get("controller_latency_s") or record.get("controller_overhead_s"),
                        "planner_online_latency_s": record.get("planner_online_latency_s"),
                        "switch_step": record.get("switch_step"), "extra_commit_count": record.get("extra_commit_count"),
                        "latency_kind": "her additive estimate" if name == "collab-open-confidence" else "wall",
                        "a3_equals_planner_tokens": record.get("a3_equals_planner_tokens"),
                    }) + "\n")
                    out.flush()
                    print(f"[collab-learned] {index + 1}/{len(examples)} {example.example_id} {name} repeat={repeat}", flush=True)
    (args.output_dir / "completion.json").write_text(json.dumps({"complete": True, "records": len(examples) * len(names) * args.repeats,
                                                                 "completed_at": datetime.now(timezone.utc).isoformat()}) + "\n")


if __name__ == "__main__":
    main()
