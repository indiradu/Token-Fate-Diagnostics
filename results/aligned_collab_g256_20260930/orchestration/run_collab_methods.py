"""Run the collaborator's decoders (commit 24cd51b) under the aligned protocol.

Runs in HER environment: PYTHONPATH=source_collab/src:source_collab/scripts. Batch 1, 256/256/32,
our prompts (patched into her build_prompt), 1 warm-up per method, 3 timed repeats per question
with rotating method order. Records token ids; scoring happens later in our environment.

Method dispatch mirrors her benchmark_phaselock_online_screen.py (dense/surelock/hybrid) and
benchmark_commitment_safety_policy.py (capped_confidence). Learned methods are added once the
retrained controller and predictor exist.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

CAMPAIGN = Path(__file__).resolve().parents[1]

import collab_compat  # noqa: E402,F401  (transformers-5 loader shim; must precede her modules)
import benchmark_surelock_vs_phaselock as surelock_mod  # noqa: E402
import benchmark_commitment_safety_policy as policy_mod  # noqa: E402
import benchmark_phaselock_sure_hybrid as hybrid_mod  # noqa: E402
import benchmark_tracelock_vs_phaselock as tracelock_mod  # noqa: E402
from regret_remasking.data import Example  # noqa: E402
from regret_remasking.llada_trace import DecodeConfig, infer_mask_token_id, load_llada  # noqa: E402

PROMPTS: dict[tuple[str, str], str] = {}


def aligned_build_prompt(question: str, dataset: str) -> str:
    return PROMPTS[(str(dataset), str(question))]


# Every script binds build_prompt at import; patch each module's global.
for _module in (surelock_mod, policy_mod, hybrid_mod, tracelock_mod):
    _module.build_prompt = aligned_build_prompt

# Settings copied from her online screen (benchmark_phaselock_online_screen.py:111-115) and
# configs/phaselock_h2_controller_screen_v1.json line 17; capped-confidence settings from her
# matched-60 run (cluster/run_commitment_safety_benchmark.slurm and argparse defaults).
COMMON = dict(epsilon=0.005, confidence_percentile=20.0, bucket_width_multiple=16)
HYBRID = dict(**COMMON, representation_layers=[8, 16, 24], influence_layer=24, drift_threshold=0.01,
              influence_threshold=0.5, representation_patience=1, representation_min_age=1,
              representation_lock_fraction=0.04, max_queries=8)
H2_GUARD90_DELAY1 = {"semantic_min_confidence": 0.9, "speed_reference_policy": "delay", "representation_min_age": 1}
CAPPED = dict(method="capped_confidence", threshold=0.95, safety_scorer=None, trace_layers=[30, 31, 32],
              representation_layers=[8, 16, 24], influence_layer=24, max_queries=8, window_width=32,
              maximum_extra_per_update=2, capped_lock_fraction=0.06, capped_confidence_threshold=0.95,
              capped_optional_steps_per_block=2)

METHODS = ["collab-dense", "collab-SureLock-5e-4", "collab-SureLock-5e-3", "collab-capped-confidence", "collab-H2-guard90-delay1"]


def run_method(name, model, tokenizer, example, config):
    if name == "collab-dense":
        return surelock_mod.decode_surelock(model, tokenizer, [example], config, **{**COMMON, "epsilon": None})
    if name == "collab-SureLock-5e-4":
        return surelock_mod.decode_surelock(model, tokenizer, [example], config, **{**COMMON, "epsilon": 5e-4})
    if name == "collab-SureLock-5e-3":
        return surelock_mod.decode_surelock(model, tokenizer, [example], config, **{**COMMON, "epsilon": 5e-3})
    if name == "collab-capped-confidence":
        return policy_mod.decode_policy(model, tokenizer, None, None, [example], config, **CAPPED)
    if name == "collab-H2-guard90-delay1":
        return hybrid_mod.decode_hybrid(model, tokenizer, [example], config, mode="h2", **{**HYBRID, **H2_GUARD90_DELAY1})
    raise ValueError(name)


def gpu_guard(initial=False):
    props = torch.cuda.get_device_properties(0)
    if props.name != "NVIDIA RTX 5000 Ada Generation":
        raise RuntimeError(f"unvalidated GPU: {props.name}")
    uuid = "GPU-" + str(props.uuid).removeprefix("GPU-")
    out = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [l for l in out.stdout.splitlines() if l.split(",")[0].strip().isdigit() and int(l.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    free, total = torch.cuda.mem_get_info(0)
    if initial and free < 30 * 1024**3:
        raise RuntimeError(f"GPU not empty before load: {free} bytes free")
    return {"gpu": props.name, "uuid": uuid, "free_bytes": free, "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def to_example(row):
    dataset = "math500" if row["dataset"] == "math500" else "humaneval"
    return Example(example_id=row["example_id"], question=row["question"], gold_answer=row["gold_answer"], dataset=dataset, metadata=dict(row["metadata"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    names = args.methods.split(",")
    rows = json.loads(args.prompts.read_text())
    for row in rows:
        key = ("math500" if row["dataset"] == "math500" else "humaneval", row["question"])
        if key in PROMPTS and PROMPTS[key] != row["prompt_text"]:
            raise ValueError("ambiguous question text")
        PROMPTS[key] = row["prompt_text"]
    examples = [to_example(r) for r in rows]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    preflight = gpu_guard(initial=True)
    torch.manual_seed(23)
    np.random.seed(23)
    model, tokenizer = load_llada("GSAI-ML/LLaDA-8B-Instruct", device="cuda", dtype="bf16")
    config = DecodeConfig(steps=256, gen_length=256, block_length=32, mask_id=infer_mask_token_id(tokenizer), temperature=0.0)
    root = CAMPAIGN / "source_collab"
    source_sha = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list((root / "src/regret_remasking").glob("*.py")) + list((root / "scripts").glob("*.py")))}
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "protocol": "docs/aligned_protocol_2026-09-29.md", "methods": names, "repeats": args.repeats, "warmups": 1,
        "decode_config": {f.name: getattr(config, f.name) for f in fields(config)}, "batch_size": 1,
        "prompts": str(args.prompts), "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "source_commit": "24cd51b", "source_sha256": source_sha, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"common": COMMON, "hybrid": HYBRID, "h2_guard90_delay1": H2_GUARD90_DELAY1, "capped": {k: v for k, v in CAPPED.items() if k != "safety_scorer"}},
        "preflight": preflight, "torch": torch.__version__, "python": platform.python_version(),
        "model_commit": getattr(model.config, "_commit_hash", None), "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str) + "\n")
    with (args.output_dir / "records.jsonl").open("x") as out:
        for name in names:
            print(f"[collab] warmup {name}", flush=True)
            run_method(name, model, tokenizer, examples[0], config)
        for index, example in enumerate(examples):
            for repeat in range(1, args.repeats + 1):
                shift = (index + repeat - 1) % len(names)
                order = names[shift:] + names[:shift]
                for name in order:
                    gpu_guard()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    record = run_method(name, model, tokenizer, example, config)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - started
                    tokens = record["token_ids"][0]
                    if config.mask_id in tokens or len(tokens) != 256:
                        raise RuntimeError(f"{name} left masked or wrong-length output")
                    out.write(json.dumps({
                        "dataset": example.dataset, "example_id": example.example_id, "policy_id": name, "repeat": repeat,
                        "order": order, "token_ids": tokens, "her_generation": record["generations"][0],
                        "latency_s": wall, "internal_latency_s": record["latency_s"], "nfe": record["nfe"],
                        "prompt_length": record.get("prompt_len"), "lock_count": record.get("lock_count"),
                        "removed_row_token_forwards": record.get("removed_row_token_forwards"),
                        "controller_latency_s": record.get("controller_latency_s"),
                        "extra_commit_count": record.get("extra_commit_count"),
                    }) + "\n")
                    out.flush()
                    print(f"[collab] {index + 1}/{len(examples)} {example.example_id} {name} repeat={repeat}", flush=True)
    (args.output_dir / "completion.json").write_text(json.dumps({"complete": True, "records": len(examples) * len(names) * args.repeats, "completed_at": datetime.now(timezone.utc).isoformat()}) + "\n")


if __name__ == "__main__":
    main()
