"""Rerun OUR standard B0 in the same job as the collaborator's methods (our environment).

Uses the frozen aligned-campaign source, runtime cache and B0 policy unchanged, so outputs must equal
results/aligned_standard_g256_20260929 B0 token-for-token.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

CAMPAIGN = Path(__file__).resolve().parents[1]
ALIGNED = CAMPAIGN.parent / "aligned_standard_g256_20260929"
CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_MODULES_CACHE"] = str(ALIGNED / "runtime_cache/model_modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ALIGNED / "source/src"))
    sys.path.insert(0, str(ALIGNED / "orchestration"))
    import run_aligned as ra
    from regret_remasking.paper_experiment import run_paper_experiment, source_provenance
    from regret_remasking.paper_decode import decode_paper_policy
    from regret_remasking.locking_experiment import load_local_llada
    gate = ra.read_json(ALIGNED / "source_gate.json")
    if source_provenance(ALIGNED / "source")["sha256"] != gate["source_sha256"]:
        raise ValueError("aligned source changed")
    policies = [p for p in ra.read_json(ALIGNED / "policies.json") if p["policy_id"] == "B0"]
    preflight = ra.gpu_guard(initial=True)
    model, tokenizer = load_local_llada(CHECKPOINT, "cuda", "bf16")

    def checked_decode(*positional, **kwargs):
        ra.gpu_guard()
        return decode_paper_policy(*positional, **kwargs)

    manifests = [ALIGNED / "manifests/smoke_pair.jsonl"] if args.stage == "smoke" else [ALIGNED / "manifests/eval_math.jsonl", ALIGNED / "manifests/eval_code.jsonl"]
    run_paper_experiment(manifest_paths=manifests, output_dir=args.output_dir, checkpoint=CHECKPOINT, policies=policies,
                         repeats=1 if args.stage == "smoke" else 3, device="cuda", dtype="bf16", seed=23, gen_length=256,
                         model=model, tokenizer=tokenizer, decoder=checked_decode, collect_events=False, warmup=True,
                         purpose="same-job B0 anchor for collaborator methods")
    (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")


if __name__ == "__main__":
    main()
