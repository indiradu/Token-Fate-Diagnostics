"""Run the aligned standard-LLaDA comparison (block 32, one token per step, 256/256, batch 1).

Stages: smoke (2 cases, timed + diagnostic, parity and mechanism checks), full (124 cases x
6 policies x 3 repeats), finalize (validation, parity report, FLOPs, paired bootstrap, results.md).
Adapted from results/surelock_token_vs_tracelock_20260922_2145/orchestration/run_comparison_frozen.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


CAMPAIGN = Path(__file__).resolve().parents[1]
ROOT = CAMPAIGN.parents[1]
SOURCE = CAMPAIGN / "source"
CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
P4 = ROOT / "results/representation_kl_last_hidden_20260921_203301"
P7 = ROOT / "results/surelock_token_vs_tracelock_20260922_2145"
# New policy id -> (earlier campaign, earlier policy id) with an identical decode configuration.
PARITY = {
    "B0": (P4, "B0"),
    "RepKL-freeze-0.0005": (P4, "R-sparse-eps-0.0005"),
    "TraceLock-0.95": (P7, "T-adaptive-0.95"),
}
TOKEN_ONLY_ARMS = {"B0", "C-block", "T-adaptive"}
ORDER = ["B0", "RepKL-freeze-0.0005", "SureLock-gen-0.005", "SureLock-prompt-0.005", "TraceLock-0.95", "Confidence-0.9"]
WORK_FIELDS = ["token_ids", "nfe", "frozen_count", "total_query_rows", "total_projection_rows", "total_ffn_rows", "total_output_head_rows", "controller_calls"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_guard(initial=False):
    import torch
    props = torch.cuda.get_device_properties(0)
    if props.name != "NVIDIA RTX 5000 Ada Generation":
        raise RuntimeError(f"unvalidated GPU: {props.name}")
    gpu_uuid = "GPU-" + str(props.uuid).removeprefix("GPU-")
    output = subprocess.run(["nvidia-smi", "--id=" + gpu_uuid, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [line for line in output.stdout.splitlines() if line.split(",")[0].strip().isdigit() and int(line.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    free, total = torch.cuda.mem_get_info(0)
    if initial and free < 30 * 1024**3:
        raise RuntimeError(f"GPU not empty before load: {free} bytes free")
    return {"gpu": props.name, "uuid": gpu_uuid, "free_bytes": free, "total_bytes": total, "other_processes": others, "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def key(row):
    return row["dataset"], row["example_id"], row["policy_id"], row["repeat"]


def validate_run(run_dir, expected_cases, repeats):
    policies = read_json(CAMPAIGN / "policies.json")
    policy_by_id = {p["policy_id"]: p for p in policies}
    config_path = run_dir / "resolved_config.json"
    config = read_json(config_path)
    config_sha = sha256(config_path)
    gate = read_json(CAMPAIGN / "source_gate.json")
    expected_panel = {(e["dataset"], e["example_id"]) for e in config["examples"]}
    data = rows(run_dir / "records.jsonl")
    expected_ids = {p["policy_id"] for p in policies}
    seen = set()
    grouped = defaultdict(list)
    for row in data:
        if row["source_sha256"] != gate["source_sha256"] or row["config_sha256"] != config_sha:
            raise ValueError("record source/config provenance differs")
        if row["checkpoint_index_sha256"] != config["checkpoint"]["checkpoint_index_sha256"]:
            raise ValueError("model provenance differs")
        policy = policy_by_id[row["policy_id"]]
        for field in ["arm", "epsilon", "threshold", "confidence_percentile", "kl_space", "allow_newly_unmasked", "controller_checkpoint_sha256", "projector_checkpoint_sha256"]:
            if row[field] != policy[field]:
                raise ValueError(f"record policy differs: {field}")
        if row["status"] != "ok" or type(row["correct"]) is not bool:
            raise ValueError("failed or unscorable record")
        if row["final_mask_count"] or row["immutable_token_violations"] or len(row["token_ids"]) != 256:
            raise ValueError("invalid terminal output")
        if row["nfe"] != row["forward_calls"] or not 0 < row["nfe"] <= 256:
            raise ValueError("invalid model-call count")
        if row["arm"] == "B0" and row["nfe"] != 256:
            raise ValueError("standard baseline must make exactly 256 calls")
        if key(row) in seen:
            raise ValueError("duplicate evaluation record")
        seen.add(key(row))
        grouped[key(row)[:3]].append(row)
    cases = {k[:2] for k in grouped}
    if cases != expected_panel or len(cases) != expected_cases or len(data) != expected_cases * len(policies) * repeats:
        raise ValueError("incomplete evaluation panel")
    for case in cases:
        for policy in expected_ids:
            group = grouped[(*case, policy)]
            if {r["repeat"] for r in group} != set(range(1, repeats + 1)):
                raise ValueError("incomplete repeat set")
            if any(any(r[f] != group[0][f] for f in WORK_FIELDS + ["correct"]) for r in group):
                raise ValueError("repeat outputs/quality/work differ")
            if group[0]["arm"] in TOKEN_ONLY_ARMS and group[0]["frozen_count"]:
                raise ValueError("token-only method cached representations")
            if policy_by_id[policy]["freeze_scope"] == "generated" and group[0]["frozen_prompt_count"]:
                raise ValueError("generated-only policy froze prompt rows")
    return data


def _old_records(campaign, subsets):
    """Repeat-1 records of one earlier campaign, keyed by (dataset, example_id, policy_id)."""
    found = {}
    for path in subsets:
        for row in rows(campaign / path):
            if row["repeat"] == 1:
                found[(row["dataset"], row["example_id"], row["policy_id"])] = row
    return found


def parity(data, old, fields):
    """Compare repeat-1 outputs with an earlier identical configuration; return per-policy counts."""
    report = {}
    for new_id, (campaign, old_id) in PARITY.items():
        compared = matched = 0
        mismatched = []
        for row in data:
            if row["policy_id"] != new_id or row["repeat"] != 1:
                continue
            # Look up in that campaign only: policy ids such as "B0" mean different things in different campaigns.
            reference = old[campaign].get((row["dataset"], row["example_id"], old_id))
            if reference is None:
                continue
            compared += 1
            diffs = [f for f in fields if row.get(f) != reference.get(f)]
            if diffs:
                mismatched.append({"dataset": row["dataset"], "example_id": row["example_id"], "fields": diffs})
            else:
                matched += 1
        report[new_id] = {"reference": f"{campaign.name}:{old_id}", "compared": compared, "matched": matched, "mismatched": mismatched}
    return report


def smoke_checks():
    timed = validate_run(CAMPAIGN / "smoke_timed", 2, 1)
    diagnostic = validate_run(CAMPAIGN / "smoke_diagnostic", 2, 1)
    by_key = {key(row): row for row in diagnostic}
    for row in timed:
        other = by_key[key(row)]
        for name in WORK_FIELDS + ["frozen_prompt_count", "correct"]:
            if row[name] != other[name]:
                raise ValueError(f"timed/diagnostic mismatch: {name}")
    old = {
        P4: _old_records(P4, ["e1_smoke/smoke_math_timed/attempt_01/records.jsonl", "e1_smoke/smoke_code_timed/attempt_01/records.jsonl"]),
        P7: _old_records(P7, ["smoke_timed/records.jsonl"]),
    }
    report = parity(timed, old, ["token_ids", "nfe", "frozen_count", "total_query_rows", "total_output_head_rows"])
    for policy_id, result in report.items():
        if result["compared"] != 2 or result["matched"] != 2:
            raise ValueError(f"earlier identical configuration not reproduced for {policy_id}: {result}")
    by_case = defaultdict(dict)
    for row in timed:
        by_case[(row["dataset"], row["example_id"])][row["policy_id"]] = row
    for case, group in by_case.items():
        if group["B0"]["nfe"] != 256 or group["B0"]["frozen_count"]:
            raise ValueError("baseline is not one token per step")
        for policy_id in ["RepKL-freeze-0.0005", "SureLock-gen-0.005", "SureLock-prompt-0.005"]:
            row = group[policy_id]
            if not row["frozen_count"] or not row["total_query_rows"] < row["nfe"] * (row["prompt_length"] + 256):
                raise ValueError(f"{policy_id} did not remove work on {case}")
        if group["SureLock-gen-0.005"]["frozen_prompt_count"]:
            raise ValueError("generated-only SureLock froze prompt rows")
        if not group["SureLock-prompt-0.005"]["frozen_prompt_count"]:
            raise ValueError(f"prompt-eligible SureLock never froze a prompt row on {case}")
        if not group["SureLock-prompt-0.005"]["total_query_rows"] < group["SureLock-gen-0.005"]["total_query_rows"]:
            raise ValueError(f"prompt freezing did not remove extra work on {case}")
        if not group["Confidence-0.9"]["nfe"] < 256:
            raise ValueError("confidence threshold never committed more than one token")
    freeze_events = masked_freezes = multi_commit_steps = 0
    for record in rows(CAMPAIGN / "smoke_diagnostic/events.jsonl"):
        commits_per_step = defaultdict(int)
        for event in record["events"]:
            if event["type"] == "reference_freeze":
                freeze_events += 1
                masked_freezes += int(bool(event["input_was_masked"]))
            if record["policy_id"] == "Confidence-0.9" and event["type"] == "token_commit":
                commits_per_step[event["global_step"]] += 1
        multi_commit_steps += sum(1 for count in commits_per_step.values() if count > 1)
    if masked_freezes:
        raise ValueError("cached a MASK-input representation")
    if not freeze_events or not multi_commit_steps:
        raise ValueError("smoke did not exercise freezing and multi-token commits")
    write_json(CAMPAIGN / "validation/smoke_pass.json", {
        "passed": True, "cases": 2, "policies": len(ORDER), "timed_and_diagnostic_records": len(timed) + len(diagnostic),
        "freeze_events": freeze_events, "confidence_multi_commit_steps": multi_commit_steps, "parity": report,
        "prompt_rows_frozen": {c: g["SureLock-prompt-0.005"]["frozen_prompt_count"] for c, g in ((f"{k[0]}/{k[1]}", v) for k, v in by_case.items())},
    })


def _bootstrap(cases, stat, reps=10000, seed=23):
    rng = random.Random(seed)
    n = len(cases)
    values = sorted(stat([cases[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return values[int(0.025 * reps)], values[int(0.975 * reps) - 1]


def pooled_stats(data):
    """Paired, example-clustered statistics over the whole panel against B0."""
    latency = defaultdict(list)
    first = {}
    for row in data:
        latency[(row["dataset"], row["example_id"], row["policy_id"])].append(row["latency_s"])
        if row["repeat"] == 1:
            first[(row["dataset"], row["example_id"], row["policy_id"])] = row
    cases = sorted({(d, e) for d, e, _ in first})
    out = {}
    for policy in ORDER:
        paired = []
        for dataset, example in cases:
            base, cand = first[(dataset, example, "B0")], first[(dataset, example, policy)]
            paired.append({
                "dataset": dataset,
                "base_correct": int(base["correct"]), "cand_correct": int(cand["correct"]),
                "base_time": statistics.median(latency[(dataset, example, "B0")]),
                "cand_time": statistics.median(latency[(dataset, example, policy)]),
                "changed": int(base["token_ids"] != cand["token_ids"]),
                "nfe": cand["nfe"],
            })
        def acc_delta(sample):
            return sum(p["cand_correct"] - p["base_correct"] for p in sample) / len(sample)
        def speedup(sample):
            return 100.0 * (sum(p["base_time"] for p in sample) / sum(p["cand_time"] for p in sample) - 1.0)
        summary = {}
        for scope in ["math500", "humaneval", "overall"]:
            subset = [p for p in paired if scope == "overall" or p["dataset"] == scope]
            summary[scope] = {
                "n": len(subset),
                "correct": sum(p["cand_correct"] for p in subset),
                "baseline_correct": sum(p["base_correct"] for p in subset),
                "accuracy_delta_pp": 100 * acc_delta(subset),
                "accuracy_delta_pp_ci95": [100 * v for v in _bootstrap(subset, acc_delta)] if policy != "B0" else [0.0, 0.0],
                "seconds_per_example": sum(p["cand_time"] for p in subset) / len(subset),
                "speedup_pct": speedup(subset),
                "speedup_pct_ci95": list(_bootstrap(subset, speedup)) if policy != "B0" else [0.0, 0.0],
                "mean_nfe": sum(p["nfe"] for p in subset) / len(subset),
                "sequence_changed": sum(p["changed"] for p in subset),
                "correct_to_wrong": sum(1 for p in subset if p["base_correct"] and not p["cand_correct"]),
                "wrong_to_correct": sum(1 for p in subset if p["cand_correct"] and not p["base_correct"]),
            }
        out[policy] = summary
    return out


def finalize():
    sys.path.insert(0, str(CAMPAIGN / "orchestration"))
    from locking_flop_accounting import analyze_records, load_model_config, ProjectorConfig, ControllerConfig
    from regret_remasking.locking_analysis import analyze_run
    data = validate_run(CAMPAIGN / "full", 124, 3)
    old = {
        P4: _old_records(P4, ["e1_eval/merged_records/records.jsonl"]),
        P7: _old_records(P7, ["full/records.jsonl"]),
    }
    trace_fields = ["token_ids", "nfe", "prompt_length", "total_query_rows", "total_projection_rows", "total_ffn_rows", "total_output_head_rows", "controller_calls", "controller_checkpoint_sha256", "projector_checkpoint_sha256"]
    report = parity(data, old, ["token_ids", "nfe", "frozen_count", "total_query_rows", "total_output_head_rows"])
    trace_report = parity([r for r in data if r["policy_id"] == "TraceLock-0.95"], old, trace_fields)["TraceLock-0.95"]
    write_json(CAMPAIGN / "analysis/parity.json", {"output_parity": report, "tracelock_profile_parity": trace_report})
    profile_path = None
    if trace_report["compared"] == 124 and trace_report["matched"] == 124:
        profiles = rows(P7 / "analysis/verified_controller_profiles.jsonl")
        for profile in profiles:
            if profile["policy_id"] != "T-adaptive-0.95":
                raise ValueError("unexpected controller profile policy")
            profile["policy_id"] = "TraceLock-0.95"
        profile_path = CAMPAIGN / "analysis/verified_controller_profiles.jsonl"
        profile_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in profiles))
        pc, cc = profiles[0]["projector_config"], profiles[0]["controller_config"]
        flops = analyze_records(CAMPAIGN / "full/records.jsonl", load_model_config(CHECKPOINT / "config.json"), CAMPAIGN / "analysis/flops", expected_cases=124,
            purpose="aligned standard LLaDA comparison", profiles_path=profile_path,
            projector_config=ProjectorConfig(pc["hidden_dim"], pc["state_hidden_dim"], pc["state_dim"], pc["delta_hidden_dim"], pc["delta_dim"]),
            controller_config=ControllerConfig(cc["model_dim"], cc["layers"], cc["ff_dim"]))
    else:
        flops = analyze_records(CAMPAIGN / "full/records.jsonl", load_model_config(CHECKPOINT / "config.json"), CAMPAIGN / "analysis/flops", expected_cases=124,
            purpose="aligned standard LLaDA comparison (controller FLOPs pending: TraceLock outputs differ from profiled run)")
    analyze_run(CAMPAIGN / "full", CAMPAIGN / "analysis/paired")
    pooled = pooled_stats(data)
    write_json(CAMPAIGN / "analysis/pooled_summary.json", pooled)
    overall = {r["policy_id"]: r for r in flops["aggregate"] if r["dataset"] == "overall"}
    base_flops = overall["B0"]["total_with_lm_head_tflops_per_example"]
    lines = [
        "# Aligned standard-LLaDA comparison",
        "",
        "Setting: LLaDA-8B-Instruct BF16, standard low-confidence sampler (block 32, one token per step), 256 output tokens / 256 steps, batch 1, seed 23, 54 MATH Level-5 + 70 HumanEval, one warm-up and three timing repeats with rotating policy order. Correctness uses repeat 1; speed uses paired sums of per-example median latency. Intervals are 10,000-sample bootstraps over examples.",
        "",
        "| Policy | MATH /54 | Code /70 | Total /124 | Δ correct vs B0 (pp) [95% CI] | s / example | Speedup vs B0 [95% CI] | Mean calls | Extended TFLOPs | FLOP cut | Outputs changed | Broken / fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ORDER:
        p = pooled[policy]
        o = p["overall"]
        r = overall[policy]
        controller = r.get("controller_total_tflops_per_example") or 0.0
        total = r["total_with_lm_head_tflops_per_example"] + controller
        pending = " (controller pending)" if r.get("controller_profile_pending_cases") else ""
        lo, hi = o["accuracy_delta_pp_ci95"]
        slo, shi = o["speedup_pct_ci95"]
        lines.append(
            f"| {policy} | {p['math500']['correct']} | {p['humaneval']['correct']} | {o['correct']} | "
            f"{o['accuracy_delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {o['seconds_per_example']:.3f} | "
            f"{o['speedup_pct']:+.2f}% [{slo:+.2f}, {shi:+.2f}] | {o['mean_nfe']:.1f} | {total:,.1f}{pending} | {100*(1-total/base_flops):.2f}% | "
            f"{o['sequence_changed']}/124 | {o['correct_to_wrong']} / {o['wrong_to_correct']} |"
        )
    lines += [
        "",
        "Policies: B0 is the standard LLaDA sampler. RepKL-freeze-0.0005 freezes a generated token's all-layer state and skips its rows once final-layer representation KL ≤ 0.0005. SureLock-gen/-prompt use vocabulary KL ≤ 0.005 plus the top-20% confidence gate; -gen freezes generated tokens only, -prompt also freezes prompt rows. Both freeze only after the token has been seen unmasked, and never EOS/EOT. TraceLock-0.95 is the trained controller with its own 32-token sliding window. Confidence-0.9 commits every block position with confidence ≥ 0.9 (else the single most confident), Fast-dLLM-style, inside the standard blocks.",
        "",
        "Parity with earlier identical configurations is in analysis/parity.json; per-dataset bootstrap tables are in analysis/paired/summary.md; FLOPs follow the SureLock §2.4 convention (2 FLOPs/MAC, GEMMs only) plus the vocabulary head and controller.",
    ]
    (CAMPAIGN / "results.md").write_text("\n".join(lines) + "\n")
    write_json(CAMPAIGN / "completion.json", {"complete": True, "cases": 124, "policies": len(ORDER), "repeats": 3, "records": len(data), "records_sha256": sha256(CAMPAIGN / "full/records.jsonl"), "source": read_json(CAMPAIGN / "source_gate.json"), "parity_summary": {k: f"{v['matched']}/{v['compared']}" for k, v in report.items()}, "completed_at": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "smoke_check", "full", "finalize"])
    args = parser.parse_args()
    os.environ["HF_MODULES_CACHE"] = str(CAMPAIGN / "runtime_cache/model_modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(SOURCE / "src"))
    from regret_remasking.paper_experiment import run_paper_experiment, source_provenance
    from regret_remasking.paper_decode import decode_paper_policy
    from regret_remasking.locking_experiment import load_local_llada
    source = source_provenance(SOURCE)
    gate = read_json(CAMPAIGN / "source_gate.json")
    if source["sha256"] != gate["source_sha256"] or sha256(CAMPAIGN / "policies.json") != gate["policy_sha256"]:
        raise ValueError("frozen source or policies changed")
    for name, expected in gate["manifest_sha256"].items():
        if sha256(CAMPAIGN / "manifests" / name) != expected:
            raise ValueError("frozen manifest changed")
    try:
        write_json(CAMPAIGN / "state.json", {"active": True, "stage": args.stage, "pid": os.getpid(), "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "updated_at": datetime.now(timezone.utc).isoformat()})
        if args.stage == "finalize":
            finalize()
        elif args.stage == "smoke_check":
            # Re-run only the acceptance checks on completed smoke outputs (CPU).
            smoke_checks()
        else:
            if args.stage == "full" and not read_json(CAMPAIGN / "validation/smoke_pass.json")["passed"]:
                raise ValueError("smoke acceptance required")
            write_json(CAMPAIGN / f"validation/{args.stage}_gpu_preflight.json", gpu_guard(initial=True))
            from regret_remasking.code_eval import evaluate_humaneval_program
            for expression, expected in [("x+1", "passed"), ("0", "failed_tests")]:
                result = evaluate_humaneval_program(f"def solve(x):\n    return {expression}\n", "def solve(x):\n", "def check(candidate):\n    assert candidate(2)==3\n", "solve")
                if result.status != expected:
                    raise ValueError(f"scoring preflight failed: {result.status}")
            model, tokenizer = load_local_llada(CHECKPOINT, "cuda", "bf16")

            def checked_decode(*positional, **kwargs):
                gpu_guard()
                result = decode_paper_policy(*positional, **kwargs)
                gpu_guard()
                return result

            policies = read_json(CAMPAIGN / "policies.json")
            stages = [("smoke_timed", False), ("smoke_diagnostic", True)] if args.stage == "smoke" else [("full", False)]
            manifests = [CAMPAIGN / "manifests/smoke_pair.jsonl"] if args.stage == "smoke" else [CAMPAIGN / "manifests/eval_math.jsonl", CAMPAIGN / "manifests/eval_code.jsonl"]
            for name, events in stages:
                run_paper_experiment(manifest_paths=manifests, output_dir=CAMPAIGN / name, checkpoint=CHECKPOINT, policies=policies,
                    repeats=1 if args.stage == "smoke" else 3, device="cuda", dtype="bf16", seed=23, gen_length=256,
                    model=model, tokenizer=tokenizer, decoder=checked_decode, collect_events=events, warmup=True,
                    purpose="smoke" if args.stage == "smoke" else "evaluation")
            if args.stage == "smoke":
                smoke_checks()
        write_json(CAMPAIGN / "state.json", {"active": False, "stage": args.stage + "_complete", "updated_at": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        write_json(CAMPAIGN / "state.json", {"active": False, "stage": args.stage + "_failed", "error": f"{type(error).__name__}: {error}", "updated_at": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
