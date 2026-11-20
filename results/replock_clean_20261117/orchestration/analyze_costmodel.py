"""Experiment 5 (premise P2, CPU only): does the two-parameter cost model (time per call = F + c x mean rows over layers; F, c measured once per model on this GPU) predict the measured speed of every arm, and does a FLOP-proportional model (time ~ rows only) fail?
For every engine-arm record with per-call rows per layer (R0, controller, Elastic-Cache, plain cache) the predicted time is the sum over calls of F + c x mean_l(rows_l); for dense decodes every call has T = prompt + gen rows.
Per (model, set): measured speed of each arm over dense Conf-0.9 (ratio of summed latencies), the cost-model prediction, and the FLOP-proportional prediction (rows only). Output: results_costmodel.md."""
import json
from pathlib import Path

import numpy as np

R = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results")
REC = R / "replock_clean_20261117/records"
BENCH = {"llada8b": R / "refresh_family_20261111/records/bench_layer_cost.json", "llada15": R / "model_transfer_20261113/llada15/records/bench_layer_cost.json",
         "dream": R / "model_transfer_20261113/dream/records/bench_layer_cost.json", "moe": REC / "moe_bench_layer_cost.json"}


def fc(path):
    b = json.loads(Path(path).read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


out = ["# Cost-model prediction of speed (Study 5 engine arms)", "Time per call = F + c x (mean rows over layers); F, c from each model's cost benchmark. 'FLOP-proportional' = rows only (F = 0), scaled to match the dense decode.",
       "Speed = dense Conf-0.9 time / arm time (sum over prompts). Error = predicted / measured speed - 1.", "",
       "| model | set | arm | measured speed | cost-model speed (error) | FLOP-proportional speed (error) | prompts |", "|---|---|---|---|---|---|---|"]
allerr_cm, allerr_fl = [], []
for model, tag in (("LLaDA-8B", "llada8b"), ("LLaDA-1.5", "llada15"), ("Dream", "dream"), ("LLaDA-MoE", "moe")):
    F, c = fc(BENCH[tag])
    for ds in ("he164", "kod_held", "gsm_held", "math108"):
        recs = rows(REC / f"eng_{tag}_{ds}.jsonl")
        if not recs:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["arm"], {})[r["example_id"]] = r
        if "DENSE_C09" not in by or "R0" not in by:
            continue
        gen = recs[0]["gen"]
        T = {i: by["R0"][i]["prompt_len"] + gen for i in by["R0"]}
        dense_meas = {i: by["DENSE_C09"][i]["latency_s"] for i in by["DENSE_C09"]}
        dense_pred = {i: by["DENSE_C09"][i]["nfe"] * (F + c * T[i]) / 1000 for i in by["DENSE_C09"] if i in T}
        dense_rows = {i: by["DENSE_C09"][i]["nfe"] * T[i] for i in by["DENSE_C09"] if i in T}
        for arm in ("R0", "CTL", "ELASTIC"):
            if arm not in by:
                continue
            ids = [i for i in by[arm] if i in dense_pred and "layer_rows" in by[arm][i]]
            if not ids:
                continue
            pred = {i: sum(F + c * float(np.mean(lr)) for lr in by[arm][i]["layer_rows"]) / 1000 for i in ids}
            rws = {i: sum(float(np.mean(lr)) for lr in by[arm][i]["layer_rows"]) for i in ids}
            meas = sum(dense_meas[i] for i in ids) / sum(by[arm][i]["latency_s"] for i in ids)
            cm = sum(dense_pred[i] for i in ids) / sum(pred.values())
            fl = sum(dense_rows[i] for i in ids) / sum(rws.values())
            allerr_cm.append(abs(cm / meas - 1)); allerr_fl.append(abs(fl / meas - 1))
            out.append(f"| {model} | {ds} | {arm} | {meas:.2f}x | {cm:.2f}x ({100 * (cm / meas - 1):+.0f}%) | {fl:.2f}x ({100 * (fl / meas - 1):+.0f}%) | {len(ids)} |")
out += ["", f"Mean absolute error over {len(allerr_cm)} arm-cells: cost model {100 * np.mean(allerr_cm):.1f}%, FLOP-proportional {100 * np.mean(allerr_fl):.1f}%."]
(REC.parent / "results_costmodel.md").write_text("\n".join(out))
print("\n".join(out))
