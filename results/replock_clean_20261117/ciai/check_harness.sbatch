#!/bin/bash
#SBATCH -J pl-check
#SBATCH -p cscc-cpu-p
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o /nfs-stor/rui.gao/Researches/phaselock/logs/check-%j.out
# CPU-only dry check of the harness: imports resolve with the frozen source, prompts load, scorer for GSM8K works, and R0 derives for all four sets with both cost benchmarks (the RTX one here; this GPU's benchmark comes later).
P=/nfs-stor/rui.gao/Researches/phaselock
export PHASELOCK_ROOT=$P MODEL_TAG=original MODEL_CHECKPOINT=/nfs-stor/rui.gao/models/LLaDA-8B-Instruct FAMILY_SRC=$P/results/model_transfer_20261113/source_m1 DEPTH_DEFER_SCORING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
cd $P/results/model_transfer_20261113/orchestration
/nfs-stor/rui.gao/envs/phaselock/bin/python - <<'PY'
import sys, json
sys.path.insert(0, ".")
import common as C, run_rule as R
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.refresh_rule import Setting, derive, config_kwargs
print("imports ok; source:", C.SRC)
for ds in ("gsm_held", "he164"):
    es = R.questions(ds); print(ds, "prompts", len(es), "gen", R.SETS[ds])
b = json.load(open(f"{C.ROOT}/results/refresh_family_20261111/records/bench_layer_cost.json"))["cells"]
print("RTX bench cells:", {k: round(v["wall_med_ms"], 1) for k, v in b.items()})
import numpy as np
pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
print("RTX F, c:", round(float(F), 2), round(float(c), 4))
for ds, (G, Pm, cpb) in {"he164": (512, 144, 3.19), "kod_held": (512, 369, 5.64), "gsm_held": (256, 98, 12.27), "math108": (512, 136, 12.14)}.items():
    d = derive(Setting(32, 32, G, Pm, cpb, float(F), float(c))); print(ds, d["partial_join"], "k", d["partial_every_k"], "m", d["refreshes_per_block"])
PY
echo "check exit $?"
