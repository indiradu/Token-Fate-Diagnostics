"""The pre-registered gate (PREREGISTERED.md, section Gate). Importable (evaluate_gate) and runnable: <python> check_gate.py -> exit 0 if the gate passes, 1 otherwise (and if the ablation result is missing).
Reads validation/ablation_S32c.json written by fit_ablation.py. Writes validation/gate.json either way when the ablation result exists. Pure python (json), no numpy needed.

"Best set" = the one of six+a, six+b, six+c, six+d, six+all with the largest admitted share at C-0.8's pooled precision (point estimate).
G0  validity: V2 (logged extras equal their post-hoc derivation on all 80 traces) and V3 (the six-input control reproduces the refit: AUROC within 0.002 of the refit's, admitted share within 0.01).
G1  admitted share of the best set minus that of the six-input control >= 0.05.
G2  AUROC difference to the control >= +0.01 AND the 2.5th percentile of its paired cluster-bootstrap distribution > 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
CANDIDATES = ["six+a", "six+b", "six+c", "six+d", "six+all"]
G1_MIN_SHARE_GAIN = 0.05
G2_MIN_AUROC_GAIN = 0.01
V3_AUROC_TOL = 0.002
V3_SHARE_TOL = 0.01


def evaluate_gate(res: dict) -> dict:
    sets = res["sets"]
    best = max(CANDIDATES, key=lambda k: sets[k]["share_diff_vs_six"]["point"])
    b = sets[best]
    v = res["validity"]
    g0 = bool(v["V2_derived_features_equal"] and v["V3_control_reproduces_refit"])
    g1 = bool(b["share_diff_vs_six"]["point"] >= G1_MIN_SHARE_GAIN)
    g2 = bool(b["auroc_diff_vs_six"]["point"] >= G2_MIN_AUROC_GAIN and b["auroc_diff_vs_six"]["ci95"][0] > 0.0)
    return {"best_set": best, "G0_valid": g0, "G1_share_gain_ge_5pt": g1, "G2_auroc_gain_ge_0.01_and_ci_above_0": g2, "passes": bool(g0 and g1 and g2),
            "best_share_diff": b["share_diff_vs_six"], "best_auroc_diff": b["auroc_diff_vs_six"], "thresholds": {"G1": G1_MIN_SHARE_GAIN, "G2": G2_MIN_AUROC_GAIN, "V3_auroc_tol": V3_AUROC_TOL, "V3_share_tol": V3_SHARE_TOL}}


def main() -> int:
    path = OUT / "validation/ablation_S32c.json"
    if not path.exists():
        print("GATE FAILED: validation/ablation_S32c.json does not exist (collection or ablation did not finish)", file=sys.stderr)
        return 1
    res = json.loads(path.read_text())
    gate = evaluate_gate(res)
    if gate["passes"]:
        meta = OUT / "rules/X/rules.json"
        rule = OUT / "rules/X/Lctx_X_c0.8.pkl"
        if not (meta.exists() and rule.exists()):
            gate["passes"] = False
            gate["note"] = "gate conditions hold but the rule file was not written"
    (OUT / "validation/gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps(gate, indent=1))
    if not gate["passes"]:
        print("GATE FAILED: stage H will not run", file=sys.stderr)
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
