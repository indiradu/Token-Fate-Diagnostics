"""CPU gate for the evaluation job: both refit rule files exist, their sha256 match their rules.json, they load, and the original rule is untouched (sha256 equals the original rules.json).
Usage: <python> check_rules.py   -> exit 0 or a message on stderr and exit 1. Imports only numpy / sklearn / pickle (no GPU)."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / "source/src"))
from regret_remasking.context_rules import ContextRule  # noqa: E402

WL = OUT.parents[1] / "results/longlen_wordlock_20261103"
bad = []
for path, meta, key in [(WL / "rules/math512/Lctx_c0.8.pkl", WL / "rules/math512/rules.json", "Lctx_c0.8"),
                        (OUT / "rules/R-S16/Lctx_refit_S16_c0.8.pkl", OUT / "rules/R-S16/rules.json", "Lctx_refit_S16_c0.8"),
                        (OUT / "rules/R-S32c/Lctx_refit_S32c_c0.8.pkl", OUT / "rules/R-S32c/rules.json", "Lctx_refit_S32c_c0.8")]:
    if not path.exists() or not meta.exists():
        bad.append(f"missing: {path if not path.exists() else meta}")
        continue
    m = json.loads(meta.read_text())["rules"][key]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != m["sha256"]:
        bad.append(f"sha256 mismatch: {path}")
        continue
    r = ContextRule.load(path)
    if abs(r.lam - m["lambda"]) > 1e-12 or r.kind != "gbm":
        bad.append(f"rule does not match its rules.json: {path}")
    print("ok", path.name, sha[:12], "lambda", r.lam)
if bad:
    print("RULE CHECK FAILED:", *bad, sep="\n  ", file=sys.stderr)
    sys.exit(1)
