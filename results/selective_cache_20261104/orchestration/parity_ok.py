"""Exit 0 iff the pre-registered parity gate passed: our engine on the OFFICIAL model object (manual softmax, window head) reproduced the official tokens on every question."""
import json, sys
from pathlib import Path
d = json.loads((Path(__file__).resolve().parents[1] / "validation/parity_official.json").read_text())
ok = all(q["mine_on_official_model"]["manual_window_head"]["tokens_equal"] for q in d["questions"])
print("parity gate (official model, manual softmax):", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
