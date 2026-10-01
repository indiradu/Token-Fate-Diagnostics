"""Freeze the aligned-protocol prompts (our build_prompt) so the collaborator's code sees identical text.

Runs in OUR environment (aligned campaign source). Writes prompts_eval.json and prompts_smoke.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

CAMPAIGN = Path(__file__).resolve().parents[1]
ALIGNED = CAMPAIGN.parent / "aligned_standard_g256_20260929"
sys.path.insert(0, str(ALIGNED / "source/src"))

from regret_remasking.paper_experiment import _resolve_prompt  # noqa: E402


def main() -> None:
    for name, files in [("eval", ["eval_math.jsonl", "eval_code.jsonl"]), ("smoke", ["smoke_pair.jsonl"])]:
        rows = []
        for file in files:
            path = ALIGNED / "manifests" / file
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                prompt = _resolve_prompt(entry)
                rows.append({
                    "dataset": entry["dataset"], "example_id": entry["example_id"], "question": entry["question"],
                    "gold_answer": entry["gold_answer"], "metadata": entry["metadata"], "prompt_text": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "manifest": file,
                    "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        out = CAMPAIGN / f"prompts_{name}.json"
        out.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
        print(name, len(rows), hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
