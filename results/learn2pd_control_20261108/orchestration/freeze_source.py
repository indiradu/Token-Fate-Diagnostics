"""Freeze the decoder source used by GPU runs: copies src/regret_remasking into source/ and records sha256 per file; also records which files differ from the far-cache
campaign's frozen source (results/farcache_stack_20261105/source) so the additions of this campaign are auditable.
Run (any python): freeze_source.py   -- re-run before each GPU claim whenever src/ changed; run_l2pd_stack.py verifies the hashes at start and imports source/src first."""
import difflib
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
dst = OUT / "source"
shutil.rmtree(dst / "src", ignore_errors=True)
shutil.copytree(ROOT / "src/regret_remasking", dst / "src/regret_remasking", ignore=shutil.ignore_patterns("__pycache__"))
files = sorted(p for p in dst.rglob("*.py"))
sha = {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
agg = hashlib.sha256("".join(f"{k}{v}" for k, v in sha.items()).encode()).hexdigest()
(dst / "SHA256.json").write_text(json.dumps({"files": sha, "aggregate": agg}, indent=1) + "\n")
old = ROOT / "results/farcache_stack_20261105/source/src/regret_remasking"
diff = {}
for p in sorted((dst / "src/regret_remasking").glob("*.py")):
    q = old / p.name
    if not q.exists():
        diff[p.name] = "NEW FILE"
    elif q.read_bytes() != p.read_bytes():
        diff[p.name] = "".join(difflib.unified_diff(q.read_text().splitlines(True), p.read_text().splitlines(True), "farcache/" + p.name, "learn2pd/" + p.name, n=0))
(OUT / "validation").mkdir(exist_ok=True)
(OUT / "validation/source_diff_vs_farcache.json").write_text(json.dumps({"changed_or_new_files": sorted(diff), "diffs": diff}, indent=1) + "\n")
print("frozen", len(files), "files; aggregate", agg[:16], "; differing from the far-cache source:", sorted(diff))
