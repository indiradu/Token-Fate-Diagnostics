"""Freeze the decoder source used by GPU runs: copies src/regret_remasking and scripts/run_elastic_cache_official_eval.py into source/ and records sha256 per file.
Run (any python): freeze_source.py   -- re-run before each GPU claim whenever src/ changed; GPU scripts import from source/src first."""
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
dst = OUT / "source"
shutil.rmtree(dst / "src", ignore_errors=True)
shutil.copytree(ROOT / "src/regret_remasking", dst / "src/regret_remasking", ignore=shutil.ignore_patterns("__pycache__"))
(dst / "scripts").mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "scripts/run_elastic_cache_official_eval.py", dst / "scripts/run_elastic_cache_official_eval.py")
files = sorted(p for p in dst.rglob("*.py"))
sha = {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
agg = hashlib.sha256("".join(f"{k}{v}" for k, v in sha.items()).encode()).hexdigest()
(dst / "SHA256.json").write_text(json.dumps({"files": sha, "aggregate": agg}, indent=1) + "\n")
print("frozen", len(files), "files; aggregate", agg[:16], "selective_cache.py", sha["src/regret_remasking/selective_cache.py"][:16], "paper_decode.py", sha["src/regret_remasking/paper_decode.py"][:16])
