"""S3: GPU and stack sanity on the allocated GPU: torch sees it, the LLaDA-8B checkpoint loads, three dense forwards are timed. Writes records/sanity_<gpu>.json."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(os.environ.get("PHASELOCK_ROOT", "/nfs-stor/rui.gao/Researches/phaselock"))
sys.path.insert(0, str(ROOT / "results/model_transfer_20261113/orchestration"))
import common as C
import torch
C.gpu_guard()
info = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__, "cuda": torch.version.cuda, "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
print(info, flush=True)
t = time.time(); model, tok = C.load_model(); info["load_s"] = round(time.time() - t, 1)
x = torch.randint(1000, 50000, (1, 690), device="cuda")
with torch.no_grad():
    for _ in range(3):
        torch.cuda.synchronize(); t = time.perf_counter(); model(x); torch.cuda.synchronize(); info.setdefault("dense_forward_ms", []).append(round((time.perf_counter() - t) * 1e3, 1))
info["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
(ROOT / "records").mkdir(exist_ok=True)
(ROOT / "records" / "sanity.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))
