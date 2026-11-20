"""GPU-transfer smoke (ciai, A100): LLaDA-8B, four arms interleaved prompt by prompt in ONE process, word lock Conf-0.9, no first-EOS truncation (all positions decoded), block 32, batch 1, bf16:
  DENSE_C09  dense Conf-0.9 (paper_decode C-block)                      the word lock only
  S32        plain selective cache (dense call at every block start; compute the current block and the rows written on the previous call; everything else served from stored K/V)
  R0_OWN     R0 derived from THIS GPU's cost benchmark (F, c from bench_layer_cost.json measured here)
  R0_RTX     R0 derived from the RTX 5000 Ada cost benchmark (the schedule we used on that GPU), run on THIS GPU
Both R0 arms use the same calls-per-block and prompt length (measured here by an 8-prompt S32 dry run) and differ only in F and c, so they differ only in the derived schedule (rejoining rows, period k, refreshes per block).
Usage: python run_transfer.py SET [SMOKE]   SET in gsm_held | he164; env: PHASELOCK_ROOT, MODEL_TAG=original, MODEL_CHECKPOINT, FAMILY_SRC, BENCH_JSON (this GPU), TAG (record name, default a100), PUB_LIMIT (default 40), SLICE_SECONDS (work budget; exit 75 when used up, records resume).
Records: <root>/records/transfer_<TAG>_<SET>.jsonl; derived rules: transfer_<TAG>_<SET>.rules.json. HumanEval answers are deferred (no bwrap on ciai): score them on the other cluster."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PHASELOCK_ROOT", "/nfs-stor/rui.gao/Researches/phaselock"))
sys.path.insert(0, str(ROOT / "results/model_transfer_20261113/orchestration"))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import run_rule as R  # noqa: E402
import torch  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.refresh_rule import Setting, config_kwargs, derive  # noqa: E402

OUT = ROOT / "records"
TAG = os.environ.get("TAG", "a100")
OWN_BENCH = Path(os.environ["BENCH_JSON"])
RTX_BENCH = ROOT / "results/refresh_family_20261111/records/bench_layer_cost.json"
NT = dict(eos_truncation=False)
ARMS = os.environ.get("ARMS_LIST", "DENSE_C09,S32,R0_OWN,R0_RTX").split(",")      # ORIG = the original decoder (B0: one token per step, block 32, all gen_length calls)


def cost_params(path):
    b = json.loads(Path(path).read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def decode(ctx, arm, e, gen, rules):
    model, tok = ctx["model"], ctx["tok"]
    if arm == "ORIG":
        cfg = PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="B0")
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "prompt_len": None}
    if arm == "DENSE_C09":
        cfg = PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=0.9)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "prompt_len": None}
    kw = {} if arm == "S32" else config_kwargs(rules[arm])
    res = C.decode_selective(model, C.prompt_ids(ctx, e), C.cached_cfg("C-0.9", ctx, gen_length=gen, **kw, **NT))
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)), "prompt_len": res.prompt_length,
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "refresh_kind": list(res.refresh_kind), "rows_per_call": list(res.rows_per_call)}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = R.SETS[ds]
    es = R.questions(ds)[: int(os.environ.get("PUB_LIMIT", "40"))]
    es_run = es[:2] if smoke else es
    budget = float(os.environ.get("SLICE_SECONDS", "1e9"))
    t_start = time.time()
    pre0 = f"{'smoke_' if smoke else ''}transfer_{TAG}_{ds}"
    done0 = {(r["example_id"], r["arm"]) for r in C.rows(OUT / f"{pre0}.jsonl") if r.get("status") == "ok"}
    if all((e["example_id"], a) in done0 for e in es_run for a in ARMS):                 # nothing left: exit before loading the model (successor slices of a finished chain)
        print(f"[transfer {TAG} {ds}] already complete", flush=True)
        return
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    OUT.mkdir(parents=True, exist_ok=True)
    pre = f"{'smoke_' if smoke else ''}transfer_{TAG}_{ds}"
    cal = [decode(ctx, "S32", e, gen, None) for e in es[: R.N_CAL]]
    setting_kw = dict(n_layers=32, block_length=32, gen_length=gen, prompt_len=float(np.mean([r["prompt_len"] for r in cal])), calls_per_block=float(np.mean([r["nfe"] / max(1, r["dense_calls"]) for r in cal])))
    rules, info = {}, {}
    for arm, bench in (("R0_OWN", OWN_BENCH), ("R0_RTX", RTX_BENCH)):
        F, c = cost_params(bench)
        rules[arm] = derive(Setting(fixed_ms=F, row_ms=c, **setting_kw))
        info[arm] = {"F_ms": F, "c_ms_per_row_layer": c, **{k: v for k, v in rules[arm].items() if k != "detail"}}
    (OUT / f"{pre}.rules.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "setting": setting_kw, "rules": info}, indent=1))
    print(json.dumps({"setting": setting_kw, "rules": info}, indent=1), flush=True)
    path = OUT / f"{pre}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in ARMS:
        decode(ctx, arm, es_run[0], gen, rules)                                   # warm-up
    for i, e in enumerate(es_run):
        if time.time() - t_start > budget:
            print(f"[transfer] work budget used after {i} prompts; records resume in the next slice", flush=True)
            sys.exit(75)
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            d = decode(ctx, arm, e, gen, rules)
            correct, status = (None, "deferred") if (R.DEFER and ds in R.CODE) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            C.append(path, dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=C.first_eos(d["tokens"]),
                                status="ok", gpu=torch.cuda.get_device_name(0)))
        print(f"[transfer {TAG} {ds}] {i + 1}/{len(es_run)} {e['example_id']}", flush=True)
    print(f"[transfer {TAG} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
