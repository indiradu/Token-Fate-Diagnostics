"""GPU parity gate for src/regret_remasking/selective_cache.py against the OFFICIAL Elastic-Cache generator (3 MATH train questions, gen 256).

Usage: <abs venv python> parity_gate.py official    official model object (meta device + dispatch): official generator (plain and instrumented copy),
                                                     the official full-recompute baseline, and OUR engine run on the same official-model weights/code
                                                     (isolates our algorithm from the model implementation)
       <abs venv python> parity_gate.py ours        our HF LLaDA model: our engine with the manual-bf16 and the fused-SDPA attention, compared with the saved official logs
Writes validation/parity_{official,ours}.json (tokens and per-call logs of every variant, first diverging call).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
STAGE = ROOT / "results/repfreeze_tests_20261029/compare_stage"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(STAGE / "hf_modules")
sys.path.insert(0, str(STAGE / "pylib"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / "source/scripts"))
sys.path.insert(0, str(OUT / "source/src"))   # frozen copy first (freeze_source.py)
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
GEN = 256
MASK_ID, EOS_ID = 126336, 126081
IDX = (0, 20, 1)       # two L5 and one L1-4 questions of the Part A training manifest


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def prompt_ids_official(tokenizer, entry):
    from regret_remasking.data import build_prompt
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(entry["question"], entry["dataset"])}], add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(rendered)["input_ids"], device="cuda").unsqueeze(0)


def mine_cfg(attn, head_rows, refresh="trigger"):
    return SelectiveConfig(gen_length=GEN, window=16, far_mask_cache=True, eos_truncation=True, refresh=refresh, gamma=0.9, trigger_cos="elastic_heads", track_num=1,
                           attn_impl=attn, head_rows=head_rows, threshold=0.9)


def compare(off_log, off_tokens, mine):
    """First diverging call between the official log and our SelectiveResult."""
    out = {"tokens_equal": list(off_tokens) == list(mine.tokens), "nfe_official": len(off_log), "nfe_mine": mine.nfe, "first_divergence": None}
    for i, (o, m_q, m_tr, m_join, m_w, m_wt) in enumerate(zip(off_log, mine.query_sets, mine.tracked, mine.refresh_layers, mine.written, mine.written_tokens)):
        diffs = []
        if i > 0 and o["query"] is not None and sorted(o["query"]) != sorted(m_q):
            diffs.append("query_set")
        if i > 0 and o["query"] is not None and o["query"] != m_q:
            diffs.append("query_order") if "query_set" not in diffs else None
        if sorted(o["tracked_next"]) != sorted(m_tr):
            diffs.append("tracked_rows")
        mj = m_join if (i > 0 and mine.refresh_kind[i] not in ("first", "dense")) else -1
        if o["join_layer"] != mj:
            diffs.append(f"join_layer(off {o['join_layer']} vs mine {mj})")
        if sorted(o["written"]) != sorted(m_w) or [t for _, t in sorted(zip(o["written"], o["written_tokens"]))] != [t for _, t in sorted(zip(m_w, m_wt))]:
            diffs.append("written")
        if diffs and out["first_divergence"] is None:
            out["first_divergence"] = {"call": i, "fields": diffs}
    out["n_calls_compared"] = min(len(off_log), len(mine.query_sets))
    out["join_calls_official"] = sum(1 for o in off_log if o["join_layer"] >= 0)
    out["join_calls_mine"] = sum(1 for k in mine.refresh_kind if k == "trigger")
    return out


def summarize_res(res):
    return {"tokens": res.tokens, "nfe": res.nfe, "latency_s": res.latency_s, "join_layers": res.refresh_layers, "refresh_kind": res.refresh_kind, "written": res.written,
            "layer_rows_sum": [sum(r) for r in res.layer_rows]}


def run_official():
    import run_elastic_cache_official_eval as EC
    from official_instrumented import generate_instrumented
    up = ROOT / ".omx/external/elastic-cache-reference"
    model, tokenizer, elastic_generate = EC.load_official_elastic(up, MODEL, "cuda", "bf16")
    entries = rows(OUT / "manifests/math40_train.jsonl")
    kw = dict(gen_length=GEN, window_length=16, threshold=0.9, gamma=0.9, track_num=1)
    result = {"upstream": EC.upstream_commit(up), "questions": []}
    for idx in IDX:
        e = entries[idx]
        p = prompt_ids_official(tokenizer, e)
        rec = {"example_id": e["example_id"], "prompt_len": int(p.shape[1])}
        EC.reset_elastic_state(model)
        toks, nfe, frac = elastic_generate(model, p, mask_id=MASK_ID, eos_id=EOS_ID, block_caching=True, **kw)
        off_tokens = toks[:, p.shape[1]:].cpu()[0].tolist()
        rec["official"] = {"tokens": off_tokens, "nfe": int(nfe), "layer_recompute_fraction": float(frac)}
        toks2, nfe2, log = generate_instrumented(model, p, mask_id=MASK_ID, eos_id=EOS_ID, block_caching=True, **kw)
        ins = toks2[:, p.shape[1]:].cpu()[0].tolist()
        rec["instrumented_copy_equals_official"] = ins == off_tokens and int(nfe2) == int(nfe)
        rec["official_log"] = log
        EC.reset_elastic_state(model)
        base, bnfe, _ = EC.generate_full_recompute(model, p, **kw)
        base_tokens = base[:, p.shape[1]:].cpu()[0].tolist()
        rec["official_full_recompute"] = {"tokens": base_tokens, "nfe": int(bnfe)}
        rec["mine_on_official_model"] = {}
        for name, attn, hr in (("manual_all_rows_head", "manual", "all"), ("manual_window_head", "manual", "window")):
            res = decode_selective(model, p, mine_cfg(attn, hr))
            rec["mine_on_official_model"][name] = {**compare(log, off_tokens, res), "latency_s": res.latency_s}
            rec["mine_on_official_model"][name]["summary"] = summarize_res(res) if name == "manual_window_head" else None
        res = decode_selective(model, p, mine_cfg("manual", "window", refresh="every_call"))
        rec["control_every_call_vs_official_full_recompute"] = {"tokens_equal": res.tokens == base_tokens, "nfe_equal": res.nfe == int(bnfe), "nfe_mine": res.nfe, "nfe_official": int(bnfe)}
        result["questions"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k in ("example_id", "instrumented_copy_equals_official", "control_every_call_vs_official_full_recompute")}), flush=True)
        for name, v in rec["mine_on_official_model"].items():
            print("  ", name, json.dumps({k: v[k] for k in ("tokens_equal", "nfe_official", "nfe_mine", "first_divergence", "join_calls_official", "join_calls_mine")}), flush=True)
    (OUT / "validation/parity_official.json").write_text(json.dumps(result) + "\n")


def run_ours():
    from regret_remasking.locking_experiment import load_local_llada
    from regret_remasking.locking_policy import _prepare_prompt
    from regret_remasking.paper_experiment import _resolve_prompt
    checkpoint = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
    off = json.loads((OUT / "validation/parity_official.json").read_text())
    model, tok = load_local_llada(checkpoint, "cuda", "bf16")
    model.eval()
    entries = rows(OUT / "manifests/math40_train.jsonl")
    result = {"questions": []}
    for rec_off in off["questions"]:
        e = next(x for x in entries if x["example_id"] == rec_off["example_id"])
        ids, _ = _prepare_prompt(tok, _resolve_prompt(e), "cuda")
        rec = {"example_id": e["example_id"], "prompt_ids_equal_official": ids.shape[1] == rec_off["prompt_len"], "variants": {}}
        for name, attn, hr in (("manual_window_head", "manual", "window"), ("sdpa_window_head", "sdpa", "window")):
            res = decode_selective(model, ids, mine_cfg(attn, hr))
            rec["variants"][name] = {**compare(rec_off["official_log"], rec_off["official"]["tokens"], res), "latency_s": res.latency_s}
        res = decode_selective(model, ids, mine_cfg("manual", "window", refresh="every_call"))
        rec["control_every_call_vs_official_full_recompute"] = {"tokens_equal": res.tokens == rec_off["official_full_recompute"]["tokens"], "nfe_mine": res.nfe,
                                                                "nfe_official": rec_off["official_full_recompute"]["nfe"]}
        result["questions"].append(rec)
        print(json.dumps({"example_id": rec["example_id"], **{n: {k: v[k] for k in ("tokens_equal", "nfe_mine", "first_divergence")} for n, v in rec["variants"].items()},
                          "control": rec["control_every_call_vs_official_full_recompute"]}), flush=True)
    (OUT / "validation/parity_ours.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    {"official": run_official, "ours": run_ours}[sys.argv[1]]()
