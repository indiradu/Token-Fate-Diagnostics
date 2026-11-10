"""Diagnostic copy of upstream ``generate_with_elastic_cache`` (VILA-Lab/Elastic-Cache @ 1960d8f, llada/generate.py lines 44-146) that logs per call what the
loop decides. The loop body is copied unchanged; only ``log`` appends are added. Upstream files are not modified. Used for the parity gate only.

Logged per call: initial query positions (tracked, rows written last call, window), tracked rows chosen on this call (union over layers), the layer at which
rows joined (lengths[1] after the call, = L when no trigger), written positions and tokens.
"""
from __future__ import annotations

import sys

import torch


def generate_instrumented(model, prompt, gen_length=512, window_length=16, mask_id=126336, eos_id=126081, threshold=0.9, tokens_per_iter=1, gamma=0.9, track_num=1,
                          block_caching=True):
    gen_mod = sys.modules["generate"]                      # upstream module, imported by load_official_elastic
    get_decoded_token_confident = gen_mod.get_decoded_token_confident
    for l, block in enumerate(model.model.transformer.blocks):
        block.x_cache = None
        block.q_cache = None
        block.k_cache = None
        block.v_cache = None
        block.track_token = None
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    nfe = 0
    query_position = torch.arange(prompt.shape[1] + gen_length, device=model.device)
    track_position = query_position[:0].clone()
    new_decoded_position = query_position[:prompt.shape[1]].clone()
    masked_position = query_position[prompt.shape[1]:].clone()
    log = []
    i = 0
    decoded_eos = False
    L = len(model.model.transformer.blocks)
    threshold_ = threshold
    while True:
        if block_caching:
            query_masked_position = masked_position[:window_length]
        else:
            query_masked_position = masked_position
        if i == 0:
            x_query = x
            start_reset = -1
        else:
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query = x[:, query_position]
            start_reset = L
        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths = [x.shape[1], start_reset, gamma, track_num]
        entry = {"call": i, "query": query_position.tolist() if i > 0 else None, "n_track": int(track_position.numel()), "n_blank": int(masked_position.numel())}
        output = model(x_query, use_cache=True, lengths=lengths, positions=positions)
        logits = output.logits
        if logits.shape[1] == x.shape[1]:
            logits = logits[:, query_masked_position, :]
        else:
            logits = logits[:, -query_masked_position.shape[0]:, :]
        if not block_caching:
            query_masked_position = query_masked_position[:window_length]
            logits = logits[:, :window_length]
        track_position = torch.cat([block.track_token for block in model.model.transformer.blocks], dim=0).unique(sorted=False)
        entry["join_layer"] = int(lengths[1]) if (i > 0 and lengths[1] < L) else -1
        entry["tracked_next"] = track_position.tolist()
        x, new_decoded_position, eos_pos = get_decoded_token_confident(logits, query_masked_position, x, threshold_, eos_id)
        entry["written"] = new_decoded_position.tolist()
        entry["written_tokens"] = x[0, new_decoded_position].tolist()
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]
        nfe += 1
        if not decoded_eos:
            if eos_pos.shape[0] > 0:
                eos_pos = eos_pos.min().item()
                decoded_eos = True
                masked_position = masked_position[masked_position <= eos_pos]
        log.append(entry)
        if masked_position.shape[0] == 0:
            break
        i += 1
    x[x.eq(mask_id)] = eos_id
    return x, nfe, log
