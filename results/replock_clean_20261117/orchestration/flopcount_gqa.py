"""flopcount.count with a grouped-query-attention aware SDPA formula (Dream / Qwen-style models have fewer K/V heads than Q heads; torch's built-in formula asserts equal head counts).
SDPA FLOPs = 4 * batch * q_heads * q_len * k_len * head_dim (QK^T and PV), identical to torch's formula when the heads match."""
import contextlib

import torch
from torch.utils.flop_counter import FlopCounterMode

N_FLOPS = 20


def _sdpa(query_shape, key_shape, value_shape, *args, **kwargs):
    b, h, sq, d = query_shape
    return 4 * b * h * sq * key_shape[2] * d


_OPS = [getattr(torch.ops.aten, n) for n in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention") if hasattr(torch.ops.aten, n)]


@contextlib.contextmanager
def count(out: dict):
    with FlopCounterMode(display=False, custom_mapping={op: _sdpa for op in _OPS}) as f:
        yield
    fc = f.get_flop_counts()
    out["flops_total"] = int(f.get_total_flops())
    out["flops_head"] = int(sum(sum(v.values()) for k, v in fc.items() if k.endswith("ff_out") or k.endswith("lm_head")))
