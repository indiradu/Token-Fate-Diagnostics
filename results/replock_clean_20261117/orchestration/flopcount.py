"""FLOP counter shared by the three runners (env FLOPS=1): torch.utils.flop_counter.FlopCounterMode around one decode. Counts the GEMM and attention FLOPs actually executed (released code and our engine alike);
flops_head = the vocabulary head (module ff_out / lm_head), flops_total - flops_head = backbone (SureLock's convention reports the backbone separately from the head). Timing of a FLOPS run is NOT a speed measurement."""
import contextlib

from torch.utils.flop_counter import FlopCounterMode

N_FLOPS = 20                                                    # prompts per cell in the FLOP pass


@contextlib.contextmanager
def count(out: dict):
    with FlopCounterMode(display=False) as f:
        yield
    fc = f.get_flop_counts()
    out["flops_total"] = int(f.get_total_flops())
    out["flops_head"] = int(sum(sum(v.values()) for k, v in fc.items() if k.endswith("ff_out") or k.endswith("lm_head")))
