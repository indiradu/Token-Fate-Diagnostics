from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from regret_remasking import FEATURE_NAMES
from regret_remasking.features import entropy_from_log_probs, local_window_average, safe_kl

MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348
MASK_TOKEN_CANDIDATES = ("<|mdm_mask|>", "<[MASK]>", "<mask>", "[MASK]")


@dataclass
class DecodeConfig:
    steps: int = 64
    gen_length: int = 64
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    policy: str = "confidence"
    beta_kl: float = 1.0
    beta_regret: float = 2.0
    cpv_window: int = 4
    mask_id: int = MASK_ID
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False


@dataclass
class DecodeResult:
    texts: list[str]
    token_ids: torch.Tensor
    trace_rows: list[dict[str, Any]]
    nfe: int
    latency_s: float


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for idx in range(mask_num.size(0)):
        out[idx, : remainder[idx]] += 1
    return out


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def prepare_prompts(tokenizer: Any, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def infer_mask_token_id(tokenizer: Any, default: int = MASK_ID) -> int:
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        return int(mask_token_id)
    for token in MASK_TOKEN_CANDIDATES:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    return default


def ensure_transformers_tied_weight_compat() -> None:
    """Bridge older remote model wrappers to newer Transformers loaders."""
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    @property
    def all_tied_weights_keys(self: Any) -> dict[str, None]:
        keys = getattr(self, "_tied_weights_keys", None) or []
        if isinstance(keys, dict):
            return keys
        return {str(key): None for key in keys}

    PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys  # type: ignore[attr-defined]


def ensure_remote_llada_compat(model_name: str) -> None:
    """Patch known LLaDA remote-code signatures for newer Transformers."""
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    try:
        model_cls = get_class_from_dynamic_module("modeling_llada.LLaDAModelLM", model_name)
    except Exception:
        return

    try:
        parameters = inspect.signature(model_cls.tie_weights).parameters
    except (TypeError, ValueError):
        return
    if "missing_keys" in parameters:
        return

    original_tie_weights = model_cls.tie_weights

    def tie_weights(self: Any, missing_keys: set[str] | None = None, recompute_mapping: bool = True) -> Any:
        del missing_keys, recompute_mapping
        return original_tie_weights(self)

    model_cls.tie_weights = tie_weights


def load_llada(model_name: str, device: str = "cuda", dtype: str = "bf16") -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    ensure_transformers_tied_weight_compat()
    ensure_remote_llada_compat(model_name)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype.lower()]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    device_map_mode = os.environ.get("LLADA_DEVICE_MAP")
    if device_map_mode == "1" and device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}
    elif device_map_mode == "auto" and device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            0: os.environ.get("LLADA_GPU_MAX_MEMORY", "26GiB"),
            "cpu": os.environ.get("LLADA_CPU_MAX_MEMORY", "80GiB"),
        }
        load_kwargs["offload_folder"] = os.environ.get("LLADA_OFFLOAD_FOLDER", "/tmp/llada_offload")
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not hasattr(model.config, "use_return_dict"):
        model.config.use_return_dict = True
    if "device_map" not in load_kwargs:
        model = model.to(device)
    model.eval()
    mask_id = infer_mask_token_id(tokenizer)
    if tokenizer.pad_token_id == mask_id:
        raise RuntimeError("tokenizer pad_token_id equals LLaDA mask id; trace generator needs separate padding")
    return model, tokenizer


def _feature_value(feature_map: dict[str, torch.Tensor], name: str, batch: int, pos: int) -> float:
    return float(feature_map[name][batch, pos].detach().cpu())


def _regret_tensor(
    scorer: Any,
    feature_map: dict[str, torch.Tensor],
    allowed: torch.Tensor,
) -> torch.Tensor:
    regret = torch.zeros_like(feature_map["confidence"], dtype=torch.float32)
    positions = torch.nonzero(allowed, as_tuple=False)
    if positions.numel() == 0:
        return regret
    rows = []
    for batch, pos in positions.tolist():
        rows.append([_feature_value(feature_map, name, batch, pos) for name in scorer.features])
    probs = scorer.score_matrix(np.asarray(rows, dtype=np.float32))
    for (batch, pos), prob in zip(positions.tolist(), probs):
        regret[batch, pos] = float(prob)
    return regret.to(allowed.device)


@torch.no_grad()
def decode_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    example_ids: list[str],
    config: DecodeConfig,
    regret_scorer: Any | None = None,
    collect_trace: bool = False,
) -> DecodeResult:
    if len(prompts) != len(example_ids):
        raise ValueError("prompts and example_ids must have the same length")
    if config.gen_length % config.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    start_time = time.perf_counter()
    device = model_device(model)
    input_ids, attention_mask = prepare_prompts(tokenizer, prompts, device)
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + config.gen_length

    x = torch.full((batch_size, total_len), config.mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((batch_size, config.gen_length), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )
    prompt_index = x != config.mask_id
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    if steps_per_block < 1 or config.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by gen_length / block_length")

    trace_rows: list[dict[str, Any]] = []
    prev_log_probs: torch.Tensor | None = None
    prev_top1 = torch.full_like(x, fill_value=-1)
    runlength = torch.zeros_like(x, dtype=torch.float32)
    nfe = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * config.block_length
        block_end = prompt_len + (block_idx + 1) * config.block_length
        block_mask_index = x[:, block_start:block_end] == config.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_in_block in range(steps_per_block):
            global_step = block_idx * steps_per_block + step_in_block
            mask_index = x == config.mask_id
            if config.cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = config.mask_id
                x_for_model = torch.cat([x, un_x], dim=0)
                attn_for_model = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_for_model, attention_mask=attn_for_model).logits
                logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + (config.cfg_scale + 1.0) * (logits - uncond_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            nfe += 1

            if config.logits_eos_inf:
                logits[:, :, EOS_ID] = -torch.inf
            logits_for_noise = logits
            if config.confidence_eos_eot_inf:
                logits_for_noise = logits_for_noise.clone()
                logits_for_noise[:, :, EOS_ID] = -torch.inf
                logits_for_noise[:, :, EOT_ID] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits_for_noise, config.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            x0_p = probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            top_probs, top_ids = probs.topk(k=2, dim=-1)
            confidence = top_probs[:, :, 0]
            margin = top_probs[:, :, 0] - top_probs[:, :, 1]
            entropy = entropy_from_log_probs(log_probs)
            kl = safe_kl(log_probs, prev_log_probs)
            top1_flip = ((prev_top1 >= 0) & (prev_top1 != x0)).float()
            runlength = torch.where(prev_top1 == x0, runlength + 1.0, torch.ones_like(runlength))
            local_mask_ratio = local_window_average(mask_index.float(), config.cpv_window)
            context_volatility = local_window_average(kl, config.cpv_window)

            allowed = mask_index.clone()
            allowed[:, :prompt_len] = False
            allowed[:, block_end:] = False
            x0 = torch.where(mask_index, x0, x)
            score = torch.full_like(confidence, fill_value=-torch.inf)
            feature_map = {
                "confidence": confidence,
                "entropy": entropy,
                "margin": margin,
                "kl": kl,
                "top1_flip": top1_flip,
                "runlength": runlength,
                "t_frac": torch.full_like(confidence, (global_step + 1) / config.steps),
                "local_mask_ratio": local_mask_ratio,
                "context_volatility": context_volatility,
            }

            if config.policy == "random":
                raw_score = torch.rand_like(confidence)
            elif config.policy == "confidence":
                raw_score = x0_p
            elif config.policy == "confidence_kl":
                raw_score = x0_p * torch.exp(-config.beta_kl * kl)
            elif config.policy in {"regret", "regret_no_cpv"}:
                if regret_scorer is None:
                    raise ValueError(f"{config.policy} policy requires regret_scorer")
                regret = _regret_tensor(regret_scorer, feature_map, allowed)
                raw_score = x0_p * torch.exp(-config.beta_regret * regret)
            else:
                raise ValueError(f"unknown decode policy: {config.policy}")
            score[allowed] = raw_score[allowed]

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for batch_idx in range(batch_size):
                k = int(num_transfer_tokens[batch_idx, step_in_block].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(score[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True

            if collect_trace:
                positions = torch.nonzero(allowed, as_tuple=False)
                for batch_idx, pos in positions.tolist():
                    row: dict[str, Any] = {
                        "example_id": example_ids[batch_idx],
                        "batch_index": batch_idx,
                        "global_step": global_step,
                        "block": block_idx,
                        "step_in_block": step_in_block,
                        "position": pos,
                        "relative_position": pos - prompt_len,
                        "top_token": int(x0[batch_idx, pos].detach().cpu()),
                        "score": float(score[batch_idx, pos].detach().cpu()),
                        "selected": bool(transfer_index[batch_idx, pos].detach().cpu()),
                        "mask_token_id": config.mask_id,
                    }
                    for name in FEATURE_NAMES:
                        row[name] = _feature_value(feature_map, name, batch_idx, pos)
                    trace_rows.append(row)

            x[transfer_index] = x0[transfer_index]
            prev_log_probs = log_probs.detach()
            prev_top1 = x0.detach()

    final_tokens = x[:, prompt_len:].detach().cpu()
    for row in trace_rows:
        rel_pos = int(row["relative_position"])
        batch_idx = int(row["batch_index"])
        final_token = int(final_tokens[batch_idx, rel_pos])
        row["final_token"] = final_token
        row["trace_regret"] = int(int(row["top_token"]) != final_token)

    texts = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
    latency = time.perf_counter() - start_time
    return DecodeResult(
        texts=[text.strip() for text in texts],
        token_ids=x.detach().cpu(),
        trace_rows=trace_rows,
        nfe=nfe,
        latency_s=latency,
    )
