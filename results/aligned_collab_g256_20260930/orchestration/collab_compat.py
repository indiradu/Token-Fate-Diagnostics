"""Environment shim so the collaborator's code (written for transformers 4.57) loads LLaDA under 5.12.

Copies the two compatibility patches our own llada_trace.load_llada applies, verbatim, and wraps her
load_llada with them. Her decoding, training and method code are untouched. Import this module
before any of her script modules so their `from regret_remasking.llada_trace import load_llada`
binds the wrapped loader.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

import regret_remasking.llada_trace as _llada_trace
import regret_remasking.tracelock_baseline as _tracelock_baseline


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



_original_load_llada = _llada_trace.load_llada


def load_llada(model_name: str, device: str = "cuda", dtype: str = "bf16") -> tuple[Any, Any]:
    ensure_transformers_tied_weight_compat()
    ensure_remote_llada_compat(model_name)
    model, tokenizer = _original_load_llada(model_name, device=device, dtype=dtype)
    # Same two config defaults our llada_trace.load_llada sets after loading under transformers 5.
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not hasattr(model.config, "use_return_dict"):
        model.config.use_return_dict = True
    return model, tokenizer


_llada_trace.load_llada = load_llada


def extend_controller_length(minimum: int) -> None:
    """Declared adaptation: two panel prompts exceed 512 tokens, so allow controller positions up to `minimum`."""
    original = _tracelock_baseline.TraceLockConfig

    def config_factory(*args: Any, **kwargs: Any) -> Any:
        config = original(*args, **kwargs)
        if config.max_sequence_length < minimum:
            config = original(*args, **{**kwargs, "max_sequence_length": minimum})
        return config

    _tracelock_baseline.TraceLockConfig = config_factory  # type: ignore[assignment]


def _alias_bare_gsm8k() -> None:
    """datasets 5.x rejects the bare id "gsm8k" her data.py uses; it is the same dataset as "openai/gsm8k"."""
    import datasets

    original = datasets.load_dataset

    def load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        return original("openai/gsm8k" if path == "gsm8k" else path, *args, **kwargs)

    datasets.load_dataset = load_dataset
    import regret_remasking.data as _data

    if hasattr(_data, "load_dataset"):
        _data.load_dataset = load_dataset


_alias_bare_gsm8k()

if os.environ.get("COLLAB_MIN_CONTROLLER_LENGTH"):
    extend_controller_length(int(os.environ["COLLAB_MIN_CONTROLLER_LENGTH"]))
