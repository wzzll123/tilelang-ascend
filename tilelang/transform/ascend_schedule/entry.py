# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fail-closed entry for opt-in lowered-TIR scheduling during migration."""

from .policy import resolve_options, schedule_enabled
from .tir_adapter import extract_tasks
from .conditions import normalize_conditions
from .gemm_decomposition import decompose_function_gemms
from .workspace_tasks import prepare_cross_core_copies


class AutoScheduleError(NotImplementedError):
    """Requested automatic scheduling cannot yet produce executable TIR."""


def schedule_module(mod, target_model, config):
    """Normalize and extract enabled modules, then stop before legacy sync.

    This bootstrap intentionally cannot return a scheduled module: ring
    materialization and unified synchronization are not implemented. Disabling
    the opt-in flag restores the existing lowering path.
    """
    config = {} if config is None else config
    # Disabled policy does not inspect targets, platform attrs, or Stage metadata.
    if not schedule_enabled(config):
        return mod
    for function in mod.functions.values():
        attrs = getattr(function, "attrs", None)
        platform = str(attrs.get("npu_platform", "")) if attrs is not None else ""
        resolve_options(config, target_model, platform)
    mod = prepare_cross_core_copies(mod)
    for function in mod.functions.values():
        extract_tasks(normalize_conditions(decompose_function_gemms(function)))
    raise AutoScheduleError(
        "tl.ascend_auto_schedule extraction succeeded, but the scheduling materializer "
        "and unified synchronization emitter are not implemented. Disable "
        "tl.ascend_auto_schedule to use existing lowering; no legacy synchronization "
        "fallback was applied.")
