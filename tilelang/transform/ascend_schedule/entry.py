# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Opt-in lowered-TIR scheduling with explicit unsupported-structure errors."""

from .policy import resolve_options, schedule_enabled
from .tir_adapter import extract_tasks
from .conditions import normalize_conditions
from .gemm_decomposition import decompose_function_gemms
from .workspace_tasks import prepare_cross_core_copies
from .ub_versions import prepare_ub_versions
from .core_scopes import normalize_core_scopes
from .materializer import materialize_flat_function, MaterializationError


class AutoScheduleError(NotImplementedError):
    """Requested automatic scheduling cannot yet produce executable TIR."""


def schedule_module(mod, target_model, config):
    """Materialize supported epochs; never silently use legacy synchronization.

    The flat epoch path is executable. Structured owner-loop/guard reconstruction
    still reports a migration error. Disabling the flag preserves legacy lowering.
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
    functions = {}
    for name, function in mod.functions.items():
        try:
            function = normalize_conditions(decompose_function_gemms(function, root_scratch=True))
        except ValueError as error:
            raise AutoScheduleError("tl.ascend_auto_schedule GEMM preparation: " + str(error) +
                                    "; no legacy synchronization fallback was applied.") from error
        extraction = extract_tasks(function)
        try:
            try:
                scoped = normalize_core_scopes(function, extraction)
                if scoped is not function:
                    function, extraction = scoped, extract_tasks(scoped)
                function, extraction = prepare_ub_versions(function, extraction,
                                                           str(function.attrs["npu_platform"]))
            except ValueError as error:
                raise MaterializationError(str(error)) from error
            functions[name] = materialize_flat_function(function, extraction,
                                                        str(function.attrs["npu_platform"]))
        except MaterializationError as error:
            raise AutoScheduleError(
                "tl.ascend_auto_schedule extraction succeeded, but the scheduling materializer: " +
                str(error) + "; no legacy synchronization fallback was applied.") from error
    from tvm import IRModule
    return IRModule(functions, mod.type_definitions, mod.attrs)
