# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""JIT adapter for A2/A3 static scheduling and future functional simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .bridge import build_kernel_program
from .config import SimulatorConfig
from .errors import SimulatorConfigError, UnsupportedSimOpError
from .scheduler import DiscreteEventScheduler, ScheduleResult
from .sync import FlagBarrierSynchronizationModel
from .trace import ChromeTraceExporter


class SimulatorKernelAdapter:
    """Expose lowered TIR, SimIR, schedule statistics, and trace.

    Functional tensor execution is deliberately fail-closed until operation executors are
    implemented. Static scheduling is available through :meth:`schedule` immediately.
    """

    def __init__(
        self,
        *,
        optimized_mod: Any,
        params: list[Any],
        result_idx: list[int] | int | None,
        workspace_idx: list[int] | int | None,
        config: SimulatorConfig,
        program: Any,
    ) -> None:
        self.optimized_mod = optimized_mod
        self.params = params
        self.result_idx = self._normalize_indices(result_idx, "result_idx")
        self.workspace_idx = self._normalize_indices(workspace_idx, "workspace_idx")
        self.config = config
        self.program = program
        self.artifact = None
        self.dynamic_symbolic_map = self._dynamic_symbolic_map()
        self.last_schedule: Optional[ScheduleResult] = None
        self.last_stats = None
        self.last_trace: Optional[Path] = None
        self.func = self._functional_execution_unavailable

    def _normalize_indices(
        self, indices: list[int] | int | None, name: str
    ) -> list[int]:
        if indices is None:
            return []
        values = [indices] if isinstance(indices, int) else list(indices)
        normalized = []
        for index in values:
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"{name} must contain integers")
            if index < 0:
                index += len(self.params)
            if index < 0 or index >= len(self.params):
                raise ValueError(
                    f"{name} index must be between {-len(self.params)} and "
                    f"{len(self.params) - 1}"
                )
            normalized.append(index)
        return normalized

    def _dynamic_symbolic_map(self) -> Mapping[Any, tuple[int, int]]:
        try:
            from tvm import tir
        except (ImportError, OSError):
            return {}
        result = {}
        for parameter_index, parameter in enumerate(self.params):
            if parameter_index in self.result_idx or parameter_index in self.workspace_idx:
                continue
            for shape_index, extent in enumerate(getattr(parameter, "shape", ())):
                if isinstance(extent, tir.Var) and extent not in result:
                    result[extent] = (parameter_index, shape_index)
        return result

    def schedule(self) -> ScheduleResult:
        """Run discrete-event scheduling and optionally emit a Chrome/Perfetto trace."""
        scheduler = DiscreteEventScheduler(
            self.config,
            synchronization=FlagBarrierSynchronizationModel(),
        )
        result = scheduler.run(self.program)
        self.last_schedule = result
        self.last_stats = result.stats
        if self.config.trace_path is not None:
            exporter = ChromeTraceExporter(
                self.config.platform,
                self.config.timing_profile.calibration,
            )
            self.last_trace = exporter.write(self.config.trace_path, result.records)
        return result

    def get_kernel_source(self) -> str:
        """Return the authoritative final pre-codegen TIR script."""
        script = getattr(self.optimized_mod, "script", None)
        return script() if callable(script) else str(self.optimized_mod)

    def get_simulator_ir(self) -> Any:
        """Return the validated backend-neutral simulator program."""
        return self.program

    def _functional_execution_unavailable(self, *_args: Any, **_kwargs: Any) -> Any:
        raise UnsupportedSimOpError(
            "TileLang Ascend functional tensor execution is not implemented yet; "
            "use kernel.adapter.schedule() for static A2/A3 trace generation."
        )


def create_simulator_adapter(
    *,
    func: Any,
    out_idx: list[int] | int | None,
    workspace_idx: list[int] | int | None,
    target: Any,
    target_host: Any,
    platform: str,
    pass_configs: dict[str, Any],
    sim_config: Any | None,
    verbose: bool,
) -> SimulatorKernelAdapter:
    """Lower ``func`` and create the A2/A3 static simulator adapter."""
    del target_host, verbose
    config = _resolve_config(platform, sim_config)

    try:
        from tilelang import tvm
        from tilelang.engine.lower import lower_ascend_ir
    except (ImportError, OSError) as error:
        raise UnsupportedSimOpError(
            "creating a simulator adapter requires the TileLang TVM runtime"
        ) from error

    with tvm.transform.PassContext(opt_level=3, config=pass_configs):
        optimized_mod, params = lower_ascend_ir(
            func,
            target=target,
            platform=config.platform,
        )
    program = build_kernel_program(
        optimized_mod,
        platform=config.platform,
        timing_profile=config.timing_profile,
    )
    return SimulatorKernelAdapter(
        optimized_mod=optimized_mod,
        params=params,
        result_idx=out_idx,
        workspace_idx=workspace_idx,
        config=config,
        program=program,
    )


def _resolve_config(platform: str, value: Any | None) -> SimulatorConfig:
    if value is None:
        return SimulatorConfig(platform=platform)
    if isinstance(value, SimulatorConfig):
        if value.platform != platform.upper():
            raise SimulatorConfigError(
                f"sim_config platform {value.platform} does not match JIT platform {platform}"
            )
        return value
    if isinstance(value, Mapping):
        options = dict(value)
        configured_platform = str(options.pop("platform", platform))
        if configured_platform.upper() != platform.upper():
            raise SimulatorConfigError(
                f"sim_config platform {configured_platform} does not match JIT platform "
                f"{platform}"
            )
        return SimulatorConfig(platform=platform, **options)
    raise SimulatorConfigError("sim_config must be SimulatorConfig, mapping, or None")
