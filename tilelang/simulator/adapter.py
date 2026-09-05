# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""JIT adapter for A2/A3 static scheduling and functional simulation."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .bridge import build_kernel_program
from .config import SimulatorConfig
from .errors import ProgramValidationError, SimulatorConfigError, UnsupportedSimOpError
from .executor import FunctionalExecutionResult, FunctionalSimulator
from .program import BufferRegion, BufferSpec, MemoryScope
from .scheduler import DiscreteEventScheduler, ScheduleResult
from .sync import FlagBarrierSynchronizationModel, validate_memory_synchronization
from .trace import ChromeTraceExporter


class SimulatorKernelAdapter:
    """Expose CPU tensor execution, lowered TIR, SimIR, scheduling, and trace."""

    def __init__(
        self,
        *,
        optimized_mod: Any,
        params: list[Any],
        result_idx: list[int] | int | None,
        workspace_idx: list[int] | int | None,
        config: SimulatorConfig,
        program: Any,
        validate_sync: bool = False,
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
        self.last_execution: Optional[FunctionalExecutionResult] = None
        self._parameter_names = self._extract_parameter_names()
        self._buffer_specs = {buffer.name: buffer for buffer in self.program.buffers}
        self.sync_diagnostics = (
            validate_memory_synchronization(
                self.program, hazard_check=self.config.hazard_check
            )
            if validate_sync
            else ()
        )
        self.func = self._functional_execute

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
        self._record_schedule(result)
        return result

    def _record_schedule(self, result: ScheduleResult) -> None:
        self.last_schedule = result
        self.last_stats = result.stats
        if self.config.trace_path is not None:
            exporter = ChromeTraceExporter(
                self.config.platform,
                self.config.timing_profile.calibration,
            )
            self.last_trace = exporter.write(self.config.trace_path, result.records)

    def get_kernel_source(self) -> str:
        """Return the authoritative final pre-codegen TIR script."""
        script = getattr(self.optimized_mod, "script", None)
        return script() if callable(script) else str(self.optimized_mod)

    def get_simulator_ir(self) -> Any:
        """Return the validated backend-neutral simulator program."""
        return self.program

    def _extract_parameter_names(self) -> list[Optional[str]]:
        try:
            import tvm
        except (ImportError, OSError):
            return []
        value = self.optimized_mod
        if isinstance(value, tvm.IRModule):
            functions = [
                function for _, function in value.functions_items()
                if isinstance(function, tvm.tir.PrimFunc)
            ]
            if len(functions) != 1:
                return []
            value = functions[0]
        if not isinstance(value, tvm.tir.PrimFunc):
            return []
        names: list[Optional[str]] = []
        for parameter in value.params:
            if parameter in value.buffer_map:
                names.append(str(value.buffer_map[parameter].name))
            else:
                names.append(str(getattr(parameter, "name", parameter)))
        return names

    def _functional_execute(self, *arguments: Any) -> Any:
        if len(self._parameter_names) != len(self.params):
            raise UnsupportedSimOpError(
                "functional simulator requires final PrimFunc parameter metadata"
            )
        supplied_indices = [
            index for index in range(len(self.params))
            if index not in self.result_idx and index not in self.workspace_idx
        ]
        dynamic_count = len(self.dynamic_symbolic_map)
        if len(arguments) not in {
            len(supplied_indices), len(supplied_indices) + dynamic_count,
        }:
            raise ValueError(
                f"expected {len(supplied_indices)} simulator inputs, got "
                f"{len(arguments)}"
            )
        parameter_values: list[Any] = [None] * len(self.params)
        for index, value in zip(supplied_indices, arguments):
            parameter_values[index] = value

        bindings: dict[str, int | float] = {}
        for variable, (parameter_index, shape_index) in self.dynamic_symbolic_map.items():
            value = parameter_values[parameter_index]
            if value is None or not hasattr(value, "shape"):
                raise ProgramValidationError(
                    f"cannot bind dynamic extent {variable} from parameter "
                    f"{parameter_index}"
                )
            bindings[str(getattr(variable, "name", variable))] = int(
                value.shape[shape_index]
            )
        for index, name in enumerate(self._parameter_names):
            if name in self._buffer_specs or parameter_values[index] is None:
                continue
            value = parameter_values[index]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
                raise ProgramValidationError(
                    f"simulator scalar parameter {name!r} must be numeric"
                )
            bindings[name or ""] = (
                int(value) if isinstance(value, Integral) else float(value)
            )

        simulator = FunctionalSimulator(self.program, self.config, bindings=bindings)
        framework = "numpy"
        for index in supplied_indices:
            name = self._parameter_names[index]
            if name not in self._buffer_specs:
                continue
            value = parameter_values[index]
            array, value_framework = self._as_numpy(value, name)
            if value_framework == "torch":
                framework = "torch"
            simulator.write(self._full_region(self._buffer_specs[name]), array)

        execution = simulator.run()
        self.last_execution = execution
        self._record_schedule(execution.schedule)
        outputs = [
            simulator.read(self._full_region(self._result_buffer(index)))
            for index in self.result_idx
        ]
        converted = [self._from_numpy(output, framework) for output in outputs]
        return converted[0] if len(converted) == 1 else converted

    def _result_buffer(self, parameter_index: int) -> BufferSpec:
        name = self._parameter_names[parameter_index]
        if name not in self._buffer_specs:
            raise ProgramValidationError(
                f"simulator result parameter {parameter_index} is not a buffer"
            )
        return self._buffer_specs[name]

    @staticmethod
    def _full_region(spec: BufferSpec) -> BufferRegion:
        if spec.scope is not MemoryScope.GM:
            raise ProgramValidationError(
                f"simulator parameter buffer {spec.name!r} must use GM scope"
            )
        return BufferRegion(spec.name, spec.scope, spec.shape, spec.dtype)

    @staticmethod
    def _as_numpy(value: Any, name: Optional[str]) -> tuple[np.ndarray, str]:
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value), "numpy"
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ProgramValidationError(
                    f"simulator input {name!r} must be a CPU tensor, got {value.device}"
                )
            try:
                return np.ascontiguousarray(value.detach().numpy()), "torch"
            except TypeError as error:
                raise UnsupportedSimOpError(
                    f"simulator cannot convert tensor dtype {value.dtype} to NumPy"
                ) from error
        raise TypeError(
            f"simulator input {name!r} must be a NumPy array or CPU torch.Tensor"
        )

    @staticmethod
    def _from_numpy(value: np.ndarray, framework: str) -> Any:
        if framework == "numpy":
            return value
        import torch

        return torch.from_numpy(np.ascontiguousarray(value))


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
        validate_sync=bool(pass_configs.get("tl.ascend_auto_sync", False)),
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
