# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Lower final A2/A3 TIR into the simulator's backend-neutral program model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from .errors import ProgramValidationError, UnsupportedSimOpError
from .profile import TimingProfile, default_timing_profile, normalize_platform
from .program import BufferSpec, CoreProgram, KernelProgram, Lane, MemoryScope, Pipe, Task


_VECTOR_OPS = frozenset({
    "abs", "add", "adds", "arith_progression", "axpy", "bilinear_interpolation",
    "bitwise_and", "bitwise_lshift", "bitwise_not", "bitwise_or", "bitwise_rshift",
    "bitwise_xor", "block_reduce_max", "block_reduce_min", "block_reduce_sum",
    "broadcast", "brcb_experiment", "cast", "clamp", "clamp_max", "clamp_min",
    "compare", "compare_scalar", "cos", "createvecindex", "div", "divs", "duplicate",
    "exp", "fill", "gather", "gather_mask", "gather_mask_experiment", "gatherb",
    "init_sort_buf", "leaky_relu", "ln", "max", "maxs", "merge_sort", "min", "mins",
    "mul", "muls", "pow", "reciprocal", "reduce", "relu", "round", "rsqrt", "select",
    "sigmoid", "sin", "sort", "sort32", "sqrt", "sub", "subs", "tail_binary",
    "tail_broadcast", "tail_compare", "tail_compare_scalar", "tail_reduce", "tail_scalar",
    "tail_select", "tail_unary", "topk", "transpose", "wholereducemax",
    "wholereducemin", "wholereducesum", "abs_experiment", "brcb_experiment",
    "datacachecleanandinvalid_experiment", "exp_experiment", "fill_experiment",
    "gather_mask_experiment", "mins_experiment", "reducesum_experiment",
    "reducesum_mask_experiment", "row_expand_div_experiment",
    "row_expand_mul_experiment", "row_expand_sub_experiment", "sub_experiment",
    "sum_experiment",
})


@dataclass(frozen=True)
class _Context:
    core_id: int = 0
    lane: Lane = Lane.CONTROL
    vector_index: Optional[int] = None
    environment: Mapping[Any, int] = None

    def __post_init__(self) -> None:
        if self.environment is None:
            object.__setattr__(self, "environment", {})


def classify_operation(operation: str, lane: Lane) -> Tuple[Lane, Pipe, str]:
    """Map one lowered operation name to the A2/A3 execution resource."""
    normalized = operation.strip().lower()
    short = normalized
    for prefix in ("tl.ascend_", "tl::ascend::", "ascendc::"):
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    if short == "tl.arith_progression":
        short = "arith_progression"

    if "shmem" in short:
        raise UnsupportedSimOpError(
            f"Ascend shmem operation {operation!r} is intentionally unsupported"
        )
    if short in {
        "set_flag", "wait_flag", "auto_set_flag", "auto_wait_flag",
        "set_cross_flag", "wait_cross_flag", "auto_set_cross_flag",
        "auto_wait_cross_flag", "barrier_all", "pipe_barrier", "auto_barrier",
    }:
        return lane, Pipe.SCALAR, short
    if "im2col" in short:
        return Lane.CUBE, Pipe.MTE1, short
    if "gemm" in short or short == "mma":
        return Lane.CUBE, Pipe.MATRIX, short
    if "copy" in short or "data_copy" in short or "datacopy" in short:
        if "l1_to_l0" in short:
            return Lane.CUBE, Pipe.MTE1, short
        if "l0c_to_gm" in short or "copy_cv" in short:
            return Lane.CUBE, Pipe.FIX, short
        if "ub_to_gm" in short:
            return _vector_lane(lane), Pipe.MTE3, short
        if "copy_vc" in short:
            return _vector_lane(lane), Pipe.VECTOR, short
        return lane if lane is not Lane.CONTROL else Lane.CUBE, Pipe.MTE2, short
    if "atomic" in short:
        if "l0c" in short or lane is Lane.CUBE:
            return Lane.CUBE, Pipe.FIX, short
        return _vector_lane(lane), Pipe.MTE3, short
    if short in _VECTOR_OPS:
        return _vector_lane(lane), Pipe.VECTOR, short
    if short in {"scalar", "printf", "dump_tensor", "free_pipe"}:
        return lane, Pipe.SCALAR, short
    if normalized == "buffer_store":
        return _vector_lane(lane), Pipe.VECTOR, normalized
    raise UnsupportedSimOpError(f"unsupported lowered simulator operation: {operation!r}")


def _vector_lane(lane: Lane) -> Lane:
    return lane if lane in {Lane.VECTOR_0, Lane.VECTOR_1} else Lane.VECTOR_0


def build_kernel_program(
    func_or_mod: Any,
    *,
    platform: str,
    timing_profile: Optional[TimingProfile] = None,
    max_unrolled_iterations: int = 65536,
) -> KernelProgram:
    """Build a simulator program from final optimized TIR.

    TVM is imported lazily so the simulator's program, memory, scheduler, and trace
    foundations remain importable in CPU-only environments that do not have TileLang built.
    """
    try:
        from tilelang import tvm
        from tvm import arith, tir
    except (ImportError, OSError) as error:
        raise UnsupportedSimOpError(
            "building a simulator program requires the TileLang TVM runtime"
        ) from error

    normalized_platform = normalize_platform(platform)
    profile = timing_profile or default_timing_profile(normalized_platform)
    bridge = _TirBridge(
        tvm=tvm,
        tir=tir,
        analyzer=arith.Analyzer(),
        platform=normalized_platform,
        timing_profile=profile,
        max_unrolled_iterations=max_unrolled_iterations,
    )
    return bridge.build(func_or_mod)


class _TirBridge:
    def __init__(
        self,
        *,
        tvm: Any,
        tir: Any,
        analyzer: Any,
        platform: str,
        timing_profile: TimingProfile,
        max_unrolled_iterations: int,
    ) -> None:
        self.tvm = tvm
        self.tir = tir
        self.analyzer = analyzer
        self.platform = platform
        self.timing_profile = timing_profile
        self.max_unrolled_iterations = max_unrolled_iterations
        self.tasks: Dict[int, list[Task]] = defaultdict(list)
        self.buffers: Dict[str, BufferSpec] = {}
        self.storage_scope_by_var: Dict[str, MemoryScope] = {}
        self.task_counter = 0
        self.kernel_name = "main"

    def build(self, func_or_mod: Any) -> KernelProgram:
        func = self._select_prim_func(func_or_mod)
        if func.attrs is not None and "global_symbol" in func.attrs:
            self.kernel_name = str(func.attrs["global_symbol"])
        self._collect_parameter_buffers(func)
        self._visit(func.body, _Context())
        cores = tuple(
            CoreProgram(core_id, tuple(self.tasks[core_id])) for core_id in sorted(self.tasks)
        )
        if not cores:
            cores = (CoreProgram(0),)
        return KernelProgram(
            self.kernel_name,
            self.platform,
            cores,
            tuple(self.buffers.values()),
            metadata={
                "timing_calibration": self.timing_profile.calibration,
                "source": "final-optimized-tir",
            },
        )

    def _select_prim_func(self, value: Any) -> Any:
        if isinstance(value, self.tir.PrimFunc):
            return value
        if isinstance(value, self.tvm.IRModule):
            functions = [func for _, func in value.functions_items()
                         if isinstance(func, self.tir.PrimFunc)]
            if len(functions) != 1:
                raise ProgramValidationError(
                    "simulator bridge requires an IRModule containing exactly one PrimFunc"
                )
            return functions[0]
        raise TypeError("simulator bridge input must be a PrimFunc or IRModule")

    def _collect_parameter_buffers(self, func: Any) -> None:
        for _, buffer in func.buffer_map.items():
            name = str(buffer.name)
            shape = tuple(self._extent_or_symbol(extent, {}) for extent in buffer.shape)
            self.buffers.setdefault(
                name,
                BufferSpec(name, MemoryScope.GM, shape, str(buffer.dtype)),
            )

    def _visit(self, stmt: Any, context: _Context) -> None:
        tir = self.tir
        if stmt is None:
            return
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                self._visit(child, context)
            return
        if isinstance(stmt, tir.AttrStmt):
            self._visit_attr(stmt, context)
            return
        if isinstance(stmt, tir.For):
            minimum = self._require_int(stmt.min, context.environment, "loop minimum")
            extent = self._require_int(stmt.extent, context.environment, "loop extent")
            if extent < 0 or extent > self.max_unrolled_iterations:
                raise UnsupportedSimOpError(
                    f"loop extent {extent} exceeds simulator bridge limit "
                    f"{self.max_unrolled_iterations}"
                )
            for value in range(minimum, minimum + extent):
                environment = dict(context.environment)
                environment[stmt.loop_var] = value
                self._visit(stmt.body, replace(context, environment=environment))
            return
        if isinstance(stmt, tir.IfThenElse):
            condition = self._require_int(stmt.condition, context.environment, "if condition")
            self._visit(stmt.then_case if condition else stmt.else_case, context)
            return
        if isinstance(stmt, tir.LetStmt):
            value = self._require_int(stmt.value, context.environment, "let binding")
            environment = dict(context.environment)
            environment[stmt.var] = value
            self._visit(stmt.body, replace(context, environment=environment))
            return
        if isinstance(stmt, tir.Allocate):
            self._collect_allocate(stmt, context)
            self._visit(stmt.body, context)
            return
        if hasattr(tir, "DeclBuffer") and isinstance(stmt, tir.DeclBuffer):
            self._visit(stmt.body, context)
            return
        if hasattr(tir, "BufferRealize") and isinstance(stmt, tir.BufferRealize):
            self._visit(stmt.body, context)
            return
        if isinstance(stmt, tir.BlockRealize):
            self._visit(stmt.block, context)
            return
        if isinstance(stmt, tir.Block):
            self._visit(stmt.init, context)
            self._visit(stmt.body, context)
            return
        if isinstance(stmt, tir.AssertStmt):
            condition = self._require_int(stmt.condition, context.environment, "assertion")
            if not condition:
                raise ProgramValidationError(f"TIR assertion failed: {stmt.message}")
            self._visit(stmt.body, context)
            return
        if isinstance(stmt, tir.Evaluate):
            if self._is_zero(stmt.value, context.environment):
                return
            if isinstance(stmt.value, tir.Call):
                self._emit_call(stmt.value, context)
                return
        if isinstance(stmt, tir.BufferStore):
            self._emit_task(
                "buffer_store",
                context,
                metadata={"buffer": str(stmt.buffer.name), "tir": str(stmt)},
            )
            return
        raise UnsupportedSimOpError(
            f"unsupported final TIR statement {type(stmt).__name__} in {self.kernel_name}"
        )

    def _visit_attr(self, stmt: Any, context: _Context) -> None:
        key = str(stmt.attr_key)
        if key in {"thread_extent", "virtual_thread"}:
            tag = str(getattr(stmt.node, "thread_tag", ""))
            variable = getattr(stmt.node, "var", stmt.node)
            extent = self._require_int(stmt.value, context.environment, f"{tag} extent")
            if tag == "blockIdx.x":
                for core_id in range(extent):
                    environment = dict(context.environment)
                    environment[variable] = core_id
                    self._visit(
                        stmt.body,
                        replace(context, core_id=core_id, environment=environment),
                    )
                return
            if tag in {"blockIdx.y", "threadIdx.x"}:
                for vector_index in range(extent):
                    environment = dict(context.environment)
                    environment[variable] = vector_index
                    self._visit(
                        stmt.body,
                        replace(
                            context,
                            vector_index=vector_index,
                            environment=environment,
                        ),
                    )
                return
        if key == "resource_scope":
            scope_value = self._require_int(stmt.value, context.environment, key)
            if scope_value == 0:
                if context.vector_index not in {None, 0}:
                    return
                self._visit(stmt.body, replace(context, lane=Lane.CUBE))
                return
            if scope_value == 1:
                vector_index = context.vector_index or 0
                lane = Lane.VECTOR_0 if vector_index == 0 else Lane.VECTOR_1
                self._visit(stmt.body, replace(context, lane=lane))
                return
            raise UnsupportedSimOpError(f"unsupported resource_scope value: {scope_value}")
        if key == "storage_scope":
            name = self._var_name(stmt.node)
            self.storage_scope_by_var[name] = MemoryScope.parse(self._literal(stmt.value))
        self._visit(stmt.body, context)

    def _collect_allocate(self, stmt: Any, context: _Context) -> None:
        name = self._var_name(stmt.buffer_var)
        scope = self.storage_scope_by_var.get(name)
        if scope is None:
            annotation = getattr(stmt.buffer_var, "type_annotation", None)
            storage_scope = getattr(annotation, "storage_scope", "")
            scope = MemoryScope.parse(str(storage_scope or "local.var"))
        shape = tuple(
            self._extent_or_symbol(extent, context.environment) for extent in stmt.extents
        )
        self.buffers.setdefault(name, BufferSpec(name, scope, shape, str(stmt.dtype)))

    def _emit_call(self, call: Any, context: _Context) -> None:
        operation, arguments = self._call_operation(call)
        metadata = {
            "arguments": tuple(self._literal(arg) for arg in arguments),
            "tir": str(call),
        }
        metadata.update(self._sync_metadata(operation, arguments))
        span = getattr(call, "span", None)
        if span is not None:
            metadata["span"] = str(span)
        self._emit_task(operation, context, metadata=metadata)

    def _emit_task(
        self, operation: str, context: _Context, *, metadata: Mapping[str, Any]
    ) -> None:
        try:
            lane, pipe, normalized = classify_operation(operation, context.lane)
        except UnsupportedSimOpError as error:
            span = metadata.get("span", "unknown")
            raise UnsupportedSimOpError(
                f"{error}; platform={self.platform}; span={span}; "
                f"lane={context.lane.value}"
            ) from error
        task_id = f"c{context.core_id}-{lane.value}-{self.task_counter}"
        self.task_counter += 1
        task = Task(
            task_id,
            normalized,
            context.core_id,
            lane,
            pipe,
            self.timing_profile.estimate_cycles(normalized),
            metadata=metadata,
        )
        self.tasks[context.core_id].append(task)

    def _call_operation(self, call: Any) -> Tuple[str, Tuple[Any, ...]]:
        name = str(call.op.name)
        arguments = tuple(call.args)
        if name == "tir.call_extern":
            if not arguments:
                raise UnsupportedSimOpError("tir.call_extern has no operation name")
            operation = self._literal(arguments[0])
            if not isinstance(operation, str):
                raise UnsupportedSimOpError("tir.call_extern operation name is not a string")
            return operation, arguments[1:]
        return name, arguments

    def _sync_metadata(self, operation: str, arguments: Tuple[Any, ...]) -> Dict[str, Any]:
        normalized = operation.lower()
        metadata: Dict[str, Any] = {}
        if "set_flag" in normalized or "wait_flag" in normalized:
            if len(arguments) >= 3:
                metadata.update({
                    "src_pipe": str(self._literal(arguments[0])).lower(),
                    "dst_pipe": str(self._literal(arguments[1])).lower(),
                    "flag_id": self._literal(arguments[2]),
                })
        if "cross_flag" in normalized:
            flag_arg = 1 if "set_" in normalized else 0
            if len(arguments) > flag_arg:
                metadata["flag_id"] = self._literal(arguments[flag_arg])
            metadata["channel"] = "cv"
        if "barrier" in normalized and arguments:
            metadata["target_pipe"] = str(self._literal(arguments[0])).lower()
        return metadata

    def _require_int(self, value: Any, environment: Mapping[Any, int], what: str) -> int:
        result = self._const_int(value, environment)
        if result is None:
            raise UnsupportedSimOpError(
                f"dynamic {what} is not supported by the first A2/A3 simulator bridge: {value}"
            )
        return result

    def _const_int(self, value: Any, environment: Mapping[Any, int]) -> Optional[int]:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        substituted = value
        if environment:
            replacements = {
                var: self.tir.IntImm(getattr(var, "dtype", "int32"), number)
                for var, number in environment.items()
            }
            substituted = self.tir.stmt_functor.substitute(value, replacements)
        simplified = self.analyzer.simplify(substituted)
        literal = getattr(simplified, "value", None)
        return int(literal) if isinstance(literal, (bool, int)) else None

    def _extent_or_symbol(self, value: Any, environment: Mapping[Any, int]) -> Any:
        literal = self._const_int(value, environment)
        return literal if literal is not None else str(value)

    def _is_zero(self, value: Any, environment: Mapping[Any, int]) -> bool:
        literal = self._const_int(value, environment)
        return literal == 0

    @staticmethod
    def _literal(value: Any) -> Any:
        literal = getattr(value, "value", None)
        if isinstance(literal, (bool, int, float, str)):
            return literal
        return str(value)

    @staticmethod
    def _var_name(value: Any) -> str:
        return str(getattr(value, "name", getattr(value, "name_hint", value)))
