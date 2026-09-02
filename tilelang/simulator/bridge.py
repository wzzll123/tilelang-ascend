# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Lower final A2/A3 TIR into the simulator's backend-neutral program model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from .errors import ProgramValidationError, UnsupportedSimOpError
from .layout import BYTE_PER_C0, C0_NUM_PER_FRACTAL, storage_elements
from .memory import contiguous_strides_bytes, dtype_size_bytes
from .profile import TimingProfile, default_timing_profile, normalize_platform
from .program import (
    AffineInt,
    BufferRegion,
    BufferSpec,
    CoreProgram,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    SymbolicInt,
    Task,
)


_VECTOR_OPS = frozenset({
    "abs", "add", "adds", "arith_progression", "axpy", "bilinear_interpolation",
    "bitwise_and", "bitwise_lshift", "bitwise_not", "bitwise_or", "bitwise_rshift",
    "bitwise_xor", "block_reduce_max", "block_reduce_min", "block_reduce_sum",
    "broadcast", "brcb_experiment", "cast", "clamp", "clamp_max", "clamp_min",
    "compare", "compare_scalar", "cos", "createvecindex", "div", "divs", "duplicate",
    "exp", "fill", "gather", "gather_mask", "gather_mask_experiment", "gatherb",
    "init_sort_buf", "leaky_relu", "ln", "max", "maxs", "merge_sort", "min", "mins",
    "mul", "muls", "pow", "reciprocal", "reduce", "reduce_max", "reduce_min",
    "reduce_sum", "relu", "round", "rsqrt", "select",
    "sigmoid", "silu", "sin", "sort", "sort32", "sqrt", "sub", "subs", "tail_binary",
    "tail_broadcast", "tail_compare", "tail_compare_scalar", "tail_reduce", "tail_scalar",
    "tail_select", "tail_unary", "topk", "transpose", "wholereducemax",
    "wholereducemin", "wholereducesum", "abs_experiment", "brcb_experiment",
    "datacachecleanandinvalid_experiment", "exp_experiment", "fill_experiment",
    "gather_mask_experiment", "mins_experiment", "reducesum_experiment",
    "reducesum_mask_experiment", "row_expand_div_experiment",
    "row_expand_mul_experiment", "row_expand_sub_experiment", "sub_experiment",
    "sum_experiment",
})

_TAIL_OPERATIONS = {
    "tail_binary": frozenset({"add", "div", "max", "min", "mul", "sub"}),
    "tail_scalar": frozenset({"adds", "maxs", "mins", "muls"}),
    "tail_unary": frozenset({
        "abs", "exp", "ln", "reciprocal", "relu", "rsqrt", "sqrt",
    }),
}

_TAIL_REDUCE_OPERATIONS = frozenset({"reduce_max", "reduce_min", "reduce_sum"})


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
    short = _short_operation(normalized)

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


def _short_operation(operation: str) -> str:
    short = operation.strip().lower()
    for prefix in ("tl.ascend_", "tl::ascend::", "ascendc::"):
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    if "<" in short:
        short = short.split("<", 1)[0]
    if short == "tl.arith_progression":
        short = "arith_progression"
    return short


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
        import tvm
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
        self.buffer_name_by_data_var: Dict[str, str] = {}
        self.storage_scope_by_var: Dict[str, MemoryScope] = {}
        self.address_by_var: Dict[str, int] = {}
        self.size_by_var: Dict[str, int] = {}
        self.last_writes: list[Tuple[BufferRegion, str]] = []
        self.last_reads: list[Tuple[BufferRegion, str]] = []
        self.task_counter = 0
        self.kernel_name = "main"

    def build(self, func_or_mod: Any) -> KernelProgram:
        func = self._select_prim_func(func_or_mod)
        if func.attrs is not None and "global_symbol" in func.attrs:
            self.kernel_name = str(func.attrs["global_symbol"])
        self._collect_memory_plan(func)
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

    def _collect_memory_plan(self, func: Any) -> None:
        if func.attrs is None:
            return
        for attribute, destination in (
            ("address_map", self.address_by_var),
            ("size_map", self.size_by_var),
        ):
            if attribute not in func.attrs:
                continue
            for variable, value in func.attrs[attribute].items():
                number = self._const_int(value, {})
                if number is None or number < 0:
                    raise ProgramValidationError(
                        f"PrimFunc {attribute} must contain non-negative integers"
                    )
                destination[self._var_name(variable)] = number

    def _buffer_spec(
        self,
        name: str,
        scope: MemoryScope,
        shape: Tuple[Any, ...],
        dtype: str,
    ) -> BufferSpec:
        address = self.address_by_var.get(name)
        size_bytes = self.size_by_var.get(name)
        metadata = {"planned_address": True} if address is not None else {}
        return BufferSpec(
            name,
            scope,
            shape,
            dtype,
            size_bytes=size_bytes,
            address=address,
            metadata=metadata,
        )

    def _collect_parameter_buffers(self, func: Any) -> None:
        for _, buffer in func.buffer_map.items():
            name = str(buffer.name)
            shape = tuple(self._extent_or_symbol(extent, {}) for extent in buffer.shape)
            self.buffers.setdefault(
                name,
                self._buffer_spec(name, MemoryScope.GM, shape, str(buffer.dtype)),
            )
            self.buffer_name_by_data_var[self._var_name(buffer.data)] = name

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
            for buffer in stmt.alloc_buffers:
                self._collect_block_buffer(buffer, context)
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
        self.buffers.setdefault(
            name, self._buffer_spec(name, scope, shape, str(stmt.dtype))
        )

    def _collect_block_buffer(self, buffer: Any, context: _Context) -> None:
        name = str(buffer.name)
        shape = tuple(
            self._extent_or_symbol(extent, context.environment) for extent in buffer.shape
        )
        self.buffers.setdefault(
            name,
            self._buffer_spec(
                name,
                MemoryScope.parse(str(buffer.scope())),
                shape,
                str(buffer.dtype),
            ),
        )
        self.buffer_name_by_data_var[self._var_name(buffer.data)] = name

    def _emit_call(self, call: Any, context: _Context) -> None:
        operation, arguments = self._call_operation(call)
        lowered_operation = operation
        tail_kind = _short_operation(operation)
        if tail_kind in _TAIL_OPERATIONS:
            if not arguments:
                raise ProgramValidationError(f"{tail_kind} requires an operation tag")
            operation_tag = self._literal(arguments[0])
            if not isinstance(operation_tag, str):
                raise ProgramValidationError(
                    f"{tail_kind} operation tag must be a string, got {operation_tag!r}"
                )
            operation = _short_operation(operation_tag)
            if operation not in _TAIL_OPERATIONS[tail_kind]:
                raise UnsupportedSimOpError(
                    f"operation tag {operation_tag!r} is not valid for {tail_kind}"
                )
            arguments = arguments[1:]
        elif tail_kind == "tail_reduce":
            if not arguments:
                raise ProgramValidationError("tail_reduce requires a reduction kind")
            operation_tag = self._literal(arguments[0])
            operation = _short_operation(str(operation_tag))
            if operation not in _TAIL_REDUCE_OPERATIONS:
                raise UnsupportedSimOpError(
                    f"unsupported tail_reduce kind {operation_tag!r}"
                )
            arguments = arguments[1:]
        metadata = {
            "arguments": tuple(self._literal(arg) for arg in arguments),
            "tir": str(call),
        }
        if tail_kind in _TAIL_OPERATIONS or tail_kind == "tail_reduce":
            metadata.update({
                "lowered_operation": lowered_operation,
                "tail_kind": tail_kind,
            })
        else:
            tail_kind = None
        metadata.update(
            self._functional_metadata(operation, arguments, context, tail_kind=tail_kind)
        )
        metadata.update(self._sync_metadata(operation, arguments))
        span = getattr(call, "span", None)
        if span is not None:
            metadata["span"] = str(span)
        self._emit_task(operation, context, metadata=metadata)

    def _functional_metadata(
        self,
        operation: str,
        arguments: Tuple[Any, ...],
        context: _Context,
        *,
        tail_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract executable operands for currently supported copy and vector forms."""
        normalized = operation.lower()
        short = _short_operation(normalized)
        if short in {"add", "sub", "mul", "div", "min", "max"}:
            return self._binary_metadata(arguments, context, tail=tail_kind is not None)
        if short in {
            "adds", "subs", "muls", "divs", "mins", "maxs", "leaky_relu",
        }:
            return self._scalar_metadata(arguments, context, tail=tail_kind is not None)
        if short == "axpy":
            metadata = self._scalar_metadata(arguments, context, tail=False)
            if isinstance(metadata.get("dst"), BufferRegion):
                metadata["accumulator"] = metadata["dst"]
            return metadata
        if short in {"abs", "exp", "ln", "reciprocal", "relu", "rsqrt", "sqrt"}:
            return self._unary_metadata(arguments, context, tail=tail_kind is not None)
        if short in {"sigmoid", "silu", "sin", "cos"}:
            return self._scratch_unary_metadata(arguments, context)
        if short == "pow":
            return self._pow_metadata(arguments, context)
        if short in {"clamp", "clamp_max", "clamp_min"}:
            return self._clamp_metadata(short, arguments, context)
        if short == "broadcast":
            return self._broadcast_metadata(arguments, context)
        if short == "tail_broadcast":
            return self._tail_broadcast_metadata(arguments, context)
        if short in {"compare", "compare_scalar"}:
            return self._compare_metadata(short, arguments, context)
        if short in {"tail_compare", "tail_compare_scalar"}:
            return self._tail_compare_metadata(short, arguments, context)
        if short == "tail_select":
            return self._tail_select_metadata(arguments, context)
        if short == "select":
            return self._select_metadata(arguments, context)
        if short == "cast":
            return self._cast_metadata(arguments, context)
        if short == "fill":
            return self._fill_metadata(arguments, context)
        if short == "reduce":
            return self._reduce_metadata(arguments, context)
        if short == "copy_gm_to_l1_linear":
            return self._gm_to_l1_linear_metadata(arguments, context)
        if short == "copy_gm_to_l1":
            return self._gm_to_l1_zn_metadata(arguments, context)
        if short in {"copy_l1_to_l0a", "copy_l1_to_l0b"}:
            return self._l1_to_l0_metadata(
                short, normalized, arguments, context
            )
        if short == "copy_l0c_to_gm":
            return self._l0c_to_gm_metadata(normalized, arguments, context)
        if short == "mma":
            return self._mma_metadata(arguments, context)
        if short in _TAIL_REDUCE_OPERATIONS and tail_kind == "tail_reduce":
            return self._tail_reduce_metadata(arguments, context)
        if not any(name in normalized for name in ("copy_gm_to_ub", "copy_ub_to_gm")):
            return {}
        if len(arguments) < 3:
            return {}
        if len(arguments) >= 5:
            valid_rows = self._runtime_int(arguments[3], context.environment)
            valid_cols = self._runtime_int(arguments[4], context.environment)
            if valid_rows is None or valid_cols is None:
                return {}
            if ((isinstance(valid_rows, int) and valid_rows < 0)
                    or (isinstance(valid_cols, int) and valid_cols < 0)):
                raise ProgramValidationError("copy valid rows/columns must not be negative")
            shape = (valid_rows, valid_cols)
            source = self._access_buffer_region(arguments[0], shape, context)
            destination = self._access_buffer_region(arguments[1], shape, context)
            details: Dict[str, Any] = {
                "valid_rows": valid_rows,
                "valid_cols": valid_cols,
                "stride_n": self._literal(arguments[2]),
            }
            if len(arguments) > 5:
                details["pad_value"] = self._literal(
                    self.analyzer.simplify(arguments[5])
                )
            if len(arguments) > 6:
                physical_rows_arg = arguments[6] if len(arguments) > 7 else 1
                physical_cols_arg = arguments[7] if len(arguments) > 7 else arguments[6]
                physical_rows = self._runtime_int(
                    physical_rows_arg, context.environment
                )
                physical_cols = self._runtime_int(
                    physical_cols_arg, context.environment
                )
                if physical_rows is None or physical_cols is None:
                    return {}
                if all(isinstance(value, int) for value in (
                    valid_rows, valid_cols, physical_rows, physical_cols
                )) and (valid_rows > physical_rows or valid_cols > physical_cols):
                    raise ProgramValidationError(
                        "copy valid rectangle must fit its physical destination tile"
                    )
                details["physical_rows"] = physical_rows
                details["physical_cols"] = physical_cols
        else:
            length = self._runtime_int(arguments[2], context.environment)
            if length is None or (isinstance(length, int) and length < 0):
                return {}
            source = self._access_buffer_region(arguments[0], (length,), context)
            destination = self._access_buffer_region(arguments[1], (length,), context)
            details = {"valid_elements": length}
        if source is None or destination is None:
            return {}
        metadata = {"src": source, "dst": destination, "copy": details}
        if (
            "copy_gm_to_ub" in normalized
            and "pad_value" in details
            and "physical_rows" in details
        ):
            pad_destination = self._access_buffer_region(
                arguments[1],
                (details["physical_rows"], details["physical_cols"]),
                context,
            )
            if pad_destination is not None and self._region_fits_buffer(pad_destination):
                metadata["pad_dst"] = pad_destination
            else:
                metadata["pad_disabled_reason"] = "physical tile exceeds buffer view"
        return metadata

    def _gm_to_l1_linear_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Final copy ABI: src, dst, realSrcN, validM, validN, dstM, dstN.
        if len(arguments) != 7:
            return {}
        dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[2:7]
        )
        if any(value is None for value in dimensions):
            return {}
        source_cols, valid_rows, valid_cols, physical_rows, physical_cols = dimensions
        if any(isinstance(value, int) and value < 0 for value in dimensions):
            raise ProgramValidationError("GM-to-L1 linear extents must not be negative")
        if all(isinstance(value, int) for value in dimensions):
            if valid_rows > physical_rows or valid_cols > physical_cols:
                raise ProgramValidationError(
                    "GM-to-L1 valid rectangle must fit its physical L1 tile"
                )
            if valid_cols > source_cols:
                raise ProgramValidationError(
                    "GM-to-L1 valid columns exceed the GM row stride"
                )
        source = self._access_buffer_region(
            arguments[0], (valid_rows, valid_cols), context
        )
        destination = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        if source is None or destination is None:
            return {}
        if source.scope is not MemoryScope.GM or destination.scope is not MemoryScope.L1:
            raise ProgramValidationError(
                "copy_gm_to_l1_linear requires GM source and L1 destination"
            )
        if source.dtype != destination.dtype:
            raise ProgramValidationError(
                "copy_gm_to_l1_linear requires matching source/destination dtype"
            )
        itemsize = dtype_size_bytes(source.dtype)
        if isinstance(valid_cols, int) and (valid_cols * itemsize) % 32:
            raise ProgramValidationError(
                "copy_gm_to_l1_linear row width must be 32-byte aligned"
            )
        return {
            "src": source,
            "dst": destination,
            "copy": {
                "layout": "row_major",
                "valid_rows": valid_rows,
                "valid_cols": valid_cols,
                "source_cols": source_cols,
                "physical_rows": physical_rows,
                "physical_cols": physical_cols,
            },
        }

    def _gm_to_l1_zn_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Final copy ABI: src, dst, realSrcN, validM, validN, dstM, dstN.
        if len(arguments) != 7:
            return {}
        dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[2:7]
        )
        if any(value is None for value in dimensions):
            return {}
        if not all(isinstance(value, int) for value in dimensions):
            raise UnsupportedSimOpError(
                "functional copy_gm_to_l1 requires static source and tile extents"
            )
        source_cols, valid_rows, valid_cols, physical_rows, physical_cols = dimensions
        if any(value < 0 for value in dimensions):
            raise ProgramValidationError("GM-to-L1 zN extents must not be negative")
        if physical_rows == 0 or physical_cols == 0:
            raise ProgramValidationError(
                "GM-to-L1 zN physical tile extents must be positive"
            )
        if valid_rows > physical_rows or valid_cols > physical_cols:
            raise ProgramValidationError(
                "GM-to-L1 valid rectangle must fit its physical L1 tile"
            )
        if valid_cols > source_cols:
            raise ProgramValidationError(
                "GM-to-L1 valid columns exceed the GM row stride"
            )

        source = self._access_buffer_region(
            arguments[0], (valid_rows, valid_cols), context
        )
        destination_name = self._access_ptr_data_name(arguments[1])
        if source is None or destination_name is None:
            return {}
        destination_buffer = self.buffer_name_by_data_var.get(destination_name)
        if destination_buffer is None:
            return {}
        destination_spec = self.buffers[destination_buffer]
        if (
            source.scope is not MemoryScope.GM
            or destination_spec.scope is not MemoryScope.L1
        ):
            raise ProgramValidationError(
                "copy_gm_to_l1 requires GM source and L1 destination"
            )
        if source.dtype != destination_spec.dtype:
            raise ProgramValidationError(
                "copy_gm_to_l1 requires matching source/destination dtype"
            )

        itemsize = dtype_size_bytes(source.dtype)
        elements_per_c0 = BYTE_PER_C0 // itemsize
        if (
            physical_rows % C0_NUM_PER_FRACTAL
            or physical_cols % elements_per_c0
        ):
            raise UnsupportedSimOpError(
                "functional copy_gm_to_l1 requires a fractal/C0-aligned physical tile"
            )
        physical_elements = storage_elements(
            "zN", (physical_rows, physical_cols), itemsize
        )
        destination = self._access_buffer_region(
            arguments[1], (physical_elements,), context
        )
        if destination is None:
            return {}
        tile_bytes = physical_elements * itemsize
        if (
            not isinstance(destination.byte_offset, int)
            or destination.byte_offset % tile_bytes
        ):
            raise UnsupportedSimOpError(
                "functional copy_gm_to_l1 requires a tile-base-aligned destination"
            )
        destination_size = _buffer_size_bytes(destination_spec)
        if (
            destination_size is not None
            and destination.byte_offset + tile_bytes > destination_size
        ):
            raise ProgramValidationError(
                "GM-to-L1 zN physical tile exceeds the destination buffer"
            )
        return {
            "src": source,
            "dst": destination,
            "copy": {
                "layout": "zN",
                "valid_rows": valid_rows,
                "valid_cols": valid_cols,
                "source_cols": source_cols,
                "physical_rows": physical_rows,
                "physical_cols": physical_cols,
                "need_clear": (
                    valid_rows != physical_rows or valid_cols != physical_cols
                ),
            },
        }

    def _l1_to_l0_metadata(
        self,
        operation: str,
        operation_tag: str,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Final copy ABI: src, dst, dstM, dstN. The source shape and transpose
        # flag are compile-time template arguments on the operation name.
        if len(arguments) != 4:
            return {}
        marker = f"{operation}<"
        marker_start = operation_tag.find(marker)
        if marker_start < 0 or not operation_tag.endswith(">"):
            return {}
        parameters = [
            parameter.strip()
            for parameter in operation_tag[
                marker_start + len(marker):-1
            ].split(",")
        ]
        if len(parameters) != 4:
            raise UnsupportedSimOpError(
                f"malformed {operation} template {operation_tag!r}"
            )
        try:
            source_rows, source_cols = (int(parameters[index]) for index in (1, 2))
        except ValueError as error:
            raise UnsupportedSimOpError(
                f"{operation} requires static source template extents"
            ) from error
        if parameters[3] not in {"true", "false"}:
            raise UnsupportedSimOpError(
                f"{operation} requires a literal transpose template flag"
            )
        transpose = parameters[3] == "true"
        destination_dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[2:4]
        )
        if any(value is None for value in destination_dimensions):
            return {}
        if not all(isinstance(value, int) for value in destination_dimensions):
            raise UnsupportedSimOpError(
                f"functional {operation} requires static destination extents"
            )
        destination_rows, destination_cols = destination_dimensions
        if min(source_rows, source_cols, destination_rows, destination_cols) <= 0:
            raise ProgramValidationError(
                f"{operation} matrix extents must be positive"
            )

        source_shape = (
            (source_cols, source_rows) if transpose else (source_rows, source_cols)
        )
        destination_shape = (destination_rows, destination_cols)
        if (
            destination_rows > source_shape[0]
            or destination_cols > source_shape[1]
        ):
            raise ProgramValidationError(
                f"{operation} destination tile must fit its logical L1 source"
            )
        source_layout = "nZ" if transpose else "zN"
        destination_layout = "l0a" if operation.endswith("l0a") else "l0b"

        source_name = self._access_ptr_data_name(arguments[0])
        destination_name = self._access_ptr_data_name(arguments[1])
        if source_name is None or destination_name is None:
            return {}
        source_buffer = self.buffer_name_by_data_var.get(source_name)
        destination_buffer = self.buffer_name_by_data_var.get(destination_name)
        if source_buffer is None or destination_buffer is None:
            return {}
        source_spec = self.buffers[source_buffer]
        destination_spec = self.buffers[destination_buffer]
        expected_destination_scope = (
            MemoryScope.L0A if operation.endswith("l0a") else MemoryScope.L0B
        )
        if (
            source_spec.scope is not MemoryScope.L1
            or destination_spec.scope is not expected_destination_scope
        ):
            raise ProgramValidationError(
                f"{operation} requires L1 source and {expected_destination_scope.value} destination"
            )
        if source_spec.dtype != destination_spec.dtype:
            raise ProgramValidationError(
                f"{operation} requires matching source/destination dtype"
            )

        itemsize = dtype_size_bytes(source_spec.dtype)
        source_elements = storage_elements(source_layout, source_shape, itemsize)
        destination_elements = storage_elements(
            destination_layout, destination_shape, itemsize
        )
        source = self._access_buffer_region(
            arguments[0], (source_elements,), context
        )
        destination = self._access_buffer_region(
            arguments[1], (destination_elements,), context
        )
        if source is None or destination is None:
            return {}
        for label, region, elements, spec in (
            ("source", source, source_elements, source_spec),
            ("destination", destination, destination_elements, destination_spec),
        ):
            tile_bytes = elements * itemsize
            if (
                not isinstance(region.byte_offset, int)
                or region.byte_offset % tile_bytes
            ):
                raise UnsupportedSimOpError(
                    f"functional {operation} requires a tile-base-aligned {label}"
                )
            buffer_size = _buffer_size_bytes(spec)
            if buffer_size is not None and region.byte_offset + tile_bytes > buffer_size:
                raise ProgramValidationError(
                    f"{operation} physical {label} tile exceeds its buffer"
                )
        return {
            "src": source,
            "dst": destination,
            "copy": {
                "layout_transform": True,
                "source_layout": source_layout,
                "destination_layout": destination_layout,
                "source_shape": source_shape,
                "destination_shape": destination_shape,
                "transpose": transpose,
            },
        }

    def _l0c_to_gm_metadata(
        self,
        operation_tag: str,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Final ABI: src, dst, dstStrideN, validM, validN, srcM, srcN,
        # enableRelu, unitFlag.
        if len(arguments) != 9:
            return {}
        marker = "copy_l0c_to_gm<"
        marker_start = operation_tag.find(marker)
        if marker_start < 0 or not operation_tag.endswith(">"):
            return {}
        parameters = [
            parameter.strip()
            for parameter in operation_tag[
                marker_start + len(marker):-1
            ].split(",")
        ]
        if len(parameters) != 6:
            raise UnsupportedSimOpError(
                f"malformed copy_l0c_to_gm template {operation_tag!r}"
            )
        if parameters[2] not in {"layout::rowmajor", "layout::row_major"}:
            raise UnsupportedSimOpError(
                f"functional copy_l0c_to_gm requires RowMajor GM, got {parameters[2]!r}"
            )
        try:
            template_rows, template_cols = (int(parameters[index]) for index in (3, 4))
        except ValueError as error:
            raise UnsupportedSimOpError(
                "copy_l0c_to_gm requires static source template extents"
            ) from error
        if parameters[5] not in {"true", "false"}:
            raise UnsupportedSimOpError(
                "copy_l0c_to_gm requires a literal ReLU template flag"
            )
        template_relu = parameters[5] == "true"
        dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[2:7]
        )
        if any(value is None for value in dimensions):
            return {}
        if not all(isinstance(value, int) for value in dimensions):
            raise UnsupportedSimOpError(
                "functional copy_l0c_to_gm requires static dimensions"
            )
        destination_cols, valid_rows, valid_cols, physical_rows, physical_cols = dimensions
        if min(destination_cols, physical_rows, physical_cols) <= 0:
            raise ProgramValidationError(
                "copy_l0c_to_gm physical extents and GM stride must be positive"
            )
        if valid_rows < 0 or valid_cols < 0:
            raise ProgramValidationError(
                "copy_l0c_to_gm valid extents must not be negative"
            )
        if valid_rows > physical_rows or valid_cols > physical_cols:
            raise ProgramValidationError(
                "copy_l0c_to_gm valid rectangle must fit its physical L0C tile"
            )
        if valid_cols > destination_cols:
            raise ProgramValidationError(
                "copy_l0c_to_gm valid columns exceed the GM row stride"
            )
        if (physical_rows, physical_cols) != (template_rows, template_cols):
            raise ProgramValidationError(
                "copy_l0c_to_gm physical extents disagree with its template"
            )
        relu_argument = self._literal(self.analyzer.simplify(arguments[7]))
        if not isinstance(relu_argument, (bool, int)):
            raise UnsupportedSimOpError(
                "copy_l0c_to_gm requires a literal enable_relu argument"
            )
        if bool(relu_argument) != template_relu:
            raise ProgramValidationError(
                "copy_l0c_to_gm ReLU argument disagrees with its template"
            )
        unit_flag = self._runtime_int(arguments[8], context.environment)
        if unit_flag != 0:
            raise UnsupportedSimOpError(
                "functional copy_l0c_to_gm supports standalone unitFlag=0 only"
            )

        source_name = self._access_ptr_data_name(arguments[0])
        if source_name is None:
            return {}
        source_buffer = self.buffer_name_by_data_var.get(source_name)
        if source_buffer is None:
            return {}
        source_spec = self.buffers[source_buffer]
        source_elements = storage_elements(
            "l0c",
            (physical_rows, physical_cols),
            dtype_size_bytes(source_spec.dtype),
        )
        source = self._access_buffer_region(
            arguments[0], (source_elements,), context
        )
        destination = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        if source is None or destination is None:
            return {}
        if source.scope is not MemoryScope.L0C or destination.scope is not MemoryScope.GM:
            raise ProgramValidationError(
                "copy_l0c_to_gm requires L0C source and GM destination"
            )
        source_template_dtype = _ascend_template_dtype(parameters[0])
        destination_template_dtype = _ascend_template_dtype(parameters[1])
        if source_template_dtype is None or destination_template_dtype is None:
            raise UnsupportedSimOpError(
                "copy_l0c_to_gm uses an unsupported template dtype"
            )
        if (
            source.dtype != source_template_dtype
            or destination.dtype != destination_template_dtype
        ):
            raise ProgramValidationError(
                "copy_l0c_to_gm buffer dtypes disagree with its template"
            )
        source_bytes = source_elements * dtype_size_bytes(source.dtype)
        if (
            not isinstance(source.byte_offset, int)
            or source.byte_offset % source_bytes
        ):
            raise UnsupportedSimOpError(
                "functional copy_l0c_to_gm requires a tile-base-aligned source"
            )
        source_size = _buffer_size_bytes(source_spec)
        if source_size is not None and source.byte_offset + source_bytes > source_size:
            raise ProgramValidationError(
                "copy_l0c_to_gm physical source tile exceeds its buffer"
            )
        return {
            "src": source,
            "dst": destination,
            "copy": {
                "layout_transform": True,
                "source_layout": "l0c",
                "destination_layout": "row_major",
                "source_shape": (physical_rows, physical_cols),
                "destination_shape": (valid_rows, valid_cols),
                "relu": template_relu,
                "unit_flag": unit_flag,
                "destination_cols": destination_cols,
            },
        }

    def _mma_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Final intrinsic ABI: tag, A, B, C, init, K[, n_actual, unitFlag].
        if len(arguments) not in {6, 8}:
            return {}
        tag = self._literal(arguments[0])
        if not isinstance(tag, str) or not tag.startswith("mma<") or not tag.endswith(">"):
            raise UnsupportedSimOpError(
                f"functional mma supports the non-bias mma template, got {tag!r}"
            )
        parameters = [part.strip() for part in tag[4:-1].split(",")]
        if len(parameters) != 4:
            raise UnsupportedSimOpError(f"malformed mma template {tag!r}")
        try:
            rows, cols = (int(parameters[index]) for index in (2, 3))
        except ValueError as error:
            raise UnsupportedSimOpError(
                "functional mma requires static M/N template extents"
            ) from error
        inner = self._runtime_int(arguments[5], context.environment)
        if inner is None:
            return {}
        if not isinstance(inner, int):
            raise UnsupportedSimOpError("functional mma requires a static K extent")
        if min(rows, cols, inner) <= 0:
            raise ProgramValidationError("mma M/N/K extents must be positive")
        init_value = self._literal(self.analyzer.simplify(arguments[4]))
        if not isinstance(init_value, (bool, int)):
            raise UnsupportedSimOpError("functional mma requires a literal init flag")
        initialize = bool(init_value)
        actual_cols = cols
        unit_flag = 0
        if len(arguments) == 8:
            actual_cols = self._runtime_int(arguments[6], context.environment)
            unit_flag = self._runtime_int(arguments[7], context.environment)
            if actual_cols != cols:
                raise UnsupportedSimOpError(
                    "functional mma does not yet support n_actual smaller than N"
                )
            if unit_flag != 0:
                raise UnsupportedSimOpError(
                    "functional mma supports standalone unitFlag=0 only"
                )

        input_dtype = _ascend_template_dtype(parameters[0])
        accumulator_dtype = _ascend_template_dtype(parameters[1])
        if (input_dtype, accumulator_dtype) != ("float16", "float32"):
            raise UnsupportedSimOpError(
                "functional mma currently supports half inputs and float accumulation"
            )
        a_elements = storage_elements(
            "l0a", (rows, inner), dtype_size_bytes(input_dtype)
        )
        b_elements = storage_elements(
            "l0b", (inner, cols), dtype_size_bytes(input_dtype)
        )
        c_elements = storage_elements(
            "l0c", (rows, cols), dtype_size_bytes(accumulator_dtype)
        )
        left = self._access_buffer_region(arguments[1], (a_elements,), context)
        right = self._access_buffer_region(arguments[2], (b_elements,), context)
        destination = self._access_buffer_region(
            arguments[3], (c_elements,), context
        )
        if left is None or right is None or destination is None:
            return {}
        if (
            left.scope is not MemoryScope.L0A
            or right.scope is not MemoryScope.L0B
            or destination.scope is not MemoryScope.L0C
        ):
            raise ProgramValidationError("mma requires L0A, L0B, and L0C operands")
        if (
            left.dtype != input_dtype
            or right.dtype != input_dtype
            or destination.dtype != accumulator_dtype
        ):
            raise ProgramValidationError("mma buffer dtypes disagree with its template")
        for label, region, elements in (
            ("L0A", left, a_elements),
            ("L0B", right, b_elements),
            ("L0C", destination, c_elements),
        ):
            tile_bytes = elements * dtype_size_bytes(region.dtype)
            if (
                not isinstance(region.byte_offset, int)
                or region.byte_offset % tile_bytes
            ):
                raise UnsupportedSimOpError(
                    f"functional mma requires a tile-base-aligned {label} operand"
                )
            buffer_size = _buffer_size_bytes(self.buffers[region.buffer])
            if buffer_size is not None and region.byte_offset + tile_bytes > buffer_size:
                raise ProgramValidationError(
                    f"mma {label} physical tile exceeds its buffer"
                )
        metadata: Dict[str, Any] = {
            "lhs": left,
            "rhs": right,
            "dst": destination,
            "mma": {
                "rows": rows,
                "cols": cols,
                "inner": inner,
                "init": initialize,
                "n_actual": actual_cols,
                "unit_flag": unit_flag,
            },
        }
        if not initialize:
            metadata["accumulator"] = destination
        return metadata

    def _reduce_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) < 4:
            return {}
        tag = self._literal(arguments[0])
        if not isinstance(tag, str) or "<" not in tag or not tag.endswith(">"):
            return {}
        kind, encoded = tag.split("<", 1)
        kind = _short_operation(kind)
        if kind not in _TAIL_REDUCE_OPERATIONS:
            raise UnsupportedSimOpError(f"unsupported reduce kind {kind!r}")
        parameters = [part.strip() for part in encoded[:-1].split(",")]
        if len(parameters) != 4:
            raise UnsupportedSimOpError(f"malformed reduce tag {tag!r}")
        try:
            rows, cols, axis = (int(parameters[index]) for index in (1, 2, 3))
        except ValueError as error:
            raise UnsupportedSimOpError(
                f"reduce tag requires static M/N/axis, got {tag!r}"
            ) from error
        if rows < 0 or cols < 0:
            raise ProgramValidationError("reduce extents must not be negative")
        if axis not in {0, -1}:
            raise UnsupportedSimOpError(f"reduce supports axis 0 or -1, got {axis}")

        cursor = len(arguments) - 1
        physical_cols = cols
        if str(getattr(arguments[cursor], "dtype", "")) != "bool":
            candidate = self._const_int(arguments[cursor], context.environment)
            if candidate is None:
                return {}
            physical_cols = candidate
            cursor -= 1
        clear_value = self._literal(arguments[cursor])
        if clear_value not in {False, True, 0, 1}:
            return {}
        clear = bool(clear_value)
        scratch_arguments = arguments[3:cursor]
        if len(scratch_arguments) > 2:
            return {}
        if not clear:
            expected_tmp_counts = {0: {1}, -1: {1, 2}}[axis]
            if len(scratch_arguments) not in expected_tmp_counts:
                raise UnsupportedSimOpError(
                    f"reduce clear=false axis {axis} requires "
                    f"{sorted(expected_tmp_counts)} tmp view count, got "
                    f"{len(scratch_arguments)}"
                )
        source = self._access_buffer_region(arguments[2], (rows, cols), context)
        output_count = cols if axis == 0 else rows
        destination = self._access_buffer_region(
            arguments[1], (output_count,), context
        )
        if source is None or destination is None:
            return {}
        metadata: Dict[str, Any] = {
            "src": source,
            "dst": destination,
            "reduce_kind": kind,
            "reduce_axis": axis,
            "clear": clear,
            "rows": rows,
            "cols": cols,
            "physical_cols": physical_cols,
        }
        if not clear:
            metadata["accumulator"] = destination
        for index, argument in enumerate(scratch_arguments):
            extent = self._access_ptr_extent(argument, context)
            if extent is None:
                return {}
            scratch = self._access_buffer_region(argument, (extent,), context)
            if scratch is None:
                return {}
            metadata["scratch" if index == 0 else "output_scratch"] = scratch
        return metadata

    def _region_fits_buffer(self, region: BufferRegion) -> bool:
        bounds = _region_bounds(region)
        spec = self.buffers[region.buffer]
        if bounds is None:
            return region.byte_offset == 0
        if any(not isinstance(extent, int) for extent in spec.shape):
            return True
        size_bytes = spec.size_bytes
        if size_bytes is None:
            size_bytes = dtype_size_bytes(spec.dtype)
            for extent in spec.shape:
                size_bytes *= extent
        return bounds[0] >= 0 and bounds[1] <= size_bytes

    def _binary_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
        *,
        tail: bool = False,
    ) -> Dict[str, Any]:
        expected_arguments = 6 if tail else 4
        if len(arguments) != expected_arguments:
            return {}
        shape = self._vector_shape(arguments, context, tail=tail, count_index=3)
        if shape is None:
            return {}
        destination = self._access_buffer_region(arguments[0], shape, context)
        left = self._access_buffer_region(arguments[1], shape, context)
        right = self._access_buffer_region(arguments[2], shape, context)
        if destination is None or left is None or right is None:
            return {}
        metadata: Dict[str, Any] = {"dst": destination, "lhs": left, "rhs": right}
        if tail:
            metadata["tail"] = self._tail_details(arguments, context, 3)
        return metadata

    def _unary_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
        *,
        tail: bool = False,
    ) -> Dict[str, Any]:
        expected_arguments = 5 if tail else 3
        if len(arguments) != expected_arguments:
            return {}
        shape = self._vector_shape(arguments, context, tail=tail, count_index=2)
        if shape is None:
            return {}
        destination = self._access_buffer_region(arguments[0], shape, context)
        source = self._access_buffer_region(arguments[1], shape, context)
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {"dst": destination, "src": source}
        if tail:
            metadata["tail"] = self._tail_details(arguments, context, 2)
        return metadata

    def _scratch_unary_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) not in {3, 4}:
            return {}
        count = self._runtime_int(arguments[-1], context.environment)
        if count is None:
            return {}
        if isinstance(count, int) and count < 0:
            raise ProgramValidationError("vector count must not be negative")
        destination = self._access_buffer_region(arguments[0], (count,), context)
        source = self._access_buffer_region(arguments[1], (count,), context)
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {"dst": destination, "src": source}
        if len(arguments) == 4:
            scratch = self._access_buffer_region(arguments[2], (count,), context)
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _pow_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) not in {3, 4}:
            return {}
        count = self._access_ptr_extent(arguments[0], context)
        if count is None:
            return {}
        destination = self._access_buffer_region(arguments[0], (count,), context)
        left = self._access_buffer_region(arguments[1], (count,), context)
        right = self._access_buffer_region(arguments[2], (count,), context)
        if destination is None or left is None or right is None:
            return {}
        metadata: Dict[str, Any] = {"dst": destination, "lhs": left, "rhs": right}
        if len(arguments) == 4:
            scratch = self._access_buffer_region(arguments[3], (count,), context)
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _clamp_metadata(
        self,
        operation: str,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        offset = (
            1
            if arguments and isinstance(getattr(arguments[0], "value", None), str)
            else 0
        )
        operands = arguments[offset:]
        scalar_count = 2 if operation == "clamp" else 1
        base_count = 2 + scalar_count + 1
        if len(operands) not in {base_count, base_count + 1}:
            return {}
        has_scratch = len(operands) == base_count + 1
        destination_arg, source_arg = operands[:2]
        scalar_start = 3 if has_scratch else 2
        scalar_args = operands[scalar_start:scalar_start + scalar_count]
        count_arg = operands[scalar_start + scalar_count]
        count = self._runtime_int(count_arg, context.environment)
        scalars = tuple(self._literal(self.analyzer.simplify(arg)) for arg in scalar_args)
        if count is None:
            return {}
        if isinstance(count, int) and count < 0:
            raise ProgramValidationError("clamp count must not be negative")
        if any(not isinstance(value, (bool, int, float)) for value in scalars):
            raise UnsupportedSimOpError(
                f"functional {operation} requires literal bounds, got {scalars!r}"
            )
        destination = self._access_buffer_region(destination_arg, (count,), context)
        source = self._access_buffer_region(source_arg, (count,), context)
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {"dst": destination, "src": source}
        if operation == "clamp":
            metadata.update(min_value=scalars[0], max_value=scalars[1])
        else:
            metadata["scalar"] = scalars[0]
        if has_scratch:
            scratch = self._access_buffer_region(operands[2], (count,), context)
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _access_ptr_extent(
        self,
        pointer: Any,
        context: _Context,
    ) -> Optional[Any]:
        if isinstance(pointer, self.tir.Call) and str(pointer.op.name) == "tir.tvm_access_ptr":
            if len(pointer.args) < 4:
                return None
            extent = self._runtime_int(pointer.args[3], context.environment)
            if isinstance(extent, int) and extent < 0:
                raise ProgramValidationError("access pointer extent must not be negative")
            return extent
        return None

    def _broadcast_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        offset = (
            1
            if arguments and isinstance(getattr(arguments[0], "value", None), str)
            else 0
        )
        operands = arguments[offset:]
        if len(operands) < 5:
            return {}
        destination_arg, source_arg = operands[:2]
        cursor = 2
        scratch_arg = None
        if (
            isinstance(operands[cursor], self.tir.Call)
            and str(operands[cursor].op.name) == "tir.tvm_access_ptr"
        ):
            scratch_arg = operands[cursor]
            cursor += 1
        dimension = self._const_int(operands[cursor], context.environment)
        if dimension not in {1, 2}:
            raise UnsupportedSimOpError(
                f"functional broadcast supports only rank 1 or 2, got {dimension!r}"
            )
        cursor += 1
        if len(operands) != cursor + 2 * dimension:
            return {}
        destination_shape = tuple(
            self._runtime_int(value, context.environment)
            for value in operands[cursor:cursor + dimension]
        )
        source_shape = tuple(
            self._runtime_int(value, context.environment)
            for value in operands[cursor + dimension:]
        )
        if any(value is None for value in destination_shape + source_shape):
            return {}
        if any(
            isinstance(value, int) and value < 0
            for value in destination_shape + source_shape
        ):
            raise ProgramValidationError("broadcast extents must not be negative")
        if all(isinstance(value, int) for value in destination_shape + source_shape):
            for source_extent, destination_extent in zip(
                source_shape, destination_shape
            ):
                if source_extent not in {1, destination_extent}:
                    raise ProgramValidationError(
                        f"cannot broadcast source shape {source_shape} to {destination_shape}"
                    )
        destination = self._access_buffer_region(
            destination_arg, destination_shape, context
        )
        source = self._access_buffer_region(source_arg, source_shape, context)
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "src": source,
            "broadcast": {"dimension": dimension},
        }
        if scratch_arg is not None:
            scratch_extent = self._access_ptr_extent(scratch_arg, context)
            if scratch_extent is None:
                return {}
            scratch = self._access_buffer_region(
                scratch_arg, (scratch_extent,), context
            )
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _tail_broadcast_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Original broadcast ABI, including its name and optional tmp, plus
        # output/input valid rows and columns appended by tail propagation.
        if len(arguments) not in {12, 13}:
            return {}
        name = self._literal(arguments[0])
        if not isinstance(name, str):
            return {}
        has_scratch = len(arguments) == 13
        dimension_index = 4 if has_scratch else 3
        if self._const_int(arguments[dimension_index], context.environment) != 2:
            raise UnsupportedSimOpError("tail broadcast supports only rank 2")
        shape_index = dimension_index + 1
        physical_shape = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[shape_index:shape_index + 4]
        )
        valid_shape = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[-4:]
        )
        if any(value is None for value in physical_shape + valid_shape):
            return {}
        if any(
            isinstance(value, int) and value < 0
            for value in physical_shape + valid_shape
        ):
            raise ProgramValidationError("tail broadcast extents must not be negative")
        dst_rows, dst_cols, src_rows, src_cols = physical_shape
        valid_rows, valid_cols, src_valid_rows, src_valid_cols = valid_shape
        if all(isinstance(value, int) for value in physical_shape + valid_shape):
            if valid_rows > dst_rows or valid_cols > dst_cols:
                raise ProgramValidationError(
                    "tail broadcast valid output must fit its physical tile"
                )
            if src_valid_rows > src_rows or src_valid_cols > src_cols:
                raise ProgramValidationError(
                    "tail broadcast valid source must fit its physical tile"
                )
            if src_cols == 1 and src_rows != 1:
                axis = 1
                if valid_rows > src_valid_rows or src_valid_cols != 1:
                    raise ProgramValidationError(
                        "row tail broadcast requires one valid source column"
                    )
                source_shape = (valid_rows, 1)
            elif src_rows == 1 and src_cols != 1:
                axis = 0
                if valid_cols > src_valid_cols or src_valid_rows != 1:
                    raise ProgramValidationError(
                        "column tail broadcast requires one valid source row"
                    )
                source_shape = (1, valid_cols)
            else:
                raise UnsupportedSimOpError(
                    f"tail broadcast requires [M,1] or [1,N], got {(src_rows, src_cols)}"
                )
        else:
            # The propagation pass chooses the axis from static physical shape.
            raise UnsupportedSimOpError(
                "dynamic physical shape is not supported for tail broadcast"
            )
        destination = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        source = self._access_buffer_region(arguments[2], source_shape, context)
        if destination is None or source is None:
            return {}
        if axis == 1:
            itemsize = dtype_size_bytes(source.dtype)
            elements_per_block = 32 // itemsize
            aligned_cols = (
                (src_cols + elements_per_block - 1) // elements_per_block
            ) * elements_per_block
            source = replace(
                source, strides_bytes=(aligned_cols * itemsize, itemsize)
            )
        metadata: Dict[str, Any] = {
            "dst": destination,
            "src": source,
            "broadcast": {"dimension": 2, "axis": axis, "tail": True},
            "valid_rows": valid_rows,
            "valid_cols": valid_cols,
            "src_valid_rows": src_valid_rows,
            "src_valid_cols": src_valid_cols,
            "physical_rows": dst_rows,
            "physical_cols": dst_cols,
            "src_physical_rows": src_rows,
            "src_physical_cols": src_cols,
        }
        if has_scratch:
            scratch_extent = self._access_ptr_extent(arguments[3], context)
            if scratch_extent is None:
                return {}
            scratch = self._access_buffer_region(
                arguments[3], (scratch_extent,), context
            )
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _compare_metadata(
        self,
        operation: str,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) != 5:
            # The six-argument compare_scalar form reads its scalar through a
            # buffer plus index.  Keep it fail-closed until scalar BufferLoad
            # addressing is represented explicitly in SimIR.
            return {}
        count = self._runtime_int(arguments[4], context.environment)
        mode = getattr(arguments[3], "value", None)
        if count is None or not isinstance(mode, str):
            return {}
        mode = mode.upper()
        if mode not in {"EQ", "NE", "GT", "GE", "LT", "LE"}:
            raise UnsupportedSimOpError(f"unsupported compare mode {mode!r}")
        if isinstance(count, int) and count < 0:
            raise ProgramValidationError("compare count must not be negative")
        mask_extent = self._access_ptr_extent(arguments[0], context)
        if mask_extent is None:
            return {}
        destination = self._access_buffer_region(
            arguments[0], (mask_extent,), context
        )
        left = self._access_buffer_region(arguments[1], (count,), context)
        if destination is None or left is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "lhs": left,
            "compare_mode": mode,
            "count": count,
        }
        if operation == "compare":
            right = self._access_buffer_region(arguments[2], (count,), context)
            if right is None:
                return {}
            metadata["rhs"] = right
        else:
            scalar = self._literal(self.analyzer.simplify(arguments[2]))
            if not isinstance(scalar, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"functional compare_scalar requires a literal scalar, got {scalar!r}"
                )
            metadata["scalar"] = scalar
        return metadata

    def _select_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) < 7:
            return {}
        destination_arg, mask_arg, source_arg = arguments[:3]
        cursor = 3
        scratch_arg = None
        if (
            isinstance(arguments[cursor], self.tir.Call)
            and str(arguments[cursor].op.name) == "tir.tvm_access_ptr"
        ):
            scratch_arg = arguments[cursor]
            cursor += 1
        source_type = self._const_int(arguments[cursor], context.environment)
        cursor += 1
        if source_type == 1:
            if len(arguments) != cursor + 5:
                return {}
            source1_arg, mode_arg, count_arg = arguments[cursor:cursor + 3]
            scalar = self._literal(self.analyzer.simplify(source1_arg))
            if not isinstance(scalar, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"functional scalar select requires a literal, got {scalar!r}"
                )
            expected_mode = "VSEL_TENSOR_SCALAR_MODE"
        elif source_type == 2:
            if len(arguments) != cursor + 3:
                return {}
            source1_arg, mode_arg, count_arg = arguments[cursor:cursor + 3]
            scalar = None
            expected_mode = "VSEL_TENSOR_TENSOR_MODE"
        else:
            raise UnsupportedSimOpError(
                f"functional select supports source type 1 or 2, got {source_type!r}"
            )
        mode = getattr(mode_arg, "value", None)
        if mode != expected_mode:
            raise UnsupportedSimOpError(
                f"select source type {source_type} requires {expected_mode}, got {mode!r}"
            )
        count = self._runtime_int(count_arg, context.environment)
        if count is None:
            return {}
        if isinstance(count, int) and count < 0:
            raise ProgramValidationError("select count must not be negative")
        mask_extent = self._access_ptr_extent(mask_arg, context)
        if mask_extent is None:
            return {}
        destination = self._access_buffer_region(destination_arg, (count,), context)
        mask = self._access_buffer_region(mask_arg, (mask_extent,), context)
        source = self._access_buffer_region(source_arg, (count,), context)
        if destination is None or mask is None or source is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "mask": mask,
            "lhs": source,
            "select_mode": mode,
            "source_type": source_type,
        }
        if source_type == 1:
            metadata["scalar"] = scalar
        else:
            source1 = self._access_buffer_region(source1_arg, (count,), context)
            if source1 is None:
                return {}
            metadata["rhs"] = source1
        if scratch_arg is not None:
            scratch_extent = self._access_ptr_extent(scratch_arg, context)
            if scratch_extent is None:
                return {}
            scratch = self._access_buffer_region(
                scratch_arg, (scratch_extent,), context
            )
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _tail_compare_metadata(
        self,
        operation: str,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Internal contract emitted by AscendTailMaskPropagation:
        # dst, src0, src1/scalar, mode, validRow, validCol, physRow,
        # physCol, storageCol.
        if len(arguments) != 9:
            return {}
        mode = getattr(arguments[3], "value", None)
        if mode not in {"EQ", "NE", "GT", "GE", "LT", "LE"}:
            raise UnsupportedSimOpError(f"unsupported tail compare mode {mode!r}")
        dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[4:9]
        )
        if any(value is None for value in dimensions):
            return {}
        valid_rows, valid_cols, physical_rows, physical_cols, storage_cols = dimensions
        if any(isinstance(value, int) and value < 0 for value in dimensions):
            raise ProgramValidationError("tail compare dimensions must not be negative")
        if all(isinstance(value, int) for value in dimensions):
            if valid_rows > physical_rows or valid_cols > physical_cols:
                raise ProgramValidationError(
                    "tail compare valid rectangle must fit its physical tile"
                )
            if storage_cols < (physical_cols + 7) // 8:
                raise ProgramValidationError(
                    "tail compare packed storage width is too small"
                )
        packed_valid_cols: Any
        if isinstance(valid_cols, int):
            packed_valid_cols = (valid_cols + 7) // 8
        else:
            packed_valid_cols = SymbolicInt(
                "floordiv", (SymbolicInt("add", (valid_cols, 7)), 8)
            )
        destination = self._access_buffer_region(
            arguments[0], (valid_rows, packed_valid_cols), context
        )
        source = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "lhs": source,
            "compare_mode": mode,
            "valid_rows": valid_rows,
            "valid_cols": valid_cols,
            "physical_rows": physical_rows,
            "physical_cols": physical_cols,
            "storage_cols": storage_cols,
        }
        if operation == "tail_compare_scalar":
            scalar = self._literal(self.analyzer.simplify(arguments[2]))
            if not isinstance(scalar, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"tail compare scalar requires a literal, got {scalar!r}"
                )
            metadata["scalar"] = scalar
        else:
            right = self._access_buffer_region(
                arguments[2], (valid_rows, valid_cols), context
            )
            if right is None:
                return {}
            metadata["rhs"] = right
        return metadata

    def _tail_select_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Internal contract emitted by AscendTailMaskPropagation:
        # kind, dst, mask, src0, tmp, src1Type, src1, mode, validRow,
        # validCol, physRow, physCol, storageCol.
        if len(arguments) != 13:
            return {}
        kind = self._literal(arguments[0])
        source_type = self._const_int(arguments[5], context.environment)
        expected = {
            "Scalar": (1, "VSEL_TENSOR_SCALAR_MODE"),
            "Tensor": (2, "VSEL_TENSOR_TENSOR_MODE"),
        }.get(kind)
        mode = getattr(arguments[7], "value", None)
        if expected is None or (source_type, mode) != expected:
            raise UnsupportedSimOpError(
                f"unsupported tail select kind/type/mode: "
                f"{kind!r}/{source_type!r}/{mode!r}"
            )
        dimensions = tuple(
            self._runtime_int(argument, context.environment)
            for argument in arguments[8:13]
        )
        if any(value is None for value in dimensions):
            return {}
        valid_rows, valid_cols, physical_rows, physical_cols, storage_cols = dimensions
        if any(isinstance(value, int) and value < 0 for value in dimensions):
            raise ProgramValidationError("tail select dimensions must not be negative")
        if all(isinstance(value, int) for value in dimensions):
            if valid_rows > physical_rows or valid_cols > physical_cols:
                raise ProgramValidationError(
                    "tail select valid rectangle must fit its physical tile"
                )
            if storage_cols < (physical_cols + 7) // 8:
                raise ProgramValidationError(
                    "tail select packed storage width is too small"
                )
        if isinstance(valid_cols, int):
            packed_valid_cols: Any = (valid_cols + 7) // 8
        else:
            packed_valid_cols = SymbolicInt(
                "floordiv", (SymbolicInt("add", (valid_cols, 7)), 8)
            )
        destination = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        mask = self._access_buffer_region(
            arguments[2], (valid_rows, packed_valid_cols), context
        )
        source = self._access_buffer_region(
            arguments[3], (valid_rows, valid_cols), context
        )
        if destination is None or mask is None or source is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "mask": mask,
            "lhs": source,
            "select_kind": kind,
            "select_mode": mode,
            "source_type": source_type,
            "valid_rows": valid_rows,
            "valid_cols": valid_cols,
            "physical_rows": physical_rows,
            "physical_cols": physical_cols,
            "storage_cols": storage_cols,
        }
        if source_type == 1:
            scalar = self._literal(self.analyzer.simplify(arguments[6]))
            if not isinstance(scalar, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"tail scalar select requires a literal, got {scalar!r}"
                )
            metadata["scalar"] = scalar
        else:
            right = self._access_buffer_region(
                arguments[6], (valid_rows, valid_cols), context
            )
            if right is None:
                return {}
            metadata["rhs"] = right

        mask_data = self._access_ptr_data_name(arguments[2])
        scratch_data = self._access_ptr_data_name(arguments[4])
        if scratch_data is not None and scratch_data != mask_data:
            scratch_extent = self._access_ptr_extent(arguments[4], context)
            if scratch_extent is None:
                return {}
            scratch = self._access_buffer_region(
                arguments[4], (scratch_extent,), context
            )
            if scratch is None:
                return {}
            metadata["scratch"] = scratch
        return metadata

    def _access_ptr_data_name(self, pointer: Any) -> Optional[str]:
        if (
            isinstance(pointer, self.tir.Call)
            and str(pointer.op.name) == "tir.tvm_access_ptr"
            and len(pointer.args) >= 2
        ):
            return self._var_name(pointer.args[1])
        return None

    def _scalar_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
        *,
        tail: bool = False,
    ) -> Dict[str, Any]:
        expected_arguments = 6 if tail else 4
        if len(arguments) != expected_arguments:
            return {}
        shape = self._vector_shape(arguments, context, tail=tail, count_index=3)
        if shape is None:
            return {}
        destination = self._access_buffer_region(arguments[0], shape, context)
        source = self._access_buffer_region(arguments[1], shape, context)
        if destination is None or source is None:
            return {}
        metadata: Dict[str, Any] = {
            "dst": destination,
            "lhs": source,
            "scalar": self._literal(arguments[2]),
        }
        if tail:
            metadata["tail"] = self._tail_details(arguments, context, 3)
        return metadata

    def _cast_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) != 4:
            return {}
        shape = self._vector_shape(arguments, context, tail=False, count_index=3)
        if shape is None:
            return {}
        destination = self._access_buffer_region(arguments[0], shape, context)
        source = self._access_buffer_region(arguments[1], shape, context)
        round_mode = self._literal(arguments[2])
        if destination is None or source is None or not isinstance(round_mode, str):
            return {}
        return {
            "dst": destination,
            "src": source,
            "round_mode": round_mode.upper(),
        }

    def _fill_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        # Native tl.ascend_fill carries a backend template name before the
        # destination pointer.  Accept the equivalent call_extern form as well
        # so bridge tests and already-lowered external calls share one contract.
        argument_offset = 1 if len(arguments) == 4 else 0
        if len(arguments) - argument_offset != 3:
            return {}
        destination_arg, scalar_arg, count_arg = arguments[argument_offset:]
        count = self._runtime_int(count_arg, context.environment)
        scalar = self._literal(self.analyzer.simplify(scalar_arg))
        if count is None:
            return {}
        if isinstance(count, int) and count < 0:
            raise ProgramValidationError("fill count must not be negative")
        if not isinstance(scalar, (bool, int, float)):
            raise UnsupportedSimOpError(
                f"functional fill requires a literal scalar, got {scalar!r}"
            )
        destination = self._access_buffer_region(destination_arg, (count,), context)
        if destination is None:
            return {}
        return {"dst": destination, "scalar": scalar}

    def _tail_reduce_metadata(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
    ) -> Dict[str, Any]:
        if len(arguments) != 7:
            raise UnsupportedSimOpError(
                "functional tail_reduce currently requires the no-workspace 8-argument form"
            )
        dimension = self._const_int(arguments[2], context.environment)
        clear = self._const_int(arguments[6], context.environment)
        if dimension != 0 or clear != 1:
            raise UnsupportedSimOpError(
                "functional tail_reduce supports only dim=0 and clear=true"
            )
        valid_rows = self._runtime_int(arguments[3], context.environment)
        valid_cols = self._runtime_int(arguments[4], context.environment)
        physical_cols = self._runtime_int(arguments[5], context.environment)
        if valid_rows is None or valid_cols is None or physical_cols is None:
            raise UnsupportedSimOpError(
                "tail_reduce extents must be executable runtime integer expressions"
            )
        if all(isinstance(value, int) for value in (
            valid_rows, valid_cols, physical_cols
        )) and valid_cols > physical_cols:
            raise ProgramValidationError(
                "tail_reduce valid columns must not exceed physical columns"
            )
        source = self._access_buffer_region(
            arguments[1], (valid_rows, valid_cols), context
        )
        destination = self._access_buffer_region(arguments[0], (valid_cols,), context)
        if source is None or destination is None:
            raise UnsupportedSimOpError(
                "tail_reduce source and destination pointers must resolve to buffers"
            )
        if source.dtype != "float32" or destination.dtype != "float32":
            raise UnsupportedSimOpError(
                "functional tail_reduce currently supports only float32"
            )
        if (
            isinstance(physical_cols, int)
            and source.strides_bytes is not None
            and isinstance(source.strides_bytes[0], int)
            and source.strides_bytes[0] != physical_cols * dtype_size_bytes(source.dtype)
        ):
            raise ProgramValidationError(
                "tail_reduce physical columns disagree with source row stride"
            )
        return {
            "src": source,
            "dst": destination,
            "reduce": {
                "dimension": dimension,
                "clear": True,
                "valid_rows": valid_rows,
                "valid_cols": valid_cols,
                "physical_cols": physical_cols,
            },
        }

    def _vector_shape(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
        *,
        tail: bool,
        count_index: int,
    ) -> Optional[Tuple[Any, ...]]:
        indices = (
            (count_index, count_index + 1, count_index + 2)
            if tail
            else (count_index,)
        )
        values = tuple(
            self._runtime_int(arguments[index], context.environment)
            for index in indices
        )
        if any(value is None for value in values):
            return None
        if any(isinstance(value, int) and value < 0 for value in values):
            raise ProgramValidationError("vector extents must not be negative")
        if tail and all(isinstance(value, int) for value in values):
            _, valid_cols, physical_cols = values
            if valid_cols > physical_cols:
                raise ProgramValidationError(
                    "tail valid columns must not exceed physical columns"
                )
        return values[:2] if tail else values

    def _tail_details(
        self,
        arguments: Tuple[Any, ...],
        context: _Context,
        count_index: int,
    ) -> Dict[str, Any]:
        return {
            "valid_rows": self._runtime_int(
                arguments[count_index], context.environment
            ),
            "valid_cols": self._runtime_int(
                arguments[count_index + 1], context.environment
            ),
            "physical_cols": self._runtime_int(
                arguments[count_index + 2], context.environment
            ),
        }

    def _access_buffer_region(
        self,
        pointer: Any,
        shape: Tuple[Any, ...],
        context: _Context,
    ) -> Optional[BufferRegion]:
        data_var = pointer
        element_offset = 0
        if isinstance(pointer, self.tir.Call) and str(pointer.op.name) == "tir.tvm_access_ptr":
            if len(pointer.args) < 4:
                return None
            data_var = pointer.args[1]
            offset = self._runtime_int(pointer.args[2], context.environment)
            if offset is None:
                return None
            element_offset = offset
        name = self.buffer_name_by_data_var.get(self._var_name(data_var))
        if name is None:
            return None
        spec = self.buffers[name]
        itemsize = dtype_size_bytes(spec.dtype)
        strides = None
        if len(shape) == 2:
            physical_cols = spec.shape[-1]
            if not isinstance(physical_cols, (int, AffineInt, SymbolicInt)):
                return None
            row_stride = (
                physical_cols * itemsize
                if isinstance(physical_cols, int)
                else physical_cols.scaled(itemsize)
            )
            strides = (row_stride, itemsize)
        byte_offset = (
            element_offset * itemsize
            if isinstance(element_offset, int)
            else element_offset.scaled(itemsize)
        )
        return BufferRegion(
            name,
            spec.scope,
            shape,
            spec.dtype,
            byte_offset=byte_offset,
            strides_bytes=strides,
            core_id=(
                None
                if spec.scope in {MemoryScope.GM, MemoryScope.WORKSPACE}
                else context.core_id
            ),
        )

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
        dependencies = self._memory_dependencies(metadata, context.core_id)
        task = Task(
            task_id,
            normalized,
            context.core_id,
            lane,
            pipe,
            self.timing_profile.estimate_cycles(normalized),
            dependencies=dependencies,
            metadata=metadata,
        )
        self.tasks[context.core_id].append(task)
        self._record_memory_accesses(task, context.core_id)

    def _memory_dependencies(
        self,
        metadata: Mapping[str, Any],
        core_id: int,
    ) -> Tuple[str, ...]:
        reads = self._operand_regions(
            metadata, ("src", "lhs", "rhs", "mask", "accumulator")
        )
        writes = self._operand_regions(
            metadata, ("dst", "pad_dst", "scratch", "output_scratch")
        )
        dependencies = {
            task_id
            for region in reads
            for previous, task_id in self.last_writes
            if self._regions_overlap(region, previous, core_id)
        }
        for region in writes:
            dependencies.update(
                task_id
                for previous, task_id in self.last_writes + self.last_reads
                if self._regions_overlap(region, previous, core_id)
            )
        return tuple(sorted(dependencies))

    def _record_memory_accesses(self, task: Task, core_id: int) -> None:
        reads = self._operand_regions(
            task.metadata, ("src", "lhs", "rhs", "mask", "accumulator")
        )
        writes = self._operand_regions(
            task.metadata, ("dst", "pad_dst", "scratch", "output_scratch")
        )
        for region in writes:
            self.last_writes = [
                entry for entry in self.last_writes
                if not self._regions_overlap(region, entry[0], core_id)
            ]
            self.last_reads = [
                entry for entry in self.last_reads
                if not self._regions_overlap(region, entry[0], core_id)
            ]
            self.last_writes.append((region, task.task_id))
        self.last_reads.extend((region, task.task_id) for region in reads)

    @staticmethod
    def _operand_regions(
        metadata: Mapping[str, Any], names: Tuple[str, ...]
    ) -> Tuple[BufferRegion, ...]:
        return tuple(
            value for name in names
            if isinstance((value := metadata.get(name)), BufferRegion)
        )

    def _regions_overlap(
        self, left: BufferRegion, right: BufferRegion, core_id: int
    ) -> bool:
        if left.scope != right.scope:
            return False
        left_owner = left.core_id if left.core_id is not None else core_id
        right_owner = right.core_id if right.core_id is not None else core_id
        if left.scope not in {MemoryScope.GM, MemoryScope.WORKSPACE}:
            if left_owner != right_owner:
                return False
        left_spec = self.buffers[left.buffer]
        right_spec = self.buffers[right.buffer]
        if left.buffer == right.buffer:
            left_base = right_base = 0
        else:
            left_base = left_spec.address
            right_base = right_spec.address
            if left_base is None or right_base is None:
                return False
        left_bounds = _region_bounds(left)
        right_bounds = _region_bounds(right)
        if left_bounds is None or right_bounds is None:
            left_size = _buffer_size_bytes(left_spec)
            right_size = _buffer_size_bytes(right_spec)
            if left_size is None or right_size is None:
                return True
            return (
                left_base < right_base + right_size
                and right_base < left_base + left_size
            )
        left_start, left_end = (
            left_base + left_bounds[0], left_base + left_bounds[1]
        )
        right_start, right_end = (
            right_base + right_bounds[0], right_base + right_bounds[1]
        )
        return left_start < right_end and right_start < left_end

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

    def _affine_int(
        self, value: Any, environment: Mapping[Any, int]
    ) -> Optional[Any]:
        constant = self._const_int(value, environment)
        if constant is not None:
            return constant
        substituted = value
        if environment:
            replacements = {
                var: self.tir.IntImm(getattr(var, "dtype", "int32"), number)
                for var, number in environment.items()
            }
            substituted = self.tir.stmt_functor.substitute(value, replacements)
        simplified = self.analyzer.simplify(substituted)
        if isinstance(simplified, self.tir.Var):
            return AffineInt.variable(self._var_name(simplified))
        if isinstance(simplified, (self.tir.Add, self.tir.Sub)):
            left = self._affine_int(simplified.a, {})
            right = self._affine_int(simplified.b, {})
            if left is None or right is None:
                return None
            left_expr = left if isinstance(left, AffineInt) else AffineInt((), left)
            right_expr = right if isinstance(right, AffineInt) else AffineInt((), right)
            if isinstance(simplified, self.tir.Sub):
                right_expr = right_expr.scaled(-1)
            result = left_expr.plus(right_expr)
            return result.constant if not result.terms else result
        if isinstance(simplified, self.tir.Mul):
            left = self._affine_int(simplified.a, {})
            right = self._affine_int(simplified.b, {})
            if isinstance(left, int) and isinstance(right, AffineInt):
                return right.scaled(left)
            if isinstance(right, int) and isinstance(left, AffineInt):
                return left.scaled(right)
            if isinstance(left, int) and isinstance(right, int):
                return left * right
        return None

    def _extent_or_symbol(self, value: Any, environment: Mapping[Any, int]) -> Any:
        expression = self._runtime_int(value, environment)
        if expression is None:
            raise UnsupportedSimOpError(
                f"unsupported dynamic integer expression in buffer extent: {value}"
            )
        return expression

    def _runtime_int(
        self, value: Any, environment: Mapping[Any, int]
    ) -> Optional[Any]:
        affine = self._affine_int(value, environment)
        if affine is not None:
            return affine
        substituted = value
        if environment:
            replacements = {
                var: self.tir.IntImm(getattr(var, "dtype", "int32"), number)
                for var, number in environment.items()
            }
            substituted = self.tir.stmt_functor.substitute(value, replacements)
        simplified = self.analyzer.simplify(substituted)
        operation = type(simplified).__name__
        binary_operations = {
            "Add": "add",
            "And": "and",
            "Div": "truncdiv",
            "EQ": "eq",
            "FloorDiv": "floordiv",
            "FloorMod": "floormod",
            "GE": "ge",
            "GT": "gt",
            "LE": "le",
            "LT": "lt",
            "Max": "max",
            "Min": "min",
            "Mul": "mul",
            "NE": "ne",
            "Mod": "truncmod",
            "Or": "or",
            "Sub": "sub",
        }
        if operation in binary_operations:
            left = self._runtime_int(simplified.a, {})
            right = self._runtime_int(simplified.b, {})
            if left is None or right is None:
                return None
            return SymbolicInt(binary_operations[operation], (left, right))
        if operation == "Not":
            operand = self._runtime_int(simplified.a, {})
            return None if operand is None else SymbolicInt("not", (operand,))
        if operation == "Cast":
            dtype = str(simplified.dtype)
            if not (dtype.startswith("int") or dtype.startswith("uint")):
                return None
            return self._runtime_int(simplified.value, {})
        if operation == "Select":
            arguments = (
                self._runtime_int(simplified.condition, {}),
                self._runtime_int(simplified.true_value, {}),
                self._runtime_int(simplified.false_value, {}),
            )
            if any(argument is None for argument in arguments):
                return None
            return SymbolicInt("select", arguments)
        return None

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


def _region_bounds(region: BufferRegion) -> Optional[Tuple[int, int]]:
    values = (region.byte_offset,) + region.shape + (region.strides_bytes or ())
    if any(isinstance(value, (AffineInt, SymbolicInt)) for value in values):
        return None
    if any(extent == 0 for extent in region.shape):
        return region.byte_offset, region.byte_offset
    itemsize = dtype_size_bytes(region.dtype)
    strides = region.strides_bytes or contiguous_strides_bytes(region.shape, itemsize)
    last_offset = sum(
        (extent - 1) * stride for extent, stride in zip(region.shape, strides)
    )
    return region.byte_offset, region.byte_offset + last_offset + itemsize


def _buffer_size_bytes(spec: BufferSpec) -> Optional[int]:
    if spec.size_bytes is not None:
        return spec.size_bytes
    if any(not isinstance(extent, int) for extent in spec.shape):
        return None
    size = dtype_size_bytes(spec.dtype)
    for extent in spec.shape:
        size *= extent
    return size


def _ascend_template_dtype(token: str) -> Optional[str]:
    return {
        "half": "float16",
        "float": "float32",
        "bfloat16_t": "bfloat16",
        "int8_t": "int8",
        "int16_t": "int16",
        "int": "int32",
        "int64_t": "int64",
        "uint8_t": "uint8",
        "uint16_t": "uint16",
        "uint32_t": "uint32",
        "uint64_t": "uint64",
    }.get(token.strip().lower())
