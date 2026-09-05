# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Functional execution of concrete simulator tasks on CPU memory."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np

from .config import SimulatorConfig
from .errors import ProgramValidationError, UnsupportedSimOpError
from .layout import pack_matrix, unpack_matrix
from .memory import MemoryRuntime, MemoryView
from .program import (
    AffineInt,
    BufferRegion,
    KernelProgram,
    MemoryScope,
    SymbolicInt,
    Task,
)
from .scheduler import DiscreteEventScheduler, ScheduleResult
from .sync import FlagBarrierSynchronizationModel

# Row-wise broadcast binary experiment ops: dst row i = src0 row i op scalar_i.
_ROW_EXPAND_OPERATIONS = {
    "row_expand_mul_experiment": np.multiply,
    "row_expand_sub_experiment": np.subtract,
    "row_expand_div_experiment": np.divide,
}

# InitSortBuf writes the fp32 -inf bit pattern into even int32 lanes and -1
# into odd lanes of the sort workspace's reinterpreted int32 view
# (src/tl_templates/ascend/common.h).
_INIT_SORT_BUF_VALUE_INT32 = np.int32(-8388608)  # 0xFF800000


_BINARY_OPERATIONS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
    "sub_experiment": np.subtract,
}

_SCALAR_OPERATIONS = {
    "adds": np.add,
    "subs": np.subtract,
    "muls": np.multiply,
    "divs": np.divide,
    "maxs": np.maximum,
    "mins": np.minimum,
    "leaky_relu": lambda value, slope: np.where(value >= 0, value, value * slope),
    "mins_experiment": np.minimum,
}

_UNARY_OPERATIONS = {
    "abs": np.abs,
    "abs_experiment": np.abs,
    "exp": np.exp,
    "exp_experiment": np.exp,
    "ln": np.log,
    "reciprocal": lambda value: np.divide(1, value),
    "relu": lambda value: np.maximum(value, 0),
    "rsqrt": lambda value: np.divide(1, np.sqrt(value)),
    "sqrt": np.sqrt,
    "sigmoid": lambda value: np.exp(-np.logaddexp(0, -value)),
    "silu": lambda value: value * np.exp(-np.logaddexp(0, -value)),
    "sin": np.sin,
    "cos": np.cos,
    "round": lambda value: np.copysign(np.floor(np.abs(value) + 0.5), value),
}

_REDUCE_OPERATIONS = {
    "reduce_max": lambda value: np.max(value, axis=0),
    "reduce_min": lambda value: np.min(value, axis=0),
    "reduce_sum": lambda value: np.sum(value, axis=0),
}

_COMPARE_OPERATIONS = {
    "EQ": np.equal,
    "NE": np.not_equal,
    "GT": np.greater,
    "GE": np.greater_equal,
    "LT": np.less,
    "LE": np.less_equal,
}

_BITWISE_BINARY_OPERATIONS = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
}


@dataclass(frozen=True)
class FunctionalExecutionResult:
    """Schedule and final memory state produced by functional execution."""

    schedule: ScheduleResult
    memory: MemoryRuntime


class FunctionalSimulator:
    """Execute explicit copy and vector tasks using byte-accurate simulator memory."""

    def __init__(
        self,
        program: KernelProgram,
        config: Optional[SimulatorConfig] = None,
        bindings: Optional[Mapping[str, int | float]] = None,
    ) -> None:
        self.program = program
        self.config = config or SimulatorConfig(platform=program.platform)
        if self.config.platform != program.platform:
            raise ProgramValidationError(
                "functional simulator config platform does not match the program"
            )
        self.bindings = dict(bindings or {})
        self._validate_runtime_contracts()
        self.memory = MemoryRuntime.from_program(
            program,
            hazard_check=self.config.hazard_check,
            bindings=self.bindings,
        )
        self._active_lane = None

    def _validate_runtime_contracts(self) -> None:
        for task in self.program.tasks:
            details = task.metadata.get("mma")
            if isinstance(details, Mapping):
                actual_cols = _resolve_int(details["n_actual"], self.bindings)
                template_cols = _resolve_int(details["cols"], self.bindings)
                if actual_cols <= 0 or actual_cols > template_cols:
                    raise ProgramValidationError(
                        f"mma n_actual must be in [1, {template_cols}], got {actual_cols}"
                    )
                if actual_cols % 16:
                    raise ProgramValidationError(
                        "mma n_actual must be a multiple of 16, "
                        f"got {actual_cols}"
                    )
            topk = task.metadata.get("topk")
            if isinstance(topk, Mapping):
                actual_num = _resolve_int(topk["actual_num"], self.bindings)
                k = _resolve_int(topk["k"], self.bindings)
                max_actual_num = _resolve_int(
                    topk["max_actual_num"], self.bindings
                )
                if actual_num < k or actual_num > max_actual_num:
                    raise ProgramValidationError(
                        f"topk actual_num must be in [{k}, {max_actual_num}], "
                        f"got {actual_num}"
                    )
            sort = task.metadata.get("sort")
            if isinstance(sort, Mapping):
                actual_num = _resolve_int(sort["actual_num"], self.bindings)
                repeat_times = _resolve_int(sort["repeat_times"], self.bindings)
                capacity = _resolve_int(sort["source_capacity"], self.bindings)
                if actual_num <= 0 or actual_num > capacity:
                    raise ProgramValidationError(
                        f"sort actual_num must be in [1, {capacity}], got {actual_num}"
                    )
                expected_repeats = (actual_num + 31) // 32
                if repeat_times != expected_repeats:
                    raise ProgramValidationError(
                        "sort repeatTimes must equal ceil(actual_num / 32), "
                        f"got {repeat_times}"
                    )

    def write(self, region: BufferRegion, values: Any, *, task_core_id: int = 0) -> None:
        """Initialize a concrete region from an array-like CPU value."""
        view = self._resolve(region, task_core_id)
        array = np.asarray(values, dtype=_numpy_dtype(region.dtype))
        if array.shape != view.shape:
            raise ProgramValidationError(
                f"input for {region.buffer!r} has shape {array.shape}, expected {view.shape}"
            )
        view.allocation.write(view, np.ascontiguousarray(array).tobytes(order="C"))

    def read(self, region: BufferRegion, *, task_core_id: int = 0) -> np.ndarray:
        """Read a concrete region into an independent NumPy array."""
        view = self._resolve(region, task_core_id)
        payload = view.allocation.read(view)
        return np.frombuffer(payload, dtype=_numpy_dtype(region.dtype)).reshape(
            view.shape
        ).copy()

    def run(self) -> FunctionalExecutionResult:
        """Schedule the program, then apply operations in deterministic event order."""
        schedule = DiscreteEventScheduler(
            self.config,
            synchronization=FlagBarrierSynchronizationModel(),
        ).run(self.program)
        task_by_id = {task.task_id: task for task in self.program.tasks}
        for record in schedule.records:
            if record.category == "wait":
                continue
            task = task_by_id[record.task_id]
            self._active_lane = task.lane
            try:
                self._execute(task)
            finally:
                self._active_lane = None
        return FunctionalExecutionResult(schedule=schedule, memory=self.memory)

    def _execute(self, task: Task) -> None:
        if task.metadata.get("trace_only") is True:
            return
        operation = task.operation.lower()
        if "copy" in operation or "datacopy" in operation or "data_copy" in operation:
            self._copy(task)
            return
        if operation in _BINARY_OPERATIONS:
            self._binary(task, operation)
            return
        if operation in _BITWISE_BINARY_OPERATIONS:
            self._bitwise_binary(task, operation)
            return
        if operation in _SCALAR_OPERATIONS:
            self._binary(task, operation)
            return
        if operation == "axpy":
            self._axpy(task)
            return
        if operation == "mul_add_dst":
            self._mul_add_dst(task)
            return
        if operation in {"mma", "mma_bias"}:
            self._mma(task)
            return
        if operation == "gemm_v0":
            self._gemm_v0(task)
            return
        if operation == "im2col":
            self._im2col(task)
            return
        if operation in _UNARY_OPERATIONS:
            self._unary(task, operation)
            return
        if operation == "bitwise_not":
            self._bitwise_not(task)
            return
        if operation in {"bitwise_lshift", "bitwise_rshift"}:
            self._bitwise_shift(task, operation)
            return
        if operation == "cast":
            self._cast(task)
            return
        if operation == "fill":
            self._fill(task)
            return
        if operation == "reducesum_experiment":
            self._reduce_sum_experiment(task)
            return
        if operation == "sum_experiment":
            self._sum_experiment(task)
            return
        if operation == "brcb_experiment":
            self._brcb(task)
            return
        if operation in {
            "row_expand_mul_experiment",
            "row_expand_sub_experiment",
            "row_expand_div_experiment",
        }:
            self._row_expand_binop(task)
            return
        if operation in {"arith_progression", "createvecindex"}:
            self._sequence(task)
            return
        if operation == "gather":
            self._gather(task)
            return
        if operation == "gatherb":
            self._gatherb(task)
            return
        if operation == "gather_mask":
            self._gather_mask(task)
            return
        if operation == "transpose":
            self._transpose(task)
            return
        if operation == "reinterpretcast":
            self._reinterpretcast(task)
            return
        if operation == "topk":
            self._topk(task)
            return
        if operation == "sort32":
            self._sort32(task)
            return
        if operation == "sort":
            self._sort(task)
            return
        if operation == "init_sort_buf":
            self._init_sort_buf(task)
            return
        if operation == "merge_sort":
            self._merge_sort(task)
            return
        if operation == "atomic_add_ub_to_gm":
            self._atomic_add(task)
            return
        if operation == "atomic_add_l0c_to_gm":
            self._atomic_add_l0c(task)
            return
        if operation in {"clamp", "clamp_max", "clamp_min"}:
            self._clamp(task, operation)
            return
        if operation in {"broadcast", "tail_broadcast"}:
            self._broadcast(task)
            return
        if operation in {"compare", "compare_scalar"}:
            self._compare(task)
            return
        if operation in {"tail_compare", "tail_compare_scalar"}:
            self._tail_compare(task)
            return
        if operation == "select":
            self._select(task)
            return
        if operation == "tail_select":
            self._tail_select(task)
            return
        if operation == "reduce":
            kind = task.metadata.get("reduce_kind")
            if kind not in _REDUCE_OPERATIONS:
                raise UnsupportedSimOpError(f"unsupported reduce kind {kind!r}")
            self._reduce(task, kind)
            return
        if operation in {"block_reduce_max", "block_reduce_min", "block_reduce_sum"}:
            self._block_reduce(task)
            return
        if operation in {"wholereducemax", "wholereducemin", "wholereducesum"}:
            self._whole_reduce(task)
            return
        if operation in _REDUCE_OPERATIONS:
            self._reduce(task, operation)
            return
        if operation in {"set_flag", "wait_flag", "auto_set_flag", "auto_wait_flag",
                         "set_cross_flag", "wait_cross_flag", "auto_set_cross_flag",
                         "auto_wait_cross_flag", "barrier_all", "pipe_barrier",
                         "auto_barrier"}:
            return
        raise UnsupportedSimOpError(
            f"functional execution is not implemented for {task.operation!r} "
            f"(task {task.task_id!r})"
        )

    def _copy(self, task: Task) -> None:
        source = _operand(task, "src")
        pad_destination = task.metadata.get("pad_dst")
        copy_details = task.metadata.get("copy", {})
        written_regions = task.metadata.get("dst_regions")
        if isinstance(written_regions, (tuple, list)):
            destinations = list(written_regions)
        else:
            destinations = [_operand(task, "dst")]
        destination = destinations[0]
        if isinstance(pad_destination, BufferRegion):
            pad_value = copy_details.get("pad_value")
            if not isinstance(pad_value, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"copy task {task.task_id!r} has non-literal pad value {pad_value!r}"
                )
            pad_shape = tuple(
                _resolve_int(value, self.bindings) for value in pad_destination.shape
            )
            self.write(
                pad_destination,
                np.full(pad_shape, pad_value, dtype=_numpy_dtype(pad_destination.dtype)),
                task_core_id=task.core_id,
            )
        source_regions = task.metadata.get("src_regions")
        if isinstance(source_regions, (tuple, list)):
            fragments = [
                self.read(region, task_core_id=task.core_id)
                for region in source_regions
            ]
            values = np.concatenate(
                fragments, axis=copy_details["source_region_axis"]
            )
        else:
            values = self.read(source, task_core_id=task.core_id)
        if copy_details.get("layout_transform") is True:
            source_shape = tuple(copy_details["source_shape"])
            destination_shape = tuple(copy_details["destination_shape"])
            if (
                copy_details.get("source_window_direct") is True
                or isinstance(source_regions, (tuple, list))
            ):
                tile = values
            else:
                logical = unpack_matrix(
                    values, copy_details["source_layout"], source_shape
                )
                source_row, source_col = copy_details.get("source_origin", (0, 0))
                window_shape = tuple(
                    copy_details.get("source_window_shape", destination_shape)
                )
                tile = logical[
                    source_row:source_row + window_shape[0],
                    source_col:source_col + window_shape[1],
                ]
            if copy_details.get("transpose_after_slice") is True:
                tile = tile.T
            if copy_details.get("relu") is True:
                tile = np.maximum(tile, 0)
            values = pack_matrix(
                tile,
                copy_details["destination_layout"],
            )
            if len(destination.shape) > 1:
                values = values.reshape(destination_shape)
        elif copy_details.get("layout") == "ub_to_ub":
            cast_mode = copy_details.get("cast_mode")
            destination_dtype = _numpy_dtype(destination.dtype)
            if cast_mode == "CAST_RINT" and np.issubdtype(
                destination_dtype, np.integer
            ):
                # Rounding to nearest-even applies to integer destinations;
                # float destinations convert with IEEE nearest-even in
                # np.asarray below.
                values = np.rint(values.astype(np.float64)).astype(values.dtype)
            values = np.asarray(values, dtype=destination_dtype)
        elif copy_details.get("layout") == "zN":
            physical_shape = (
                _resolve_int(copy_details["physical_rows"], self.bindings),
                _resolve_int(copy_details["physical_cols"], self.bindings),
            )
            written_rows = copy_details.get("written_rows")
            if written_rows is not None:
                # Spliced sub-tile copy: only the fractal rows touched by the
                # valid rectangle are written, one packed segment per
                # fractal-column band, so bands written by earlier DMAs of the
                # same zN view stay intact.
                written_rows = _resolve_int(written_rows, self.bindings)
                logical = np.zeros(
                    (written_rows, physical_shape[1]),
                    dtype=_numpy_dtype(destination.dtype),
                )
                logical[:values.shape[0], :values.shape[1]] = values
                packed = pack_matrix(logical, "zN")
                chunks = packed.reshape(
                    len(destinations), packed.size // len(destinations)
                )
                for region, chunk in zip(destinations, chunks):
                    self.write(
                        region,
                        np.ascontiguousarray(chunk),
                        task_core_id=task.core_id,
                    )
                return
            logical = np.zeros(
                physical_shape, dtype=_numpy_dtype(destination.dtype)
            )
            logical[:values.shape[0], :values.shape[1]] = values
            values = pack_matrix(logical, "zN")
        self.write(destination, values, task_core_id=task.core_id)

    def _binary(self, task: Task, operation: str) -> None:
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        left_values = self.read(left, task_core_id=task.core_id)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif isinstance(task.metadata.get("scalar_src"), BufferRegion):
            scalar_source = task.metadata["scalar_src"]
            right_values = self.read(
                scalar_source, task_core_id=task.core_id
            ).reshape(-1)[0]
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
            if isinstance(right_values, str) and right_values in self.bindings:
                right_values = self.bindings[right_values]
        else:
            raise ProgramValidationError(
                f"task {task.task_id!r} requires a BufferRegion 'rhs' or scalar"
            )
        implementation = (
            _BINARY_OPERATIONS[operation]
            if operation in _BINARY_OPERATIONS
            else _SCALAR_OPERATIONS[operation]
        )
        result = implementation(left_values, right_values)
        self.write(destination, result, task_core_id=task.core_id)

    def _unary(self, task: Task, operation: str) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        result = _UNARY_OPERATIONS[operation](values)
        self.write(destination, result, task_core_id=task.core_id)

    def _axpy(self, task: Task) -> None:
        source = _operand(task, "lhs")
        destination = _operand(task, "dst")
        accumulator = _operand(task, "accumulator")
        scalar = task.metadata.get("scalar")
        if not isinstance(scalar, (bool, int, float)):
            raise UnsupportedSimOpError(
                f"functional axpy requires a literal scalar, got {scalar!r}"
            )
        source_values = self.read(source, task_core_id=task.core_id)
        accumulator_values = self.read(accumulator, task_core_id=task.core_id)
        self.write(
            destination,
            scalar * source_values + accumulator_values,
            task_core_id=task.core_id,
        )

    def _mul_add_dst(self, task: Task) -> None:
        left = _operand(task, "lhs")
        right = _operand(task, "rhs")
        destination = _operand(task, "dst")
        accumulator = _operand(task, "accumulator")
        result = (
            self.read(left, task_core_id=task.core_id)
            * self.read(right, task_core_id=task.core_id)
            + self.read(accumulator, task_core_id=task.core_id)
        )
        self.write(destination, result, task_core_id=task.core_id)

    def _mma(self, task: Task) -> None:
        left = _operand(task, "lhs")
        right = _operand(task, "rhs")
        destination = _operand(task, "dst")
        details = task.metadata.get("mma", {})
        actual_cols = _resolve_int(details["n_actual"], self.bindings)
        shape_a = (details["rows"], details["inner"])
        shape_b = (details["inner"], actual_cols)
        shape_c = (details["rows"], actual_cols)
        a_values = unpack_matrix(
            self.read(left, task_core_id=task.core_id), "l0a", shape_a
        )
        b_values = unpack_matrix(
            self.read(right, task_core_id=task.core_id), "l0b", shape_b
        )
        compute_dtype = _numpy_dtype(destination.dtype)
        result = np.matmul(
            a_values.astype(compute_dtype), b_values.astype(compute_dtype)
        )
        bias = task.metadata.get("bias")
        if isinstance(bias, BufferRegion):
            bias_values = self.read(bias, task_core_id=task.core_id)
            result = result + bias_values.reshape(1, actual_cols)
        elif not details["init"]:
            accumulator = _operand(task, "accumulator")
            previous = unpack_matrix(
                self.read(accumulator, task_core_id=task.core_id), "l0c", shape_c
            )
            result = previous + result
        self.write(
            destination,
            pack_matrix(result.astype(compute_dtype), "l0c"),
            task_core_id=task.core_id,
        )

    def _gemm_v0(self, task: Task) -> None:
        left = _operand(task, "lhs")
        right = _operand(task, "rhs")
        destination = _operand(task, "dst")
        details = task.metadata.get("gemm", {})
        a_values = unpack_matrix(
            self.read(left, task_core_id=task.core_id),
            "zn",
            tuple(details["shape_a"]),
        )
        b_values = unpack_matrix(
            self.read(right, task_core_id=task.core_id),
            "zn",
            tuple(details["shape_b"]),
        )
        if details["transpose_a"]:
            a_values = a_values.T
        if details["transpose_b"]:
            b_values = b_values.T
        compute_dtype = _numpy_dtype(destination.dtype)
        result = np.matmul(
            a_values.astype(compute_dtype), b_values.astype(compute_dtype)
        )
        if not details["init"]:
            accumulator = _operand(task, "accumulator")
            previous = unpack_matrix(
                self.read(accumulator, task_core_id=task.core_id),
                "l0c",
                (details["rows"], details["cols"]),
            )
            result = previous + result
        self.write(
            destination,
            pack_matrix(result.astype(compute_dtype), "l0c"),
            task_core_id=task.core_id,
        )

    def _im2col(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        details = task.metadata.get("im2col")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"im2col task {task.task_id!r} requires im2col metadata"
            )
        hi, wi = details["image_shape"]
        kh, kw = details["kernel"]
        stride_h, stride_w = details["stride"]
        dilation_h, dilation_w = details["dilation"]
        pad_left, _pad_right, pad_top, _pad_bottom = details["padding"]
        _output_h, output_w = details["output_shape"]
        channels = details["channels"]
        valid_m = details["valid_m"]
        valid_k = details["valid_k"]

        feature = unpack_matrix(
            self.read(source, task_core_id=task.core_id),
            "zn",
            (hi * wi, channels),
        )
        tile = np.zeros(
            (valid_m, valid_k), dtype=_numpy_dtype(destination.dtype)
        )
        for m in range(valid_m):
            output_row, output_col = divmod(m, output_w)
            for k in range(valid_k):
                filter_point, channel = divmod(k, channels)
                filter_row, filter_col = divmod(filter_point, kw)
                image_row = (
                    output_row * stride_h - pad_top
                    + filter_row * dilation_h
                )
                image_col = (
                    output_col * stride_w - pad_left
                    + filter_col * dilation_w
                )
                if 0 <= image_row < hi and 0 <= image_col < wi:
                    tile[m, k] = feature[image_row * wi + image_col, channel]
        self.write(
            destination,
            pack_matrix(tile, "l0a"),
            task_core_id=task.core_id,
        )

    def _cast(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        round_mode = task.metadata.get("round_mode")
        if round_mode == "CAST_NONE":
            result = values
        elif round_mode == "CAST_RINT":
            result = np.rint(values)
        elif round_mode == "CAST_FLOOR":
            result = np.floor(values)
        elif round_mode == "CAST_CEIL":
            result = np.ceil(values)
        elif round_mode == "CAST_ROUND":
            result = np.copysign(np.floor(np.abs(values) + 0.5), values)
        elif round_mode == "CAST_TRUNC":
            result = np.trunc(values)
        else:
            raise UnsupportedSimOpError(
                f"functional cast does not support round mode {round_mode!r}"
            )
        self.write(destination, result, task_core_id=task.core_id)

    def _fill(self, task: Task) -> None:
        destination = _operand(task, "dst")
        scalar = task.metadata.get("scalar")
        if not isinstance(scalar, (bool, int, float)):
            raise UnsupportedSimOpError(
                f"functional fill requires a literal scalar, got {scalar!r}"
            )
        shape = tuple(_resolve_int(value, self.bindings) for value in destination.shape)
        result = np.full(shape, scalar, dtype=_numpy_dtype(destination.dtype))
        self.write(destination, result, task_core_id=task.core_id)

    def _brcb(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        details = task.metadata.get("brcb")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"brcb task {task.task_id!r} requires brcb metadata"
            )
        repeat = _resolve_int(details["repeat"], self.bindings)
        blk_stride = _resolve_int(details["blk_stride"], self.bindings)
        rep_stride = _resolve_int(details["rep_stride"], self.bindings)
        if repeat < 0 or blk_stride < 0 or rep_stride < 0:
            raise ProgramValidationError(
                "brcb repeat and strides must not be negative"
            )
        if repeat == 0:
            return
        values = self.read(source, task_core_id=task.core_id).reshape(-1)
        if values.size < repeat * 8:
            raise ProgramValidationError(
                f"brcb source provides {values.size} elements, requires "
                f"{repeat * 8}"
            )
        itemsize = _numpy_dtype(destination.dtype).itemsize
        block_elements = 32 // itemsize
        block_bytes = block_elements * itemsize
        base_offset = destination.byte_offset
        for r in range(repeat):
            for b in range(8):
                scalar = values[r * 8 + b]
                offset = base_offset + (r * rep_stride + b * blk_stride) * block_bytes
                self.write(
                    replace(
                        destination,
                        shape=(block_elements,),
                        byte_offset=offset,
                    ),
                    np.full(
                        block_elements,
                        scalar,
                        dtype=_numpy_dtype(destination.dtype),
                    ),
                    task_core_id=task.core_id,
                )

    def _row_expand_binop(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        scalar_source = _operand(task, "scalar_src")
        details = task.metadata.get("row_expand")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"row-expand task {task.task_id!r} requires row_expand metadata"
            )
        rows = _resolve_int(details["rows"], self.bindings)
        row_elements = _resolve_int(details["row_elements"], self.bindings)
        elements_per_block = row_elements // 8
        scalars = self.read(scalar_source, task_core_id=task.core_id).reshape(-1)
        scratch = task.metadata.get("scratch")
        if isinstance(scratch, BufferRegion):
            # With tmp the codegen expands brcb(src1 -> tmp) before the masked
            # op; reproduce that broadcast so the scratch end state matches.
            if scalars.size != rows:
                raise ProgramValidationError(
                    f"row-expand scalar source provides {scalars.size} elements, "
                    f"requires {rows}"
                )
            self.write(
                scratch,
                np.repeat(scalars, elements_per_block).astype(
                    _numpy_dtype(scratch.dtype)
                ),
                task_core_id=task.core_id,
            )
        else:
            if scalars.size != rows * elements_per_block:
                raise ProgramValidationError(
                    f"row-expand scalar source provides {scalars.size} elements, "
                    f"requires {rows * elements_per_block} (one packed block per row)"
                )
            scalars = scalars[::elements_per_block]
        values = self.read(source, task_core_id=task.core_id).reshape(
            rows, row_elements
        )
        implementation = _ROW_EXPAND_OPERATIONS[task.operation]
        result = implementation(
            values, scalars.astype(values.dtype)[:, None]
        )
        self.write(destination, result, task_core_id=task.core_id)

    def _bitwise_binary(self, task: Task, operation: str) -> None:
        destination = _operand(task, "dst")
        left = _operand(task, "lhs")
        right = _operand(task, "rhs")
        result = _BITWISE_BINARY_OPERATIONS[operation](
            self.read(left, task_core_id=task.core_id),
            self.read(right, task_core_id=task.core_id),
        )
        self.write(destination, result, task_core_id=task.core_id)

    def _bitwise_not(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        self.write(
            destination,
            np.bitwise_not(self.read(source, task_core_id=task.core_id)),
            task_core_id=task.core_id,
        )

    def _bitwise_shift(self, task: Task, operation: str) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        values = self.read(source, task_core_id=task.core_id)
        shift = _resolve_int(task.metadata.get("shift"), self.bindings)
        bits = values.dtype.itemsize * 8
        if shift < 0 or shift > bits:
            raise ProgramValidationError(
                f"bitwise shift must be in [0, {bits}], got {shift}"
            )
        implementation = (
            np.left_shift if operation == "bitwise_lshift" else np.right_shift
        )
        result = implementation(values, shift).astype(values.dtype, copy=False)
        self.write(destination, result, task_core_id=task.core_id)

    def _sequence(self, task: Task) -> None:
        destination = _operand(task, "dst")
        count = _resolve_int(task.metadata.get("count"), self.bindings)
        first = _resolve_number(task.metadata.get("first_value"), self.bindings)
        difference = _resolve_number(task.metadata.get("difference"), self.bindings)
        values = first + difference * np.arange(count, dtype=np.int64)
        self.write(
            destination,
            np.asarray(values, dtype=_numpy_dtype(destination.dtype)),
            task_core_id=task.core_id,
        )

    def _reduce_sum_experiment(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        count = _resolve_int(task.metadata.get("count"), self.bindings)
        if count < 0:
            raise ProgramValidationError(
                "reducesum_experiment count must not be negative"
            )
        values = self.read(source, task_core_id=task.core_id).reshape(-1)
        result = np.sum(values, dtype=values.dtype)
        self.write(
            destination,
            np.asarray([result], dtype=_numpy_dtype(destination.dtype)),
            task_core_id=task.core_id,
        )

    def _sum_experiment(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        details = task.metadata.get("sum_experiment")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"sum_experiment task {task.task_id!r} requires metadata"
            )
        outer = _resolve_int(details["outer"], self.bindings)
        inner = _resolve_int(details["inner"], self.bindings)
        valid = _resolve_int(details["valid"], self.bindings)
        if min(outer, inner, valid) < 0:
            raise ProgramValidationError("sum_experiment extents must not be negative")
        if valid > inner:
            raise ProgramValidationError(
                "sum_experiment valid width must not exceed inner width"
            )
        if inner * _numpy_dtype(source.dtype).itemsize % 32:
            raise ProgramValidationError(
                "sum_experiment inner rows must be 32-byte aligned"
            )
        values = self.read(source, task_core_id=task.core_id)
        result = np.sum(values[:, :valid], axis=1, dtype=values.dtype)
        self.write(
            destination,
            np.asarray(result, dtype=_numpy_dtype(destination.dtype)),
            task_core_id=task.core_id,
        )

    def _gather(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        offsets = _operand(task, "offsets")
        source_values = self.read(source, task_core_id=task.core_id).reshape(-1)
        offset_values = self.read(offsets, task_core_id=task.core_id).reshape(-1)
        base = _resolve_int(task.metadata.get("base"), self.bindings)
        if base < 0:
            raise ProgramValidationError(
                f"gather base byte address must not be negative, got {base}"
            )
        itemsize = source_values.dtype.itemsize
        byte_addresses = offset_values.astype(np.int64) + base
        misaligned = byte_addresses % itemsize != 0
        if np.any(misaligned):
            bad = int(byte_addresses[np.flatnonzero(misaligned)[0]])
            raise ProgramValidationError(
                f"gather byte address must align with element size {itemsize}, got {bad}"
            )
        indices = byte_addresses // itemsize
        invalid = (indices < 0) | (indices >= source_values.size)
        if np.any(invalid):
            bad = int(indices[np.flatnonzero(invalid)[0]])
            raise ProgramValidationError(
                f"gather source index out of range: {bad} for {source_values.size} elements"
            )
        self.write(
            destination,
            source_values[indices].astype(source_values.dtype, copy=False),
            task_core_id=task.core_id,
        )

    def _gatherb(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        offsets = _operand(task, "offsets")
        repeat = _resolve_int(task.metadata.get("repeat"), self.bindings)
        dst_block_stride = _resolve_int(
            task.metadata.get("dst_block_stride"), self.bindings
        )
        dst_repeat_stride = _resolve_int(
            task.metadata.get("dst_repeat_stride"), self.bindings
        )
        if min(repeat, dst_block_stride, dst_repeat_stride) < 0:
            raise ProgramValidationError("gatherb repeat/strides must not be negative")
        if repeat > 255:
            raise ProgramValidationError(
                f"gatherb repeat must not exceed 255, got {repeat}"
            )
        source_values = self.read(source, task_core_id=task.core_id).reshape(-1)
        offset_values = self.read(offsets, task_core_id=task.core_id)
        block_elements = _resolve_int(
            task.metadata.get("elements_per_block"), self.bindings
        )
        blocks = np.empty(
            offset_values.shape + (block_elements,), dtype=source_values.dtype
        )
        for index in np.ndindex(offset_values.shape):
            byte_offset = int(offset_values[index])
            if byte_offset < 0 or byte_offset % 32:
                raise ProgramValidationError(
                    f"gatherb byte offset must be non-negative and 32-byte aligned, got {byte_offset}"
                )
            source_index = byte_offset // source_values.dtype.itemsize
            if source_index + block_elements > source_values.size:
                raise ProgramValidationError(
                    f"gatherb source block out of range at byte offset {byte_offset}"
                )
            blocks[index] = source_values[source_index:source_index + block_elements]
        self.write(destination, blocks, task_core_id=task.core_id)

    def _gather_mask(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        source_values = self.read(source, task_core_id=task.core_id).reshape(-1)
        mode = task.metadata.get("mode")
        if mode == "fixed":
            pattern = task.metadata.get("pattern")
            residues = {
                "P0101": (0, 2),
                "P1010": (1, 3),
                "P0001": (0,),
                "P0010": (1,),
                "P0100": (2,),
                "P1000": (3,),
                "P1111": (0, 1, 2, 3),
            }[pattern]
            indices = np.flatnonzero(
                np.isin(np.arange(source_values.size) % 4, residues)
            )
        elif mode == "custom":
            offsets = _operand(task, "offsets")
            indices = self.read(
                offsets, task_core_id=task.core_id
            ).reshape(-1).astype(np.int64)
            invalid = indices >= source_values.size
            if np.any(invalid):
                bad = int(indices[np.flatnonzero(invalid)[0]])
                raise ProgramValidationError(
                    f"gather_mask source index out of range: {bad}"
                )
        else:
            raise UnsupportedSimOpError(f"unsupported gather_mask mode {mode!r}")
        result = np.zeros(destination.shape, dtype=source_values.dtype).reshape(-1)
        if indices.size > result.size:
            raise ProgramValidationError(
                "gather_mask destination cannot hold all selected elements"
            )
        result[:indices.size] = source_values[indices]
        self.write(
            destination,
            result.reshape(destination.shape),
            task_core_id=task.core_id,
        )

    def _transpose(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        values = self.read(source, task_core_id=task.core_id)
        self.write(
            destination,
            np.ascontiguousarray(values.T),
            task_core_id=task.core_id,
        )

    def _reinterpretcast(self, task: Task) -> None:
        destination = _operand(task, "dst")
        source = _operand(task, "src")
        if task.metadata.get("materialize_view"):
            source_values = np.ascontiguousarray(
                self.read(source, task_core_id=task.core_id)
            )
            destination_dtype = _numpy_dtype(destination.dtype)
            self.write(
                destination,
                source_values.view(destination_dtype).reshape(destination.shape),
                task_core_id=task.core_id,
            )
            return
        self.memory.alias(
            destination.buffer,
            source.buffer,
            scope=destination.scope,
            core_id=task.core_id,
        )

    def _topk(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        details = task.metadata.get("topk")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"topk task {task.task_id!r} requires topk metadata"
            )
        k = _resolve_int(details["k"], self.bindings)
        values = self.read(source, task_core_id=task.core_id).reshape(-1)
        order = np.argsort(-values.astype(np.float64), kind="stable")[:k]
        result = np.empty(2 * k, dtype=_numpy_dtype(destination.dtype))
        result[0::2] = values[order]
        result[1::2] = order
        self.write(destination, result, task_core_id=task.core_id)

    def _sort32(self, task: Task) -> None:
        source = _operand(task, "src")
        indices = _operand(task, "offsets")
        destination = _operand(task, "dst")
        repeat_times = _resolve_int(
            task.metadata.get("repeat_times"), self.bindings
        )
        if not (1 <= repeat_times <= 255):
            raise ProgramValidationError(
                f"sort32 repeatTimes must be in [1, 255], got {repeat_times}"
            )
        source_values = self.read(
            source, task_core_id=task.core_id
        ).reshape(repeat_times, 32)
        index_values = self.read(
            indices, task_core_id=task.core_id
        ).reshape(repeat_times, 32)
        multiplier = _resolve_int(
            task.metadata.get("output_multiplier"), self.bindings
        )
        result = np.zeros(
            repeat_times * 32 * multiplier,
            dtype=_numpy_dtype(destination.dtype),
        ).reshape(repeat_times, 32, multiplier)
        for repeat in range(repeat_times):
            order = np.argsort(
                -source_values[repeat].astype(np.float64), kind="stable"
            )
            result[repeat, :, 0] = source_values[repeat, order]
            encoded_indices = np.ascontiguousarray(
                index_values[repeat, order]
            )
            if destination.dtype == "float32":
                result[repeat, :, 1] = encoded_indices.view(np.float32)
            else:
                result[repeat, :, 2:4] = encoded_indices.view(np.float16).reshape(
                    32, 2
                )
        self.write(
            destination,
            result.reshape(-1),
            task_core_id=task.core_id,
        )

    def _sort(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id).reshape(-1)
        order = np.argsort(-values.astype(np.float64), kind="stable")
        result = np.empty(values.size * 2, dtype=_numpy_dtype(destination.dtype))
        result[0::2] = values[order]
        result[1::2] = order
        self.write(destination, result, task_core_id=task.core_id)

    def _init_sort_buf(self, task: Task) -> None:
        destination = _operand(task, "dst")
        details = task.metadata.get("init_sort_buf")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"init_sort_buf task {task.task_id!r} requires init_sort_buf metadata"
            )
        covered = _resolve_int(details["covered_elements"], self.bindings)
        if covered < 0:
            raise ProgramValidationError(
                "init_sort_buf covered element count must not be negative, "
                f"got {covered}"
            )
        if covered == 0:
            return
        region = replace(destination, shape=(covered,))
        pattern = np.empty(covered, dtype=np.int32)
        pattern[0::2] = _INIT_SORT_BUF_VALUE_INT32
        pattern[1::2] = -1
        self.write(region, pattern.view(np.float32), task_core_id=task.core_id)

    def _merge_sort(self, task: Task) -> None:
        destination = _operand(task, "dst")
        details = task.metadata.get("merge_sort")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"merge_sort task {task.task_id!r} requires merge_sort metadata"
            )
        record_width = _resolve_int(details["record_width"], self.bindings)
        sources = task.metadata.get("src_regions")
        if not isinstance(sources, (tuple, list)) or not all(
            isinstance(source, BufferRegion) for source in sources
        ):
            raise ProgramValidationError(
                f"merge_sort task {task.task_id!r} requires source regions"
            )
        values = []
        indices = []
        for source_number, source in enumerate(sources):
            records = self.read(
                source, task_core_id=task.core_id
            ).reshape(-1, record_width)
            if np.any(np.isnan(records[:, 0])):
                raise ProgramValidationError(
                    "merge_sort NaN ordering has no confirmed A2/A3 contract"
                )
            if np.any(records[:-1, 0] < records[1:, 0]):
                raise ProgramValidationError(
                    f"merge_sort source{source_number} is not descending"
                )
            values.append(records[:, 0])
            indices.append(records[:, 1:])
        merged_values = np.concatenate(values)
        merged_payloads = np.concatenate(indices)
        order = np.argsort(-merged_values.astype(np.float64), kind="stable")
        result = np.empty(
            (merged_values.size, record_width),
            dtype=_numpy_dtype(destination.dtype),
        )
        result[:, 0] = merged_values[order]
        result[:, 1:] = merged_payloads[order]
        self.write(destination, result.reshape(-1), task_core_id=task.core_id)

    def _atomic_add(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        source_values = self.read(source, task_core_id=task.core_id)
        destination_values = self.read(destination, task_core_id=task.core_id)
        self.write(
            destination,
            destination_values + source_values,
            task_core_id=task.core_id,
        )

    def _atomic_add_l0c(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        details = task.metadata.get("atomic")
        if not isinstance(details, Mapping):
            raise ProgramValidationError(
                f"atomic task {task.task_id!r} requires atomic metadata"
            )
        logical = unpack_matrix(
            self.read(source, task_core_id=task.core_id),
            "l0c",
            tuple(details["source_shape"]),
        )
        rows = _resolve_int(details["rows"], self.bindings)
        cols = _resolve_int(details["cols"], self.bindings)
        converted = logical[:rows, :cols].astype(
            _numpy_dtype(destination.dtype)
        )
        previous = self.read(destination, task_core_id=task.core_id)
        self.write(
            destination,
            previous + converted,
            task_core_id=task.core_id,
        )

    def _block_reduce(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        result = self.read(destination, task_core_id=task.core_id)
        mask = _resolve_int(task.metadata.get("mask"), self.bindings)
        elements_per_block = _resolve_int(
            task.metadata.get("elements_per_block"), self.bindings
        )
        kind = task.metadata.get("reduce_kind")
        for block_index in range(values.shape[1]):
            active = min(elements_per_block, mask - block_index * elements_per_block)
            if active <= 0:
                continue
            block = values[:, block_index, :active]
            compute = block.astype(np.float32) if block.dtype == np.float16 else block
            if kind == "reduce_sum":
                result[:, block_index] = np.sum(compute, axis=1)
            elif kind == "reduce_max":
                result[:, block_index] = np.max(compute, axis=1)
            elif kind == "reduce_min":
                result[:, block_index] = np.min(compute, axis=1)
            else:
                raise UnsupportedSimOpError(
                    f"functional block reduction does not support kind {kind!r}"
                )
        self.write(destination, result, task_core_id=task.core_id)

    def _whole_reduce(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        result = self.read(destination, task_core_id=task.core_id)
        mask = _resolve_int(task.metadata.get("mask"), self.bindings)
        if mask == 0:
            self.write(destination, result, task_core_id=task.core_id)
            return
        elements_per_block = _resolve_int(
            task.metadata.get("elements_per_block"), self.bindings
        )
        kind = task.metadata.get("reduce_kind")
        order = task.metadata.get("reduce_order")
        for repeat_index in range(values.shape[0]):
            remaining = mask
            pieces = []
            for block_index in range(values.shape[1]):
                active = min(elements_per_block, remaining)
                pieces.append(values[repeat_index, block_index, :active])
                remaining -= active
            lanes = np.concatenate(pieces)
            compute = lanes.astype(np.float32) if lanes.dtype == np.float16 else lanes
            if kind == "reduce_sum":
                result[repeat_index, 0] = np.sum(compute)
            elif kind in {"reduce_max", "reduce_min"}:
                reducer = np.argmax if kind == "reduce_max" else np.argmin
                index = int(reducer(compute))
                result[repeat_index, 0] = compute[index]
                if order == "ORDER_VALUE_INDEX":
                    index_dtype = np.uint16 if result.dtype.itemsize == 2 else np.uint32
                    result[repeat_index, 1] = np.asarray(
                        [index], dtype=index_dtype
                    ).view(result.dtype)[0]
            else:
                raise UnsupportedSimOpError(
                    f"functional whole reduction does not support kind {kind!r}"
                )
        self.write(destination, result, task_core_id=task.core_id)

    def _clamp(self, task: Task, operation: str) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        if operation == "clamp":
            minimum = task.metadata.get("min_value")
            maximum = task.metadata.get("max_value")
            if not isinstance(minimum, (bool, int, float)) or not isinstance(
                maximum, (bool, int, float)
            ):
                raise UnsupportedSimOpError(
                    f"functional clamp requires literal bounds, got {minimum!r}, {maximum!r}"
                )
            if minimum > maximum:
                raise ProgramValidationError(
                    f"clamp minimum {minimum} exceeds maximum {maximum}"
                )
            result = np.clip(values, minimum, maximum)
        else:
            scalar = task.metadata.get("scalar")
            if not isinstance(scalar, (bool, int, float)):
                raise UnsupportedSimOpError(
                    f"functional {operation} requires a literal bound, got {scalar!r}"
                )
            result = (
                np.minimum(values, scalar)
                if operation == "clamp_max"
                else np.maximum(values, scalar)
            )
        self.write(destination, result, task_core_id=task.core_id)

    def _broadcast(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        destination_shape = tuple(
            _resolve_int(value, self.bindings) for value in destination.shape
        )
        try:
            result = np.broadcast_to(values, destination_shape)
        except ValueError as error:
            raise ProgramValidationError(
                f"cannot broadcast source shape {values.shape} to {destination_shape}"
            ) from error
        self.write(destination, result, task_core_id=task.core_id)

    def _compare(self, task: Task) -> None:
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        left_values = self.read(left, task_core_id=task.core_id)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif isinstance(task.metadata.get("scalar_src"), BufferRegion):
            scalar_source = _operand(task, "scalar_src")
            right_values = self.read(
                scalar_source, task_core_id=task.core_id
            ).reshape(-1)[0]
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
        else:
            raise ProgramValidationError(
                f"compare task {task.task_id!r} requires rhs or scalar metadata"
            )
        mode = task.metadata.get("compare_mode")
        implementation = _COMPARE_OPERATIONS.get(mode)
        if implementation is None:
            raise UnsupportedSimOpError(f"unsupported compare mode {mode!r}")
        predicate = np.asarray(implementation(left_values, right_values)).reshape(-1)
        packed = np.packbits(predicate, bitorder="little")
        destination_shape = tuple(
            _resolve_int(value, self.bindings) for value in destination.shape
        )
        required = int(np.prod(destination_shape))
        if packed.size > required:
            raise ProgramValidationError(
                f"packed compare result needs {packed.size} bytes, destination has {required}"
            )
        result = np.zeros(required, dtype=_numpy_dtype(destination.dtype))
        result[:packed.size] = packed
        self.write(
            destination,
            result.reshape(destination_shape),
            task_core_id=task.core_id,
        )

    def _select(self, task: Task) -> None:
        mask = _operand(task, "mask")
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        mask_values = self.read(mask, task_core_id=task.core_id)
        left_values = self.read(left, task_core_id=task.core_id)
        unpacked = np.unpackbits(
            np.asarray(mask_values, dtype=np.uint8).reshape(-1), bitorder="little"
        )
        if unpacked.size < left_values.size:
            raise ProgramValidationError(
                f"select mask provides {unpacked.size} bits for {left_values.size} values"
            )
        predicate = unpacked[:left_values.size].reshape(left_values.shape).astype(bool)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif isinstance(task.metadata.get("scalar_src"), BufferRegion):
            scalar_source = _operand(task, "scalar_src")
            right_values = self.read(
                scalar_source, task_core_id=task.core_id
            ).reshape(-1)[0]
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
        else:
            raise ProgramValidationError(
                f"select task {task.task_id!r} requires rhs, scalar_src, "
                "or scalar metadata"
            )
        self.write(
            destination,
            np.where(predicate, left_values, right_values),
            task_core_id=task.core_id,
        )

    def _tail_compare(self, task: Task) -> None:
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        left_values = self.read(left, task_core_id=task.core_id)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
        else:
            raise ProgramValidationError(
                f"tail compare task {task.task_id!r} requires rhs or scalar metadata"
            )
        mode = task.metadata.get("compare_mode")
        implementation = _COMPARE_OPERATIONS.get(mode)
        if implementation is None:
            raise UnsupportedSimOpError(f"unsupported tail compare mode {mode!r}")
        predicate = np.asarray(implementation(left_values, right_values))
        if predicate.ndim != 2:
            raise ProgramValidationError(
                f"tail compare requires a 2D valid rectangle, got {predicate.shape}"
            )
        self.write(
            destination,
            np.packbits(predicate, axis=1, bitorder="little"),
            task_core_id=task.core_id,
        )

    def _tail_select(self, task: Task) -> None:
        mask = _operand(task, "mask")
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        mask_values = self.read(mask, task_core_id=task.core_id)
        left_values = self.read(left, task_core_id=task.core_id)
        if mask_values.ndim != 2 or left_values.ndim != 2:
            raise ProgramValidationError(
                "tail select requires 2D mask and source valid rectangles"
            )
        unpacked = np.unpackbits(
            np.asarray(mask_values, dtype=np.uint8), axis=1, bitorder="little"
        )
        if unpacked.shape[1] < left_values.shape[1]:
            raise ProgramValidationError(
                f"tail select mask row provides {unpacked.shape[1]} bits for "
                f"{left_values.shape[1]} values"
            )
        predicate = unpacked[:, :left_values.shape[1]].astype(bool)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
        else:
            raise ProgramValidationError(
                f"tail select task {task.task_id!r} requires rhs or scalar metadata"
            )
        self.write(
            destination,
            np.where(predicate, left_values, right_values),
            task_core_id=task.core_id,
        )

    def _reduce(self, task: Task, operation: str) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        if values.shape[0] == 0 or values.shape[1] == 0:
            return
        axis = task.metadata.get("reduce_axis", 0)
        numpy_axis = 0 if axis == 0 else 1
        implementation = {
            "reduce_sum": np.sum,
            "reduce_max": np.max,
            "reduce_min": np.min,
        }[operation]
        result = implementation(values, axis=numpy_axis)
        if not task.metadata.get("clear", True):
            accumulator = _operand(task, "accumulator")
            previous = self.read(accumulator, task_core_id=task.core_id)
            result = {
                "reduce_sum": np.add,
                "reduce_max": np.maximum,
                "reduce_min": np.minimum,
            }[operation](previous, result)
        self.write(destination, result, task_core_id=task.core_id)

    def _resolve(self, region: BufferRegion, task_core_id: int) -> MemoryView:
        owner = region.core_id if region.core_id is not None else task_core_id
        if (
            region.scope not in {MemoryScope.GM, MemoryScope.WORKSPACE}
            and self._active_lane is not None
            and self._active_lane.value == "vector1"
        ):
            owner = self.memory.vector1_owner(task_core_id)
        allocation = self.memory.get(
            region.buffer,
            scope=region.scope,
            core_id=None if region.scope in {MemoryScope.GM, MemoryScope.WORKSPACE} else owner,
        )
        shape = tuple(_resolve_int(value, self.bindings) for value in region.shape)
        byte_offset = _resolve_int(region.byte_offset, self.bindings)
        strides = (
            None
            if region.strides_bytes is None
            else tuple(_resolve_int(value, self.bindings) for value in region.strides_bytes)
        )
        return allocation.view(
            byte_offset=byte_offset,
            shape=shape,
            dtype=region.dtype,
            strides_bytes=strides,
        )


def _operand(task: Task, name: str) -> BufferRegion:
    value = task.metadata.get(name)
    if not isinstance(value, BufferRegion):
        raise ProgramValidationError(
            f"task {task.task_id!r} requires BufferRegion metadata {name!r}"
        )
    return value


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    normalized = dtype.strip().lower()
    if "x" in normalized:
        raise UnsupportedSimOpError(f"vector-packed dtype is not executable yet: {dtype!r}")
    try:
        return np.dtype(normalized)
    except TypeError as error:
        raise UnsupportedSimOpError(
            f"NumPy cannot represent simulator dtype {dtype!r}"
        ) from error


def _resolve_int(value: Any, bindings: Mapping[str, int | float]) -> int:
    if isinstance(value, (AffineInt, SymbolicInt)):
        return value.evaluate(bindings)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ProgramValidationError(f"simulator integer value is not executable: {value!r}")


def _resolve_number(value: Any, bindings: Mapping[str, int | float]) -> Any:
    if isinstance(value, (AffineInt, SymbolicInt)):
        return value.evaluate(bindings)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise ProgramValidationError(f"simulator numeric value is not executable: {value!r}")
