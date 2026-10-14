# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2/A3 topology, PTO-derived estimates, and guarded A3 calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import SimulatorConfigError
from .layout import physical_index


@dataclass(frozen=True)
class DeviceProfile:
    """Static topology used by the simulator.

    Timing is deliberately kept separate from topology.  The A3 core counts mirror the
    repository's current fallback table and must be revised when measured data is available.
    """

    platform: str
    cube_core_count: int
    vector_core_count: int
    vector_lanes_per_cube: int
    cube_pipes: tuple[str, ...] = ("mte2", "mte1", "m", "fix")
    vector_pipes: tuple[str, ...] = ("mte2", "v", "mte3")
    calibration: str = "uncalibrated"


@dataclass(frozen=True)
class TimingProfile:
    """Parameter table for a discrete-event scheduler.

    Explicit operation costs override the selected estimator.  The default simulator profile
    uses PTO's documented fallback formulas and labels their provenance; it does not claim
    measured A2/A3 hardware latency.
    """

    platform: str
    operation_cycles: Mapping[str, int] = field(default_factory=dict)
    fallback_cycles: int = 1
    calibration: str = "uncalibrated-unit-cost"
    estimator: str = "fixed"

    def __post_init__(self) -> None:
        get_device_profile(self.platform)
        if self.fallback_cycles <= 0:
            raise SimulatorConfigError("fallback_cycles must be positive")
        normalized = {}
        for name, cycles in self.operation_cycles.items():
            if not name:
                raise SimulatorConfigError("operation name must not be empty")
            if cycles <= 0:
                raise SimulatorConfigError(f"operation cycle count must be positive: {name}")
            normalized[str(name)] = int(cycles)
        object.__setattr__(self, "operation_cycles", MappingProxyType(normalized))
        object.__setattr__(self, "platform", normalize_platform(self.platform))
        if self.estimator not in {"fixed", "pto-fallback", "pto-cce-mock"}:
            raise SimulatorConfigError(
                "timing estimator must be one of: fixed, pto-fallback, pto-cce-mock"
            )
        if self.estimator == "pto-cce-mock" and self.platform != "A3":
            raise SimulatorConfigError("pto-cce-mock timing is currently supported only on A3")

    def estimate_cycles(self, operation: str) -> int:
        """Return a configured cost or the visibly uncalibrated fallback cost."""
        return self.operation_cycles.get(operation, self.fallback_cycles)

    def estimate_task(
        self,
        operation: str,
        *,
        pipe: str,
        metadata: Mapping[str, Any],
    ) -> int:
        """Return the duration of one bridged task."""
        return self.estimate_task_with_calls(operation, pipe=pipe, metadata=metadata)[0]

    def estimate_task_with_source(
        self,
        operation: str,
        *,
        pipe: str,
        metadata: Mapping[str, Any],
    ) -> tuple[int, str]:
        """Return a duration and its cost-model source."""
        cycles, source, _ = self.estimate_task_with_calls(operation, pipe=pipe, metadata=metadata)
        return cycles, source

    def estimate_task_with_calls(
        self,
        operation: str,
        *,
        pipe: str,
        metadata: Mapping[str, Any],
    ) -> tuple[int, str, tuple[dict[str, Any], ...]]:
        """Estimate one bridged task, preserving explicit per-operation overrides.

        ``pto-fallback`` ports the public PTO perf-sim fallback formulas for
        instructions not covered by its lightweight cost model.  It is a
        relative-performance model, not an A2/A3 hardware calibration.
        """
        configured = self.operation_cycles.get(operation)
        if configured is not None:
            return configured, "explicit", ()
        if self.estimator == "pto-cce-mock":
            calls = _pto_cce_mock_calls(operation, pipe, metadata)
            if calls is not None:
                source = (
                    "a3-hardware-calibrated"
                    if calls[0]["name"] in {
                        "vmuls_fa_runtime_count_mask_a3_calibrated_analogue",
                        "vmuls_count_mask_calibrated_analogue",
                        "copy_ubuf_to_ubuf_a3_calibrated_analogue",
                        "cast_fp32_to_fp16_a3_calibrated_analogue",
                        "row_expand_sub_a3_calibrated_analogue",
                        "binary_mask_fa_fold_a3_calibrated_analogue",
                        "vexp_fa_inplace_a3_calibrated_analogue",
                        "copy_ubuf_to_gm_fa_shape_a3_calibrated_analogue",
                    }
                    else "pto-cce-mock"
                )
                return sum(call["cycles"] for call in calls), source, calls
        elif self.estimator != "pto-fallback":
            return self.fallback_cycles, "fixed", ()

        # PTO's fallback formulas consume a logical rows*cols element count.
        # The TIR bridge retains transfer bytes and regions instead, so derive
        # that count only when dtype and shape information are concrete.
        elements = _timing_elements(metadata)
        normalized_pipe = pipe.strip().lower()
        normalized_operation = operation.strip().lower()
        if normalized_operation in {
            "set_flag", "wait_flag", "auto_set_flag", "auto_wait_flag",
            "set_cross_flag", "wait_cross_flag", "auto_set_cross_flag",
            "auto_wait_cross_flag", "pipe_barrier", "barrier_all",
        }:
            return 1, "pto-fallback", ()
        if normalized_pipe == "m":
            mma_cycles = _pto_mma_cycles(metadata)
            if mma_cycles is not None:
                return mma_cycles, "pto-fallback", ()
        if elements is None:
            return self.fallback_cycles, "fixed", ()
        if normalized_pipe == "m":
            return 4 + elements // 16, "pto-fallback", ()
        if normalized_pipe in {"mte2", "mte3", "fix"}:
            return 3 + elements * 2 // 64, "pto-fallback", ()
        if normalized_pipe == "mte1":
            return 1 + elements // 64, "pto-fallback", ()
        if normalized_pipe == "s":
            return 1, "pto-fallback", ()
        return 2 + elements // 32, "pto-fallback", ()


def _timing_elements(metadata: Mapping[str, Any]) -> int | None:
    """Recover PTO's logical element count from bridge metadata when possible."""
    transfer_bytes = metadata.get("transfer_bytes")
    if isinstance(transfer_bytes, int) and transfer_bytes >= 0:
        for key in ("src", "dst"):
            dtype = getattr(metadata.get(key), "dtype", None)
            itemsize = _DTYPE_BYTES.get(str(dtype).lower())
            if itemsize is not None:
                return transfer_bytes // itemsize
    for key in ("dst", "src"):
        shape = getattr(metadata.get(key), "shape", None)
        if isinstance(shape, tuple) and all(isinstance(extent, int) and extent >= 0 for extent in shape):
            elements = 1
            for extent in shape:
                elements *= extent
            return elements
    return None


_DTYPE_BYTES = {
    "float16": 2, "half": 2, "bfloat16": 2, "float32": 4,
    "float": 4, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4,
}


# PTO A2/A3 CCE mock: arch_config.hpp and cce_costmodel_core.hpp.  The
# bandwidth table is in GiB/s in this path, unlike the lightweight model's
# bandwidth_bytes_per_us runtime-config interface.
_PTO_CCE_FREQUENCY_HZ = 1_850_000_000
_PTO_CCE_GIB = 1024 ** 3
_PTO_CCE_GM_TO_UB_GIB_S = 100.9
_PTO_CCE_GM_TO_L1_GIB_S = 135.0
_PTO_CCE_UB_TO_GM_GIB_S = 188.46
_PTO_CCE_L1_TO_L0A_GIB_S = 441.0
_PTO_CCE_L1_TO_L0B_GIB_S = 220.5
_PTO_CCE_L1_TO_BT_GIB_S = 32.0
_PTO_CCE_L0C_TO_GM_GIB_S = 70.0
_PTO_CCE_UB_TO_UB_GIB_S = 1024.0


def _gemm_zn_col_slabs_match(
    metadata: Mapping[str, Any], source: Any, source_shape: tuple[int, int],
    origin: tuple[int, int], shape: tuple[int, int],
) -> bool:
    """Check the exact zN column-slab view of a full-row GEMM window.

    A zN tile is physically grouped by 16-column slabs. A logical GEMM
    window can therefore have several BufferRegions even though the helper
    issues ordinary L1 LoadData calls on that one zN allocation.
    """
    rows, cols = shape
    if (
        origin[0] % 16
        or origin[1] % 16
        or not isinstance(source.byte_offset, int)
        or source.byte_offset % 32
        or getattr(source, "shape", None) != (source_shape[0] * source_shape[1],)
    ):
        return False
    regions = metadata.get("src_regions")
    if not isinstance(regions, tuple) or len(regions) != cols // 16:
        return False
    for index, region in enumerate(regions):
        expected_offset = source.byte_offset + 2 * physical_index(
            "zn", origin[0], origin[1] + 16 * index, source_shape, 2
        )
        if (
            getattr(region, "buffer", None) != source.buffer
            or getattr(region, "scope", None) != source.scope
            or getattr(region, "core_id", None) != source.core_id
            or str(getattr(region, "dtype", "")).lower() != "float16"
            or getattr(region, "shape", None) != (rows, 16)
            or getattr(region, "strides_bytes", None) != (32, 2)
            or getattr(region, "byte_offset", None)
            != expected_offset
        ):
            return False
    return True


def _pto_cce_mock_calls(
    operation: str, pipe: str, metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], ...] | None:
    """Estimate only TileLang helper paths with an auditable CCE-call mapping.

    A PTO CCE mock accumulates the low-level call costs into one top-level
    instruction; this function deliberately returns one task duration, not a
    fabricated low-level instruction timeline.  Unknown helper branches fall
    back to the existing PTO-derived estimator.
    """
    normalized_pipe = pipe.strip().lower()
    # common.h::brcb -> CANN dav_c220 BrcbImpl -> one vbrcb. The PTO
    # isolated scope has zero head/slope and an 18-cycle pending tail.
    # Limit this first mapping to the standard non-overlapping packed layout.
    if operation.lower() == "brcb_experiment" and normalized_pipe == "v":
        source, destination = metadata.get("src"), metadata.get("dst")
        details = metadata.get("brcb")
        dtype = str(getattr(source, "dtype", "")).lower()
        if (
            not isinstance(details, Mapping)
            or dtype not in {"float16", "float32"}
            or not str(metadata.get("lowered_operation", "")).lower().endswith(
                "brcb_experiment"
            )
        ):
            return None
        repeat = details.get("repeat")
        if (
            type(repeat) is not int or not 1 <= repeat <= 255
            or type(details.get("blk_stride")) is not int
            or type(details.get("rep_stride")) is not int
            or details.get("blk_stride") != 1 or details.get("rep_stride") != 8
        ):
            return None
        itemsize = _DTYPE_BYTES[dtype]
        for region, shape in (
            (source, (repeat * 8,)),
            (destination, (repeat * 8 * (32 // itemsize),)),
        ):
            if (
                getattr(getattr(region, "scope", None), "value", None) != "ub"
                or str(getattr(region, "dtype", "")).lower() != dtype
                or getattr(region, "shape", None) != shape
                or getattr(region, "strides_bytes", None) not in {None, (itemsize,)}
                or type(getattr(region, "byte_offset", None)) is not int
                or region.byte_offset % 32
            ):
                return None
        if source.core_id != destination.core_id or source.buffer == destination.buffer:
            return None
        return ({
            "name": "vbrcb", "pipe": "v", "cycles": 18,
            "args": {"repeat": repeat, "dst_block_stride": 1,
                     "dst_repeat_stride": 8, "dtype": dtype,
                     "mask_setup": "CANN ResetMask (not timed)"},
        },)

    # RowExpandBinOpExperimentCodegen's no-tmp branch emits exactly one
    # full-bitmask Mul<T,false>/Div<T,false> with block strides (1,1,0).
    # CANN dav_c220 directly calls vmul/vdiv with those explicit repeats.
    # Do not apply this to count-mask Add/Mul/Div or the tmp+Brcb branch.
    row_expand_cost = {
        "row_expand_mul_experiment": ("vmul", 13, 2, 19),
        "row_expand_div_experiment": ("vdiv", 13, 8, 25),
    }.get(operation.lower())
    if row_expand_cost is not None and normalized_pipe == "v":
        details = metadata.get("row_expand")
        destination, source, packed = (
            metadata.get("dst"), metadata.get("src"), metadata.get("scalar_src")
        )
        dtype = str(getattr(destination, "dtype", "")).lower()
        if (
            not isinstance(details, Mapping)
            or dtype not in {"float16", "float32"}
            or metadata.get("scratch") is not None
            or not str(metadata.get("lowered_operation", "")).lower().endswith(
                operation.lower()
            )
        ):
            return None
        rows = details.get("rows")
        itemsize = _DTYPE_BYTES[dtype]
        block_elements = 32 // itemsize
        row_elements = 8 * block_elements
        strides = getattr(destination, "strides_bytes", None)
        if (
            type(rows) is not int or not 1 <= rows <= 255 or rows % 8
            or details.get("row_elements") != row_elements
            or not isinstance(strides, tuple) or len(strides) != 2
            or type(strides[0]) is not int
            or strides[1] != itemsize or strides[0] < 256 or strides[0] % 32
            or not 1 <= strides[0] // 32 <= 255
        ):
            return None
        for region in (destination, source, packed):
            if (
                getattr(getattr(region, "scope", None), "value", None) != "ub"
                or str(getattr(region, "dtype", "")).lower() != dtype
                or type(getattr(region, "byte_offset", None)) is not int
                or region.byte_offset % 32
                or getattr(region, "core_id", None) != destination.core_id
            ):
                return None
        if (
            destination.shape != (rows, row_elements)
            or source.shape != destination.shape or source.strides_bytes != strides
            or packed.shape != (rows * block_elements,)
            or packed.strides_bytes not in {None, (itemsize,)}
            or packed.buffer in {source.buffer, destination.buffer}
            or (source.buffer == destination.buffer
                and source.byte_offset != destination.byte_offset)
        ):
            return None
        return ({
            "name": row_expand_cost[0], "pipe": "v",
            "cycles": row_expand_cost[1] + row_expand_cost[2] * rows + row_expand_cost[3],
            "args": {
                "repeat": rows,
                "mask": ((1 << 64) - 1, (1 << 64) - 1 if itemsize == 2 else 0),
                "block_strides": (1, 1, 0),
                "repeat_strides": (strides[0] // 32, strides[0] // 32, 1),
                "dtype": dtype, "helper_form": "explicit_bitmask_no_tmp",
            },
        },)
    # CANN dav_c220 BlockReduceMax/Sum lower to grouped (32-byte block)
    # reduction, vcgmax/vcgadd. Use explicit bridge repeat/stride controls.
    # Costs are isolated scopes; FA's PIPE_V barriers between levels flush
    # the pending tail and clear the queue. Mask setup, as in the existing
    # vector mappings, stays in args without a scalar-issue timing model.
    grouped_reduce = {
        "block_reduce_max": ("vcgmax", 17),
        "block_reduce_sum": ("vcgadd", 24),
    }.get(operation.lower())
    if grouped_reduce is not None and normalized_pipe == "v":
        source, destination = metadata.get("src"), metadata.get("dst")
        dtype = str(getattr(source, "dtype", "")).lower()
        controls = tuple(metadata.get(key) for key in (
            "repeat", "mask", "dst_rep_stride", "src_blk_stride", "src_rep_stride"
        ))
        if dtype not in {"float16", "float32"} or not all(
            type(value) is int for value in controls
        ):
            return None
        repeat, mask, dst_stride, block_stride, repeat_stride = controls
        itemsize = _DTYPE_BYTES[dtype]
        block_elements = 32 // itemsize
        active_blocks = (mask + block_elements - 1) // block_elements
        if (
            not 1 <= repeat <= 255 or not 1 <= mask <= 8 * block_elements
            or any(not 0 <= stride <= 255 for stride in (
                dst_stride, block_stride, repeat_stride
            ))
            or metadata.get("elements_per_block") != block_elements
            or getattr(source, "shape", None) != (repeat, active_blocks, block_elements)
            or getattr(source, "strides_bytes", None)
            != (repeat_stride * 32, block_stride * 32, itemsize)
            or getattr(destination, "shape", None) != (repeat, active_blocks)
            or getattr(destination, "strides_bytes", None)
            != (dst_stride * 8 * itemsize, itemsize)
            or getattr(source, "core_id", None) != getattr(destination, "core_id", None)
            or any(
                getattr(getattr(region, "scope", None), "value", None) != "ub"
                or str(getattr(region, "dtype", "")).lower() != dtype
                or type(getattr(region, "byte_offset", None)) is not int
                or region.byte_offset % 32
                for region in (source, destination)
            )
        ):
            return None
        return ({
            "name": grouped_reduce[0], "pipe": "v",
            "cycles": 14 + repeat + grouped_reduce[1],
            "args": {"repeat": repeat, "mask": mask,
                     "dst_repeat_stride": dst_stride,
                     "src_block_stride": block_stride,
                     "src_repeat_stride": repeat_stride, "dtype": dtype},
        },)
    # FA v06 folds two 64-column halves of a 128-column FP32 score tile
    # with one masked Max/Add call. A same-helper A3 PMU control at 0/4/8
    # calls measured 0.147778/0.438889/0.730000 us of Vector active time
    # per AIV: 131 hardware cycles per call at 1.8 GHz. PTO's isolated
    # vmax/vadd calls instead cost 158/96 cycles. This is an aggregate PMU
    # slope analogue, not an instruction timestamp or a general binary cost.
    if operation.lower() == "binary_mask_experiment" and normalized_pipe == "v":
        destination, left, right = (
            metadata.get("dst"), metadata.get("lhs"), metadata.get("rhs")
        )
        kind = metadata.get("binary_kind")
        left_offset = getattr(left, "byte_offset", None)
        if (
            kind in {"max", "add"}
            and metadata.get("binary_mask") == ((1 << 64) - 1, 0)
            and all(
                getattr(getattr(region, "scope", None), "value", None) == "ub"
                and str(getattr(region, "dtype", "")).lower() == "float32"
                and getattr(region, "shape", None) == (64, 8, 8)
                for region in (destination, left, right)
            )
            and getattr(destination, "buffer", None) == "fold_tmp"
            and getattr(left, "buffer", None) == "acc_s_ub"
            and getattr(right, "buffer", None) == "acc_s_ub"
            and getattr(destination, "strides_bytes", None) == (256, 32, 4)
            and getattr(left, "strides_bytes", None) == (512, 32, 4)
            and getattr(right, "strides_bytes", None) == (512, 32, 4)
            and getattr(destination, "byte_offset", None) == 0
            and left_offset in (0, 32768)
            and getattr(right, "byte_offset", None) == left_offset + 256
            and destination.core_id == left.core_id == right.core_id
        ):
            cycles = round(131 * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "binary_mask_fa_fold_a3_calibrated_analogue",
                "pipe": "v", "cycles": cycles,
                "args": {"kind": kind, "repeat": 64,
                         "mask": "0xFFFFFFFFFFFFFFFF,0",
                         "src_slot_byte_offset": left_offset,
                         "pto_isolated_cycles": 158 if kind == "max" else 96,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
    # FA v06's alpha and score Exp calls each lower to one in-place
    # AscendC::Exp<float>(dst, src, count) on a contiguous UB slot. PTO's
    # isolated vexp CCE costs 13 + 4*repeat + 24 cycles, but an exact-helper
    # A3 PMU control measures only 30/284 active cycles per 64/8192-element
    # call at 1.8 GHz. Keep both values visible, and do not generalize this
    # active-time increment to other shapes or helpers.
    if operation.lower() == "exp" and normalized_pipe == "v":
        source, destination = metadata.get("src"), metadata.get("dst")
        count = getattr(source, "shape", None)
        fa_slots = {
            ("alpha_ring", (64,)): (0, 256, 512),
            ("acc_s_ub", (8192,)): (0, 32768),
        }
        allowed_offsets = fa_slots.get((getattr(source, "buffer", None), count))
        if (
            allowed_offsets is not None
            and getattr(getattr(source, "scope", None), "value", None) == "ub"
            and getattr(getattr(destination, "scope", None), "value", None) == "ub"
            and str(getattr(source, "dtype", "")).lower() == "float32"
            and str(getattr(destination, "dtype", "")).lower() == "float32"
            and getattr(destination, "shape", None) == count
            and getattr(source, "strides_bytes", None) is None
            and getattr(destination, "strides_bytes", None) is None
            and getattr(destination, "buffer", None) == source.buffer
            and getattr(destination, "core_id", None) == source.core_id
            and getattr(source, "byte_offset", None) in allowed_offsets
            and getattr(destination, "byte_offset", None) == source.byte_offset
        ):
            elements = count[0]
            repeat = elements // 64
            measured_cycles = 30 if elements == 64 else 284
            cycles = round(measured_cycles * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "vexp_fa_inplace_a3_calibrated_analogue",
                "pipe": "v", "cycles": cycles,
                "args": {"count": elements, "repeat": repeat,
                         "pto_vexp_isolated_cycles": 13 + 4 * repeat + 24,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
    # FA v06's no-tmp row-expand Sub lowers to one in-place sub_mask<float>
    # on a 64-column window of a 128-column physical UB row. PTO has only
    # generic vsub, whose isolated repeat=64 cost is 159 cycles; it does
    # not model this masked helper's physical row stride or pointer half.
    if operation.lower() == "row_expand_sub_experiment" and normalized_pipe == "v":
        details = metadata.get("row_expand")
        source = metadata.get("src")
        destination = metadata.get("dst")
        packed = metadata.get("scalar_src")
        offset = getattr(source, "byte_offset", None)
        if (
            isinstance(details, Mapping)
            and details.get("rows") == 64
            and details.get("row_elements") == 64
            and metadata.get("scratch") is None
            and all(
                getattr(getattr(region, "scope", None), "value", None) == "ub"
                and str(getattr(region, "dtype", "")).lower() == "float32"
                for region in (source, destination, packed)
            )
            and getattr(source, "shape", None) == (64, 64)
            and getattr(destination, "shape", None) == (64, 64)
            and getattr(source, "strides_bytes", None) == (512, 4)
            and getattr(destination, "strides_bytes", None) == (512, 4)
            and getattr(packed, "shape", None) == (512,)
            and getattr(packed, "strides_bytes", None) is None
            and source.buffer == destination.buffer
            and source.buffer != packed.buffer
            and source.core_id == destination.core_id == packed.core_id
            and isinstance(offset, int)
            and offset == destination.byte_offset
            and offset % 32768 in (0, 256)
            and isinstance(getattr(packed, "byte_offset", None), int)
            and packed.byte_offset % 32 == 0
        ):
            # Same generated sub_mask parameters on A3 hardware: repeat=64,
            # strides=(16,16,1), full FP32 mask. PMU Vector active-time
            # increment is 0.093889 us for the left half, 0.095556 us for
            # the right half at 1.8 GHz. This is not a PTO literal call.
            right_half = offset % 32768 == 256
            hardware_cycles = round(
                (172 if right_half else 169) * _PTO_CCE_FREQUENCY_HZ
                / 1_800_000_000
            )
            return ({
                "name": "row_expand_sub_a3_calibrated_analogue",
                "pipe": "v", "cycles": hardware_cycles,
                "args": {"repeat": 64, "mask": "full", "block_strides": (1, 1, 0),
                         "repeat_strides": (16, 16, 1),
                         "half": "right" if right_half else "left",
                         "pto_vsub_isolated_cycles": 159,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
    # common.h::copy_ub_to_ub uses one AscendC::DataCopy only for matching
    # dtypes and contiguous rows. Its FP32->FP16 path instead calls Cast.
    # PTO prices UB_TO_UB by bandwidth; preserve TileLang's Vector scheduling
    # lane, since the PTO cost key is not an AIV hardware timestamp.
    if operation.lower() == "copy_ub_to_ub" and normalized_pipe == "v":
        details = metadata.get("copy")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping) or details.get("layout") != "ub_to_ub":
            return None
        rows, cols = details.get("valid_rows"), details.get("valid_cols")
        dtype = str(getattr(source, "dtype", "")).lower()
        itemsize = _DTYPE_BYTES.get(dtype)
        if (
            details.get("cast_mode") == "CAST_RINT"
            and dtype == "float32"
            and str(getattr(destination, "dtype", "")).lower() == "float16"
            and (rows, cols) == (64, 128)
            and getattr(getattr(source, "scope", None), "value", None) == "ub"
            and getattr(getattr(destination, "scope", None), "value", None) == "ub"
            and getattr(source, "shape", None) == (64, 128)
            and getattr(destination, "shape", None) == (64, 128)
            and getattr(source, "strides_bytes", None) == (512, 4)
            and getattr(destination, "strides_bytes", None) == (256, 2)
            and getattr(source, "buffer", None) != getattr(destination, "buffer", None)
            and getattr(source, "core_id", None) == getattr(destination, "core_id", None)
            and isinstance(getattr(source, "byte_offset", None), int)
            and isinstance(getattr(destination, "byte_offset", None), int)
            and source.byte_offset % 32 == 0
            and destination.byte_offset % 32 == 0
        ):
            # Same TileLang Cast helper, 4/8/16 repeats on A3: each extra
            # copy adds 0.075 us of PMU Vector active time at 1.8 GHz.
            # This is an aggregate slope, not a literal PTO CCE call or
            # an instruction latency. Keep the default fallback unchanged.
            hardware_cycles = round(135 * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "cast_fp32_to_fp16_a3_calibrated_analogue",
                "pipe": "v", "cycles": hardware_cycles,
                "args": {"count": rows * cols, "rounding": "CAST_RINT",
                         "basis": "A3 PMU Vector active-time increment"},
            },)
        if (
            details.get("cast_mode", object()) is not None
            or dtype not in {"float16", "float32"}
            or not all(isinstance(value, int) and value > 0 for value in (rows, cols))
            or getattr(getattr(source, "scope", None), "value", None) != "ub"
            or getattr(getattr(destination, "scope", None), "value", None) != "ub"
            or str(getattr(destination, "dtype", "")).lower() != dtype
            or getattr(source, "shape", None) != (rows, cols)
            or getattr(destination, "shape", None) != (rows, cols)
            or getattr(source, "strides_bytes", None) != (cols * itemsize, itemsize)
            or getattr(destination, "strides_bytes", None) != (cols * itemsize, itemsize)
            or getattr(source, "buffer", None) == getattr(destination, "buffer", None)
            or getattr(source, "core_id", None) != getattr(destination, "core_id", None)
            or not isinstance(getattr(source, "byte_offset", None), int)
            or not isinstance(getattr(destination, "byte_offset", None), int)
            or source.byte_offset % 32 or destination.byte_offset % 32
        ):
            return None
        byte_count = rows * cols * itemsize
        if byte_count % 32:
            return None
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_UB_TO_UB_GIB_S
                     * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None  # PTO truncates tiny copies to zero; keep a scheduled task.
        if dtype == "float32" and (rows, cols) == (64, 128):
            # Same TileLang helper, 1/4/8/16 copies on A3 hardware: each
            # extra copy added 0.085 us to each AIV's PMU Vector active time
            # at 1.8 GHz. This is an aggregate increment, not an instruction
            # latency; keep the literal PTO 55-cycle result for comparison.
            hardware_cycles = round(153 * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "copy_ubuf_to_ubuf_a3_calibrated_analogue",
                "pipe": "v", "cycles": hardware_cycles,
                "args": {"n_burst": 1, "len_burst": byte_count // 32,
                         "src_gap": 0, "dst_gap": 0, "pto_cost_key": "UB_TO_UB",
                         "pto_cce_cycles": cycles,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
        return ({
            "name": "copy_ubuf_to_ubuf", "pipe": "v", "cycles": cycles,
            "args": {"n_burst": 1, "len_burst": byte_count // 32,
                     "src_gap": 0, "dst_gap": 0, "pto_cost_key": "UB_TO_UB"},
        },)
    # CANN dav-c220 Muls(dst, src, scalar, count) uses a count mask. PTO's
    # repeat-only CCE vmuls estimator loses that count, so this is explicitly
    # a hardware-calibrated analogue, not a literal PTO CCE call. The guards
    # cover separately measured FP32 in-place x2 and FA runtime-scalar paths.
    if operation.lower() == "muls" and normalized_pipe == "v":
        helper = metadata.get("scalar_helper")
        destination, source = metadata.get("dst"), metadata.get("lhs")
        count = helper.get("count") if isinstance(helper, Mapping) else None
        if (
            isinstance(helper, Mapping)
            and helper.get("form") == "count_mask"
            and count == 8192
            and metadata.get("tail_kind") is None
            and metadata.get("scalar") == "sm_scale"
            and getattr(getattr(destination, "scope", None), "value", None) == "ub"
            and getattr(getattr(source, "scope", None), "value", None) == "ub"
            and str(getattr(destination, "dtype", "")).lower() == "float32"
            and str(getattr(source, "dtype", "")).lower() == "float32"
            and getattr(destination, "shape", None) == (8192,)
            and getattr(source, "shape", None) == (8192,)
            and getattr(destination, "strides_bytes", None) is None
            and getattr(source, "strides_bytes", None) is None
            and getattr(destination, "buffer", None) == "acc_s_ub"
            and getattr(source, "buffer", None) == "acc_s_ub"
            and getattr(destination, "core_id", None) == getattr(source, "core_id", None)
            and getattr(destination, "byte_offset", None) in (0, 32768)
            and getattr(source, "byte_offset", None) == destination.byte_offset
        ):
            # Exact FA v06 runtime-scalar two-slot path on A3. A same-helper
            # 0/1/6-call PMU control adds 0.087778 us per Muls on both slots.
            # This is an aggregate Vector active-time increment, not a PTO
            # CCE call or an isolated instruction latency.
            cycles = round(158 * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "vmuls_fa_runtime_count_mask_a3_calibrated_analogue",
                "pipe": "v", "cycles": cycles,
                "args": {"count": count, "scalar": "sm_scale",
                         "slot_byte_offset": destination.byte_offset,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
        if (
            isinstance(helper, Mapping)
            and helper.get("form") == "count_mask"
            and isinstance(count, int)
            and 4096 <= count <= 32768
            and count % 64 == 0
            and metadata.get("tail_kind") is None
            and metadata.get("scalar") == 2.0
            and getattr(getattr(destination, "scope", None), "value", None) == "ub"
            and getattr(getattr(source, "scope", None), "value", None) == "ub"
            and str(getattr(destination, "dtype", "")).lower() == "float32"
            and str(getattr(source, "dtype", "")).lower() == "float32"
            and getattr(destination, "shape", None) == (count,)
            and getattr(source, "shape", None) == (count,)
            and getattr(destination, "strides_bytes", None) is None
            and getattr(source, "strides_bytes", None) is None
            and getattr(destination, "buffer", None) == getattr(source, "buffer", None)
            and getattr(destination, "core_id", None) == getattr(source, "core_id", None)
            and getattr(destination, "byte_offset", None) == 0
            and getattr(source, "byte_offset", None) == 0
        ):
            # PMU at 1.8 GHz: delta Vector active cycles ≈ 34 + 17/16
            # cycles per 64 elements. Convert to the 1.85-GHz simulator base.
            hardware_cycles = 34 + 17 * count / 1024
            cycles = round(hardware_cycles * _PTO_CCE_FREQUENCY_HZ / 1_800_000_000)
            return ({
                "name": "vmuls_count_mask_calibrated_analogue",
                "pipe": "v",
                "cycles": cycles,
                "args": {"count": count, "scalar": 2.0,
                         "basis": "A3 PMU Vector active-time increment"},
            },)
    # common.h::copy_l1_to_bt emits one 64-byte-granular DataCopy. The
    # float32 TileLang/PTO path lowers to one copy_cbuf_to_bt call.
    if operation.lower() == "copy_l1_to_bt" and normalized_pipe == "mte1":
        source, destination = metadata.get("src"), metadata.get("dst")
        details = metadata.get("copy")
        byte_count = metadata.get("transfer_bytes")
        if (
            not isinstance(details, Mapping)
            or details.get("layout") != "linear"
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or byte_count % 64
            or byte_count // 64 > 65535
            or getattr(getattr(source, "scope", None), "value", None) != "l1"
            or getattr(getattr(destination, "scope", None), "value", None) != "bt"
            or str(getattr(source, "dtype", "")).lower() != "float32"
            or str(getattr(destination, "dtype", "")).lower() != "float32"
            or not isinstance(details.get("valid_elements"), int)
            or details["valid_elements"] * 4 != byte_count
            or getattr(source, "shape", None) != (details["valid_elements"],)
            or getattr(destination, "shape", None) != (details["valid_elements"],)
            or getattr(source, "byte_offset", -1) % 64
            or getattr(destination, "byte_offset", -1) % 64
        ):
            return None
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_L1_TO_BT_GIB_S
                     * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        return ({
            "name": "copy_cbuf_to_bt", "pipe": "mte1", "cycles": cycles,
            "args": {"conv_control": 0, "n_burst": 1,
                     "len_burst": byte_count // 64,
                     "src_stride": 0, "dst_stride": 0},
        },)
    if normalized_pipe == "m" and operation.lower() in {"mma", "gemm_v0.mma"}:
        details = metadata.get("mma")
        dtype = str(getattr(metadata.get("lhs"), "dtype", "")).lower()
        if not isinstance(details, Mapping) or dtype not in _DTYPE_BYTES:
            return None
        rows, inner = details.get("rows"), details.get("inner")
        cols = details.get("n_actual", details.get("cols"))
        if not all(isinstance(value, int) and value > 0 for value in (rows, inner, cols)):
            return None
        itemsize = _DTYPE_BYTES[dtype]
        if itemsize not in {1, 2, 4}:
            return None
        repeats = _ceil_div(rows, 16) * _ceil_div(inner, 32 // itemsize) * _ceil_div(cols, 16)
        cycles = 6 + (2 if itemsize == 4 else 1) * repeats
        return ({
            "name": "mad",
            "pipe": "m",
            "cycles": cycles,
            "args": {"m": rows, "k": inner, "n": cols, "dtype": dtype},
        },)

    # common.h::tail_binary takes one explicit-mask Add/Sub/Mul branch when the
    # logical columns are shorter than the UB row and one row fits a vector
    # repeat. CANN emits one vector binary call (mask=valid_cols, repeat=valid_rows,
    # repeat_stride=physical_cols/block_elems), unlike the count-mask path.
    # PTO CCE's one-call instruction scope charges head + repeat + drain.
    vector_costs = {
        "add": ("vadd", 14, 1, 18),
        "sub": ("vsub", 13, 2, 18),
        "mul": ("vmul", 13, 2, 19),
    }
    vector_cost = vector_costs.get(operation.lower())
    if vector_cost is not None and normalized_pipe == "v":
        if (
            metadata.get("tail_kind") != "tail_binary"
            or not str(metadata.get("lowered_operation", "")).lower().endswith("tail_binary")
        ):
            return None
        tail = metadata.get("tail")
        operands = (metadata.get("dst"), metadata.get("lhs"), metadata.get("rhs"))
        if not isinstance(tail, Mapping):
            return None
        dtype = str(getattr(operands[0], "dtype", "")).lower()
        if dtype not in {"float16", "float32"}:
            return None
        itemsize = _DTYPE_BYTES[dtype]
        rows, cols, physical_cols = (
            tail.get("valid_rows"), tail.get("valid_cols"), tail.get("physical_cols")
        )
        block_elements = 32 // itemsize
        vector_elements = 256 // itemsize
        if (
            not all(isinstance(value, int) and value > 0
                    for value in (rows, cols, physical_cols))
            or cols >= physical_cols
            or cols > vector_elements
            or rows > 255
            or physical_cols % block_elements
            or physical_cols // block_elements > 255
        ):
            return None
        for region in operands:
            if (
                getattr(getattr(region, "scope", None), "value", None) != "ub"
                or str(getattr(region, "dtype", "")).lower() != dtype
                or getattr(region, "shape", None) != (rows, cols)
                or getattr(region, "strides_bytes", None)
                != (physical_cols * itemsize, itemsize)
                or not isinstance(getattr(region, "byte_offset", None), int)
                or region.byte_offset % 32
            ):
                return None
        return ({
            "name": vector_cost[0],
            "pipe": "v",
            "cycles": vector_cost[1] + vector_cost[2] * rows + vector_cost[3],
            "args": {
                "mask": cols, "repeat": rows,
                "block_stride": 1, "repeat_stride": physical_cols // block_elements,
                "dtype": dtype,
            },
        },)

    # TileLang's full RowMajor FP32->FP16/FP32 Fixpipe emits one CANN V220
    # copy_matrix_cc_to_gm. PTO's CCE mock exposes only the older packed
    # register ABI with a hard-coded 2-byte element cost. FP32 uses that
    # packed-ABI *cost analogue*, not its actual four-byte traffic: the
    # latter overstates measured A3 Fixpipe active time for FA. Neither
    # mapping is a literal replay of the CANN call.
    if operation.lower() == "copy_l0c_to_gm" and normalized_pipe == "fix":
        details = metadata.get("copy")
        helper = metadata.get("copy_helper")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping) or not isinstance(helper, Mapping):
            return None
        if getattr(getattr(source, "scope", None), "value", None) != "l0c":
            return None
        if getattr(getattr(destination, "scope", None), "value", None) != "gm":
            return None
        if str(getattr(source, "dtype", "")).lower() != "float32":
            return None
        destination_dtype = str(getattr(destination, "dtype", "")).lower()
        if destination_dtype not in {"float16", "float32"}:
            return None
        shape = details.get("destination_shape")
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 and value % 16 == 0
                       for value in shape)
            or details.get("source_shape") != shape
            or helper.get("physical_shape") != shape
            or details.get("destination_cols") != shape[1]
            or details.get("source_layout") != "l0c"
            or details.get("destination_layout") != "row_major"
            or details.get("relu") is not False
            or details.get("unit_flag") != 0
            or shape[0] > 65535
            or shape[1] > 4095
        ):
            return None
        byte_count = shape[0] * shape[1] * 2
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_L0C_TO_GM_GIB_S
                     * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        return ({
            "name": "copy_matrix_cc_to_gm",
            "pipe": "fix",
            "cycles": cycles,
            "args": {
                "m": shape[0], "n": shape[1], "output_dtype": destination_dtype,
                "mapping": (
                    "packed-abi-bandwidth-analogue"
                    if destination_dtype == "float32" else "bandwidth-analogue"
                ),
            },
        },)

    # common.h::copy_gm_to_ub uses one DataCopyPad in the full, unpadded
    # branch. CANN dav_c220 lowers FP16/FP32 to one corresponding
    # copy_gm_to_ubuf_align_b16/b32(blockCount, blockLenBytes, ...).
    if operation.lower() == "copy_gm_to_ub" and normalized_pipe == "mte2":
        details = metadata.get("copy")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping):
            return None
        if getattr(getattr(source, "scope", None), "value", None) != "gm":
            return None
        if getattr(getattr(destination, "scope", None), "value", None) != "ub":
            return None
        dtype = str(getattr(source, "dtype", "")).lower()
        if dtype not in {"float16", "float32"}:
            return None
        if str(getattr(destination, "dtype", "")).lower() != dtype:
            return None
        itemsize = _DTYPE_BYTES[dtype]
        rows, cols = details.get("valid_rows"), details.get("valid_cols")
        if not all(isinstance(value, int) and value > 0 for value in (rows, cols)):
            return None
        if (
            details.get("physical_rows") != rows
            or details.get("physical_cols") != cols
            or not isinstance(details.get("stride_n"), int)
            or (rows > 1 and details["stride_n"] < cols)
            or not isinstance(metadata.get("copy_helper"), Mapping)
            or metadata["copy_helper"].get("left_pad") != 0
            or (cols * itemsize) % 32
        ):
            return None
        byte_count = rows * cols * itemsize
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_GM_TO_UB_GIB_S * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        return ({
            "name": "copy_gm_to_ubuf_align_b16" if itemsize == 2
                    else "copy_gm_to_ubuf_align_b32",
            "pipe": "mte2",
            "cycles": cycles,
            "args": {"n_burst": rows, "len_burst_bytes": cols * itemsize},
        },)

    # common.h::copy_ub_to_gm issues one DataCopyPad in this branch.  The
    # dav_c220 FP16/FP32 overload emits one copy_ubuf_to_gm_align_b16/b32.
    if operation.lower() == "copy_ub_to_gm" and normalized_pipe == "mte3":
        details = metadata.get("copy")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping):
            return None
        if getattr(getattr(source, "scope", None), "value", None) != "ub":
            return None
        if getattr(getattr(destination, "scope", None), "value", None) != "gm":
            return None
        dtype = str(getattr(source, "dtype", "")).lower()
        if dtype != str(getattr(destination, "dtype", "")).lower():
            return None
        itemsize = _DTYPE_BYTES.get(dtype)
        if itemsize not in {2, 4}:
            return None
        rows, cols = details.get("valid_rows"), details.get("valid_cols")
        if not all(isinstance(value, int) and value > 0 for value in (rows, cols)):
            return None
        if (
            details.get("physical_rows") != rows
            or details.get("physical_cols") != cols
            or not isinstance(details.get("stride_n"), int)
            or (rows > 1 and details["stride_n"] < cols)
            or (cols * itemsize) % 32
        ):
            return None
        byte_count = rows * cols * itemsize
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_UB_TO_GM_GIB_S * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        # Exact FA-shaped full FP16 copies retain their modern helper controls
        # in bridge metadata: stride_n, valid rows/cols, physical rows/cols.
        # Device5 A3 PMU controls of 64 x 128 measured mean MTE3 active time
        # 0.177917 us at 1.8 GHz, versus isolated PTO's 149 cycles: +171 cycles
        # after rounding. The independent 128 x 128 control was held out.
        # This is an aggregate active-time analogue, not an instruction latency.
        arguments = metadata.get("arguments")
        if (
            dtype == "float16" and rows in (64, 128) and cols == 128
            and details.get("stride_n") == 128
            and "pad_value" not in details
            and metadata.get("copy_helper") in (None, {})
            and metadata.get("copy_helper_phase") is None
            and isinstance(arguments, tuple) and len(arguments) == 7
            and arguments[2:] == (128, rows, 128, rows, 128)
            and all(
                getattr(region, "shape", None) == (rows, 128)
                and getattr(region, "strides_bytes", None) == (256, 2)
                and type(getattr(region, "byte_offset", None)) is int
                and region.byte_offset >= 0 and region.byte_offset % 32 == 0
                for region in (source, destination)
            )
        ):
            return ({
                "name": "copy_ubuf_to_gm_fa_shape_a3_calibrated_analogue",
                "pipe": "mte3", "cycles": cycles + 171,
                "args": {
                    "n_burst": rows, "len_burst_bytes": 256,
                    "src_stride": 0, "dst_stride": 0,
                    "pto_isolated_cycles": cycles, "extra_cycles": 171,
                    "basis": "A3 PMU MTE3 aggregate active-time analogue",
                    "control": "bench_pto_cce_fa_ub_gm_copy_a3 rows=64 FP16",
                    "control_frequency_mhz": 1800, "holdout_rows": 128,
                    "mapping": "aggregate-active-time-analogue-not-instruction-latency",
                },
            },)
        return ({
            "name": "copy_ubuf_to_gm_align_b16" if itemsize == 2 else "copy_ubuf_to_gm_align_b32",
            "pipe": "mte3",
            "cycles": cycles,
            "args": {"n_burst": rows, "len_burst_bytes": cols * itemsize},
        },)

    # CATLASS A2 zN->zZ/nZ FP16 copies issue one LoadData2D per 16-row
    # outer fractal, not one LoadData for the entire TileLang helper.
    l1_to_l0_operations = {
        "copy_l1_to_l0a": True, "copy_l1_to_l0b": False,
        "gemm_v0.load_a": True, "gemm_v0.load_b": False,
    }
    if operation.lower() in l1_to_l0_operations and normalized_pipe == "mte1":
        details = metadata.get("copy")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping):
            return None
        to_a = l1_to_l0_operations[operation.lower()]
        expected_scope = "l0a" if to_a else "l0b"
        if getattr(getattr(source, "scope", None), "value", None) != "l1":
            return None
        if getattr(getattr(destination, "scope", None), "value", None) != expected_scope:
            return None
        if str(getattr(source, "dtype", "")).lower() != "float16":
            return None
        if str(getattr(destination, "dtype", "")).lower() != "float16":
            return None
        shape = details.get("destination_shape")
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 and value % 16 == 0 for value in shape)
            or details.get("destination_layout") != expected_scope
        ):
            return None
        if operation.lower().startswith("gemm_v0."):
            source_shape = details.get("source_shape")
            origin = details.get("source_origin")
            if (
                not isinstance(metadata.get("gemm_stage"), Mapping)
                or details.get("source_layout") != "zn"
                or details.get("source_window_shape") != shape
                or details.get("transpose_after_slice") is not False
                or details.get("source_region_axis") != 1
                or not isinstance(source_shape, tuple)
                or len(source_shape) != 2
                or not isinstance(origin, tuple)
                or len(origin) != 2
                or not all(isinstance(value, int) and value >= 0 and value % 16 == 0
                           for value in origin)
                or not all(isinstance(value, int) and value > 0 and value % 16 == 0
                           for value in source_shape)
                or any(origin[axis] + shape[axis] > source_shape[axis]
                       for axis in (0, 1))
            ):
                return None
            if (
                details.get("source_window_direct") is not True
                and not _gemm_zn_col_slabs_match(
                    metadata, source, source_shape, origin, shape
                )
            ):
                return None
        elif (
            details.get("source_shape") != shape
            or details.get("source_origin") != (0, 0)
            or details.get("source_layout") != "zN"
            or details.get("transpose") is not False
        ):
            return None
        outer_rows, repeats = shape[0] // 16, shape[1] // 16
        bytes_per_call = repeats * 16 * 32
        bandwidth = _PTO_CCE_L1_TO_L0A_GIB_S if to_a else _PTO_CCE_L1_TO_L0B_GIB_S
        cycles = int(bytes_per_call / _PTO_CCE_GIB / bandwidth * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        name = "load_cbuf_to_ca" if to_a else "load_cbuf_to_cb"
        return tuple({
            "name": name,
            "pipe": "mte1",
            "cycles": cycles,
            "args": {"outer_row": row, "repeat_times": repeats},
        } for row in range(outer_rows))

    # common.h::copy_gm_to_l1: a complete FP16 zN tile uses one linear
    # DataCopy when it is 16 columns wide and rows are contiguous; otherwise
    # CATLASS issues one Nd2Nz DataCopy if the GM row stride fits the intrinsic.
    if operation.lower() == "copy_gm_to_l1" and normalized_pipe == "mte2":
        details = metadata.get("copy")
        source, destination = metadata.get("src"), metadata.get("dst")
        if not isinstance(details, Mapping) or details.get("layout") != "zN":
            return None
        if getattr(getattr(source, "scope", None), "value", None) != "gm":
            return None
        if getattr(getattr(destination, "scope", None), "value", None) != "l1":
            return None
        if str(getattr(source, "dtype", "")).lower() != "float16":
            return None
        if str(getattr(destination, "dtype", "")).lower() != "float16":
            return None
        rows, cols = details.get("valid_rows"), details.get("valid_cols")
        source_cols = details.get("source_cols")
        if not all(isinstance(value, int) and value > 0 for value in (rows, cols, source_cols)):
            return None
        if (
            details.get("physical_rows") != rows
            or details.get("physical_cols") != cols
            or details.get("need_clear") is not False
            or (metadata.get("copy_helper") or {}).get("helper_clear") is True
            or rows % 16
            or cols % 16
            or source_cols < cols
            or source_cols >= 65536
        ):
            return None
        byte_count = rows * cols * 2
        cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_GM_TO_L1_GIB_S * _PTO_CCE_FREQUENCY_HZ)
        if cycles <= 0:
            return None
        linear_fast_path = cols == source_cols == 16
        return ({
            "name": "copy_gm_to_cbuf" if linear_fast_path else "copy_gm_to_cbuf_multi_nd2nz_b16",
            "pipe": "mte2",
            "cycles": cycles,
            "args": ({"n_burst": 1, "len_burst": byte_count // 32} if linear_fast_path else {
                "nd_num": 1, "n_value": rows, "d_value": cols,
                "src_d_value": source_cols,
            }),
        },)

    # common.h::copy_gm_to_l1_linear takes this branch as one DataCopy with
    # blockCount=valid_rows and blockLen=valid_cols*itemsize/32.  Misaligned
    # tails instead clear L1 and issue per-row copies, so are excluded here.
    if "copy_gm_to_l1_linear" not in operation.lower() or normalized_pipe != "mte2":
        return None
    details = metadata.get("copy")
    source = metadata.get("src")
    destination = metadata.get("dst")
    if not isinstance(details, Mapping) or details.get("layout") != "row_major":
        return None
    if getattr(getattr(source, "scope", None), "value", None) != "gm":
        return None
    if getattr(getattr(destination, "scope", None), "value", None) != "l1":
        return None
    dtype = str(getattr(source, "dtype", "")).lower()
    itemsize = _DTYPE_BYTES.get(dtype)
    rows, cols = details.get("valid_rows"), details.get("valid_cols")
    source_cols = details.get("source_cols")
    if (
        itemsize not in {2, 4}
        or dtype != str(getattr(destination, "dtype", "")).lower()
        or not all(isinstance(value, int) and value > 0
                   for value in (rows, cols, source_cols))
    ):
        return None
    # The helper casts DataCopyParams blockCount, blockLen and srcStride to
    # uint16. A sub-32B GM row gap is truncated rather than transferred, so
    # only treat representable strides as the intended one-copy path. Limit
    # blockCount to the conservative hardware burst range until a larger
    # TileLang DataCopy path is verified.
    row_bytes = cols * itemsize
    gap_bytes = (source_cols - cols) * itemsize
    if (
        source_cols < cols
        or rows > 4095
        or row_bytes % 32
        or row_bytes // 32 > 65535
        or (rows > 1 and (gap_bytes % 32 or gap_bytes // 32 > 65535))
    ):
        return None
    byte_count = rows * cols * itemsize
    cycles = int(byte_count / _PTO_CCE_GIB / _PTO_CCE_GM_TO_L1_GIB_S * _PTO_CCE_FREQUENCY_HZ)
    if cycles <= 0:
        return None
    return ({
        "name": "copy_gm_to_cbuf",
        "pipe": "mte2",
        "cycles": cycles,
        "args": {"n_burst": rows, "len_burst": cols * itemsize // 32},
    },)


def _pto_mma_cycles(metadata: Mapping[str, Any]) -> int | None:
    """Port ``TryEstimateMadCycles`` for TileLang's concrete ``mma`` metadata."""
    details = metadata.get("mma")
    lhs = metadata.get("lhs")
    if not isinstance(details, Mapping):
        return None
    rows = details.get("rows")
    inner = details.get("inner")
    cols = details.get("cols")
    dtype = str(getattr(lhs, "dtype", "")).lower()
    if not all(isinstance(value, int) and value > 0 for value in (rows, inner, cols)):
        return None
    # PTO formula_backend_compute.hpp: 16x16 output tiles, a 32-byte K
    # fractal, head=6, and fp32/fp16 repeat costs of 2/1 respectively.
    if dtype in {"float32", "float"}:
        element_bytes, repeat_cycles = 4, 2
    elif dtype in {"float16", "half"}:
        element_bytes, repeat_cycles = 2, 1
    else:
        return None
    k_tile = 32 // element_bytes
    repeats = _ceil_div(rows, 16) * _ceil_div(inner, k_tile) * _ceil_div(cols, 16)
    return 6 + repeat_cycles * repeats


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


_DEVICE_PROFILES = {
    "A2": DeviceProfile("A2", cube_core_count=20, vector_core_count=40,
                        vector_lanes_per_cube=2),
    "A3": DeviceProfile("A3", cube_core_count=20, vector_core_count=40,
                        vector_lanes_per_cube=2),
}


def normalize_platform(platform: str) -> str:
    """Normalize and validate a simulator platform name."""
    normalized = platform.strip().upper()
    if normalized not in _DEVICE_PROFILES:
        supported = ", ".join(sorted(_DEVICE_PROFILES))
        raise SimulatorConfigError(
            f"unsupported simulator platform {platform!r}; supported platforms: {supported}"
        )
    return normalized


def get_device_profile(platform: str) -> DeviceProfile:
    """Return the immutable device topology for A2 or A3."""
    return _DEVICE_PROFILES[normalize_platform(platform)]


def default_timing_profile(platform: str) -> TimingProfile:
    """Return the single default PTO-derived relative timing profile."""
    return pto_fallback_timing_profile(platform)


def pto_fallback_timing_profile(platform: str) -> TimingProfile:
    """Return the PTO perf-sim-derived relative timing profile.

    The formulas are ported from ``pto/costmodel/perf_sim/costmodel_provider.hpp``.
    They are deliberately labeled as derived rather than measured hardware data.
    """
    return TimingProfile(
        platform=normalize_platform(platform),
        calibration="pto-perf-sim-derived-fallback",
        estimator="pto-fallback",
    )


def pto_cce_mock_timing_profile(platform: str) -> TimingProfile:
    """Opt-in A3 CCE-mock cost model with fallback for unmatched helpers."""
    normalized = normalize_platform(platform)
    if normalized != "A3":
        raise SimulatorConfigError("pto-cce-mock timing is currently supported only on A3")
    return TimingProfile(
        platform=normalized,
        calibration="pto-a3-cce-mock-partial-with-narrow-pmu-analogue",
        estimator="pto-cce-mock",
    )
