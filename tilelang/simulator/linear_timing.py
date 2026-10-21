# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Hardware-calibration primitives; all times are ns, never assumed core cycles.

These fits do not silently replace the default estimator. A measurement family
must separately identify operation, dtype, path, geometry and measurement mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, prod
from collections.abc import Mapping

_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int16": 2, "int8": 1, "uint8": 1}


def _input_bytes(dtype):
    parts = dtype.split("_to_")
    if len(parts) > 2 or any(part not in _BYTES for part in parts):
        raise ValueError(f"unsupported dtype: {dtype}")
    return _BYTES[parts[0]]


def timing_feature(operation: str, shape: tuple[int, ...], dtype: str) -> int:
    """Return padded MNK for MMA, or one processed tensor's bytes otherwise.

    Copy uses the bytes actually transferred, not the sum of read/write bytes.
    Reduction shape describes its INPUT, not its smaller output. Layout padding
    and repeat rounding for vector helpers belong in the measurement family.
    """
    itemsize = _input_bytes(dtype)
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("shape must have positive concrete integer dimensions")
    if operation in {"mma", "mma_init"}:
        if len(shape) != 3:
            raise ValueError("MMA requires (M, N, K)")
        m, n, k = shape
        k_tile = 32 // itemsize
        return ((m + 15) // 16 * 16 * ((n + 15) // 16 * 16) * ((k + k_tile - 1) // k_tile * k_tile))
    return prod(shape) * itemsize


@dataclass(frozen=True)
class LinearCost:
    """A fit with an explicit measured feature domain and evidence identifier."""

    slope_ns: float
    intercept_ns: float
    min_feature: float
    max_feature: float
    feature: str
    evidence: str

    def __post_init__(self):
        numbers = (self.slope_ns, self.intercept_ns, self.min_feature, self.max_feature)
        if not all(isfinite(v) for v in numbers):
            raise ValueError("linear parameters must be finite")
        if self.slope_ns < 0 or not 0 < self.min_feature < self.max_feature:
            raise ValueError("invalid slope or measured domain")
        if self.slope_ns * self.min_feature + self.intercept_ns <= 0:
            raise ValueError("predicted time must be positive on the measured domain")
        if self.feature not in {"bytes", "padded_mnk"}:
            raise ValueError("unsupported feature")
        if not self.evidence.strip():
            raise ValueError("measurement evidence is required")

    def predict_ns(self, x: float, *, extrapolate: bool = False) -> float:
        if not isfinite(x) or x <= 0:
            raise ValueError("feature must be finite and positive")
        if not extrapolate and not self.min_feature <= x <= self.max_feature:
            raise ValueError("feature outside measured domain")
        result = self.slope_ns * x + self.intercept_ns
        if result <= 0:
            raise ValueError("nonpositive predicted time")
        return result

    def relative_error(self, x: float, observed_ns: float) -> float:
        if not isfinite(observed_ns) or observed_ns <= 0:
            raise ValueError("observation must be finite and positive")
        return self.predict_ns(x) / observed_ns - 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> LinearCost:
        return cls(**value)


def fit_linear_cost(points, *, feature: str, evidence: str) -> LinearCost:
    """OLS on caller-selected TRAIN observations only; no holdout leakage."""
    points = tuple(points)
    if len(points) < 2:
        raise ValueError("at least two training observations are required")
    if any(not isfinite(x) or not isfinite(y) or x <= 0 or y <= 0 for x, y in points):
        raise ValueError("measurements must be finite and positive")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    variance = sum((x - mean_x)**2 for x, _ in points)
    if not variance:
        raise ValueError("at least two distinct feature values are required")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return LinearCost(slope, mean_y - slope * mean_x, min(x for x, _ in points),
                      max(x for x, _ in points), feature, evidence)


class LinearCalibration:
    """Read measured families without guessing missing operation/dtype costs.

    This API returns ns and deliberately does not convert to assumed hardware
    cycles. It also does not reinterpret two_slot/chain as latency/issue interval.
    """

    def __init__(self, report: dict, *, platform: str = "A2"):
        if report.get("schema") != "tilelang-linear-cost-calibration-v1":
            raise ValueError("unsupported calibration schema")
        if report.get("unit") != "ns" or report.get("platform_measured") != platform:
            raise ValueError("calibration unit/platform does not match measured hardware")
        self.costs = {}
        for family in report["families"]:
            if family["status"] != "fitted":
                continue
            key = (family["operation"], family["dtype"], family["geometry"],
                   family["measurement_mode"])
            if key in self.costs:
                raise ValueError(f"duplicate calibration family: {key}")
            self.costs[key] = LinearCost.from_dict(family["cost"])

    def estimate_ns(self,
                    operation: str,
                    shape: tuple[int, ...],
                    dtype: str,
                    *,
                    measurement_mode: str = "two_slot",
                    geometry: str | None = None) -> float:
        feature = timing_feature(operation, shape, dtype)
        if geometry is None:
            geometry = (f"last_axis_width_{shape[-1]}"
                        if operation.startswith("reduce_") else f"row_elements_{shape[-1]}"
                        if operation == "copy_l0c_gm" and len(shape) == 2 else "contiguous")
        key = (operation, dtype, geometry, measurement_mode)
        if key not in self.costs:
            raise ValueError(f"no measured calibration for {key}")
        cost = self.costs[key]
        expected_feature = "padded_mnk" if operation in {"mma", "mma_init"} else "bytes"
        if cost.feature != expected_feature:
            raise ValueError("calibration feature does not match operation")
        return cost.predict_ns(feature)

    def estimate_task_ns(self, operation: str, metadata: Mapping) -> float:
        """Estimate a bridged tile using only an exact measured family match."""
        op, shape, dtype = task_feature(operation, metadata)
        geometry = None
        if op == "copy_l0c_gm":
            stride = metadata.get("copy", {}).get("destination_cols", shape[-1])
            if stride != shape[-1]:
                geometry = f"row_elements_{shape[-1]}_stride_elements_{stride}"
        if op.startswith("row_expand_"):
            src = metadata.get("src")
            strides = getattr(src, "strides_bytes", None)
            stride = strides[-2] if strides and len(
                strides) >= 2 else shape[-1] * _input_bytes(dtype)
            geometry = f"row_stride_bytes_{stride}"
        return self.estimate_ns(op, shape, dtype, geometry=geometry)


def task_feature(operation: str, metadata: Mapping) -> tuple[str, tuple[int, ...], str]:
    """Map bridge ABI fields, not flattened allocation extents, to measured inputs.

    Unsupported variants raise rather than borrowing a vaguely similar cost.
    These are descriptive family keys; no new compiler or simulator flag exists.
    """
    short = operation.split("<", 1)[0].rsplit("::", 1)[-1].rsplit(".", 1)[-1].lower()
    src = metadata.get("src", metadata.get("lhs", metadata.get("dst")))
    dst = metadata.get("dst")
    dtype = str(getattr(src, "dtype", "")).lower()
    dtype = {"half": "float16", "float": "float32"}.get(dtype, dtype)
    shape = getattr(src, "shape", None)
    if short == "mma":
        mma = metadata.get("mma", {})
        shape = tuple(mma.get(key) for key in ("rows", "cols", "inner"))
        # Existing measurements repeatedly accumulate (init=False).
        op = "mma_init" if mma.get("init", True) else "mma"
    elif short in {"reduce", "reduce_2d", "reduce_sum", "reduce_max"}:
        if metadata.get("reduce_axis", -1) != -1 or not metadata.get("clear", True):
            raise ValueError("no measured reduction for axis=0 or clear=False")
        op = metadata.get("reduce_kind", short)
        op = {"sum": "reduce_sum", "max": "reduce_max"}.get(op, op)
    elif "copy" in short:

        def scope(region):
            value = getattr(region, "scope", None)
            value = getattr(value, "value", value)
            return "gm" if value == "workspace" else value

        op = f"copy_{scope(src)}_{scope(dst)}"
        if op == "copy_l1_l0b" and metadata.get("copy", {}).get("transpose", False):
            op += "_transpose"
        if op == "copy_l0c_gm":
            details = metadata.get("copy", {})
            shape = details.get("destination_shape", shape)
        dst_dtype = str(getattr(dst, "dtype", "")).lower()
        dst_dtype = {"half": "float16", "float": "float32"}.get(dst_dtype, dst_dtype)
        if dst_dtype != dtype:
            # Keep the conversion family distinct. Its feature is source bytes.
            dtype = dtype + "_to_" + dst_dtype
        transferred = metadata.get("transfer_bytes")
        if transferred is not None and "_to_" not in dtype and op != "copy_l0c_gm":
            size = _BYTES.get(dtype)
            if type(transferred) is not int or not size or transferred % size:
                raise ValueError("copy requires a concrete integral byte count")
            shape = (transferred // size,)
    else:
        op = short
        dst_dtype = str(getattr(dst, "dtype", dtype)).lower()
        dst_dtype = {"half": "float16", "float": "float32"}.get(dst_dtype, dst_dtype)
        if dst_dtype != dtype:
            dtype += "_to_" + dst_dtype
    if shape is None:
        raise ValueError("missing logical timing shape")
    shape = tuple(shape)
    timing_feature("mma" if op in {"mma", "mma_init"} else op, shape, dtype)
    return op, shape, dtype


def measured_timing_profile(report: dict, *, platform: str = "A2"):
    """Build a strict single-duration profile on the existing scheduler grid.

    Two abstract timing ticks represent one ns, matching the existing trace
    conversion's 2000 ticks/us. This is a serialization grid, NOT a statement
    about physical AI Core frequency. Missing measured families raise.
    """
    from .profile import TimingProfile
    import hashlib
    import json

    model = LinearCalibration(report, platform=platform)
    fingerprint = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]

    class MeasuredProfile(TimingProfile):

        def estimate_task_with_source(self, operation, *, pipe, metadata):
            configured = self.operation_cycles.get(operation)
            if configured is not None:
                return configured, "override"
            if pipe.lower() == "s" or operation in {
                    "set_flag",
                    "wait_flag",
                    "auto_set_flag",
                    "auto_wait_flag",
                    "set_cross_flag",
                    "wait_cross_flag",
                    "auto_set_cross_flag",
                    "auto_wait_cross_flag",
                    "pipe_barrier",
                    "auto_barrier",
                    "barrier_all",
                    "sync_all",
            }:
                # Retain current scheduler bookkeeping; not calibrated signal latency.
                return 1, "bookkeeping"
            return max(1, ceil(model.estimate_task_ns(operation, metadata) * 2)), "measured"

        def estimate_task(self, operation, *, pipe, metadata):
            return self.estimate_task_with_source(operation, pipe=pipe, metadata=metadata)[0]

    return MeasuredProfile(
        platform=platform,
        estimator="measured-linear",
        calibration=f"measured-linear-{fingerprint}-ns-2ticks-per-ns")
