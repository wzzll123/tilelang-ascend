# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Parity with CANN/PyPTO fast CostModel (not hardware accuracy tests)."""

from types import SimpleNamespace

import pytest

from tilelang.simulator.profile import TimingProfile, default_timing_profile
from tilelang.simulator.pypto_timing import estimate_cycles


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_default_selects_measured_a2_and_uncalibrated_a3(platform):
    expected = "measured-linear" if platform == "A2" else "pypto-fast"
    assert default_timing_profile(platform).estimator == expected


def _pypto_profile(platform):
    return TimingProfile(platform=platform, estimator="pypto-fast")


@pytest.mark.parametrize("op,dtype,expected", [
    ("COPY_IN", "float16", 1748),
    ("ADD", "float16", 353),
    ("EXP", "float32", 510),
    ("A_MUL_B", "float16", 535),
])
def test_reference_costs(op, dtype, expected):
    assert estimate_cycles(op, [(128, 128)], dtype) == expected


def test_shape_count_and_minimum_operand_shape():
    single = estimate_cycles("ADD", [(16, 32)], "float16")
    assert estimate_cycles("ADD", [(3, 16, 32)], "float16") == 3 * single
    assert estimate_cycles("ADD", [(128, 128), (16, 32)], "float16") == single
    assert estimate_cycles("COPY_IN", [(3, 128, 128)], "float16") == 3 * 348 + 1400


def test_fallback_and_dtype_ratio():
    assert estimate_cycles("unknown", [(128, 128)], "float16") == 265
    assert estimate_cycles("UB_EXP", [(128, 128)], "float32") == 523
    assert estimate_cycles("MUL", [(128, 128)],
                           "int32") < estimate_cycles("MUL", [(128, 128)], "float32")
    assert estimate_cycles("EXP", [(128, 128)],
                           "float32") == estimate_cycles("EXP", [(128, 128)], "float16")


def test_mma_uses_pypto_not_old_repeat_formula():

    def region(shape):
        return SimpleNamespace(shape=shape, dtype="float16")

    metadata = {
        "lhs": region((128 * 128,)),
        "rhs": region((128 * 128,)),
        "dst": region((128 * 128,)),
        "mma": {
            "rows": 128,
            "cols": 128,
            "inner": 128,
            "init": True
        }
    }
    profile = _pypto_profile("A2")
    assert profile.estimate_task("mma", pipe="m", metadata=metadata) == 535
    metadata["mma"]["inner"] = 64
    assert profile.estimate_task(
        "mma", pipe="m", metadata=metadata) == estimate_cycles("A_MUL_B", [(128, 64), (64, 128),
                                                                           (128, 128)], "float16")


def test_operation_mapping_and_override():
    region = SimpleNamespace(shape=(128, 128), dtype="float16", scope="ub")
    profile = _pypto_profile("A2")
    assert profile.estimate_task(
        "tl::ascend::exp<half>", pipe="v", metadata={
            "src": region,
            "dst": region
        }) == 510
    assert profile.estimate_task("set_flag", pipe="mte2", metadata={}) == 1
    assert profile.estimate_task("unknown", pipe="v", metadata={}) == 1


# Frozen by compiling upstream PostSimulateForPass with Clang, without FMA.
# Each tuple is (opcode, FP16 cycles, FP32 cycles), for shape (128, 128).
@pytest.mark.parametrize("op,half_cycles,float_cycles", [
    ("COPY_IN", 1748, 2090),
    ("L1_COPY_IN", 348, 690),
    ("UB_COPY_IN", 348, 690),
    ("COPY_OUT", 735, 1166),
    ("L0C_COPY_OUT", 435, 866),
    ("UB_COPY_OUT", 435, 866),
    ("ADD", 353, 668),
    ("SUB", 259, 474),
    ("MUL", 243, 458),
    ("DIV", 552, 552),
    ("ADDS", 145, 245),
    ("SUBS", 145, 245),
    ("RANGE", 145, 245),
    ("MULS", 155, 280),
    ("DIVS", 155, 280),
    ("SQRT", 526, 526),
    ("EXP", 510, 510),
    ("NEG", 155, 280),
    ("LN", 510, 510),
    ("WHERE_TT", 353, 668),
    ("WHERE_TS", 353, 668),
    ("WHERE_ST", 353, 668),
    ("WHERE_SS", 353, 668),
    ("RECIPROCAL", 150, 279),
    ("ROWSUMLINE", 2058, 3970),
    ("ROWMAX_SINGLE", 2061, 4104),
    ("ROWMIN_SINGLE", 2061, 4104),
    ("EXPAND", 105, 181),
    ("CAST", 151, 271),
    ("TRANSPOSE_MOVEOUT", 68, 68),
    ("L1_TO_L0B", 294, 564),
    ("L1_TO_L0A", 178, 326),
    ("L1_TO_L0Bt", 275, 520),
    ("A_MUL_B", 535, 1047),
    ("A_MUL_Bt", 535, 1048),
    ("A_MULACC_B", 533, 1044),
    ("A_MULACC_Bt", 533, 1045),
    ("PAIRMAX", 2528, 5056),
    ("PAIRSUM", 2567, 5134),
    ("MAXIMUM", 40466, 80931),
])
def test_all_coefficients_against_upstream_cpp(op, half_cycles, float_cycles):
    assert estimate_cycles(op, [(128, 128)], "float16") == half_cycles
    assert estimate_cycles(op, [(128, 128)], "float32") == float_cycles


@pytest.mark.parametrize("src_scope,dst_scope,pipe,opcode,original", [
    ("gm", "ub", "mte2", "UB_COPY_IN", "COPY_IN"),
    ("gm", "l1", "mte2", "L1_COPY_IN", "COPY_IN"),
    ("ub", "gm", "mte3", "UB_COPY_OUT", "COPY_OUT"),
    ("l0c", "gm", "fix", "L0C_COPY_OUT", "COPY_OUT"),
    ("l1", "l0a", "mte1", "L1_TO_L0A", "L1_TO_L0A"),
    ("l1", "l0b", "mte1", "L1_TO_L0B", "L1_TO_L0B"),
])
def test_copy_scope_mapping(src_scope, dst_scope, pipe, opcode, original):

    def region(scope):
        return SimpleNamespace(shape=(128, 128), dtype="float16", scope=scope)

    metadata = {"src": region(src_scope), "dst": region(dst_scope)}
    assert _pypto_profile("A2").estimate_task(
        "copy", pipe=pipe, metadata=metadata) == (
            estimate_cycles(opcode, [(128, 128)] * 2, "float16", original_opcode=original))


def test_explicit_override_and_missing_metadata():
    from tilelang.simulator.profile import TimingProfile

    profile = TimingProfile(platform="A2", estimator="pypto-fast", operation_cycles={"mma": 99})
    assert profile.estimate_task("mma", pipe="m", metadata={}) == 99
    assert estimate_cycles("ADD", [], "float16") is None
    assert estimate_cycles("ADD", [(128, 128)], "unknown") is None


def test_upstream_edge_cases_and_case_sensitive_opcode():
    assert estimate_cycles("unknown", [(1,)], "float16") == 11
    assert estimate_cycles("UB_ADD", [(16, 16)], "float16") == 13
    assert estimate_cycles("ADD", [(0, 128)],
                           "float16") == estimate_cycles("ADD", [(256, 128)], "float16")
    region = SimpleNamespace(shape=(128, 128), dtype="float16")
    assert _pypto_profile("A2").estimate_task(
        "A_MUL_Bt", pipe="m", metadata={
            "lhs": region,
            "dst": region
        }) == estimate_cycles("A_MUL_Bt", [(128, 128)], "float16")
