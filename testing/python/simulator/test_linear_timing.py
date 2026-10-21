# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Measured linear costs: shape features, honest domains and held-out validation."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tilelang.simulator.linear_timing import (
    LinearCalibration,
    LinearCost,
    fit_linear_cost,
    measured_timing_profile,
    task_feature,
    timing_feature,
)


def test_mma_counts_k_and_padding():
    assert timing_feature("mma", (128, 128, 256), "float16") == 128 * 128 * 256
    assert timing_feature("mma", (17, 31, 33), "float16") == 32 * 32 * 48
    with pytest.raises(ValueError, match="dtype"):
        timing_feature("mma", (16, 16, 16), "unknown")


def test_vector_bytes_are_one_tensor_not_sum_of_operands():
    assert timing_feature("add", (16, 512), "float32") == 32768
    assert timing_feature("reduce_sum", (16, 512), "float32") == 32768
    with pytest.raises(ValueError):
        timing_feature("add", (16, 0), "float32")


def test_fit_uses_train_only_and_reports_holdout():
    points = [(10, 25), (20, 45), (30, 65)]
    cost = fit_linear_cost(points, feature="bytes", evidence="sha256:example")
    assert cost.slope_ns == pytest.approx(2)
    assert cost.intercept_ns == pytest.approx(5)
    assert cost.predict_ns(25) == pytest.approx(55)
    assert cost.relative_error(25, 110) == pytest.approx(-0.5)
    assert cost.to_dict()["feature"] == "bytes"
    assert LinearCost.from_dict(cost.to_dict()) == cost


def test_no_silent_extrapolation_or_fake_measurements():
    cost = fit_linear_cost([(10, 25), (20, 45)], feature="bytes", evidence="raw.json")
    with pytest.raises(ValueError, match="domain"):
        cost.predict_ns(30)
    assert cost.predict_ns(30, extrapolate=True) == pytest.approx(65)
    for points in ([(10, 1)], [(10, 1), (10, 2)], [(10, 1), (20, float("nan"))], [(10, 2),
                                                                                  (20, 1)]):
        with pytest.raises(ValueError):
            fit_linear_cost(points, feature="bytes", evidence="raw.json")
    with pytest.raises(ValueError, match="evidence"):
        fit_linear_cost([(10, 1), (20, 2)], feature="bytes", evidence="")


def test_constant_cost_is_valid_and_negative_time_is_not():
    cost = fit_linear_cost([(10, 2), (20, 2)], feature="bytes", evidence="raw.json")
    assert cost.predict_ns(15) == 2
    with pytest.raises(ValueError):
        LinearCost(1, -100, 1, 10, "bytes", "raw.json")


def test_fitter_keeps_holdout_out_of_training_and_geometry_separate(tmp_path):
    fitter = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/fit_simulator_linear_costs.py"))
    records = []
    for rows, cols, split, ns in ((4, 512, "train", 20), (16, 512, "train", 80),
                                  (12, 512, "holdout", 1000), (16, 128, "holdout", 10)):
        records.append({
            "name": f"reduce_sum_{rows}_{cols}",
            "op": "reduce_sum",
            "shape": [rows, cols, 0],
            "dtype": "float32",
            "split": split,
            "correctness": "passed",
            "two_slot": {
                "ns_per_op": ns,
                "r_squared": 1
            },
            "chain": {
                "ns_per_op": ns,
                "r_squared": 1
            }
        })
    observations = tmp_path / "observations.json"
    observations.write_text(json.dumps(records))
    result = fitter["fit_observations"]([observations])
    families = result["families"]
    for family in families:
        if family["geometry"] == "last_axis_width_128":
            assert family["status"] == "insufficient_or_invalid_fit"
        else:
            assert family["cost"]["slope_ns"] == pytest.approx(20 / 8192)
            assert family["held_out_max_abs_relative_error"] == pytest.approx(0.94)
    with pytest.raises(ValueError, match="duplicate"):
        fitter["fit_observations"]([observations, observations])


def test_calibration_inference_never_borrows_other_dtype_or_platform():
    cost = fit_linear_cost([(16**3, 10), (128**3, 100)], feature="padded_mnk", evidence="raw.json")
    report = {
        "schema":
            "tilelang-linear-cost-calibration-v1",
        "unit":
            "ns",
        "platform_measured":
            "A2",
        "families": [{
            "operation": "mma",
            "dtype": "float16",
            "geometry": "contiguous",
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": cost.to_dict()
        }]
    }
    calibration = LinearCalibration(report)
    assert calibration.estimate_ns("mma", (64, 64, 128), "float16") > (
        calibration.estimate_ns("mma", (64, 64, 64), "float16"))
    with pytest.raises(ValueError, match="hardware"):
        LinearCalibration(report, platform="A3")
    with pytest.raises(ValueError, match="no measured"):
        calibration.estimate_ns("mma", (64, 64, 64), "bfloat16")
    with pytest.raises(ValueError, match="no measured"):
        calibration.estimate_ns("mma", (64, 64, 64), "float16", measurement_mode="latency")


def test_single_completion_measurements_are_not_renamed_issue_interval(tmp_path):
    fitter = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/fit_simulator_linear_costs.py"))
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps([{
            "name": f"add_{m}",
            "op": "add",
            "shape": [m, 512, 0],
            "dtype": "float32",
            "correctness": "passed",
            "split": "train",
            **{
                mode: {
                    "ns_per_op": m * 10,
                    "r_squared": 1,
                    "single_op_net_completion_ns": m * 10 + 20,
                    "single_op_counter_tick_ns": 20
                } for mode in ("two_slot", "chain")
            }
        } for m in (4, 16)]))
    result = fitter["fit_observations"]([observations])
    modes = {family["measurement_mode"] for family in result["families"]}
    assert modes == {"two_slot", "chain", "single_net_two_slot", "single_net_chain"}
    assert all(family["status"] == "fitted" for family in result["families"])


def test_bridge_abi_not_flat_allocation_controls_mma_and_copy_features():
    lhs = SimpleNamespace(dtype="float16", shape=(65536,), scope="l0a")
    details = {"rows": 64, "cols": 64, "inner": 256, "init": False}
    assert task_feature("mma", {"lhs": lhs, "mma": details}) == ("mma", (64, 64, 256), "float16")
    details["init"] = True
    assert task_feature("mma", {"lhs": lhs, "mma": details})[0] == "mma_init"
    src = SimpleNamespace(dtype="float32", shape=(65536,), scope="gm")
    dst = SimpleNamespace(dtype="float32", shape=(65536,), scope="ub")
    assert task_feature("tl::ascend::copy_gm_to_ub<float>", {
        "src": src,
        "dst": dst,
        "transfer_bytes": 8192
    }) == ("copy_gm_ub", (2048,), "float32")
    dst.dtype = "float16"
    assert task_feature("copy", {
        "src": src,
        "dst": dst
    }) == ("copy_gm_ub", (65536,), "float32_to_float16")
    assert timing_feature("copy_gm_ub", (128, 128), "float32_to_float16") == 65536


def test_unsupported_reduce_axis_is_not_borrowed_from_row_reduction():
    src = SimpleNamespace(dtype="float32", shape=(16, 512))
    with pytest.raises(ValueError, match="axis"):
        task_feature("reduce", {"src": src, "reduce_kind": "reduce_sum", "reduce_axis": 0})


def test_profile_uses_single_duration_grid_and_fails_on_unmeasured_operations():
    from tilelang.simulator.errors import SimulatorConfigError
    from tilelang.simulator.profile import TimingProfile

    cost = fit_linear_cost([(8192, 50), (32768, 160)], feature="bytes", evidence="raw.json")
    report = {
        "schema":
            "tilelang-linear-cost-calibration-v1",
        "unit":
            "ns",
        "platform_measured":
            "A2",
        "families": [{
            "operation": "add",
            "dtype": "float32",
            "geometry": "contiguous",
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": cost.to_dict()
        }]
    }
    region = SimpleNamespace(shape=(16, 512), dtype="float32")
    profile = measured_timing_profile(report)
    assert profile.estimate_task("add", pipe="v", metadata={"src": region}) == 320
    assert profile.estimate_task("set_flag", pipe="v", metadata={}) == 1
    with pytest.raises(ValueError, match="no measured"):
        profile.estimate_task("exp", pipe="v", metadata={"src": region})
    with pytest.raises(SimulatorConfigError, match="calibration"):
        TimingProfile("A2", estimator="measured-linear")


def test_fix_row_width_and_conversion_are_independent_measured_families():
    cost = fit_linear_cost([(16384, 100), (65536, 400)], feature="bytes", evidence="raw.json")
    report = {
        "schema":
            "tilelang-linear-cost-calibration-v1",
        "unit":
            "ns",
        "platform_measured":
            "A2",
        "families": [{
            "operation": "copy_l0c_gm",
            "dtype": "float32_to_float16",
            "geometry": "row_elements_128",
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": cost.to_dict()
        }]
    }
    model = LinearCalibration(report)
    src = SimpleNamespace(shape=(65536,), dtype="float32", scope="l0c")
    dst = SimpleNamespace(shape=(65536,), dtype="float16", scope="gm")
    metadata = {
        "src": src,
        "dst": dst,
        "copy": {
            "destination_shape": (64, 128),
            "destination_cols": 128
        }
    }
    assert model.estimate_task_ns("copy_l0c_to_gm", metadata) == pytest.approx(200)
    with pytest.raises(ValueError, match="no measured"):
        model.estimate_ns("copy_l0c_gm", (96, 96), "float32_to_float16")
    metadata["copy"]["destination_cols"] = 256
    with pytest.raises(ValueError, match="no measured"):
        model.estimate_task_ns("copy_l0c_to_gm", metadata)
    report["families"].append(
        dict(report["families"][0], geometry="row_elements_128_stride_elements_256"))
    assert LinearCalibration(report).estimate_task_ns("copy_l0c_to_gm",
                                                      metadata) == pytest.approx(200)


def test_transposed_mte1_copy_never_borrows_plain_copy_cost():
    src = SimpleNamespace(shape=(128, 128), dtype="float16", scope="l1")
    dst = SimpleNamespace(shape=(128, 128), dtype="float16", scope="l0b")
    assert task_feature("copy_l1_to_l0b", {
        "src": src,
        "dst": dst,
        "copy": {
            "transpose": True
        }
    })[0] == "copy_l1_l0b_transpose"


def test_fitter_does_not_hide_fix_geometry_outlier_in_shared_byte_fit(tmp_path):
    fitter = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/fit_simulator_linear_costs.py"))
    records = []
    for m, n, split, ns in ((32, 128, "train", 100), (128, 128, "train", 400),
                            (80, 128, "holdout", 250), (96, 96, "holdout", 1000)):
        records.append({
            "name": f"fix_{m}_{n}",
            "op": "copy_l0c_gm",
            "shape": [m, n, 128],
            "dtype": "float32",
            "split": split,
            "correctness": "passed",
            **{
                mode: {
                    "ns_per_op": ns,
                    "r_squared": 1
                } for mode in ("two_slot", "chain")
            }
        })
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(records))
    result = fitter["fit_observations"]([path])
    for family in result["families"]:
        if family["geometry"] == "row_elements_128":
            assert family["held_out_max_abs_relative_error"] == pytest.approx(0)
        else:
            assert family["geometry"] == "row_elements_96"
            assert family["status"] == "insufficient_or_invalid_fit"
            assert family["observations"][0]["observed_ns"] == 1000


def test_row_helper_uses_real_row_stride_instead_of_silent_contiguous_assumption():
    cost = fit_linear_cost([(4096, 40), (16384, 160)], feature="bytes", evidence="raw.json")
    report = {
        "schema":
            "tilelang-linear-cost-calibration-v1",
        "unit":
            "ns",
        "platform_measured":
            "A2",
        "families": [{
            "operation": "row_expand_mul_experiment",
            "dtype": "float32",
            "geometry": "row_stride_bytes_512",
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": cost.to_dict()
        }]
    }
    src = SimpleNamespace(shape=(64, 64), dtype="float32", strides_bytes=(512, 4))
    model = LinearCalibration(report)
    assert model.estimate_task_ns("row_expand_mul_experiment", {"src": src}) == 160
    src.strides_bytes = (256, 4)
    with pytest.raises(ValueError, match="no measured"):
        model.estimate_task_ns("row_expand_mul_experiment", {"src": src})


def test_packaged_default_marks_measured_fallback_override_and_bookkeeping():
    from dataclasses import replace
    from tilelang.simulator.measured_a2 import COSTS, INPUT_SHA256, calibration_report
    from tilelang.simulator.profile import default_timing_profile

    profile = default_timing_profile("A2")
    metadata = {
        "lhs": SimpleNamespace(dtype="float16", shape=(128, 128)),
        "mma": {
            "rows": 128,
            "cols": 128,
            "inner": 128,
            "init": False
        }
    }
    cycles, source = profile.estimate_task_with_source("mma", pipe="m", metadata=metadata)
    assert cycles == 569
    assert source == "measured"
    assert profile.estimate_task_with_source(
        "not_measured", pipe="v", metadata={}) == (1, "pypto-fallback")
    assert profile.estimate_task_with_source(
        "set_flag", pipe="m", metadata={}) == (1, "bookkeeping")
    overridden = replace(profile, operation_cycles={"mma": 77})
    assert overridden.estimate_task_with_source(
        "mma", pipe="m", metadata=metadata) == (77, "override")
    assert len(COSTS) == 33
    assert len(INPUT_SHA256) == 13
    model = LinearCalibration(calibration_report())
    for cost in model.costs.values():
        assert cost.predict_ns(cost.min_feature) > 0
        assert cost.predict_ns(cost.max_feature) > 0
