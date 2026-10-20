# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Keep fork-specific calls compatible with upstream Vector legalization.

Reuse the actual regression factories, but intercept the JIT compile boundary
before device compilation or tensor execution. Only native TIR lowering and
AscendC source generation run; Tensor.npu is explicitly forbidden.
"""

from __future__ import annotations

import importlib
import re
import runpy
from pathlib import Path

import pytest
import torch

import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
JIT_MODULE = importlib.import_module("tilelang.jit")


class _LoweringCaptured(Exception):
    """Stop the original test immediately after source generation."""

    def __init__(self, artifact):
        super().__init__("native source generated without device execution")
        self.artifact = artifact


def _source_from_factory(monkeypatch, platform, relative_path, invoke):
    def forbid_npu(*args, **kwargs):
        pytest.fail("CPU codegen regression must not transfer tensors to NPU")

    def lower_only(func, **kwargs):
        with tvm.transform.PassContext(opt_level=3, config=kwargs.get("pass_configs") or {}):
            artifact = tilelang.lower(func, target="ascendc", platform=platform)
        # _run_silu evaluates kernel() before x.npu(). Stop here so neither
        # the tensor transfer nor the returned kernel can execute.
        raise _LoweringCaptured(artifact)

    monkeypatch.setattr(torch.Tensor, "npu", forbid_npu, raising=False)
    monkeypatch.setattr(JIT_MODULE, "compile", lower_only)
    module = runpy.run_path(str(ROOT / relative_path))
    with pytest.raises(_LoweringCaptured) as captured:
        invoke(module)
    return captured.value.artifact.kernel_source


def _compact(source):
    return re.sub(r"\s+", "", source)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize(
    ("dtype", "inplace"),
    [("float16", True), ("float32", True), ("float32", False)],
)
def test_silu_hidden_tmp_survives_vector_selection(monkeypatch, platform, dtype, inplace):
    source = _source_from_factory(
        monkeypatch,
        platform,
        "testing/python/language/test_tilelang_ascend_language_silu_inplace.py",
        lambda module: module["_run_silu"]("ascendc", dtype, inplace),
    )
    compact = _compact(source)
    calls = re.findall(r"AscendC::Silu\(([^,;]+),([^,;]+),256\);", compact)
    assert len(calls) == 1, source
    silu_dst, silu_src = calls[0]
    assert silu_dst != silu_src, "Silu must never receive overlapping operands"
    token = "half" if dtype == "float16" else "float"
    # UB tensors are declared with auto from TBuf::GetWithOffset<T>(), not
    # an explicit LocalTensor<T> declaration.
    assert f"ascend_ub.GetWithOffset<{token}>" in compact, source
    if inplace:
        # The computed temporary must be copied back to the original input.
        assert f"AscendC::DataCopy({silu_src},{silu_dst},256);" in compact, source
        assert "silu_inplace_tmp" in source
    else:
        assert "silu_inplace_tmp" not in source


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("tail", [False, True])
def test_mma_bias_preserves_linear_l1_and_bt_lowering(monkeypatch, platform, tail):
    suffix = "_n_tail" if tail else ""
    source = _source_from_factory(
        monkeypatch,
        platform,
        f"testing/python/language/test_tilelang_ascend_language_mma_bias{suffix}.py",
        lambda module: module["build"](),
    )
    compact = _compact(source)
    assert "tl::ascend::copy_gm_to_l1_linear<float," in compact, source
    assert "tl::ascend::copy_l1_to_bt<float,float>" in compact, source
    assert "tl::ascend::mma_bias<half,float," in compact, source
    assert "AscendC::TPosition::C2" in compact, "Bias must retain its BT allocation"
    bt_calls = re.findall(r"copy_l1_to_bt<float,float>\(([^;]+)\);", compact)
    mma_calls = re.findall(r"mma_bias<[^>]+>\(([^;]+)\);", compact)
    assert len(bt_calls) == len(mma_calls) == 1
    bt_args = bt_calls[0].split(",")
    mma_args = mma_calls[0].split(",")
    assert len(bt_args) == 3 and len(mma_args) >= 6
    assert "[" in mma_args[3], "The fourth MMA operand must be a BT access, not init"
    assert mma_args[3] == bt_args[0], "MMA must consume the BT destination just loaded"
    # The ordinary three-operand MMA must not silently replace bias init.
    assert "tl::ascend::mma<" not in compact


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_im2col_survives_resource_scope_verification(monkeypatch, platform, dtype):
    source = _source_from_factory(
        monkeypatch,
        platform,
        "testing/python/jit/test_simulator_im2col_conv2d_fp32.py",
        lambda module: module["_conv2d_kernel"](platform, dtype=dtype),
    )
    compact = _compact(source)
    token = "half" if dtype == "float16" else "float"
    calls = re.findall(rf"tl::ascend::im2col<{token}>\(([^;]+)\);", compact)
    assert len(calls) == 1, source
    args = calls[0].split(",")
    assert len(args) == 18, "im2col requires two pointers and 16 geometry arguments"
    assert args[2:6] == ["8", "8", "3", "3"]
    valid_k = "144" if dtype == "float16" else "72"
    assert args[-4:] == ["0", "0", "64", valid_k]
    assert "tl::ascend::copy_l1_to_l0b<" in compact
    assert f"tl::ascend::mma<{token},float," in compact
