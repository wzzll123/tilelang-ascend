# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tanh frontend, workspace, selected-mask, and native codegen contracts."""

import pytest

import tilelang
import tilelang.language as T
from tilelang.engine.lower import device_codegen
from tvm import IRModule, tir
from tvm.ir import Op, Range
from tvm.target import Target
from tvm.ir.transform import PassContext


TARGET = Target({"kind": "llvm", "model": "ascendc"})


def _buffers(dtype="float32", size=128, tmp_bytes=4096):
    return (
        tir.decl_buffer((size,), dtype, name="dst", scope="shared.ub"),
        tir.decl_buffer((size,), dtype, name="src", scope="shared.ub"),
        tir.decl_buffer((tmp_bytes,), "uint8", name="scratch", scope="shared.ub"),
    )


def _module(call, buffers):
    statement = call if isinstance(call, tir.Stmt) else tir.Evaluate(call)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, statement)
    block = tir.Block([], [], [], "tilelang_root", body, alloc_buffers=list(buffers))
    func = tir.PrimFunc([], tir.BlockRealize([], True, block)).with_attr("global_symbol", "main")
    return IRModule({"main": func})


def _calls(mod):
    result = []

    def visit(node):
        if isinstance(node, tir.Call):
            result.append(node)

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return result


def tanh_program(dtype="float32", explicit=False):
    @T.prim_func
    def main(source: T.Tensor((128,), dtype), output: T.Tensor((128,), dtype)):
        with T.Kernel(1, is_npu=True):
            src_ub = T.alloc_ub((128,), dtype)
            dst_ub = T.alloc_ub((128,), dtype)
            scratch = T.alloc_ub((1024,), "uint8")
            with T.Scope("V"):
                T.copy(source, src_ub)
                if explicit:
                    T.tile.tanh(dst_ub, src_ub, tmp=scratch)
                else:
                    T.tile.tanh(dst_ub, src_ub)
                T.copy(dst_ub, output)

    return main


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("explicit", [False, True])
def test_tanh_full_lowering_generates_native_api(dtype, platform, explicit):
    artifact = tilelang.lower(tanh_program(dtype, explicit), target="ascendc", platform=platform)
    assert "AscendC::Tanh(" in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("size", [16, 128, 256])
def test_tanh_workspace_uses_official_maximum(dtype, size):
    dst, src, _ = _buffers(dtype, size)
    mod = tilelang.transform.InjectTmpBuffer(TARGET)(_module(T.tile.tanh(dst, src), [dst, src]))
    call = next(call for call in _calls(mod) if call.op.name == "tl.ascend_tanh")
    scratch = call.args[2]
    factor, element_bytes = (4, 2) if dtype == "float16" else (1, 4)
    assert scratch.args[0].dtype == "uint8"
    assert int(scratch.args[3]) == factor * max(size * element_bytes, 256)
    assert int(scratch.args[4]) == 2  # Scratch is clobbered, not input data.


@pytest.mark.parametrize("dtype,minimum", [("float16", 1024), ("float32", 256)])
def test_tanh_explicit_workspace_minimum_is_checked(dtype, minimum):
    dst, src, scratch = _buffers(dtype, tmp_bytes=minimum - 32)
    with pytest.raises(Exception, match="Tanh.*workspace.*too small|Tanh.*tmp.*too small"):
        tilelang.transform.InjectTmpBuffer(TARGET)(_module(T.tile.tanh(dst, src, tmp=scratch), [dst, src, scratch]))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_tanh_selected_terminal_reaches_native_codegen(dtype, platform):
    dst, src, scratch = _buffers(dtype)
    mod = tilelang.transform.InjectTmpBuffer(TARGET)(_module(T.tile.tanh(dst, src, tmp=scratch), [dst, src, scratch]))
    # Eliminate high-level Blocks before selection; nothing may transform the
    # IR after mask legalization. Full-pipeline coverage is tested separately.
    mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    with PassContext(config={"tl.ascend_memory_planning": True}):
        mod = tilelang.transform.AscendMemoryPlanning()(mod)
    selected = tilelang.transform.AscendVectorInstructionSelection(TARGET, platform)(mod)
    names = [call.op.name for call in _calls(selected)]
    assert "tl.ascend_tanh_self_contained_normal_full" in names
    legalized = tilelang.transform.AscendVectorMaskLegalize(TARGET, platform)(selected)
    source = device_codegen(legalized, TARGET, platform).get_source()
    assert "AscendC::Tanh(" in source
    assert "scratch" in source


@pytest.mark.parametrize("dtype", ["bfloat16", "int32"])
def test_tanh_unsupported_dtype_is_rejected(dtype):
    dst, src, _ = _buffers(dtype)
    with pytest.raises(ValueError, match="float16.*float32"):
        T.tile.tanh(dst, src)


def test_tanh_mixed_dtype_is_rejected():
    dst, _, _ = _buffers("float16")
    _, src, _ = _buffers("float32")
    with pytest.raises(ValueError, match="same dtype"):
        T.tile.tanh(dst, src)


def test_tanh_overlapping_operands_are_rejected():
    dst, src, _ = _buffers()
    with pytest.raises(ValueError, match="overlap"):
        T.tile.tanh(src, src)
    with pytest.raises(ValueError, match="overlap"):
        T.tile.tanh(dst, src, tmp=src)


def test_tanh_noncontiguous_region_is_rejected():
    dst = tir.decl_buffer((4, 32), "float32", scope="shared.ub")
    src = tir.decl_buffer((4, 64), "float32", scope="shared.ub")
    region = tir.BufferRegion(src, [Range(0, 4), Range(0, 32)])
    with pytest.raises(ValueError, match="contiguous"):
        T.tile.tanh(dst, region)


def test_tanh_pto_is_explicitly_unsupported():
    dst, src, _ = _buffers()
    with pytest.raises(Exception, match="not supported.*PTO|PTO.*not supported"):
        tilelang.transform.InjectTmpBuffer(Target({"kind": "llvm", "model": "pto"}))(_module(T.tile.tanh(dst, src), [dst, src]))


@pytest.mark.parametrize("after_count,expected_counter_sets", [(64, 1), (96, 2)])
def test_tanh_repairs_mask_state_between_managed_vector_ops(after_count, expected_counter_sets):
    dst, src, scratch = _buffers()
    base = T.tile.add(dst, src, src)
    # 96 fp32 elements selects counter mode; 128 selects two full normal repeats.
    before = tir.Call("handle", base.op, [*base.args[:-1], 96])
    after = tir.Call("handle", before.op, [dst.access_ptr("w"), src.access_ptr("r"), src.access_ptr("r"), after_count])
    body = tir.SeqStmt(
        [
            tir.Evaluate(before),
            tir.Evaluate(T.tile.tanh(dst, src, tmp=scratch)),
            tir.Evaluate(after),
        ]
    )
    mod = tilelang.transform.InjectTmpBuffer(TARGET)(_module(body, [dst, src, scratch]))
    mod = tilelang.transform.AscendVectorInstructionSelection(TARGET, "A2")(mod)
    mod = tilelang.transform.AscendVectorMaskLegalize(TARGET, "A2")(mod)
    calls = _calls(mod)
    mode_sets = [call for call in calls if call.op.name == "tl.ascend_set_mask_mode"]
    # The first Add enters counter mode. Tanh restores normal/full internally.
    # A following full normal Add needs no repair; a counter Add must re-enter.
    assert len(mode_sets) == expected_counter_sets
    assert all(int(call.args[0]) == 1 for call in mode_sets)


@pytest.mark.parametrize("failure", ["dtype", "overlap", "scratch", "alignment"])
def test_tanh_selector_rejects_invalid_raw_intrinsics(failure):
    dst, src, scratch = _buffers()
    dst_ptr, src_ptr, scratch_ptr = dst.access_ptr("w"), src.access_ptr("r"), scratch.access_ptr("w")
    message = {
        "dtype": "same dtype",
        "overlap": "overlap",
        "scratch": "workspace is too small",
        "alignment": "32-byte aligned",
    }[failure]
    if failure == "dtype":
        src = tir.decl_buffer((128,), "float16", name="mixed_src", scope="shared.ub")
        src_ptr = src.access_ptr("r")
    elif failure == "overlap":
        dst_ptr = src.access_ptr("w")
    elif failure == "scratch":
        scratch = tir.decl_buffer((32,), "uint8", name="tiny", scope="shared.ub")
        scratch_ptr = scratch.access_ptr("w")
    else:
        src_ptr = src.access_ptr("r", offset=1)
    raw = tir.Call("handle", Op.get("tl.ascend_tanh"), [dst_ptr, src_ptr, scratch_ptr, 64])
    with pytest.raises(Exception, match=message):
        tilelang.transform.AscendVectorInstructionSelection(TARGET, "A2")(_module(raw, [dst, src, scratch]))
