# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native DMA geometry, not access_ptr extent, certifies L1 overwrite epochs."""

import pytest
import tilelang  # noqa: F401
from tvm import arith, ir, tir


def fixture(skv=128, offset=0):
    epoch = tir.Var("epoch", "int32")
    buffer = tir.decl_buffer((64, 128), "float16", name="kv", scope="shared.l1")
    gm = tir.decl_buffer((skv, 128), "float16", name="gm")
    call = tir.call_extern("handle", "copy_gm_to_l1<half,64,128>", gm.access_ptr("r"),
                           buffer.access_ptr("w", offset=offset), 128,
                           tir.min(tir.max(skv - epoch * 64, 0), 64), 128, 64, 128)
    analyzer = arith.Analyzer()
    analyzer.bind(epoch, ir.Range.from_min_extent(0, (skv + 63) // 64))
    return buffer, call, analyzer


@pytest.mark.parametrize("skv", [64, 128, 192])
def test_aligned_full_dma_has_physical_overwrite_certificate(skv):
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture(skv)
    proof = certify_l1_epoch_write(buffer, call, analyzer)
    assert proof.elements == 8192
    assert proof.byte_footprint == ((0, 16384),)


def test_partial_tail_cannot_reuse_offset_zero_clear_as_ring_proof():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture(65)
    with pytest.raises(ValueError, match="full rows"):
        certify_l1_epoch_write(buffer, call, analyzer)


def test_pointer_extent_is_not_a_full_overwrite_proof():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture()
    args = list(call.args)
    args[4] = tir.IntImm("int32", 32)
    call = tir.Call(call.dtype, call.op, args)
    with pytest.raises(ValueError, match="full rows"):
        certify_l1_epoch_write(buffer, call, analyzer)


def test_splice_base_is_not_a_version_base():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture(offset=16)
    with pytest.raises(ValueError, match="base"):
        certify_l1_epoch_write(buffer, call, analyzer)


def test_source_first_tir_abi_is_required():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture()
    args = list(call.args)
    args[1], args[2] = args[2], args[1]
    with pytest.raises(ValueError, match="destination"):
        certify_l1_epoch_write(buffer, tir.Call(call.dtype, call.op, args), analyzer)


def test_unbounded_runtime_tail_is_not_certified():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture()
    args = list(call.args)
    args[4] = tir.Var("tail", "int32")
    with pytest.raises(ValueError, match="full rows"):
        certify_l1_epoch_write(buffer, tir.Call(call.dtype, call.op, args), analyzer)


def test_template_geometry_must_match_allocation():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture()
    args = list(call.args)
    args[0] = tir.StringImm("copy_gm_to_l1<half,32,128>")
    with pytest.raises(ValueError, match="geometry"):
        certify_l1_epoch_write(buffer, tir.Call(call.dtype, call.op, args), analyzer)


def test_native_literal_zero_means_full_dimension_not_empty_dma():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    buffer, call, analyzer = fixture()
    args = list(call.args)
    args[4] = args[5] = tir.IntImm("int32", 0)
    assert certify_l1_epoch_write(buffer, tir.Call(call.dtype, call.op, args),
                                  analyzer).elements == 8192


def test_mixed_dtype_allocation_cannot_share_fp16_geometry_proof():
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_write
    _, call, analyzer = fixture()
    other = tir.decl_buffer((64, 128), "float32", name="other", scope="shared.l1")
    with pytest.raises(ValueError, match="FP16"):
        certify_l1_epoch_write(other, call, analyzer)


@pytest.mark.parametrize("transpose", [False, True])
def test_reader_source_geometry_is_bounded_in_native_layout(transpose):
    from tilelang.transform.ascend_schedule.l1_epoch import certify_l1_epoch_read
    buffer, _, analyzer = fixture()
    dest = tir.decl_buffer((128, 128), "float16", name="l0", scope="local.L0B")
    rows, cols = (128, 64) if transpose else (64, 128)
    call = tir.call_extern("handle", f"copy_l1_to_l0b<half,64,128,{str(transpose).lower()}>",
                           buffer.access_ptr("r"), dest.access_ptr("w"), rows, cols)
    assert certify_l1_epoch_read(buffer, call, analyzer) == ((0, 16384),)
    args = list(call.args)
    args[3] = tir.IntImm("int32", rows + 1)
    with pytest.raises(ValueError, match="escapes"):
        certify_l1_epoch_read(buffer, tir.Call(call.dtype, call.op, args), analyzer)


@pytest.mark.parametrize("causal", [False, True])
def test_actual_fa_lowered_kv_dma_and_l0_readers_are_certified(causal):
    import tvm
    from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                             flash_attention_auto128_fwd,
                                                             pass_configs)
    from tilelang.engine.phase import LowerAndLegalize
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms
    from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    from tilelang.transform.ascend_schedule.l1_epoch import (certify_l1_epoch_write,
                                                             certify_l1_epoch_read)

    function = flash_attention_auto128_fwd.__wrapped__(
        FlashAttentionConfig(1, 1, 128, 192, 128, causal=causal)).with_attr("npu_platform", "A2")
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=pass_configs):
        mod = LowerAndLegalize(tvm.IRModule({"main": function}), target)
        mod = prepare_cross_core_copies(mod)
        function = decompose_function_gemms(mod["main"], root_scratch=True)
    buffers, owners, calls = [], [], []

    def collect(node):
        if isinstance(node, tir.Block):
            buffers.extend(node.alloc_buffers)
        elif isinstance(node, tir.For):
            owners.append(node)
        elif isinstance(node, tir.Call) and node.op.name == "tir.call_extern":
            calls.append(node)

    tir.stmt_functor.post_order_visit(function.body, collect)
    assert len(owners) == 1
    owner = owners[0]
    analyzer = certify_uniform_clock(function, owner).analyzer
    analyzer.bind(owner.loop_var, ir.Range.from_min_extent(owner.min, owner.extent))
    tested = []
    for buffer in buffers:
        if buffer.name not in ("k", "v"):
            continue
        writes, reads = [], []
        for call in calls:
            name = str(call.args[0].value).split("::")[-1].replace(" ", "")
            if name.startswith("copy_gm_to_l1<") and call.args[2].args[1].same_as(buffer.data):
                writes.append(certify_l1_epoch_write(buffer, call, analyzer))
            elif name.startswith("copy_l1_to_l0") and call.args[1].args[1].same_as(buffer.data):
                reads.append(certify_l1_epoch_read(buffer, call, analyzer))
        assert len(writes) == len(reads) == 1
        tested.append(buffer.name)
    assert sorted(tested) == ["k", "v"]
