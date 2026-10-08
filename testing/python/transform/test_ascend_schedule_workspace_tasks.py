# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Expose the existing numerical GM-hop lowering before task analysis."""

import pytest
import tvm
import tilelang  # noqa: F401
from tvm import tir
from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies
from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
from tilelang.transform.ascend_schedule.dependency import analyze_dependencies
from tilelang.transform.ascend_schedule.entry import schedule_module, AutoScheduleError


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_ub_to_l1_is_two_real_pipes_with_a_shared_gm_dependency(platform):
    ub = tir.decl_buffer((16, 16), "float16", name="ub", scope="shared.ub")
    l1 = tir.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    l0 = tir.decl_buffer((16, 16), "float16", name="l0", scope="wmma.matrix_a")
    virtual = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_ub_to_l1<half,16,16>", ub.access_ptr("r"),
                        l1.access_ptr("w"), 16, 16, 16, 16))
    consume = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_l1_to_l0a<half,16,16,false>",
                        l1.access_ptr("r"), l0.access_ptr("w"), 16, 16))
    body = tir.SeqStmt([
        tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, virtual),
        tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, consume),
    ])
    block = tir.Block([], [], [], "root", body, alloc_buffers=[ub, l1, l0])
    body = tir.BlockRealize([], True, block)
    for tag in ("threadIdx.x", "blockIdx.x"):
        axis = tir.IterVar(None, tir.Var(tag.replace(".", "_"), "int32"), tir.IterVar.ThreadIndex,
                           tag)
        body = tir.AttrStmt(axis, "thread_extent", 1, body)
    function = tir.PrimFunc([], body).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    mod = tilelang.transform.BufferShapeCollector()(mod)
    prepared = prepare_cross_core_copies(mod)
    extracted = extract_tasks(prepared["main"])
    assert [(task.core, task.pipe) for task in extracted.tasks] == [("V", "PIPE_MTE3"),
                                                                    ("C", "PIPE_MTE2"),
                                                                    ("C", "PIPE_MTE1")]
    dependencies = analyze_dependencies(extracted.tasks, max_distance=0)
    assert any(d.source == 0 and d.target == 1 and d.kind == "RAW" for d in dependencies)
    assert len(prepared["main"].params) == 1  # Generated GM workspace parameter.
    assert tvm.ir.structural_equal(prepared, prepare_cross_core_copies(prepared))
    assert all("flag" not in node.value.op.name for node in extracted.task_nodes.values())
    with pytest.raises(AutoScheduleError, match="extraction succeeded"):
        schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_l0c_to_ub_exposes_fix_write_and_vector_mte2_read(platform):
    acc = tir.decl_buffer((16, 16), "float32", name="acc", scope="wmma.accumulator")
    ub = tir.decl_buffer((16, 16), "float32", name="ub", scope="shared.ub")
    out = tir.decl_buffer((16, 16), "float32", name="out", scope="shared.ub")
    virtual = tir.Evaluate(
        tir.call_extern("handle",
                        "tl::ascend::copy_l0c_to_ub<float,float,layout::RowMajor,16,16,false>",
                        acc.access_ptr("r"), ub.access_ptr("w"), 16, 16, 16, 16))
    consume = tir.Evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_add"), out.access_ptr("w"),
                        ub.access_ptr("r"), ub.access_ptr("r"), 256))
    body = tir.SeqStmt([
        tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, virtual),
        tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, consume),
    ])
    root = tir.Block([], [], [], "root", body, alloc_buffers=[acc, ub, out])
    body = tir.BlockRealize([], True, root)
    for tag in ("threadIdx.x", "blockIdx.x"):
        axis = tir.IterVar(None, tir.Var(tag.replace(".", "_"), "int32"), tir.IterVar.ThreadIndex,
                           tag)
        body = tir.AttrStmt(axis, "thread_extent", 1, body)
    function = tir.PrimFunc([], body).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    prepared = prepare_cross_core_copies(tilelang.transform.BufferShapeCollector()(mod))
    extracted = extract_tasks(prepared["main"])
    assert [(task.core, task.pipe) for task in extracted.tasks] == [("C", "PIPE_FIX"),
                                                                    ("V", "PIPE_MTE2"),
                                                                    ("V", "PIPE_V")]
    dependencies = analyze_dependencies(extracted.tasks, max_distance=0)
    assert any(d.source == 0 and d.target == 1 and d.kind == "RAW" for d in dependencies)
    assert len(prepared["main"].params) == 1
    assert all("flag" not in node.value.op.name for node in extracted.task_nodes.values())
    with pytest.raises(AutoScheduleError, match="extraction succeeded"):
        schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})
