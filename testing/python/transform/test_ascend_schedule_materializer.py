# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The enabled compiler entry must generate executable fences, not only inspect."""

import json
import pytest
import tilelang
import tilelang.language as T
import tvm
from tvm import tir
from tilelang.transform.ascend_schedule.entry import schedule_module


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("versions", [None, 1, 2, 3])
def test_automatic_ub_versions_are_selected_materialized_and_executed(versions, platform):
    pytest.importorskip("z3")
    import numpy as np
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)

    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    handle, index = tir.Var("output_handle", "handle"), tir.Var("epoch", "int32")
    operations = tir.SeqStmt([
        tir.Evaluate(T.tile.fill(ub, index)),
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm<float,8>", ub.access_ptr("r"),
                            output.access_ptr("w"), 8)),
    ])
    loop = tir.For(index, 3, 5, tir.ForKind.SERIAL, operations)
    annotations = {} if versions is None else {"tl.buffer_versions_map": {ub.data: versions}}
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, loop),
                     alloc_buffers=[ub],
                     annotations=annotations)
    function = tir.PrimFunc([handle], tir.BlockRealize([], True, root), buffer_map={
        handle: output
    }).with_attr("npu_platform", platform)
    scheduled = schedule_module(
        tvm.IRModule({"main": function}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    if versions == 1:
        assert int(scheduled.attrs["tl.ascend.selected_buffer_versions"][ub.data]) == 1
        assert "tl.ascend.iteration_versions" not in scheduled.attrs
    else:
        selection = json.loads(str(scheduled.attrs["tl.ascend.version_selection"]))
        chosen = selection["schedule"]["buffer_versions"]["scratch"]
        assert chosen == (versions or 2)
        assert int(scheduled.attrs["tl.ascend.iteration_versions"][ub.data][0]) == chosen
    for sync_only in (False, True):
        sim = FunctionalSimulator(
            build_kernel_program(scheduled, platform=platform),
            SimulatorConfig(platform=platform, sync_only=sync_only, flag_balance_check="error"))
        sim.run()
        if not sync_only:
            np.testing.assert_array_equal(
                sim.read(BufferRegion("output", MemoryScope.GM, (8,), "float32")), np.full(8, 7))
    from tilelang.engine.phase import OptimizeForTarget
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    native = tir.transform.BindTarget(target)(
        tvm.IRModule({"probe": scheduled.with_attr("global_symbol", "probe")}))
    with tvm.transform.PassContext(config={"tl.ascend_auto_schedule": True}):
        native = OptimizeForTarget(native, target, platform)
    source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                 platform).get_source()
    assert f"GetWithOffset<float>({8 * (versions or 2)}," in source
    assert "SetFlag<" in source and "WaitFlag<" in source
    if versions == 2:
        from tilelang.transform.ascend_schedule.entry import AutoScheduleError

        def partial_fill(node):
            if isinstance(node, tir.Call) and node.op.name == "tl.ascend_fill":
                return tir.Call(node.dtype, node.op, [*node.args[:3], tir.IntImm("int32", 7)])
            return None

        partial = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, partial_fill))
        with pytest.raises(AutoScheduleError, match="version hints require owner materialization"):
            schedule_module(
                tvm.IRModule({"main": partial}), "ascendc", {"tl.ascend_auto_schedule": True})

        def initial_read(node):
            if isinstance(node, tir.For):
                return tir.For(node.loop_var, node.min, node.extent, node.kind,
                               tir.SeqStmt(list(node.body.seq)[::-1]))
            return None

        carries_state = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, initial_read))
        with pytest.raises(AutoScheduleError, match="version hints require owner materialization"):
            schedule_module(
                tvm.IRModule({"main": carries_state}), "ascendc", {"tl.ascend_auto_schedule": True})

    if versions == 1:
        from tilelang.transform.ascend_schedule.entry import AutoScheduleError
        alias = tir.Block([], [], [],
                          "alias",
                          function.body,
                          annotations={"tl.buffer_version_mode": {
                              ub.data: "iteration"
                          }})
        conflicting = function.with_body(tir.BlockRealize([], True, alias))
        with pytest.raises(AutoScheduleError, match="inconsistent buffer"):
            schedule_module(
                tvm.IRModule({"main": conflicting}), "ascendc", {"tl.ascend_auto_schedule": True})


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("iterations", [0, 1, 2, 5])
def test_automatic_single_loop_preserves_reuse_and_drains(platform, iterations):
    from tilelang.simulator import build_kernel_program, FunctionalSimulator, SimulatorConfig

    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    handle = tir.Var("output_handle", "handle")
    index = tir.Var("iteration", "int32")
    operations = tir.SeqStmt([
        tir.Evaluate(T.tile.fill(ub, 1)),
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm<float,8>", ub.access_ptr("r"),
                            output.access_ptr("w"), 8)),
    ])
    loop = tir.For(index, 3, iterations, tir.ForKind.SERIAL, operations)
    body = tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, loop)
    root = tir.Block([], [], [], "root", body, alloc_buffers=[ub])
    function = tir.PrimFunc([handle], tir.BlockRealize([], True, root), buffer_map={
        handle: output
    }).with_attr("npu_platform", platform)
    scheduled = schedule_module(
        tvm.IRModule({"main": function}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    report = json.loads(str(scheduled.attrs["tl.ascend.schedule_report"]))
    assert report["materialization"] == "single local owner loop"
    sim = FunctionalSimulator(
        build_kernel_program(scheduled, platform=platform),
        SimulatorConfig(platform=platform, sync_only=True, flag_balance_check="error"))
    sim.run()
    # A protocol with its final drain removed must not pass accounting, even
    # when the owner executes zero iterations. This checks actual execution.
    from tilelang.simulator import SimulationDeadlockError

    def remove_drain(node):
        if isinstance(node, tir.SeqStmt) and any(isinstance(child, tir.For) for child in node.seq):
            return tir.SeqStmt(list(node.seq)[:-1])
        return None

    mutated = scheduled.with_body(tir.stmt_functor.ir_transform(scheduled.body, None, remove_drain))
    with pytest.raises(SimulationDeadlockError, match="FLAG ACCOUNTING"):
        FunctionalSimulator(
            build_kernel_program(mutated, platform=platform),
            SimulatorConfig(platform=platform, sync_only=True, flag_balance_check="error")).run()


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_automatic_flat_vector_entry_executes_without_manual_flags(platform):
    from tilelang.simulator import build_kernel_program, FunctionalSimulator, SimulatorConfig

    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    output_handle = tir.Var("output_handle", "handle")
    body = tir.SeqStmt([
        tir.Evaluate(T.tile.fill(ub, 1)),
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm<float,8>", ub.access_ptr("r"),
                            output.access_ptr("w"), 8)),
    ])
    body = tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, body)
    block = tir.Block([], [], [], "root", body, alloc_buffers=[ub])
    function = tir.PrimFunc([output_handle],
                            tir.BlockRealize([], True, block),
                            buffer_map={
                                output_handle: output
                            }).with_attr("npu_platform", platform)
    mod = tvm.IRModule({"main": function})
    result = schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})
    assert int(result["main"].attrs["tl.ascend.schedule_materialized"]) == 1
    report = json.loads(str(result["main"].attrs["tl.ascend.schedule_report"]))
    assert report["materialization"] == "flat local/CV epoch"
    FunctionalSimulator(
        build_kernel_program(result["main"], platform=platform),
        SimulatorConfig(platform=platform, sync_only=True)).run()
    from contextlib import ExitStack
    from unittest.mock import patch
    from tilelang.engine.phase import OptimizeForTarget
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    result = tir.transform.BindTarget(target)(result)
    with ExitStack() as stack:
        for name in ("CrossCorePipeline", "CombineCV", "PipelinePlanning", "InjectSoftwarePipeline",
                     "AscendStorageRewrite", "AscendSyncInsert", "AscendSyncInsertVS"):
            stack.enter_context(
                patch.object(
                    tilelang.transform, name, side_effect=AssertionError("legacy pass was called")))
        stack.enter_context(
            tvm.transform.PassContext(config={
                "tl.ascend_auto_schedule": True,
                "tl.ascend_memory_planning": True
            }))
        optimized = OptimizeForTarget(result, target, platform)
    optimized = tvm.IRModule({"main": optimized["main"].with_attr("global_symbol", "probe")})
    generated = tvm.get_global_func("target.build.tilelang_ascend")(optimized, target,
                                                                    platform).get_source()
    assert "SetFlag<" in generated and "WaitFlag<" in generated


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_automatic_gemm_and_fixpipe_epoch_has_correct_cpu_values(platform):
    import numpy as np
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)

    a = tir.decl_buffer((16, 16), "float16", name="a", scope="shared.l1")
    b = tir.decl_buffer((16, 16), "float16", name="b", scope="shared.l1")
    c = tir.decl_buffer((16, 16), "float32", name="c", scope="wmma.accumulator")
    output = tir.decl_buffer((16, 16), "float32", name="output", scope="global")
    handle = tir.Var("output_handle", "handle")
    gemm = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_gemm_v0"),
                           "gemm_v0<half,float,16,16,16,false,false,128>", a.access_ptr("r"),
                           b.access_ptr("r"), c.access_ptr("w"), True, 16)
    fix = tir.call_extern("handle", "copy_l0c_to_gm<float,float,layout::RowMajor,16,16,false>",
                          c.access_ptr("r"), output.access_ptr("w"), 16, 0, 0)
    body = tir.AttrStmt(
        tir.IntImm("int32", 0), "resource_scope", 0,
        tir.SeqStmt([tir.Evaluate(gemm), tir.Evaluate(fix)]))
    root = tir.Block([], [], [], "root", body, alloc_buffers=[a, b, c])
    function = tir.PrimFunc([handle], tir.BlockRealize([], True, root), buffer_map={
        handle: output
    }).with_attr("npu_platform", platform)
    scheduled = schedule_module(
        tvm.IRModule({"main": function}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    sim = FunctionalSimulator(
        build_kernel_program(scheduled, platform=platform), SimulatorConfig(platform=platform))
    for name in ("a", "b"):
        sim.write(BufferRegion(name, MemoryScope.L1, (256,), "float16", core_id=0), np.ones(256))
    sim.run()
    result = sim.read(BufferRegion("output", MemoryScope.GM, (16, 16), "float32"))
    np.testing.assert_array_equal(result, np.full((16, 16), 16, dtype="float32"))
