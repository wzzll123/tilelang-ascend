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


def assert_mode2_execution(function, extracted, dependencies, platform, source_name):
    import numpy as np
    from dataclasses import replace
    from tilelang.transform.ascend_schedule.cross_sync import emit_mode2_sync
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    SimulationDeadlockError, MemoryHazardError)

    synced = emit_mode2_sync(function, extracted, dependencies, cross_ids=(5,), local_ids=(0, 1))
    calls = []
    tir.stmt_functor.post_order_visit(
        synced.body, lambda node: calls.append(node) if isinstance(node, tir.Call) else None)
    sets = [call for call in calls if call.op.name == "tl.ascend_set_cross_flag"]
    waits = [call for call in calls if call.op.name == "tl.ascend_wait_cross_flag"]
    assert len(sets) == len(waits) == 1
    assert int(sets[0].args[2]) == 2
    assert waits[0].args[1].value == ""  # A2/A3 scalar wait, not A5 pipe wait.
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        mod = tvm.IRModule({"probe": synced.with_attr("global_symbol", "probe")})
        mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
        mod = tilelang.transform.Flatten2DBuffer()(mod)
        mod = tilelang.transform.FlattenBuffer()(mod)
        mod = tilelang.transform.AscendMemoryPlanning()(mod)
        source = tvm.get_global_func("target.build.tilelang_ascend")(mod, function.attrs["target"],
                                                                     platform).get_source()
    assert "CrossCoreSetFlag<0x2, PIPE_" + sets[0].args[0].value + ">(5)" in source
    assert "CrossCoreWaitFlag(5)" in source

    def simulator(candidate, *, delayed_producer=False):
        program = build_kernel_program(candidate, platform=platform)
        if delayed_producer:
            # Adversarial scheduling probe, not a hardware timing prediction.
            def delay(task):
                region = task.metadata.get("src")
                return replace(
                    task, duration_cycles=1000
                ) if region is not None and region.buffer == source_name else task

            program = replace(
                program,
                cores=tuple(
                    replace(core, tasks=tuple(delay(t)
                                              for t in core.tasks))
                    for core in program.cores))
        sim = FunctionalSimulator(program, SimulatorConfig(platform=platform, sync_only=True))
        for task in program.tasks:
            region = task.metadata.get("src")
            if region is not None and region.buffer == source_name:
                owner = sim.memory.vector1_owner(
                    task.core_id) if task.lane.value == "vector1" else task.core_id
                sim.write(replace(region, core_id=owner), np.ones(region.shape), task_core_id=owner)
        return sim

    simulator(synced).run()
    simulator(synced, delayed_producer=True).run()

    def remove_set(node):
        if (isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) and
                node.value.op.name == "tl.ascend_set_cross_flag"):
            return tir.Evaluate(0)
        return None

    bad = synced.with_body(tir.stmt_functor.ir_transform(synced.body, None, remove_set))
    with pytest.raises(SimulationDeadlockError):
        simulator(bad).run()

    def remove_wait(node):
        if (isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) and
                node.value.op.name == "tl.ascend_wait_cross_flag"):
            return tir.Evaluate(0)
        return None

    bad_wait = synced.with_body(tir.stmt_functor.ir_transform(synced.body, None, remove_wait))
    with pytest.raises(MemoryHazardError):
        simulator(bad_wait, delayed_producer=True).run()

    from tilelang.transform.ascend_schedule.cross_sync import emit_mode2_loop_sync
    for extent in (0, 1, 2, 5):
        owners = []

        def wrap(node, extent=extent, owners=owners):
            if isinstance(node, tir.Block) and node.name_hint == "root":
                owner = tir.For(tir.Var("epoch", "int32"), 3, extent, tir.ForKind.SERIAL, node.body)
                owners.append(owner)
                return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint, owner,
                                 node.init, node.alloc_buffers, node.match_buffers,
                                 node.annotations)
            return None

        candidate = function.with_body(tir.stmt_functor.ir_transform(function.body, None, wrap))
        assert len(owners) == 1
        owner = owners[0]
        loop_extraction = extract_tasks(owner.body)
        loop_dependencies = analyze_dependencies(loop_extraction.tasks, max_distance=1)
        synchronized = emit_mode2_loop_sync(
            candidate,
            owner,
            loop_extraction,
            loop_dependencies,
            cross_ids=(0, 1, 2, 3),
            local_ids=(0, 1, 2, 3))
        simulator(synchronized).run()
        simulator(synchronized, delayed_producer=True).run()
        from tilelang.transform.ascend_schedule.buffer_versions import materialize_iteration_versions
        from tilelang.transform.ascend_schedule.ownership import EpochPlan, OwnerEpochPlan
        workspace = next(iter(candidate.buffer_map.values()))
        pitch = int(np.prod([int(x) for x in workspace.shape])) * tvm.DataType(
            workspace.dtype).bits // 8
        plan = EpochPlan(workspace.data, True, 2, (OwnerEpochPlan(owner, "iteration", object()),))
        versioned = materialize_iteration_versions(candidate, workspace, plan)
        loops = []
        tir.stmt_functor.post_order_visit(
            versioned.body,
            lambda node, loops=loops: loops.append(node) if isinstance(node, tir.For) else None)
        assert len(loops) == 1
        ring_owner = loops[0]
        ring_extraction = extract_tasks(ring_owner.body)

        def is_workspace(storage, workspace=workspace):
            return hasattr(storage, "var") and storage.var.same_as(workspace.data)

        # Conservative full-allocation regions for this one-core fixture cover
        # both Vector shards; no inferred edge is silently turned into a fence.
        ring_extraction.tasks = tuple(
            replace(
                task,
                accesses=tuple(
                    replace(
                        access,
                        offset=0,
                        extent=pitch,
                        intervals=None,
                        iteration_stride=pitch,
                        ring_size=2) if is_workspace(access.storage) else access
                    for access in task.accesses))
            for task in ring_extraction.tasks)
        ring_dependencies = analyze_dependencies(ring_extraction.tasks, max_distance=2)
        counts = {
            (d.source, d.target, d.distance): 2 if is_workspace(d.storage) else 1
            for d in ring_dependencies
        }
        ring_synced = emit_mode2_loop_sync(
            versioned,
            ring_owner,
            ring_extraction,
            ring_dependencies,
            cross_ids=tuple(range(8)),
            local_ids=(0, 1, 2, 3),
            slot_counts=counts)
        simulator(ring_synced).run()
        simulator(ring_synced, delayed_producer=True).run()
        missing_set = synchronized.with_body(
            tir.stmt_functor.ir_transform(synchronized.body, None, remove_set))
        with pytest.raises(SimulationDeadlockError):
            simulator(missing_set).run()

        def remove_drain(node):
            if isinstance(node, tir.SeqStmt):
                index = next((i for i, child in enumerate(node.seq) if isinstance(child, tir.For)),
                             None)
                if index is not None and index < len(node.seq) - 1:
                    return tir.SeqStmt(list(node.seq)[:index + 1])
            return None

        missing_drain = synchronized.with_body(
            tir.stmt_functor.ir_transform(synchronized.body, None, remove_drain))
        with pytest.raises(SimulationDeadlockError, match="FLAG ACCOUNTING"):
            simulator(missing_drain).run()


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_ub_to_l1_is_two_real_pipes_with_a_shared_gm_dependency(platform):
    ub = tir.decl_buffer((8, 16), "float16", name="ub", scope="shared.ub")
    l1 = tir.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    l0 = tir.decl_buffer((16, 16), "float16", name="l0", scope="wmma.matrix_a")
    virtual = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_ub_to_l1<half,16,8>", ub.access_ptr("r"),
                        l1.access_ptr("w"), 16, 8, 16, 16))
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
        body = tir.AttrStmt(axis, "thread_extent", 2 if tag == "threadIdx.x" else 1, body)
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
    assert_mode2_execution(prepared["main"], extracted, dependencies, platform, "ub")


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_l0c_to_ub_exposes_fix_write_and_vector_mte2_read(platform):
    acc = tir.decl_buffer((16, 16), "float32", name="acc", scope="wmma.accumulator")
    ub = tir.decl_buffer((8, 16), "float32", name="ub", scope="shared.ub")
    out = tir.decl_buffer((8, 16), "float32", name="out", scope="shared.ub")
    virtual = tir.Evaluate(
        tir.call_extern("handle",
                        "tl::ascend::copy_l0c_to_ub<float,float,layout::RowMajor,16,16,false>",
                        acc.access_ptr("r"), ub.access_ptr("w"), 16, 16, 8, 16))
    consume = tir.Evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_add"), out.access_ptr("w"),
                        ub.access_ptr("r"), ub.access_ptr("r"), 128))
    body = tir.SeqStmt([
        tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, virtual),
        tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, consume),
    ])
    root = tir.Block([], [], [], "root", body, alloc_buffers=[acc, ub, out])
    body = tir.BlockRealize([], True, root)
    for tag in ("threadIdx.x", "blockIdx.x"):
        axis = tir.IterVar(None, tir.Var(tag.replace(".", "_"), "int32"), tir.IterVar.ThreadIndex,
                           tag)
        body = tir.AttrStmt(axis, "thread_extent", 2 if tag == "threadIdx.x" else 1, body)
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
    assert_mode2_execution(prepared["main"], extracted, dependencies, platform, "acc")
