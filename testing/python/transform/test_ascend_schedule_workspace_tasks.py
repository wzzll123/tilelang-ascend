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
from tilelang.transform.ascend_schedule.entry import schedule_module


def assert_mode2_execution(function,
                           extracted,
                           dependencies,
                           platform,
                           source_name,
                           automatic_function=None):
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
    if automatic_function is not None:
        simulator(automatic_function).run()
        simulator(automatic_function, delayed_producer=True).run()
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

        from tilelang.transform.ascend_schedule.workspace_proof import certify_workspace_versions
        ring_extraction, diagnostics = certify_workspace_versions(versioned, ring_extraction)
        assert not diagnostics
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


def assert_automatic_cv_loop(function, platform, source_name):
    import numpy as np
    from dataclasses import replace
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    SimulationDeadlockError)
    from tilelang.transform.ascend_schedule.entry import AutoScheduleError

    for extent in (0, 1, 2, 5):
        epoch = tir.Var("auto_epoch", "int32")

        def wrap(node, epoch=epoch, extent=extent):
            if isinstance(node, tir.Block) and node.name_hint == "root":
                return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint,
                                 tir.For(epoch, 3, extent, tir.ForKind.SERIAL,
                                         node.body), node.init, node.alloc_buffers,
                                 node.match_buffers, node.annotations)
            return None

        candidate = function.with_body(tir.stmt_functor.ir_transform(function.body, None, wrap))
        automatic = schedule_module(
            tvm.IRModule({"main": candidate}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]

        def run(candidate):
            sim = FunctionalSimulator(
                build_kernel_program(candidate, platform=platform),
                SimulatorConfig(platform=platform, sync_only=True, flag_balance_check="error"))
            for task in sim.program.tasks:
                region = task.metadata.get("src")
                if region is not None and region.buffer == source_name:
                    owner = sim.memory.vector1_owner(
                        task.core_id) if task.lane.value == "vector1" else task.core_id
                    sim.write(
                        replace(region, core_id=owner), np.ones(region.shape), task_core_id=owner)
            sim.run()

        run(automatic)
        if extent == 2:
            from tilelang.engine.phase import OptimizeForTarget
            target = function.attrs["target"]
            native = tvm.IRModule({"probe": automatic.with_attr("global_symbol", "probe")})
            with tvm.transform.PassContext(config={"tl.ascend_auto_schedule": True}):
                native = OptimizeForTarget(native, target, platform)
            source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                         platform).get_source()
            assert "CrossCoreSetFlag<0x2," in source
            assert "CrossCoreWaitFlag(" in source

        def remove_drain(node):
            if isinstance(node, tir.SeqStmt) and any(
                    isinstance(child, tir.For) for child in node.seq):
                return tir.SeqStmt(list(node.seq)[:-1])
            return None

        mutated = automatic.with_body(
            tir.stmt_functor.ir_transform(automatic.body, None, remove_drain))
        with pytest.raises(SimulationDeadlockError, match="FLAG ACCOUNTING"):
            run(mutated)

        if extent == 1:
            axes = []
            tir.stmt_functor.post_order_visit(
                candidate.body,
                lambda node, axes=axes: axes.append(node.node)
                if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and node.node
                .thread_tag == "threadIdx.x" else None)
            axis = axes[0]

            def relocate_participants(node, axis=axis):
                if isinstance(node, tir.AttrStmt
                             ) and node.attr_key == "thread_extent" and node.node.same_as(axis):
                    return node.body
                if isinstance(node, tir.For):
                    return tir.For(node.loop_var, node.min, node.extent, node.kind,
                                   tir.AttrStmt(axis, "thread_extent", 2, node.body))
                return None

            wrong_scope = candidate.with_body(
                tir.stmt_functor.ir_transform(candidate.body, None, relocate_participants))
            with pytest.raises(AutoScheduleError, match="participant scope must dominate"):
                schedule_module(
                    tvm.IRModule({"main": wrong_scope}), "ascendc",
                    {"tl.ascend_auto_schedule": True})

            def nonuniform_clock(node, axis=axis):
                if isinstance(node, tir.For):
                    return tir.For(node.loop_var, node.min, tir.Select(axis.var == 0, 1, 2),
                                   node.kind, node.body)
                return None

            wrong_clock = candidate.with_body(
                tir.stmt_functor.ir_transform(candidate.body, None, nonuniform_clock))
            with pytest.raises(AutoScheduleError, match="uniform clock"):
                schedule_module(
                    tvm.IRModule({"main": wrong_clock}), "ascendc",
                    {"tl.ascend_auto_schedule": True})


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
    automatic = schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    assert int(automatic.attrs["tl.ascend.schedule_materialized"]) == 1
    assert_mode2_execution(prepared["main"], extracted, dependencies, platform, "ub", automatic)
    assert_automatic_cv_loop(prepared["main"], platform, "ub")


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
    automatic = schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    assert int(automatic.attrs["tl.ascend.schedule_materialized"]) == 1
    assert_mode2_execution(prepared["main"], extracted, dependencies, platform, "acc", automatic)
    assert_automatic_cv_loop(prepared["main"], platform, "acc")
    assert_automatic_cv_phases(prepared["main"], platform, ub, out)


def assert_automatic_cv_phases(function, platform, ub, out):
    import numpy as np
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)
    from tilelang.engine.phase import OptimizeForTarget
    import tilelang.language as T

    axes = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: axes.append(node.node.var) if isinstance(node, tir.AttrStmt) and
        node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x" else None)
    vid = axes[0]
    output = tir.decl_buffer((16, 16), "float32", name="phase_output", scope="global")
    handle = tir.Var("phase_output_handle", "handle")
    for extent in (0, 1, 3):

        def wrap(node, extent=extent):
            if isinstance(node, tir.Block) and node.name_hint == "root":
                prefix = tir.AttrStmt(
                    tir.IntImm("int32", 1), "resource_scope", 1,
                    tir.SeqStmt(
                        [tir.Evaluate(T.tile.fill(ub, 0)),
                         tir.Evaluate(T.tile.fill(out, 0))]))
                owner = tir.For(
                    tir.Var("phase_epoch", "int32"), 3, extent, tir.ForKind.SERIAL, node.body)
                suffix = tir.AttrStmt(
                    tir.IntImm("int32", 1), "resource_scope", 1,
                    tir.Evaluate(
                        tir.call_extern("handle", "copy_ub_to_gm<float,16,8>", out.access_ptr("r"),
                                        output.access_ptr("w", offset=vid * 128), 16, 8, 16)))
                return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint,
                                 tir.SeqStmt([prefix, owner, suffix]), node.init,
                                 node.alloc_buffers, node.match_buffers, node.annotations)
            return None

        body = tir.stmt_functor.ir_transform(function.body, None, wrap)
        buffers = dict(function.buffer_map.items())
        buffers[handle] = output
        candidate = tir.PrimFunc([*function.params, handle], body, function.ret_type, buffers,
                                 function.attrs)
        automatic = schedule_module(
            tvm.IRModule({"main": candidate}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
        if extent == 1:
            from tilelang.transform.ascend_schedule.entry import AutoScheduleError

            def enclose_cut(node):
                if isinstance(node, tir.For):
                    return tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, node)
                return None

            enclosed = candidate.with_body(
                tir.stmt_functor.ir_transform(candidate.body, None, enclose_cut))
            with pytest.raises(AutoScheduleError, match="contradictory nested core scopes"):
                schedule_module(
                    tvm.IRModule({"main": enclosed}), "ascendc", {"tl.ascend_auto_schedule": True})

            axes = []
            tir.stmt_functor.post_order_visit(
                candidate.body,
                lambda node, axes=axes: axes.append(node.node)
                if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and node.node
                .thread_tag == "threadIdx.x" else None)
            axis = axes[0]

            def nested_participants(node, axis=axis):
                if isinstance(node, tir.For):
                    return tir.AttrStmt(axis, "thread_extent", 2, node)
                return None

            nested = candidate.with_body(
                tir.stmt_functor.ir_transform(candidate.body, None, nested_participants))
            with pytest.raises(AutoScheduleError, match="participant context must be shared"):
                schedule_module(
                    tvm.IRModule({"main": nested}), "ascendc", {"tl.ascend_auto_schedule": True})
        for sync_only in (False, True):
            sim = FunctionalSimulator(
                build_kernel_program(automatic, platform=platform),
                SimulatorConfig(platform=platform, sync_only=sync_only, flag_balance_check="error"))
            if extent:
                sim.write(
                    BufferRegion("acc", MemoryScope.L0C, (256,), "float32", core_id=0),
                    np.ones(256))
            sim.run()
            if not sync_only:
                np.testing.assert_array_equal(
                    sim.read(BufferRegion("phase_output", MemoryScope.GM, (16, 16), "float32")),
                    np.full((16, 16), 2 if extent else 0))
        native = tvm.IRModule({"probe": automatic.with_attr("global_symbol", "probe")})
        with tvm.transform.PassContext(config={"tl.ascend_auto_schedule": True}):
            native = OptimizeForTarget(native, function.attrs["target"], platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native,
                                                                     function.attrs["target"],
                                                                     platform).get_source()
        assert "CrossCoreSetFlag<0x2," in source
