# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native multistage C/V bundles with physically rotated generated workspaces.

Source-generation tests only: the fixture's L0C input is externally initialized.
This is not a numerical FA implementation or an automatic eligibility proof.
"""

from math import prod
import pytest
import tilelang
import tvm
from tvm import tir

from tilelang.transform.ascend_schedule.buffer_versions import materialize_iteration_versions
from tilelang.transform.ascend_schedule.cross_sync import emit_mode2_loop_sync
from tilelang.transform.ascend_schedule.dependency import analyze_dependencies, validate_dependencies
from tilelang.transform.ascend_schedule.ownership import EpochPlan, OwnerEpochPlan
from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
from tilelang.transform.ascend_schedule.workspace_proof import certify_workspace_versions
from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies


def fixture(platform, trips):
    acc = tir.decl_buffer((16, 16), "float32", name="acc", scope="wmma.accumulator")
    ub = tir.decl_buffer((8, 16), "float16", name="ub", scope="shared.ub")
    out = tir.decl_buffer((8, 16), "float16", name="out", scope="shared.ub")
    l1 = tir.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    l0 = tir.decl_buffer((16, 16), "float16", name="l0", scope="wmma.matrix_a")
    score = tir.Evaluate(
        tir.call_extern("handle",
                        "tl::ascend::copy_l0c_to_ub<float,half,layout::RowMajor,16,16,false>",
                        acc.access_ptr("r"), ub.access_ptr("w"), 16, 16, 8, 16))
    vector = tir.Evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_add"), out.access_ptr("w"),
                        ub.access_ptr("r"), ub.access_ptr("r"), 128))
    probability = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_ub_to_l1<half,16,8>", out.access_ptr("r"),
                        l1.access_ptr("w"), 16, 8, 16, 16))
    consume = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_l1_to_l0a<half,16,16,false>",
                        l1.access_ptr("r"), l0.access_ptr("w"), 16, 16))
    owner = tir.For(
        tir.Var("epoch", "int32"), 0, trips, tir.ForKind.SERIAL,
        tir.SeqStmt([
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, score),
            tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, vector),
            tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, probability),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, consume)
        ]))
    body = tir.BlockRealize([], True,
                            tir.Block([], [], [],
                                      "root",
                                      owner,
                                      alloc_buffers=[acc, ub, out, l1, l0]))
    for tag, count in (("threadIdx.x", 2), ("blockIdx.x", 1)):
        axis = tir.IterVar(None, tir.Var(tag.replace(".", "_"), "int32"), tir.IterVar.ThreadIndex,
                           tag)
        body = tir.AttrStmt(axis, "thread_extent", count, body)
    function = tir.PrimFunc([], body).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = tir.transform.BindTarget(target)(tvm.IRModule({"probe": function}))
    prepared = prepare_cross_core_copies(tilelang.transform.BufferShapeCollector()(mod))["probe"]
    assert len(prepared.params) == 2
    return prepared, target


def owner_of(function):
    owners = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda n: owners.append(n) if isinstance(n, tir.For) else None)
    assert len(owners) == 1
    return owners[0]


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("trips", [0, 1, 2, 5])
def test_three_stage_native_cv_workspaces_keep_slots_and_mode2_protocols(platform, trips):
    function, target = fixture(platform, trips)
    # Deliberately supplied physical-version plans for this native fixture.
    # Production discovery must prove collective writes rather than using this
    # explicit eligible=True token as evidence of full overwrite.
    for handle in function.params:
        buffer = function.buffer_map[handle]
        owner = owner_of(function)
        plan = EpochPlan(buffer.data, True, 3, (OwnerEpochPlan(owner, "iteration", object()),))
        function = materialize_iteration_versions(function, buffer, plan)
    owner = owner_of(function)
    extraction = extract_tasks(owner.body)
    extraction, diagnostics = certify_workspace_versions(function, extraction)
    assert not diagnostics
    assert [(t.core, t.pipe) for t in extraction.tasks] == [("C", "PIPE_FIX"), ("V", "PIPE_MTE2"),
                                                            ("V", "PIPE_V"), ("V", "PIPE_MTE3"),
                                                            ("C", "PIPE_MTE2"), ("C", "PIPE_MTE1")]
    dependencies = analyze_dependencies(extraction.tasks)
    validate_dependencies(extraction.tasks, dependencies)
    counts = {}
    for edge in dependencies:
        accesses = [a for t in extraction.tasks for a in t.accesses if a.storage is edge.storage]
        ring = max((a.ring_size or 1 for a in accesses), default=1)
        if ring > 1:
            counts[(edge.source, edge.target, edge.distance)] = ring
    synchronized = emit_mode2_loop_sync(
        function,
        owner,
        extraction,
        dependencies,
        cross_ids=tuple(range(16)),
        local_ids=tuple(range(8)),
        slot_counts=counts)
    if trips == 2:
        from tilelang.transform.ascend_schedule.local_sync import EventCapacityError
        # This three-slot fixture uses twelve cross channels; it is not a
        # proof of admission under the production eight-channel default.
        with pytest.raises(EventCapacityError, match="capacity exhausted"):
            emit_mode2_loop_sync(
                function,
                owner,
                extraction,
                dependencies,
                cross_ids=tuple(range(8)),
                local_ids=tuple(range(8)),
                slot_counts=counts)
    result = materialize_staged_owner(synchronized, owner_of(synchronized), extraction.task_nodes, {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2
    }, dependencies)
    if trips == 2:
        from tilelang.transform.ascend_schedule.stage_skew import TASK_BUNDLE_ATTR

        def lose_bundle_boundary(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == TASK_BUNDLE_ATTR:
                return node.body
            return None

        flattened = synchronized.with_body(
            tir.stmt_functor.ir_transform(synchronized.body, None, lose_bundle_boundary))
        with pytest.raises(ValueError, match="orphan synchronization"):
            materialize_staged_owner(flattened, owner_of(flattened), extraction.task_nodes, {
                0: 0,
                1: 1,
                2: 1,
                3: 1,
                4: 2,
                5: 2
            }, dependencies)
    assert int(owner_of(result).extent) == trips + 2
    assert all(prod(map(int, result.buffer_map[h].shape)) == 768 for h in result.params)
    native = tvm.IRModule({"probe": result.with_attr("global_symbol", "probe")})
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        native = tilelang.transform.AscendLowerOpaqueBlock()(native)
        native = tilelang.transform.Flatten2DBuffer()(native)
        native = tilelang.transform.FlattenBuffer()(native)
        native = tilelang.transform.AscendMemoryPlanning()(native)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    assert "epoch_wave" in source
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>" in source
    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>" in source
    assert "CrossCoreWaitFlag" in source
    assert "copy_l0c_to_gm" in source and "copy_gm_to_ub" in source
    assert "copy_ub_to_gm" in source and "copy_gm_to_l1" in source
    # GM is a runtime parameter, not a local GetWithOffset allocation. Its
    # expanded 768-element capacity is asserted in buffer_map above; native
    # address expressions must contain the original 256-element slot pitch.
    assert "256" in source
    gm_reads = [line for line in source.splitlines() if "copy_gm_to_ub<" in line]
    gm_writes = [line for line in source.splitlines() if "copy_ub_to_gm<" in line]
    cube_reads = [line for line in source.splitlines() if "copy_gm_to_l1<" in line]
    assert len(gm_reads) == len(gm_writes) == len(cube_reads) == 1
    if trips == 1:
        # Bounds prove the only active logical epoch is zero.
        assert all("epoch_wave" not in line for line in (*gm_reads, *gm_writes, *cube_reads))
    else:
        assert all("epoch_wave - 1" in line for line in (*gm_reads, *gm_writes))
        assert "epoch_wave - 2" in cube_reads[0]
    if trips >= 3:
        assert all("tl_floormod(((int64_t)(epoch_wave - 1)), (int64_t)3)" in line
                   for line in (*gm_reads, *gm_writes))
        assert "tl_floormod(((int64_t)(epoch_wave - 2)), (int64_t)3)" in cube_reads[0]
    assert f"(epoch_wave >= 1) && (epoch_wave < {trips + 1})" in source
    assert f"(epoch_wave >= 2) && (epoch_wave < {trips + 2})" in source
    # A removed physical reuse edge must fail graph validation before emission.
    carried = next(d for d in dependencies if d.distance == 3 and d.kind == "WAR")
    with pytest.raises(ValueError, match="missing dependency"):
        validate_dependencies(extraction.tasks, [d for d in dependencies if d != carried])
