# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Real TIR decomposition of the FA128 GEMM shape family."""

import pytest
import tilelang  # noqa: F401 - Register the native Ascend intrinsic operators.
from tvm import tir
from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_gemm_v0


def example(transpose_b=True):
    a = tir.decl_buffer((128, 128), "float16", scope="shared.l1")
    b = tir.decl_buffer((128, 128), "float16", scope="shared.l1")
    c = tir.decl_buffer((128, 128), "float32", scope="wmma.accumulator")
    x = tir.decl_buffer((128 * 128,), "float16", scope="wmma.matrix_a")
    y = tir.decl_buffer((128 * 128,), "float16", scope="wmma.matrix_b")
    template = f"gemm_v0<half,float,128,128,128,false,{str(transpose_b).lower()},128>"
    call = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_gemm_v0"), template,
                           a.access_ptr("r"), b.access_ptr("r"), c.access_ptr("rw"), False, 64)
    return call, x, y


@pytest.mark.parametrize("transpose_b", [True, False])
def test_three_actual_pipe_tasks_without_hidden_flags(transpose_b):
    call, a, b = example(transpose_b)
    result = decompose_gemm_v0(call, a, b)
    assert len(result.seq) == 3
    first, second, last = [node.value for node in result.seq]
    assert first.args[0].value == "tl::ascend::copy_l1_to_l0a<half,128,128,false>"
    expected = "tl::ascend::copy_l1_to_l0b<half,128,128" + (",true>" if transpose_b else ",false>")
    assert second.args[0].value == expected
    assert last.op.name == "tl.ascend_mma"
    assert last.args[4].same_as(call.args[4])
    assert int(last.args[5]) == 128
    assert int(last.args[6]) == (64 if transpose_b else 128)
    assert len(last.args) == 8 and int(last.args[7]) == 0
    assert first.args[1].same_as(call.args[1])
    assert second.args[1].same_as(call.args[2])
    assert last.args[3].same_as(call.args[3])


def test_scratch_capacity_must_be_checked():
    call, _, b = example()
    small = tir.decl_buffer((16,), "float16", scope="wmma.matrix_a")
    with pytest.raises(ValueError, match="capacity"):
        decompose_gemm_v0(call, small, b)


def test_scratch_cannot_be_same_storage():
    call, a, _ = example()
    with pytest.raises(ValueError):
        decompose_gemm_v0(call, a, a)


def test_operand_extent_must_cover_tile():
    call, a, b = example()
    original = call.args[1]
    short = tir.Call(
        original.dtype, original.op,
        [*original.args[:3], tir.IntImm("int32", 16), original.args[4]])
    invalid = tir.Call(call.dtype, call.op, [call.args[0], short, *call.args[2:]])
    with pytest.raises(ValueError, match="operand capacity"):
        decompose_gemm_v0(invalid, a, b)


@pytest.mark.parametrize("template", [
    "gemm_v0<half,float,128,128,128,false,true,256>",
    "gemm_v0<half,float,128,128,256,false,true,128>",
    "gemm_v0<float,float,128,128,128,false,true,128>",
    "gemm_v0<half,float,128,256,128,false,false,128>",
])
def test_outside_single_tile_family_fails_explicitly(template):
    call, a, b = example()
    invalid = tir.Call(call.dtype, call.op, [tir.StringImm(template), *call.args[1:]])
    with pytest.raises(ValueError):
        decompose_gemm_v0(invalid, a, b)


def test_decomposition_is_visible_to_dependency_and_sync_pipeline():
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    from tilelang.transform.ascend_schedule.dependency import analyze_dependencies
    from tilelang.transform.ascend_schedule.local_sync import emit_local_sync

    call, a, b = example()
    extraction = extract_tasks(decompose_gemm_v0(call, a, b))
    assert [task.pipe for task in extraction.tasks] == ["PIPE_MTE1", "PIPE_MTE1", "PIPE_M"]
    assert all(task.core == "C" for task in extraction.tasks)
    dependencies = analyze_dependencies(extraction.tasks, max_distance=0)
    assert {(edge.source, edge.target, edge.kind) for edge in dependencies} == {(0, 2, "RAW"),
                                                                                (1, 2, "RAW")}
    emitted = emit_local_sync(extraction, dependencies, event_ids=(0, 1))
    assert len(emitted.assignments) == 2
    assert all(event.source_pipe == "MTE1" and event.destination_pipe == "M"
               for event in emitted.assignments)


def test_function_decomposition_preserves_loop_stage_guard_and_attributes():
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms

    call, _, _ = example()
    i = tir.Var("i", "int32")
    predicate = tir.Var("predicate", "bool")
    body = tir.For(
        i, 0, 2, tir.ForKind.SERIAL,
        tir.AttrStmt(i, "tl.stage", 1, tir.IfThenElse(predicate, tir.Evaluate(call), None)))
    function = tir.PrimFunc([predicate], body).with_attr("global_symbol", "test")
    result = decompose_function_gemms(function)
    assert result.attrs["global_symbol"] == "test"
    assert result.body.loop_var.same_as(i)
    assert result.body.body.attr_key == "tl.stage"
    branch = result.body.body.body
    assert branch.condition.same_as(predicate)
    alloc_a = branch.then_case
    alloc_b = alloc_a.body
    assert isinstance(alloc_a, tir.Allocate) and isinstance(alloc_b, tir.Allocate)
    assert [int(x) for x in alloc_a.extents] == [128, 128]
    assert [int(x) for x in alloc_b.extents] == [128, 128]
    assert alloc_a.buffer_var.type_annotation.storage_scope == "wmma.matrix_a"
    assert alloc_b.buffer_var.type_annotation.storage_scope == "wmma.matrix_b"
    assert len(alloc_b.body.seq) == 3
    assert decompose_function_gemms(result).same_as(result)


def test_each_callsite_has_distinct_scratch_and_avoids_existing_names():
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms

    call, _, _ = example()
    occupied = tir.Var("auto_l0a_0", "int32")
    function = tir.PrimFunc([occupied], tir.SeqStmt([tir.Evaluate(call), tir.Evaluate(call)]))
    result = decompose_function_gemms(function)
    first, second = result.body.seq
    assert first.buffer_var.name != occupied.name
    assert not first.buffer_var.same_as(second.buffer_var)
    assert not first.body.buffer_var.same_as(second.body.buffer_var)


def test_no_gemm_preserves_function_identity():
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms

    function = tir.PrimFunc([], tir.Evaluate(0))
    assert decompose_function_gemms(function).same_as(function)


def test_root_scratch_hoist_is_explicit_distinct_and_has_carried_hazards():
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    from tilelang.transform.ascend_schedule.dependency import analyze_dependencies

    call, _, _ = example()
    epoch = tir.Var("epoch", "int32")
    owner = tir.For(epoch, 0, 2, tir.ForKind.SERIAL,
                    tir.SeqStmt([tir.Evaluate(call), tir.Evaluate(call)]))
    root = tir.Block([], [], [], "root", owner)
    function = tir.PrimFunc([], tir.BlockRealize([], True, root))
    result = decompose_function_gemms(function, root_scratch=True)
    buffers = result.body.block.alloc_buffers
    assert len(buffers) == 4
    assert len({buffer.data.name for buffer in buffers}) == 4
    assert [buffer.scope() for buffer in buffers] == ["wmma.matrix_a", "wmma.matrix_b"] * 2
    allocations = []
    tir.stmt_functor.post_order_visit(
        result.body, lambda node: allocations.append(node)
        if isinstance(node, tir.Allocate) else None)
    assert not allocations
    assert result.body.block.body.loop_var.same_as(epoch)
    extraction = extract_tasks(result)
    edges = analyze_dependencies(extraction.tasks, max_distance=1)
    # Hoisting creates real reuse across epochs; the MMA read must finish before
    # the next MTE1 load overwrites that same callsite's L0 scratch.
    assert {(edge.source, edge.target, edge.kind, edge.distance) for edge in edges
           } >= {(2, 0, "WAR", 1), (2, 1, "WAR", 1), (5, 3, "WAR", 1), (5, 4, "WAR", 1)}
    assert decompose_function_gemms(result, root_scratch=True).same_as(result)


def test_root_scratch_does_not_hoist_user_allocations_or_cross_conditional_roots():
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms

    call, _, _ = example()
    user = tir.decl_buffer((8,), "float16", name="auto_l0a_user", scope="wmma.matrix_a")
    root = tir.Block([], [], [], "root",
                     tir.Allocate(user.data, user.dtype, user.shape, tir.const(True, "bool"),
                                  tir.Evaluate(call)))
    result = decompose_function_gemms(
        tir.PrimFunc([], tir.BlockRealize([], True, root)), root_scratch=True)
    assert len(result.body.block.alloc_buffers) == 2
    assert result.body.block.body.buffer_var.same_as(user.data)
    predicate = tir.Var("predicate", "bool")
    conditional = tir.PrimFunc([predicate],
                               tir.IfThenElse(predicate, tir.BlockRealize([], True, root), None))
    with pytest.raises(ValueError, match="unconditional common root"):
        decompose_function_gemms(conditional, root_scratch=True)


def test_enabled_entry_decomposes_gemm_but_still_rejects_missing_materializer():
    from tvm import IRModule
    from tilelang.transform.ascend_schedule.entry import schedule_module, AutoScheduleError

    call, _, _ = example()
    function = tir.PrimFunc([], tir.Evaluate(call)).with_attr("npu_platform", "A2")
    mod = IRModule({"main": function})
    with pytest.raises(AutoScheduleError, match="unconditional common root"):
        schedule_module(mod, "ascendc", {"tl.ascend_auto_schedule": True})
    assert mod["main"].body.value.same_as(call)


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_new_scratch_is_discovered_by_existing_native_planner_and_codegen(platform):
    import tvm
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms

    a = tir.decl_buffer((16, 16), "float16", name="a", scope="shared.l1")
    b = tir.decl_buffer((16, 16), "float16", name="b", scope="shared.l1")
    c = tir.decl_buffer((16, 16), "float32", name="c", scope="wmma.accumulator")
    call = tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_gemm_v0"),
                           "gemm_v0<half,float,16,16,16,false,false,128>", a.access_ptr("r"),
                           b.access_ptr("r"), c.access_ptr("w"), True, 16)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, tir.Evaluate(call))
    block = tir.Block([], [], [], "root", body, alloc_buffers=[a, b, c])
    function = tir.PrimFunc([], tir.BlockRealize([], True, block)).with_attr(
        "global_symbol", "probe").with_attr("npu_platform", platform)
    mod = tvm.IRModule({"probe": function})
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        mod = tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.BufferShapeCollector()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        # Insert after the initial shape collection, exactly at the new entry.
        mod = tvm.IRModule({"probe": decompose_function_gemms(mod["probe"])})
        mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
        mod = tilelang.transform.Flatten2DBuffer()(mod)
        mod = tilelang.transform.FlattenBuffer()(mod)
        mod = tilelang.transform.AscendMemoryPlanning()(mod)
        source = tvm.get_global_func("target.build.tilelang_ascend")(mod, target,
                                                                     platform).get_source()
    assert "auto_l0a_0 = ascend_l0a.GetWithOffset<half>(256," in source
    assert "auto_l0b_0 = ascend_l0b.GetWithOffset<half>(256," in source
    assert "copy_l1_to_l0a<half,16,16,false>(auto_l0a_0[0], a[0], 16, 16)" in source
    assert "copy_l1_to_l0b<half,16,16,false>(auto_l0b_0[0], b[0], 16, 16)" in source
    assert "mma<half,float,16,16>(auto_l0a_0[0],auto_l0b_0[0],c[0]" in source
    assert "gemm_v0<" not in source
    # Source-only: this intentionally unsynchronized probe is never launched.
