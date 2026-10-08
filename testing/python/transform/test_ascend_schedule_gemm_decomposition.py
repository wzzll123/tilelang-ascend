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
