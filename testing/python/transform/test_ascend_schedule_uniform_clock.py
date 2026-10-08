# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Loop counts may vary by core group, never by participant or mutable data."""

import pytest
import tilelang  # noqa: F401
from tvm import tir


def fixture(extent):
    cid = tir.IterVar(None, tir.Var("cid", "int32"), tir.IterVar.ThreadIndex, "blockIdx.x")
    vid = tir.IterVar(None, tir.Var("vid", "int32"), tir.IterVar.ThreadIndex, "threadIdx.x")
    owner = tir.For(
        tir.Var("epoch", "int32"), 0, extent(cid.var, vid.var), tir.ForKind.SERIAL, tir.Evaluate(0))
    body = tir.AttrStmt(cid, "thread_extent", 12, tir.AttrStmt(vid, "thread_extent", 2, owner))
    return tir.PrimFunc([], body), owner, cid, vid


def test_causal_group_clock_is_bounded_without_becoming_static():
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    function, owner, cid, _ = fixture(
        lambda c, v: tir.Min(8, tir.floordiv((tir.floormod(c, 4) + 1) * 32 + 63, 64)))
    certificate = certify_uniform_clock(function, owner)
    assert certificate.extent_min == 1
    assert certificate.extent_max == 2
    for group, expected in ((0, 1), (1, 1), (2, 2), (3, 2), (4, 1)):
        actual = tir.stmt_functor.substitute(owner.extent, {cid.var: tir.IntImm("int32", group)})
        assert int(certificate.analyzer.simplify(actual)) == expected


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_empty_and_short_static_clocks_remain_legal(iterations):
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    function, owner, _, _ = fixture(lambda c, v: tir.IntImm("int32", iterations))
    certificate = certify_uniform_clock(function, owner)
    assert certificate.extent_min == certificate.extent_max == iterations


def test_loop_end_condition_must_not_overflow_even_if_last_index_fits():
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    function, owner, _, _ = fixture(lambda c, v: tir.IntImm("int32", 1))
    owner = tir.For(owner.loop_var, 2147483647, 1, tir.ForKind.SERIAL, owner.body)
    function = function.with_body(owner)
    with pytest.raises(ValueError, match="clock"):
        certify_uniform_clock(function, owner)


@pytest.mark.parametrize("bad",
                         ["vid", "load", "call", "unbound", "overflow", "zero_divisor", "negative"])
def test_unproven_clocks_fail_closed(bad):
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock

    def extent(c, v):
        if bad == "vid":
            return v + 1
        if bad == "load":
            return tir.decl_buffer((1,), "int32", name="mutable")[0]
        if bad == "call":
            return tir.call_extern("int32", "mutable_trip_count")
        if bad == "unbound":
            return tir.Var("unknown", "int32") + 1
        if bad == "overflow":
            return tir.Min(8, c * 1000000000 + 1)
        if bad == "zero_divisor":
            return tir.FloorDiv(c + 1, c)
        return c - 1

    function, owner, _, _ = fixture(extent)
    with pytest.raises(ValueError, match="clock"):
        certify_uniform_clock(function, owner)


def test_sibling_thread_range_cannot_certify_owner():
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    function, owner, cid, _ = fixture(lambda c, v: c + 1)
    sibling = tir.AttrStmt(cid, "thread_extent", 12, tir.Evaluate(0))
    function = function.with_body(tir.SeqStmt([sibling, owner]))
    with pytest.raises(ValueError, match="clock"):
        certify_uniform_clock(function, owner)


@pytest.mark.parametrize("same_var", [False, True])
def test_ambiguous_thread_binding_cannot_certify_clock(same_var):
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    function, owner, cid, _ = fixture(lambda c, v: c + 1)
    axis = tir.IterVar(None, cid.var if same_var else tir.Var("other_cid", "int32"),
                       tir.IterVar.ThreadIndex, "threadIdx.x" if same_var else "blockIdx.x")
    function = function.with_body(tir.AttrStmt(axis, "thread_extent", 2, function.body))
    with pytest.raises(ValueError, match="clock"):
        certify_uniform_clock(function, owner)


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_dynamic_clock_reaches_native_three_phase_emission(platform):
    import tvm
    import tilelang.language as T
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from tilelang.engine.phase import OptimizeForTarget
    function, original, cid, vid = fixture(lambda c, v: tir.Min(8, c + 1))
    acc = tir.decl_buffer((8,), "float32", name="acc", scope="shared.ub")
    ones = tir.decl_buffer((8,), "float32", name="ones", scope="shared.ub")
    output = tir.decl_buffer((96,), "float32", name="output", scope="global")
    handle = tir.Var("output_handle", "handle")
    owner = tir.For(
        original.loop_var, 0, original.extent, tir.ForKind.SERIAL,
        tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_add"), acc.access_ptr("w"),
                            acc.access_ptr("r"), ones.access_ptr("r"), 8)))
    body = tir.SeqStmt([
        tir.Evaluate(T.tile.fill(acc, 0)),
        tir.Evaluate(T.tile.fill(ones, 1)), owner,
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm<float,8>", acc.access_ptr("r"),
                            output.access_ptr("w", offset=cid.var * 8), 8))
    ])
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, body),
                     alloc_buffers=[acc, ones])
    function = tir.PrimFunc([handle],
                            tir.AttrStmt(
                                cid, "thread_extent", 12,
                                tir.AttrStmt(vid, "thread_extent", 2,
                                             tir.BlockRealize([], True, root))),
                            buffer_map={
                                handle: output
                            }).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = schedule_module(
        tvm.IRModule({"probe": function.with_attr("global_symbol", "probe")}), "ascendc",
        {"tl.ascend_auto_schedule": True})
    owners = []
    tir.stmt_functor.post_order_visit(
        mod["probe"].body, lambda node: owners.append(node) if isinstance(node, tir.For) else None)
    assert len(owners) == 1
    assert tvm.ir.structural_equal(owners[0].extent, owner.extent)
    with tvm.transform.PassContext(config={"tl.ascend_auto_schedule": True}):
        native = OptimizeForTarget(tir.transform.BindTarget(target)(mod), target, platform)
    source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                 platform).get_source()
    assert "SetFlag<" in source and "WaitFlag<" in source
    assert "for (" in source


def test_version_bounds_bind_core_range_before_owner_extent():
    from tilelang.transform.ascend_schedule.buffer_versions import materialize_iteration_versions
    from tilelang.transform.ascend_schedule.ownership import EpochPlan, OwnerEpochPlan
    buffer = tir.decl_buffer((16, 16), "float16", name="scratch", scope="wmma.matrix_a")
    function, original, cid, vid = fixture(lambda c, v: c + 1)
    pointer = buffer.access_ptr("w")
    owner = tir.For(original.loop_var, 0, original.extent, tir.ForKind.SERIAL,
                    tir.Evaluate(tir.call_extern("handle", "probe", pointer)))
    function = function.with_body(
        tir.AttrStmt(
            cid, "thread_extent", 12,
            tir.AttrStmt(
                vid, "thread_extent", 2,
                tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"),
                             owner))))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(owner, "iteration", object()),))
    result = materialize_iteration_versions(function, buffer, plan)
    allocation = result.body.body.body
    assert [int(value) for value in allocation.extents] == [32, 16]
    assert allocation.body.extent.same_as(owner.extent)
