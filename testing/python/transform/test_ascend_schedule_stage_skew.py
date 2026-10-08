# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Executable logical epochs, including synchronization operands, not reports."""

import pytest
import tilelang  # noqa: F401
from tvm import tir, arith
from tilelang.transform.ascend_schedule.model import Dependency


def fixture(trips, slots=2):
    epoch = tir.Var("epoch", "int32")
    nodes = {
        i: tir.Evaluate(tir.call_extern("handle", f"op_{i}", epoch, epoch % slots)) for i in range(4)
    }
    groups = []
    for node in nodes.values():
        wait = tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_flag"), "MTE2", "MTE1",
                            epoch % slots))
        ready = tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_flag"), "MTE2", "MTE1",
                            epoch % slots))
        groups.append(
            tir.AttrStmt(
                tir.IntImm("int32", 0), "resource_scope", 0, tir.SeqStmt([wait, node, ready])))
    owner = tir.For(epoch, 0, trips, tir.ForKind.SERIAL, tir.SeqStmt(groups))
    function = tir.PrimFunc([], owner)
    dependencies = [
        Dependency(0, 1, "RAW"),
        Dependency(2, 3, "RAW"),
        Dependency(1, 0, "WAR", 2),
        Dependency(3, 2, "WAR", 2)
    ]
    return function, owner, nodes, dependencies


@pytest.mark.parametrize("trips", [0, 1, 2, 5])
@pytest.mark.parametrize("stages,slots", [({
    0: 0,
    1: 1,
    2: 2,
    3: 2
}, 3), ({
    0: 0,
    1: 2,
    2: 3,
    3: 3
}, 4)])
def test_multistage_epochs_and_flag_slots_cover_each_logical_task_once(trips, stages, slots):
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, _ = fixture(trips, slots)
    dependencies = [
        Dependency(0, 1, "RAW"),
        Dependency(1, 2, "RAW"),
        Dependency(2, 3, "RAW"),
        Dependency(3, 0, "WAR", slots)
    ]
    result = materialize_staged_owner(function, owner, nodes, stages, dependencies)
    wave = result.body
    assert int(wave.extent) == trips + max(stages.values())
    events = []
    analyzer = arith.Analyzer()
    for w in range(int(wave.extent)):
        bindings = {wave.loop_var: tir.IntImm("int32", w)}
        for guard in wave.body.seq:
            if not int(analyzer.simplify(tir.stmt_functor.substitute(guard.condition, bindings))):
                continue
            calls = []
            tir.stmt_functor.post_order_visit(
                guard.then_case, lambda n: calls.append(n.value)
                if isinstance(n, tir.Evaluate) else None)
            operation = calls[1]
            epoch = int(analyzer.simplify(tir.stmt_functor.substitute(operation.args[1], bindings)))
            op_slot = int(
                analyzer.simplify(tir.stmt_functor.substitute(operation.args[2], bindings)))
            assert op_slot == epoch % slots
            for flag in (calls[0], calls[2]):
                assert int(analyzer.simplify(tir.stmt_functor.substitute(flag.args[2],
                                                                         bindings))) == op_slot
            events.append((str(operation.args[0].value), epoch))
    assert sorted(events) == sorted((f"op_{i}", e) for e in range(trips) for i in range(4))
    if trips > max(stages.values()):
        assert events.index(("op_0", 1)) < events.index(("op_3", 0))


def test_multistage_reuse_at_equal_wave_is_rejected_in_actual_bundle_order():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, _ = fixture(5)
    with pytest.raises(ValueError, match="dependence"):
        materialize_staged_owner(function, owner, nodes, {
            0: 0,
            1: 1,
            2: 2,
            3: 2
        }, [Dependency(3, 0, "WAR", 2)])


def test_multistage_wave_extent_checks_maximum_stage_not_only_one():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, _ = fixture(2147483646, 3)
    with pytest.raises(ValueError, match="overflow"):
        materialize_staged_owner(function, owner, nodes, {0: 0, 1: 1, 2: 2, 3: 2}, [])


@pytest.mark.parametrize("bad_stages", [{
    0: -1,
    1: 1,
    2: 2,
    3: 2
}, {
    0: False,
    1: 1,
    2: 2,
    3: 2
}, {
    0: 1,
    1: 2,
    2: 3,
    3: 3
}, {
    0: 0,
    1: 0,
    2: 0,
    3: 0
}])
def test_multistage_assignments_remain_fail_closed(bad_stages):
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, _ = fixture(5, 4)
    with pytest.raises(ValueError, match="stage"):
        materialize_staged_owner(function, owner, nodes, bad_stages, [])


def test_multistage_preserves_group_dependent_clock():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    cid = tir.IterVar(None, tir.Var("cid", "int32"), tir.IterVar.ThreadIndex, "blockIdx.x")
    trips = tir.floormod(cid.var, 4)
    function, owner, nodes, _ = fixture(trips, 3)
    function = function.with_body(tir.AttrStmt(cid, "thread_extent", 8, owner))
    result = materialize_staged_owner(function, owner, nodes, {
        0: 0,
        1: 1,
        2: 2,
        3: 2
    }, [Dependency(3, 0, "WAR", 3)])
    wave = result.body.body
    analyzer = arith.Analyzer()
    for group in range(8):
        group_binding = {cid.var: tir.IntImm("int32", group)}
        extent = int(analyzer.simplify(tir.stmt_functor.substitute(wave.extent, group_binding)))
        assert extent == group % 4 + 2
        counts = [0] * 4
        for w in range(extent):
            for index, guard in enumerate(wave.body.seq):
                bindings = {**group_binding, wave.loop_var: tir.IntImm("int32", w)}
                counts[index] += int(
                    analyzer.simplify(tir.stmt_functor.substitute(guard.condition, bindings)))
        assert counts == [group % 4] * 4


@pytest.mark.parametrize("trips", [0, 1, 2, 3])
def test_skew_moves_next_load_before_current_compute_and_flags_follow_epoch(trips):
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, dependencies = fixture(trips)
    result = materialize_staged_owner(function, owner, nodes, {
        0: 0,
        1: 1,
        2: 0,
        3: 1
    }, dependencies)
    wave = result.body
    assert int(wave.extent) == trips + 1
    events = []
    for w in range(trips + 1):
        for guard in wave.body.seq:
            condition = tir.stmt_functor.substitute(guard.condition,
                                                    {wave.loop_var: tir.IntImm("int32", w)})
            if not int(arith.Analyzer().simplify(condition)):
                continue
            calls = []
            tir.stmt_functor.post_order_visit(
                guard.then_case,
                lambda n, calls=calls: calls.append(n.value)
                if isinstance(n, tir.Evaluate) else None)
            operation = calls[1]
            logical_epoch = int(arith.Analyzer().simplify(
                tir.stmt_functor.substitute(operation.args[1],
                                            {wave.loop_var: tir.IntImm("int32", w)})))
            for flag in (calls[0], calls[2]):
                slot = int(arith.Analyzer().simplify(
                    tir.stmt_functor.substitute(flag.args[2],
                                                {wave.loop_var: tir.IntImm("int32", w)})))
                assert slot == logical_epoch % 2
            events.append((str(operation.args[0].value), logical_epoch))
    assert sorted(events) == sorted((f"op_{i}", e) for e in range(trips) for i in range(4))
    if trips > 1:
        assert events.index(("op_2", 1)) < events.index(("op_1", 0))


def test_one_version_reuse_edge_cannot_be_skewed_into_a_race():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, dependencies = fixture(3)
    dependencies[-1] = Dependency(3, 2, "WAR", 1)
    with pytest.raises(ValueError, match="dependence"):
        materialize_staged_owner(function, owner, nodes, {0: 0, 1: 1, 2: 0, 3: 1}, dependencies)


def test_unguarded_orphan_wait_is_not_accepted_as_an_operation_bundle():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, dependencies = fixture(3)
    orphan = tir.Evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_flag"), "MTE2", "MTE1", 0))
    owner = tir.For(owner.loop_var, 0, 3, owner.kind, tir.SeqStmt([orphan, *owner.body.seq]))
    function = function.with_body(owner)
    with pytest.raises(ValueError, match="bundle"):
        materialize_staged_owner(function, owner, nodes, {0: 0, 1: 1, 2: 0, 3: 1}, dependencies)


def test_wave_prologue_extent_cannot_overflow():
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, dependencies = fixture(2147483647)
    with pytest.raises(ValueError, match="overflow"):
        materialize_staged_owner(function, owner, nodes, {0: 0, 1: 1, 2: 0, 3: 1}, dependencies)


def test_actual_mode2_emitter_bundles_survive_wave_reconstruction():
    from types import SimpleNamespace
    from tilelang.transform.ascend_schedule.model import Task
    from tilelang.transform.ascend_schedule.cross_sync import emit_mode2_loop_sync
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, nodes, dependencies = fixture(3)
    body = tir.SeqStmt([
        tir.AttrStmt(tir.IntImm("int32", int(i == 3)), "resource_scope", int(i == 3), node)
        for i, node in nodes.items()
    ])
    owner = tir.For(owner.loop_var, 0, 3, owner.kind, body)
    vid = tir.IterVar(None, tir.Var("vid", "int32"), tir.IterVar.ThreadIndex, "threadIdx.x")
    function = function.with_body(tir.AttrStmt(vid, "thread_extent", 2, owner))
    extraction = SimpleNamespace(
        tasks=(Task(0, "MTE2", "C"), Task(1, "MTE1", "C"), Task(2, "FIX",
                                                                "C"), Task(3, "MTE2", "V")),
        task_nodes=nodes,
        loops={i: () for i in nodes},
        scalar_bindings=())
    dependencies[-1] = Dependency(3, 2, "WAR", 1)
    synchronized = emit_mode2_loop_sync(
        function,
        owner,
        extraction,
        dependencies,
        local_ids=tuple(range(8)),
        cross_ids=tuple(range(8)),
        slot_counts={
            (0, 1, 0): 2,
            (1, 0, 2): 2
        })
    loops = []
    tir.stmt_functor.post_order_visit(synchronized.body, lambda n: loops.append(n)
                                      if isinstance(n, tir.For) else None)
    result = materialize_staged_owner(synchronized, loops[0], nodes, {
        0: 0,
        1: 1,
        2: 1,
        3: 1
    }, dependencies)
    protocol = result.body.body
    assert isinstance(protocol, tir.SeqStmt)
    assert len(protocol.seq) > 1  # Seed/drain remain outside the wave loop.
    wave = next(node for node in protocol.seq if isinstance(node, tir.For))
    assert all(isinstance(node, tir.IfThenElse) for node in wave.body.seq)
    guarded_calls = []
    tir.stmt_functor.post_order_visit(
        wave.body, lambda n: guarded_calls.append(n.value.op.name)
        if isinstance(n, tir.Evaluate) else None)
    assert "tl.ascend_set_cross_flag" in guarded_calls
    assert "tl.ascend_wait_cross_flag" in guarded_calls
