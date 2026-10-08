# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Generated workspace candidates retain typed group coverage, not overwrite hints."""

import pytest

from test_ascend_schedule_workspace_epoch import prepared
from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks


def discover():
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function = prepared(3)
    extraction = extract_tasks(function)
    return function, extraction, discover_workspace_epochs(function, extraction)


def test_discovery_does_not_invent_whole_world_overwrite():
    function, extraction, discovery = discover()
    assert not discovery.diagnostics and len(discovery.candidates) == 2
    for candidate in discovery.candidates:
        assert candidate.size_bytes == 3 * candidate.group_bytes == 1536
        assert candidate.writer_id < min(candidate.reader_ids)
        assert candidate.clock.extent_min == candidate.clock.extent_max == 3
        assert candidate.certificate.owner.same_as(candidate.owner)
    # Neither task nor metadata is rewritten by discovery.
    assert tuple(discovery.tasks) == tuple(extraction.tasks)
    assert not any(access.full_overwrite for task in discovery.tasks for access in task.accesses)


@pytest.mark.parametrize("reader_stage", [1, 2, 4])
def test_partition_lifecycle_uses_exact_stage_span_and_world_pitch(reader_stage):
    from tilelang.transform.ascend_schedule.workspace_planning import plan_workspace_epochs
    _, extraction, discovery = discover()
    candidate = discovery.candidates[0]
    stages = {task.id: 0 for task in extraction.tasks}
    for task_id in candidate.reader_ids:
        stages[task_id] = reader_stage
    plan = plan_workspace_epochs(candidate, stages)
    assert plan.eligible and plan.count == reader_stage + 1
    assert len(plan.owners) == 1 and plan.owners[0].loop.same_as(candidate.owner)
    assert plan.owners[0].clock is candidate.clock and plan.owners[0].mode == "iteration"


@pytest.mark.parametrize(
    "bad",
    ["reader_before_writer", "missing_reader", "bool_stage", "negative_stage", "too_few_slots"])
def test_invalid_stage_assignment_cannot_authorize_versions(bad):
    from tilelang.transform.ascend_schedule.workspace_planning import plan_workspace_epochs
    _, extraction, discovery = discover()
    candidate = discovery.candidates[0]
    stages = {task.id: 0 for task in extraction.tasks}
    for task_id in candidate.reader_ids:
        stages[task_id] = 1
    count = None
    if bad == "reader_before_writer":
        stages[candidate.writer_id] = 2
    elif bad == "missing_reader":
        del stages[candidate.reader_ids[0]]
    elif bad == "bool_stage":
        stages[candidate.writer_id] = False
    elif bad == "negative_stage":
        stages[candidate.writer_id] = -1
    else:
        stages[candidate.reader_ids[0]] = 2
        count = 2
    with pytest.raises(ValueError, match="workspace"):
        plan_workspace_epochs(candidate, stages, count=count)


def test_raw_parameter_escape_prevents_candidate_admission():
    from tvm import tir
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function = prepared(3)
    handle = function.params[0]
    escape = tir.Evaluate(tir.call_extern("handle", "unknown_escape", handle))
    function = function.with_body(tir.SeqStmt([function.body, escape]))
    # Use the pre-escape extraction deliberately: the function-level complete
    # endpoint proof must detect escapes even when metadata omitted them.
    # IDs/data identities must come from the same function, without the escape.
    clean = function.with_body(function.body.seq[0])
    discovery = discover_workspace_epochs(function, extract_tasks(clean))
    assert len(discovery.candidates) == 1
    assert any("escape" in message for message in discovery.diagnostics)


@pytest.mark.parametrize("bad", ["core", "pipe", "private_gm_domain"])
def test_execution_domain_metadata_must_match_native_endpoints(bad):
    from dataclasses import replace
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function, extraction, baseline = discover()
    candidate = baseline.candidates[0]
    if bad == "private_gm_domain":
        private = replace(candidate.storage, domain="C")
        tasks = tuple(
            replace(
                task,
                accesses=tuple(
                    replace(access, storage=private) if access.storage is
                    candidate.storage else access
                    for access in task.accesses))
            for task in extraction.tasks)
    else:
        tasks = tuple(
            replace(task, **
                    {"core" if bad == "core" else "pipe": "V" if bad == "core" else "PIPE_V"}
                   ) if task.id == candidate.writer_id else task for task in extraction.tasks)
    discovery = discover_workspace_epochs(function, replace(extraction, tasks=tasks))
    assert len(discovery.candidates) == 1 and discovery.diagnostics


@pytest.mark.parametrize("hint", ["disable_versions", "counter_mode", "stage"])
def test_explicit_constraints_do_not_enter_unconstrained_workspace_discovery(hint):
    from tvm import tir
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function = prepared(3)
    if hint == "stage":
        function = function.with_body(
            tir.AttrStmt(tir.IntImm("int32", 0), "tl.ascend_stage", 1, function.body))
    else:
        buffer = function.buffer_map[function.params[0]]
        key = "tl.buffer_versions_map" if hint == "disable_versions" else "tl.buffer_version_mode"
        value = 1 if hint == "disable_versions" else "counter"

        def annotate(node):
            if not isinstance(node, tir.Block):
                return None
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations={
                    **dict(node.annotations), key: {
                        buffer.data: value
                    }
                })

        function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, annotate))
    discovery = discover_workspace_epochs(function, extract_tasks(function))
    assert not discovery.candidates
    assert any("explicit" in message for message in discovery.diagnostics)


def test_address_planned_input_needs_async_alias_contract():
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function = prepared(3).with_attr("address_map", {"scratch": 0})
    with pytest.raises(ValueError, match="alias contract"):
        discover_workspace_epochs(function, extract_tasks(function))


@pytest.mark.parametrize("hint", ["stage", "disable_versions"])
def test_stale_extraction_cannot_hide_current_function_constraints(hint):
    from tvm import tir
    from tilelang.transform.ascend_schedule.workspace_planning import discover_workspace_epochs
    function = prepared(3)
    extraction = extract_tasks(function)
    if hint == "stage":
        function = function.with_body(
            tir.AttrStmt(tir.IntImm("int32", 2), "tl.ascend_stage", 1, function.body))
    else:
        buffer = function.buffer_map[function.params[0]]

        def annotate(node):
            if not isinstance(node, tir.Block):
                return None
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations={"tl.buffer_versions_map": {
                    buffer.data: 1
                }})

        function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, annotate))
    discovery = discover_workspace_epochs(function, extraction)
    assert not discovery.candidates
    assert any("explicit" in message for message in discovery.diagnostics)
