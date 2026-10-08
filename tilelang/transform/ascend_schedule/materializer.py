# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Executable flat-epoch bootstrap for the structured scheduling materializer."""

import json
from dataclasses import replace
from .dependency import analyze_dependencies
from .local_sync import emit_local_sync, emit_loop_sync, LocalSyncError
from .cross_sync import emit_mode2_sync, emit_mode2_loop_sync
from .analysis_report import build_schedule_report
from .solver import TaskCost, StorageCost
from .resources import allocation_budgets

MARKER = "tl.ascend.schedule_materialized"


class MaterializationError(ValueError):
    """The requested structure has no complete executable materialization yet."""


def _validate_cv_owner(function, loop, extraction):
    """Prove participant context and exact task ownership, not just presence."""
    from tvm import tir

    if not isinstance(loop.min, tir.IntImm) or not isinstance(loop.extent, tir.IntImm):
        raise MaterializationError("CV owner bounds require a uniform static clock")
    paths = []

    def find(node, path):
        if node.same_as(loop):
            paths.append(path)
            return
        children = []
        if isinstance(node, (tir.AttrStmt, tir.Allocate, tir.Block)):
            children = [node.body]
        elif isinstance(node, tir.BlockRealize):
            if not isinstance(node.predicate, tir.IntImm) or not int(node.predicate):
                raise MaterializationError("predicated allocation scopes need guard reconstruction")
            children = [node.block]
        elif isinstance(node, tir.SeqStmt):
            children = list(node.seq)
        for child in children:
            find(child, (*path, node))

    find(function.body, ())
    if len(paths) != 1:
        raise MaterializationError("CV owner has no unique allocation/thread scope path")
    ancestors = paths[0]
    if any(
            isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope"
            for node in ancestors):
        raise MaterializationError("shared CV owner cannot be enclosed by a single-core scope")
    if not any(
            isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and
            isinstance(node.node, tir.IterVar) and node.node.thread_tag in
        ("threadIdx.x",
         "blockIdx.y") and isinstance(node.value, tir.IntImm) and int(node.value) == 2
            for node in ancestors):
        raise MaterializationError("two-Vector participant scope must dominate the CV owner")
    seen = []
    by_id = {task.id: task for task in extraction.tasks}

    def validate(node, core=None):
        if isinstance(node, tir.SeqStmt):
            for child in node.seq:
                validate(child, core)
        elif (isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope" and
              isinstance(node.value, tir.IntImm) and int(node.value) in (0, 1)):
            validate(node.body, "C" if int(node.value) == 0 else "V")
        elif isinstance(node, tir.Evaluate):
            task_id = next(
                (i for i, original in extraction.task_nodes.items() if node.same_as(original)),
                None)
            if task_id is None or core != by_id[task_id].core or task_id in seen:
                raise MaterializationError(
                    "CV owner task must occur once in its matching core scope")
            seen.append(task_id)
        else:
            raise MaterializationError("CV owner body requires flat scoped tasks")

    validate(loop.body)
    if seen != [task.id for task in extraction.tasks]:
        raise MaterializationError("CV owner does not cover its complete lexical task stream")


def materialize_flat_function(function, extraction, platform):
    """Emit an unconditional single epoch with disjoint allocation contracts.

    Unit costs test feasibility only: they are not calibrated NPU timings.
    Real data fences and same-pipe FIFO implement execution ordering. General
    guarded/cyclic schedule reconstruction remains an explicit error, not a
    silent fallback to legacy passes.
    """
    from tvm import DataType, tir

    tasks = tuple(extraction.tasks)
    if (not isinstance(function, tir.PrimFunc) or not tasks or extraction.scalar_bindings or
            any(task.guard not in (None, ()) for task in tasks)):
        raise MaterializationError("structured loop/guard/scalar materializer is not integrated")
    owner_loop = None
    phased = any(extraction.loops.values()) and any(
        not context for context in extraction.loops.values())
    if any(extraction.loops.values()) and not phased:
        contexts = [extraction.loops[task.id] for task in tasks]
        if (any(len(context) != 1 for context in contexts) or
                any(not context[0].same_as(contexts[0][0]) for context in contexts)):
            raise MaterializationError("only a single unconditional owner loop is integrated")
        owner_loop = contexts[0][0]
        if owner_loop.kind != tir.ForKind.SERIAL:
            raise MaterializationError("owner loop must be serial")
        nodes = [extraction.task_nodes[task.id] for task in tasks]
        expected = nodes[0] if len(nodes) == 1 else tir.SeqStmt(nodes)
        from tvm import ir
        if len({task.core for task in tasks}) > 1:
            _validate_cv_owner(function, owner_loop, extraction)
        elif not ir.structural_equal(owner_loop.body, expected):
            raise MaterializationError("owner loop must contain only its extracted flat tasks")
        loop_nodes = []
        tir.stmt_functor.post_order_visit(
            function.body, lambda node: loop_nodes.append(node)
            if isinstance(node, tir.For) else None)
        if len(loop_nodes) != 1:
            raise MaterializationError("additional loops need structured lifetime reconstruction")
    if any(stage != 0 for stage in extraction.stages.values()):
        raise MaterializationError("nonzero Stage requires an owner-loop materializer")
    for annotation in extraction.annotations:
        if hasattr(annotation, "keys") and any(
                str(key).startswith("tl.buffer_version") for key in annotation.keys()):
            records = function.attrs.get("tl.ascend.iteration_versions", {})
            selected = function.attrs.get("tl.ascend.selected_buffer_versions", {})
            for var, count in annotation.get("tl.buffer_versions_map", {}).items():
                if int(count) == 1 and var in selected and int(selected[var]) == 1:
                    continue
                if var not in records or int(records[var][0]) != int(count):
                    raise MaterializationError("buffer version hints require owner materialization")
            for var, mode in annotation.get("tl.buffer_version_mode", {}).items():
                if (var not in records and
                        var not in selected) or str(mode) not in ("auto", "iteration"):
                    raise MaterializationError("buffer version hints require owner materialization")
    if function.attrs is not None and function.attrs.get("address_map", {}):
        raise MaterializationError(
            "physical addresses need a validated asynchronous alias contract")
    allocations = []
    scalar_variables = [var for var in function.params if str(var.dtype) != "handle"]
    if owner_loop is not None:
        scalar_variables.append(owner_loop.loop_var)
    if phased:
        scalar_variables.extend(
            context[0].loop_var for context in extraction.loops.values() if context)
    scope_names = {
        "shared.ub": "UB",
        "shared.l1": "L1",
        "wmma.matrix_a": "L0A",
        "wmma.matrix_b": "L0B",
        "wmma.accumulator": "L0C"
    }

    def add(var, dtype, shape):
        if any(var.same_as(existing[0]) for existing in allocations):
            raise MaterializationError("duplicate physical allocation identity")
        scope = getattr(var.type_annotation, "storage_scope", "")
        if scope not in scope_names or any(
                not isinstance(dim, tir.IntImm) or int(dim) < 1 for dim in shape):
            raise MaterializationError("allocation scope/shape lacks a static capacity proof")
        elements = 1
        for dim in shape:
            size = int(dim)
            elements *= ((size + 15) // 16 *
                         16) if scope != "shared.ub" and len(shape) == 2 else size
        ty = DataType(dtype)
        byte_count = elements * ty.bits * ty.lanes // 8
        allocations.append((var, scope_names[scope], (byte_count + 31) // 32 * 32))

    def collect(node):
        if isinstance(node, tir.For):
            scalar_variables.append(node.loop_var)
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and isinstance(
                node.node, tir.IterVar):
            scalar_variables.append(node.node.var)
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                add(buffer.data, buffer.dtype, buffer.shape)
        elif isinstance(node, tir.Allocate):
            add(node.buffer_var, node.dtype, node.extents)

    tir.stmt_functor.post_order_visit(function.body, collect)
    known = [item[0] for item in allocations
            ] + [buffer.data for buffer in function.buffer_map.values()]
    if any(not any(access.storage.var.same_as(var)
                   for var in known + scalar_variables)
           for task in tasks
           for access in task.accesses):
        raise MaterializationError("unallocated storage needs a physical lifetime contract")
    permitted = {
        "C": {"shared.l1", "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"},
        "V": {"shared.ub"}
    }
    parameters = [buffer.data for buffer in function.buffer_map.values()]
    for task in tasks:
        for access in task.accesses:
            var = access.storage.var
            if str(var.dtype) != "handle" or any(
                    var.same_as(parameter) for parameter in parameters):
                continue
            scope = str(getattr(var.type_annotation, "storage_scope", ""))
            if scope not in permitted.get(task.core, set()):
                raise MaterializationError(
                    "on-chip storage is inaccessible from its task core domain")
    slot_count = max((access.ring_size or 1 for task in tasks for access in task.accesses),
                     default=1)
    dependencies = list(
        analyze_dependencies(tasks, max_distance=slot_count if owner_loop is not None else 0))
    if phased:
        body_tasks = tuple(task for task in tasks if extraction.loops[task.id])
        dependencies.extend(
            dep for dep in analyze_dependencies(body_tasks, max_distance=1) if dep.distance)
    # Preserve actual hardware FIFO in the feasibility model as well as emission.
    from .model import Dependency
    last = {}
    for task in tasks:
        key = task.core, task.pipe
        if key in last:
            dependencies.append(Dependency(last[key], task.id, "RAW"))
        last[key] = task.id
    costs = {task.id: TaskCost(1, 1, task.core + ":" + task.pipe) for task in tasks}
    storage = {
        f"allocation_{i}": StorageCost(size, scope, max_versions=1, fixed_versions=1)
        for i, (_, scope, size) in enumerate(allocations)
    }
    explicit_stages = any(
        getattr(annotation, "attr_key", "") == "tl.ascend_stage"
        for annotation in extraction.annotations)
    report = build_schedule_report(
        extraction,
        dependencies,
        costs,
        stages=extraction.stages if explicit_stages else None,
        storage=storage,
        capacities=allocation_budgets(platform),
        resource_model=platform)
    report.update(
        materialization="prefix/owner/suffix phases" if phased else
        ("single CV owner loop" if len({task.core for task in tasks}) > 1 else
         "single local owner loop") if owner_loop is not None else "flat local/CV epoch",
        timing="unit-cost feasibility; not calibrated NPU timing",
        alias_policy="disjoint physical allocations until async alias contracts are integrated")
    if phased:
        from .phases import emit_three_phase_sync
        try:
            result = emit_three_phase_sync(function, extraction)
        except LocalSyncError as error:
            raise MaterializationError(str(error)) from error
    elif owner_loop is not None:
        body_extraction = replace(extraction, loops={task.id: () for task in tasks})
        try:
            if len({task.core for task in tasks}) > 1:
                result = emit_mode2_loop_sync(
                    function,
                    owner_loop,
                    body_extraction,
                    dependencies,
                    cross_ids=tuple(range(8)),
                    local_ids=tuple(range(8)))
                return result.with_attr(MARKER, 1).with_attr("tl.ascend.schedule_report",
                                                             json.dumps(report, sort_keys=True))
            synchronized = emit_loop_sync(
                body_extraction,
                owner_loop,
                dependencies,
                event_ids=tuple(range(8)),
                slot_count=slot_count)
        except LocalSyncError as error:
            raise MaterializationError(str(error)) from error

        def rewrite_loop(node):
            return synchronized if node.same_as(owner_loop) else None

        result = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, rewrite_loop))
    elif len({task.core for task in tasks}) > 1:
        result = emit_mode2_sync(
            function,
            extraction,
            dependencies,
            cross_ids=tuple(range(8)),
            local_ids=tuple(range(8)))
    else:
        emitted = emit_local_sync(extraction, dependencies, event_ids=tuple(range(8)))
        before, after = {task.id: [] for task in tasks}, {task.id: [] for task in tasks}
        for assignment in emitted.assignments:
            for kind, task_id, placement in (("wait", assignment.target, before),
                                             ("set", assignment.source, after)):
                placement[task_id].append(
                    tir.Evaluate(
                        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_" + kind + "_flag"),
                                        assignment.source_pipe, assignment.destination_pipe,
                                        assignment.event_id)))

        def rewrite(node):
            task_id = next(
                (i for i, original in extraction.task_nodes.items() if node.same_as(original)),
                None)
            if task_id is None:
                return None
            sequence = [*before[task_id], node, *after[task_id]]
            return sequence[0] if len(sequence) == 1 else tir.SeqStmt(sequence)

        result = function.with_body(tir.stmt_functor.ir_transform(function.body, None, rewrite))
    return result.with_attr(MARKER, 1).with_attr("tl.ascend.schedule_report",
                                                 json.dumps(report, sort_keys=True))
