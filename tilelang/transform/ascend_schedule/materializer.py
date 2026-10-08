# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Executable flat-epoch bootstrap for the structured scheduling materializer."""

import json
from dataclasses import replace
from .dependency import analyze_dependencies
from .local_sync import emit_local_sync, emit_loop_sync, LocalSyncError
from .cross_sync import emit_mode2_sync
from .analysis_report import build_schedule_report
from .solver import TaskCost, StorageCost
from .resources import allocation_budgets

MARKER = "tl.ascend.schedule_materialized"


class MaterializationError(ValueError):
    """The requested structure has no complete executable materialization yet."""


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
    if any(extraction.loops.values()):
        contexts = [extraction.loops[task.id] for task in tasks]
        if (len({task.core for task in tasks}) != 1 or
                any(len(context) != 1 for context in contexts) or
                any(not context[0].same_as(contexts[0][0]) for context in contexts)):
            raise MaterializationError("only a single unconditional local owner loop is integrated")
        owner_loop = contexts[0][0]
        if owner_loop.kind != tir.ForKind.SERIAL:
            raise MaterializationError("owner loop must be serial")
        nodes = [extraction.task_nodes[task.id] for task in tasks]
        expected = nodes[0] if len(nodes) == 1 else tir.SeqStmt(nodes)
        from tvm import ir
        if not ir.structural_equal(owner_loop.body, expected):
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
            raise MaterializationError("buffer version hints require owner materialization")
    if function.attrs is not None and function.attrs.get("address_map", {}):
        raise MaterializationError(
            "physical addresses need a validated asynchronous alias contract")
    allocations = []
    scalar_variables = [var for var in function.params if str(var.dtype) != "handle"]
    if owner_loop is not None:
        scalar_variables.append(owner_loop.loop_var)
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
    dependencies = list(
        analyze_dependencies(tasks, max_distance=1 if owner_loop is not None else 0))
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
        materialization="single local owner loop"
        if owner_loop is not None else "flat local/CV epoch",
        timing="unit-cost feasibility; not calibrated NPU timing",
        alias_policy="disjoint physical allocations until async alias contracts are integrated")
    if owner_loop is not None:
        body_extraction = replace(extraction, loops={task.id: () for task in tasks})
        try:
            synchronized = emit_loop_sync(
                body_extraction, owner_loop, dependencies, event_ids=tuple(range(8)))
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
