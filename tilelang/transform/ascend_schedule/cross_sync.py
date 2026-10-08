# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Materialize local events and mode-2 notifications in an unguarded CV slice."""

from types import SimpleNamespace
from .local_sync import LocalSyncError, emit_local_sync, _pipe


def emit_mode2_sync(function, extraction, dependencies, *, cross_ids, local_ids):
    """Keep each operation in its original C/V scope and add real fences.

    A2/A3 CrossCoreWaitFlag executes on scalar control (no destination pipe
    parameter). Producer notification is submitted to its actual source pipe.
    The native thread/vid structure supplies the mode-2 AIC/AIV participants.
    This emitter owns distinct channels for this single, unconditional epoch;
    loop reuse, guards and initialization/drain require the owner protocol.
    """
    from tvm import tir

    vector_extents = []

    def participants(node):
        if (isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and
                isinstance(node.node, tir.IterVar) and
                node.node.thread_tag in ("threadIdx.x", "blockIdx.y")):
            vector_extents.append(int(node.value) if isinstance(node.value, tir.IntImm) else None)

    tir.stmt_functor.post_order_visit(function.body, participants)
    if not vector_extents or any(extent != 2 for extent in vector_extents):
        raise LocalSyncError(
            "A2/A3 mode-2 emission requires an explicit two Vector participant group")
    tasks, dependencies = tuple(extraction.tasks), tuple(dependencies)
    if (extraction.scalar_bindings or any(extraction.loops.values()) or
            any(t.guard not in (None, ()) for t in tasks)):
        raise LocalSyncError("mode-2 slice requires flat unguarded tasks without scalar bindings")
    if type(cross_ids) is not tuple or any(type(i) is not int or i < 0 for i in cross_ids) or len(
            set(cross_ids)) != len(cross_ids):
        raise LocalSyncError("cross_ids must be distinct nonnegative explicit integers")
    positions = {task.id: i for i, task in enumerate(tasks)}
    by_id = {task.id: task for task in tasks}
    if len(positions) != len(tasks) or set(extraction.task_nodes) != set(positions):
        raise LocalSyncError("mode-2 task identities must be unique and complete")
    if any(t.core not in ("C", "V") for t in tasks):
        raise LocalSyncError("mode-2 tasks require explicit C/V domains")
    for d in dependencies:
        if (d.kind not in ("RAW", "WAR", "WAW") or d.distance != 0 or d.source not in positions or
                d.target not in positions or positions[d.source] >= positions[d.target]):
            raise LocalSyncError("mode-2 slice requires forward distance-zero dependencies")
    before, after = {t.id: [] for t in tasks}, {t.id: [] for t in tasks}

    def local_flag(kind, assignment):
        return tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_" + kind + "_flag"),
                            assignment.source_pipe, assignment.destination_pipe,
                            assignment.event_id))

    for core in ("C", "V"):
        subset = tuple(t for t in tasks if t.core == core)
        if not subset:
            continue
        ids = {t.id for t in subset}
        local = SimpleNamespace(
            tasks=subset,
            task_nodes={i: extraction.task_nodes[i] for i in ids},
            loops={i: () for i in ids},
            scalar_bindings=())
        emitted = emit_local_sync(
            local, [d for d in dependencies if d.source in ids and d.target in ids],
            event_ids=local_ids)
        for assignment in emitted.assignments:
            before[assignment.target].append(local_flag("wait", assignment))
            after[assignment.source].append(local_flag("set", assignment))
    pairs = sorted(
        {(d.source, d.target) for d in dependencies if by_id[d.source].core != by_id[d.target].core
        },
        key=lambda pair: (positions[pair[0]], positions[pair[1]]))
    if len(pairs) > len(cross_ids):
        raise LocalSyncError("mode-2 channel capacity exhausted")
    for (source, target), event in zip(pairs, cross_ids):
        after[source].append(
            tir.Evaluate(
                tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_cross_flag"),
                                _pipe(by_id[source].pipe), event, 2)))
        before[target].append(
            tir.Evaluate(
                tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_cross_flag"), event, "")))
    seen = set()

    def rewrite(node):
        if not isinstance(node, tir.Evaluate):
            return None
        task_id = next(
            (i for i, original in extraction.task_nodes.items() if node.same_as(original)), None)
        if task_id is None:
            return None
        seen.add(task_id)
        statements = [*before[task_id], node, *after[task_id]]
        return statements[0] if len(statements) == 1 else tir.SeqStmt(statements)

    body = tir.stmt_functor.ir_transform(function.body, None, rewrite)
    if seen != set(positions):
        raise LocalSyncError("mode-2 extraction does not belong to the supplied function")
    return function.with_body(body)
