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


def emit_mode2_loop_sync(function,
                         loop,
                         extraction,
                         dependencies,
                         *,
                         cross_ids,
                         local_ids,
                         slot_counts=None):
    """Materialize per-edge slot protocols in a shared regular CV owner loop.

    Every cross-pipe edge owns ready/release channels for each certified slot.
    Carried reuse edges seed ready;
    same-iteration edges seed release. Drain consumes exactly the remaining
    initial credit, including when the loop executes zero times. The extraction
    describes the loop body; conditional/counter owners are not yet supported.
    slot_counts maps (source,target,distance) to the certified owner version
    count. It must come from physical-region/owner proof, not guessed timing.
    """
    from tvm import tir

    if not isinstance(loop, tir.For) or loop.kind != tir.ForKind.SERIAL:
        raise LocalSyncError("mode-2 owner must be a serial loop")
    dependencies = tuple(dependencies)
    slot_counts = {} if slot_counts is None else dict(slot_counts)
    dependency_keys = {(d.source, d.target, d.distance) for d in dependencies}
    if any(key not in dependency_keys for key in slot_counts) or any(
            type(count) is not int or count < 1 for count in slot_counts.values()):
        raise LocalSyncError(
            "slot counts require known dependency keys and positive integer counts")
    if any(type(d.distance) is not int or d.distance < 0 for d in dependencies):
        raise LocalSyncError("mode-2 dependency distances must be nonnegative integers")
    # Validate core/participant/node ownership using the existing flat emitter.
    emit_mode2_sync(
        function,
        extraction, [d for d in dependencies if d.distance == 0],
        cross_ids=cross_ids,
        local_ids=local_ids)
    by_id = {task.id: task for task in extraction.tasks}
    positions = {task.id: i for i, task in enumerate(extraction.tasks)}
    before, after = {i: [] for i in by_id}, {i: [] for i in by_id}
    pairs = {}
    minimum_distance = {}
    for d in dependencies:
        pair = d.source, d.target
        minimum_distance[pair] = min(minimum_distance.get(pair, d.distance), d.distance)
    for d in dependencies:
        if d.kind not in ("RAW", "WAR", "WAW") or d.source not in by_id or d.target not in by_id:
            raise LocalSyncError("mode-2 owner requires analyzed RAW/WAR/WAW dependencies")
        if d.distance == 0 and positions[d.source] >= positions[d.target]:
            raise LocalSyncError("mode-2 same-epoch edges must preserve lexical order")
        # Actual pipe FIFO orders source(i) before source(i+1). A real emitted
        # source(i+1)->target(i+1) fence therefore already proves this carried
        # edge. Adding another credit protocol would impose a false cycle.
        if d.distance > minimum_distance[d.source, d.target]:
            continue
        count = slot_counts.get((d.source, d.target, d.distance), 1)
        if d.distance not in (0, count):
            raise LocalSyncError("carried mode-2 distance must equal its certified slot count")
        a, b = by_id[d.source], by_id[d.target]
        if a.core != b.core or _pipe(a.pipe) != _pipe(b.pipe):
            key = d.source, d.target, d.distance
            if key in pairs and pairs[key] != count:
                raise LocalSyncError("conflicting slot counts for one dependency channel")
            pairs[key] = count
    used_cross, local_pools = set(), {}

    def channel(a, b):
        cross = a.core != b.core
        pool = used_cross if cross else local_pools.setdefault((a.core, _pipe(b.pipe)), set())
        candidates = cross_ids if cross else local_ids
        event = next((i for i in candidates if i not in pool), None)
        if event is None:
            raise LocalSyncError("mode-2 owner event channel capacity exhausted")
        pool.add(event)
        return a.core, _pipe(a.pipe), b.core, _pipe(b.pipe), event

    def flag(kind, key, *, scoped=False):
        source_core, source_pipe, target_core, target_pipe, event = key
        if source_core != target_core:
            args = (source_pipe, event, 2) if kind == "set" else (event, "")
            op = "tl.ascend_" + kind + "_cross_flag"
        else:
            args = source_pipe, target_pipe, event
            op = "tl.ascend_" + kind + "_flag"
        node = tir.Evaluate(tir.call_intrin("handle", tir.op.Op.get(op), *args))
        if scoped:
            core = source_core if kind == "set" else target_core
            code = 0 if core == "C" else 1
            node = tir.AttrStmt(tir.IntImm("int32", code), "resource_scope", code, node)
        return node

    initialize, drain = [], []

    def selected(keys, count):
        slot = tir.floormod(loop.loop_var - loop.min, count)
        event = tir.IntImm("int32", keys[-1][-1])
        for i in reversed(range(count - 1)):
            event = tir.Select(slot == i, tir.IntImm("int32", keys[i][-1]), event)
        return (*keys[0][:-1], event)

    for (source, target, distance), count in pairs.items():
        ready = tuple(channel(by_id[source], by_id[target]) for _ in range(count))
        release = tuple(channel(by_id[target], by_id[source]) for _ in range(count))
        before[source].append(flag("wait", selected(release, count)))
        after[source].append(flag("set", selected(ready, count)))
        before[target].append(flag("wait", selected(ready, count)))
        after[target].append(flag("set", selected(release, count)))
        initial = ready if distance else release
        initialize.extend(flag("set", key, scoped=True) for key in initial)
        drain.extend(flag("wait", key, scoped=True) for key in initial)
    seen = set()

    def task_rewrite(node):
        if not isinstance(node, tir.Evaluate):
            return None
        task_id = next(
            (i for i, original in extraction.task_nodes.items() if node.same_as(original)), None)
        if task_id is None:
            return None
        seen.add(task_id)
        sequence = [*before[task_id], node, *after[task_id]]
        return sequence[0] if len(sequence) == 1 else tir.SeqStmt(sequence)

    body = tir.stmt_functor.ir_transform(loop.body, None, task_rewrite)
    if seen != set(by_id):
        raise LocalSyncError("mode-2 tasks must all belong to the certified owner body")
    rewritten = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, body, loop.thread_binding,
                        loop.annotations, getattr(loop, "span", None))
    sequence = [*initialize, rewritten, *drain]
    replacement = sequence[0] if len(sequence) == 1 else tir.SeqStmt(sequence)
    found = []

    def replace_owner(node):
        if isinstance(node, tir.For) and node.same_as(loop):
            found.append(node)
            return replacement
        return None

    result = tir.stmt_functor.ir_transform(function.body, None, replace_owner)
    if len(found) != 1:
        raise LocalSyncError("mode-2 owner must appear exactly once in the function")
    return function.with_body(result)
