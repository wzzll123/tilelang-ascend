# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""One-shot fences around a regular prefix / owner loop / suffix program.

Boundary hazards use whole-storage envelopes across all loop epochs. A one-shot
notification is never consumed inside the repeated owner. Persisted state is
not rotated, and no event ID is reused without a completion proof.
"""

from dataclasses import replace, asdict
import json
from .dependency import analyze_dependencies
from .events import EventRegistry
from .local_sync import emit_local_sync, emit_loop_sync, LocalSyncError, _pipe
from .cross_sync import emit_mode2_sync, emit_mode2_loop_sync

_DEFAULT_IDS = tuple(range(8))


def _subset(extraction, tasks):
    ids = {task.id for task in tasks}
    return replace(
        extraction,
        tasks=tuple(tasks),
        task_nodes={i: extraction.task_nodes[i] for i in ids},
        loops={i: () for i in ids},
        stages={i: extraction.stages[i] for i in ids})


def _flag(kind, channel, *, scoped=False):
    from tvm import tir
    if channel.cross:
        args = (channel.source_pipe, channel.event_id, 2) if kind == "set" else (channel.event_id,
                                                                                 "")
        name = "tl.ascend_" + kind + "_cross_flag"
    else:
        args = channel.source_pipe, channel.destination_pipe, channel.event_id
        name = "tl.ascend_" + kind + "_flag"
    result = tir.Evaluate(tir.call_intrin("handle", tir.op.Op.get(name), *args))
    if scoped:
        core = channel.source_core if kind == "set" else channel.destination_core
        code = 0 if core == "C" else 1
        result = tir.AttrStmt(tir.IntImm("int32", code), "resource_scope", code, result)
    return result


def _validate_tree(function, extraction, owner, mixed, participant_nodes):
    """Prove each operation's unique lexical, core and allocation ownership."""
    from tvm import tir
    nodes = list(extraction.task_nodes.items())
    if any(
            left.same_as(right)
            for index, (_, left) in enumerate(nodes)
            for _, right in nodes[index + 1:]):
        raise LocalSyncError("phase task statements require unique physical node identities")
    by_id = {task.id: task for task in extraction.tasks}
    parameters = [buffer.data for buffer in function.buffer_map.values()]
    seen, loops = [], []

    def visit(node, core=None, allocated=()):
        if isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, core, allocated)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key == "resource_scope":
                if not isinstance(node.value, tir.IntImm) or int(node.value) not in (0, 1):
                    raise LocalSyncError("phase core scope must be explicit C/V")
                selected = "C" if int(node.value) == 0 else "V"
                if core is not None and selected != core:
                    raise LocalSyncError("contradictory nested core scopes are not executable")
                core = selected
            elif node.attr_key == "thread_extent" and isinstance(
                    node.node,
                    tir.IterVar) and node.node.thread_tag in ("threadIdx.x", "blockIdx.y"):
                if mixed and not any(node.same_as(group) for group in participant_nodes):
                    raise LocalSyncError("participant context must be shared by every phase")
            visit(node.body, core, allocated)
        elif isinstance(node, tir.For):
            if not node.same_as(owner) or (mixed and core is not None):
                raise LocalSyncError("phase owner requires one core-neutral shared cut")
            loops.append(node)
            visit(node.body, core, allocated)
        elif isinstance(node, tir.BlockRealize):
            if not isinstance(node.predicate, tir.IntImm) or not int(node.predicate):
                raise LocalSyncError("predicated phase blocks require guard reconstruction")
            visit(node.block, core, allocated)
        elif isinstance(node, tir.Block):
            if node.init is not None or node.match_buffers or node.iter_vars:
                raise LocalSyncError("phase blocks require lowered initialization/alias bindings")
            visit(node.body, core, (*allocated, *(buffer.data for buffer in node.alloc_buffers)))
        elif isinstance(node, tir.Allocate):
            if not isinstance(node.condition, tir.IntImm) or not int(node.condition):
                raise LocalSyncError("conditional phase allocation needs lifecycle reconstruction")
            visit(node.body, core, (*allocated, node.buffer_var))
        elif isinstance(node, tir.Evaluate):
            task_id = next((i for i, original in nodes if node.same_as(original)), None)
            if task_id is None:
                if isinstance(node.value, tir.IntImm):
                    return
                raise LocalSyncError("phase operation is absent from the task extraction")
            if task_id in seen or core != by_id[task_id].core:
                raise LocalSyncError("phase task must occur once in its matching core scope")
            seen.append(task_id)
            for access in by_id[task_id].accesses:
                var = access.storage.var
                if str(var.dtype) == "handle" and not any(
                        var.same_as(bound) for bound in (*allocated, *parameters)):
                    raise LocalSyncError("phase allocation must dominate every accessing task")
        else:
            raise LocalSyncError("phase structure contains unreconstructed control or declarations")

    visit(function.body)
    if seen != [task.id for task in extraction.tasks] or len(loops) != 1:
        raise LocalSyncError("phase extraction must cover one complete lexical program")


def _flat(function, extraction, registry):
    from tvm import tir
    if not extraction.tasks:
        return function
    dependencies = analyze_dependencies(extraction.tasks, max_distance=0)
    if len({task.core for task in extraction.tasks}) > 1:
        return emit_mode2_sync(
            function,
            extraction,
            dependencies,
            local_ids=registry.local_ids,
            cross_ids=registry.cross_ids,
            registry=registry)
    emitted = emit_local_sync(
        extraction, dependencies, event_ids=registry.local_ids, registry=registry)
    before, after = {
        task.id: [] for task in extraction.tasks
    }, {
        task.id: [] for task in extraction.tasks
    }
    for item in emitted.assignments:
        from .events import EventChannel
        channel = EventChannel(item.core, item.source_pipe, item.core, item.destination_pipe,
                               item.event_id)
        before[item.target].append(_flag("wait", channel))
        after[item.source].append(_flag("set", channel))

    def rewrite(node):
        key = next((i for i, original in extraction.task_nodes.items() if node.same_as(original)),
                   None)
        if key is None:
            return None
        sequence = [*before[key], node, *after[key]]
        return sequence[0] if len(sequence) == 1 else tir.SeqStmt(sequence)

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, rewrite))


def emit_three_phase_sync(function, extraction, *, local_ids=_DEFAULT_IDS, cross_ids=_DEFAULT_IDS):
    from tvm import tir
    if (extraction.scalar_bindings or
            any(task.guard not in (None, ()) for task in extraction.tasks) or any(
                access.ring_size not in (None, 1)
                for task in extraction.tasks
                for access in task.accesses)):
        raise LocalSyncError("three-phase reconstruction requires unguarded single-version state")
    contexts = [context for context in extraction.loops.values() if context]
    if (not contexts or any(len(context) != 1 for context in contexts) or
            any(not context[0].same_as(contexts[0][0]) for context in contexts)):
        raise LocalSyncError("three-phase reconstruction requires one shared serial owner")
    owner = contexts[0][0]
    partition, phase = {0: [], 1: [], 2: []}, 0
    for task in extraction.tasks:
        if extraction.loops[task.id]:
            if phase == 2:
                raise LocalSyncError("owner tasks cannot straddle the suffix")
            phase = 1
        elif phase == 1:
            phase = 2
        partition[phase].append(task)
    phase_of = {task.id: index for index, tasks in partition.items() for task in tasks}
    body = _subset(extraction, partition[1])
    mixed = len({task.core for task in extraction.tasks}) > 1
    participant_nodes = []
    if mixed:
        # Participant context must dominate all three phases, not appear only
        # inside the owner or in an unrelated sibling.
        cursor, participants = function.body, False
        while isinstance(cursor, (tir.AttrStmt, tir.BlockRealize, tir.Block, tir.Allocate)):
            if isinstance(cursor, tir.AttrStmt):
                if cursor.attr_key == "resource_scope":
                    raise LocalSyncError("mixed phases need a shared outer core context")
                if cursor.attr_key == "thread_extent" and isinstance(
                        cursor.node,
                        tir.IterVar) and cursor.node.thread_tag in ("threadIdx.x", "blockIdx.y"):
                    if not isinstance(cursor.value, tir.IntImm) or int(cursor.value) != 2:
                        raise LocalSyncError("mixed phases require two Vector participants")
                    participants = True
                    participant_nodes.append(cursor)
                cursor = cursor.body
            elif isinstance(cursor, tir.BlockRealize):
                cursor = cursor.block
            else:
                cursor = cursor.body
        if not participants:
            raise LocalSyncError("two-Vector context must dominate all phases")
        if len(participant_nodes) != 1:
            raise LocalSyncError("mixed phases require one unambiguous Vector participant group")
    _validate_tree(function, extraction, owner, mixed, participant_nodes)
    from .uniform_clock import certify_uniform_clock
    try:
        certify_uniform_clock(function, owner)
    except ValueError as error:
        raise LocalSyncError(f"three-phase owner requires a uniform clock: {error}") from error
    registry = EventRegistry(local_ids=local_ids, cross_ids=cross_ids)
    # Whole-storage envelopes conservatively cover every possible ring/loop
    # offset. Boundary analysis is not restricted to iteration zero.
    broad = tuple(
        replace(
            task,
            accesses=tuple(
                replace(access, offset=None, extent=None, intervals=None)
                for access in task.accesses))
        for task in extraction.tasks)
    boundaries = [
        dep for dep in analyze_dependencies(broad, max_distance=0)
        if phase_of[dep.source] != phase_of[dep.target]
    ]
    by_id = {task.id: task for task in extraction.tasks}
    channels = {}
    for dep in boundaries:
        source, target = by_id[dep.source], by_id[dep.target]
        if source.core == target.core and _pipe(source.pipe) == _pipe(target.pipe):
            continue
        key = phase_of[dep.source], phase_of[dep.target], source.core, _pipe(
            source.pipe), target.core, _pipe(target.pipe)
        if key not in channels:
            channels[key] = registry.allocate(*key[2:])
    result = _flat(function, _subset(extraction, partition[0]), registry)
    result = _flat(result, _subset(extraction, partition[2]), registry)
    entry_sets, entry_waits, exit_sets, exit_waits = [], [], [], []
    for (source_phase, target_phase, *_), channel in channels.items():
        (entry_sets if source_phase == 0 else exit_sets).append(_flag("set", channel, scoped=True))
        (entry_waits if target_phase == 1 else exit_waits).append(
            _flag("wait", channel, scoped=True))

    def wrap_protocol(protocol):
        sequence = [*entry_sets, *entry_waits, protocol, *exit_sets, *exit_waits]
        return sequence[0] if len(sequence) == 1 else tir.SeqStmt(sequence)

    dependencies = analyze_dependencies(body.tasks, max_distance=1)
    if len({task.core for task in body.tasks}) > 1:
        from .materializer import _validate_cv_owner
        _validate_cv_owner(function, owner, body)
        result = emit_mode2_loop_sync(
            result,
            owner,
            body,
            dependencies,
            local_ids=local_ids,
            cross_ids=cross_ids,
            registry=registry,
            protocol_wrapper=wrap_protocol)
    else:
        synchronized = emit_loop_sync(
            body, owner, dependencies, event_ids=local_ids, registry=registry)
        result = result.with_body(
            tir.stmt_functor.ir_transform(
                result.body, None, lambda node: wrap_protocol(synchronized)
                if node.same_as(owner) else None))
    return result.with_attr(
        "tl.ascend.phase_events",
        json.dumps([asdict(channel) for channel in registry.records], sort_keys=True))
