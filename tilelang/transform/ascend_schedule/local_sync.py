# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Emit actual local SetFlag/WaitFlag TIR for a flat, single-core task slice.

This prototype does not reconstruct loops, guards, scalar bindings, multi-pipe
operations, cross-core execution, or a software pipeline. Event IDs are explicit
caller inputs and are never reused within a core/destination channel pool. This
does not assume that hardware event keys include source, or that a textually
earlier wait has completed before another asynchronous source submits a set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .async_alias import ExecutionEdge


class LocalSyncError(ValueError):
    """The input is outside the proven local-emission slice."""


class EventCapacityError(LocalSyncError):
    """Caller-supplied event IDs cannot cover all live channel intervals."""


@dataclass(frozen=True)
class ChannelAssignment:
    source: object
    target: object
    core: str
    source_pipe: str
    destination_pipe: str
    event_id: int
    source_position: int
    wait_position: int
    dependency_kinds: tuple[str, ...]
    reused_after: tuple[object, object] | None = None


@dataclass(frozen=True)
class LocalSyncResult:
    statement: object
    assignments: tuple[ChannelAssignment, ...]
    execution_edges: tuple[ExecutionEdge, ...]
    diagnostics: tuple[str, ...]


_PIPES = {"MTE1", "MTE2", "MTE3", "M", "V", "FIX"}
_CORE_PIPES = {"C": {"MTE2", "MTE1", "M", "FIX"}, "V": {"MTE2", "V", "MTE3"}}


def _pipe(value):
    if type(value) is not str:
        raise LocalSyncError("every task must have one explicit pipe")
    value = value.upper()
    value = value[5:] if value.startswith("PIPE_") else value
    if value not in _PIPES:
        raise LocalSyncError(f"unsupported or multi-pipe task: {value}")
    return value


def emit_local_sync(extraction, dependencies, *, event_ids, order=None):
    """Return a SeqStmt with producer sets and consumer waits plus channel proofs.

    ``order`` contains original task IDs, not solver indexes. Every edge must be
    distance zero and forward in that order. Same-pipe lexical FIFO is preserved
    and represented explicitly in ``execution_edges``. Different-pipe hazards
    become real builtin calls with exactly ``(src, dst, eventId)`` arguments.
    IDs stay allocated for the entire slice: submission positions do not prove
    that asynchronous destination waits have consumed their event credits.
    No inferred data edge is presented as an execution proof without emission.
    """
    tasks = tuple(extraction.tasks)
    dependencies = tuple(dependencies)
    if not tasks:
        raise LocalSyncError("local synchronization requires at least one task")
    if type(event_ids) is not tuple or any(type(i) is not int or i < 0 for i in event_ids):
        raise LocalSyncError("event_ids must be an explicit tuple of nonnegative integers")
    if len(set(event_ids)) != len(event_ids):
        raise LocalSyncError("event_ids must be unique")
    if extraction.scalar_bindings:
        raise LocalSyncError("scalar bindings require reconstruction outside this slice")
    if any(metadata for metadata in extraction.loops.values()):
        raise LocalSyncError("loop metadata is outside the flat single-instance slice")
    task_ids = tuple(t.id for t in tasks)
    if len(set(task_ids)) != len(tasks):
        raise LocalSyncError("task IDs must be unique")
    if set(extraction.task_nodes) != set(task_ids):
        raise LocalSyncError("task_nodes must contain exactly every task")
    if any(
            t.guard is not None and not (type(t.guard) is tuple and len(t.guard) == 0)
            for t in tasks):
        raise LocalSyncError("guarded tasks require reconstruction outside this slice")
    cores = {t.core for t in tasks}
    if len(cores) != 1 or next(iter(cores)) not in ("C", "V"):
        raise LocalSyncError("local emission requires one explicit C or V core")
    core = next(iter(cores))
    pipes = {t.id: _pipe(t.pipe) for t in tasks}
    if any(pipe not in _CORE_PIPES[core] for pipe in pipes.values()):
        raise LocalSyncError("task pipe is outside its explicit core execution domain")
    order = task_ids if order is None else tuple(order)
    if len(order) != len(task_ids) or set(order) != set(task_ids):
        raise LocalSyncError("chosen order must be a permutation of every task ID")
    positions = {task: position for position, task in enumerate(order)}
    execution_edges, last_on_pipe = [], {}
    for task in tasks:
        previous = last_on_pipe.get(pipes[task.id])
        if previous is not None:
            if positions[previous] >= positions[task.id]:
                raise LocalSyncError("chosen order must preserve lexical same-pipe FIFO")
            execution_edges.append(ExecutionEdge(previous, task.id, "fifo"))
        last_on_pipe[pipes[task.id]] = task.id
    hazards = {}
    for dependency in dependencies:
        if dependency.kind not in ("RAW", "WAR", "WAW"):
            raise LocalSyncError("only analyzed RAW/WAR/WAW dependencies are supported")
        if type(dependency.distance) is not int or dependency.distance != 0:
            raise LocalSyncError("carried dependencies require loop reconstruction")
        if dependency.source not in positions or dependency.target not in positions:
            raise LocalSyncError("dependency references missing task")
        if positions[dependency.source] >= positions[dependency.target]:
            raise LocalSyncError("chosen order violates dependency geometry")
        if pipes[dependency.source] == pipes[dependency.target]:
            continue
        pair = dependency.source, dependency.target
        kinds = hazards.setdefault(pair, [])
        if dependency.kind not in kinds:
            kinds.append(dependency.kind)

    assignments, pools = [], {}
    pairs = sorted(hazards, key=lambda pair: (positions[pair[0]], positions[pair[1]]))
    for source, target in pairs:
        key = core, pipes[target]
        pool = pools.setdefault(key, {})
        selected = next((event for event in event_ids if event not in pool), None)
        if selected is None:
            raise EventCapacityError(f"event capacity exhausted for core/destination {key}")
        assignment = ChannelAssignment(source, target, core, pipes[source], pipes[target], selected,
                                       positions[source], positions[target],
                                       tuple(hazards[(source, target)]))
        assignments.append(assignment)
        pool[selected] = assignment
        execution_edges.append(ExecutionEdge(source, target, "sync"))

    # Lazy import leaves validation/contract users independent of the native runtime.
    from tvm import tir

    for task in tasks:
        node = extraction.task_nodes[task.id]
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            raise LocalSyncError("only original flat Evaluate(Call) task nodes are supported")
        if node.value.op.name in (
                "tl.ascend_set_flag",
                "tl.ascend_wait_flag",
                "tl.ascend_set_cross_flag",
                "tl.ascend_wait_cross_flag",
                "tl.ascend_pipe_barrier",
                "tl.ascend_sync_all",
        ):
            raise LocalSyncError("preexisting synchronization needs separate event ownership")
    statements = []
    for task_id in order:
        for assignment in assignments:
            if assignment.target == task_id:
                statements.append(
                    tir.Evaluate(
                        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_flag"),
                                        assignment.source_pipe, assignment.destination_pipe,
                                        assignment.event_id)))
        statements.append(extraction.task_nodes[task_id])
        for assignment in assignments:
            if assignment.source == task_id:
                statements.append(
                    tir.Evaluate(
                        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_flag"),
                                        assignment.source_pipe, assignment.destination_pipe,
                                        assignment.event_id)))
    return LocalSyncResult(
        tir.SeqStmt(statements), tuple(assignments), tuple(execution_edges),
        ("flat single-core local synchronization only",
         "event IDs supplied by caller; no ID reuse within destination pools"))


def emit_loop_sync(extraction, loop, dependencies, *, event_ids, slot_count=1):
    """Emit bounded-credit local protocols around one unconditional serial loop.

    The extraction describes the flat loop *body*, not the surrounding For.
    Distance-zero edges transfer ready then release in the same iteration;
    carried edges transfer ready to the next reuse of the same physical slot.
    Each edge owns
    two destination channels so an asynchronous producer cannot overwrite a
    ready notification before its consumer takes it. Initial credits and final
    drain are paired even for an empty loop. This is not slot rotation or the
    general guarded/multicore software-pipeline materializer.
    """
    from tvm import ir, tir

    if not isinstance(loop, tir.For) or loop.kind != tir.ForKind.SERIAL:
        raise LocalSyncError("loop synchronization requires a serial For")
    if type(slot_count) is not int or slot_count < 1:
        raise LocalSyncError("slot_count must be a positive integer")
    dependencies = tuple(dependencies)
    if any(type(d.distance) is not int or d.distance not in (0, slot_count) for d in dependencies):
        raise LocalSyncError("dependency distance must be zero or the certified slot count")
    # Reuse strict flat-domain/guard/node validation, without pretending carried
    # edges are distance-zero edges in its execution proof.
    emit_local_sync(extraction, [d for d in dependencies if d.distance == 0], event_ids=event_ids)
    tasks = tuple(extraction.tasks)
    nodes = [extraction.task_nodes[t.id] for t in tasks]
    expected = nodes[0] if len(nodes) == 1 else tir.SeqStmt(nodes)
    if not ir.structural_equal(expected, loop.body):
        raise LocalSyncError("extraction must cover the complete flat loop body in lexical order")
    positions = {t.id: i for i, t in enumerate(tasks)}
    pipes = {t.id: _pipe(t.pipe) for t in tasks}
    pairs = {}
    for d in dependencies:
        if d.kind not in ("RAW", "WAR", "WAW"):
            raise LocalSyncError("only RAW/WAR/WAW loop dependencies are supported")
        if d.source not in positions or d.target not in positions:
            raise LocalSyncError("dependency references missing loop task")
        if d.distance == 0 and positions[d.source] >= positions[d.target]:
            raise LocalSyncError("same-iteration dependency contradicts lexical order")
        if pipes[d.source] != pipes[d.target]:
            pairs[d.source, d.target, d.distance] = None

    pools = {}

    def channel(source_pipe, destination_pipe):
        used = pools.setdefault(destination_pipe, set())
        event = next((i for i in event_ids if i not in used), None)
        if event is None:
            raise EventCapacityError(
                f"loop event capacity exhausted for destination {destination_pipe}")
        used.add(event)
        return source_pipe, destination_pipe, event

    def flag(kind, key):
        return tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_" + kind + "_flag"), *key))

    slot = tir.floormod(loop.loop_var - loop.min, slot_count)

    def selected(keys):
        event = tir.IntImm("int32", keys[-1][2])
        for index in reversed(range(len(keys) - 1)):
            event = tir.Select(slot == index, tir.IntImm("int32", keys[index][2]), event)
        return keys[0][0], keys[0][1], event

    before = {t.id: [] for t in tasks}
    after = {t.id: [] for t in tasks}
    initialize, drain = [], []
    for source, target, distance in pairs:
        ready = tuple(channel(pipes[source], pipes[target]) for _ in range(slot_count))
        release = tuple(channel(pipes[target], pipes[source]) for _ in range(slot_count))
        before[source].append(flag("wait", selected(release)))
        after[source].append(flag("set", selected(ready)))
        before[target].append(flag("wait", selected(ready)))
        after[target].append(flag("set", selected(release)))
        initial = ready if distance else release
        initialize.extend(flag("set", key) for key in initial)
        drain.extend(flag("wait", key) for key in initial)
    statements = []
    for task in tasks:
        statements.extend(before[task.id])
        statements.append(extraction.task_nodes[task.id])
        statements.extend(after[task.id])
    rewritten = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, tir.SeqStmt(statements),
                        loop.thread_binding, loop.annotations, getattr(loop, "span", None))
    return tir.SeqStmt([*initialize, rewritten, *drain])
