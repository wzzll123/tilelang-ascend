# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Certified Cube-prefix / complete remaining-epoch overlap.

No buffer names or FA task IDs identify candidates. The earliest complete
Cube→Vector workspace must follow a pure Cube prefix with an MMA producer.
Only that GM tile is rotated; complete dependence checks keep all remaining
state in its original epoch. Unit costs establish feasibility, not performance.
"""


def prepare_score_pipeline(function, extraction, platform):
    from tvm import DataType, tir
    from .workspace_planning import discover_workspace_epochs, plan_workspace_epochs
    from .buffer_versions import materialize_iteration_versions
    from .workspace_proof import certify_workspace_versions
    from .tir_adapter import extract_tasks
    from .dependency import analyze_dependencies
    from .l1_prefetch import PrefetchMaterialization
    from .solver import TaskCost, StorageCost, ScheduleInfeasible
    from .analysis_report import build_schedule_report
    from .resources import allocation_budgets

    if function.attrs.get("tl.ascend.iteration_versions", {}):
        return None, None
    discovery = discover_workspace_epochs(function, extraction)
    by_id = {task.id: task for task in extraction.tasks}
    candidates = [
        candidate for candidate in discovery.candidates
        if by_id[candidate.writer_id].core == "C" and all(
            by_id[reader].core == "V" for reader in candidate.reader_ids)
    ]
    if not candidates:
        return None, None
    positions = {task.id: index for index, task in enumerate(extraction.tasks)}
    candidate = min(candidates, key=lambda item: positions[item.writer_id])
    body = tuple(task for task in extraction.tasks if extraction.loops[task.id])
    if not body or any(
            len(extraction.loops[task.id]) != 1 or
            not extraction.loops[task.id][0].same_as(candidate.owner) or task.guard not in (None,
                                                                                            ())
            for task in body):
        return None, None
    prefix = [task for task in body if positions[task.id] <= positions[candidate.writer_id]]
    if any(task.core != "C" for task in prefix) or not any(task.pipe == "PIPE_M"
                                                           for task in prefix):
        return None, None
    stages = {task.id: int(positions[task.id] > positions[candidate.writer_id]) for task in body}
    if set(stages.values()) != {0, 1}:
        return None, None
    plan = plan_workspace_epochs(candidate, stages)
    function = materialize_iteration_versions(function, candidate.buffer, plan)
    fresh = extract_tasks(function)
    if tuple(task.id for task in fresh.tasks) != tuple(task.id for task in extraction.tasks):
        raise ValueError("score pipeline physical rewrite changed task identities")
    fresh, diagnostics = certify_workspace_versions(function, fresh)
    if diagnostics:
        raise ValueError("score pipeline physical slot proof: " + "; ".join(diagnostics))
    owners = [fresh.loops[task_id] for task_id in stages]
    if any(len(context) != 1 for context in owners) or any(
            not context[0].same_as(owners[0][0]) for context in owners):
        raise ValueError("score pipeline lost its common owner")
    owner = owners[0][0]
    body = tuple(task for task in fresh.tasks if task.id in stages)
    dependencies = analyze_dependencies(body, max_distance=plan.count)
    wave_order = sorted(body, key=lambda task: (stages[task.id], positions[task.id]))
    wave_positions = {task.id: index for index, task in enumerate(wave_order)}
    for edge in dependencies:
        source, target = stages[edge.source], stages[edge.target] + edge.distance
        if source > target or (source == target and
                               wave_positions[edge.source] >= wave_positions[edge.target]):
            raise ScheduleInfeasible(
                "score pipeline violates a true carried or same-wave dependency")
    slot_counts = {
        (edge.source, edge.target, edge.distance): plan.count
        for edge in dependencies
        if edge.storage is not None and edge.storage.var.same_as(candidate.buffer.data)
    }

    # Charge complete physical local allocations, not estimated live ranges.
    buffers, lowered_allocations = [], []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: buffers.extend(node.alloc_buffers)
        if isinstance(node, tir.Block) else lowered_allocations.append(node)
        if isinstance(node, tir.Allocate) else None)
    if lowered_allocations:
        raise ValueError("score pipeline capacity requires Block allocations before flattening")
    scopes = {
        "shared.ub": "UB",
        "shared.l1": "L1",
        "wmma.matrix_a": "L0A",
        "wmma.matrix_b": "L0B",
        "wmma.accumulator": "L0C"
    }
    storage, seen = {}, []

    def byte_size(buffer, local):
        if buffer.strides or any(
                not isinstance(dim, tir.IntImm) or int(dim) <= 0 for dim in buffer.shape):
            raise ValueError("score pipeline allocation needs a compact static geometry")
        elements = 1
        for dim in buffer.shape:
            value = int(dim)
            elements *= (value + 15) // 16 * 16 if local and buffer.scope() != "shared.ub" and len(
                buffer.shape) == 2 else value
        dtype = DataType(buffer.dtype)
        bits = dtype.bits * dtype.lanes
        if bits % 8:
            raise ValueError("score pipeline allocation requires whole-byte elements")
        size = elements * bits // 8
        return (size + 31) // 32 * 32 if local else size

    for index, buffer in enumerate(buffers):
        if any(buffer.data.same_as(var) for var in seen) or buffer.scope() not in scopes:
            raise ValueError("score pipeline requires unique known local allocations")
        seen.append(buffer.data)
        storage[f"local_{index}"] = StorageCost(
            byte_size(buffer, True), scopes[buffer.scope()], 1, 1)
    generated = {int(index) for index in function.attrs.get("auto_gm_indices", ())}
    for index in sorted(generated):
        buffer = function.buffer_map[function.params[index]]
        selected = buffer.data.same_as(candidate.buffer.data)
        storage[f"workspace_{index}"] = StorageCost(
            candidate.size_bytes if selected else byte_size(buffer, False), "GM",
            plan.count if selected else 1, plan.count if selected else 1)
    from .model import Dependency
    feasibility = list(dependencies)
    last = {}
    for task in body:
        key = stages[task.id], task.core, task.pipe
        if key in last:
            feasibility.append(Dependency(last[key], task.id, "RAW"))
        last[key] = task.id
    report = build_schedule_report(
        body,
        feasibility, {task.id: TaskCost(1, 1, task.core + ":" + task.pipe) for task in body},
        stages=stages,
        storage=storage,
        capacities=allocation_budgets(platform),
        resource_model=platform)
    selected = dict(function.attrs.get("tl.ascend.selected_buffer_versions", {}).items())
    selected[candidate.buffer.data] = tir.IntImm("int32", plan.count)
    function = function.with_attr("tl.ascend.selected_buffer_versions", selected)
    # with_attr retains body/owner nodes; refresh the exact extraction required
    # by phase emission without losing certified physical access envelopes.
    report["pipeline_kind"] = "cube_prefix_workspace_overlap"
    report["workspace_version_pitch_bytes"] = candidate.size_bytes
    return PrefetchMaterialization(function, fresh, owner, stages, tuple(dependencies),
                                   slot_counts), report
