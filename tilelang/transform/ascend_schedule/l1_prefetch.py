# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Discover certified L1 input-prefetch owners before joint version solving.

No buffer names identify candidates. Generated workspaces and mutable sources
are not input prefetches. This module does not emit flags or executable stages;
those must consume the same selected owner/version plan during reconstruction.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PrefetchCandidate:
    buffer: object
    storage: object
    owner: object
    writer_id: object
    size_bytes: int
    plan: object


@dataclass(frozen=True)
class PrefetchDiscovery:
    candidates: tuple
    tasks: tuple
    stages: dict
    diagnostics: tuple


@dataclass(frozen=True)
class PrefetchMaterialization:
    function: object
    extraction: object
    owner: object
    stages: dict
    dependencies: tuple
    slot_counts: dict


def discover_l1_prefetch(function, extraction):
    from tvm import tir
    from .l1_epoch import certify_l1_epoch_write, certify_l1_epoch_read
    from .uniform_clock import certify_uniform_clock
    from .ownership import OwnerDescriptor, plan_storage_epochs

    buffers = []
    physical_allocations = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: buffers.extend(node.alloc_buffers)
        if isinstance(node, tir.Block) else physical_allocations.append(node)
        if isinstance(node, tir.Allocate) else None)
    if physical_allocations:
        raise ValueError("joint prefetch capacity needs complete Block allocation accounting")
    buffers = [buffer for buffer in buffers if buffer.scope() == "shared.l1"]
    diagnostics, candidates = [], []
    if extraction.scalar_bindings:
        return PrefetchDiscovery((), (), {}, ("prefetch owner contains scalar bindings",))
    generated = {int(index) for index in function.attrs.get("auto_gm_indices", ())}
    readonly_inputs = [
        view.data
        for handle, view in function.buffer_map.items()
        if view.scope() == "global" and any(
            handle.same_as(param) and index not in generated
            for index, param in enumerate(function.params)) and not any(
                access.writes and access.storage.var.same_as(view.data)
                for task in extraction.tasks
                for access in task.accesses)
    ]
    proven = list(extraction.tasks)
    common_owner = None
    for buffer in buffers:
        try:
            storages = [
                access.storage
                for task in extraction.tasks
                for access in task.accesses
                if access.storage.var.same_as(buffer.data)
            ]
            if not storages:
                continue
            storage = storages[0]
            if any(token is not storage for token in storages):
                raise ValueError("prefetch storage has conflicting physical domains")
            touching = [
                task for task in extraction.tasks
                if any(access.storage is storage for access in task.accesses)
            ]
            contexts = [extraction.loops[task.id] for task in touching]
            if any(len(context) != 1 for context in contexts) or any(
                    not context[0].same_as(contexts[0][0]) for context in contexts):
                raise ValueError("prefetch accesses do not share one exclusive owner")
            owner = contexts[0][0]
            if common_owner is not None and not owner.same_as(common_owner):
                raise ValueError("joint prefetch requires one common owner")
            from tvm import ir
            analyzer = certify_uniform_clock(function, owner).analyzer
            analyzer.bind(owner.loop_var, ir.Range.from_min_extent(owner.min, owner.extent))
            writes, footprints = [], {}
            for task in touching:
                if task.guard not in (None, ()):
                    raise ValueError("prefetch needs unconditional owner accesses")
                access_list = [access for access in task.accesses if access.storage is storage]
                if len(access_list) != 1:
                    raise ValueError("prefetch helper has multiple aliased tile accesses")
                access = access_list[0]
                call = extraction.task_nodes[task.id].value
                if access.mode == "W":
                    proof = certify_l1_epoch_write(buffer, call, analyzer)
                    source = call.args[1]
                    if (not isinstance(source, tir.Call) or
                            source.op.name != "tir.tvm_access_ptr" or len(source.args) != 5 or
                            not any(source.args[1].same_as(var) for var in readonly_inputs)):
                        raise ValueError("prefetch source is not a readonly kernel input")
                    writes.append(task.id)
                    footprints[task.id] = proof.byte_footprint
                elif access.mode == "R":
                    footprints[task.id] = certify_l1_epoch_read(buffer, call, analyzer)
                else:
                    raise ValueError("prefetch cannot rotate read-modify-write storage")
            if touching[0].id not in writes:
                raise ValueError("prefetch first access is not a complete write")
            owner_tasks = tuple(
                task for task in proven if any(
                    loop.same_as(owner) for loop in extraction.loops[task.id]))
            if any(
                    len(extraction.loops[task.id]) != 1 or task.guard not in (None, ())
                    for task in owner_tasks):
                raise ValueError("prefetch owner contains nested or conditional tasks")
            first = touching[0].id
            owner_tasks = tuple(
                replace(
                    task,
                    accesses=tuple(
                        replace(
                            access,
                            offset=None,
                            extent=None,
                            intervals=footprints[task.id],
                            full_overwrite=task.id in writes,
                            owner=owner) if access.storage is storage else access
                        for access in task.accesses))
                for task in owner_tasks)
            stages = {task.id: int(task.id != first) for task in owner_tasks}
            size = footprints[first][0][1]
            descriptor = OwnerDescriptor(
                owner, owner_tasks, stages, full_footprints=((storage, ((0, size),)),))
            plan = plan_storage_epochs(storage, (descriptor,))
            if not plan.eligible or plan.owners[0].mode != "iteration":
                raise ValueError("prefetch lacks a proven iteration lifecycle")
            # The existing physical pass also catches escaped/raw pointers,
            # logical aliases, outside-owner accesses and malformed endpoints.
            # Run it as a proof preflight; do not retain its temporary rewrite.
            from .buffer_versions import materialize_iteration_versions
            materialize_iteration_versions(function, buffer, plan)
            replacements = {task.id: task for task in owner_tasks}
            proven = [replacements.get(task.id, task) for task in proven]
            candidates.append(PrefetchCandidate(buffer, storage, owner, first, size, plan))
            common_owner = owner
        except ValueError as error:
            diagnostics.append(f"{buffer.name}: {error}")
    owner_tasks = tuple(
        task for task in proven if common_owner is not None and any(
            loop.same_as(common_owner) for loop in extraction.loops[task.id]))
    writers = {candidate.writer_id for candidate in candidates}
    stages = {task.id: int(task.id not in writers) for task in owner_tasks}
    return PrefetchDiscovery(tuple(candidates), owner_tasks, stages, tuple(diagnostics))


def solve_l1_prefetch(function, discovery, platform):
    """Joint feasibility solve; persistent allocations remain single-version.

    Costs are unit feasibility costs, not a calibrated timing prediction.
    Storage reuse edges alone acquire the certified slot distance. Every other true
    RAW/WAR/WAW edge retains its original distance. The proposed two-stage
    prefetch partition is checked, not assumed executable on solver failure.
    Counts currently use the lifecycle span+1 requirement: unconstrained modulo
    starts can select one version using a different within-wave issue order,
    which the current lexical stage-wave reconstruction cannot safely execute.
    This is joint capacity/feasibility validation, not optimal version search.
    """
    from tvm import DataType, tir
    from .analysis_report import build_schedule_report
    from .dependency import analyze_dependencies
    from .resources import allocation_budgets
    from .solver import StorageCost, TaskCost, ReuseDependency

    if not discovery.candidates or not discovery.tasks:
        raise ValueError("joint prefetch has no certified storage candidates")
    dependencies = analyze_dependencies(discovery.tasks, max_distance=1)
    indexes = {task.id: index for index, task in enumerate(discovery.tasks)}
    names = {
        id(candidate.storage): f"prefetch_{index}"
        for index, candidate in enumerate(discovery.candidates)
    }
    reuse, fixed = [], []
    for dependency in dependencies:
        if dependency.distance and id(dependency.storage) in names:
            reuse.append(
                ReuseDependency(indexes[dependency.source], indexes[dependency.target],
                                names[id(dependency.storage)]))
        else:
            fixed.append(dependency)
    buffers, physical_allocations = [], []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: buffers.extend(node.alloc_buffers)
        if isinstance(node, tir.Block) else physical_allocations.append(node)
        if isinstance(node, tir.Allocate) else None)
    if physical_allocations:
        raise ValueError("joint prefetch capacity needs complete Block allocation accounting")
    scopes = {
        "shared.l1": "L1",
        "shared.ub": "UB",
        "wmma.matrix_a": "L0A",
        "wmma.matrix_b": "L0B",
        "wmma.accumulator": "L0C"
    }
    allocated, seen = {}, []
    for index, buffer in enumerate(buffers):
        if any(buffer.data.same_as(var) for var in seen):
            raise ValueError("joint prefetch capacity needs unique physical allocations")
        seen.append(buffer.data)
        if buffer.scope() not in scopes:
            raise ValueError("joint prefetch capacity has unsupported local allocation scope")
        if any(not isinstance(dim, tir.IntImm) or int(dim) <= 0 for dim in buffer.shape):
            raise ValueError("joint prefetch capacity needs static allocation sizes")
        elements = 1
        for dim in buffer.shape:
            size = int(dim)
            elements *= (size + 15) // 16 * 16 if buffer.scope() != "shared.ub" and len(
                buffer.shape) == 2 else size
        dtype = DataType(buffer.dtype)
        bits = int(dtype.bits) * int(dtype.lanes)
        if bits % 8:
            raise ValueError("joint prefetch capacity needs whole-byte element widths")
        size_bytes = (elements * bits // 8 + 31) // 32 * 32
        candidate = next((candidate for candidate in discovery.candidates
                          if candidate.buffer.data.same_as(buffer.data)), None)
        if candidate is None:
            allocated[f"fixed_{index}"] = StorageCost(size_bytes, scopes[buffer.scope()], 1, 1)
        else:
            allocated[names[id(candidate.storage)]] = StorageCost(size_bytes,
                                                                  scopes[buffer.scope()],
                                                                  candidate.plan.count,
                                                                  candidate.plan.count)
    if any(name not in allocated for name in names.values()):
        raise ValueError("joint prefetch candidate lacks its physical allocation")
    costs = {task.id: TaskCost(1, 1, task.core + ":" + task.pipe) for task in discovery.tasks}
    return build_schedule_report(
        discovery.tasks,
        fixed,
        costs,
        stages=discovery.stages,
        storage=allocated,
        capacities=allocation_budgets(platform),
        reuse_dependencies=reuse,
        resource_model=platform)


def materialize_l1_prefetch(function, discovery, counts):
    """Apply jointly validated counts and refresh all physical identities.

    The caller must obtain counts from joint capacity validation. This step
    checks their lifecycle requirement again, but does not create events or
    issue reordered tasks. It supplies the exact ring dependencies and event
    slot counts needed by the subsequent synchronization/wave reconstruction.
    """
    from tvm import tir
    from .buffer_versions import materialize_iteration_versions
    from .tir_adapter import extract_tasks
    from .dependency import analyze_dependencies

    expected = {f"prefetch_{index}" for index in range(len(discovery.candidates))}
    if not expected or not expected <= counts.keys():
        raise ValueError("joint materialization needs every certified prefetch selection")
    original_ids = tuple(task.id for task in extract_tasks(function).tasks)
    chosen = []
    for index, candidate in enumerate(discovery.candidates):
        count = counts[f"prefetch_{index}"]
        if type(count) is not int or count != candidate.plan.count or count < 2:
            raise ValueError("selected slot count violates the certified stage lifecycle")
        # The previous allocation rewrite rebuilt the owner node. Never reuse
        # its old object identity for the next candidate in the same owner.
        fresh = extract_tasks(function)
        contexts = [
            fresh.loops[task.id] for task in fresh.tasks if any(
                access.storage.var.same_as(candidate.buffer.data) for access in task.accesses)
        ]
        if not contexts or any(len(context) != 1 for context in contexts) or any(
                not context[0].same_as(contexts[0][0]) for context in contexts):
            raise ValueError("joint materialization lost the exclusive candidate owner")
        owner = contexts[0][0]
        plan = replace(candidate.plan, owners=(replace(candidate.plan.owners[0], loop=owner),))
        function = materialize_iteration_versions(function, candidate.buffer, plan)
        chosen.append((candidate, count))
    fresh = extract_tasks(function)
    if tuple(task.id for task in fresh.tasks) != original_ids:
        raise ValueError("joint materialization changed task identity/order unexpectedly")
    owners = [fresh.loops[task.id][0] for task in fresh.tasks if task.id in discovery.stages]
    if not owners or any(not owner.same_as(owners[0]) for owner in owners):
        raise ValueError("joint materialization lost the common owner")
    owner = owners[0]
    selected = dict(function.attrs.get("tl.ascend.selected_buffer_versions", {}).items())
    for candidate, count in chosen:
        selected[candidate.buffer.data] = tir.IntImm("int32", count)
    function = function.with_attr("tl.ascend.selected_buffer_versions", selected)

    def ring_access(access):
        match = next(((candidate, count)
                      for candidate, count in chosen
                      if access.storage.var.same_as(candidate.buffer.data)), None)
        if match is None:
            return access
        candidate, count = match
        return replace(
            access,
            offset=0,
            extent=candidate.size_bytes,
            intervals=None,
            iteration_stride=candidate.size_bytes,
            ring_size=count,
            owner=owner,
            full_overwrite=access.mode == "W")

    tasks = tuple(
        replace(task, accesses=tuple(ring_access(access)
                                     for access in task.accesses))
        for task in fresh.tasks)
    fresh = replace(fresh, tasks=tasks)
    body = tuple(task for task in tasks if task.id in discovery.stages)
    dependencies = tuple(analyze_dependencies(body, max_distance=max(count for _, count in chosen)))
    slot_counts = {}
    for edge in dependencies:
        match = next(
            ((candidate, count)
             for candidate, count in chosen
             if edge.storage is not None and edge.storage.var.same_as(candidate.buffer.data)), None)
        if match is None:
            continue
        count = match[1]
        key = edge.source, edge.target, edge.distance
        if key in slot_counts and slot_counts[key] != count:
            raise ValueError("one event pair has conflicting version clocks")
        slot_counts[key] = count
    return PrefetchMaterialization(function, fresh, owner, dict(discovery.stages), dependencies,
                                   slot_counts)
