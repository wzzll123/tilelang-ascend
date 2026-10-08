# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Connect regular UB overwrite owners to version selection and physical rings.

Eligibility comes from actual native fill/store operands, never pointer extent
or estimated timing. This initial driver supports one compact UB scratch ring
in a local owner; other storage remains single-version, including accumulated
state. Its unit-cost solver is a feasibility model, not calibrated performance.
"""

from dataclasses import replace
import json


def prepare_ub_versions(function, extraction, platform):
    from tvm import DataType, tir
    from .ownership import OwnerDescriptor, plan_storage_epochs
    from .dependency import analyze_dependencies
    from .analysis_report import build_schedule_report
    from .solver import TaskCost, StorageCost, ReuseDependency
    from .resources import allocation_budgets
    from .buffer_versions import materialize_iteration_versions
    from .tir_adapter import extract_tasks

    selected = dict(function.attrs.get("tl.ascend.selected_buffer_versions",
                                       {}).items()) if function.attrs is not None else {}
    alias_specs = {}
    for annotation in extraction.annotations:
        if not hasattr(annotation, "get"):
            continue
        modes = annotation.get("tl.buffer_version_mode", {})
        counts = annotation.get("tl.buffer_versions_map", {})
        for var in {*counts.keys(), *modes.keys()}:
            count, mode = counts.get(var), modes.get(var)
            value = (int(count), str(mode)) if count is not None and mode is not None else int(
                count) if count is not None else str(mode)
            alias_specs.setdefault(var, []).append(value)
        for var, count in annotation.get("tl.buffer_versions_map", {}).items():
            if int(count) == 1 and var not in modes:
                selected[var] = tir.IntImm("int32", 1)
    for var, specifications in alias_specs.items():
        plan_storage_epochs(var, (), specifications)
    if selected:
        function = function.with_attr("tl.ascend.selected_buffer_versions", selected)
    tasks = extraction.tasks
    if (not tasks or len({task.core for task in tasks}) != 1 or extraction.scalar_bindings or
            any(task.guard not in (None, ()) for task in tasks)):
        return function, extraction
    contexts = [extraction.loops[task.id] for task in tasks]
    if (any(len(context) != 1 for context in contexts) or
            any(not context[0].same_as(contexts[0][0]) for context in contexts)):
        return function, extraction
    owner = contexts[0][0]
    nodes = [extraction.task_nodes[task.id] for task in tasks]
    from tvm import ir
    if not ir.structural_equal(owner.body, nodes[0] if len(nodes) == 1 else tir.SeqStmt(nodes)):
        return function, extraction
    buffers = []
    physical_allocations = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: buffers.extend(node.alloc_buffers)
        if isinstance(node, tir.Block) else physical_allocations.append(node)
        if isinstance(node, tir.Allocate) else None)
    if physical_allocations:
        return function, extraction
    candidates = []
    for buffer in buffers:
        if (buffer.scope() != "shared.ub" or buffer.strides or
                not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset) or
                any(not isinstance(dim, tir.IntImm) or int(dim) <= 0 for dim in buffer.shape)):
            continue
        pitch = 1
        for dim in buffer.shape:
            pitch *= int(dim)
        ty = DataType(buffer.dtype)
        if int(ty.bits) * int(ty.lanes) % 8 or int(ty.bits) < 8:
            continue
        width = int(ty.bits) * int(ty.lanes) // 8
        if pitch * width % 32:
            continue
        storage = next((access.storage
                        for task in tasks
                        for access in task.accesses
                        if access.storage.var.same_as(buffer.data)), None)
        if storage is None:
            continue
        touching = [
            task for task in tasks if any(access.storage is storage for access in task.accesses)
        ]
        overwrite_ids = set()
        supported = True
        for task in touching:
            call = extraction.task_nodes[task.id].value
            pointers = [access for access in task.accesses if access.storage is storage]
            if len(pointers) != 1:
                supported = False
                break
            access = pointers[0]
            pointer = None
            if call.op.name == "tl.ascend_fill" and len(call.args) == 4:
                pointer = call.args[1]
                if (access.mode != "W" or not isinstance(call.args[3], tir.IntImm) or
                        int(call.args[3]) != pitch):
                    supported = False
                    break
                overwrite_ids.add(task.id)
            elif call.op.name == "tir.call_extern":
                tag = str(call.args[0].value)
                name = tag.split("<", 1)[0].split("::")[-1]
                if name != "copy_ub_to_gm" or access.mode != "R" or len(call.args) not in (4, 6):
                    supported = False
                    break
                try:
                    fields = tag.split("<", 1)[1].removesuffix(">").split(",")
                    cols, rows = int(fields[1]), int(fields[2]) if len(fields) == 3 else 1
                except (ValueError, IndexError):
                    supported = False
                    break
                if fields[0].strip() != {
                        "float16": "half",
                        "float32": "float"
                }.get(str(buffer.dtype)):
                    supported = False
                    break
                if cols <= 0 or rows <= 0 or cols * rows != pitch:
                    supported = False
                    break
                if len(call.args) == 6 and (not isinstance(call.args[4], tir.IntImm) or int(
                        call.args[4]) != rows or not isinstance(call.args[5], tir.IntImm) or
                                            int(call.args[5]) != cols):
                    supported = False
                    break
                pointer = call.args[1]
            else:
                supported = False
                break
            if (not isinstance(pointer, tir.Call) or pointer.op.name != "tir.tvm_access_ptr" or
                    str(pointer.args[0].dtype) != str(buffer.dtype) or
                    not isinstance(pointer.args[2], tir.IntImm) or int(pointer.args[2]) or
                    not isinstance(pointer.args[4], tir.IntImm) or
                    int(pointer.args[4]) != (2 if task.id in overwrite_ids else 1)):
                supported = False
                break
        if not supported or touching[0].id not in overwrite_ids:
            continue
        annotations = []
        for annotation in extraction.annotations:
            if not hasattr(annotation, "get"):
                continue
            counts = annotation.get("tl.buffer_versions_map", {})
            modes = annotation.get("tl.buffer_version_mode", {})
            count, mode = counts.get(buffer.data), modes.get(buffer.data)
            if count is not None or mode is not None:
                annotations.append((int(count), str(mode)) if count is not None and mode is not None
                                   else int(count) if count is not None else str(mode))
        proven_tasks = tuple(
            replace(
                task,
                accesses=tuple(
                    replace(
                        access,
                        offset=0,
                        extent=pitch * width,
                        intervals=None,
                        full_overwrite=task.id in overwrite_ids,
                        owner=owner) if access.storage is storage else access
                    for access in task.accesses))
            for task in tasks)
        descriptor = OwnerDescriptor(
            owner,
            proven_tasks,
            extraction.stages,
            full_footprints=((storage, ((0, pitch * width),)),))
        plan = plan_storage_epochs(storage, (descriptor,), annotations)
        if plan.eligible and plan.owners[0].mode == "iteration":
            candidates.append((buffer, plan, proven_tasks, pitch, width,
                               any(not isinstance(value, str) for value in annotations)))
    if len(candidates) != 1:
        return function, extraction
    buffer, plan, proven_tasks, pitch, width, hinted = candidates[0]
    # Report all true edges. Only a proven scratch owner's carried hazards may
    # become version-distance reuse constraints; persistent state stays fixed.
    dependencies = analyze_dependencies(proven_tasks, max_distance=1)
    indexes = {task.id: i for i, task in enumerate(tasks)}
    reuse = [
        ReuseDependency(indexes[d.source], indexes[d.target], "scratch")
        for d in dependencies
        if d.distance and d.storage is plan.storage
    ]
    fixed = [d for d in dependencies if not (d.distance and d.storage is plan.storage)]
    costs = {task.id: TaskCost(1, 1, task.core + ":" + task.pipe) for task in tasks}
    allocated = {}
    scopes = {
        "shared.ub": "UB",
        "shared.l1": "L1",
        "wmma.matrix_a": "L0A",
        "wmma.matrix_b": "L0B",
        "wmma.accumulator": "L0C"
    }
    for index, view in enumerate(buffers):
        if view.scope() not in scopes or any(not isinstance(dim, tir.IntImm) for dim in view.shape):
            return function, extraction
        elements = 1
        for dim in view.shape:
            size = int(dim)
            elements *= (size + 15) // 16 * 16 if view.scope() != "shared.ub" and len(
                view.shape) == 2 else size
        dtype = DataType(view.dtype)
        size = (elements * int(dtype.bits) * int(dtype.lanes) // 8 + 31) // 32 * 32
        if view.data.same_as(buffer.data):
            allocated["scratch"] = StorageCost(
                size, "UB", max_versions=plan.count, fixed_versions=plan.count if hinted else None)
        else:
            allocated[f"fixed_{index}"] = StorageCost(size, scopes[view.scope()], 1, 1)
    report = build_schedule_report(
        extraction,
        fixed,
        costs,
        storage=allocated,
        capacities=allocation_budgets(platform),
        reuse_dependencies=reuse)
    count = report["schedule"]["buffer_versions"]["scratch"]
    selected[buffer.data] = tir.IntImm("int32", count)
    function = function.with_attr("tl.ascend.selected_buffer_versions", selected)
    function = function.with_attr("tl.ascend.version_selection", json.dumps(report, sort_keys=True))
    if count == 1:
        return function, extraction
    function = materialize_iteration_versions(function, buffer, replace(plan, count=count))
    fresh = extract_tasks(function)
    # Re-derive identities from rewritten TIR. The physical rewrite checks every
    # original pointer against the slot pitch and preserves the helper operands.
    storage = next(
        access.storage
        for task in fresh.tasks
        for access in task.accesses
        if access.storage.var.same_as(buffer.data))
    ring_tasks = tuple(
        replace(
            task,
            accesses=tuple(
                replace(
                    access,
                    offset=0,
                    extent=pitch * width,
                    intervals=None,
                    iteration_stride=pitch * width,
                    ring_size=count) if access.storage is storage else access
                for access in task.accesses))
        for task in fresh.tasks)
    return function, replace(fresh, tasks=ring_tasks)
