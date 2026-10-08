# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Typed generated-workspace discovery and group-relative epoch lifecycles.

This path never turns a collective partition write into Access.full_overwrite.
It consumes the all-endpoint certificate and participant-uniform owner clock.
Joint capacities, all other dependencies and event allocation are separate
obligations before a caller may emit an executable stage schedule.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class WorkspaceCandidate:
    function: object
    buffer: object
    storage: object
    owner: object
    certificate: object
    clock: object
    writer_id: object
    reader_ids: tuple
    size_bytes: int
    group_bytes: int


@dataclass(frozen=True)
class WorkspaceDiscovery:
    candidates: tuple
    tasks: tuple
    diagnostics: tuple


def discover_workspace_epochs(function, extraction):
    """Certify generated parameters before any physical slot/wave rewriting."""
    from tvm import tir
    from .workspace_epoch import certify_workspace_epoch
    from .uniform_clock import certify_uniform_clock
    from .tir_adapter import load_operation_configs, operation_config, normalize_operation

    if not isinstance(function, tir.PrimFunc):
        raise ValueError("workspace discovery requires a PrimFunc")
    attrs = {} if function.attrs is None else function.attrs
    if attrs.get("address_map", {}):
        raise ValueError(
            "workspace physical addresses need a validated asynchronous alias contract")
    generated = {int(index) for index in attrs.get("auto_gm_indices", ())}
    if any(index < 0 or index >= len(function.params) for index in generated):
        raise ValueError("workspace generated parameter index is invalid")
    if extraction.scalar_bindings:
        return WorkspaceDiscovery((), tuple(extraction.tasks),
                                  ("workspace owner contains scalar bindings",))
    current_annotations = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: current_annotations.append(node)
        if isinstance(node, tir.AttrStmt) else current_annotations.append(node.annotations)
        if isinstance(node, tir.Block) else None)
    if any(
            getattr(annotation, "attr_key", "") == "tl.ascend_stage" or
        (hasattr(annotation, "keys") and any(
            str(key).startswith("tl.buffer_version")
            for key in annotation.keys()))
            for annotation in (*extraction.annotations, *current_annotations)):
        return WorkspaceDiscovery(
            (), tuple(extraction.tasks),
            ("workspace explicit stage/version constraints require the validated hint path",))
    configs = load_operation_configs()
    candidates, diagnostics, clocks = [], [], []
    for index in sorted(generated):
        try:
            parameter = function.params[index]
            if parameter not in function.buffer_map:
                raise ValueError("workspace parameter has no physical view")
            buffer = function.buffer_map[parameter]
            certificate = certify_workspace_epoch(function, buffer)
            touching = [
                task for task in extraction.tasks if any(
                    access.storage.var.same_as(buffer.data) for access in task.accesses)
            ]
            if not touching:
                raise ValueError("workspace has no extracted endpoints")
            storages = [
                access.storage
                for task in touching
                for access in task.accesses
                if access.storage.var.same_as(buffer.data)
            ]
            storage = storages[0]
            if storage.domain is not None:
                raise ValueError("workspace GM storage cannot have a core-private domain")
            if any(token is not storage for token in storages):
                raise ValueError("workspace has conflicting physical domains")
            for task in touching:
                loops = extraction.loops[task.id]
                if len(loops) != 1 or not loops[0].same_as(
                        certificate.owner) or task.guard not in (None, ()):
                    raise ValueError("workspace endpoints need one exclusive unguarded owner")
                accesses = [access for access in task.accesses if access.storage is storage]
                if len(accesses) != 1:
                    raise ValueError("workspace task has multiple aliased accesses")
            calls = [certificate.writer.call, *(reader.call for reader in certificate.readers)]
            ids = []
            for call in calls:
                matches = [
                    task for task in touching
                    if isinstance(extraction.task_nodes[task.id], tir.Evaluate) and
                    extraction.task_nodes[task.id].value.same_as(call)
                ]
                if len(matches) != 1:
                    raise ValueError("workspace certificate endpoint has no unique extracted task")
                name = normalize_operation(call.args[0].value)
                core = "C" if name in ("copy_l0c_to_gm", "copy_gm_to_l1") else "V"
                if matches[0].core != core or matches[0].pipe != operation_config(name,
                                                                                  configs).pipe:
                    raise ValueError("workspace task execution domain disagrees with native helper")
                ids.append(matches[0].id)
            if len(ids) != len(touching) or len(set(ids)) != len(ids):
                raise ValueError("workspace extraction and all-endpoint certificate disagree")
            modes = [
                next(access.mode
                     for task in touching
                     if task.id == task_id
                     for access in task.accesses if access.storage is storage)
                for task_id in ids
            ]
            if modes != ["W", *(["R"] * (len(ids) - 1))]:
                raise ValueError("workspace task access roles disagree with physical pointers")
            clock = next((clock for owner, clock in clocks if owner.same_as(certificate.owner)),
                         None)
            if clock is None:
                clock = certify_uniform_clock(function, certificate.owner)
                clocks.append((certificate.owner, clock))
            writer = certificate.writer
            width = writer.group_byte_extent // writer.group_pitch_elements
            candidates.append(
                WorkspaceCandidate(function, buffer, storage, certificate.owner, certificate, clock,
                                   ids[0], tuple(ids[1:]), writer.version_pitch_elements * width,
                                   writer.group_byte_extent))
        except ValueError as error:
            diagnostics.append(f"workspace parameter {index}: {error}")
    return WorkspaceDiscovery(tuple(candidates), tuple(extraction.tasks), tuple(diagnostics))


def plan_workspace_epochs(candidate, stages, count=None):
    """Plan iteration slots from typed group coverage, not whole-world hints.

    For ascending stage-wave reconstruction the next writer cannot share a wave
    with the prior epoch's last reader. Therefore count must exceed stage span.
    This validates this workspace only, not other storage dependencies, local
    capacity, GM allocation budgets or available event channels.
    """
    from .ownership import EpochPlan, OwnerEpochPlan
    from .buffer_versions import materialize_iteration_versions

    if not isinstance(candidate, WorkspaceCandidate):
        raise ValueError("workspace plan requires a typed discovered candidate")
    endpoint_ids = (candidate.writer_id, *candidate.reader_ids)
    if any(
            task_id not in stages or type(stages[task_id]) is not int or stages[task_id] < 0
            for task_id in endpoint_ids):
        raise ValueError("workspace endpoints require explicit nonnegative integer stages")
    writer_stage = stages[candidate.writer_id]
    reader_stages = [stages[task_id] for task_id in candidate.reader_ids]
    if any(stage < writer_stage for stage in reader_stages):
        raise ValueError("workspace reader stage precedes its producer")
    minimum = max(2, max(reader_stages) - writer_stage + 1)
    count = minimum if count is None else count
    if type(count) is not int or count < minimum:
        raise ValueError("workspace slot count violates its stage lifecycle")
    plan = EpochPlan(candidate.storage, True, count,
                     (OwnerEpochPlan(candidate.owner, "iteration", candidate.clock),))
    # Preflight the existing physical rewriter's aliases, all-use accounting,
    # pointer spans and integer-index safety. Do not retain this temporary IR.
    materialize_iteration_versions(candidate.function, candidate.buffer, plan)
    return plan
