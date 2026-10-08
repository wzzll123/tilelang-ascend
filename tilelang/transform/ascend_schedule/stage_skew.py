# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Reconstruct executable stage-skew waves from already synchronized epochs.

This component requires complete analyzed dependences and certified storage
versions from its caller. It neither invents storage eligibility nor removes
events. The main scheduling driver must supply that proof before using it.
"""


def materialize_staged_owner(function, owner, task_nodes, stages, dependencies):
    """Move whole task/event bundles to their logical epoch in a staged wave.

    Task(e) with stage s runs in wave e+s. An edge a(e)->b(e+d) therefore
    requires s[a] <= s[b]+d; equality additionally requires actual wave order.
    Timing estimates are irrelevant to this proof. Endpoint flag slot formulas
    are substituted together with the task's pointer/address expressions.
    """
    from tvm import DataType, tir
    from .uniform_clock import certify_uniform_clock

    certificate = certify_uniform_clock(function, owner)
    if not task_nodes or set(stages) != set(task_nodes) or any(
            type(stage) is not int or stage < 0 for stage in stages.values()):
        raise ValueError("stage reconstruction requires complete nonnegative integer assignments")
    max_stage = max(stages.values())
    if min(stages.values()) != 0 or max_stage == 0:
        raise ValueError("stage reconstruction requires stage zero and a later stage")
    dtype = DataType(owner.loop_var.dtype)
    if not str(dtype).startswith("int") or dtype.lanes != 1:
        raise ValueError("stage wave requires a signed scalar integer owner")
    if certificate.extent_max + max_stage > (1 << (dtype.bits - 1)) - 1:
        raise ValueError("stage wave extent may overflow")
    markers = list(task_nodes.items())
    if any(
            left.same_as(right)
            for i, (_, left) in enumerate(markers)
            for _, right in markers[i + 1:]):
        raise ValueError("stage task nodes must have unique identities")
    sync_ops = {
        "tl.ascend_set_flag", "tl.ascend_wait_flag", "tl.ascend_set_cross_flag",
        "tl.ascend_wait_cross_flag", "tl.ascend_pipe_barrier"
    }

    def identities(node):
        if isinstance(node, tir.Evaluate):
            found = [key for key, original in markers if node.same_as(original)]
            if not found and not (isinstance(node.value, tir.Call) and
                                  getattr(node.value.op, "name", None) in sync_ops):
                raise ValueError("stage bundle contains an unclassified operation")
            return found
        if isinstance(node, tir.SeqStmt):
            return [key for child in node.seq for key in identities(child)]
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope":
            return identities(node.body)
        raise ValueError("stage bundle requires flat unconditional scoped operations")

    bundles = []

    def collect(node, scopes=()):
        keys = identities(node)
        if not keys:
            raise ValueError("stage bundle has an orphan synchronization instruction")
        if len(keys) == 1:
            wrapped = node
            for scope in reversed(scopes):
                wrapped = tir.AttrStmt(scope.node, scope.attr_key, scope.value, wrapped)
            bundles.append((keys[0], wrapped))
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                collect(child, scopes)
        elif isinstance(node, tir.AttrStmt):
            collect(node.body, (*scopes, node))
        else:
            raise ValueError("stage bundle cannot be split safely")

    collect(owner.body)
    ids = [key for key, _ in bundles]
    if len(ids) != len(task_nodes) or set(ids) != set(task_nodes):
        raise ValueError("stage bundles must cover every task exactly once")
    positions = {key: index for index, key in enumerate(ids)}
    ordered = sorted(bundles, key=lambda item: (stages[item[0]], positions[item[0]]))
    wave_positions = {key: index for index, (key, _) in enumerate(ordered)}
    for edge in dependencies:
        if (edge.source not in stages or edge.target not in stages or
                edge.kind not in ("RAW", "WAR", "WAW") or type(edge.distance) is not int or
                edge.distance < 0):
            raise ValueError("stage reconstruction requires valid analyzed dependences")
        source_wave, target_wave = stages[edge.source], stages[edge.target] + edge.distance
        if source_wave > target_wave or (source_wave == target_wave and wave_positions[edge.source]
                                         >= wave_positions[edge.target]):
            raise ValueError("stage skew violates an analyzed dependence")
    wave = tir.Var(owner.loop_var.name + "_wave", owner.loop_var.dtype)
    body = []
    for key, bundle in ordered:
        stage = stages[key]
        logical_epoch = owner.min + (wave - stage)
        rewritten = tir.stmt_functor.substitute(bundle, {owner.loop_var: logical_epoch})
        # Every wait/set shares the operation's guard, including unused slots
        # in zero/one-trip owners. Seed and drain remain outside this loop.
        valid = tir.And(wave >= stage, wave < owner.extent + stage)
        body.append(tir.IfThenElse(valid, rewritten, None))
    replacement = tir.For(
        wave,
        0,
        owner.extent + max_stage,
        tir.ForKind.SERIAL,
        tir.SeqStmt(body),
        annotations=owner.annotations,
        span=owner.span)
    rewritten = tir.stmt_functor.ir_transform(
        function.body, None, lambda node: replacement if node.same_as(owner) else None)
    return function.with_body(rewritten)
