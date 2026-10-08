# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Assign physical task domains without invoking legacy CombineCV rewriting."""


def normalize_core_scopes(function, extraction):
    """Place proven C/V operations while keeping a mixed owner core-neutral.

    Only unconditional standalone physical calls are supported. Immediate tensor
    scalar reads need a separate scalar task/fence and are deliberately rejected;
    the enclosing vector call is not a proof that the scalar read runs on V.
    """
    from tvm import tir

    tasks = {task.id: task for task in extraction.tasks}
    nodes = list(extraction.task_nodes.items())
    if not tasks:
        return function
    if extraction.scalar_bindings or any(
            task.guard not in (None, ()) or task.pipe == "PIPE_S" for task in tasks.values()):
        # Do not weaken the materializer's existing structured-control gate.
        return function
    if any(
            left.same_as(right)
            for index, (_, left) in enumerate(nodes)
            for _, right in nodes[index + 1:]):
        raise ValueError("core placement requires unique physical node identities")
    if any(task.core not in ("C", "V") for task in tasks.values()):
        raise ValueError("core placement requires proven C/V task ownership")
    missing, seen = [], []

    def visit(node, core=None):
        if isinstance(node, tir.AttrStmt):
            if node.attr_key == "resource_scope":
                if not isinstance(node.value, tir.IntImm) or int(node.value) not in (0, 1):
                    raise ValueError("core placement requires explicit C/V scopes")
                selected = "C" if int(node.value) == 0 else "V"
                if core is not None and selected != core:
                    raise ValueError("contradictory nested core scopes are not executable")
                core = selected
            visit(node.body, core)
        elif isinstance(node, (tir.For, tir.Allocate, tir.Block)):
            visit(node.body, core)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block, core)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, core)
        elif isinstance(node, tir.Evaluate):
            task_id = next((i for i, original in nodes if node.same_as(original)), None)
            if task_id is None:
                if isinstance(node.value, tir.IntImm):
                    return
                raise ValueError("core placement operation is absent from extraction")
            if task_id in seen:
                raise ValueError("core placement task must occur exactly once")
            seen.append(task_id)
            call = node.value
            if not isinstance(call, tir.Call):
                raise ValueError("core placement requires standalone physical calls")
            immediate = []
            tir.stmt_functor.post_order_visit(
                call, lambda item: immediate.append(item)
                if isinstance(item, tir.BufferLoad) else None)
            # CodeGenTileLangAscend::AddsAndMulsOpCodegen reads arg2.GetValue
            # immediately and injects PIPE_ALL for the pointer-scalar variant.
            # Neither action is represented as this call's physical V task.
            scalar_ops = {"tl.ascend_adds", "tl.ascend_subs", "tl.ascend_muls", "tl.ascend_divs"}
            pointer_scalar = (
                call.op.name in scalar_ops and len(call.args) > 2 and
                isinstance(call.args[2], tir.Call) and call.args[2].op.name == "tir.tvm_access_ptr")
            if immediate or pointer_scalar:
                raise ValueError("immediate scalar tensor read requires scalar task decomposition; "
                                 "use a vector broadcast instead")
            if core is not None and core != tasks[task_id].core:
                raise ValueError("task ownership contradicts its enclosing core scope")
            if core is None:
                missing.append((node, tasks[task_id].core))
        else:
            raise ValueError("core placement contains unreconstructed scalar/control statements")

    visit(function.body)
    if seen != [task.id for task in extraction.tasks]:
        raise ValueError("core placement must cover the complete lexical task stream")
    if not missing:
        return function

    def scope(node, core):
        code = 0 if core == "C" else 1
        return tir.AttrStmt(tir.IntImm("int32", code), "resource_scope", code, node)

    cores = {task.core for task in tasks.values()}
    if len(cores) == 1:
        # Keep a local owner body flat; initialization/drain inherit this scope.
        return function.with_body(scope(function.body, next(iter(cores))))

    def rewrite(node):
        for original, core in missing:
            if node.same_as(original):
                return scope(node, core)
        return None

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, rewrite))
