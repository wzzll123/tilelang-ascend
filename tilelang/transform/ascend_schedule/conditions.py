# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Make runtime branch decisions explicit at their original lexical position."""

CONDITION_SNAPSHOT = "tl.ascend_condition_snapshot"


def normalize_conditions(node):
    """Bind each If decision once at its original position, retaining spans.

    Marked wrappers are skipped on repeat calls. Snapshots stay inside loops
    and preceding scalar bindings. The compiler applies this helper only in
    the opted-in scheduling path; disabled scheduling preserves the input.
    """
    from tvm import tir

    function = node if isinstance(node, tir.PrimFunc) else None
    body = function.body if function is not None else node
    names = set()
    tir.stmt_functor.post_order_visit(
        body, lambda item: names.add(item.name) if isinstance(item, tir.Var) else None)
    counter = 0

    def before(item):
        if isinstance(item, tir.AttrStmt) and item.attr_key == CONDITION_SNAPSHOT:
            return item
        return None

    def after(item):
        nonlocal counter
        if not isinstance(item, tir.IfThenElse):
            return None
        while f"ascend_condition_snapshot_{counter}" in names:
            counter += 1
        name = f"ascend_condition_snapshot_{counter}"
        names.add(name)
        counter += 1
        snapshot = tir.Var(name, "bool", span=item.condition.span)
        branch = tir.IfThenElse(snapshot, item.then_case, item.else_case, span=item.span)
        binding = tir.LetStmt(snapshot, item.condition, branch, span=item.span)
        return tir.AttrStmt(
            snapshot, CONDITION_SNAPSHOT, tir.const(1, "int32"), binding, span=item.span)

    normalized = tir.stmt_functor.ir_transform(body, before, after)
    return function.with_body(
        normalized, span=function.span) if function is not None else normalized
