# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Reuse numerical CV-copy lowering without inserting legacy synchronization."""


def prepare_cross_core_copies(mod):
    """Expose existing AscendC workspace writes/reads as separate pipe tasks.

    The native workspace pass owns row pitches, dtype conversions, per-core/vid
    offsets, workspace parameters and first-use reload placement. It emits no
    flags. Scheduling must separately order the resulting C/V physical tasks
    with a proven mode-2 protocol; a GM data edge is not an execution fence.
    """
    import tilelang
    from tvm import tir
    from .tir_adapter import normalize_operation

    composites = []

    def find(node):
        if (isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tir.call_extern" and
                node.args and isinstance(node.args[0], tir.StringImm) and
                normalize_operation(node.args[0].value) in ("copy_ub_to_l1", "copy_l0c_to_ub")):
            composites.append(node)

    for function in mod.functions.values():
        if isinstance(function, tir.PrimFunc):
            tir.stmt_functor.post_order_visit(function.body, find)
    # The native pass also refreshes workspace attrs on composite-free input;
    # do not rerun it or mutate unrelated functions when no GM hop is present.
    if not composites:
        return mod
    return tilelang.transform.AscendWorkspaceReduction()(mod)
