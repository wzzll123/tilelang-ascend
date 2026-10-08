# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Expose the single-tile FP16 GEMM's physical pipe work in existing TIR.

The numerical copy/MMA recipe follows common.h gemm_v0, without its hidden
events or implicit scratch. Scratch allocation/lifetimes and synchronization
are the caller's responsibility; this helper never claims to emit a pipeline.
"""

import re


def _gemm_geometry(call):
    """Validate the supported template before creating any scratch allocations."""
    from tvm import tir

    if not isinstance(call, tir.Call) or call.op.name != "tl.ascend_gemm_v0" or len(call.args) != 6:
        raise ValueError("expected the six-argument gemm_v0 intrinsic")
    if not isinstance(call.args[0], tir.StringImm):
        raise ValueError("gemm_v0 template must be static")
    match = re.fullmatch(
        r"gemm_v0<\s*half\s*,\s*float\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
        r"\s*(true|false)\s*,\s*(true|false)\s*,\s*(\d+)\s*>", call.args[0].value)
    if match is None:
        raise ValueError("GEMM decomposition requires static FP16/FP32 template dimensions")
    m, n, k = map(int, match.group(1, 2, 3))
    transpose_a, transpose_b = match.group(4) == "true", match.group(5) == "true"
    tile_k = int(match.group(6))
    if any(dim < 16 or dim > 128 or dim % 16 for dim in (m, n, k)) or tile_k != 128:
        raise ValueError("GEMM decomposition supports aligned M/N/K<=128 with kL0Size=128")
    return m, n, k, transpose_a, transpose_b


def decompose_gemm_v0(call, l0a, l0b):
    """Return MTE1(A), MTE1(B), M tasks using caller-owned L0 buffers.

    First supported family is FP16 input/FP32 accumulation with static aligned
    M/N/K <=128 and one K tile. Runtime n_actual is retained only on transpose-B,
    exactly as in the original helper. Unsupported forms fail rather than leave
    an opaque multi-pipe function in the task graph.
    """
    from tvm import arith, tir

    m, n, k, transpose_a, transpose_b = _gemm_geometry(call)

    def pointer(ptr, dtype, scope, required_mask):
        if (not isinstance(ptr, tir.Call) or ptr.op.name != "tir.tvm_access_ptr" or
                len(ptr.args) != 5 or str(ptr.args[0].dtype) != dtype or
                not isinstance(ptr.args[1], tir.Var)):
            raise ValueError("invalid GEMM operand access pointer")
        annotation = ptr.args[1].type_annotation
        if str(getattr(annotation, "storage_scope", "")) != scope:
            raise ValueError(f"GEMM operand requires storage scope {scope}")
        mask = ptr.args[4]
        if not isinstance(mask, tir.IntImm) or not int(mask) & required_mask:
            raise ValueError("GEMM operand access mask contradicts its operation")

    for ptr in call.args[1:3]:
        pointer(ptr, "float16", "shared.l1", 1)
    pointer(call.args[3], "float32", "wmma.accumulator", 2)
    analyzer = arith.Analyzer()
    for ptr, required in zip(call.args[1:4], (m * k, n * k, m * n)):
        extent = analyzer.simplify(ptr.args[3])
        if not isinstance(extent, tir.IntImm) or int(extent) < required:
            raise ValueError("GEMM operand capacity must cover its complete static tile")
    for buffer, scope, count in ((l0a, "wmma.matrix_a", m * k), (l0b, "wmma.matrix_b", n * k)):
        if not isinstance(buffer, tir.Buffer) or str(
                buffer.dtype) != "float16" or buffer.scope() != scope:
            raise ValueError(f"GEMM scratch must be FP16 {scope}")
        if buffer.strides or not isinstance(buffer.elem_offset, tir.IntImm) or int(
                buffer.elem_offset) != 0:
            raise ValueError("GEMM scratch requires a compact allocation base")
        size = 1
        for extent in buffer.shape:
            if not isinstance(extent, tir.IntImm):
                raise ValueError("GEMM scratch capacity must be static")
            size *= int(extent)
        if size < count or size * 2 > 65536:
            raise ValueError("GEMM scratch capacity does not fit the tile or A2/A3 L0 budget")
    if l0a.data.same_as(l0b.data) or any(
            scratch.data.same_as(ptr.args[1]) for scratch in (l0a, l0b) for ptr in call.args[1:4]):
        raise ValueError("GEMM scratch must have independent allocation identities")
    actual_n = call.args[5] if transpose_b else tir.IntImm("int32", n)
    if isinstance(actual_n, tir.IntImm) and (int(actual_n) <= 0 or int(actual_n) > n or
                                             int(actual_n) % 16):
        raise ValueError("GEMM actual N must be positive, <=N, and fractal aligned")

    a_template = f"copy_l1_to_l0a<half,{k},{m},true>" if transpose_a else f"copy_l1_to_l0a<half,{m},{k},false>"
    b_template = f"copy_l1_to_l0b<half,{n},{k},true>" if transpose_b else f"copy_l1_to_l0b<half,{k},{n},false>"
    copy_a = tir.call_extern("handle", "tl::ascend::" + a_template, call.args[1],
                             l0a.access_ptr("w"), m, k)
    copy_b = tir.call_extern("handle", "tl::ascend::" + b_template, call.args[2],
                             l0b.access_ptr("w"), k, actual_n)
    mma = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_mma"), f"mma<half,float,{m},{n}>",
                          l0a.access_ptr("r"), l0b.access_ptr("r"), call.args[3], call.args[4], k,
                          actual_n, 0)
    return tir.SeqStmt([tir.Evaluate(copy_a), tir.Evaluate(copy_b), tir.Evaluate(mma)])


def decompose_function_gemms(function, *, root_scratch=False):
    """Expose GEMM tasks and allocate logical scratch at each statement site.

    This transformation deliberately preserves enclosing loops, branches and
    Stage annotations. It does not schedule, emit synchronization, rotate slots,
    or grant permission to alias these allocations. Those are separate steps.
    Late Flatten2DBuffer/AscendMemoryPlanning discover the new 2D allocations.
    With root_scratch, only buffers created in this invocation are allocated at
    an unconditional common root outside all serial owners. This deliberately
    retains distinct callsite identities and exposes their carried reuse edges.
    It never hoists user allocations based on their names.
    """
    from tvm import tir

    if not isinstance(function, tir.PrimFunc):
        raise TypeError("GEMM decomposition requires a PrimFunc")
    names = {param.name for param in function.params}
    gemms = []

    def collect(node):
        if isinstance(node, tir.Var):
            names.add(node.name)
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tl.ascend_gemm_v0":
            gemms.append(node)

    tir.stmt_functor.post_order_visit(function.body, collect)
    if not gemms:
        return function

    root = None
    if root_scratch:
        candidates = []

        def find_root(node, unconditional=True):
            if isinstance(node, tir.BlockRealize):
                valid = (
                    unconditional and isinstance(node.predicate, tir.IntImm) and
                    bool(int(node.predicate)) and not node.iter_values and
                    not node.block.iter_vars and node.block.init is None and
                    not node.block.match_buffers)
                contained = []
                tir.stmt_functor.post_order_visit(
                    node.block.body,
                    lambda item: contained.append(item) if isinstance(item, tir.Call) and getattr(
                        item.op, "name", None) == "tl.ascend_gemm_v0" else None)
                if valid and len(contained) == len(gemms):
                    candidates.append(node.block)
                find_root(node.block.body, valid)
            elif isinstance(node, (tir.AttrStmt, tir.Allocate)):
                if isinstance(node, tir.Allocate):
                    unconditional = (
                        unconditional and isinstance(node.condition, tir.IntImm) and
                        bool(int(node.condition)))
                find_root(node.body, unconditional)
            elif isinstance(node, tir.SeqStmt):
                for child in node.seq:
                    find_root(child, unconditional)
            # Roots beneath a loop/branch do not establish a kernel lifetime.

        find_root(function.body)
        if not candidates:
            raise ValueError("GEMM scratch requires an unconditional common root outside owners")
        root = candidates[0]
    scratch_buffers = []

    def fresh_name(prefix):
        index = 0
        while f"{prefix}_{index}" in names:
            index += 1
        name = f"{prefix}_{index}"
        names.add(name)
        return name

    def rewrite(node):
        if (not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call) or
                getattr(node.value.op, "name", None) != "tl.ascend_gemm_v0"):
            return None
        m, n, k, _, _ = _gemm_geometry(node.value)
        a = tir.decl_buffer((m, k), "float16", name=fresh_name("auto_l0a"), scope="wmma.matrix_a")
        b = tir.decl_buffer((k, n), "float16", name=fresh_name("auto_l0b"), scope="wmma.matrix_b")
        body = decompose_gemm_v0(node.value, a, b)
        if root is not None:
            scratch_buffers.extend((a, b))
            return body
        for scratch in (b, a):
            body = tir.Allocate(scratch.data, scratch.dtype, scratch.shape, tir.const(True, "bool"),
                                body)
        return body

    if root is not None:
        # Replace the certified root pre-order; only its body is decomposed.
        # The fresh buffer registry is populated exactly once, before attaching
        # it to that root. No rebuilt-node name matching or arbitrary hoist.
        def attach(node):
            if node.same_as(root):

                def expand(item):
                    result = rewrite(item)
                    if result is not None:
                        return result
                    if isinstance(item, tir.SeqStmt):
                        children = []
                        for child in item.seq:
                            children.extend(
                                child.seq if isinstance(child, tir.SeqStmt) else [child])
                        return children[0] if len(children) == 1 else tir.SeqStmt(children)
                    return None

                rewritten = tir.stmt_functor.ir_transform(node.body, None, expand)
                return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint, rewritten,
                                 node.init, [*node.alloc_buffers, *scratch_buffers],
                                 node.match_buffers, node.annotations, getattr(node, "span", None))
            return None

        body = tir.stmt_functor.ir_transform(function.body, attach, None)
    else:
        body = tir.stmt_functor.ir_transform(function.body, None, rewrite)
    remaining = []
    tir.stmt_functor.post_order_visit(
        body, lambda node: remaining.append(node) if isinstance(node, tir.Call) and getattr(
            node.op, "name", None) == "tl.ascend_gemm_v0" else None)
    if remaining:
        raise ValueError("GEMM decomposition requires standalone Evaluate statements")
    return function.with_body(body)
