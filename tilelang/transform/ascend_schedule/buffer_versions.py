# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Physical storage materialization for certified regular iteration owners.

The storage-pitch/clock separation follows the design in upstream TileLang
47976f2 src/ascend/transform/materialize_multi_buffer.cc (Apache-2.0). This is
an independent older-TIR implementation, not a TIRX source copy. Synchronization
must use the same owner clock and protect slot reuse; this pass emits no events.
"""


def materialize_iteration_versions(function, buffer, plan):
    """Expand a compact allocation and shift lowered access_ptr element offsets.

    Accepts one already-proven iteration owner. Counter clocks, logical scalar
    buffer indexing, noncompact/fractal-unaligned storage and accesses outside
    the owner are rejected, not silently versioned. Per-access extent remains
    unchanged. The allocation identity is preserved for alias/dependency maps.
    """
    from tvm import DataType, ir, tir
    from .resources import allocation_budgets

    if not isinstance(function, tir.PrimFunc) or not isinstance(buffer, tir.Buffer):
        raise ValueError("version materialization requires a PrimFunc and Buffer")
    storage = getattr(plan.storage, "var", plan.storage)
    if not isinstance(storage, tir.Var) or not storage.same_as(buffer.data):
        raise ValueError("version plan must describe the physical buffer storage")
    if not plan.eligible or type(plan.count) is not int or plan.count < 2:
        raise ValueError("version materialization requires a proven eligible multi-version plan")
    if len(plan.owners) != 1 or plan.owners[0].mode != "iteration":
        raise ValueError("version materialization currently requires one iteration owner")
    owner = plan.owners[0].loop
    if not isinstance(owner, tir.For) or owner.kind != tir.ForKind.SERIAL:
        raise ValueError("iteration owner must be a serial For")
    if buffer.strides or not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset):
        raise ValueError("version materialization requires compact zero-base storage")
    shape = tuple(buffer.shape)
    if not shape or any(not isinstance(x, tir.IntImm) or int(x) <= 0 for x in shape):
        raise ValueError("version allocation shape must be positive and static")
    scope = buffer.scope()
    if scope not in ("global", "shared.ub", "shared.l1", "wmma.matrix_a", "wmma.matrix_b",
                     "wmma.accumulator"):
        raise ValueError("unsupported versioned storage scope")
    if scope not in ("global", "shared.ub") and (len(shape) != 2 or
                                                 any(int(x) % 16 for x in shape)):
        raise ValueError("Cube version tiles require aligned 2D fractal dimensions")
    pitch = 1
    for extent in shape:
        pitch *= int(extent)
    # A whole version must also satisfy DMA/vector block alignment.
    dtype = DataType(buffer.dtype)
    bits = int(dtype.bits) * int(dtype.lanes)
    if bits % 8 or pitch * bits // 8 % 32:
        raise ValueError("version storage pitch must be 32-byte aligned")
    budget_key = {
        "shared.ub": "UB",
        "shared.l1": "L1",
        "wmma.matrix_a": "L0A",
        "wmma.matrix_b": "L0B",
        "wmma.accumulator": "L0C"
    }.get(scope)
    if budget_key is not None and pitch * bits // 8 * plan.count > allocation_budgets(
            "A2")[budget_key]:
        raise ValueError("version allocation exceeds the A2/A3 per-core storage budget")
    enlarged = [shape[0] * plan.count, *shape[1:]]
    active, found_owner, allocations, accesses = [], 0, 0, 0
    gm_handle = None
    if scope == "global":
        handles = [
            handle for handle, view in function.buffer_map.items() if view.data.same_as(storage)
        ]
        if len(handles) != 1:
            raise ValueError("versioned GM workspace needs one unambiguous parameter view")
        gm_handle = handles[0]
        view = function.buffer_map[gm_handle]
        if (str(view.dtype) != str(buffer.dtype) or view.strides or len(view.shape) != len(shape) or
                any(int(a) != int(b) for a, b in zip(view.shape, shape))):
            raise ValueError("GM version pitch must match the actual parameter allocation view")
        indices = function.attrs.get("auto_gm_indices", ()) if function.attrs is not None else ()
        position = next((i for i, param in enumerate(function.params) if param.same_as(gm_handle)),
                        None)
        if position not in {int(i) for i in indices}:
            raise ValueError("only generated scratch GM workspaces may be versioned")
        allocations = 1
    from .uniform_clock import certify_uniform_clock
    analyzer = certify_uniform_clock(function, owner).analyzer
    analyzer.bind(owner.loop_var, ir.Range.from_min_extent(owner.min, owner.extent))
    calls = []
    allocation_scopes = []

    def gm_copy_span(parent, pointer):
        from .workspace_proof import workspace_copy_span
        return workspace_copy_span(parent, pointer, analyzer)

    def bind_thread(node):
        if (isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and
                isinstance(node.node, tir.IterVar) and isinstance(node.value, tir.IntImm)):
            analyzer.bind(node.node.var, ir.Range.from_min_extent(0, node.value))

    tir.stmt_functor.post_order_visit(function.body, bind_thread)

    def before(node):
        nonlocal found_owner
        if isinstance(node, tir.Allocate):
            allocation_scopes.append(node.buffer_var.same_as(storage))
        elif isinstance(node, tir.Block):
            allocation_scopes.append(any(view.data.same_as(storage) for view in node.alloc_buffers))
        if isinstance(node, tir.BlockRealize) and not analyzer.can_prove(node.predicate):
            raise ValueError("predicated block allocation needs a version-aware lifecycle proof")
        if isinstance(node, tir.Call):
            calls.append(node)
        if isinstance(node, tir.For):
            matching = node.same_as(owner)
            if matching and gm_handle is None and not any(allocation_scopes):
                raise ValueError("version allocation must dominate the owner on its ancestor path")
            found_owner += int(matching)
            active.append(matching)
        return None

    def after(node):
        nonlocal allocations, accesses
        if isinstance(node, (tir.Allocate, tir.Block)):
            allocation_scopes.pop()
        parent_call = calls[-2] if isinstance(node, tir.Call) and len(calls) >= 2 else None
        if isinstance(node, tir.Call):
            calls.pop()
        if isinstance(node, tir.For):
            active.pop()
        if isinstance(node,
                      (tir.BufferLoad, tir.BufferStore)) and node.buffer.data.same_as(storage):
            raise ValueError(
                "logical buffer indexing must be lowered before version materialization")
        if isinstance(node, tir.DeclBuffer) and node.buffer.data.same_as(storage):
            raise ValueError("logical buffer declarations need a version-aware view rewrite")
        if isinstance(node, tir.Call) and getattr(
                node.op, "name", None) != "tir.tvm_access_ptr" and any(
                    isinstance(arg, tir.Var) and arg.same_as(storage) for arg in node.args):
            raise ValueError(
                "raw buffer handles cannot escape the versioned access-pointer rewrite")
        if (isinstance(node, tir.Call) and
                getattr(node.op, "name", None) == "tir.tvm_access_ptr" and len(node.args) == 5 and
                isinstance(node.args[1], tir.Var) and node.args[1].same_as(storage)):
            if not any(active):
                raise ValueError("versioned storage is accessed outside its certified owner")
            if str(node.args[0].dtype) != str(buffer.dtype):
                raise ValueError("mixed-dtype aliases require a physical bit-pitch proof")
            span = node.args[3]
            if scope == "global" and not analyzer.can_prove(node.args[2] + span <= pitch):
                proven_span = gm_copy_span(parent_call, node)
                if proven_span is not None:
                    span = tir.IntImm("int32", proven_span)
            if not (analyzer.can_prove(node.args[2] >= 0) and analyzer.can_prove(span >= 0) and
                    analyzer.can_prove(node.args[2] + span <= pitch)):
                raise ValueError("access pointer is not proven to stay inside one version")
            slot = tir.floormod(owner.loop_var - owner.min, plan.count)
            offset_dtype = str(node.args[2].dtype)
            index_type = DataType(offset_dtype)
            if index_type.lanes != 1 or not offset_dtype.startswith(("int", "uint")):
                raise ValueError("version offset requires a scalar integer index")
            index_high = (1 << (index_type.bits - 1)) - 1 if offset_dtype.startswith(
                "int") else (1 << index_type.bits) - 1
            offset_bound = analyzer.const_int_bound(node.args[2])
            if (plan.count - 1) * pitch + offset_bound.max_value > index_high:
                raise ValueError("version slot offset arithmetic may overflow")
            offset = analyzer.simplify(node.args[2] + tir.Cast(offset_dtype, slot) *
                                       tir.IntImm(offset_dtype, pitch))
            accesses += 1
            return tir.Call(node.dtype, node.op, [*node.args[:2], offset, *node.args[3:]],
                            getattr(node, "span", None))
        if isinstance(node, tir.Allocate) and node.buffer_var.same_as(storage):
            if any(active) or not analyzer.can_prove(node.condition):
                raise ValueError("version allocation must unconditionally dominate its owner")
            if str(node.dtype) != str(buffer.dtype) or len(node.extents) != len(shape) or any(
                    not analyzer.can_prove_equal(a, b) for a, b in zip(node.extents, shape)):
                raise ValueError("physical allocation does not match certified version pitch")
            allocations += 1
            return tir.Allocate(node.buffer_var, node.dtype, enlarged, node.condition, node.body,
                                node.annotations, getattr(node, "span", None))
        if isinstance(node, tir.Block) and any(b.data.same_as(storage) for b in node.alloc_buffers):
            if any(active
                  ) or node.reads or node.writes or node.match_buffers or node.init is not None:
                raise ValueError(
                    "logical block views or inner allocation need lifecycle reconstruction")
            buffers = []
            for view in node.alloc_buffers:
                if not view.data.same_as(storage):
                    buffers.append(view)
                    continue
                if (str(view.dtype) != str(buffer.dtype) or view.strides or
                        not analyzer.can_prove_equal(view.elem_offset, 0) or
                        len(view.shape) != len(shape) or
                        any(not analyzer.can_prove_equal(a, b) for a, b in zip(view.shape, shape))):
                    raise ValueError("physical allocation does not match certified version pitch")
                allocations += 1
                buffers.append(
                    tir.decl_buffer(
                        enlarged,
                        view.dtype,
                        name=view.name,
                        data=storage,
                        scope=scope,
                        data_alignment=view.data_alignment,
                        offset_factor=view.offset_factor))
            return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint, node.body,
                             node.init, buffers, node.match_buffers, node.annotations,
                             getattr(node, "span", None))
        return None

    body = tir.stmt_functor.ir_transform(function.body, before, after)
    if found_owner != 1 or allocations != 1 or not accesses:
        raise ValueError(
            "version materialization needs one owner, one allocation and actual accesses")
    buffers = dict(function.buffer_map.items())
    if gm_handle is not None:
        buffers[gm_handle] = tir.decl_buffer(
            enlarged,
            buffer.dtype,
            name=buffer.name,
            data=storage,
            scope=scope,
            data_alignment=buffer.data_alignment,
            offset_factor=buffer.offset_factor)
    result = tir.PrimFunc(function.params, body, function.ret_type, buffers, function.attrs,
                          getattr(function, "span", None))
    from .workspace_proof import VERSION_ATTR
    records = dict(function.attrs.get(VERSION_ATTR,
                                      {}).items()) if function.attrs is not None else {}
    records[storage] = [
        tir.IntImm("int32", plan.count),
        tir.IntImm("int64", pitch), owner.loop_var, owner.min
    ]
    return result.with_attr(VERSION_ATTR, records)
