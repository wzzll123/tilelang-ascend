# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Typed group-partition coverage, not whole-workspace initialization hints.

A writer certificate alone NEVER authorizes versioning. All access endpoints
must subsequently prove the same group partition and a common owner clock.
The physical version pitch is the entire generated GM allocation, whereas
coverage below is relative to one kernel group. No Access metadata is changed.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class WorkspaceWritePartition:
    storage: object
    handle: object
    call: object
    pointer: object
    group_var: object
    participant_var: object
    group_count: int
    group_pitch_elements: int
    version_pitch_elements: int
    group_byte_extent: int
    participant_intervals: tuple


def certify_workspace_epoch_write(function, buffer, call, core):
    """Certify an unconditional native writer's complete per-group partition.

    Currently accepts generated compact RowMajor FIX writes and two-Vector
    contiguous half-tile MTE3 writes, with one dominating blockIdx.x axis.
    Unknown geometry/axes and guarded writes fail closed. Ordinary GM inputs,
    wrong pointer roles, gaps, overlapping shards and sibling axes are rejected.
    This does not certify any reader or a synchronization/lifetime protocol.
    """
    from tvm import DataType, arith, ir, tir

    def require(condition, message):
        if not condition:
            raise ValueError("workspace epoch: " + message)

    require(
        isinstance(function, tir.PrimFunc) and isinstance(buffer, tir.Buffer),
        "requires a PrimFunc and Buffer")
    require(
        core in ("C", "V") and isinstance(call, tir.Call) and
        getattr(call.op, "name", None) == "tir.call_extern" and len(call.args) >= 6 and
        isinstance(call.args[0], tir.StringImm), "requires an actual native write helper")
    name, sep, template = call.args[0].value.removeprefix("tl::ascend::").partition("<")
    fields = template.removesuffix(">").split(",")
    types = {"half": "float16", "float": "float32"}
    require(bool(sep) and call.args[0].value.endswith(">"), "unrecognized native template")
    try:
        if core == "C":
            require(
                name == "copy_l0c_to_gm" and len(fields) == 6 and
                fields[2].strip() == "layout::RowMajor", "FIX requires RowMajor geometry")
            # This parameter is enRelu, not atomic mode. Both recognized
            # values preserve full write coverage; atomic has another helper.
            require(fields[5].strip() in ("true", "false"), "unknown enRelu template")
            source_dtype, dtype = types[fields[0].strip()], types[fields[1].strip()]
            rows, cols = int(fields[3]), int(fields[4])
            shard_rows, participants = rows, 1
        else:
            require(name == "copy_ub_to_gm" and len(fields) == 3,
                    "Vector requires the generated contiguous MTE3 helper")
            source_dtype = dtype = types[fields[0].strip()]
            cols, shard_rows = int(fields[1]), int(fields[2])
            rows, participants = 2 * shard_rows, 2
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError("workspace epoch: unsupported helper dtype/geometry") from error
    require(rows > 0 and cols > 0, "positive static geometry is required")
    require(
        buffer.scope() == "global" and str(buffer.dtype) == dtype and not buffer.strides and
        isinstance(buffer.elem_offset, tir.IntImm) and int(buffer.elem_offset) == 0 and
        all(isinstance(dim, tir.IntImm) and int(dim) > 0 for dim in buffer.shape),
        "requires a compact matching GM allocation")
    handles = [h for h, view in function.buffer_map.items() if view.data.same_as(buffer.data)]
    require(
        len(handles) == 1 and function.buffer_map[handles[0]].same_as(buffer),
        "requires one unambiguous parameter view")
    handle = handles[0]
    positions = [i for i, param in enumerate(function.params) if param.same_as(handle)]
    indices = function.attrs.get("auto_gm_indices", ()) if function.attrs is not None else ()
    require(
        len(positions) == 1 and positions[0] in {int(i) for i in indices},
        "only generated scratch GM parameters are eligible")
    source, destination = call.args[1:3]
    for pointer, role, expected_dtype in ((source, 1, source_dtype), (destination, 2, dtype)):
        require(
            isinstance(pointer, tir.Call) and pointer.op.name == "tir.tvm_access_ptr" and
            len(pointer.args) == 5 and str(pointer.args[0].dtype) == expected_dtype and
            isinstance(pointer.args[1], tir.Var) and isinstance(pointer.args[4], tir.IntImm) and
            int(pointer.args[4]) == role, "pointer dtype or read/write role is inconsistent")
    require(destination.args[1].same_as(buffer.data), "destination is not the parameter storage")

    ancestors, occurrences = [], []
    path_types = (tir.AttrStmt, tir.IfThenElse, tir.BlockRealize, tir.Block, tir.Allocate)

    def before(node):
        if isinstance(node, path_types):
            ancestors.append(node)
        if node.same_as(call):
            occurrences.append(tuple(ancestors))
        return None

    def after(node):
        if isinstance(node, path_types):
            ancestors.pop()
        return None

    tir.stmt_functor.ir_transform(function.body, before, after)
    require(len(occurrences) == 1, "writer must have one physical call identity")
    path = occurrences[0]
    require(not any(isinstance(node, tir.Allocate) for node in path),
            "lowered Allocate needs a separate dominating allocation proof")
    source_buffers = [
        view for node in path if isinstance(node, tir.Block) for view in node.alloc_buffers
        if view.data.same_as(source.args[1])
    ]
    require(not any(node.match_buffers for node in path if isinstance(node, tir.Block)),
            "match-buffer aliases require a separate physical view proof")
    require(
        not any(
            isinstance(node, tir.IfThenElse) or
            (isinstance(node, tir.BlockRealize) and
             (not isinstance(node.predicate, tir.IntImm) or not int(node.predicate)))
            for node in path), "guarded writers need an active-epoch coverage proof")
    scopes = [
        node.value
        for node in path
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope"
    ]
    require(
        bool(scopes) and all(
            isinstance(value, tir.IntImm) and int(value) == (0 if core == "C" else 1)
            for value in scopes), "writer has no consistent dominating core scope")
    axes = [
        node for node in path if isinstance(node, tir.AttrStmt) and
        node.attr_key == "thread_extent" and isinstance(node.node, tir.IterVar)
    ]
    require(
        len(axes) == 2 and
        all(node.node.thread_tag in ("blockIdx.x", "threadIdx.x") for node in axes),
        "unrecognized participant axis")
    group_axes = [node for node in axes if node.node.thread_tag.startswith("blockIdx.")]
    vector_axes = [node for node in axes if node.node.thread_tag == "threadIdx.x"]
    require(
        len(group_axes) == 1 and group_axes[0].node.thread_tag == "blockIdx.x" and
        isinstance(group_axes[0].value, tir.IntImm) and int(group_axes[0].value) > 0,
        "requires one dominating static group axis")
    require(
        len(vector_axes) == 1 and isinstance(vector_axes[0].value, tir.IntImm) and
        int(vector_axes[0].value) == 2, "requires two dominating Vector participants")
    group, vector = group_axes[0], vector_axes[0]
    require(not group.node.var.same_as(vector.node.var), "group and participant axes alias")
    group_count, pitch = int(group.value), rows * cols
    capacity = 1
    for dim in buffer.shape:
        capacity *= int(dim)
    require(capacity == group_count * pitch, "allocation capacity disagrees with group partitions")
    analyzer = arith.Analyzer()
    analyzer.bind(group.node.var, ir.Range(0, group.value))
    analyzer.bind(vector.node.var, ir.Range(0, vector.value))

    def check_address_arithmetic(expression):
        # Proving an algebraic equality does not prove the native integer
        # intermediates can represent it. Check each node before accepting
        # the cid/vid affine partition formula, including narrowing casts.
        nodes = []
        tir.stmt_functor.post_order_visit(
            expression, lambda node: nodes.append(node) if isinstance(node, tir.PrimExpr) else None)
        for node in nodes:
            dtype = DataType(node.dtype)
            require(
                str(dtype).startswith(("int", "uint")) and dtype.lanes == 1,
                "address arithmetic requires scalar integers")
            signed = str(dtype).startswith("int")
            low = -(1 << (dtype.bits - 1)) if signed else 0
            high = (1 << (dtype.bits - (1 if signed else 0))) - 1
            bound = analyzer.const_int_bound(node)
            require(low <= bound.min_value <= bound.max_value <= high,
                    "address arithmetic may overflow its native integer type")
            if isinstance(node, tir.Cast):
                # TVM intersects a Cast's bound with the target dtype range;
                # that clipped result is not evidence the original value fits.
                original = analyzer.const_int_bound(node.value)
                require(low <= original.min_value <= original.max_value <= high,
                        "address cast may overflow its destination integer type")

    for expression in (group.node.var, vector.node.var, source.args[2], destination.args[2]):
        check_address_arithmetic(expression)
    dimensions = [analyzer.simplify(arg) for arg in call.args[3:6]]
    require(all(isinstance(arg, tir.IntImm) for arg in dimensions), "dynamic DMA geometry")
    stride, valid_rows, valid_cols = map(int, dimensions)
    if core == "C":
        valid_rows, valid_cols = valid_rows or rows, valid_cols or cols
    require((stride, valid_rows, valid_cols) == (cols, shard_rows, cols),
            "partial or noncontiguous DMA does not cover its partition")
    expected = group.node.var * pitch
    if core == "V":
        expected += vector.node.var * shard_rows * cols
    require(
        analyzer.can_prove_equal(destination.args[2], expected),
        "destination offset is not its exact group/participant partition")
    require(len(source_buffers) == 1, "requires one known source allocation")
    source_buffer = source_buffers[0]
    require(
        str(source_buffer.dtype) == source_dtype and not source_buffer.strides and
        isinstance(source_buffer.elem_offset, tir.IntImm) and
        int(source_buffer.elem_offset) == 0 and
        source_buffer.scope() == ("wmma.accumulator" if core == "C" else "shared.ub") and
        all(isinstance(dim, tir.IntImm) and int(dim) > 0 for dim in source_buffer.shape),
        "source physical allocation is unsupported")
    source_capacity = 1
    for dim in source_buffer.shape:
        source_capacity *= int(dim)
    require(
        analyzer.can_prove(source.args[2] >= 0) and
        analyzer.can_prove(source.args[2] + shard_rows * cols <= source_capacity),
        "native read exceeds the physical source allocation")
    if core == "C":
        # Until a sub-tile fractal proof is available, accept only one whole
        # aligned L0C tile. Logical capacity alone cannot prove native span.
        require(
            rows % 16 == cols % 16 == 0 and
            tuple(map(int, source_buffer.shape)) in ((rows, cols), (rows * cols,)) and
            analyzer.can_prove_equal(source.args[2], 0),
            "FIX requires a whole aligned L0C source tile")
    else:
        require(cols * (DataType(source_dtype).bits // 8) % 32 == 0,
                "MTE3 contiguous local row pitch must be 32-byte aligned")
        require(
            analyzer.can_prove_equal((source.args[2] * (DataType(source_dtype).bits // 8)) % 32, 0),
            "MTE3 source base must be 32-byte aligned")
    width = DataType(dtype).bits // 8
    shard_bytes = shard_rows * cols * width
    return WorkspaceWritePartition(
        buffer.data, handle, call, destination, group.node.var,
        vector.node.var if core == "V" else None, group_count, pitch, capacity, pitch * width,
        tuple((i * shard_bytes, (i + 1) * shard_bytes) for i in range(participants)))
