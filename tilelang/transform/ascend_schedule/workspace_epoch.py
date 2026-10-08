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


@dataclass(frozen=True, eq=False)
class WorkspaceReadPartition:
    writer: WorkspaceWritePartition
    call: object
    pointer: object
    owner: object
    participant_intervals: tuple


@dataclass(frozen=True, eq=False)
class WorkspaceEpochPartition:
    writer: WorkspaceWritePartition
    readers: tuple
    owner: object


def _checked_address_arithmetic(expression, analyzer):
    from tvm import DataType, arith, tir
    nodes = []
    tir.stmt_functor.post_order_visit(
        expression, lambda node: nodes.append(node) if isinstance(node, tir.PrimExpr) else None)
    for node in nodes:
        dtype = DataType(node.dtype)
        if not str(dtype).startswith(("int", "uint")) or dtype.lanes != 1:
            raise ValueError("workspace epoch: address arithmetic requires scalar integers")
        signed = str(dtype).startswith("int")
        low = -(1 << (dtype.bits - 1)) if signed else 0
        high = (1 << (dtype.bits - (1 if signed else 0))) - 1
        bound = analyzer.const_int_bound(node.value if isinstance(node, tir.Cast) else node)
        if bound.min_value == arith.ConstIntBound.NEG_INF or bound.max_value == arith.ConstIntBound.POS_INF:
            raise ValueError(
                "workspace epoch: address arithmetic has an unbounded or overflowing range")
        # Cast bounds in TVM are clipped to the target range. Check the
        # preconversion value instead; child nodes are checked independently.
        if not low <= bound.min_value <= bound.max_value <= high:
            raise ValueError(
                "workspace epoch: address arithmetic may overflow its native integer type")


def _call_path(function, call):
    from tvm import tir
    stack, found = [], []
    types = (tir.AttrStmt, tir.IfThenElse, tir.BlockRealize, tir.Block, tir.Allocate, tir.For,
             tir.Evaluate)

    def before(node):
        if isinstance(node, types):
            stack.append(node)
        if node.same_as(call):
            found.append(tuple(stack))
        return None

    def after(node):
        if isinstance(node, types):
            stack.pop()
        return None

    tir.stmt_functor.ir_transform(function.body, before, after)
    if len(found) != 1:
        raise ValueError("workspace epoch: native call must have one physical identity")
    evaluations = [node for node in found[0] if isinstance(node, tir.Evaluate)]
    if len(evaluations) != 1 or not evaluations[0].value.same_as(call):
        raise ValueError("workspace epoch: native endpoint must be a direct Evaluate statement")
    if any(isinstance(node, tir.Block) and node.init is not None for node in found[0]):
        raise ValueError("workspace epoch: block initialization needs a separate execution proof")
    return found[0]


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
            require(fields[5].strip() in ("true", "false", "0", "1"), "unknown enRelu template")
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

    path = _call_path(function, call)
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
        node for node in path if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent"
    ]
    require(
        len(axes) == 2 and all(
            isinstance(node.node, tir.IterVar) and node.node.thread_tag in ("blockIdx.x",
                                                                            "threadIdx.x")
            for node in axes), "unrecognized participant axis")
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

    for expression in (group.node.var, vector.node.var, source.args[2], destination.args[2]):
        _checked_address_arithmetic(expression, analyzer)
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


def certify_workspace_epoch_read(function, writer, call, core):
    """Prove a complete read of the same group's partition in the same owner.

    Revalidates the writer in the current function; certificates cannot silently
    survive a changed writer or participant context. This proves geometry and
    loop identity only, not uniform trip counts, protocol order or alias safety.
    """
    from tvm import DataType, arith, ir, tir

    def require(condition, message):
        if not condition:
            raise ValueError("workspace epoch: " + message)

    require(
        isinstance(function, tir.PrimFunc) and isinstance(writer, WorkspaceWritePartition),
        "requires a function and typed writer certificate")
    buffers = [
        view for handle, view in function.buffer_map.items() if handle.same_as(writer.handle)
    ]
    require(len(buffers) == 1, "writer parameter is absent from this function")
    writer = certify_workspace_epoch_write(function, buffers[0], writer.call,
                                           "C" if writer.participant_var is None else "V")
    require(
        core in ("C", "V") and isinstance(call, tir.Call) and
        getattr(call.op, "name", None) == "tir.call_extern" and call.args and
        isinstance(call.args[0], tir.StringImm), "requires an actual native reader")
    name, sep, template = call.args[0].value.removeprefix("tl::ascend::").partition("<")
    fields = template.removesuffix(">").split(",")
    require(
        bool(sep) and call.args[0].value.endswith(">") and len(fields) == 3 and
        name == ("copy_gm_to_l1" if core == "C" else "copy_gm_to_ub") and
        len(call.args) == (6 if core == "C" else 7), "unrecognized reader ABI")
    try:
        dtype = {"half": "float16", "float": "float32"}[fields[0].strip()]
        first, second = int(fields[1]), int(fields[2])
    except (KeyError, ValueError) as error:
        raise ValueError("workspace epoch: unknown reader template geometry") from error
    rows, cols = (first, second) if core == "C" else (second, first)
    participants = 1 if core == "C" else 2
    require(
        rows > 0 and cols > 0 and rows * cols * participants == writer.group_pitch_elements and
        dtype == str(buffers[0].dtype), "reader tile disagrees with writer partition")
    source, destination = call.args[1:3]
    for pointer, role in ((source, 1), (destination, 2)):
        require(
            isinstance(pointer, tir.Call) and
            getattr(pointer.op, "name", None) == "tir.tvm_access_ptr" and len(pointer.args) == 5 and
            str(pointer.args[0].dtype) == dtype and isinstance(pointer.args[1], tir.Var) and
            isinstance(pointer.args[4], tir.IntImm) and int(pointer.args[4]) == role,
            "reader pointer dtype/role is inconsistent")
    require(source.args[1].same_as(writer.storage), "reader uses a different workspace")
    path, write_path = _call_path(function, call), _call_path(function, writer.call)
    require(
        not any(
            isinstance(node, (tir.IfThenElse, tir.Allocate)) or
            (isinstance(node, tir.BlockRealize) and
             (not isinstance(node.predicate, tir.IntImm) or not int(node.predicate))) or
            (isinstance(node, tir.Block) and node.match_buffers)
            for node in path), "guarded or aliased reader needs a separate active-epoch proof")
    owners = [node for node in path if isinstance(node, tir.For)]
    write_owners = [node for node in write_path if isinstance(node, tir.For)]
    require(
        bool(owners) and bool(write_owners) and owners[-1].same_as(write_owners[-1]) and
        owners[-1].kind == tir.ForKind.SERIAL, "reader and writer need the same serial owner")
    scopes = [
        node.value
        for node in path
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope"
    ]
    require(
        bool(scopes) and all(
            isinstance(value, tir.IntImm) and int(value) == (0 if core == "C" else 1)
            for value in scopes), "reader core scope is inconsistent")

    def axes_in(ancestors):
        return [
            node for node in ancestors
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent"
        ]

    axes, write_axes = axes_in(path), axes_in(write_path)
    require(
        len(axes) == len(write_axes) == 2 and all(
            isinstance(axis.node, tir.IterVar) and isinstance(axis.value, tir.IntImm) and any(
                isinstance(old.node, tir.IterVar) and axis.node.var.same_as(old.node.var) and
                axis.node.thread_tag == old.node.thread_tag and int(axis.value) == int(old.value)
                for old in write_axes)
            for axis in axes), "reader participant context differs from writer")
    analyzer = arith.Analyzer()
    for axis in axes:
        analyzer.bind(axis.node.var, ir.Range(0, axis.value))
    for expression in (source.args[2], destination.args[2]):
        _checked_address_arithmetic(expression, analyzer)
    vector = next(axis.node.var for axis in axes if axis.node.thread_tag == "threadIdx.x")
    expected = writer.group_var * writer.group_pitch_elements
    if core == "V":
        expected += vector * rows * cols
    require(
        analyzer.can_prove_equal(source.args[2], expected),
        "reader accesses another group or shard")
    dims = [analyzer.simplify(arg) for arg in call.args[3:6]]
    require(all(isinstance(arg, tir.IntImm) for arg in dims), "dynamic reader DMA geometry")
    stride, valid_rows, valid_cols = map(int, dims)
    if core == "C":
        valid_rows, valid_cols = valid_rows or rows, valid_cols or cols
    require((stride, valid_rows, valid_cols) == (cols, rows, cols), "partial or strided reader")
    if core == "V":
        require(
            isinstance(call.args[6], (tir.IntImm, tir.FloatImm)) and float(call.args[6].value) == 0,
            "reader requires the generated zero-pad ABI")
    allocations = [
        view for node in path if isinstance(node, tir.Block) for view in node.alloc_buffers
        if view.data.same_as(destination.args[1])
    ]
    require(len(allocations) == 1, "reader needs one dominating destination allocation")
    local = allocations[0]
    require(
        str(local.dtype) == dtype and
        local.scope() == ("shared.l1" if core == "C" else "shared.ub") and not local.strides and
        isinstance(local.elem_offset, tir.IntImm) and int(local.elem_offset) == 0 and
        all(isinstance(dim, tir.IntImm) and int(dim) > 0 for dim in local.shape),
        "reader destination physical allocation is unsupported")
    capacity = 1
    for dim in local.shape:
        capacity *= int(dim)
    width = DataType(dtype).bits // 8
    if core == "C":
        require(
            rows % 16 == cols % 16 == 0 and
            tuple(map(int, local.shape)) in ((rows, cols), (rows * cols,)) and
            analyzer.can_prove_equal(destination.args[2], 0),
            "reader needs a whole aligned L1 tile")
    else:
        require(
            cols * width % 32 == 0 and analyzer.can_prove(destination.args[2] >= 0) and
            analyzer.can_prove(destination.args[2] + rows * cols <= capacity) and
            analyzer.can_prove_equal((destination.args[2] * width) % 32, 0),
            "reader destination span or local alignment is unproven")
    shard_bytes = rows * cols * width
    return WorkspaceReadPartition(
        writer, call, source, owners[-1],
        tuple((i * shard_bytes, (i + 1) * shard_bytes) for i in range(participants)))


def certify_workspace_epoch(function, buffer):
    """Audit every use of one generated GM tile before admitting epoch coverage.

    Requires one complete writer before all same-owner readers. Pointer escapes,
    direct tensor loads/stores, unknown helpers and multiple writers fail closed.
    Uniform execution clocks, slot counts and synchronization remain separate
    obligations; this certificate does not set Access.full_overwrite.
    """
    from tvm import tir

    def require(condition, message):
        if not condition:
            raise ValueError("workspace epoch: " + message)

    require(
        isinstance(function, tir.PrimFunc) and isinstance(buffer, tir.Buffer),
        "requires a function and workspace buffer")
    handles = [
        handle for handle, view in function.buffer_map.items() if view.data.same_as(buffer.data)
    ]
    require(len(handles) == 1, "workspace needs one parameter handle")
    handle = handles[0]
    stack, endpoints = [], []

    def before(node):
        if isinstance(node, tir.Var) and node.same_as(handle) and not handle.same_as(buffer.data):
            raise ValueError(
                "workspace epoch: parameter handle escapes its certified workspace view")
        if isinstance(node,
                      (tir.BufferLoad, tir.BufferStore)) and node.buffer.data.same_as(buffer.data):
            raise ValueError(
                "workspace epoch: direct tensor access is not a certified DMA endpoint")
        if isinstance(node, tir.Var) and node.same_as(buffer.data):
            require(
                bool(stack) and getattr(stack[-1].op, "name", None) == "tir.tvm_access_ptr" and
                len(stack[-1].args) == 5 and stack[-1].args[1].same_as(node),
                "workspace data pointer escapes its access_ptr")
        if isinstance(node, tir.Call):
            if getattr(node.op, "name", None) == "tir.tvm_access_ptr" and len(node.args) == 5 and \
                    isinstance(node.args[1], tir.Var) and node.args[1].same_as(buffer.data):
                require(
                    bool(stack) and getattr(stack[-1].op, "name", None) == "tir.call_extern" and
                    any(arg.same_as(node) for arg in stack[-1].args),
                    "workspace pointer escapes a direct native endpoint")
                endpoints.append((stack[-1], node))
            stack.append(node)
        return None

    def after(node):
        if isinstance(node, tir.Call):
            stack.pop()
        return None

    tir.stmt_functor.ir_transform(function.body, before, after)
    require(bool(endpoints), "workspace has no certified endpoints")
    writes, reads = [], []
    for position, (call, pointer) in enumerate(endpoints):
        require(isinstance(pointer.args[4], tir.IntImm), "dynamic pointer access role")
        role = int(pointer.args[4])
        require(role in (1, 2), "read-modify-write workspace access is unsupported")
        require(isinstance(call.args[0], tir.StringImm), "unnamed native endpoint")
        name = call.args[0].value.removeprefix("tl::ascend::").split("<", 1)[0]
        if role == 2:
            require(name in ("copy_l0c_to_gm", "copy_ub_to_gm"), "unknown workspace writer")
            writes.append((position, call, "C" if name == "copy_l0c_to_gm" else "V"))
        else:
            require(name in ("copy_gm_to_l1", "copy_gm_to_ub"), "unknown workspace reader")
            reads.append((position, call, "C" if name == "copy_gm_to_l1" else "V"))
    require(len(writes) == 1 and bool(reads), "requires exactly one writer and at least one reader")
    position, call, core = writes[0]
    writer = certify_workspace_epoch_write(function, buffer, call, core)
    require(
        all(index > position for index, _, _ in reads), "read precedes this epoch's complete write")
    certificates = tuple(
        certify_workspace_epoch_read(function, writer, call, core) for _, call, core in reads)
    owner = certificates[0].owner
    require(
        all(item.owner.same_as(owner) for item in certificates),
        "workspace endpoints have different owners")
    return WorkspaceEpochPartition(writer, certificates, owner)
