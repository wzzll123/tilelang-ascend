# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative slot envelopes proved from generated workspace copy TIR."""

from dataclasses import replace

VERSION_ATTR = "tl.ascend.iteration_versions"


def workspace_copy_span(parent, pointer, analyzer):
    from tvm import tir
    # Native workspace ptr.extent advertises a whole core tile even for a
    # half-tile Vector shard. Prove the helper's actual row-major span;
    # never reinterpret that broad metadata as bytes the DMA must touch.
    if parent is None or getattr(parent.op, "name", None) != "tir.call_extern":
        return None
    tag = parent.args[0].value if isinstance(parent.args[0], tir.StringImm) else ""
    name = tag.split("<", 1)[0].removeprefix("tl::ascend::")
    sides = {"copy_ub_to_gm": 2, "copy_gm_to_ub": 1, "copy_gm_to_l1": 1, "copy_l0c_to_gm": 2}
    if name not in sides or len(parent.args) < 6 or not pointer.same_as(parent.args[sides[name]]):
        return None
    try:
        fields = tag.split("<", 1)[1].removesuffix(">").split(",")
        if name == "copy_l0c_to_gm":
            if fields[2].strip() != "layout::RowMajor":
                return None
            rows, cols = int(fields[3]), int(fields[4])
        elif name == "copy_gm_to_l1":
            rows, cols = int(fields[1]), int(fields[2])
        else:
            cols, rows = int(fields[1]), int(fields[2])
    except (ValueError, IndexError):
        return None
    dims = [analyzer.simplify(arg) for arg in parent.args[3:6]]
    if not all(isinstance(arg, tir.IntImm) for arg in dims):
        return None
    stride, valid_m, valid_n = map(int, dims)
    if name in ("copy_gm_to_l1", "copy_l0c_to_gm"):
        valid_m, valid_n = valid_m or rows, valid_n or cols
    if not (0 <= valid_m <= rows and 0 <= valid_n <= cols and valid_n <= stride):
        return None
    return 0 if valid_m == 0 or valid_n == 0 else (valid_m - 1) * stride + valid_n


def certify_workspace_versions(function, extraction):
    """Prove conservative whole-slot envelopes without inventing execution edges.

    A certificate records the physical pitch/count and owner variable introduced
    by materialization. Each copy must independently prove the same slot formula
    and a bounded row-major span. Unknown operations, clocks or offsets retain
    their original unknown footprints. This does not prove full overwrite,
    synchronization, alias safety or that a guard advances an owner counter.
    """
    from tvm import DataType, arith, ir, tir

    records = function.attrs.get(VERSION_ATTR, {}) if function.attrs is not None else {}
    if not records:
        return extraction, ()
    analyzer = arith.Analyzer()
    loops, diagnostics = [], []

    def bounds(node):
        if isinstance(node, tir.For):
            loops.append(node)
        if (isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and
                isinstance(node.node, tir.IterVar) and isinstance(node.value, tir.IntImm)):
            analyzer.bind(node.node.var, ir.Range.from_min_extent(0, node.value))

    tir.stmt_functor.post_order_visit(function.body, bounds)
    certificates = []
    for storage, info in records.items():
        if len(info) != 4 or not isinstance(info[0], tir.IntImm) or not isinstance(
                info[1], tir.IntImm):
            diagnostics.append("invalid iteration version certificate")
            continue
        count, pitch, iv, minimum = info
        count, pitch = int(count), int(pitch)
        owners = [loop for loop in loops if loop.loop_var.same_as(iv)]
        buffers = [
            buffer for buffer in function.buffer_map.values() if buffer.data.same_as(storage)
        ]
        if count < 2 or pitch <= 0 or len(owners) != 1 or len(buffers) != 1:
            diagnostics.append("workspace certificate needs one owner and parameter view")
            continue
        owner, buffer = owners[0], buffers[0]
        handles = [
            handle for handle, view in function.buffer_map.items() if view.data.same_as(storage)
        ]
        indices = function.attrs.get("auto_gm_indices", ())
        parameter_positions = {
            i for i, param in enumerate(function.params) if any(
                param.same_as(handle) for handle in handles)
        }
        if not parameter_positions or not parameter_positions.issubset({int(i) for i in indices}):
            diagnostics.append("workspace certificate does not name generated scratch parameters")
            continue
        if (owner.kind != tir.ForKind.SERIAL or not analyzer.can_prove_equal(owner.min, minimum) or
                buffer.scope() != "global" or buffer.strides or
                any(not isinstance(dim, tir.IntImm) for dim in buffer.shape)):
            diagnostics.append("workspace certificate clock or allocation is unsupported")
            continue
        capacity = 1
        for dim in buffer.shape:
            capacity *= int(dim)
        if capacity != count * pitch:
            diagnostics.append("workspace version certificate disagrees with allocation capacity")
            continue
        iv_type = DataType(owner.loop_var.dtype)
        minimum_bound = analyzer.const_int_bound(owner.min)
        extent_bound = analyzer.const_int_bound(owner.extent)
        iv_low = -(1 << (iv_type.bits - 1)) if str(iv_type).startswith("int") else 0
        iv_high = (1 << (iv_type.bits - 1)) - 1 if str(iv_type).startswith(
            "int") else (1 << iv_type.bits) - 1
        if (extent_bound.min_value < 0 or minimum_bound.min_value < iv_low or
                minimum_bound.max_value + max(0, extent_bound.max_value - 1) > iv_high or
                minimum_bound.max_value - minimum_bound.min_value +
                max(0, extent_bound.max_value - 1) > iv_high):
            diagnostics.append("workspace owner index/relative clock may overflow")
            continue
        analyzer.bind(owner.loop_var, ir.Range.from_min_extent(owner.min, owner.extent))
        certificates.append((storage, count, pitch, iv, minimum, str(buffer.dtype)))

    tasks = []
    for task in extraction.tasks:
        accesses = []
        node = extraction.task_nodes[task.id]
        call = node.value if isinstance(node, tir.Evaluate) and isinstance(node.value,
                                                                           tir.Call) else None
        for access in task.accesses:
            var = getattr(access.storage, "var", None)
            certificate = next(
                (item for item in certificates if var is not None and item[0].same_as(var)), None)
            if certificate is None:
                accesses.append(access)
                continue
            storage, count, pitch, iv, minimum, dtype = certificate
            pointers = [] if call is None else [
                arg for arg in call.args if isinstance(arg, tir.Call) and
                getattr(arg.op, "name", None) == "tir.tvm_access_ptr" and len(arg.args) == 5 and
                isinstance(arg.args[1], tir.Var) and arg.args[1].same_as(storage)
            ]
            proved = bool(pointers) and task.guard in (None, ())
            details = []
            for pointer in pointers:
                span = workspace_copy_span(call, pointer, analyzer)
                offset_dtype = str(pointer.args[2].dtype)
                if not offset_dtype.startswith(
                    ("int", "uint")) or DataType(offset_dtype).lanes != 1:
                    proved = False
                    details.append("non-integer pointer offset")
                    continue
                slot = tir.floormod(iv - minimum, count)
                base = analyzer.simplify(pointer.args[2] - tir.Cast(offset_dtype, slot) *
                                         tir.IntImm(offset_dtype, pitch))
                bits = DataType(offset_dtype).bits
                maximum = (1 <<
                           (bits - 1)) - 1 if offset_dtype.startswith("int") else (1 << bits) - 1
                proved = proved and (str(pointer.args[0].dtype) == dtype and span is not None and
                                     count * pitch <= maximum and analyzer.can_prove(base >= 0) and
                                     analyzer.can_prove(base + (span or 0) <= pitch))
                details.append(f"span={span}, base={base}, pitch={pitch}")
            if not proved:
                diagnostics.append(f"task {task.id}: workspace slot/physical span unproven; " +
                                   "; ".join(details))
                accesses.append(access)
                continue
            width = DataType(dtype).bits * DataType(dtype).lanes // 8
            accesses.append(
                replace(
                    access,
                    offset=0,
                    extent=pitch * width,
                    intervals=None,
                    iteration_stride=pitch * width,
                    ring_size=count))
        tasks.append(replace(task, accesses=tuple(accesses)))
    return replace(extraction, tasks=tuple(tasks)), tuple(diagnostics)
