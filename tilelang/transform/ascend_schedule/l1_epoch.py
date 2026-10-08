# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Geometry certificates for native FP16 L1 prefetch epochs.

This is an eligibility proof, not a scheduler. Callers must additionally prove
unique ownership, guards and lifetimes. In particular the primary offset-zero
tail clear is not a certificate for a rotated slot: codegen may suppress that
clear at a nonzero base. Admit complete DMA rectangles without relying on it.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class L1EpochWrite:
    elements: int
    byte_footprint: tuple[tuple[int, int], ...]


def _geometry(buffer, rows, cols):
    from tvm import tir
    if (buffer.scope() != "shared.l1" or str(buffer.dtype) != "float16" or buffer.strides or
            len(buffer.shape) not in (1, 2) or
            any(not isinstance(dim, tir.IntImm) or int(dim) <= 0 for dim in buffer.shape) or
            not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset)):
        raise ValueError(f"L1 epoch requires compact FP16 tile geometry: {buffer.scope()}, "
                         f"{buffer.shape}, {buffer.strides}, {buffer.elem_offset}")
    shape = tuple(map(int, buffer.shape))
    if rows <= 0 or cols <= 0 or shape not in ((rows, cols), (rows * cols,)):
        raise ValueError("L1 epoch template geometry does not match allocation")
    # FP16 zN pads each axis to 16. Only exact physical/logical rectangles
    # are admitted; padding holes would need a separately certified clear.
    if rows % 16 or cols % 16:
        raise ValueError("L1 epoch geometry has uninitialized fractal padding")
    return rows, cols


def _pointer(pointer, buffer, mode, analyzer, role):
    from tvm import tir
    if (not isinstance(pointer, tir.Call) or pointer.op.name != "tir.tvm_access_ptr" or
            len(pointer.args) != 5 or not pointer.args[1].same_as(buffer.data) or
            not isinstance(pointer.args[4], tir.IntImm) or int(pointer.args[4]) != mode):
        raise ValueError(f"L1 epoch {role} pointer does not match its allocation/mode")
    if str(pointer.args[0].dtype) != "float16":
        raise ValueError(f"L1 epoch {role} pointer dtype does not match its allocation")
    if not analyzer.can_prove(pointer.args[2] == 0):
        raise ValueError(f"L1 epoch {role} must start at the logical tile base")


def _template(call, pattern):
    from tvm import tir
    if (not isinstance(call, tir.Call) or call.op.name != "tir.call_extern" or not call.args or
            not isinstance(call.args[0], tir.StringImm)):
        raise ValueError("L1 epoch requires a recognized native copy")
    match = re.fullmatch(pattern, str(call.args[0].value).split("::")[-1].replace(" ", ""))
    if match is None:
        raise ValueError("L1 epoch native template geometry is unsupported")
    return match.groups()


def certify_l1_epoch_write(buffer, call, analyzer):
    """Certify full initialization from GM->L1 helper's actual copy geometry.

    TIR ABI is source first (codegen reverses it for the C++ helper). Operands
    3/4/5 are source stride and copied row/column counts. The helper's literal
    zero count means full dimension, not an empty transfer. Pointer extent is
    deliberately NOT used to infer DMA coverage.
    """
    from tvm import tir
    template = _template(call, r"(?:tl::)?copy_gm_to_l1<half,(\d+),(\d+)>")
    rows, cols = _geometry(buffer, *map(int, template))
    if len(call.args) not in (6, 8):
        raise ValueError("L1 epoch native geometry does not match allocation")
    _pointer(call.args[2], buffer, 2, analyzer, "destination")
    for operand, size, label in ((call.args[4], rows, "rows"), (call.args[5], cols, "columns")):
        default = isinstance(operand, tir.IntImm) and int(operand) == 0
        if not default and not analyzer.can_prove(operand == size):
            raise ValueError(f"L1 epoch DMA must write full {label} in every epoch")
    if not analyzer.can_prove(call.args[3] >= cols):
        raise ValueError("L1 epoch source stride is smaller than the copy rectangle")
    return L1EpochWrite(rows * cols, ((0, rows * cols * 2),))


def certify_l1_epoch_read(buffer, call, analyzer):
    """Bound native L1->L0A/B source geometry inside one initialized tile."""
    template = _template(call, r"(?:tl::)?copy_l1_to_l0[ab]<half,(\d+),(\d+),(true|false)>")
    rows, cols = _geometry(buffer, *map(int, template[:2]))
    if len(call.args) != 5:
        raise ValueError("L1 epoch reader geometry does not match allocation")
    _pointer(call.args[1], buffer, 1, analyzer, "source")
    limits = (cols, rows) if template[2] == "true" else (rows, cols)
    for operand, limit in zip(call.args[3:5], limits):
        if not analyzer.can_prove(operand >= 1) or not analyzer.can_prove(operand <= limit):
            raise ValueError("L1 epoch reader geometry escapes the initialized tile")
    # Conservative envelope covers the whole zN tile; partial reads are not
    # reinterpreted as a contiguous row-major byte span.
    return ((0, rows * cols * 2),)
