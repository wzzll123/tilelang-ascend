# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tvm._ffi.registry import register_func
from tvm.ir import make_node


@register_func("tvm.info.mem.local.var")
def mem_info_local_var():
    """Get memory information for local variable memory.

    Returns:
        tvm.ir.make_node: A node containing memory information
    """
    return make_node(
        "MemoryInfo",
        unit_bits=8,
        max_num_bits=64,
        max_simd_bits=128,
        head_address=None,
    )


@register_func("tvm.info.mem.shared.bt")
def mem_info_shared_bt():
    """BT (bias table, C2) memory info: 1KB on A2/A3."""
    return make_node(
        "MemoryInfo",
        unit_bits=8,
        max_num_bits=1024 * 8,
        max_simd_bits=512,
        head_address=None,
    )
