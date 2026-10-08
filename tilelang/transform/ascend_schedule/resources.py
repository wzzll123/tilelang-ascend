# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2/A3 *compiler allocation budgets*, not A5 hardware capacities.

These conservative byte budgets match the constants in
src/transform/ascend_memory_planning.cc and the corresponding generated TBuf
reservations. In particular UB reserves 256 bytes; this differs deliberately
from the simulator's full address-space capacity for manually placed buffers.
Every budget is per execution-core instance, never a sum across the two AIVs.
"""

from types import MappingProxyType

_BUDGETS = MappingProxyType({
    "UB": 196352,
    "L1": 524032,
    "L0A": 65536,
    "L0B": 65536,
    "L0C": 131072,
    "BT": 1024,
})


def allocation_budgets(platform):
    """Return an independent capacity map for a validated A2/A3 target."""
    if platform not in ("A2", "A3"):
        raise ValueError("automatic scheduling allocation budgets require A2 or A3")
    return dict(_BUDGETS)
