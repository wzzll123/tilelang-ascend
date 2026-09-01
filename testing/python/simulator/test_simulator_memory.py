# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the A2/A3 simulator memory and hazard model."""

import pytest

from tilelang.simulator import (
    A2_A3_LOCAL_CAPACITIES,
    AddressRange,
    BufferSpec,
    CoreProgram,
    KernelProgram,
    MemoryBoundsError,
    MemoryCapacityError,
    MemoryHazardError,
    MemoryRuntime,
    MemoryScope,
    SimulatorHazardWarning,
    UninitializedMemoryError,
    dtype_size_bytes,
)


def test_program_memory_shares_gm_and_workspace_but_isolates_local_scopes() -> None:
    program = KernelProgram(
        "memory-layout",
        "A2",
        (CoreProgram(0), CoreProgram(1)),
        buffers=(
            BufferSpec("x", MemoryScope.GM, (8,), "uint8"),
            BufferSpec("scratch", MemoryScope.WORKSPACE, (8,), "uint8"),
            BufferSpec("tile", MemoryScope.UB, (8,), "uint8"),
        ),
    )
    memory = MemoryRuntime.from_program(program)

    gm_from_core_0 = memory.get("x", scope=MemoryScope.GM, core_id=0)
    gm_from_core_1 = memory.get("x", scope=MemoryScope.GM, core_id=1)
    workspace_0 = memory.get("scratch", scope=MemoryScope.WORKSPACE, core_id=0)
    workspace_1 = memory.get("scratch", scope=MemoryScope.WORKSPACE, core_id=1)
    ub_0 = memory.get("tile", scope=MemoryScope.UB, core_id=0)
    ub_1 = memory.get("tile", scope=MemoryScope.UB, core_id=1)

    assert gm_from_core_0 is gm_from_core_1
    assert workspace_0 is workspace_1
    assert ub_0 is not ub_1
    ub_0.write(ub_0.view(), bytes(range(8)))
    assert ub_0.read(ub_0.view()) == bytes(range(8))
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        ub_1.read(ub_1.view())


@pytest.mark.parametrize(
    ("tir_scope", "scope"),
    [
        ("wmma.matrix_a", MemoryScope.L0A),
        ("wmma.matrix_b", MemoryScope.L0B),
        ("wmma.accumulator", MemoryScope.L0C),
    ],
)
def test_lowered_wmma_scopes_map_to_a2_a3_local_memory(
    tir_scope: str, scope: MemoryScope
) -> None:
    assert MemoryScope.parse(tir_scope) is scope


def test_poison_tracking_is_independent_of_byte_value_and_policy() -> None:
    spec = BufferSpec("out", MemoryScope.GM, (4,), "uint8")

    disabled = MemoryRuntime((0,), hazard_check="off").allocate(spec)
    assert disabled.read(disabled.view()) == b"\xff" * 4
    assert disabled.initialized() is False
    disabled.write(disabled.view(), b"\xff" * 4)
    assert disabled.initialized() is True
    assert disabled.read(disabled.view()) == b"\xff" * 4

    warning_runtime = MemoryRuntime((0,), hazard_check="warn")
    warning_buffer = warning_runtime.allocate(spec)
    with pytest.warns(SimulatorHazardWarning, match="read-before-write"):
        assert warning_buffer.read(warning_buffer.view()) == b"\xff" * 4
    assert warning_runtime.reporter.diagnostics[0].kind == "read-before-write"


def test_partial_write_and_absolute_address_range() -> None:
    allocation = MemoryRuntime((0,)).allocate(
        BufferSpec("gm", MemoryScope.GM, (16,), "uint8"), address=128
    )
    middle = AddressRange(132, 136)

    allocation.write(middle, b"abcd")
    assert allocation.read(middle) == b"abcd"
    assert allocation.initialized(middle)
    assert not allocation.initialized()
    with pytest.raises(MemoryBoundsError, match="escapes buffer"):
        allocation.read(AddressRange(127, 129))


def test_views_check_physical_extent_for_contiguous_and_strided_layouts() -> None:
    allocation = MemoryRuntime((0,), hazard_check="off").allocate(
        BufferSpec("matrix", MemoryScope.GM, (4, 4), "float16")
    )

    view = allocation.view(byte_offset=2, shape=(2, 2), strides_bytes=(8, 2))
    assert view.byte_range == AddressRange(2, 14)
    assert view.nbytes == 8
    assert view.address_ranges == (AddressRange(2, 6), AddressRange(10, 14))
    allocation.write(view, b"abcdefgh")
    assert allocation.read(view) == b"abcdefgh"
    assert allocation.read(AddressRange(6, 10)) == b"\xff" * 4
    with pytest.raises(MemoryBoundsError, match="reaches byte"):
        allocation.view(byte_offset=24, shape=(2, 2), dtype="float16")
    with pytest.raises(MemoryBoundsError, match="same rank"):
        allocation.view(shape=(2, 2), strides_bytes=(2,))


@pytest.mark.parametrize("scope", list(A2_A3_LOCAL_CAPACITIES))
def test_every_local_scope_enforces_a2_a3_capacity(scope: MemoryScope) -> None:
    capacity = A2_A3_LOCAL_CAPACITIES[scope]
    memory = MemoryRuntime((0,))
    memory.allocate(BufferSpec("fits", scope, (capacity,), "uint8"), core_id=0)

    with pytest.raises(MemoryCapacityError, match="capacity exceeded"):
        MemoryRuntime((0,)).allocate(
            BufferSpec("too-large", scope, (capacity + 1,), "uint8"), core_id=0
        )


def test_explicit_overlapping_addresses_follow_hazard_policy() -> None:
    first = BufferSpec("first", MemoryScope.UB, (16,), "uint8")
    second = BufferSpec("second", MemoryScope.UB, (8,), "uint8")

    strict = MemoryRuntime((0,), hazard_check="error")
    strict.allocate(first, core_id=0, address=0)
    with pytest.raises(MemoryHazardError, match="overlaps"):
        strict.allocate(second, core_id=0, address=4)

    permissive = MemoryRuntime((0,), hazard_check="off")
    permissive.allocate(first, core_id=0, address=0)
    permissive.allocate(second, core_id=0, address=4)
    assert permissive.reporter.diagnostics == ()


def test_non_overlapping_lifetimes_share_planned_physical_storage() -> None:
    first = BufferSpec(
        "first", MemoryScope.UB, (8,), "uint8", address=16, lifetime=(0, 4)
    )
    second = BufferSpec(
        "second", MemoryScope.UB, (4,), "uint8", address=20, lifetime=(4, 8)
    )
    memory = MemoryRuntime((0,))
    first_allocation = memory.allocate(first, core_id=0)
    second_allocation = memory.allocate(second, core_id=0)

    first_allocation.write(first_allocation.view(), b"abcdefgh")
    assert second_allocation.read(second_allocation.view()) == b"efgh"

    second_allocation.write(second_allocation.view(), b"WXYZ")
    assert first_allocation.read(first_allocation.view()) == b"abcdWXYZ"


def test_live_alias_requires_an_explicit_alias_relationship() -> None:
    base = BufferSpec("base", MemoryScope.UB, (8,), "uint8", address=0)
    alias = BufferSpec(
        "alias",
        MemoryScope.UB,
        (4,),
        "uint8",
        address=4,
        metadata={"alias_of": "base"},
    )
    memory = MemoryRuntime((0,))
    base_allocation = memory.allocate(base, core_id=0)
    alias_allocation = memory.allocate(alias, core_id=0)

    base_allocation.write(base_allocation.view(), b"12345678")
    assert alias_allocation.read(alias_allocation.view()) == b"5678"


def test_program_uses_storage_rewrite_address() -> None:
    program = KernelProgram(
        "planned-address",
        "A2",
        (CoreProgram(0),),
        buffers=(BufferSpec("tile", MemoryScope.UB, (8,), "uint8", address=64),),
    )

    allocation = MemoryRuntime.from_program(program).get(
        "tile", scope=MemoryScope.UB, core_id=0
    )
    assert allocation.address == 64


@pytest.mark.parametrize(
    ("dtype", "size"),
    [("bool", 1), ("int8", 1), ("float16", 2), ("bfloat16", 2), ("float32x4", 16)],
)
def test_dtype_byte_width(dtype: str, size: int) -> None:
    assert dtype_size_bytes(dtype) == size
