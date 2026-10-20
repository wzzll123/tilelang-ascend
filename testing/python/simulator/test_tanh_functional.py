# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tanh's source is immutable and workspace is an unspecified clobber."""

import numpy as np
import pytest

from tilelang.simulator import (
    BufferRegion,
    BufferSpec,
    CoreProgram,
    FunctionalSimulator,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    SimulatorConfig,
    Task,
    classify_operation,
)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("sync_only", [False, True])
def test_tanh_preserves_source_and_does_not_read_scratch(dtype, sync_only):
    source = BufferRegion("source", MemoryScope.UB, (64,), dtype)
    destination = BufferRegion("destination", MemoryScope.UB, (64,), dtype)
    scratch = BufferRegion("scratch", MemoryScope.UB, (4096,), "uint8")
    task = Task(
        "tanh",
        "tanh",
        0,
        Lane.VECTOR_0,
        Pipe.VECTOR,
        1,
        metadata={"src": source, "dst": destination, "scratch": scratch, "scratch_clobber": True},
    )
    program = KernelProgram(
        "tanh",
        "A2",
        (CoreProgram(0, (task,)),),
        buffers=tuple(BufferSpec(region.buffer, region.scope, region.shape, region.dtype) for region in (source, destination, scratch)),
    )
    simulator = FunctionalSimulator(program, SimulatorConfig(sync_only=sync_only))
    values = np.linspace(-4, 4, 64).astype(dtype)
    simulator.write(source, values)
    simulator.run()
    assert not simulator._resolve(scratch, 0).allocation.initialized(simulator._resolve(scratch, 0))
    assert simulator._resolve(destination, 0).allocation.initialized(simulator._resolve(destination, 0))
    if not sync_only:
        np.testing.assert_array_equal(simulator.read(source), values)
        np.testing.assert_array_equal(simulator.read(destination), np.tanh(values))
    assert classify_operation("tl.ascend_tanh", Lane.VECTOR_0)[:2] == (Lane.VECTOR_0, Pipe.VECTOR)
