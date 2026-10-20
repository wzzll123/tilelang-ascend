# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Versioned FIA vector helpers: functional semantics and opaque fail-closed."""

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
    UnsupportedSimOpError,
)
from tilelang.simulator.fia_helpers import helper_metadata, validate_contract
from tilelang.simulator.errors import UninitializedMemoryError


def _regions():
    return {
        name: BufferRegion(name, MemoryScope.UB, shape, dtype)
        for name, shape, dtype in (
            ("score", (16, 512), "float32"),
            ("maxv", (64,), "float32"),
            ("sums", (192,), "float32"),
            ("alpha", (192,), "float32"),
            ("denom", (64,), "float32"),
            ("scratch", (49152,), "uint8"),
            ("dst", (64, 128), "float32"),
            ("brcb", (64, 8), "float32"),
            ("col", (512,), "float32"),
        )
    }


def _sim(kind, *, sync_only=False, slot=1, row=16, valid=257):
    regions = _regions()
    metadata = helper_metadata(kind, regions, slot=slot, row=row, valid=valid)
    task = Task("helper", "fia_helper", 0, Lane.VECTOR_0, Pipe.VECTOR, 1, metadata=metadata)
    program = KernelProgram(
        "helper", "A2", (CoreProgram(0, (task,)),), buffers=tuple(BufferSpec(r.buffer, r.scope, r.shape, r.dtype) for r in regions.values())
    )
    return FunctionalSimulator(program, SimulatorConfig(sync_only=sync_only)), regions, metadata


def test_flash_init_does_not_read_state_or_initialize_alpha():
    sim, r, m = _sim("flash_init", sync_only=True)
    sim.write(r["score"], np.zeros((16, 512), np.float32))
    sim.run()
    for dst in m["dst_regions"]:
        view = sim._resolve(dst, 0)
        assert view.allocation.check_initialized(view)
    view = sim._resolve(r["alpha"], 0)
    assert not view.allocation.initialized(view.byte_range)
    view = sim._resolve(r["scratch"], 0)
    assert not view.allocation.initialized(view.byte_range)
    assert len(m["src_regions"]) == 1


def test_flash_update_numerical_state_slices():
    sim, r, m = _sim("flash_update")
    x = np.linspace(-3, 2, 8192, dtype=np.float32).reshape(16, 512)
    old_max = np.full(64, 1.5, np.float32)
    old_sum = np.full(192, 7, np.float32)
    sim.write(r["score"], x)
    sim.write(r["maxv"], old_max)
    sim.write(r["sums"], old_sum)
    sim.run()
    new_max = np.maximum(x.max(axis=1), 1.5)
    p = np.exp(x - new_max[:, None])
    alpha = np.exp(1.5 - new_max)
    np.testing.assert_allclose(sim.read(r["score"]), p, rtol=1e-6)
    np.testing.assert_allclose(sim.read(m["max_out"]), new_max)
    np.testing.assert_allclose(sim.read(m["alpha_out"]), alpha)
    np.testing.assert_allclose(sim.read(m["sum_out"]), p.sum(axis=1) + alpha * 7)
    np.testing.assert_allclose(sim.read(m["old_sum"]), alpha * 7)
    assert m["old_sum"] in m["dst_regions"]


@pytest.mark.parametrize("kind", ["rescale", "normalize"])
def test_row_operation(kind):
    sim, r, _ = _sim(kind)
    sim.write(r["dst"], np.full((64, 128), 6, np.float32))
    sim.write(r["alpha"], np.full(192, 2, np.float32))
    sim.write(r["denom"], np.full(64, 2, np.float32))
    sim.run()
    np.testing.assert_array_equal(sim.read(r["dst"]), 12 if kind == "rescale" else 3)
    np.testing.assert_array_equal(sim.read(r["brcb"]), 2)


def test_tail():
    sim, r, _ = _sim("tail")
    sim.write(r["score"], np.ones((16, 512), np.float32))
    sim.run()
    value = sim.read(r["score"])
    assert np.all(value[:, :257] == 1)
    assert np.isneginf(value[:, 257:]).all()


@pytest.mark.parametrize("kind", ["flash_init", "flash_update", "rescale", "normalize", "tail"])
def test_sync_only_uninitialized_input_propagates_poison(kind):
    sim, _, m = _sim(kind, sync_only=True)
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        sim.run()
    for dst in m["dst_regions"]:
        if dst in m.get("scratch_regions", ()):
            continue
        view = sim._resolve(dst, 0)
        assert not view.allocation.initialized(view.byte_range)


def test_contract_unknown_and_source_mismatch_fail_closed():
    with pytest.raises(UnsupportedSimOpError):
        validate_contract("sim.fia.v2.flash_init", "anything")
    with pytest.raises(UnsupportedSimOpError):
        validate_contract("sim.fia.v1.flash_init", "evil();")
    assert validate_contract("sim.fia.v1.normalize", "fia_aligned::Normalize(acc_o,denom,brcb);") == "normalize"


def _bridge_program(kind="normalize", *, fence=True, mutation="raw"):
    import tvm
    from tilelang.simulator.bridge import build_kernel_program

    t = tvm.tir
    out = t.decl_buffer((64, 128), "float32", name="acc_o", scope="shared.ub")
    rows = t.decl_buffer((64,), "float32", name="denom", scope="shared.ub")
    tmp = t.decl_buffer((64, 8), "float32", name="brcb", scope="shared.ub")
    gm = t.decl_buffer((8192,), "float32", name="gm")
    function = "NormalizeDivMul" if kind == "normalize_divmul" else "Normalize"
    version = 2 if kind == "normalize_divmul" else 1
    opaque = t.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.ascend_src_code"),
        f"fia_aligned::{function}(acc_o,denom,brcb);",
        out.access_ptr("rw"),
        rows.access_ptr("rw"),
        tmp.access_ptr("rw"),
        f"sim.fia.v{version}.{kind}",
    )
    # MTE2 write -> vector read is RAW; vector read -> MTE2 overwrite is WAR.
    target = tmp if mutation == "waw" else rows
    count = 512 if mutation == "waw" else 64
    copy = t.call_extern("handle", "copy_gm_to_ub", gm.data, target.data, count)
    calls = [copy, opaque] if mutation == "raw" else [opaque, copy]
    if fence:
        src, dst = ("mte2", "v") if mutation == "raw" else ("v", "mte2")
        calls[1:1] = [t.call_extern("handle", "set_flag", src, dst, 0), t.call_extern("handle", "wait_flag", src, dst, 0)]
    body = t.SeqStmt([t.Evaluate(c) for c in calls])
    for buffer in (out, rows, tmp):
        body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
    return build_kernel_program(t.PrimFunc([gm.data], body, buffer_map={gm.data: gm}), platform="A2")


@pytest.mark.parametrize("mutation", ["raw", "war", "waw"])
def test_bridge_helper_memory_edges_require_real_fence(mutation):
    from tilelang.simulator.sync import validate_memory_synchronization
    from tilelang.simulator.errors import MemoryHazardError

    safe = _bridge_program(mutation=mutation)
    assert not validate_memory_synchronization(safe)
    broken = _bridge_program(fence=False, mutation=mutation)
    with pytest.raises(MemoryHazardError, match="missing A2/A3 pipe synchronization"):
        validate_memory_synchronization(broken)
    helper = next(t for t in broken.tasks if t.operation == "fia_helper")
    assert helper.pipe is Pipe.VECTOR
    assert helper.metadata["helper_contract"] == "sim.fia.v1.normalize"


def test_bridge_unknown_opaque_contract_rejected():
    with pytest.raises(UnsupportedSimOpError, match="does not match"):
        _bridge_program(kind="unknown")


@pytest.mark.parametrize("update", [False, True])
def test_flash_bridge_explicit_scalars_and_flattened_stride(update):
    import tvm
    from tilelang.simulator.bridge import build_kernel_program

    t = tvm.tir
    sizes = {"score": 8192, "maxv": 64, "sums": 192, "alpha": 192, "denom": 64, "scratch": 49152}
    buffers = [t.decl_buffer((n,), "uint8" if name == "scratch" else "float32", name=name, scope="shared.ub") for name, n in sizes.items()]
    kind = "flash_update" if update else "flash_init"
    source = "fia_aligned::Flash<" + ("true" if update else "false") + ">(score,maxv,sums,alpha,denom,scratch,sm_slot,sm_row);"
    opaque = t.call_intrin(
        "handle", tvm.ir.Op.get("tl.ascend_src_code"), source, *[b.access_ptr("rw") for b in buffers], 0, 0, f"sim.fia.v1.{kind}", 2, 48
    )
    body = t.Evaluate(opaque)
    for buffer in buffers:
        body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
    program = build_kernel_program(t.PrimFunc([], body), platform="A2")
    task = program.tasks[0]
    assert task.metadata["dst"].strides_bytes == (2048, 4)
    assert task.metadata["sum_out"].byte_offset == 176 * 4
    assert task.metadata["old_sum"].byte_offset == 112 * 4
    assert len(task.metadata["src_regions"]) == (3 if update else 1)
    assert task.metadata["helper_timing_provenance"].startswith("uncalibrated")


def test_poison_propagates_when_diagnostics_warn():
    from dataclasses import replace

    sim, _, m = _sim("normalize", sync_only=True)
    sim.config = replace(sim.config, hazard_check="warn")
    # The memory reporter is configured when the simulator is constructed.
    sim = FunctionalSimulator(sim.program, sim.config)
    with pytest.warns(RuntimeWarning, match="read-before-write"):
        sim.run()
    for region in m["dst_regions"]:
        view = sim._resolve(region, 0)
        assert not view.allocation.initialized(view)


@pytest.mark.parametrize("rows,work", [(8, 8192), (16, 8192), (16, 16384), (32, 16384)])
def test_v2_variable_rows_work_numerical(rows, work):
    from dataclasses import replace

    regions = _regions()
    regions["score"] = replace(regions["score"], shape=(rows, 512))
    regions["scratch"] = replace(regions["scratch"], shape=(work,))
    m = helper_metadata("flash_update", regions, slot=0, row=64 - rows, rows=rows, work=work, version=2)
    task = Task("v2", "fia_helper", 0, Lane.VECTOR_0, Pipe.VECTOR, 1, metadata=m)
    p = KernelProgram(
        "v2", "A2", (CoreProgram(0, (task,)),), buffers=tuple(BufferSpec(r.buffer, r.scope, r.shape, r.dtype) for r in regions.values())
    )
    sim = FunctionalSimulator(p)
    sim.write(regions["score"], np.zeros((rows, 512), np.float32))
    sim.write(regions["maxv"], np.zeros(64, np.float32))
    sim.write(regions["sums"], np.ones(192, np.float32))
    sim.run()
    np.testing.assert_array_equal(sim.read(m["sum_out"]), np.full(rows, 513))
    np.testing.assert_array_equal(sim.read(m["old_sum"]), np.ones(rows))
    assert m["helper_contract"] == "sim.fia.v2.flash_update"


def test_v2_unknown_shape_fails_closed():
    with pytest.raises(UnsupportedSimOpError, match="rows"):
        helper_metadata("flash_init", _regions(), version=2, rows=24, work=8192)


def _v2_phase_alias_program(*, bank=0, fence=False, mutation="raw", rows=16, work=8192, kind="flash_init"):
    import tvm
    from tilelang.simulator.bridge import build_kernel_program

    t = tvm.tir
    sizes = {"score": 2 * rows * 512, "maxv": 64, "sums": 192, "alpha": 192, "denom": 64, "scratch": work, "brcb": 512, "tail_col": 512}
    buffers = {
        name: t.decl_buffer((n,), "uint8" if name == "scratch" else "float32", name=name, scope="shared.ub") for name, n in sizes.items()
    }
    gm = t.decl_buffer((512,), "float32", name="gm")
    if kind == "tail":
        source = f"fia_aligned::Tail<257,{rows}>(score[sm_bank * {rows * 512}], tail_col);"
        opaque = t.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.ascend_src_code"),
            source,
            buffers["score"].access_ptr("rw"),
            buffers["tail_col"].access_ptr("rw"),
            0,
            "sim.fia.v2.tail",
            257,
            rows,
            bank,
        )
    else:
        source = f"fia_aligned::Flash<false,{rows},{work}>(score[sm_bank * {rows * 512}],maxv,sums,alpha,denom,scratch,sm_slot,sm_row);"
        opaque = t.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.ascend_src_code"),
            source,
            *[buffers[n].access_ptr("rw") for n in ("score", "maxv", "sums", "alpha", "denom", "scratch")],
            0,
            0,
            0,
            "sim.fia.v2.flash_init",
            1,
            0,
            rows,
            work,
            bank,
        )
    copy = t.call_extern("handle", "copy_gm_to_ub", gm.data, buffers["brcb"].data, 512)
    calls = [copy, opaque] if mutation == "raw" else [opaque, copy]
    if fence:
        src, dst = ("mte2", "v") if mutation == "raw" else ("v", "mte2")
        calls[1:1] = [t.call_extern("handle", "set_flag", src, dst, 0), t.call_extern("handle", "wait_flag", src, dst, 0)]
    body = t.SeqStmt([t.Evaluate(c) for c in calls])
    for buffer in buffers.values():
        body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
    addresses = {}
    offset = 2 * rows * 512 * 4
    for name, b in buffers.items():
        addresses[b.data] = 0 if name in ("score", "brcb") else offset
        if name not in ("score", "brcb"):
            offset += sizes[name] * (1 if name == "scratch" else 4)
    func = t.PrimFunc([gm.data], body, buffer_map={gm.data: gm}).with_attr("address_map", addresses)
    return build_kernel_program(func, platform="A2")


@pytest.mark.parametrize("mutation", ["raw", "war"])
def test_v2_phase_alias_hazard_and_disjoint_bank(mutation):
    from tilelang.simulator.sync import validate_memory_synchronization
    from tilelang.simulator.errors import MemoryHazardError

    with pytest.raises(MemoryHazardError, match="missing A2/A3 pipe synchronization"):
        validate_memory_synchronization(_v2_phase_alias_program(mutation=mutation))
    assert not validate_memory_synchronization(_v2_phase_alias_program(fence=True, mutation=mutation))
    # brcb aliases bank 0 only; selecting bank 1 must not create a false edge.
    program = _v2_phase_alias_program(bank=1, mutation=mutation)
    assert not validate_memory_synchronization(program)
    helper = next(task for task in program.tasks if task.operation == "fia_helper")
    assert helper.metadata["dst"].byte_offset == 16 * 512 * 4
    assert helper.metadata["scratch"].shape == (8192,)


@pytest.mark.parametrize("rows,work", [(8, 8192), (32, 16384)])
def test_v2_bridge_other_supported_slice_shapes(rows, work):
    program = _v2_phase_alias_program(bank=1, rows=rows, work=work)
    helper = next(task for task in program.tasks if task.operation == "fia_helper")
    assert helper.metadata["dst"].shape == (rows, 512)
    assert helper.metadata["scratch"].shape == (work,)


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_v2_tail_bank_numerical(rows):
    program = _v2_phase_alias_program(bank=1, rows=rows, work=8192, kind="tail")
    helper = next(task for task in program.tasks if task.operation == "fia_helper")
    # Isolate the helper payload from the deliberately-unfenced alias probe.
    program = KernelProgram("tail", "A2", (CoreProgram(0, (helper,)),), buffers=program.buffers)
    sim = FunctionalSimulator(program)
    full = BufferRegion("score", MemoryScope.UB, (2, rows, 512), "float32")
    sim.write(full, np.full((2, rows, 512), 3, np.float32))
    sim.run()
    value = sim.read(full)
    np.testing.assert_array_equal(value[0], 3)
    np.testing.assert_array_equal(value[1, :, :257], 3)
    assert np.isneginf(value[1, :, 257:]).all()


def test_v2_normalize_divmul_writes_reciprocal_and_multiplies():
    r = _regions()
    m = helper_metadata("normalize_divmul", r, version=2, work=8192)
    task = Task("norm_divmul", "fia_helper", 0, Lane.VECTOR_0, Pipe.VECTOR, 1, metadata=m)
    p = KernelProgram(
        "norm_divmul",
        "A2",
        (CoreProgram(0, (task,)),),
        buffers=tuple(BufferSpec(region.buffer, region.scope, region.shape, region.dtype) for region in r.values()),
    )
    sim = FunctionalSimulator(p)
    denom = np.linspace(1.1, 7, 64, dtype=np.float32)
    value = np.linspace(-1, 1, 8192, dtype=np.float32).reshape(64, 128)
    sim.write(r["denom"], denom)
    sim.write(r["dst"], value)
    sim.run()
    reciprocal = np.float32(1) / denom
    np.testing.assert_array_equal(sim.read(r["denom"]), reciprocal)
    np.testing.assert_array_equal(sim.read(r["dst"]), value * reciprocal[:, None])
    assert r["denom"] in m["dst_regions"]
    assert validate_contract("sim.fia.v2.normalize_divmul", "fia_aligned::NormalizeDivMul(acc_o,denom,brcb);") == "normalize_divmul"


@pytest.mark.parametrize("mutation", ["raw", "war", "waw"])
def test_v2_normalize_divmul_bridge_hazards(mutation):
    from tilelang.simulator.sync import validate_memory_synchronization
    from tilelang.simulator.errors import MemoryHazardError

    program = _bridge_program(kind="normalize_divmul", fence=False, mutation=mutation)
    with pytest.raises(MemoryHazardError):
        validate_memory_synchronization(program)
    assert not validate_memory_synchronization(_bridge_program(kind="normalize_divmul", fence=True, mutation=mutation))
