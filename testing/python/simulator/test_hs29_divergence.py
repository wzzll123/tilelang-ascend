# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""HS29 divergence sentinel (hardware-vs-model known divergence).

GQA campaign, 2026-09-10: an experimental sync variant ("HS29-A": per-kernel
slot-credit preset instead of per-task preset+drain) is level-balanced by
construction and this simulator correctly reports it healthy -- yet on real
A2 hardware it deadlocks FLAKILY (~1/4 runs on the B1/S128/Nq48/Nkv4 shape;
the per-task-preset baseline is 8/8 clean). The mechanism lives outside the
modeled flag semantics (suspected: pipe-queue position / credit-fire timing
of SetFlag/WaitFlag markers), unresolved pending hardware observability
(msdebug) or official A2 flag semantics documentation.

These tests pin the CURRENT model verdict (healthy) for the distilled HS29
patterns. If a future flag-model improvement makes them report a deadlock,
that is a signal the new model may be closer to silicon -- investigate, then
flip the expectation with evidence. Do NOT delete these tests when they
"fail" after a model change.

Reference: plugins-community/tilelang2ascendc-ops-generator/workflows/
templates/archive_tasks/gqa/ascendc/GENERALIZE_DEBUG_LOG.md (ISSUE-HS29).
Full-fidelity variant: design/tile_level/gqa_hs29ab.py (run via gqa_sim.py
with GQA_SIM_VARIANT=hs29ab; cases hs29_case0/hs29_case2).
"""

import pytest

from tilelang.simulator import (
    CoreProgram,
    FunctionalSimulator,
    KernelProgram,
    Lane,
    Pipe,
    SimulationDeadlockError,
    SimulatorConfig,
    Task,
)


def _set(tid, flag_id, src="mte1", dst="mte2", pipe=None, core=0):
    if pipe is None:
        pipe = Pipe(src)
    return Task(
        tid,
        "set_flag",
        core,
        Lane.CUBE,
        pipe,
        1,
        metadata={"src_pipe": src, "dst_pipe": dst, "flag_id": flag_id},
    )


def _wait(tid, flag_id, src="mte1", dst="mte2", pipe=None, core=0):
    if pipe is None:
        pipe = Pipe(dst)
    return Task(
        tid,
        "wait_flag",
        core,
        Lane.CUBE,
        pipe,
        1,
        metadata={"src_pipe": src, "dst_pipe": dst, "flag_id": flag_id},
    )


def _kernel(name, tasks):
    return KernelProgram(name, "A2", (CoreProgram(0, tuple(tasks)),), buffers=())


def _run(program, **cfg):
    return FunctionalSimulator(
        program, SimulatorConfig(platform=program.platform, sync_only=True, **cfg)
    ).run()


def _task_cycle(prefix):
    """One HS29 task body: wait free -> produce -> set ready -> consume ->
    set free (level-balanced, mirrors the GQA K-slot lifecycle)."""
    return [
        _wait(f"{prefix}-wait-free", 0),
        _set(f"{prefix}-set-ready", 4, "mte2", "mte1"),
        _wait(f"{prefix}-wait-ready", 4, "mte2", "mte1"),
        _set(f"{prefix}-set-free", 0),
    ]


# The HS29-A pattern: credits preset ONCE at kernel entry; per-task bodies are
# exactly balanced (no per-task preset, no tail drain).
HS29A_TASKS = (
    # kernel-entry preset (the HS29-A delta)
    [_set("preset-free0", 0), _set("preset-free1", 1)]
    + _task_cycle("t0")
    + _task_cycle("t1")
    + _task_cycle("t2")
)

# The baseline pattern (online v7): per-task preset + per-task drain.
BASELINE_TASKS = []
for i in range(3):
    BASELINE_TASKS += [_set(f"t{i}-preset0", 0), _set(f"t{i}-preset1", 1)]
    BASELINE_TASKS += _task_cycle(f"t{i}")
    BASELINE_TASKS += [_wait(f"t{i}-drain0", 0), _wait(f"t{i}-drain1", 1)]


@pytest.mark.parametrize("flag_blocking", [False, True])
def test_hs29a_pattern_reports_undrained_kernel_entry_credits(flag_blocking) -> None:
    """HS29-A leaves its one-time preset credits live at the kernel boundary."""
    program = _kernel("hs29a-sentinel", HS29A_TASKS)
    with pytest.raises(SimulationDeadlockError, match=r"FLAG ACCOUNTING.*id=0.*level=1"):
        _run(program, flag_blocking=flag_blocking)


@pytest.mark.parametrize("flag_blocking", [False, True])
def test_baseline_pattern_healthy_in_current_model(flag_blocking) -> None:
    """The shipped baseline pattern must stay healthy under any flag mode."""
    program = _kernel("hs29-baseline-sentinel", BASELINE_TASKS)
    result = _run(program, flag_blocking=flag_blocking)
    assert result is not None
