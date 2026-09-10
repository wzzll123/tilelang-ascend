# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""GQA HS29 variant: full-program hardware-vs-model divergence test (A2).

This drives the REAL GQA kernel design (the HS29 A+B variant: per-kernel
slot-credit preset + C0 preload reorder + no task-tail drain -- fixture:
gqa_hs29ab_design.py) through the simulator, pinning the current model verdict
(healthy) for a program that deadlocks FLAKILY on real A2 silicon.

Hardware evidence (2026-09-10, ascendc/test_hs29_hw_vs_sim.sh):
  hardware : aicore hang within a few runs (B1/B2, S128, Nq48, Nkv4 shapes)
  simulator: completes cleanly (sync_only, full, and flag_blocking modes)

The mechanism is outside the modeled flag semantics (suspected pipe-queue
marker-position / credit-fire timing). This test MUST keep asserting the
current verdict; if a future flag-model change makes it deadlock, that is the
signal the model moved closer to silicon -- investigate and flip with evidence.
See test_hs29_divergence.py for the distilled pattern version.
"""

import os

import pytest

# The fixture reads GQA_SIM at module level (drops the F217 all-masked-row
# scalar-if, which the bridge cannot express); set it BEFORE import.
os.environ["GQA_SIM"] = "1"

import torch  # noqa: E402

import tilelang  # noqa: E402

# Smallest HS29-relevant shape: kv_loops=1 + multiple tasks per core
# (B24/Hq8/Hkv2/S8/Skv128 -> block_num=48 > 20 cores, tasks_per_core=3).
# The full hardware trigger shapes (B2/S128/Nq48/Nkv4 = 96 tasks) live in the
# task-side gqa_sim.py (cases hs29_case0/hs29_case2); this one keeps the test
# compile time tractable while carrying the identical sync structure.
HS29_MT_SHAPE = dict(
    batch=24, heads_q=8, heads_kv=2, q_seq_len=8, kv_seq_len=128, dim=128,
    causal=False, dtype="float16",
)


def _gen(shape, dtype):
    dt = torch.float16 if dtype == "float16" else torch.bfloat16
    return (torch.rand(shape, dtype=torch.float32) * 2 - 1).to(dt)


@pytest.mark.parametrize("flag_blocking", [False, True])
def test_gqa_hs29ab_variant_healthy_in_current_model(flag_blocking) -> None:
    from gqa_hs29ab_design import GQAConfig, gqa_fwd, pass_configs

    builder = gqa_fwd.__jit_impl__.func   # undecorated raw builder
    cfg = GQAConfig(**HS29_MT_SHAPE)
    torch.manual_seed(42)
    q = _gen((cfg.batch, cfg.q_seq_len, cfg.heads_q, cfg.dim), cfg.dtype)
    k = _gen((cfg.batch, cfg.kv_seq_len, cfg.heads_kv, cfg.dim), cfg.dtype)
    v = _gen((cfg.batch, cfg.kv_seq_len, cfg.heads_kv, cfg.dim), cfg.dtype)

    ker = tilelang.compile(
        builder(cfg),
        out_idx=[3],
        workspace_idx=[4, 5, 6],
        pass_configs=pass_configs,
        target="ascendc",
        simulator=True,
        platform="A2",
        sim_config={
            "hazard_check": "error",   # hard gate: any hazard fails the test
            "sync_only": True,
            "deadlock_detect": True,
            "flag_blocking": flag_blocking,
            "execution_timeout_s": 600.0,
        },
    )
    # Completes without deadlock / hazard / validation error = the model's
    # current verdict. (sync_only skips numerics by design.)
    ker(q, k, v, 1.0 / (cfg.dim ** 0.5))
