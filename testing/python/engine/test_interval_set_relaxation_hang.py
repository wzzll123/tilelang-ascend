# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regression test for a lowering hang in tvm.arith interval-set relaxation.

The weight_quant_batch_matmul (wqbm) TileLang design hangs inside
``LowerAndLegalize``'s ``tir.transform.Simplify`` when M is tiled
(``m_num = 2``, e.g. M=256 > BLOCK_M=128).  The M-tiling guard
``min(m_half, M - (cid // cubeBlockDimN * block_M + vid * m_half))`` mixes the
two thread vars ``cid``/``vid``, so their inferred domain bounds become
mutually referential.  ``IntervalSetEvaluator::Eval(IntervalSet)`` then relaxed
the bounds recursively without converging -- each round re-intersected a var's
constraints and nested the min/max endpoints one level deeper -- hanging
lowering for >10 minutes.

The fix backports apache/tvm#19670 (commit ``96b8257``): fully-relaxed variable
intervals are memoized and an in-progress set breaks cyclic dependencies.  It
therefore removes the exponential repeated expansion without imposing an
arbitrary depth or expression-complexity cutoff.  These tests run the compile
in a subprocess with a hard timeout: before the fix the subprocess is killed by
the timeout (test goes red); after the fix it completes (test goes green).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Path to the (read-only) wqbm design file that reproduces the hang.
WQBM_DESIGN_DIR = (
    "/home/wenzhongzhen/kernel-tool/cannbot-skills-0817/plugins-community/"
    "tilelang2ascendc-ops-generator/workflows/templates/archive_tasks/"
    "weight_quant_batch_matmul"
)

# Generous ceiling: the fixed compile takes ~12s; the buggy one never finishes.
COMPILE_TIMEOUT_S = 120


def _run_subprocess(source: str, timeout: int) -> subprocess.CompletedProcess:
    """Run ``source`` in a fresh python subprocess with a hard timeout."""
    env = os.environ.copy()
    # The tilelang cache key does not cover header/pass behaviour; clear it so
    # the subprocess always exercises the current libtvm.
    subprocess.run(["rm", "-rf", os.path.expanduser("~/.tilelang/cache")], check=False)
    return subprocess.run(
        [sys.executable, "-u", "-c", source],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.skipif(
    not os.path.isdir(WQBM_DESIGN_DIR),
    reason="wqbm design file not present on this machine",
)
def test_wqbm_m256_tiling_compile_does_not_hang():
    """wqbm native compile with M-tiling (M=256) must finish within timeout."""
    source = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {WQBM_DESIGN_DIR!r})
        from design.tile_level.weight_quant_batch_matmul import (
            WQBMConfig, weight_quant_batch_matmul_fwd)
        t0 = time.time()
        weight_quant_batch_matmul_fwd(
            WQBMConfig(m=256, n=512, k=512, dtype="float16",
                       has_offset=True, has_bias=True))
        print("COMPILE_OK", f"{{time.time()-t0:.1f}}s", flush=True)
        """
    )
    proc = _run_subprocess(source, COMPILE_TIMEOUT_S)
    assert "COMPILE_OK" in proc.stdout, (
        f"wqbm M=256 compile did not complete within {COMPILE_TIMEOUT_S}s "
        f"(returncode={proc.returncode}).\nstdout tail: {proc.stdout[-500:]}\n"
        f"stderr tail: {proc.stderr[-500:]}"
    )


def test_mutually_referential_var_domains_terminate():
    """Interval-set evaluation over mutually-referential var domains must
    terminate (the arithmetic-level essence of the wqbm hang)."""
    source = textwrap.dedent(
        """
        import time
        import tilelang  # noqa: F401  (sets up repo libtvm)
        from tvm import tir
        from tvm.arith import Analyzer, IntervalSet

        # A cycle of vars whose upper bounds reference one another.  The
        # in-progress set must break the cycle rather than recursively nesting
        # min/max bounds until the old depth budget is exhausted.
        N = 12
        vs = [tir.Var(f"v{i}", "int32") for i in range(N)]
        ana = Analyzer()
        dom = {}
        for i in range(N):
            nxt = vs[(i + 1) % N]
            dom[vs[i]] = IntervalSet(tir.const(0, "int32"),
                                     tir.min(tir.const(7, "int32"), nxt))
        acc = vs[0]
        for v in vs[1:]:
            acc = acc + v
        expr = tir.min(tir.const(64, "int32"), acc)
        t0 = time.time()
        ana.int_set(expr, dom)
        print("INTSET_OK", f"{time.time()-t0:.2f}s", flush=True)
        """
    )
    proc = _run_subprocess(source, 60)
    assert "INTSET_OK" in proc.stdout, (
        f"int_set over mutually-referential domains did not terminate "
        f"(returncode={proc.returncode}).\nstderr tail: {proc.stderr[-500:]}"
    )


if __name__ == "__main__":
    test_mutually_referential_var_domains_terminate()
    if os.path.isdir(WQBM_DESIGN_DIR):
        test_wqbm_m256_tiling_compile_does_not_hang()
    print("PASS")
