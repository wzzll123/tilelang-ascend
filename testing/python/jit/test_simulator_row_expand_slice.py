"""Regression: row_expand_*_experiment on 64-column UB slices must be accepted by the simulator bridge.

Bug (2026-09-18, GQA campaign): bridge._row_expand_metadata validates the row width via
``destination_spec.shape[-1]`` — the *parent buffer's* declared (possibly memory-planner
flattened) last dim — instead of the sliced region's actual 64-element row width. Kernels
that meet the hardware 256-byte-row requirement *by slicing into 64-column windows*
(the GQA Brcb-fusion idiom, ``row_expand_sub_experiment(acc[:, j*64:(j+1)*64], ...)``)
are rejected at compile time with::

    ProgramValidationError: row_expand_sub_experiment requires 256-byte (64-element) rows,
    got <parent_cols> columns

The same kernels run correctly on real hardware; this is a bridge-side slice-geometry
misread (compile-time reject, not silent wrong data). Sibling of the sliced-ops backend
defects (offset>0 strided UB slice lowering).

Fixed on branch fix/sim-bridge-rowexpand-slice: the bridge now validates the access-window
width (256-byte rows) and recovers the physical row stride from the pre-planning
``initial_buffer_shapes`` attr, mirroring ``_exp_experiment_metadata``.
"""

import numpy as np

import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

ROWS = 8
COLS = 128  # parent row width; slices are the legal 64-element (256-byte fp32) windows


def _row_expand_sliced_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([ROWS, COLS], "float32"),
            output: T.Tensor([ROWS, COLS], "float32"),
            scalars: T.Tensor([ROWS, 1], "float32"),
        ):
            with T.Kernel(1, is_npu=True):
                src_ub = T.alloc_ub([ROWS, COLS], "float32")
                row_scalar = T.alloc_ub([ROWS, 1], "float32")
                expand_tmp = T.alloc_ub([ROWS * 8], "float32")
                with T.Scope("V"):
                    T.copy(source, src_ub)
                    T.copy(scalars, row_scalar)
                    # GQA Brcb-fusion idiom: slide a 64-column window over the wide tile.
                    for j in range(COLS // 64):
                        T.tile.row_expand_sub_experiment(
                            src_ub[:, j * 64:(j + 1) * 64],
                            src_ub[:, j * 64:(j + 1) * 64],
                            row_scalar,
                            tmp=expand_tmp,
                        )
                    T.copy(src_ub, output)

        return main

    return kernel()


def test_row_expand_sub_on_64col_slices() -> None:
    rng = np.random.default_rng(0)
    source = rng.standard_normal((ROWS, COLS), dtype=np.float32)
    scalars = rng.standard_normal((ROWS, 1), dtype=np.float32)
    expected = source - scalars

    output = _row_expand_sliced_kernel("A2")(source, scalars)
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
