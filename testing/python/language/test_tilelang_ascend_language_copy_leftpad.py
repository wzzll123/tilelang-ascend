import pytest
import tilelang
import tilelang.language as T
import torch

"""
GM -> UB copy ``left_pad`` (stencil shift-window) regression suite.

Feature under test
------------------
``T.copy(src_gm, dst_ub, left_pad=PL)`` (tilelang/language/copy_op.py ::
npu_copy_v2) is an OPT-IN extension of the plain GM->UB load.  It loads the
valid source region ``src[0:W]`` into the destination window starting at column
offset ``PL`` (i.e. dst[PL : PL+W] = src[0:W]), and fills the left gap
dst[0:PL] with the pad value (default 0) via the DataCopyPad ``leftPadding``
hardware field.

This is the load primitive a stencil / sliding-window kernel (depthwise conv,
pooling) needs: the halo on the left of a row window must read as 0 (the golden
zero-padding semantics), while the W valid samples land shifted right by PL.
Without ``left_pad`` the only way to express this is a separate Duplicate to
zero the halo plus an offset copy -- and an offset UB destination whose column
start is not 32-Byte aligned (PL=1 fp16 = 2 B) is physically un-writable, so
the hardware leftPadding path is the only correct expression.

Semantics pinned by this suite (bit-exact against a torch golden):
  * dst[0 : PL]        == 0        (left halo filled by leftPadding)
  * dst[PL : PL+W]     == src[0:W] (valid window, shifted right by PL)
  * dst[PL+W : WP]     == 0        (right tail of the aligned window stays 0)

The destination UB window width WP = PL + W is chosen to be 32-Byte aligned so
the full window (halo + data + right tail) can be stored back to GM for a
bit-exact check:
  * fp16 (2 B): W=31, PL=1  -> WP=32 elements = 64 B  (aligned)
  * fp32 (4 B): W=7,  PL=1  -> WP=8  elements = 32 B  (aligned)

Opt-in contract (regression guard): a copy that does NOT pass ``left_pad``
must emit byte-identical codegen to before this feature existed.  Group 2
re-locks the plain path so the new parameter can never hijack an existing
copy (the agent-2 dst_offset hijack regression).

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run
in a CPU-only environment.
"""

# Minimal vector config: auto in-core sync + memory planning.  No CV combine
# (pure AIV load/store, no cube work).
VEC_MIN_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {"float": torch.float32, "float32": torch.float32,
            "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


# =============================================================================
# Group 1 - left_pad shift-window (the feature under test)
# Load src[0:W] into a WP-wide UB window shifted right by PL, store the full
# window back, and check halo==0 / data==src / tail==0 bit-exactly.
# =============================================================================
def leftpad_window(W, PL, WP, dtype):
    """Single-block shift-window load.  ``win`` is a WP-wide UB buffer;
    ``T.copy(X[0:W], win, left_pad=PL)`` must place the W valid samples at
    win[PL:PL+W] and zero the PL-wide left halo.  The full WP window is stored
    to GM so the halo and right tail are observable."""

    @T.prim_func
    def main(X: T.Tensor((W,), dtype), Y: T.Tensor((WP,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            win = T.alloc_ub((WP,), dtype)
            # Opt-in shift-window load: dst base is the window start, the W
            # valid samples land at column offset PL, left halo zero-filled.
            T.copy(X[0:W], win, left_pad=PL)
            T.copy(win, Y[0:WP])

    return main


def run_test_leftpad_window(W, PL, WP, dtype):
    torch.manual_seed(0)
    func = leftpad_window(W, PL, WP, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS,
                            target="ascendc")
    td = _torch_dtype(dtype)
    x = torch.randn(W, dtype=td).npu()
    torch.npu.synchronize()
    y = func(x)
    torch.npu.synchronize()
    y = y.cpu()
    x = x.cpu()
    # Left halo [0:PL] must be exactly 0 (hardware leftPadding).
    assert torch.all(y[0:PL] == 0), f"left halo not zero: {y[0:PL]}"
    # Valid window [PL:PL+W] must equal the source bit-for-bit.
    torch.testing.assert_close(y[PL:PL + W], x, rtol=0, atol=0)
    # Right tail [PL+W:WP] must be 0 (aligned-window residue).
    if WP > PL + W:
        assert torch.all(y[PL + W:WP] == 0), \
            f"right tail not zero: {y[PL + W:WP]}"


@pytest.mark.parametrize(
    "W,PL,WP,dtype",
    [
        (31, 1, 32, "float16"),  # fp16: WP=32 elem = 64 B aligned
        (7, 1, 8, "float"),      # fp32: WP=8  elem = 32 B aligned
    ],
)
def test_leftpad_window(W, PL, WP, dtype):
    run_test_leftpad_window(W, PL, WP, dtype)


# =============================================================================
# Group 2 - plain copy unchanged (opt-in regression guard)
# A copy that does NOT pass left_pad must behave exactly as before: the valid
# region lands at the destination base with no left shift.  This locks the
# opt-in contract so the new parameter can never hijack an existing copy.
# =============================================================================
def plain_copy(N, dtype):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            T.copy(A[0:N], a_ub)  # no left_pad -> plain load at base
            T.copy(a_ub, C[0:N])

    return main


def run_test_plain_copy(N, dtype):
    torch.manual_seed(0)
    func = plain_copy(N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS,
                            target="ascendc")
    td = _torch_dtype(dtype)
    a = torch.randn(N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c.cpu(), a.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("N,dtype", [(32, "float16"), (8, "float")])
def test_plain_copy_unchanged(N, dtype):
    run_test_plain_copy(N, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
