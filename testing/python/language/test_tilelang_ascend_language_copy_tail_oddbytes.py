import pytest
import tilelang
import tilelang.language as T
import torch

"""
GM -> UB copy tail with a non-32-Byte-aligned full-row width (isPad routing).

Feature under test
------------------
``copy_gm_to_ub`` (src/tl_templates/ascend/common.h) decides whether the
DataCopyPad needs padding via ``isPad``.  The legacy routing was::

    if (maskShapeN == dstN || (maskShapeN * sizeof(T)) % 32 == 0)
        isPad = false;   // "no padding needed"

The ``maskShapeN == dstN`` arm is WRONG: it assumes "reading the full row means
no padding", ignoring the case where the full row's byte count is NOT a 32-Byte
multiple.  DataCopyPad moves data in whole 32-Byte blocks; with isPad=false a
124-Byte row (31 fp32) is read as 3 full 32-Byte blocks (96 B) and the trailing
28 B (7 elements) are silently dropped / read out of bounds.

The fix routes purely on byte alignment::

    isPad = (maskShapeN * sizeof(T)) % 32 != 0;
    rightPadding = elements needed to pad the row up to a 32-Byte boundary;

so a full-row-but-odd-byte copy (maskShapeN == dstN == 31 fp32) is sent down the
padded path and the tail lands correctly.

This suite loads a 31-element fp32 row into a 31-wide UB window and stores it
back, checking the round trip bit-exactly.  On the legacy routing the last 7
elements are garbage; on the fixed routing all 31 match.

NOTE: executes on real NPU hardware (``.npu()``).
"""

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


def odd_full_row(N, dtype):
    """Full-row copy whose byte width N*sizeof(T) is NOT a 32B multiple.
    maskShapeN == dstN == N (read the whole row), so the legacy
    ``maskShapeN == dstN`` arm sends it down the unpadded path and truncates the
    sub-32B tail."""

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            win = T.alloc_ub((N,), dtype)
            T.copy(X[0:N], win)
            T.copy(win, Y[0:N])

    return main


def run_test_odd_full_row(N, dtype):
    torch.manual_seed(0)
    func = odd_full_row(N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS,
                            target="ascendc")
    td = _torch_dtype(dtype)
    x = torch.randn(N, dtype=td).npu()
    torch.npu.synchronize()
    y = func(x)
    torch.npu.synchronize()
    # Bit-exact round trip: every one of the N elements (incl. the sub-32B tail)
    # must survive.
    torch.testing.assert_close(y.cpu(), x.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "N,dtype",
    [
        (31, "float"),    # 31*4 = 124 B, not a 32B multiple -> tail 7 elements
        (15, "float16"),  # 15*2 = 30 B,  not a 32B multiple -> tail 7 elements
    ],
)
def test_odd_full_row(N, dtype):
    run_test_odd_full_row(N, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
