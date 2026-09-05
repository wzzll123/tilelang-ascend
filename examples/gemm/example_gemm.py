import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
parser.add_argument(
    "--simulator", action="store_true", help="Run the CPU A2/A3 simulator"
)
parser.add_argument(
    "--platform", choices=["A2", "A3"], default="A2",
    help="Simulator platform (used with --simulator)",
)
parser.add_argument(
    "--trace", type=str, default=None,
    help="Optional Chrome/Perfetto trace path in simulator mode",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
jit_options = {}
if args.simulator:
    jit_options = {
        "simulator": True,
        "platform": args.platform,
        "sim_config": {"trace_path": args.trace} if args.trace else None,
    }


@tilelang.jit(out_idx=[-1], **jit_options)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


if M % 128 or N % 256 or K % 64:
    raise ValueError("this example requires M % 128 == N % 256 == K % 64 == 0")

func = matmul(M, N, K, 128, 256, 64)

torch.manual_seed(0)

a = torch.randn(M, K).half()
b = torch.randn(K, N).half()
if not args.simulator:
    a = a.npu()
    b = b.npu()
print("init successful!")

c = func(a, b)

ref_c = a.cpu() @ b.cpu()

torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
print(f"mode={'simulator' if args.simulator else 'npu'}")
