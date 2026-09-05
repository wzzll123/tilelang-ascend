import argparse

import tilelang as tl
import tilelang.language as T
import torch

tl.disable_cache()

ELEMENT_SIZE = 2
VALUE_POSITION = 0
INDEX_POSITION = 1

N = 64

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_2way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 2 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 2 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)

            T.tile.merge_sort(merge_output, src0, src1)

            T.copy(merge_output, output)

    return main


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_3way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 3 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 3 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)

            T.tile.merge_sort(merge_output, src0, src1, src2)

            T.copy(merge_output, output)

    return main


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_merge_sort_4way():
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block3: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 4 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src3 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 4 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)
            T.copy(block3, src3)

            T.tile.merge_sort(merge_output, src0, src1, src2, src3)

            T.copy(merge_output, output)

    return main


def create_sorted_block(N):
    values = torch.randn(N, dtype=torch.float32)
    sorted_indices = torch.argsort(values, descending=True)
    sorted_values = values[sorted_indices]

    block = torch.zeros(N * ELEMENT_SIZE, dtype=torch.float32)
    for i in range(N):
        block[i * ELEMENT_SIZE + VALUE_POSITION] = sorted_values[i]
        block[i * ELEMENT_SIZE + INDEX_POSITION] = float(sorted_indices[i].item())

    return block


def ref_program(blocks):
    merge_num = len(blocks)
    sequences = []
    for block in blocks:
        pairs = []
        for j in range(N):
            value = block[j * ELEMENT_SIZE + VALUE_POSITION].item()
            index = block[j * ELEMENT_SIZE + INDEX_POSITION].item()
            pairs.append((value, index))
        sequences.append(pairs)

    import heapq

    neg_seqs = [[(-v, i) for v, i in seq] for seq in sequences]
    merged = list(heapq.merge(*neg_seqs))
    merged = [(-v, i) for v, i in merged]

    result = torch.zeros(N * merge_num * ELEMENT_SIZE, dtype=torch.float32)
    for i, (value, index) in enumerate(merged):
        result[i * ELEMENT_SIZE + VALUE_POSITION] = value
        result[i * ELEMENT_SIZE + INDEX_POSITION] = index

    return result


def format_block(block):
    elements = []
    total = len(block) // ELEMENT_SIZE
    for i in range(total):
        value = block[i * ELEMENT_SIZE + VALUE_POSITION].item()
        index = block[i * ELEMENT_SIZE + INDEX_POSITION].item()
        elements.extend([value, index])
    return elements


def test_merge(
    merge_num, *, simulator=False, platform="A2", trace=None, verbose=False
):
    print(f"\n{'=' * 60}")
    print(f"Testing {merge_num}-way merge sort (value-index pair format):")
    print(f"N = {N} elements per block, each element = {ELEMENT_SIZE} floats")
    print("=" * 60)

    blocks = [create_sorted_block(N) for _ in range(merge_num)]
    if verbose:
        print("blocks", blocks)
        print("\nInput blocks (value, index pairs, all elements):")
        for i in range(merge_num):
            formatted = format_block(blocks[i])
            print(f"  Block {i}: {formatted}")

    device_blocks = blocks if simulator else [b.npu() for b in blocks]

    if merge_num == 2:
        kernel_factory = generate_merge_sort_2way
    elif merge_num == 3:
        kernel_factory = generate_merge_sort_3way
    elif merge_num == 4:
        kernel_factory = generate_merge_sort_4way
    else:
        raise ValueError(f"unsupported merge count: {merge_num}")

    if simulator:
        kernel_factory = tl.jit(
            out_idx=[-1],
            simulator=True,
            platform=platform,
            sim_config={"trace_path": trace} if trace else None,
            pass_configs=pass_configs,
        )(kernel_factory.__wrapped__)
    kernel = kernel_factory()

    print("\nGenerated kernel source:")

    if not simulator:
        torch.npu.synchronize()
    print("init successful!")

    result = kernel(*device_blocks)
    if not simulator:
        torch.npu.synchronize()

    ref_result = ref_program(blocks)

    result_cpu = result.cpu()
    if verbose:
        print(f"\nOutput (all elements): {format_block(result_cpu)}")
        print(f"\nref_result output (all elements): {format_block(ref_result)}")

    output_values = [result_cpu[i * ELEMENT_SIZE + VALUE_POSITION].item() for i in range(N * merge_num)]
    is_sorted = all(output_values[i] >= output_values[i + 1] for i in range(len(output_values) - 1))
    print(f"\nIs output sorted (descending): {is_sorted}")

    records_match = torch.equal(result_cpu, ref_result)
    print(f"Full value/index records match: {records_match}")

    if not records_match:
        matching = int((result_cpu == ref_result).sum().item())
        print(f"Matching scalar slots: {matching}/{ref_result.numel()}")

    return is_sorted and records_match


def main():
    parser = argparse.ArgumentParser(description="Merge-sort example")
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
    parser.add_argument(
        "--verbose", action="store_true", help="Print every input/output record"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    tl.disable_cache()

    results = []
    for merge_num in [2, 3, 4]:
        success = test_merge(
            merge_num,
            simulator=args.simulator,
            platform=args.platform,
            trace=args.trace,
            verbose=args.verbose,
        )
        results.append((merge_num, success))

    print(f"\n{'=' * 60}")
    print("Summary:")
    for merge_num, success in results:
        status = "Kernel Output Match!" if success else "FAIL"
        print(f"  {merge_num}-way merge: {status}")
    if not all(success for _, success in results):
        raise AssertionError("one or more merge-sort variants failed")
    print(f"mode={'simulator' if args.simulator else 'npu'}")


if __name__ == "__main__":
    main()
