# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2 resident-data compute controls, derived from the expert FA timer probe.

GetSystemCycle is a 50 MHz system counter, NOT an AI Core cycle counter.
Only the repeated compute region and its completion drain are timed. Inputs,
priming, output copies and cache-cleaned diagnostic export are outside it.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import tilelang
from tilelang import language as T
from tilelang.language.ascend import _src_code
from tilelang.intrinsics import make_zn_layout

PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
COUNTS = (0, 16, 64, 256)


def cube(m, n, k, op="mma"):
    fix_copy = op in {"copy_l0c_gm", "copy_l0c_gm_half", "copy_l0c_gm_half_strided"}
    output_dtype = "float16" if op in {"copy_l0c_gm_half", "copy_l0c_gm_half_strided"
                                      } else "float32"
    destination_cols = 2 * n if op == "copy_l0c_gm_half_strided" else n
    # A2 L0C is 128 KiB. Wide FIX reads an immutable source, so one source
    # slot suffices while the two GM output slots remain independent.
    c_slots = 1 if m * n * 4 > 65536 else 2
    if not fix_copy and c_slots != 2:
        raise ValueError("compute controls require two accumulator slots within 128 KiB L0C")
    if c_slots * m * n * 4 > 131072 or m * k * 2 > 65536 or k * n * 2 > 65536:
        raise ValueError("probe tile exceeds A2 L0 capacity")

    @T.prim_func
    def kernel(a: T.Tensor((m, k), "float16"), b: T.Tensor((k, n),
                                                           "float16"), out: T.Tensor(
                                                               (2, m, destination_cols),
                                                               output_dtype),
               stamp: T.Tensor((8,), "int64"), count: T.int32, chain: T.int32, enabled: T.int32):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            al1 = T.alloc_L1((m, k), "float16")
            bl1 = T.alloc_L1((k, n), "float16")
            T.annotate_layout({al1: make_zn_layout(al1), bl1: make_zn_layout(bl1)})
            al0 = T.alloc_L0A((m, k), "float16")
            bl0 = T.alloc_L0B((k, n), "float16")
            c = T.alloc_L0C((c_slots, m, n), "float32")
            with T.Scope("C"):
                T.copy(a, al1)
                T.copy(b, bl1)
                T.set_flag("MTE2", "MTE1", 0)
                T.wait_flag("MTE2", "MTE1", 0)
                T.copy(al1, al0)
                T.copy(bl1, bl0)
                T.set_flag("MTE1", "M", 0)
                T.wait_flag("MTE1", "M", 0)
                for slot in T.unroll(c_slots):
                    T.mma(al0, bl0, c[slot, :, :], init=True)
                T.barrier_all()
                _src_code("int64_t bench_before = AscendC::GetSystemCycle();")
                for i in T.serial(count):
                    slot = T.if_then_else(chain != 0, 0, i % 2)
                    c_slot = slot % c_slots
                    if enabled != 0:
                        if op == "copy_gm_l1":
                            T.copy(a, al1)
                        elif op == "copy_l1_l0a":
                            T.copy(al1, al0)
                        elif op == "copy_l1_l0b":
                            T.copy(bl1, bl0)
                        elif op == "copy_l1_l0b_transpose":
                            T.copy(bl1, bl0, transpose=True)
                        elif fix_copy:
                            T.copy(c[c_slot, :, :], out[slot, :, :n])
                        elif op == "mma_init":
                            T.mma(al0, bl0, c[c_slot, :, :], init=True)
                        else:
                            T.mma(al0, bl0, c[c_slot, :, :], init=False)
                    if chain != 0:
                        if op == "copy_gm_l1":
                            T.pipe_barrier("MTE2")
                        elif op in ("copy_l1_l0a", "copy_l1_l0b", "copy_l1_l0b_transpose"):
                            T.pipe_barrier("MTE1")
                        elif fix_copy:
                            T.pipe_barrier("FIX")
                        else:
                            T.pipe_barrier("M")
                T.barrier_all()
                _src_code("int64_t bench_after = AscendC::GetSystemCycle();")
                if op in ("copy_gm_l1", "copy_l1_l0a", "copy_l1_l0b", "copy_l1_l0b_transpose"):
                    if op == "copy_gm_l1":
                        T.copy(al1, al0)
                    T.set_flag("MTE1", "M", 1)
                    T.wait_flag("MTE1", "M", 1)
                    for slot in T.unroll(c_slots):
                        T.mma(al0, bl0, c[slot, :, :], init=True)
                T.set_flag("M", "FIX", 0)
                T.wait_flag("M", "FIX", 0)
                for slot in T.unroll(2):
                    if fix_copy:
                        if enabled == 0 or count == 0 or (slot == 1 and (chain != 0 or count == 1)):
                            T.copy(c[slot % c_slots, :, :], out[slot, :, :n])
                    else:
                        T.copy(c[slot % c_slots, :, :], out[slot, :, :n])
                _src_code("stamp.SetValue(0, bench_before); stamp.SetValue(1, bench_after);")
                stamp[2] = T.cast(count, "int64")
                T.tile.datacachecleanandinvalid_experiment(stamp, "SINGLE_CACHE_LINE",
                                                           "CACHELINE_OUT")

    return kernel


def vector(op, rows=16, cols=512, dtype="float32"):
    if op not in {
            "add", "sub", "mul", "div", "exp", "fill", "axpy", "axpy_mixed", "max", "muls",
            "mul_add_dst", "copy_ub_ub", "copy_gm_ub", "copy_ub_gm", "brcb_experiment",
            "row_expand_sub_experiment", "row_expand_mul_experiment", "row_expand_div_experiment",
            "cast_f16_f32", "cast_f32_f16", "reduce_sum", "reduce_max"
    }:
        raise ValueError(f"unsupported probe operation: {op}")
    padded_rows = (rows + 7) // 8 * 8
    source_dtype = "float16" if op in ("axpy_mixed", "cast_f16_f32") else dtype
    output_dtype = "float16" if op == "cast_f32_f16" else dtype
    w_shape = (rows, cols) if op == "mul_add_dst" else (1, 8)
    uses_coeff = op in {
        "brcb_experiment", "row_expand_sub_experiment", "row_expand_mul_experiment",
        "row_expand_div_experiment"
    }
    coeff_rows = rows if uses_coeff else 8
    y_shape = (1, 8) if uses_coeff else (rows, cols)

    @T.prim_func
    def kernel(
            a: T.Tensor((2, 2, rows, cols), output_dtype), b: T.Tensor((2, rows, cols),
                                                                       source_dtype), out: T.Tensor(
                                                                           (2, 2, rows, cols),
                                                                           output_dtype),
            stamp: T.Tensor((2, 8), "int64"), count: T.int32, chain: T.int32, enabled: T.int32):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x = T.alloc_ub((2, rows, cols), output_dtype)
            y = T.alloc_ub(y_shape, source_dtype)
            w = T.alloc_ub(w_shape, source_dtype)
            coeff_in = T.alloc_ub((coeff_rows,), "float32")
            coeff = T.alloc_ub((coeff_rows, 8), "float32")
            z = T.alloc_ub((2, padded_rows), dtype)
            tmp0 = T.alloc_ub((4096,), "float32")
            tmp1 = T.alloc_ub((4096,), "float32")
            with T.Scope("V"):
                for slot in T.unroll(2):
                    T.copy(a[vid, slot, :, :], x[slot, :, :])
                if not uses_coeff:
                    T.copy(b[vid, :, :], y)
                if op == "mul_add_dst":
                    T.copy(b[vid, :, :], w)
                if uses_coeff:
                    T.copy(b[vid, 0, :rows], coeff_in)
                T.set_flag("MTE2", "V", 0)
                T.wait_flag("MTE2", "V", 0)
                if uses_coeff:
                    T.tile.brcb_experiment(coeff, coeff_in, rows // 8, 1, 8)
                for slot in T.unroll(2):
                    if op == "reduce_sum":
                        if slot == 0:
                            T.reduce_sum(x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp0)
                        else:
                            T.reduce_sum(x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp1)
                    elif op == "reduce_max":
                        if slot == 0:
                            T.reduce_max(x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp0)
                        else:
                            T.reduce_max(x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp1)
                T.barrier_all()
                _src_code("int64_t bench_before = AscendC::GetSystemCycle();")
                for i in T.serial(count):
                    slot = T.if_then_else(chain != 0, 0, i % 2)
                    if enabled != 0:
                        if op == "add":
                            T.tile.add(x[slot, :, :], x[slot, :, :], y)
                        elif op == "sub":
                            T.tile.sub(x[slot, :, :], x[slot, :, :], y)
                        elif op == "mul":
                            T.tile.mul(x[slot, :, :], x[slot, :, :], y)
                        elif op == "div":
                            T.tile.div(x[slot, :, :], x[slot, :, :], y)
                        elif op == "exp":
                            T.tile.exp(x[slot, :, :], y)
                        elif op == "fill":
                            T.tile.fill(x[slot, :, :], 0.25)
                        elif op == "axpy":
                            T.tile.axpy(x[slot, :, :], y, 0.125)
                        elif op == "axpy_mixed":
                            T.tile.axpy(x[slot, :, :], y, 1.0)
                        elif op in ("cast_f16_f32", "cast_f32_f16"):
                            T.tile.cast(x[slot, :, :], y, "CAST_NONE", rows * cols)
                        elif op == "muls":
                            T.tile.mul(x[slot, :, :], x[slot, :, :], 0.999)
                        elif op == "max":
                            T.tile.max(x[slot, :, :], x[slot, :, :], y)
                        elif op == "mul_add_dst":
                            T.tile.mul_add_dst(x[slot, :, :], y, w)
                        elif op == "copy_ub_ub":
                            T.copy(y, x[slot, :, :])
                        elif op == "brcb_experiment":
                            T.tile.brcb_experiment(x[slot, :rows // 8, :], coeff_in, rows // 8, 1,
                                                   8)
                        elif op == "row_expand_sub_experiment":
                            T.tile.row_expand_sub_experiment(x[slot, :, :64], x[slot, :, :64],
                                                             coeff)
                        elif op == "row_expand_mul_experiment":
                            T.tile.row_expand_mul_experiment(x[slot, :, :64], x[slot, :, :64],
                                                             coeff)
                        elif op == "row_expand_div_experiment":
                            T.tile.row_expand_div_experiment(x[slot, :, :64], x[slot, :, :64],
                                                             coeff)
                        elif op == "copy_gm_ub":
                            T.copy(a[vid, slot, :, :], x[slot, :, :])
                        elif op == "copy_ub_gm":
                            T.copy(x[slot, :, :], out[vid, slot, :, :])
                        elif op == "reduce_sum":
                            if slot == 0:
                                T.reduce_sum(
                                    x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp0)
                            else:
                                T.reduce_sum(
                                    x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp1)
                        else:
                            if slot == 0:
                                T.reduce_max(
                                    x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp0)
                            else:
                                T.reduce_max(
                                    x[slot, :, :], z[slot, :], dim=-1, clear=True, tmp=tmp1)
                    if chain != 0:
                        if op == "copy_gm_ub":
                            T.pipe_barrier("MTE2")
                        elif op == "copy_ub_gm":
                            T.pipe_barrier("MTE3")
                        else:
                            T.pipe_barrier("V")
                T.barrier_all()
                _src_code("int64_t bench_after = AscendC::GetSystemCycle();")
                T.set_flag("V", "MTE3", 0)
                T.wait_flag("V", "MTE3", 0)
                for slot in T.unroll(2):
                    if op not in ("reduce_sum", "reduce_max"):
                        T.copy(x[slot, :, :], out[vid, slot, :, :])
                    else:
                        T.copy(z[slot, :rows], out[vid, slot, 0, :rows])
                _src_code(
                    "stamp.SetValue(vid * 8, bench_before); stamp.SetValue(vid * 8 + 1, bench_after);"
                )
                stamp[vid, 2] = T.cast(count, "int64")
                _src_code("AscendC::DataCacheCleanAndInvalid<int64_t, "
                          "AscendC::CacheLine::SINGLE_CACHE_LINE, "
                          "AscendC::DcciDst::CACHELINE_OUT>(stamp[vid * 8]);")

    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument(
        "--suite",
        choices=("initial", "linear", "copy", "vector", "cube_copy", "fa_cube", "vector_small",
                 "row_helpers", "cast", "fa_extra"),
        default="initial")
    parser.add_argument("--ops", default="all")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--counts", default="0,16,64,256")
    args = parser.parse_args()
    if args.dtype != "float32" and args.suite != "copy":
        parser.error("--dtype currently applies to the copy suite only")
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(set(counts)) < 2 or 0 not in counts or min(counts) < 0:
        parser.error("counts must include zero and at least one distinct positive count")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)  # ASCEND_RT_VISIBLE_DEVICES=6 is set by the launcher.
    torch.manual_seed(20261021)
    cases = [("mma", 128, 128, 128), ("mma", 128, 128, 256), ("mma", 64, 64, 512),
             ("add", 16, 512, 0), ("reduce_sum", 16, 512, 0), ("reduce_max", 16, 512, 0)]
    if args.suite == "linear":
        # Freeze shape-level train/holdout membership BEFORE collecting data.
        # Equal-MNK pairs deliberately test whether geometry needs a separate family.
        cases = [("mma", *shape)
                 for shape in ((64, 64, 64), (64, 64, 128), (64, 64, 256), (128, 128,
                                                                            64), (128, 128, 128),
                               (128, 128, 256), (64, 64, 512), (64, 128, 128), (128, 64, 256))]
        cases += [(op, rows, cols, 0)
                  for op in ("add", "reduce_sum", "reduce_max")
                  for rows, cols in ((4, 512), (8, 512), (12, 512), (16, 512), (24, 512), (16, 128),
                                     (16, 256))]
    if args.suite == "copy":
        cases = [
            (op, rows, 512, 0) for op in ("copy_gm_ub", "copy_ub_gm") for rows in (4, 8, 12, 16, 24)
        ]
    if args.suite == "vector":
        cases = [(op, rows, 512, 0)
                 for op in ("sub", "mul", "div", "exp", "fill", "axpy", "muls")
                 for rows in (4, 8, 12, 16, 24)]
    if args.suite == "vector_small":
        cases = [(op, 1, cols, 0)
                 for op in ("sub", "exp", "fill", "max", "muls", "mul_add_dst", "copy_ub_ub")
                 for cols in (16, 32, 64, 128, 256, 512)]
        cases += [("axpy_mixed", rows, 512, 0) for rows in (4, 8, 12, 16, 24)]
    if args.suite == "row_helpers":
        cases = [("brcb_experiment", rows, 64, 0) for rows in (8, 16, 32, 48, 64)]
        cases += [("row_expand_sub_experiment", rows, 512, 0) for rows in (8, 16, 24, 32)]
        cases += [(op, rows, 128, 0)
                  for op in ("row_expand_mul_experiment", "row_expand_div_experiment")
                  for rows in (16, 32, 48, 64)]
    if args.suite == "cast":
        cases = [(op, rows, 512, 0)
                 for op in ("cast_f16_f32", "cast_f32_f16")
                 for rows in (4, 8, 12, 16, 24)]
    cube_ops = {
        "mma", "mma_init", "copy_gm_l1", "copy_l1_l0a", "copy_l1_l0b", "copy_l1_l0b_transpose",
        "copy_l0c_gm", "copy_l0c_gm_half", "copy_l0c_gm_half_strided"
    }
    if args.suite == "cube_copy":
        cases = [(op, *shape)
                 for op in ("copy_gm_l1", "copy_l1_l0a", "copy_l1_l0b", "copy_l0c_gm")
                 for shape in ((32, 32, 128), (64, 64, 128), (96, 96, 128), (128, 128, 128),
                               (64, 64, 256))]
    if args.suite == "fa_cube":
        cases = [(op, m, 128, 128)
                 for op in ("copy_l0c_gm", "copy_l0c_gm_half", "mma_init")
                 for m in (32, 64, 80, 96, 128)]
        cases += [(op, 256, 64, 128) for op in ("copy_gm_l1", "copy_l1_l0a", "copy_l1_l0b")]
    if args.suite == "fa_extra":
        cases = [("copy_l0c_gm_half_strided", m, 256, 128) for m in (32, 64, 80, 96, 128)]
        cases += [("copy_l1_l0b_transpose", 64, size, size) for size in (32, 64, 96, 128)]
    held_out = {"mma_64_64_256", "mma_64_64_512", "mma_128_64_256"}
    held_out |= {
        f"{op}_{rows}_{cols}_0" for op in ("add", "reduce_sum", "reduce_max")
        for rows, cols in ((12, 512), (16, 128), (16, 256))
    }
    held_out |= {f"{op}_12_512_0" for op in ("copy_gm_ub", "copy_ub_gm")}
    held_out |= {f"{op}_12_512_0" for op in ("sub", "mul", "div", "exp", "fill", "axpy")}
    held_out.add("muls_12_512_0")
    held_out |= {f"{op}_96_96_128" for op in cube_ops}
    held_out |= {f"{op}_80_128_128" for op in cube_ops}
    held_out |= {"copy_l0c_gm_half_strided_80_256_128", "copy_l1_l0b_transpose_64_96_96"}
    held_out |= {
        f"{op}_1_{cols}_0" for op in ("sub", "exp", "fill", "max", "muls", "mul_add_dst",
                                      "copy_ub_ub") for cols in (64, 256)
    }
    held_out.add("axpy_mixed_12_512_0")
    held_out |= {"cast_f16_f32_12_512_0", "cast_f32_f16_12_512_0"}
    held_out |= {
        "brcb_experiment_48_64_0", "row_expand_sub_experiment_24_512_0",
        "row_expand_mul_experiment_48_128_0", "row_expand_div_experiment_48_128_0"
    }
    if not any((args.ops == "all" or op in args.ops.split(",")) and args.case in
               {"all", f"{op}_{m}_{n}_{k}" + (f"_{args.dtype}" if args.dtype != "float32" else "")}
               for op, m, n, k in cases):
        parser.error("no case matches the requested suite/ops/case/dtype")
    manifest = {
        "physical_device": 6,
        "counter_mhz": 50,
        "warmup": 3,
        "samples": 5,
        "counts": counts,
        "python_package": tilelang.__file__,
        "native": tilelang._LIB_PATH,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(Path(tilelang._LIB_PATH).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "pass_configs": {
            str(k): v for k, v in PASSES.items()
        },
        "scope": "resident-data repeated operations with matched empty-loop/drain controls",
        "not_bare_instruction_latency": True,
        "suite": args.suite,
        "held_out": sorted(held_out),
        "a3_hardware_validated": False
    }
    if args.suite in ("copy", "cube_copy"):
        manifest["copy_cache_condition"] = ("warm repeated addresses, two AIV lanes, one core"
                                            if args.suite == "copy" else
                                            "warm repeated addresses, one AIC, one core")
    (root / "bench_source.py").write_bytes(Path(__file__).read_bytes())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for op, m, n, k in cases:
        if args.ops != "all" and op not in args.ops.split(","):
            continue
        base_name = f"{op}_{m}_{n}_{k}"
        name = base_name + (f"_{args.dtype}" if args.dtype != "float32" else "")
        if args.case not in {"all", name}:
            continue
        begin = time.monotonic()
        fn = cube(m, n, k, op) if op in cube_ops else vector(op, m, n, args.dtype)
        kernel = tilelang.compile(
            fn.with_attr("global_symbol", name),
            out_idx=[2, 3],
            target="ascendc",
            platform="A2",
            pass_configs=PASSES,
            compile_flags=["--cce-auto-sync=off", "-O3"])
        source = kernel.get_kernel_source()
        (root / f"{name}.cpp").write_text(source)
        assert source.count("AscendC::GetSystemCycle()") == 2
        assert "count;" in source and "enabled" in source
        print("COMPILED", name, time.monotonic() - begin, flush=True)
        if op in cube_ops:
            a = (torch.randn((m, k)) * 0.01).half().npu()
            b = (torch.randn((k, n)) * 0.01).half().npu()
            golden = a.cpu().float() @ b.cpu().float()
            transposed_golden = a.cpu().float() @ b.cpu().float(
            ).T if op == "copy_l1_l0b_transpose" else None
            dtype = ("float32_to_float16" if op in {"copy_l0c_gm_half", "copy_l0c_gm_half_strided"}
                     else "float32" if op == "copy_l0c_gm" else "float16")
        else:
            a = (torch.randn((2, 2, m, n)) * 0.1).to(getattr(torch, args.dtype)).npu()
            b = (torch.rand((2, m, n)) * 0.001 + 0.001).to(getattr(torch, args.dtype)).npu()
            if op in ("mul", "div", "row_expand_mul_experiment", "row_expand_div_experiment"):
                # Avoid overflow/underflow in a 256-update dependent chain.
                b = (torch.rand((2, m, n)) * 0.0001 + 1).npu()
            if op in ("axpy_mixed", "cast_f16_f32"):
                b = b.half()
            if op == "cast_f32_f16":
                a = a.half()
            golden = a.cpu()
            dtype = ("float16_to_float32" if op in ("axpy_mixed", "cast_f16_f32") else
                     "float32_to_float16" if op == "cast_f32_f16" else args.dtype)
        samples = []
        for chain in (0, 1):
            for count in counts:
                for enabled in (0, 1):
                    for iteration in range(8):
                        out, stamp = kernel(a, b, count, chain, enabled)
                        torch.npu.synchronize()
                        ticks = stamp.cpu().reshape(-1, 8).tolist()
                        assert all(t[1] >= t[0] and t[2] == count for t in ticks)
                        if iteration == 0:
                            actual = out.cpu()
                            for slot in (0, 1):
                                steps = (count if slot == 0 else 0) if chain else (
                                    (count + 1 - slot) // 2)
                                steps *= enabled
                                if op in cube_ops:
                                    expected = golden * (1 + steps if op == "mma" else 1)
                                    if op == "copy_l1_l0b_transpose" and enabled and count:
                                        expected = transposed_golden
                                    got = actual[slot, :, :n]
                                elif op == "add":
                                    expected = golden[:, slot] + steps * b.cpu()
                                    got = actual[:, slot]
                                elif op == "axpy":
                                    expected = golden[:, slot] + steps * b.cpu() * 0.125
                                    got = actual[:, slot]
                                elif op in ("axpy_mixed", "muls", "max", "mul_add_dst",
                                            "copy_ub_ub"):
                                    values = b.cpu().float()
                                    if op == "axpy_mixed":
                                        expected = golden[:, slot] + steps * values
                                    elif op == "muls":
                                        expected = golden[:, slot] * (0.999**steps)
                                    elif op == "max":
                                        expected = torch.maximum(
                                            golden[:, slot], values) if steps else golden[:, slot]
                                    elif op == "mul_add_dst":
                                        expected = golden[:, slot] + steps * values * values
                                    else:
                                        expected = values if steps else golden[:, slot]
                                    got = actual[:, slot]
                                elif op in ("sub", "mul", "div"):
                                    values = b.cpu()
                                    expected = (
                                        golden[:, slot] -
                                        steps * values if op == "sub" else golden[:, slot] *
                                        values.pow(steps) if op == "mul" else golden[:, slot] /
                                        values.pow(steps))
                                    got = actual[:, slot]
                                elif op in ("exp", "fill"):
                                    values = b.cpu()
                                    computed = {
                                        "exp": values.exp(),
                                        "fill": torch.full_like(values, 0.25)
                                    }[op]
                                    expected = computed if steps else golden[:, slot]
                                    got = actual[:, slot]
                                elif op in ("cast_f16_f32", "cast_f32_f16"):
                                    converted = b.cpu().to(golden.dtype)
                                    expected = converted if steps else golden[:, slot]
                                    got = actual[:, slot]
                                elif op in ("brcb_experiment", "row_expand_sub_experiment",
                                            "row_expand_mul_experiment",
                                            "row_expand_div_experiment"):
                                    values = b.cpu()[:, 0, :m]
                                    expected = golden[:, slot].clone()
                                    if op == "brcb_experiment":
                                        if steps:
                                            expected.reshape(2, -1)[:, :m *
                                                                    8] = values.repeat_interleave(
                                                                        8, dim=-1)
                                    elif op == "row_expand_sub_experiment":
                                        expected[:, :, :64] -= steps * values[:, :, None]
                                    elif op == "row_expand_mul_experiment":
                                        expected[:, :, :64] *= values[:, :, None].pow(steps)
                                    else:
                                        expected[:, :, :64] /= values[:, :, None].pow(steps)
                                    got = actual[:, slot]
                                elif op in ("copy_gm_ub", "copy_ub_gm"):
                                    expected = golden[:, slot]
                                    got = actual[:, slot]
                                else:
                                    values = golden[:, slot]
                                    expected = (
                                        values.sum(-1)
                                        if op == "reduce_sum" else values.max(-1).values)
                                    got = actual[:, slot, 0, :m]
                                assert torch.isfinite(got).all()
                                assert torch.allclose(
                                    got.float(), expected.float(), rtol=2e-3, atol=2e-4), name
                        if iteration >= 3:
                            samples.append({
                                "chain": chain,
                                "count": count,
                                "enabled": enabled,
                                "sample": iteration - 3,
                                "ticks": ticks,
                                "ns": [(t[1] - t[0]) * 20 for t in ticks]
                            })
        record = {
            "name": name,
            "op": op,
            "shape": [m, n, k],
            "split": "holdout" if base_name in held_out else "train",
            "dtype": dtype,
            "operand_pattern":
                ("inplace_distinct_rhs" if op in ("add", "sub", "mul", "div", "axpy") else "other"),
            "samples": samples,
            "compile_s": time.monotonic() - begin,
            "correctness": "passed"
        }
        if args.suite == "fa_cube":
            record["name"] = "fa_cube:" + name
        if op in cube_ops:
            record["l0c_source_slots"] = 1 if m * n * 4 > 65536 else 2
            record["gm_destination_cols"] = n * 2 if op.endswith("_strided") else n
        if op == "axpy_mixed":
            record["op"] = "axpy"
        if op in ("cast_f16_f32", "cast_f32_f16"):
            record["op"] = "cast"
        if op in ("copy_l0c_gm", "copy_l0c_gm_half", "copy_l0c_gm_half_strided"):
            record["op"] = "copy_l0c_gm"
            record["geometry"] = (f"row_elements_{n}_stride_elements_{2 * n}"
                                  if op.endswith("_strided") else f"row_elements_{n}")
        if op in ("copy_gm_l1", "copy_l1_l0a"):
            record["timing_shape"] = [m, k]
        elif op in ("copy_l1_l0b", "copy_l1_l0b_transpose"):
            record["timing_shape"] = [k, n]
        elif op == "brcb_experiment":
            record["timing_shape"] = [m]
        elif op.startswith("row_expand_"):
            record["timing_shape"] = [m, 64]
            record["geometry"] = f"row_stride_bytes_{n * 4}"
        for chain in (0, 1):
            deltas = []
            for count in counts:
                values = {}
                for enabled in (0, 1):
                    values[enabled] = median(
                        t for s in samples
                        if (s["chain"], s["count"], s["enabled"]) == (chain, count, enabled)
                        for t in s["ns"])
                deltas.append(values[1] - values[0])
            slope, intercept = np.polyfit(counts, deltas, 1)
            predicted = slope * np.array(counts) + intercept
            residual = sum((np.array(deltas) - predicted)**2)
            total = sum((np.array(deltas) - np.mean(deltas))**2)
            record["chain" if chain else "two_slot"] = {
                "net_ns": deltas,
                "ns_per_op": float(slope),
                "intercept_ns": float(intercept),
                "r_squared": float(1 - residual / total) if total else None
            }
            if 1 in counts:
                record["chain" if chain else "two_slot"]["single_op_net_completion_ns"] = (
                    deltas[counts.index(1)] - deltas[counts.index(0)])
                record["chain" if chain else "two_slot"]["single_op_counter_tick_ns"] = 20
        results.append(record)
        (root / "observations.json").write_text(json.dumps(results, indent=2))
        print(
            "MEASURED",
            json.dumps({
                key: value for key, value in record.items() if key != "samples"
            }),
            flush=True)


if __name__ == "__main__":
    main()
