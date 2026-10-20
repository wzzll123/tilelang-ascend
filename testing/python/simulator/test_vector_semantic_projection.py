# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native, CPU-only compatibility tests for compiler-selected Vector IR."""

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")

import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402
from tilelang.engine.lower import lower_ascend_ir  # noqa: E402
from tilelang.simulator import Lane, Pipe, build_kernel_program  # noqa: E402


TARGET = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
PASS_CONFIGS = {"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True}


def _kernel():
    @T.prim_func
    def main(source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True):
            a = T.alloc_ub((128,), "float32")
            b = T.alloc_ub((128,), "float32")
            with T.Scope("V"):
                T.copy(source, a)
                T.copy(a, b)
                T.tile.add(b, a, b)
                T.copy(b, output)

    return main


def _lower(platform="A2"):
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        return lower_ascend_ir(_kernel(), target=TARGET, platform=platform)[0]


def _project(module, platform="A2"):
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        return tilelang.transform.AscendVectorSemanticProjection(TARGET, platform)(module)


def _op_names(module):
    names = set()

    def collect(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.add(node.op.name)

    for _, func in module.functions_items():
        tvm.tir.stmt_functor.post_order_visit(func.body, collect)
    return names


def _rewrite(module, mutate):
    result = tvm.IRModule(module.functions)
    for name, func in module.functions_items():
        body = tvm.tir.stmt_functor.ir_transform(func.body, None, mutate, ["tir.Evaluate"])
        result[name] = func.with_body(body)
    return result


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_projection_preserves_sync_and_routes_ub_copy_to_vector(platform):
    final = _lower(platform)
    projected = _project(final, platform)
    assert "tl.ascend_add_raw" in _op_names(final)
    assert "tl.ascend_add_raw" not in _op_names(projected)
    assert not any(name.startswith("tl.ascend_set_mask_") for name in _op_names(projected))
    original_program = build_kernel_program(projected, platform=platform)
    copies = [task for task in original_program.tasks if task.operation == "copy_ub_to_ub"]
    assert copies and all(task.pipe is Pipe.VECTOR and task.lane is Lane.VECTOR_0 for task in copies)
    assert any(task.operation in {"set_flag", "auto_set_flag"} for task in original_program.tasks)


def test_projection_rejects_missing_mask_repair():
    def remove_setter(stmt):
        value = stmt.value
        if isinstance(value, tvm.tir.Call) and value.op.name.startswith("tl.ascend_set_mask_"):
            return tvm.tir.Evaluate(0)
        return None

    with pytest.raises(tvm.error.TVMError, match="canonical|round trip"):
        _project(_rewrite(_lower(), remove_setter))


def test_projection_rejects_changed_terminal_payload():
    def change_payload(stmt):
        value = stmt.value
        if isinstance(value, tvm.tir.Call) and value.op.name == "tl.ascend_add_raw":
            args = list(value.args)
            # Float32 NORMAL masks use the low 64-bit word; the high word is
            # already zero, so mutating it to zero would not be a negative.
            args[-2] = tvm.tir.const(0, "uint64")
            return tvm.tir.Evaluate(tvm.tir.Call(value.dtype, value.op, args))
        return None

    with pytest.raises(tvm.error.TVMError, match="payload|canonical|round trip"):
        _project(_rewrite(_lower(), change_payload))


def test_projection_rejects_non_integer_mask_payload():
    def change_payload(stmt):
        value = stmt.value
        if isinstance(value, tvm.tir.Call) and value.op.name == "tl.ascend_set_mask_payload":
            args = [tvm.tir.const(1.0, "float32"), value.args[1]]
            return tvm.tir.Evaluate(tvm.tir.Call(value.dtype, value.op, args))
        return None

    with pytest.raises(tvm.error.TVMError, match="payload must be a scalar integer"):
        _project(_rewrite(_lower(), change_payload))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_selected_vector_compiles_and_executes_without_npu(platform, tmp_path):
    kernel = tilelang.compile(
        _kernel(),
        out_idx=[1],
        target=TARGET,
        platform=platform,
        simulator=True,
        sim_config={"trace_path": tmp_path / "trace.json"},
        pass_configs=PASS_CONFIGS,
    )
    source = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(kernel(source), source * 2)
    report = kernel.get_simulator_report()
    assert report.timing_limitations
    assert report.to_dict()["timing"]["limitations"]
    assert "mask setter timing" in (tmp_path / "trace.json").read_text()


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("sync_only", [False, True])
def test_reduce2d_projection_preserves_clear_false(platform, sync_only):
    @T.prim_func
    def main(
        source: T.Tensor((2, 128), "float32"),
        output: T.Tensor((2, 1), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            values = T.alloc_ub((2, 128), "float32")
            rows = T.alloc_ub((2, 1), "float32")
            with T.Scope("V"):
                T.copy(source, values)
                T.tile.fill(rows, 2.0)
                T.reduce_sum(values, rows, dim=-1, clear=False)
                T.copy(rows, output)

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target=TARGET,
        platform=platform,
        simulator=True,
        sim_config={"sync_only": sync_only},
        pass_configs=PASS_CONFIGS,
    )
    assert "tl.ascend_reduce_fp32_v2" in _op_names(kernel.adapter.optimized_mod)
    source = np.arange(256, dtype=np.float32).reshape(2, 128)
    result = kernel(source)
    if sync_only:
        assert result is None
    else:
        np.testing.assert_array_equal(result, source.sum(axis=1, keepdims=True) + 2.0)
    assert kernel.get_simulator_report().sync_only is sync_only
