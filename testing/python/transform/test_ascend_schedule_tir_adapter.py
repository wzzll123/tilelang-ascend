# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Extraction contracts; real old-TVM tests skip if the runtime is unavailable."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_tir_adapter_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "tilelang/transform/ascend_schedule")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".tir_adapter", ROOT / "tilelang/transform/ascend_schedule/tir_adapter.py")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


class SourceContracts(unittest.TestCase):

    def test_parse_repository_operation_configuration(self):
        configs = adapter.load_operation_configs()
        self.assertEqual(configs["copy_gm_to_l1"].pipe, "PIPE_MTE2")
        self.assertEqual(configs["AscendC::Add"].accesses, ((0, "W"), (1, "R"), (2, "R")))

    def test_template_normalization_preserves_unknown_namespace(self):
        self.assertEqual(adapter.normalize_operation("tl::ascend::mma<float>"), "mma")
        self.assertEqual(adapter.normalize_operation("AscendC::Add<float>"), "AscendC::Add")
        self.assertEqual(adapter.normalize_operation("foreign::Add<float>"), "foreign::Add")

    def test_multi_pipe_gemm_and_unknown_operations_fail_closed(self):
        for name in ("gemm_v0", "gemm_v1", "tl.ascend_gemm_v0", "foreign::Add"):
            with self.subTest(name=name), self.assertRaises(adapter.ExtractionError):
                adapter.operation_config(name)

    def test_storage_interning_uses_same_as_not_wrapper_identity(self):

        class Var:

            def __init__(self, handle):
                self.handle = handle

            def same_as(self, other):
                return self.handle == other.handle

        pool = adapter.StorageInterner()
        first = pool.intern(Var(1))
        self.assertIs(first, pool.intern(Var(1)))
        self.assertIsNot(first, pool.intern(Var(2)))

    def test_dtype_byte_sizes_and_masks(self):
        self.assertEqual(adapter.dtype_bytes("float16"), 2)
        self.assertEqual(adapter.dtype_bytes("float32x4"), 16)
        self.assertIsNone(adapter.dtype_bytes("int4"))
        self.assertEqual([adapter.mask_mode(value) for value in (1, 2, 3)], ["R", "W", "RW"])
        for value in (0, 4, None):
            with self.assertRaises(adapter.ExtractionError):
                adapter.mask_mode(value)


class RealTIRContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from tvm import tir
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Old TVM unavailable: {error}") from error
        cls.tir = tir

    def pointer(self, var, offset=0, extent=8, mask=1):
        T = self.tir
        return T.tvm_access_ptr(T.type_annotation("float16"), var, offset, extent, mask)

    def test_rw_bytes_storage_alias_and_stage(self):
        T = self.tir
        a, b = T.Var("a", "handle"), T.Var("b", "handle")
        call = T.Evaluate(
            T.call_extern("handle", "AscendC::Add<float>", self.pointer(a, 2, 8, 3),
                          self.pointer(b), self.pointer(b)))
        result = adapter.extract_tasks(T.AttrStmt(T.IntImm("int64", 4), "tl.ascend_stage", 1, call))
        task = result.tasks[0]
        self.assertEqual((task.pipe, task.core), ("PIPE_V", "V"))
        self.assertEqual((task.accesses[0].mode, task.accesses[0].offset, task.accesses[0].extent),
                         ("RW", 4, 16))
        self.assertIs(task.accesses[1].storage, task.accesses[2].storage)
        self.assertEqual(result.stages[task.id], 4)

    def test_guard_snapshot_precedes_write_and_preserves_scalar_binding(self):
        T = self.tir
        buffer = T.decl_buffer((8,), "int32", name="state")
        scalar = T.Var("scalar", "int32")
        body = T.LetStmt(
            scalar, buffer[0],
            T.IfThenElse(scalar > 0, T.BufferStore(buffer, T.IntImm("int32", 0), [0]), None))
        result = adapter.extract_tasks(body)
        self.assertEqual(len(result.scalar_bindings), 2)
        self.assertEqual(len(result.tasks), 3)
        self.assertTrue(result.tasks[-1].guard)
        self.assertEqual(result.tasks[0].accesses[0].mode, "R")
        writes = [access for access in result.tasks[-1].accesses if access.writes]
        self.assertEqual(len(writes), 1)
        self.assertIs(writes[0].storage, result.tasks[0].accesses[0].storage)
        self.assertEqual(result.tasks[-1].accesses[-1].mode, "R")  # Guard snapshot read.

    def test_symbolic_pointer_footprint_is_unknown(self):
        T = self.tir
        var, offset = T.Var("buffer", "handle"), T.Var("offset", "int32")
        call = T.Evaluate(
            T.call_extern("handle", "AscendC::Duplicate<float>", self.pointer(var, offset, 8, 2), 0,
                          8))
        result = adapter.extract_tasks(call)
        self.assertIsNone(result.tasks[0].accesses[0].offset)
        self.assertIsNone(result.tasks[0].accesses[0].extent)

    def test_unknown_extern_rejected(self):
        T = self.tir
        with self.assertRaises(adapter.ExtractionError):
            adapter.extract_tasks(T.Evaluate(T.call_extern("handle", "unverified_op")))

    def test_additional_scratch_pointer_is_not_ignored(self):
        T = self.tir
        output, scratch = T.Var("out", "handle"), T.Var("scratch", "handle")
        call = T.Evaluate(
            T.call_extern("handle", "AscendC::Duplicate<float>", self.pointer(output, mask=2), 0, 8,
                          self.pointer(scratch, mask=3)))
        result = adapter.extract_tasks(call)
        self.assertEqual([access.mode for access in result.tasks[0].accesses], ["W", "RW"])

    def test_real_vector_intrinsic_primfunc(self):
        import tilelang.language as L
        T = self.tir
        dst = T.decl_buffer((8,), "float16", name="dst", scope="shared")
        a = T.decl_buffer((8,), "float16", name="a", scope="shared")
        b = T.decl_buffer((8,), "float16", name="b", scope="shared")
        call = L.tile.add(dst, a, b)
        self.assertEqual(call.op.name, "tl.ascend_add")
        self.assertEqual(len(call.args), 4)  # No extern-name/template offset.
        result = adapter.extract_tasks(T.PrimFunc([dst.data, a.data, b.data], T.Evaluate(call)))
        self.assertEqual((result.tasks[0].pipe, result.tasks[0].core), ("PIPE_V", "V"))
        self.assertEqual([access.mode for access in result.tasks[0].accesses], ["W", "R", "R"])

    def test_real_template_intrinsic_argument_index(self):
        import tilelang.language as L
        T = self.tir
        dst = T.decl_buffer((8,), "float16", name="dst", scope="shared")
        call = L.tile.fill(dst, 1)
        self.assertEqual(call.op.name, "tl.ascend_fill")
        self.assertIsInstance(call.args[0], T.StringImm)
        result = adapter.extract_tasks(T.PrimFunc([dst.data], T.Evaluate(call)))
        self.assertEqual(len(result.tasks[0].accesses), 1)
        self.assertEqual(result.tasks[0].accesses[0].mode, "W")

    def test_builtin_multipipe_gemm_diagnostic(self):
        import tilelang  # noqa: F401 - Registers builtin Ops in the native library.
        T = self.tir
        for op_name in ("tl.ascend_gemm_v0", "tl.ascend_gemm_v1"):
            with self.subTest(op_name=op_name), \
                    self.assertRaisesRegex(adapter.ExtractionError, "multi-pipe.*decompose"):
                call = T.call_intrin("handle", T.op.Op.get(op_name), *[T.IntImm("int32", 0)] * 10)
                adapter.extract_tasks(T.Evaluate(call))


if __name__ == "__main__":
    unittest.main()
