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

    def test_cross_core_gm_hop_composites_require_decomposition(self):
        for name in ("copy_ub_to_l1", "copy_l0c_to_ub"):
            with self.subTest(name=name), \
                    self.assertRaisesRegex(adapter.ExtractionError, "cross-core.*decompose"):
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
        from importlib import import_module
        normalize = import_module(PACKAGE + ".conditions").normalize_conditions
        result = adapter.extract_tasks(
            T.AttrStmt(T.IntImm("int32", 0), "resource_scope", 0, normalize(body)))
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

    def test_copy_logical_extent_does_not_hide_second_physical_row(self):
        from importlib import import_module
        dependencies = import_module(PACKAGE + ".dependency")
        T = self.tir
        gm = T.decl_buffer((2, 16), "float16", name="gm")
        ub = T.decl_buffer((2, 16), "float16", name="ub", scope="shared")
        scalar = T.decl_buffer((1,), "float16", name="scalar", scope="local")
        # src/op/ascend.cc CopyNode::Lower supplies product(region extents)=8,
        # although a 2x4 region in 16-column storage touches elements 16..19.
        copy = T.Evaluate(
            T.call_extern("handle", "tl::ascend::copy_gm_to_ub<half,16,2>",
                          self.pointer(gm.data, 0, 8, 1), self.pointer(ub.data, 0, 8, 2), 16, 2, 4,
                          T.FloatImm("float16", 0), 2, 16))
        later_read = T.BufferStore(scalar, ub[1, 0], [0])
        result = adapter.extract_tasks(T.SeqStmt([copy, later_read]))
        copy_write = [access for access in result.tasks[0].accesses if access.writes][0]
        self.assertIsNone(copy_write.offset)
        self.assertIsNone(copy_write.extent)
        self.assertEqual(copy_write.mode, "W")
        precise_read = result.tasks[1].accesses[0]
        self.assertEqual((precise_read.offset, precise_read.extent), (32, 2))
        hazards = dependencies.analyze_dependencies(result.tasks, max_distance=0)
        self.assertTrue((0, 1,
                         "RAW") in [(edge.source, edge.target, edge.kind) for edge in hazards])
        reversed_result = adapter.extract_tasks(T.SeqStmt([later_read, copy]))
        hazards = dependencies.analyze_dependencies(reversed_result.tasks, max_distance=0)
        self.assertTrue((0, 1,
                         "WAR") in [(edge.source, edge.target, edge.kind) for edge in hazards])

    def test_builtin_mma_tail_logical_counts_preserve_hazards(self):
        from importlib import import_module
        import tilelang  # noqa: F401 - Registers tl.ascend_mma in the native library.
        dependencies = import_module(PACKAGE + ".dependency")
        T = self.tir
        a = T.decl_buffer((256,), "float16", name="a", scope="shared.l0a")
        b = T.decl_buffer((256,), "float16", name="b", scope="shared.l0b")
        c = T.decl_buffer((256,), "float32", name="c", scope="shared.l0c")
        # MmaCodegen reads template/ptr args 0..3 and runtime init/K args 4..5.
        # Logical 16x5 counts cannot prove padded/fractal physical read spans.
        for k_actual in (1, 5, 15):
            mma = T.Evaluate(
                T.call_intrin("handle", T.op.Op.get("tl.ascend_mma"), "mma<half,float,16,16>",
                              self.pointer(a.data, 0, 80, 1), self.pointer(b.data, 0, 80, 1),
                              T.tvm_access_ptr(T.type_annotation("float32"), c.data, 0, 256, 3),
                              T.IntImm("bool", 0), k_actual))
            prior_write = T.BufferStore(a, T.FloatImm("float16", 1), [80])
            result = adapter.extract_tasks(T.SeqStmt([prior_write, mma]))
            matrix = result.tasks[1]
            self.assertEqual((matrix.pipe, matrix.core), ("PIPE_M", "C"))
            self.assertEqual([access.mode for access in matrix.accesses], ["R", "R", "RW"])
            self.assertTrue(all(access.footprint() is None for access in matrix.accesses))
            self.assertIs(matrix.accesses[0].storage, result.tasks[0].accesses[0].storage)
            hazards = dependencies.analyze_dependencies(result.tasks, max_distance=0)
            self.assertTrue((0, 1,
                             "RAW") in [(edge.source, edge.target, edge.kind) for edge in hazards])
            # A following write must also retain WAR on the matrix's reads.
            result = adapter.extract_tasks(T.SeqStmt([mma, prior_write]))
            hazards = dependencies.analyze_dependencies(result.tasks, max_distance=0)
            self.assertTrue((0, 1,
                             "WAR") in [(edge.source, edge.target, edge.kind) for edge in hazards])

    def test_vector_pipe_ub_copy_is_also_conservative(self):
        T = self.tir
        source, destination = T.Var("src", "handle"), T.Var("dst", "handle")
        copy = T.Evaluate(
            T.call_extern("handle", "copy_ub_to_ub", self.pointer(source),
                          self.pointer(destination, mask=3)))
        result = adapter.extract_tasks(copy)
        self.assertEqual(result.tasks[0].pipe, "PIPE_V")
        self.assertEqual([access.mode for access in result.tasks[0].accesses], ["R", "RW"])
        self.assertTrue(all(access.footprint() is None for access in result.tasks[0].accesses))

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

    def test_unscoped_scalar_producer_resolves_to_single_operation_core(self):
        T = self.tir
        scalar = T.Var("count", "int32")
        source, destination = T.Var("source", "handle"), T.Var("destination", "handle")
        for name, core in (("copy_gm_to_l1", "C"), ("AscendC::Duplicate<float>", "V")):
            args = ([self.pointer(source),
                     self.pointer(destination, mask=2), scalar]
                    if core == "C" else [self.pointer(destination, mask=2), 0, scalar])
            call = T.Evaluate(T.call_extern("handle", name, *args))
            result = adapter.extract_tasks(T.LetStmt(scalar, T.IntImm("int32", 8), call))
            self.assertEqual([task.core for task in result.tasks], [core, core])
            scalar_write = result.scalar_bindings[0].storage
            self.assertEqual(scalar_write.domain, core)
            self.assertTrue(
                any(access.storage is scalar_write and access.reads
                    for access in result.tasks[1].accesses))

    def test_mixed_core_unscoped_scalar_controls_require_normalization(self):
        T = self.tir
        scalar = T.Var("count", "int32")
        source, destination = T.Var("source", "handle"), T.Var("destination", "handle")
        c = T.Evaluate(
            T.call_extern("handle", "copy_gm_to_l1", self.pointer(source),
                          self.pointer(destination, mask=2), scalar))
        v = T.Evaluate(
            T.call_extern("handle", "AscendC::Duplicate<float>", self.pointer(destination, mask=2),
                          0, scalar))
        body = T.LetStmt(scalar, T.IntImm("int32", 8), T.SeqStmt([c, v]))
        with self.assertRaisesRegex(adapter.ExtractionError, "per-core control normalization"):
            adapter.extract_tasks(body)

    def test_scoped_mixed_scalar_storage_is_core_private(self):
        T = self.tir
        scalar = T.Var("same_scalar", "int32")
        local = T.decl_buffer((1,), "int32", name="same_local", scope="local")
        children = []
        for scope in (0, 1):
            binding = T.LetStmt(scalar, T.IntImm("int32", scope), T.BufferStore(local, scalar, [0]))
            children.append(T.AttrStmt(T.IntImm("int32", 0), "resource_scope", scope, binding))
        result = adapter.extract_tasks(T.SeqStmt(children))
        self.assertEqual([task.core for task in result.tasks], ["C", "C", "V", "V"])
        c_scalar, v_scalar = [binding.storage for binding in result.scalar_bindings]
        self.assertIsNot(c_scalar, v_scalar)
        self.assertEqual((c_scalar.domain, v_scalar.domain), ("C", "V"))
        self.assertIs(result.tasks[1].accesses[0].storage, c_scalar)
        self.assertIs(result.tasks[3].accesses[0].storage, v_scalar)
        self.assertIsNot(result.tasks[1].accesses[-1].storage, result.tasks[3].accesses[-1].storage)

    def test_scalar_only_unscoped_domain_is_not_invented(self):
        T = self.tir
        local = T.decl_buffer((1,), "int32", name="local", scope="local")
        with self.assertRaisesRegex(adapter.ExtractionError, "per-core control normalization"):
            adapter.extract_tasks(T.BufferStore(local, T.IntImm("int32", 0), [0]))

    def test_actual_extern_gm_hop_composites_rejected(self):
        T = self.tir
        source, destination, workspace = (
            T.Var(name, "handle") for name in ("source", "destination", "workspace"))
        names = ("tl::ascend::copy_ub_to_l1<half,16,2>",
                 "tl::ascend::copy_l0c_to_ub<float,half,layout::RowMajor,16,16,0>")
        for name in names:
            call = T.Evaluate(
                T.call_extern("handle", name, self.pointer(source),
                              self.pointer(destination, mask=2), 16, 2, 2, 16,
                              self.pointer(workspace, mask=3)))
            with self.subTest(name=name), \
                    self.assertRaisesRegex(adapter.ExtractionError, "cross-core.*decompose"):
                adapter.extract_tasks(call)

    def test_lower_level_gm_hop_components_remain_supported(self):
        T = self.tir
        source, destination = T.Var("source", "handle"), T.Var("destination", "handle")
        for name, pipe, core in (("copy_gm_to_l1", "PIPE_MTE2", "C"), ("copy_ub_to_gm", "PIPE_MTE3",
                                                                       "V")):
            call = T.Evaluate(
                T.call_extern("handle", name, self.pointer(source),
                              self.pointer(destination, mask=2)))
            result = adapter.extract_tasks(call)
            self.assertEqual((result.tasks[0].pipe, result.tasks[0].core), (pipe, core))
            self.assertEqual([access.mode for access in result.tasks[0].accesses], ["R", "W"])
            self.assertTrue(all(access.footprint() is None for access in result.tasks[0].accesses))

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
