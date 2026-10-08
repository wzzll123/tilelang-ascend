# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Real old-TVM byte proofs; no device execution or dependency pruning."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[3] / "tilelang/transform/ascend_schedule/region_proof.py"
spec = importlib.util.spec_from_file_location("_ascend_region_proof_tests", PATH)
proof = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = proof
spec.loader.exec_module(proof)


class RegionProofTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import tvm
            import z3
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Old TVM and optional Z3 required: {error}") from error
        cls.T, cls.z3 = tvm.tir, z3

    def prove(self, a, extent_a, b, extent_b, bounds=(), **kwargs):
        return proof.prove_byte_regions_disjoint(a, extent_a, b, extent_b, bounds, **kwargs)

    def test_constants_partial_overlap_and_physical_byte_views(self):
        self.assertTrue(self.prove(16, 32, 48, 8).proved_disjoint)
        self.assertFalse(self.prove(16, 32, 40, 16).proved_disjoint)
        self.assertTrue(self.prove(16, 0, 16, 32).proved_disjoint)

    def test_mixed_dtype_aliases_are_explicitly_converted_to_bytes(self):
        T = self.T
        half = T.decl_buffer((16,), "float16", name="physical", elem_offset=0)
        storage = half.data
        wide = T.decl_buffer((4,), "float32", data=storage, elem_offset=4)
        self.assertTrue(half.data.same_as(wide.data))
        self.assertFalse(
            self.prove(half.elem_offset * 2, 16 * 2, wide.elem_offset * 4, 4 * 4).proved_disjoint)
        disjoint = T.decl_buffer((4,), "float32", data=storage, elem_offset=8)
        self.assertTrue(
            self.prove(half.elem_offset * 2, 16 * 2, disjoint.elem_offset * 4,
                       4 * 4).proved_disjoint)

    def test_same_storage_var_wrapper_identity_and_equal_names(self):
        T = self.T
        i, j = T.Var("i", "int32"), T.Var("i", "int32")
        wrapper = T.Add(i, T.IntImm("int32", 0)).a
        self.assertTrue(wrapper.same_as(i))
        self.assertTrue(
            self.prove(i * 16, 16, wrapper * 16 + 16, 16, [(i, (0, 2))]).proved_disjoint)
        self.assertFalse(
            self.prove(i * 16, 16, j * 16 + 16, 16, [(i, (0, 2)), (j, (0, 2))]).proved_disjoint)

    def test_bounded_moving_offsets_and_assumptions(self):
        T = self.T
        i = T.Var("moving", "int32")
        self.assertTrue(self.prove(i * 16, 8, 128, 8, {i: (0, 7)}).proved_disjoint)
        self.assertFalse(self.prove(i * 16, 8, 128, 8, {i: (0, 10)}).proved_disjoint)
        self.assertTrue(
            self.prove(i * 16, 8, 128, 8, {
                i: (0, 10)
            }, assumptions=(T.And(i >= 0, i < 8),)).proved_disjoint)

    def test_comparison_and_logical_assumptions(self):
        T = self.T
        i = T.Var("i", "int32")
        conditions = (T.LT(i, 8), T.LE(i, 7), T.GT(i, 8), T.GE(i, 9), T.EQ(i, 0), T.NE(i, 8),
                      T.Not(T.EQ(i, 8)), T.Or(T.LT(i, 8), T.GT(i, 8)))
        for condition in conditions:
            with self.subTest(kind=type(condition).__name__):
                self.assertTrue(
                    self.prove(i * 16, 8, 128, 8, [(i, (0, 10))],
                               assumptions=(condition,)).proved_disjoint)

    def test_subtraction_and_negative_constant_affine_mul(self):
        T = self.T
        i = T.Var("i", "int32")
        self.assertTrue(self.prove(16 + i * -4, 4, 20, 4, [(i, (0, 4))]).proved_disjoint)
        self.assertTrue(self.prove(16 - i * 4, 4, 20, 4, [(i, (0, 4))]).proved_disjoint)

    def test_unsupported_calls_cast_division_and_nonlinear_mul(self):
        T = self.T
        i, j = T.Var("i", "int32"), T.Var("j", "int32")
        for expr in (T.Cast("int64", i), T.FloorDiv(i, T.IntImm("int32", 2)), i * j,
                     T.call_extern("int32", "mutable_offset", i)):
            with self.subTest(kind=type(expr).__name__):
                result = self.prove(expr, 4, 100, 4, [(i, (0, 4)), (j, (0, 4))])
                self.assertFalse(result.proved_disjoint)
                self.assertTrue(result.diagnostics)

    def test_signed_and_unsigned_intermediate_overflow_rejected(self):
        T = self.T
        for dtype, bounds in [("int8", (120, 127)), ("uint8", (250, 255))]:
            i = T.Var("overflow", dtype)
            expr = T.Add(i, T.IntImm(dtype, 8))
            result = self.prove(expr, 4, 0, 4, [(i, bounds)])
            self.assertFalse(result.proved_disjoint)
            self.assertTrue("overflow" in " ".join(result.diagnostics))

    def test_assumptions_do_not_silently_bypass_overflow_range_check(self):
        T = self.T
        i = T.Var("overflow", "int8")
        result = self.prove(
            T.Add(i, T.IntImm("int8", 8)), 4, 0, 4, [(i, (100, 127))], assumptions=(i < 110,))
        self.assertFalse(result.proved_disjoint)
        self.assertTrue("overflow" in " ".join(result.diagnostics))

    def test_negative_extent_and_missing_bound_rejected(self):
        self.assertFalse(self.prove(0, -1, 8, 4).proved_disjoint)
        result = self.prove(self.T.Var("unbounded", "int32"), 4, 100, 4)
        self.assertFalse(result.proved_disjoint)
        self.assertTrue("bound" in " ".join(result.diagnostics))

    def test_missing_z3_is_explicit_conservative_result(self):
        with patch.object(proof, "_load_z3", return_value=None):
            result = self.prove(0, 4, 8, 4)
        self.assertFalse(result.proved_disjoint)
        self.assertTrue("z3-solver" in " ".join(result.diagnostics))

    def test_timeout_unknown_is_conservative(self):
        z3 = self.z3

        class UnknownSolver:

            def add(self, *args):
                pass

            def check(self):
                return z3.unknown

            def reason_unknown(self):
                return "timeout"

        with patch.object(proof, "_new_solver", return_value=UnknownSolver()):
            result = self.prove(0, 4, 8, 4, timeout_ms=1)
        self.assertFalse(result.proved_disjoint)
        self.assertTrue("timeout" in " ".join(result.diagnostics))
