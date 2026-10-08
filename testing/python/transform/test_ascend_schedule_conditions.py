# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Real-TIR condition snapshots preserve lexical evaluation and updates."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_ascend_condition_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "tilelang/transform/ascend_schedule")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(
        PACKAGE + "." + name, ROOT / "tilelang/transform/ascend_schedule" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


conditions = load("conditions")
adapter = load("tir_adapter")


class ConditionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import tvm
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Old TVM unavailable: {error}") from error
        cls.tvm, cls.T = tvm, tvm.tir

    def nodes(self, node, kind):
        result = []
        self.T.stmt_functor.post_order_visit(
            node.body if isinstance(node, self.T.PrimFunc) else node,
            lambda item: result.append(item) if isinstance(item, kind) else None)
        return result

    def stateful_if(self):
        T = self.T
        state = T.decl_buffer((1,), "int32", name="state")
        return state, T.IfThenElse(state[0] > 0, T.BufferStore(state, T.IntImm("int32", 0), [0]),
                                   T.BufferStore(state, T.IntImm("int32", 1), [0]))

    def extract_in_core(self, node, core=0):
        T = self.T
        return adapter.extract_tasks(T.AttrStmt(T.IntImm("int32", 0), "resource_scope", core, node))

    def test_buffer_condition_snapshot_is_real_let_before_write(self):
        state, original = self.stateful_if()
        normalized = conditions.normalize_conditions(original)
        self.assertIsInstance(normalized, self.T.AttrStmt)
        self.assertEqual(normalized.attr_key, "tl.ascend_condition_snapshot")
        binding = normalized.body
        self.assertIsInstance(binding, self.T.LetStmt)
        self.assertTrue(binding.value.same_as(original.condition))
        self.assertTrue(binding.body.condition.same_as(binding.var))
        result = self.extract_in_core(normalized)
        self.assertEqual(len(result.scalar_bindings), 1)
        self.assertEqual(len(result.tasks), 3)  # Actual Let + then/else writes.
        self.assertTrue(result.scalar_bindings[0].var.same_as(binding.var))
        self.assertTrue(result.task_nodes[0].same_as(binding))
        self.assertEqual(result.tasks[1].guard[0][1], True)
        self.assertEqual(result.tasks[2].guard[0][1], False)
        self.assertIs(result.tasks[1].guard[0][0], result.scalar_bindings[0].storage)

    def test_nested_conditions_normalized_once_and_idempotent(self):
        T = self.T
        state, child = self.stateful_if()
        outer = T.IfThenElse(state[0] == 1, child, T.Evaluate(0))
        first = conditions.normalize_conditions(outer)
        second = conditions.normalize_conditions(first)
        self.assertTrue(self.tvm.ir.structural_equal(first, second))
        self.assertEqual(len(self.nodes(first, T.LetStmt)), 2)
        result = self.extract_in_core(first)
        self.assertEqual(len(result.scalar_bindings), 2)
        self.assertEqual(len(result.tasks[-1].guard), 2)
        for initial in (0, 1):
            self.assertEqual(self.execute(outer, initial), self.execute(first, initial))

    def test_loop_snapshot_remains_inside_each_iteration(self):
        T = self.T
        _, body = self.stateful_if()
        loop = T.For(T.Var("i", "int32"), 0, 4, T.ForKind.SERIAL, body)
        normalized = conditions.normalize_conditions(loop)
        self.assertIsInstance(normalized, T.For)
        self.assertIsInstance(normalized.body, T.AttrStmt)
        self.assertIsInstance(normalized.body.body, T.LetStmt)
        result = self.extract_in_core(normalized)
        self.assertTrue(all(len(result.loops[task.id]) == 1 for task in result.tasks))
        for initial in (0, 1):
            self.assertEqual(self.execute(loop, initial), self.execute(normalized, initial))
            self.assertEqual(self.execute(normalized, initial), initial)

    def execute(self, statement, initial):
        """Tiny scalar interpreter for this test's real TIR, independent of extraction."""
        T = self.T
        memory, variables = {0: initial}, {}

        def evaluate(expr):
            if isinstance(expr, T.IntImm):
                return int(expr.value)
            if isinstance(expr, T.Var):
                return variables[expr]
            if isinstance(expr, T.BufferLoad):
                return memory[evaluate(expr.indices[0])]
            if isinstance(expr, T.GT):
                return evaluate(expr.a) > evaluate(expr.b)
            if isinstance(expr, T.EQ):
                return evaluate(expr.a) == evaluate(expr.b)
            raise AssertionError(type(expr).__name__)

        def run(node):
            if isinstance(node, T.AttrStmt):
                run(node.body)
            elif isinstance(node, T.LetStmt):
                variables[node.var] = evaluate(node.value)
                run(node.body)
            elif isinstance(node, T.IfThenElse):
                child = node.then_case if evaluate(node.condition) else node.else_case
                if child is not None:
                    run(child)
            elif isinstance(node, T.BufferStore):
                memory[evaluate(node.indices[0])] = evaluate(node.value)
            elif isinstance(node, T.For):
                for index in range(evaluate(node.min), evaluate(node.min) + evaluate(node.extent)):
                    variables[node.loop_var] = index
                    run(node.body)
            elif isinstance(node, T.Evaluate):
                evaluate(node.value)
            else:
                raise AssertionError(type(node).__name__)

        run(statement)
        return memory[0]

    def test_if_else_toggles_and_preserves_spans(self):
        T = self.T
        _, body = self.stateful_if()
        span = self.tvm.ir.Span(self.tvm.ir.SourceName("condition_test"), 2, 4, 1, 7)
        original = T.IfThenElse(body.condition, body.then_case, body.else_case, span)
        normalized = conditions.normalize_conditions(original)
        self.assertTrue(normalized.span.same_as(span))
        self.assertTrue(normalized.body.span.same_as(span))
        self.assertTrue(normalized.body.body.span.same_as(span))
        self.assertTrue(normalized.body.body.then_case.same_as(original.then_case))
        self.assertTrue(normalized.body.body.else_case.same_as(original.else_case))
        for initial in (0, 1):
            self.assertEqual(self.execute(normalized, initial), 1 - initial)

    def test_name_collision_preserves_distinct_var_identity(self):
        T = self.T
        name_collision = T.Var("ascend_condition_snapshot_0", "bool")
        original = T.LetStmt(name_collision, T.IntImm("bool", 1),
                             T.IfThenElse(name_collision, T.Evaluate(1), None))
        normalized = conditions.normalize_conditions(original)
        new_var = normalized.body.body.var
        self.assertFalse(new_var.same_as(name_collision))
        self.assertNotEqual(new_var.name, name_collision.name)
        self.assertTrue(normalized.body.body.value.same_as(name_collision))

    def test_primfunc_attrs_and_no_if_identity(self):
        T = self.T
        _, body = self.stateful_if()
        function = T.PrimFunc([], body).with_attr("npu_platform", "A2")
        normalized = conditions.normalize_conditions(function)
        self.assertEqual(str(normalized.attrs["npu_platform"]), "A2")
        empty = T.PrimFunc([], T.Evaluate(0))
        self.assertTrue(self.tvm.ir.structural_equal(empty, conditions.normalize_conditions(empty)))

    def test_flag_off_path_does_not_normalize_source(self):
        entry = load("entry")
        _, body = self.stateful_if()
        mod = self.tvm.IRModule({"main": self.T.PrimFunc([], body).with_attr("npu_platform", "A2")})
        original = self.tvm.ir.save_json(mod)
        for config in ({}, {"tl.ascend_auto_schedule": False}):
            returned = entry.schedule_module(mod, "ascendc", config)
            self.assertIs(returned, mod)
            self.assertTrue(self.tvm.ir.structural_equal(self.tvm.ir.load_json(original), returned))
            self.assertEqual(self.nodes(returned["main"], self.T.LetStmt), [])

    def test_unnormalized_if_extraction_requires_actual_snapshot(self):
        _, body = self.stateful_if()
        with self.assertRaisesRegex(adapter.ExtractionError, "normalize_conditions"):
            adapter.extract_tasks(body)


if __name__ == "__main__":
    unittest.main()
