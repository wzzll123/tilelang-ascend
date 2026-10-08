# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Opt-in policy and fail-closed lowering bootstrap contracts."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_ascend_policy_tests"
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


policy = load("policy")
entry = load("entry")
KEY = "tl.ascend_auto_schedule"


class OptionsTests(unittest.TestCase):

    def test_default_and_explicit_false_leave_legacy_options_alone(self):
        for config in ({}, {KEY: False, "tl.ascend_auto_sync": False}):
            self.assertFalse(policy.resolve_options(config, "pto", "A5").enabled)

    def test_supported_targets_and_absent_sync(self):
        for platform in ("A2", "A3"):
            result = policy.resolve_options({KEY: True}, "ascendc", platform)
            self.assertTrue(result.enabled)
            self.assertTrue(result.unified_sync)

    def test_explicit_false_sync_rejected(self):
        with self.assertRaisesRegex(policy.OptionsError, "ascend_auto_sync"):
            policy.resolve_options({KEY: True, "tl.ascend_auto_sync": False}, "ascendc", "A2")

    def test_vs_false_does_not_select_legacy_vs_algorithm(self):
        self.assertTrue(
            policy.resolve_options({
                KEY: True,
                "tl.ascend_auto_sync_vs": False
            }, "ascendc", "A3").enabled)

    def test_unsupported_targets_fail_when_enabled(self):
        for target, platform in (("pto", "A2"), ("cuda", "A3"), ("ascendc", "A5"), ("ascendc",
                                                                                    "auto")):
            with self.subTest(target=target, platform=platform), \
                    self.assertRaises(policy.OptionsError):
                policy.resolve_options({KEY: True}, target, platform)

    def test_strict_boolean_conversion(self):
        for value in (True, 1, type("IntImm", (), {"dtype": "bool", "value": 1})()):
            self.assertTrue(policy.resolve_options({KEY: value}, "ascendc", "A2").enabled)
        for value in ("true", "False", 2, -1, None, 1.0, object()):
            with self.subTest(value=value), self.assertRaises(policy.OptionsError):
                policy.resolve_options({KEY: value}, "ascendc", "A2")


class BootstrapTests(unittest.TestCase):

    def module(self, platform="A2"):
        function = types.SimpleNamespace(attrs={"npu_platform": platform})
        return types.SimpleNamespace(functions={"main": function})

    def test_flag_off_returns_input_without_extracting(self):
        mod = self.module()
        with patch.object(entry, "extract_tasks", side_effect=AssertionError("unexpected")), \
                patch.object(entry, "normalize_conditions",
                             side_effect=AssertionError("unexpected normalization")):
            self.assertIs(entry.schedule_module(mod, "ascendc", {}), mod)
            self.assertIs(entry.schedule_module(mod, "ascendc", {KEY: False}), mod)

    def test_flag_on_extracts_then_errors_no_fallback(self):
        mod = self.module()
        with patch.object(entry, "extract_tasks") as extract, \
                patch.object(entry, "decompose_function_gemms", side_effect=lambda node, **kwargs: node) as decompose, \
                patch.object(entry, "normalize_core_scopes", side_effect=lambda node, extraction: node), \
                patch.object(entry, "normalize_conditions", side_effect=lambda node: node) as normalize:
            with self.assertRaisesRegex(entry.AutoScheduleError, "materializ"):
                entry.schedule_module(mod, "ascendc", {KEY: True})
            extract.assert_called_once_with(mod.functions["main"])
            normalize.assert_called_once_with(mod.functions["main"])
            decompose.assert_called_once_with(mod.functions["main"], root_scratch=True)

    def test_extraction_error_retains_multi_pipe_diagnostic(self):
        mod = self.module()
        with patch.object(entry, "extract_tasks", side_effect=ValueError("multi-pipe gemm_v0")), \
                patch.object(entry, "decompose_function_gemms", side_effect=lambda node, **kwargs: node), \
                patch.object(entry, "normalize_conditions", side_effect=lambda node: node), \
                self.assertRaisesRegex(ValueError, "multi-pipe gemm_v0"):
            entry.schedule_module(mod, "ascendc", {KEY: True})

    def test_phase_entry_is_after_legalization_before_pipeline(self):
        source = (ROOT / "tilelang/engine/phase.py").read_text()
        lower_index = source.index("mod = tilelang.transform.LowerTileOp()(mod)")
        scheduler_index = source.index("mod = schedule_module(", lower_index)
        tail_index = source.index("AscendTailMaskPropagation(", lower_index)
        self.assertLess(lower_index, scheduler_index)
        self.assertLess(tail_index, scheduler_index)
        self.assertLess(source.index("AscendWorkspaceReduction(", lower_index), scheduler_index)
        self.assertLess(source.index("LegalizeSafeMemoryAccess(", lower_index), scheduler_index)


class NativePolicyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import tvm
            import tilelang
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"Native TVM/TileLang unavailable: {error}") from error
        cls.tvm, cls.tilelang = tvm, tilelang

    def test_registered_native_pass_context_boolean_option(self):
        T = self.tvm.tir
        self.assertTrue(KEY in self.tvm.transform.PassContext.list_configs())
        mod = self.tvm.IRModule(
            {"main": T.PrimFunc([], T.Evaluate(0)).with_attr("npu_platform", "A2")})
        with self.tvm.transform.PassContext() as context:
            self.assertFalse(policy.resolve_options(context.config, "ascendc", "A2").enabled)
            self.assertIs(entry.schedule_module(mod, "ascendc", context.config), mod)
        with self.tvm.transform.PassContext(config={KEY: False}) as context:
            self.assertFalse(policy.resolve_options(context.config, "ascendc", "A2").enabled)
            self.assertIs(entry.schedule_module(mod, "ascendc", context.config), mod)
        with self.tvm.transform.PassContext(config={KEY: True}) as context:
            self.assertTrue(policy.resolve_options(context.config, "ascendc", "A2").enabled)
            with self.assertRaisesRegex(entry.AutoScheduleError, "materializer"):
                entry.schedule_module(mod, "ascendc", context.config)

    def test_native_intimm_options_and_module_platform_attrs(self):
        T = self.tvm.tir
        config = {KEY: T.IntImm("bool", 1)}
        mod = self.tvm.IRModule(
            {"main": T.PrimFunc([], T.Evaluate(0)).with_attr("npu_platform", "A3")})
        with self.assertRaisesRegex(entry.AutoScheduleError, "materializer"):
            entry.schedule_module(mod, "ascendc", config)
        with self.assertRaisesRegex(policy.OptionsError, "ascend_auto_sync"):
            entry.schedule_module(mod, "ascendc", {
                KEY: T.IntImm("bool", 1),
                "tl.ascend_auto_sync": T.IntImm("bool", 0)
            })

    def test_enabled_if_primfunc_reaches_materializer_after_normalization(self):
        T = self.tvm.tir
        state = T.decl_buffer((1,), "int32", name="state")
        body = T.IfThenElse(state[0] > 0, T.BufferStore(state, T.IntImm("int32", 0), [0]), None)
        body = T.AttrStmt(T.IntImm("int32", 0), "resource_scope", 0, body)
        mod = self.tvm.IRModule({"main": T.PrimFunc([], body).with_attr("npu_platform", "A2")})
        source = self.tvm.ir.save_json(mod)
        with self.assertRaisesRegex(entry.AutoScheduleError, "materializer"):
            entry.schedule_module(mod, "ascendc", {KEY: True})
        self.assertEqual(self.tvm.ir.save_json(mod), source)

    def test_unsupported_phase_stops_after_legalization_before_target_pipeline(self):
        from tilelang.engine import phase

        T = self.tvm.tir
        mod = self.tvm.IRModule(
            {"main": T.PrimFunc([], T.Evaluate(0)).with_attr("npu_platform", "A2")})
        passes = []

        def fake_pass(name):

            def factory(*args, **kwargs):

                def run(module):
                    passes.append(name)
                    return module

                return run

            return factory

        early = ("InjectTmpBuffer", "AscendInferBufferScope", "AscendVidReduction",
                 "BufferShapeCollector", "HostProcesser", "AscendLowerParallelToVector",
                 "LayoutInference", "CollectBufferShapes", "LowerTileOp",
                 "AscendTailMaskPropagation", "AscendWorkspaceReduction", "LegalizeVectorizedLoop",
                 "LegalizeSafeMemoryAccess")
        with ExitStack() as stack:
            for name in early:
                stack.enter_context(patch.object(self.tilelang.transform, name, fake_pass(name)))
            for name in ("BindTarget", "Simplify"):
                stack.enter_context(patch.object(T.transform, name, fake_pass(name)))
            stack.enter_context(
                patch.object(
                    self.tilelang.transform,
                    "get_pass_context",
                    return_value=types.SimpleNamespace(config={KEY: True})))
            with self.assertRaisesRegex(NotImplementedError, "materializer"):
                phase.LowerAndLegalize(mod, types.SimpleNamespace(model="ascendc"))
            self.assertEqual(passes[-1], "Simplify")
            self.assertIn("AscendWorkspaceReduction", passes)


if __name__ == "__main__":
    unittest.main()
