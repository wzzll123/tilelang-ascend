# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Frontend contracts runnable without a compiled TVM or an Ascend device.

The isolated tests check validation and arguments sent to the existing TIR
builder. The optional parser test checks the resulting real TIR when available.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class IntImm:

    def __init__(self, dtype, value):
        self.dtype = dtype
        self.value = value


def load_source(relative):
    """Use narrow builder doubles, without importing the tilelang package."""
    tvm = types.ModuleType("tvm")
    tvm.tir = types.SimpleNamespace(IntImm=IntImm, const=lambda value, dtype: IntImm(dtype, value))
    builder = types.ModuleType("tvm.script.ir_builder.tir")
    builder.attr = lambda node, key, value: (node, key, value)
    builder.block_attr = lambda annotations: annotations
    target = types.ModuleType("tvm.target")
    target.Target = type("Target", (), {})
    modules = {
        "tvm": tvm,
        "tvm.tir": tvm.tir,
        "tvm.target": target,
        "tvm.script.ir_builder.tir": builder
    }
    spec = importlib.util.spec_from_file_location("_auto_schedule_test_source", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class AnnotationContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api = load_source("tilelang/language/annotations.py")

    def buffer(self, data=None):
        # SimpleNamespace is unhashable; real TIR Buffers are identity keys.
        return type("Buffer", (), {"data": object() if data is None else data})()

    def test_stage_preserves_integer_in_attribute_node(self):
        for stage in (0, 1, 128):
            node, key, value = self.api.Stage(stage)
            self.assertEqual((node.dtype, node.value), ("int64", stage))
            self.assertEqual(key, "tl.ascend_stage")
            self.assertEqual((value.dtype, value.value), ("int32", 1))

    def test_invalid_stages(self):
        for stage in (-1, True, False, 1.5, "1", None):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                self.api.Stage(stage)

    def test_counts_modes_and_storage_keys(self):
        fixed, paired, inferred = self.buffer(), self.buffer(), self.buffer()
        result = self.api.annotate_buffer_versions({
            fixed: 2,
            paired: (3, "counter"),
            inferred: "iteration"
        })
        self.assertEqual(result["tl.buffer_versions_map"], {fixed.data: 2, paired.data: 3})
        self.assertEqual(result["tl.buffer_version_mode"], {
            paired.data: "counter",
            inferred.data: "iteration"
        })

    def test_one_version_distinguishes_opt_out_and_explicit_mode(self):
        buffer = self.buffer()
        self.assertEqual(
            self.api.annotate_buffer_versions({buffer: 1}),
            {"tl.buffer_versions_map": {
                buffer.data: 1
            }})
        self.assertEqual(
            self.api.annotate_buffer_versions({buffer: (1, "auto")}), {
                "tl.buffer_versions_map": {
                    buffer.data: 1
                },
                "tl.buffer_version_mode": {
                    buffer.data: "auto"
                }
            })

    def test_all_supported_modes(self):
        buffer = self.buffer()
        for mode in ("auto", "iteration", "counter"):
            self.assertEqual(
                self.api.annotate_buffer_versions({buffer: mode}),
                {"tl.buffer_version_mode": {
                    buffer.data: mode
                }})

    def test_invalid_specifications(self):
        for value in (0, -1, True, False, 1.5, None, "unknown", (), (2,), (2, "auto", 3),
                      (2, "unknown"), (0, "counter"), (True, "auto")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.api.annotate_buffer_versions({self.buffer(): value})

    def test_aliases_agree_on_storage_metadata(self):
        a = self.buffer()
        b = self.buffer(a.data)
        result = self.api.annotate_buffer_versions({a: (2, "counter"), b: (2, "counter")})
        self.assertEqual(result["tl.buffer_versions_map"], {a.data: 2})
        self.assertEqual(result["tl.buffer_version_mode"], {a.data: "counter"})
        for first, second in ((2, 3), ("iteration", "counter")):
            with self.subTest(first=first), self.assertRaises(ValueError):
                self.api.annotate_buffer_versions({a: first, b: second})

    def test_empty_annotation(self):
        self.assertEqual(self.api.annotate_buffer_versions({}), {})

    def test_alias_opt_out_conflicts_with_explicit_eligibility(self):
        a = self.buffer()
        b = self.buffer(a.data)
        for explicit in ((1, "auto"), "counter", "iteration", "auto"):
            for first, second in ((1, explicit), (explicit, 1)):
                with self.subTest(first=first, second=second), \
                        self.assertRaisesRegex(ValueError, "eligibility"):
                    self.api.annotate_buffer_versions({a: first, b: second})


class PassConfigContractTests(unittest.TestCase):

    def test_auto_schedule_is_opt_in(self):
        config = load_source("tilelang/transform/pass_config.py")
        key = config.PassConfigKey.TL_ASCEND_AUTO_SCHEDULE
        self.assertEqual(key.value, "tl.ascend_auto_schedule")
        for target in ("ascendc", "auto", "pto", "cuda"):
            self.assertFalse(config.process_default_pass_config(target, None).get(key, False))
            for enabled in (False, True):
                self.assertIs(
                    config.process_default_pass_config(target, {key: enabled})[key], enabled)


class TIRParserTests(unittest.TestCase):

    def test_real_buffer_alias_eligibility_conflicts(self):
        try:
            import tilelang.language as T
            from tvm import tir
        except (ImportError, OSError) as error:
            self.skipTest(f"Compiled TileLang/TVM unavailable: {error}")
        a = tir.decl_buffer((8,), "float32", name="a")
        b = tir.decl_buffer((8,), "float32", name="b", data=a.data)
        self.assertTrue(a.data.same_as(b.data))
        for explicit in ((1, "auto"), "counter"):
            for first, second in ((1, explicit), (explicit, 1)):
                with self.subTest(first=first, second=second), \
                        self.assertRaisesRegex(ValueError, "eligibility"):
                    T.annotate_buffer_versions({a: first, b: second})

    def test_annotations_survive_real_tir_parser(self):
        try:
            import tilelang.language as T
            from tvm import tir
        except (ImportError, OSError) as error:
            self.skipTest(f"Compiled TileLang/TVM unavailable: {error}")

        @T.prim_func
        def kernel():
            with T.block("root"):
                scratch = T.alloc_buffer((8,), "float32", scope="shared")
                T.annotate_buffer_versions({scratch: (2, "counter")})
                with T.Stage(3):
                    scratch[0] = T.float32(1)

        stages = []
        blocks = []
        tir.stmt_functor.post_order_visit(
            kernel.body, lambda node: stages.append(node) if isinstance(node, tir.AttrStmt) and node
            .attr_key == "tl.ascend_stage" else blocks.append(node)
            if isinstance(node, tir.Block) else None)
        self.assertEqual(len(stages), 1)
        self.assertEqual(int(stages[0].node), 3)
        self.assertEqual(int(stages[0].value), 1)
        annotated = [block for block in blocks if "tl.buffer_versions_map" in block.annotations]
        self.assertEqual(len(annotated), 1)
        self.assertEqual(
            [int(count) for count in annotated[0].annotations["tl.buffer_versions_map"].values()],
            [2])
        self.assertEqual(
            [str(mode) for mode in annotated[0].annotations["tl.buffer_version_mode"].values()],
            ["counter"])


if __name__ == "__main__":
    unittest.main()
