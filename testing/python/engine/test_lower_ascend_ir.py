# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Focused tests for the shared Ascend pre-codegen lowering boundary."""

from __future__ import annotations

import importlib

from tilelang import tvm
from tvm import tir


lower_module = importlib.import_module("tilelang.engine.lower")


def _empty_module() -> tvm.IRModule:
    func = tir.PrimFunc([], tir.Evaluate(0)).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


def test_lower_ascend_ir_has_one_authoritative_pass_order(monkeypatch):
    calls = []

    def lower_and_legalize(mod, target):
        calls.append(("legalize", target.model))
        return mod

    def optimize_for_target(mod, target, platform):
        calls.append(("optimize", target.model, platform))
        return mod

    class RecordSimplify:

        def __call__(self, mod):
            calls.append(("simplify",))
            return mod

    monkeypatch.setattr(lower_module, "LowerAndLegalize", lower_and_legalize)
    monkeypatch.setattr(lower_module, "OptimizeForTarget", optimize_for_target)
    monkeypatch.setattr(lower_module.tir.transform, "Simplify", RecordSimplify)

    mod, params = lower_module.lower_ascend_ir(_empty_module(), target="pto", platform="A2")

    assert calls == [("legalize", "pto"), ("optimize", "pto", "A2"), ("simplify",)]
    assert params == []
    assert mod["main"].attrs["npu_platform"] == "A2"


def test_native_lower_codegen_consumes_shared_pre_codegen_ir(monkeypatch):
    lowered_mod = _empty_module()
    params = [object()]
    calls = []

    def shared_lower(func_or_mod, target, platform):
        calls.append(("lower", func_or_mod, target.model, platform))
        return lowered_mod, params

    class CodegenModule:

        @staticmethod
        def get_source():
            return "generated source"

    def codegen(mod, target, platform):
        calls.append(("codegen", mod, target.model, platform))
        return CodegenModule()

    monkeypatch.setattr(lower_module, "lower_ascend_ir", shared_lower)
    monkeypatch.setattr(lower_module, "device_codegen", codegen)

    artifact = lower_module.lower(_empty_module(), target="ascendc", platform="A3")

    assert calls[0][0] == "lower"
    assert calls[1] == ("codegen", lowered_mod, "ascendc", "A3")
    assert artifact.device_mod is lowered_mod
    assert artifact.params is params
    assert artifact.kernel_source == "generated source"
