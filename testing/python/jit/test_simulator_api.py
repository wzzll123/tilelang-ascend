"""Focused tests for the public simulator JIT plumbing."""

from __future__ import annotations

import importlib


def test_compile_normal_path_still_uses_binary_cache(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")
    libgen_module = importlib.import_module("tilelang.jit.adapter.libgen")
    calls = {}
    expected_kernel = object()

    def fake_resolve_compile_flags(target, pass_configs, compile_flags):
        calls["resolve"] = (target, pass_configs, compile_flags)
        return ["-O3"]

    def fake_cached(**kwargs):
        calls["cached"] = kwargs
        return expected_kernel

    monkeypatch.setattr(libgen_module, "resolve_compile_flags", fake_resolve_compile_flags)
    monkeypatch.setattr(jit_module, "cached", fake_cached)

    kernel = jit_module.compile(object(), target="pto", platform="A2")

    assert kernel is expected_kernel
    assert calls["cached"]["compile_flags"] == ["-O3"]
    assert "simulator" not in calls["cached"]


def test_compile_simulator_bypasses_binary_cache(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")
    calls = {}
    func = object()
    sim_config = object()
    expected_kernel = object()

    def fake_jit_kernel(**kwargs):
        calls.update(kwargs)
        return expected_kernel

    def fail_cached(**_kwargs):
        raise AssertionError("simulator kernels must bypass the device binary cache")

    monkeypatch.setattr(jit_module, "JITKernel", fake_jit_kernel)
    monkeypatch.setattr(jit_module, "cached", fail_cached)

    kernel = jit_module.compile(
        func,
        target="pto",
        platform="A2",
        simulator=True,
        sim_config=sim_config,
    )

    assert kernel is expected_kernel
    assert calls["func"] is func
    assert calls["simulator"] is True
    assert calls["sim_config"] is sim_config
    assert calls["platform"] == "A2"


def test_jit_decorator_forwards_simulator_options(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")
    calls = {}
    program = object()
    expected_kernel = object()
    sim_config = object()

    def fake_compile(func, **kwargs):
        calls["func"] = func
        calls.update(kwargs)
        return expected_kernel

    monkeypatch.setattr(jit_module, "compile", fake_compile)

    @jit_module.jit(simulator=True, sim_config=sim_config, platform="A3")
    def make_program(size):
        assert size == 128
        return program

    kernel = make_program(128)

    assert kernel is expected_kernel
    assert calls["func"] is program
    assert calls["simulator"] is True
    assert calls["sim_config"] is sim_config
    assert calls["platform"] == "A3"


def test_jit_kernel_simulator_uses_lazy_adapter_hook(monkeypatch):
    kernel_module = importlib.import_module("tilelang.jit.kernel")
    adapter_module = importlib.import_module("tilelang.simulator.adapter")
    calls = {}
    sim_config = object()

    class FakeAdapter:
        artifact = None
        params = []
        result_idx = []
        workspace_idx = []
        func = staticmethod(lambda *args: args)

    def fake_create_simulator_adapter(**kwargs):
        calls.update(kwargs)
        return FakeAdapter()

    monkeypatch.setattr(adapter_module, "create_simulator_adapter", fake_create_simulator_adapter)

    func = object()
    kernel = kernel_module.JITKernel(
        func=func,
        simulator=True,
        sim_config=sim_config,
        target="pto",
        platform="A2",
    )

    assert kernel.simulator is True
    assert calls["func"] is func
    assert calls["sim_config"] is sim_config
    assert calls["platform"] == "A2"
