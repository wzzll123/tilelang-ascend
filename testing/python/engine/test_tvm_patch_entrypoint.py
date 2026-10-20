# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Build-entry patching stays idempotent after the pinned TVM slice fix."""

from __future__ import annotations

import ast
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "3rdparty" / "patches"
LOGGING = """void BacktraceSyminfoCallback() {
  auto str = reinterpret_cast<std::string*>(data);

  if (symname != nullptr) {
    std::string tmp(symname, symsize);
    *str = DemangleName(tmp.c_str());
  } else {
    std::ostringstream s;
  }
}
"""


@pytest.fixture
def patch_checkout(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is required to exercise the build patch entrypoint")
    patch_dir = tmp_path / "3rdparty" / "patches"
    shutil.copytree(PATCHES, patch_dir)
    tvm_dir = tmp_path / "3rdparty" / "tvm"
    logging = tvm_dir / "src" / "runtime" / "logging.cc"
    logging.parent.mkdir(parents=True)
    logging.write_text(LOGGING)
    subprocess.run(["git", "init", "-q", str(tvm_dir)], check=True)
    subprocess.run(["git", "-C", str(tvm_dir), "add", "src/runtime/logging.cc"], check=True)
    return tmp_path, logging


def _apply(root):
    return subprocess.run(
        ["bash", str(root / "3rdparty" / "patches" / "apply_tvm_patches.sh"), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_obsolete_slice_patch_is_not_reapplied():
    assert not (PATCHES / "tvm_slice_step_fix.patch").exists(), (
        "The pinned TVM already contains the unit-step slice fix; its old text patch conflicts with the committed implementation."
    )


def test_build_patch_entrypoint_is_idempotent(patch_checkout):
    root, logging = patch_checkout
    first = _apply(root)
    assert first.returncode == 0, first.stdout + first.stderr
    applied = logging.read_text()
    assert "std::string tmp(symname);" in applied
    assert "std::string tmp(symname, symsize);" not in applied
    second = _apply(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already applied, skipping" in second.stdout
    assert logging.read_text() == applied


def test_build_patch_entrypoint_still_rejects_unknown_conflicts(patch_checkout):
    root, logging = patch_checkout
    logging.write_text(LOGGING.replace("std::string tmp(symname, symsize);", 'std::string tmp("incompatible");'))
    before = logging.read_bytes()
    result = _apply(root)
    assert result.returncode != 0
    assert "cannot apply tvm_backtrace_symbol_name_length.patch" in result.stderr
    assert logging.read_bytes() == before


def test_pinned_tvm_accepts_dynamic_unit_step_slice_without_local_patch():
    path = ROOT / "3rdparty" / "tvm" / "python" / "tvm" / "tir" / "buffer.py"
    if not path.exists():
        pytest.skip("initialize the pinned TVM submodule to inspect its slice contract")
    module = ast.parse(path.read_text())
    buffer_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Buffer")
    getitem = next(node for node in buffer_class.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
    assignment = next(
        node
        for node in ast.walk(getitem)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "has_step" for target in node.targets)
    )
    # Evaluate the actual committed predicate, not a loose source substring.
    predicate = compile(ast.Expression(assignment.value), str(path), "eval")
    dynamic_stop = object()
    namespace = {"any": any, "isinstance": isinstance, "slice": slice}
    for step, expected in [(None, False), (1, False), (2, True)]:
        namespace["indices"] = [0, slice(0, dynamic_stop, step)]
        assert eval(predicate, {"__builtins__": {}, **namespace}) is expected
