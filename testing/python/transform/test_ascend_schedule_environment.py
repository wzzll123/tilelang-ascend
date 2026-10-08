# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Runtime identity checks must reject accidental source/native mixing."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts/ascend_schedule_environment.py"
_SPEC = importlib.util.spec_from_file_location("schedule_environment", _PATH)
environment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(environment)


def test_correct_component_root(tmp_path):
    source = tmp_path / "tilelang" / "__init__.py"
    assert environment.require_under(source, tmp_path, "Python") == str(source)


def test_wrong_runtime_root_fails(tmp_path):
    with pytest.raises(RuntimeError, match="TVM native library loaded from"):
        environment.require_under(tmp_path / "vendor/libtvm.so", tmp_path / "build",
                                  "TVM native library")


def test_symlink_does_not_hide_wrong_library(tmp_path):
    expected = tmp_path / "build"
    expected.mkdir()
    outside = tmp_path / "vendor.so"
    outside.write_bytes(b"not our library")
    link = expected / "libtvm.so"
    link.symlink_to(outside)
    with pytest.raises(RuntimeError, match="expected under"):
        environment.require_under(link, expected, "native")


def test_library_identity_includes_content_hash(tmp_path):
    library = tmp_path / "libtilelang.so"
    library.write_bytes(b"compiled artifact")
    identity = environment.file_identity(library)
    assert identity["size_bytes"] == 17
    assert identity["sha256"] == hashlib.sha256(b"compiled artifact").hexdigest()


def test_missing_commands_are_reported_not_faked():
    result = environment.command(["/nonexistent/bisheng"])
    assert not result["available"]


def test_cann_version_from_selected_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    (tmp_path / "version.info").write_text("Version=fixture\n")
    identity = environment.cann_identity()
    assert identity["available"]
    assert identity["version_files"][str(tmp_path / "version.info")] == "Version=fixture\n"


def test_unavailable_cann_not_zero_version(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    assert not environment.cann_identity()["available"]


def test_solver_fingerprint_records_selected_package(tmp_path, monkeypatch):
    source = tmp_path / "z3" / "__init__.py"
    monkeypatch.setitem(
        sys.modules, "z3",
        SimpleNamespace(__file__=str(source), get_version_string=lambda: "fixture-version"))
    assert environment.solver_identity() == {
        "available": True,
        "version": "fixture-version",
        "python": str(source)
    }


def test_missing_solver_fingerprint_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "z3", None)
    assert not environment.solver_identity()["available"]
