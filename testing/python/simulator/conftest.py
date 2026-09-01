# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Allow CPU-only simulator tests to run before the native TileLang build exists."""

import importlib.util
from pathlib import Path
import sys
import types


def _native_tvm_is_available() -> bool:
    try:
        return importlib.util.find_spec("tvm") is not None
    except (ImportError, OSError, RuntimeError):
        return False


if not _native_tvm_is_available() and "tilelang" not in sys.modules:
    # Importing tilelang normally loads TVM and libtilelang.  The simulator core is
    # intentionally backend-neutral, so expose only the package path for these tests.
    repository_root = Path(__file__).resolve().parents[3]
    package = types.ModuleType("tilelang")
    package.__path__ = [str(repository_root / "tilelang")]
    package.__package__ = "tilelang"
    sys.modules["tilelang"] = package
