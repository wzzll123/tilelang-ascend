# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Allow CPU-only simulator tests to run before the native TileLang build exists."""

import os
from pathlib import Path
import sys
import types


repository_root = Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "TEST_DATA_ROOT_PATH",
    str(repository_root / ".pytest_cache" / "tvm-test-data"),
)


if "tilelang" not in sys.modules:
    # Importing tilelang normally loads TVM and libtilelang.  The simulator core is
    # intentionally backend-neutral, so expose only the package path for these tests.
    package = types.ModuleType("tilelang")
    package.__path__ = [str(repository_root / "tilelang")]
    package.__package__ = "tilelang"
    sys.modules["tilelang"] = package
