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
    try:
        # Prefer the real package when the native build exists: it puts
        # 3rdparty/tvm/python on sys.path, enabling end-to-end simulator tests
        # that compile T.prim_func kernels with simulator=True.
        import tilelang  # noqa: F401
    except Exception:
        # Importing tilelang normally loads TVM and libtilelang.  The simulator
        # core is intentionally backend-neutral, so expose only the package
        # path for these tests when the native build is unavailable.
        package = types.ModuleType("tilelang")
        package.__path__ = [str(repository_root / "tilelang")]
        package.__package__ = "tilelang"
        sys.modules["tilelang"] = package
