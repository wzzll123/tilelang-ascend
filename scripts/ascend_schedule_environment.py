#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Print a verified runtime fingerprint before a compiler/FA test process.

This does not install packages, modify checkouts, or choose an NPU. Each
component root is explicit: worktrees do not inherit submodules/builds, and
source identity alone cannot establish which native libraries Python loaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def require_under(path, root, component):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"{component} loaded from {path}, expected under {root}")
    return str(path)


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": str(error)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip()
    }


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def cann_identity():
    root = os.environ.get("ASCEND_HOME_PATH")
    if not root:
        return {"available": False, "reason": "ASCEND_HOME_PATH is unset"}
    paths = [Path(root) / "version.info", Path(root) / "compiler/version.info"]
    versions = {str(path.resolve()): path.read_text() for path in paths if path.is_file()}
    return {
        "root": str(Path(root).resolve()),
        "version_files": versions,
        "available": bool(versions)
    }


def solver_identity():
    try:
        import z3
    except (ImportError, OSError) as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "version": z3.get_version_string(),
        "python": str(Path(z3.__file__).resolve())
    }


def collect(args):
    import tilelang
    import tvm
    from tvm._ffi import base

    require_under(tilelang.__file__, args.compiler_root, "TileLang Python")
    require_under(tvm.__file__, args.tvm_source, "TVM Python")
    native = getattr(tilelang, "_LIB_PATH", None)
    if native is None:
        raise RuntimeError("TileLang native loading is disabled; cannot test the compiler")
    require_under(native, args.compiler_library, "TileLang native library")
    require_under(base._LIB._name, args.tvm_library, "TVM native library")
    registered = "tl.ascend_auto_schedule" in tvm.transform.PassContext.list_configs()
    if args.require_auto_schedule and not registered:
        raise RuntimeError("loaded TileLang library does not register tl.ascend_auto_schedule")
    return {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version()
        },
        "tilelang": {
            "python": str(Path(tilelang.__file__).resolve()),
            "version": tilelang.__version__,
            "native": file_identity(native),
            "commit": command(["git", "-C", args.compiler_root, "rev-parse", "HEAD"]),
            "dirty": command(["git", "-C", args.compiler_root, "status", "--porcelain"]),
            "auto_schedule_registered": registered
        },
        "tvm": {
            "python": str(Path(tvm.__file__).resolve()),
            "version": tvm.__version__,
            "native": file_identity(base._LIB._name),
            "commit": command(["git", "-C", args.tvm_source, "rev-parse", "HEAD"]),
            "dirty": command(["git", "-C", args.tvm_source, "status", "--porcelain"])
        },
        "bisheng": command(["bisheng", "--version"]),
        "cann": cann_identity(),  # codespell:ignore cann
        "z3": solver_identity(),
        "run": {
            "platform": args.platform,
            "card": args.card,
            "case_id": args.case,
            "shape": args.shape,
            "dtype": args.dtype,
            "tiling": json.loads(args.tiling),
            "pass_configs": json.loads(args.pass_configs)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler-root", "compiler-library", "tvm-source", "tvm-library"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--require-auto-schedule", action="store_true")
    parser.add_argument("--platform", choices=("A2", "A3"), required=True)
    parser.add_argument("--card", type=int)
    parser.add_argument("--case", type=int)
    parser.add_argument("--shape", type=int, nargs="+")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tiling", default="{}", help="JSON object")
    parser.add_argument("--pass-configs", default="{}", help="JSON object")
    print(json.dumps(collect(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
