# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from .base import BaseKernelAdapter  # noqa: F401

__all__ = [
    "BaseKernelAdapter",
    "TorchDLPackKernelAdapter",
    "CtypesKernelAdapter",
    "CythonKernelAdapter",
]


def __getattr__(name: str):
    """Load execution backends only when callers actually request them.

    In particular, importing the package for CPU simulation must not compile the
    optional Cython hardware adapter as a side effect.
    """
    if name == "TorchDLPackKernelAdapter":
        from .dlpack import TorchDLPackKernelAdapter

        return TorchDLPackKernelAdapter
    if name == "CtypesKernelAdapter":
        from .ctypes import CtypesKernelAdapter

        return CtypesKernelAdapter
    if name == "CythonKernelAdapter":
        from .cython import CythonKernelAdapter

        return CythonKernelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
