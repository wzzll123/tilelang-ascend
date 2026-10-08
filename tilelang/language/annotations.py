# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Ascend automatic scheduling hints using the existing TVM TIR builder."""

from tvm import tir
from tvm.script.ir_builder.tir import attr, block_attr

__all__ = ["Stage", "annotate_buffer_versions"]

_VALID_BUFFER_VERSION_MODES = ("auto", "iteration", "counter")


def Stage(stage: int):
    """Assign a non-negative pipeline stage to enclosed complete tasks.

    Stage scopes constrain a scheduled child list; unannotated siblings use
    stage zero. Wrap complete tasks, placing Stage first in a multi-context
    ``with`` statement. The stage integer is retained in the AttrStmt node.
    This hint is consumed only when ``tl.ascend_auto_schedule`` is enabled.
    """
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError(f"Stage must be a non-negative integer, got {stage!r}")
    return attr(tir.IntImm("int64", stage), "tl.ascend_stage", tir.const(1, "int32"))


def annotate_buffer_versions(buffer_versions_map: dict):
    """Annotate per-storage version counts and indexing modes.

    Values are positive integer counts, ``(count, mode)`` tuples, or a mode
    string (``auto``, ``iteration``, or ``counter``). A bare count of one opts
    out of multibuffering; ``(1, "auto")`` retains eligibility with one slot.
    Mode-only entries leave the count to the scheduler. Buffer aliases sharing
    ``buffer.data`` must agree on each explicitly supplied count, mode, and
    eligibility. A bare ``1`` conflicts with a mode or explicit-mode tuple
    supplied for an alias of that storage.

    These block annotations are consumed when ``tl.ascend_auto_schedule`` is
    enabled. They do not add a leading dimension to the frontend buffer.
    Call inside a ``T.Kernel`` or explicit ``T.block`` frame, like other block
    annotations such as ``T.annotate_layout``.
    """
    versions = {}
    modes = {}
    eligibility = {}
    for buffer, specification in buffer_versions_map.items():
        if isinstance(specification, tuple):
            if len(specification) != 2:
                raise ValueError("buffer version tuple must be (num_versions, mode), "
                                 f"got {specification!r} for buffer {buffer}")
            num_versions, mode = specification
            has_num_versions = True
            has_mode = True
        elif isinstance(specification, str):
            num_versions, mode = None, specification
            has_num_versions = False
            has_mode = True
        else:
            num_versions, mode = specification, None
            has_num_versions = True
            has_mode = False

        data = buffer.data
        if has_num_versions:
            if (isinstance(num_versions, bool) or not isinstance(num_versions, int) or
                    num_versions < 1):
                raise ValueError(f"num_versions must be a positive integer, got {num_versions!r} "
                                 f"for buffer {buffer}")
            if data in versions and versions[data] != num_versions:
                raise ValueError("aliases of one buffer storage must use the same version count, "
                                 f"got {versions[data]} and {num_versions}")
            versions[data] = num_versions

        if has_mode:
            if mode not in _VALID_BUFFER_VERSION_MODES:
                raise ValueError(
                    f"buffer version mode must be one of {_VALID_BUFFER_VERSION_MODES}, "
                    f"got {mode!r} for buffer {buffer}")
            if data in modes and modes[data] != mode:
                raise ValueError("aliases of one buffer storage must use the same version mode, "
                                 f"got {modes[data]!r} and {mode!r}")
            modes[data] = mode

        eligible = has_mode or num_versions != 1
        if data in eligibility and eligibility[data] != eligible:
            raise ValueError("aliases of one buffer storage must agree on multibuffer eligibility; "
                             "bare count 1 opts out but an explicit mode retains eligibility")
        eligibility[data] = eligible

    annotations = {}
    if versions:
        annotations["tl.buffer_versions_map"] = versions
    if modes:
        annotations["tl.buffer_version_mode"] = modes
    return block_attr(annotations)
