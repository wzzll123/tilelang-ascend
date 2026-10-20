# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Versioned contracts for the FIA expert kernel's opaque vector helpers.

This is not a C++ interpreter. Only the five declared call forms are accepted.
The numerical contract follows SoftmaxFlashV2 WITHOUT_BRC (FP32, 512 columns;
v1: 16 rows, v2: 8/16/32 rows), and the exact RowOp/Tail wrappers in gqa_vector.h.
V2 selects one bank of a one/two-bank score allocation. Scratch footprints are
conservative clobbers, not initialized inputs. All work stays on PIPE_V;
internal vector barriers do not create cross-pipe or cross-lane ordering.
"""

from dataclasses import replace
import re

import numpy as np

from .errors import ProgramValidationError, UnsupportedSimOpError
from .program import BufferRegion, MemoryScope


_CALLS = {
    "flash_init": "fia_aligned::Flash<false>(score,maxv,sums,alpha,denom,scratch,sm_slot,sm_row);",
    "flash_update": "fia_aligned::Flash<true>(score,maxv,sums,alpha,denom,scratch,sm_slot,sm_row);",
    "rescale": "fia_aligned::Rescale(acc_o,alpha,brcb,sm_slot);",
    "normalize": "fia_aligned::Normalize(acc_o,denom,brcb);",
}


def validate_contract(tag, source):
    if isinstance(tag, str) and tag.startswith("sim.fia.v2."):
        kind = tag[len("sim.fia.v2.") :]
        compact = re.sub(r"\s+", "", source)
        if kind == "normalize_divmul" and compact == "fia_aligned::NormalizeDivMul(acc_o,denom,brcb);":
            return kind
        if kind in ("flash_init", "flash_update"):
            update = "false" if kind == "flash_init" else "true"
            pattern = (
                rf"fia_aligned::Flash<{update},(8|16|32),(8192|16384)>"
                r"\(score\[sm_bank\*(\d+)\],maxv,sums,alpha,denom,scratch,sm_slot,sm_row\);"
            )
            match = re.fullmatch(pattern, compact)
            if match and int(match.group(3)) == int(match.group(1)) * 512:
                return kind
        elif kind == "tail":
            match = re.fullmatch(r"fia_aligned::Tail<(\d+),(8|16|32)>\(score\[sm_bank\*(\d+)\],tail_col\);", compact)
            if match and 1 <= int(match.group(1)) < 512 and int(match.group(3)) == int(match.group(2)) * 512:
                return kind
        raise UnsupportedSimOpError(f"opaque source does not match {tag!r}")
    prefix = "sim.fia.v1."
    if not isinstance(tag, str) or not tag.startswith(prefix):
        raise UnsupportedSimOpError(f"unknown opaque helper contract {tag!r}")
    kind = tag[len(prefix) :]
    compact = re.sub(r"\s+", "", source)
    if kind == "tail":
        match = re.fullmatch(r"fia_aligned::Tail<(\d+)\>\(score,tail_col\);", compact)
        valid = int(match.group(1)) if match else -1
        if not 1 <= valid < 512:
            raise UnsupportedSimOpError("opaque tail source does not match its v1 contract")
    elif kind not in _CALLS or compact != _CALLS[kind]:
        raise UnsupportedSimOpError(f"opaque source does not match {tag!r}")
    return kind


def _window(region, offset, size):
    # offset and size are contract scalars already resolved by the bridge.
    return replace(region, shape=(size,), byte_offset=region.byte_offset + offset * 4, strides_bytes=None)


def helper_metadata(kind, regions, *, slot=0, row=0, valid=512, rows=16, work=49152, version=1):
    """Build exact logical reads/writes, preserving physical buffer aliases."""
    if kind not in {*_CALLS, "tail", "normalize_divmul"} or (version == 1 and kind == "normalize_divmul"):
        raise UnsupportedSimOpError(f"unknown FIA helper {kind!r}")
    if not isinstance(slot, int) or not 0 <= slot < 3:
        raise UnsupportedSimOpError("FIA v1 slot must resolve to an integer in [0,3)")
    if version not in (1, 2) or rows not in ((16,) if version == 1 else (8, 16, 32)):
        raise UnsupportedSimOpError("FIA unsupported contract version/rows")
    if not isinstance(row, int) or row % rows or not 0 <= row <= 64 - rows:
        raise UnsupportedSimOpError("FIA row must select an aligned slice within 64 rows")
    if version == 2 and (kind not in ("flash_init", "flash_update", "tail", "normalize_divmul") or work not in (8192, 16384)):
        raise UnsupportedSimOpError("FIA v2 supports Flash/Tail and work=8192/16384 only")
    if version == 1 and work != 49152:
        raise UnsupportedSimOpError("FIA v1 requires work=49152")
    needed = (
        ("score", "maxv", "sums", "alpha", "denom", "scratch")
        if kind.startswith("flash")
        else (("score", "col") if kind == "tail" else ("dst", "alpha" if kind == "rescale" else "denom", "brcb"))
    )
    expected = {
        "score": ((rows, 512), "float32"),
        "maxv": ((64,), "float32"),
        "sums": ((192,), "float32"),
        "alpha": ((192,), "float32"),
        "denom": ((64,), "float32"),
        "scratch": ((work,), "uint8"),
        "dst": ((64, 128), "float32"),
        "brcb": ((64, 8), "float32"),
        "col": ((512,), "float32"),
    }
    for name in needed:
        r = regions.get(name)
        if not isinstance(r, BufferRegion) or r.scope is not MemoryScope.UB:
            raise ProgramValidationError(f"FIA {name} must be a UB region")
        shape, dtype = expected[name]
        if tuple(r.shape) != shape or r.dtype != dtype:
            raise ProgramValidationError(f"FIA {name} requires {shape}/{dtype}")
    m = {
        "fia_helper": kind,
        "helper_contract": f"sim.fia.v{version}.{kind}",
        "timing_key": f"sim.fia.v{version}.{kind}",
        "helper_rows": rows,
        "helper_work_bytes": work,
        "helper_timing_provenance": "uncalibrated analytical vector primitive count",
        "helper_internal_pipe": "v",
        "helper_internal_barriers": "PIPE_V only",
    }
    if kind.startswith("flash"):
        score = regions["score"]
        max_out = _window(regions["maxv"], row, rows)
        sum_out = _window(regions["sums"], slot * 64 + row, rows)
        alpha_out = _window(regions["alpha"], slot * 64 + row, rows)
        old_sum = _window(regions["sums"], ((slot + 2) % 3) * 64 + row, rows)
        reads, writes = [score], [score, max_out, sum_out]
        if kind == "flash_update":
            reads.extend((max_out, old_sum))
            # CANN 9.0 v220 M1 basic-block implementation scales inExpSum
            # in place before adding it to the current block's reduction.
            # The argument is a const LocalTensor descriptor, not const data.
            writes.extend((alpha_out, old_sum))
        m.update(
            dst=score,
            max_out=max_out,
            sum_out=sum_out,
            alpha_out=alpha_out,
            old_sum=old_sum,
            scratch=regions["scratch"],
            helper_vector_passes=6 if kind == "flash_init" else 10,
        )
    elif kind == "tail":
        if not isinstance(valid, int) or not 1 <= valid < 512:
            raise ProgramValidationError("FIA tail valid must be in [1,512)")
        reads, writes = [regions["score"]], [regions["score"], regions["col"]]
        m.update(dst=regions["score"], col=regions["col"], valid=valid, helper_vector_passes=2)
    else:
        factor = _window(regions["alpha"], slot * 64, 64) if kind == "rescale" else regions["denom"]
        reads, writes = [regions["dst"], factor], [regions["dst"], regions["brcb"]]
        if kind == "normalize_divmul":
            writes.append(factor)
        m.update(dst=regions["dst"], factor=factor, brcb=regions["brcb"], helper_vector_passes=4 if kind == "normalize_divmul" else 2)
    m.update(src_regions=tuple(reads), dst_regions=tuple(writes))
    return m


def execute_helper(simulator, task):
    """Functional payload; sync-only uses the common poison metadata path."""
    m, core = task.metadata, task.core_id

    def read(region):
        return simulator.read(region, task_core_id=core)

    def write(region, values):
        simulator.write(region, values, task_core_id=core)

    kind = m["fia_helper"]
    dst = m["dst"]
    if kind.startswith("flash"):
        score = read(dst)
        new_max = score.max(axis=1)
        if kind == "flash_update":
            old_max, old_sum = read(m["max_out"]), read(m["old_sum"])
            new_max = np.maximum(old_max, new_max)
            alpha = np.exp(old_max - new_max)
        probability = np.exp(score - new_max[:, None])
        sums = probability.sum(axis=1, dtype=np.float32)
        if kind == "flash_update":
            scaled_sum = alpha * old_sum
            sums += scaled_sum
            write(m["alpha_out"], alpha)
            write(m["old_sum"], scaled_sum)
        write(dst, probability)
        write(m["max_out"], new_max)
        write(m["sum_out"], sums)
        # Workspace contents are unspecified, but prior contents are not read.
        view = simulator._resolve(m["scratch"], core)
        view.allocation.mark_initialized(view, False)
    elif kind == "tail":
        col = np.full(512, -np.inf, np.float32)
        col[: m["valid"]] = 0
        write(dst, read(dst) + col[None, :])
        write(m["col"], col)
    else:
        factor = read(m["factor"])
        value = read(dst)
        if kind == "normalize_divmul":
            factor = np.float32(1) / factor
            write(m["factor"], factor)
        value = value * factor[:, None] if kind in ("rescale", "normalize_divmul") else value / factor[:, None]
        write(dst, value)
        write(m["brcb"], np.repeat(factor[:, None], 8, axis=1))
