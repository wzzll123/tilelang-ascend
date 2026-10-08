# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Unproven physical slot formulas must not remove carried memory hazards."""

import pytest
import tilelang  # noqa: F401
from tvm import tir
from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
from tilelang.transform.ascend_schedule.workspace_proof import certify_workspace_versions, VERSION_ATTR


@pytest.mark.parametrize("case", ["missing_slot", "wrong_pitch", "unknown_helper"])
def test_invalid_slot_certificate_retains_unknown_footprint(case):
    workspace = tir.decl_buffer((32, 16), "float16", name="workspace", scope="global")
    ub = tir.decl_buffer((16, 16), "float16", name="ub", scope="shared.ub")
    handle = tir.Var("workspace_handle", "handle")
    epoch = tir.Var("epoch", "int32")
    offset = tir.floormod(epoch, 2) * 256
    if case == "missing_slot":
        offset = tir.IntImm("int32", 0)
    tag = "copy_ub_to_gm<half,16,16>"
    if case == "unknown_helper":
        tag = "atomic_add_ub_to_gm<half,16,16>"
    call = tir.call_extern("handle", tag, ub.access_ptr("r"),
                           workspace.access_ptr("w", offset=offset, extent=256), 16, 16, 16)
    body = tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, tir.Evaluate(call))
    loop = tir.For(epoch, 0, 4, tir.ForKind.SERIAL, body)
    function = tir.PrimFunc([handle], loop, buffer_map={
        handle: workspace
    }).with_attr(VERSION_ATTR, {
        workspace.data: [2, 128 if case == "wrong_pitch" else 256, epoch, 0]
    }).with_attr("auto_gm_indices", [0])
    extraction = extract_tasks(loop.body)
    proven, diagnostics = certify_workspace_versions(function, extraction)
    assert diagnostics
    original = next(
        a for a in extraction.tasks[0].accesses if a.storage.var.same_as(workspace.data))
    result = next(a for a in proven.tasks[0].accesses if a.storage.var.same_as(workspace.data))
    assert result is original
    assert result.ring_size is None
