# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Selected Vector terminals must never fall through to DMA classification."""

import pytest

from tilelang.simulator import Lane, UnsupportedSimOpError, classify_operation


def test_selected_ub_copy_requires_validated_projection():
    with pytest.raises(UnsupportedSimOpError, match="validated AscendVectorSemanticProjection"):
        classify_operation("tl.ascend_copy_ub_to_ub_selected", Lane.VECTOR_0)


@pytest.mark.parametrize("name", ["add_raw", "reduce_fp32_v2", "set_mask_mode", "set_mask_payload"])
def test_unprojected_terminals_and_mask_setters_fail_closed(name):
    with pytest.raises(UnsupportedSimOpError, match="unsupported lowered"):
        classify_operation(f"tl.ascend_{name}", Lane.VECTOR_0)
