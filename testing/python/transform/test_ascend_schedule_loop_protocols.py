# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A real ready/release protocol already implements its reciprocal reuse edge."""

import pytest


@pytest.mark.parametrize("count", [1, 2, 3])
def test_ready_release_discharge_matching_reverse_carried_edge(count):
    from tilelang.transform.ascend_schedule.loop_protocols import remove_reciprocal_protocols

    pairs = {(0, 1, 0): count, (1, 0, count): count}
    assert remove_reciprocal_protocols(pairs) == {(0, 1, 0): count}
    assert pairs == {(0, 1, 0): count, (1, 0, count): count}


def test_different_slots_or_endpoints_cannot_borrow_release_proof():
    from tilelang.transform.ascend_schedule.loop_protocols import remove_reciprocal_protocols

    pairs = {(0, 1, 0): 2, (1, 0, 1): 1, (2, 0, 2): 2}
    assert remove_reciprocal_protocols(pairs) == pairs


def test_carried_only_protocol_is_retained():
    from tilelang.transform.ascend_schedule.loop_protocols import remove_reciprocal_protocols

    assert remove_reciprocal_protocols({(1, 0, 2): 2}) == {(1, 0, 2): 2}
