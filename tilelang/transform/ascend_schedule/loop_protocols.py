# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Reduce protocol duplication using emitted credit edges, never timing estimates."""


def remove_reciprocal_protocols(pairs):
    """Keep a distance-zero protocol instead of a duplicate carried reverse.

    Callers have validated an unconditional regular owner and each slot count.
    A(i) -> B(i) owns ready and release channels for `count` slots. Its release
    set follows B(i), and A(i+count) waits on that exact slot's release. Thus it
    already implements B(i) -> A(i+count). Only an exact endpoint/slot match is
    discharged. The analyzer's hazards remain intact; this reduces real emitted
    protocols, not the dependency report or the execution model.
    """
    return {
        key: count
        for key, count in pairs.items()
        if not (key[2] == count and pairs.get((key[1], key[0], 0)) == count)
    }
