# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import importlib.util
from pathlib import Path
import re

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "schedule_resources", _ROOT / "tilelang/transform/ascend_schedule/resources.py")
resources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resources)


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_budgets_match_existing_compiler_reservations(platform):
    source = (_ROOT / "src/transform/ascend_memory_planning.cc").read_text()
    definitions = {
        key: int(value)
        for key, value in re.findall(r"#define (ASCEND_[A-Z_]+_MEM_SIZE) (\d+)", source)
    }
    scopes = {
        "UB": "ASCEND_SHARED_MEM_SIZE",
        "L1": "ASCEND_SHARED_DYN_MEM_SIZE",
        "L0A": "ASCEND_WMMA_MATRIX_A_MEM_SIZE",
        "L0B": "ASCEND_WMMA_MATRIX_B_MEM_SIZE",
        "L0C": "ASCEND_WMMA_ACCUMULATOR_MEM_SIZE",
        "BT": "ASCEND_BT_MEM_SIZE",
    }
    assert resources.allocation_budgets(platform) == {
        scope: definitions[name] for scope, name in scopes.items()
    }


def test_target_must_not_accidentally_use_a5():
    with pytest.raises(ValueError, match="A2 or A3"):
        resources.allocation_budgets("A5")


def test_budget_mutation_cannot_change_next_compilation():
    budgets = resources.allocation_budgets("A2")
    budgets["UB"] = 1
    assert resources.allocation_budgets("A2")["UB"] == 196352
