"""Control-flow tests for the additive SDK verification compatibility patch."""

import ast
import copy
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


class Tensor:

    def __init__(self, values):
        self.values = list(values)

    def all(self):
        return Tensor([all(self.values)])

    def item(self):
        return self.values[0]


def functions():
    patch = (Path(__file__).parent / "verification_repeat_finite.patch").read_text()
    hunks = patch.split("@@")
    block = next(hunk for hunk in hunks if "+def _require_finite" in hunk)
    lines = [line[1:] for line in block.splitlines() if line.startswith(("+", " "))]
    source = "\n".join(lines)
    source = source[source.index("def _require_finite"):]
    namespace = {
        "math":
            math,
        "torch":
            SimpleNamespace(
                Tensor=Tensor,
                is_floating_point=lambda _: True,
                is_complex=lambda _: False,
                isfinite=lambda value: Tensor([math.isfinite(v) for v in value.values]),
                no_grad=nullcontext),
        "_clone_value":
            copy.deepcopy,
        "_to_f64_clone":
            copy.deepcopy,
        "_normalize_output":
            lambda x: x,
        "_move_to_device":
            lambda x, _: copy.deepcopy(x),
        "_case_dtype":
            lambda _: "float16",
        "_summarize_value":
            lambda *_: ["inputs"]
    }
    compared = []

    def compare(ai, golden, native, dtype, path):
        compared.append((ai.values, golden.values, native.values, dtype, path))
        return ai.values == golden.values, "matched" if ai.values == golden.values else "mismatch"

    namespace["_compare_group_cannbench"] = compare
    exec(compile(ast.parse(source), "SDK-patch-functions", "exec"), namespace)
    return namespace, compared


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nested_nonfinite_outputs_are_rejected(value):
    namespace, _ = functions()
    with pytest.raises(ValueError, match="NaN or infinity"):
        namespace["_require_finite"]({"out": [Tensor([1, value])]}, "candidate")


def test_all_repeated_launches_are_compared_and_use_fresh_inputs():
    namespace, compared = functions()
    calls = []

    def candidate(tensor):
        calls.append(tensor.values[0])
        result = Tensor(tensor.values)
        tensor.values[0] = 99
        return result

    ok, message, _ = namespace["_run_case"](
        lambda x: Tensor(x.values),
        candidate, [Tensor([1])],
        "npu",
        42,
        check_finite=True,
        repeat_checks=3)
    assert ok and "all 3" in message
    assert calls == [1, 1, 1] and len(compared) == 3
    assert all(record[3:] == ("float16", "case[42]") for record in compared)


def test_second_launch_corruption_is_not_hidden_by_first_pass():
    namespace, compared = functions()
    calls = []

    def candidate(_):
        calls.append(1)
        return Tensor([1 if len(calls) == 1 else 2])

    ok, message, _ = namespace["_run_case"](
        lambda _: Tensor([1]), candidate, [Tensor([1])], "npu", 0, repeat_checks=3)
    assert not ok and "repeat 2/3" in message
    assert len(calls) == len(compared) == 2


def test_default_remains_one_launch_and_original_comparison_message():
    namespace, compared = functions()
    ok, message, _ = namespace["_run_case"](lambda _: Tensor([1]), lambda _: Tensor([1]),
                                            [Tensor([1])], "npu", 0)
    assert ok and message == "matched" and len(compared) == 1


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_repeat_count_is_rejected_before_any_model_call(count):
    namespace, _ = functions()

    def unexpected(_):
        raise AssertionError("invalid option must not execute models")

    with pytest.raises(ValueError, match="positive integer"):
        namespace["_run_case"](unexpected, unexpected, [Tensor([1])], "npu", 0, repeat_checks=count)
