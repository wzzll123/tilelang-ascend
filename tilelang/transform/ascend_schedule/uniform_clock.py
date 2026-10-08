# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Certificates for bounded, immutable clocks shared by one C/V core group."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UniformClock:
    analyzer: object
    extent_min: int
    extent_max: int


def certify_uniform_clock(function, owner):
    """Allow constants and dominating blockIdx.x arithmetic, never participant IDs.

    Check every intermediate before accepting the final bound: a saturating Min
    does not make an overflowing multiply safe. Sibling thread attributes do not
    establish a range. Unbound parameters, memory loads and calls fail closed.
    The certificate does not establish guards, task lifetimes or slot ownership.
    """
    from tvm import DataType, arith, ir, tir

    if not isinstance(owner, tir.For) or owner.kind != tir.ForKind.SERIAL:
        raise ValueError("uniform clock requires a serial owner")
    paths = []

    def find(node, ancestors):
        if node.same_as(owner):
            paths.append(ancestors)
            return
        children = []
        if isinstance(
                node,
            (tir.AttrStmt, tir.Allocate, tir.Block, tir.For, tir.LetStmt, tir.DeclBuffer)):
            children = [node.body]
        elif isinstance(node, tir.BlockRealize):
            children = [node.block]
        elif isinstance(node, tir.SeqStmt):
            children = list(node.seq)
        elif isinstance(node, tir.IfThenElse):
            children = [node.then_case] + ([node.else_case] if node.else_case is not None else [])
        for child in children:
            find(child, (*ancestors, node))

    find(function.body, ())
    if len(paths) != 1:
        raise ValueError("uniform clock needs one unique owner ancestor path")
    analyzer = arith.Analyzer()
    uniform, thread_variables = [], []

    def integer_limits(expression):
        dtype = DataType(expression.dtype)
        if dtype.lanes != 1 or not str(dtype).startswith(("int", "uint")):
            raise ValueError("uniform clock requires scalar integer expressions")
        signed = str(dtype).startswith("int")
        return (-(1 << (dtype.bits - 1)) if signed else 0, (1 << (dtype.bits - int(signed))) - 1)

    for ancestor in paths[0]:
        if (isinstance(ancestor, tir.AttrStmt) and ancestor.attr_key == "thread_extent" and
                isinstance(ancestor.node, tir.IterVar)):
            if any(ancestor.node.var.same_as(var) for var in thread_variables):
                raise ValueError("uniform clock has conflicting thread variable bindings")
            thread_variables.append(ancestor.node.var)
        if (isinstance(ancestor, tir.AttrStmt) and ancestor.attr_key == "thread_extent" and
                isinstance(ancestor.node, tir.IterVar) and
                ancestor.node.thread_tag == "blockIdx.x"):
            if not isinstance(ancestor.value, tir.IntImm) or int(ancestor.value) <= 0:
                raise ValueError("uniform clock needs a positive static core-group range")
            var = ancestor.node.var
            _, high = integer_limits(var)
            if int(ancestor.value) - 1 > high or uniform:
                raise ValueError("uniform clock core-group range overflows or is ambiguous")
            uniform.append(var)
            analyzer.bind(var, ir.Range.from_min_extent(0, ancestor.value))

    binary = (tir.Add, tir.Sub, tir.Mul, tir.FloorDiv, tir.FloorMod, tir.Min, tir.Max)

    def check(expression):
        if isinstance(expression, tir.IntImm):
            pass
        elif isinstance(expression, tir.Var):
            if not any(expression.same_as(var) for var in uniform):
                raise ValueError("uniform clock contains a participant or unbound variable")
        elif isinstance(expression, binary):
            check(expression.a)
            check(expression.b)
            if isinstance(expression, (tir.FloorDiv, tir.FloorMod)) and analyzer.const_int_bound(
                    expression.b).min_value <= 0:
                raise ValueError("uniform clock divisor is not proven positive")
        elif isinstance(expression, tir.Cast):
            check(expression.value)
            low, high = integer_limits(expression)
            source = analyzer.const_int_bound(expression.value)
            if source.min_value < low or source.max_value > high:
                raise ValueError("uniform clock cast may overflow")
        else:
            raise ValueError("uniform clock expression is unsupported or reads mutable data")
        low, high = integer_limits(expression)
        bound = analyzer.const_int_bound(expression)
        if bound.min_value < low or bound.max_value > high:
            raise ValueError("uniform clock intermediate arithmetic may overflow")

    check(owner.min)
    check(owner.extent)
    minimum = analyzer.const_int_bound(owner.min)
    extent = analyzer.const_int_bound(owner.extent)
    low, high = integer_limits(owner.loop_var)
    if (extent.min_value < 0 or minimum.min_value < low or
            # Native for termination computes min+extent, and the final ++ must
            # fit too. Last-index-only bounds would admit INT_MAX + one trip.
            minimum.max_value + extent.max_value > high or
            minimum.max_value - minimum.min_value + max(0, extent.max_value - 1) > high):
        raise ValueError("uniform clock owner index or relative clock may overflow")
    return UniformClock(analyzer, int(extent.min_value), int(extent.max_value))
