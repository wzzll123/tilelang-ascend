# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Optional proofs for bounded affine old-TVM byte regions.

Inputs are byte offsets/extents, never logical element ranges. Variable bounds
are inclusive integer endpoints keyed by TIR Var identity (``same_as``). Each
typed arithmetic intermediate must fit its declared integer dtype under those
bounds. Only then is it sound to use Z3 mathematical integers instead of wrapped
machine arithmetic. Assumptions may refine disjointness but do not bypass this
conservative overflow check. Unsupported expressions always retain conflicts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProofResult:
    proved_disjoint: bool
    diagnostics: tuple[str, ...] = ()


class _Unsupported(ValueError):
    pass


def _load_z3():
    try:
        import z3
    except (ImportError, OSError):
        return None
    return z3


def _new_solver(z3, context, timeout_ms):
    solver = z3.Solver(ctx=context)
    solver.set(timeout=timeout_ms)
    return solver


def _integer_range(dtype):
    name = str(dtype)
    signed = name.startswith("int")
    prefix = "int" if signed else "uint"
    if not name.startswith(prefix) or not name[len(prefix):].isdigit():
        raise _Unsupported("only scalar integer arithmetic is supported")
    bits = int(name[len(prefix):])
    if bits < 1 or bits > 64:
        raise _Unsupported("unsupported integer bit width")
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


class _Translator:

    def __init__(self, tir, z3, context, bounds):
        self.tir, self.z3, self.context = tir, z3, context
        self.bounds, self.constraints = [], []
        entries = bounds.items() if hasattr(bounds, "items") else bounds
        for var, endpoints in entries:
            if not isinstance(var, tir.Var):
                raise _Unsupported("variable bounds must be keyed by TIR Var identity")
            if (len(endpoints) != 2 or any(type(value) is not int for value in endpoints) or
                    endpoints[0] > endpoints[1]):
                raise _Unsupported("variable bounds must be inclusive ordered integer endpoints")
            lower, upper = endpoints
            dtype_lower, dtype_upper = _integer_range(var.dtype)
            if lower < dtype_lower or upper > dtype_upper:
                raise _Unsupported("variable bounds permit integer overflow")
            previous = next((item for item in self.bounds if item[0].same_as(var)), None)
            if previous is not None:
                if (lower, upper) != previous[2:]:
                    raise _Unsupported("inconsistent bounds for the same variable identity")
                continue
            symbol = z3.Int(f"byte_var_{len(self.bounds)}", ctx=context)
            self.bounds.append((var, symbol, lower, upper))
            self.constraints.extend((symbol >= lower, symbol <= upper))

    def _checked(self, expr, value, lower, upper):
        dtype_lower, dtype_upper = _integer_range(expr.dtype)
        if lower < dtype_lower or upper > dtype_upper:
            raise _Unsupported("possible integer overflow in affine expression")
        return value, lower, upper

    def integer(self, expr):
        T, z3 = self.tir, self.z3
        if type(expr) is int:
            return z3.IntVal(expr, ctx=self.context), expr, expr
        if isinstance(expr, T.IntImm):
            value = int(expr.value)
            return self._checked(expr, z3.IntVal(value, ctx=self.context), value, value)
        if isinstance(expr, T.Var):
            for var, symbol, lower, upper in self.bounds:
                if var.same_as(expr):
                    return symbol, lower, upper
            raise _Unsupported("missing bound for variable identity")
        if isinstance(expr, (T.Add, T.Sub, T.Mul)):
            a, low_a, high_a = self.integer(expr.a)
            b, low_b, high_b = self.integer(expr.b)
            if isinstance(expr, T.Add):
                return self._checked(expr, a + b, low_a + low_b, high_a + high_b)
            if isinstance(expr, T.Sub):
                return self._checked(expr, a - b, low_a - high_b, high_a - low_b)
            # Constant means a literal operand, not a coincidentally singleton
            # variable bound or an unsupported expression folded by this helper.
            if not isinstance(expr.a, T.IntImm) and not isinstance(expr.b, T.IntImm):
                raise _Unsupported("only multiplication by an integer constant is supported")
            products = (low_a * low_b, low_a * high_b, high_a * low_b, high_a * high_b)
            return self._checked(expr, a * b, min(products), max(products))
        raise _Unsupported(f"unsupported byte expression: {type(expr).__name__}")

    def boolean(self, expr):
        T, z3 = self.tir, self.z3
        if type(expr) is bool:
            return z3.BoolVal(expr, ctx=self.context)
        if isinstance(expr, T.IntImm) and str(expr.dtype) == "bool":
            return z3.BoolVal(bool(expr.value), ctx=self.context)
        if isinstance(expr, T.Not):
            return z3.Not(self.boolean(expr.a))
        if isinstance(expr, (T.And, T.Or)):
            function = z3.And if isinstance(expr, T.And) else z3.Or
            return function(self.boolean(expr.a), self.boolean(expr.b))
        comparisons = ((T.EQ, lambda a, b: a == b), (T.NE, lambda a, b: a != b),
                       (T.LT, lambda a, b: a < b), (T.LE, lambda a, b: a <= b),
                       (T.GT, lambda a, b: a > b), (T.GE, lambda a, b: a >= b))
        for kind, function in comparisons:
            if isinstance(expr, kind):
                return function(self.integer(expr.a)[0], self.integer(expr.b)[0])
        raise _Unsupported(f"unsupported region assumption: {type(expr).__name__}")


def prove_byte_regions_disjoint(offset_a,
                                extent_a,
                                offset_b,
                                extent_b,
                                variable_bounds,
                                assumptions=(),
                                timeout_ms=1000) -> ProofResult:
    """Prove no overlap for any bounded valuation; SAT/unknown retain conflict.

    Missing Z3/TVM, unsupported nodes, missing bounds, negative possible extents,
    possible arithmetic overflow, and solver timeout return explicit diagnostics.
    This helper is deliberately not connected to dependency pruning.
    """
    if type(timeout_ms) is not int or timeout_ms <= 0:
        return ProofResult(False, ("timeout_ms must be a positive integer",))
    z3 = _load_z3()
    if z3 is None:
        return ProofResult(False, ("optional z3-solver package is unavailable",))
    try:
        from tvm import tir
    except (ImportError, OSError):
        return ProofResult(False, ("old TVM TIR runtime is unavailable",))
    try:
        context = z3.Context()
        translator = _Translator(tir, z3, context, variable_bounds)
        a, _, _ = translator.integer(offset_a)
        size_a, minimum_a, _ = translator.integer(extent_a)
        b, _, _ = translator.integer(offset_b)
        size_b, minimum_b, _ = translator.integer(extent_b)
        if minimum_a < 0 or minimum_b < 0:
            raise _Unsupported("negative or potentially negative byte extent is invalid")
        conditions = tuple(translator.boolean(condition) for condition in assumptions)
        solver = _new_solver(z3, context, timeout_ms)
        solver.add(*translator.constraints, *conditions)
        # Endpoints here are mathematical byte-range limits, not an additional
        # TIR Add node. Every actual TIR arithmetic node was checked above.
        solver.add(size_a > 0, size_b > 0, a < b + size_b, b < a + size_a)
        status = solver.check()
        if status == z3.unsat:
            return ProofResult(True, ("UNSAT: bounded byte regions cannot overlap",))
        if status == z3.sat:
            return ProofResult(False, ("SAT: bounded byte regions may overlap",))
        return ProofResult(False, (f"solver unknown: {solver.reason_unknown()}",))
    except (_Unsupported, z3.Z3Exception) as error:
        return ProofResult(False, (str(error),))
