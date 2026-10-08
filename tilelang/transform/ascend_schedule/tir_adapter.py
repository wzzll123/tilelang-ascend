# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative extraction from the repository's older ``tvm.tir`` API.

No statements are rewritten. Scalar bindings (including condition snapshots)
are explicit tasks and metadata so a later reconstruction must evaluate them
at their original position. Unknown statements/externs fail closed. Footprints
are bytes, with symbolic offsets/extents deliberately left unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from .model import Access, Task


class ExtractionError(ValueError):
    """Lowered TIR cannot be modeled without losing execution semantics."""


@dataclass(frozen=True)
class OperationConfig:
    accesses: tuple
    pipe: str


def normalize_operation(name: str) -> str:
    name = name.split("<", 1)[0]
    if name.startswith("tl::ascend::"):
        name = name[len("tl::ascend::"):]
    return name


def load_operation_configs(path: Path | None = None) -> dict:
    """Read pipe/access declarations from the checked-in authoritative table."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "src/transform/common/operation_config.h"
    source = Path(path).read_text()
    entries = re.findall(r'\{"([^"\n]+)",\s*\{\{(.*?)\},\s*"(PIPE_[A-Z0-9]+)"\}\}', source, re.S)
    result = {}
    for name, accesses, pipe in entries:
        pairs = re.findall(r'\{(\d+),\s*"(read|write)"\}', accesses)
        result[name] = OperationConfig(
            tuple((int(index), "R" if mode == "read" else "W") for index, mode in pairs), pipe)
    if not result:
        raise ExtractionError(f"No operation configurations found in {path}")
    return result


def operation_config(name: str, configs: dict | None = None) -> OperationConfig:
    normalized = normalize_operation(name)
    if normalized in ("gemm_v0", "gemm_v1", "tl.ascend_gemm_v0", "tl.ascend_gemm_v1"):
        raise ExtractionError(
            f"{name} is a multi-pipe function; decompose it into copy/mma tasks before scheduling")
    configs = load_operation_configs() if configs is None else configs
    if normalized not in configs:
        raise ExtractionError(f"Unknown lowered extern operation: {name}")
    return configs[normalized]


def dtype_bytes(dtype: str) -> int | None:
    match = re.fullmatch(r"(?:u?int|float|bfloat)(\d+)(?:x(\d+))?", str(dtype))
    if not match:
        return None
    bits, lanes = int(match[1]), int(match[2] or 1)
    return bits // 8 * lanes if bits % 8 == 0 else None


def mask_mode(mask: int | None) -> str:
    if mask not in (1, 2, 3):
        raise ExtractionError(f"Unsupported or symbolic access-pointer mask: {mask}")
    return {1: "R", 2: "W", 3: "RW"}[mask]


@dataclass(eq=False, frozen=True)
class Storage:
    id: int
    var: Any = field(repr=False)


class StorageInterner:
    """Canonical tokens compared by identity; wrappers are compared by same_as."""

    def __init__(self):
        self.storages = []

    def intern(self, var) -> Storage:
        for storage in self.storages:
            if storage.var.same_as(var):
                return storage
        storage = Storage(len(self.storages), var)
        self.storages.append(storage)
        return storage


@dataclass(frozen=True)
class ScalarBinding:
    task_id: int
    storage: Storage
    expression: Any
    var: Any = None
    kind: str = "let"


@dataclass
class Extraction:
    tasks: tuple
    task_nodes: dict
    stages: dict
    loops: dict
    scalar_bindings: tuple
    storages: tuple
    annotations: tuple


def extract_tasks(node) -> Extraction:
    """Extract lexical tasks, guards, scalar dependencies, stages, and loops.

    Guards are tuples of ``(snapshot_storage, branch_truth)``. They are opaque
    ordering metadata, not proof that two tasks never conflict. Every task in a
    branch reads its snapshot token. Bindings retain the original expression
    and statement for reconstruction; this function does not insert snapshots
    into the input TIR or assert initialization/full-overwrite proofs.
    """
    from tvm import tir

    extractor = _Extractor(tir)
    if isinstance(node, tir.PrimFunc):
        for buffer in node.buffer_map.values():
            extractor.pool.intern(buffer.data)
        node = node.body
    extractor.walk(node)
    return Extraction(
        tuple(extractor.tasks), extractor.nodes, extractor.stages, extractor.loops,
        tuple(extractor.bindings), tuple(extractor.pool.storages), tuple(extractor.annotations))


class _Extractor:

    def __init__(self, tir):
        self.tir = tir
        self.pool = StorageInterner()
        self.configs = load_operation_configs()
        self.tasks, self.bindings, self.annotations = [], [], []
        self.nodes, self.stages, self.loops = {}, {}, {}
        self.stage, self.core, self.guard, self.loop_stack = 0, None, (), ()

    def integer(self, expr):
        return int(expr.value) if isinstance(expr, self.tir.IntImm) else None

    def add_task(self, node, pipe, core, accesses):
        accesses = list(accesses)
        accesses.extend(Access(storage, "R") for storage, _ in self.guard)
        task_id = len(self.tasks)
        self.tasks.append(Task(task_id, pipe, core, tuple(accesses), self.guard or None))
        self.nodes[task_id] = node
        self.stages[task_id] = self.stage
        self.loops[task_id] = self.loop_stack
        return task_id

    def expression_reads(self, expression, include_pointers=True):
        """Record scalar Vars and physical BufferLoad reads without name aliasing."""
        reads = []
        seen_vars = set()

        def visit(expr):
            if isinstance(expr, self.tir.Call):
                if expr.op.name in ("tir.call_extern", "tir.call_packed",
                                    "tir.call_cpacked", "tir.tvm_call_trace_packed") \
                        or expr.op.name.startswith("tl."):
                    raise ExtractionError(
                        "Nested external call in scalar expression requires decomposition")
                if include_pointers and expr.op.name == "tir.tvm_access_ptr":
                    reads.append(self.pointer_access(expr, ""))
            if isinstance(expr, self.tir.BufferLoad):
                reads.append(self.buffer_access(expr.buffer, expr.indices, "R"))
            elif isinstance(expr, self.tir.Var) and str(expr.dtype) != "handle":
                storage = self.pool.intern(expr)
                if storage.id not in seen_vars:
                    seen_vars.add(storage.id)
                    reads.append(Access(storage, "R"))

        self.tir.stmt_functor.post_order_visit(expression, visit)
        return reads

    def buffer_access(self, buffer, indices, mode):
        storage = self.pool.intern(buffer.data)
        size = dtype_bytes(buffer.dtype)
        offset = 0
        extents = [self.integer(extent) for extent in buffer.shape]
        positions = [self.integer(index) for index in indices]
        if buffer.strides:
            strides = [self.integer(stride) for stride in buffer.strides]
        else:
            strides = []
            stride = 1
            for extent in reversed(extents):
                strides.insert(0, stride)
                stride = None if extent is None or stride is None else stride * extent
        element_offset = self.integer(buffer.elem_offset)
        if (size is None or element_offset is None or
                any(value is None for value in positions + strides)):
            return Access(storage, mode, alias=buffer)
        offset = (element_offset + sum(i * s for i, s in zip(positions, strides))) * size
        return Access(storage, mode, offset, size, alias=buffer)

    def pointer_access(self, pointer, configured_mode):
        if (not isinstance(pointer, self.tir.Call) or pointer.op.name != "tir.tvm_access_ptr" or
                len(pointer.args) != 5):
            raise ExtractionError("Configured buffer argument is not a lowered tvm_access_ptr")
        dtype, var, offset, extent, mask = pointer.args
        if not isinstance(var, self.tir.Var):
            raise ExtractionError("Indirect access pointer storage requires alias analysis")
        mode = mask_mode(self.integer(mask))
        # Preserve declared semantics as well as mask=3, including in-place operations.
        mode = "RW" if "R" in mode + configured_mode and "W" in mode + configured_mode else mode
        size = dtype_bytes(dtype.dtype)
        start, count = self.integer(offset), self.integer(extent)
        if count is not None and count < 0:
            raise ExtractionError("Negative access-pointer extent")
        if size is None or start is None or count is None:
            start, count = None, None
        else:
            start, count = start * size, count * size
        return Access(self.pool.intern(var), mode, start, count, alias=var)

    def bind(self, node, expression, var=None, kind="let"):
        if var is None:
            # A synthetic condition variable represents a single evaluation at entry.
            var = self.tir.Var(f"ascend_guard_{len(self.bindings)}", "bool")
        storage = self.pool.intern(var)
        task_id = self.add_task(node, "PIPE_S", self.core or "C",
                                self.expression_reads(expression) + [Access(storage, "W")])
        self.bindings.append(ScalarBinding(task_id, storage, expression, var, kind))
        return storage

    def operation(self, node):
        call = node.value
        if call.op.name == "tir.call_extern":
            name = str(call.args[0].value)
            arguments = call.args[1:]
        else:
            # Intrinsic configuration indices already include any template
            # StringImm in arg0 (mma/fill/createvecindex). Binary vector APIs
            # have no template argument. See codegen_ascend.cc and ascend_tile.py.
            name = call.op.name
            arguments = call.args
        config = operation_config(name, self.configs)
        accesses = []
        for index, mode in config.accesses:
            if index >= len(arguments):
                raise ExtractionError(f"Missing configured buffer argument {index} in {name}")
            argument = arguments[index]
            if (isinstance(argument, self.tir.Call) and argument.op.name == "tir.tvm_access_ptr"):
                accesses.append(self.pointer_access(argument, mode))
            elif mode == "R":
                # Scalar vector variants declare reads in the shared table;
                # the expression contributes scalar/BufferLoad dependencies.
                accesses.extend(self.expression_reads(argument))
            else:
                raise ExtractionError(
                    f"Configured write argument {index} in {name} is not tvm_access_ptr")
        configured_indices = {index for index, _ in config.accesses}
        for index, argument in enumerate(arguments):
            accesses.extend(
                self.expression_reads(argument, include_pointers=index not in configured_indices))
        core = self.operation_core(normalize_operation(name), config.pipe)
        self.add_task(node, config.pipe, core, accesses)

    def operation_core(self, name, pipe):
        if pipe in ("PIPE_M", "PIPE_FIX", "PIPE_MTE1"):
            inferred = "C"
        elif pipe == "PIPE_V" or "ub" in name.lower():
            inferred = "V"
        elif any(scope in name.lower() for scope in ("l1", "l0", "bt")):
            inferred = "C"
        elif self.core is not None:
            inferred = self.core
        else:
            raise ExtractionError(f"Cannot determine core for lowered operation {name}")
        if self.core is not None and self.core != inferred:
            raise ExtractionError(f"Operation {name} conflicts with resource_scope {self.core}")
        return inferred

    def walk(self, node):
        T = self.tir
        if isinstance(node, T.SeqStmt):
            for child in node.seq:
                self.walk(child)
        elif isinstance(node, T.For):
            previous = self.loop_stack
            self.loop_stack += (node,)
            self.walk(node.body)
            self.loop_stack = previous
        elif isinstance(node, T.IfThenElse):
            snapshot = self.bind(node, node.condition, kind="guard")
            previous = self.guard
            self.guard += ((snapshot, True),)
            self.walk(node.then_case)
            if node.else_case is not None:
                self.guard = previous + ((snapshot, False),)
                self.walk(node.else_case)
            self.guard = previous
        elif isinstance(node, T.LetStmt):
            self.bind(node, node.value, node.var)
            self.walk(node.body)
        elif isinstance(node, T.AttrStmt):
            stage, core = self.stage, self.core
            if node.attr_key == "tl.ascend_stage":
                value = self.integer(node.node)
                if value is None or value < 0:
                    raise ExtractionError("Stage attribute must contain a nonnegative integer node")
                self.stage = value
            elif node.attr_key == "resource_scope":
                value = self.integer(node.value)
                if value not in (0, 1):
                    raise ExtractionError("Unsupported resource_scope")
                self.core = "C" if value == 0 else "V"
            self.annotations.append(node)
            self.walk(node.body)
            self.stage, self.core = stage, core
        elif isinstance(node, T.BlockRealize):
            previous = self.guard
            for variable, value in zip(node.block.iter_vars, node.iter_values):
                self.bind(node, value, variable.var, "block_iter")
            if self.integer(node.predicate) != 1:
                snapshot = self.bind(node, node.predicate, kind="guard")
                self.guard += ((snapshot, True),)
            self.annotations.append(node.block.annotations)
            if node.block.init is not None:
                self.walk(node.block.init)
            self.walk(node.block.body)
            self.guard = previous
        elif isinstance(node, T.Allocate):
            self.pool.intern(node.buffer_var)
            previous = self.guard
            if self.integer(node.condition) != 1:
                snapshot = self.bind(node, node.condition, kind="allocation_guard")
                self.guard += ((snapshot, True),)
            self.walk(node.body)
            self.guard = previous
        elif isinstance(node, T.BufferStore):
            reads = self.expression_reads(node.value)
            for index in node.indices:
                reads.extend(self.expression_reads(index))
            self.add_task(node, "PIPE_S", self.core or "C",
                          reads + [self.buffer_access(node.buffer, node.indices, "W")])
        elif isinstance(node, T.Evaluate):
            if isinstance(node.value, T.Call):
                self.operation(node)
            elif isinstance(node.value, T.IntImm):
                return
            else:
                raise ExtractionError(f"Unsupported lowered Evaluate: {node.value}")
        else:
            raise ExtractionError(f"Unsupported lowered statement: {type(node).__name__}")
