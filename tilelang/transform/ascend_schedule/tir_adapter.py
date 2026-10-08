# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative extraction from the repository's older ``tvm.tir`` API.

No statements are rewritten. Scalar bindings (including condition snapshots)
are explicit tasks and metadata so a later reconstruction must evaluate them
at their original position. Unknown statements/externs fail closed. Footprints
are bytes, with symbolic offsets/extents deliberately left unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any

from .model import Access, Task
from .conditions import CONDITION_SNAPSHOT


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
    # CopyNode::Lower marks these helpers virtual_channel and passes workspace:
    # A2/A3 execute a GM hop across cores/pipes, not an atomic L1/UB transfer.
    if normalized in ("copy_ub_to_l1", "copy_l0c_to_ub"):
        raise ExtractionError(f"{name} is a multi-pipe cross-core GM-hop composite; "
                              "decompose it into physical copy tasks before scheduling")
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
    domain: str | None = None


class StorageInterner:
    """Canonical physical or per-core scalar tokens using Var.same_as identity.

    domain=None denotes physical buffer storage. Scalar Vars and local buffers
    are private to their C/V execution domain, even with the same TIR Var.
    """

    def __init__(self):
        self.storages = []

    def intern(self, var, domain=None) -> Storage:
        for storage in self.storages:
            if storage.domain == domain and storage.var.same_as(var):
                return storage
        storage = Storage(len(self.storages), var, domain)
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
    Unscoped scalar control can be assigned only when all known tasks identify
    one execution core; mixed/unknown domains require control normalization.
    """
    from tvm import tir

    extractor = _Extractor(tir)
    if isinstance(node, tir.PrimFunc):
        for buffer in node.buffer_map.values():
            extractor.memory_storage(buffer.data, buffer.scope())
        node = node.body
    extractor.walk(node)
    extractor.resolve_domains()
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
        self.snapshot_vars = ()

    def integer(self, expr):
        return int(expr.value) if isinstance(expr, self.tir.IntImm) else None

    def scalar_storage(self, var):
        return self.pool.intern(var, self.core or "UNRESOLVED")

    def memory_storage(self, var, scope=None):
        if scope is None:
            annotation = getattr(var, "type_annotation", None)
            scope = getattr(annotation, "storage_scope", "")
        domain = (self.core or "UNRESOLVED") if str(scope).startswith("local") else None
        return self.pool.intern(var, domain)

    def resolve_domains(self):
        """Infer only a single proven core; never invent scalar C ownership."""
        cores = {task.core for task in self.tasks if task.core != "UNRESOLVED"}
        unresolved = any(task.core == "UNRESOLVED" for task in self.tasks)
        if unresolved and len(cores) != 1:
            raise ExtractionError(
                "Unscoped scalar tasks require per-core control normalization; "
                f"cannot assign execution ownership from cores {sorted(cores)}. "
                "Use explicit resource_scope or decompose/duplicate mixed-core controls.")
        inferred = next(iter(cores)) if len(cores) == 1 else None
        remap = {}
        if inferred is not None:
            for storage in tuple(self.pool.storages):
                if storage.domain == "UNRESOLVED":
                    remap[storage] = self.pool.intern(storage.var, inferred)
        rewritten = []
        for task in self.tasks:
            core = inferred if task.core == "UNRESOLVED" else task.core
            accesses = tuple(
                replace(access, storage=remap.get(access.storage, access.storage))
                for access in task.accesses)
            guard = (
                tuple((remap.get(storage, storage), branch) for storage, branch in task.guard)
                if task.guard else None)
            if any(access.storage.domain not in (None, core) for access in accesses):
                raise ExtractionError(
                    "Scalar storage crosses execution cores; per-core control normalization "
                    "is required before extraction")
            rewritten.append(replace(task, core=core, accesses=accesses, guard=guard))
        self.tasks = rewritten
        self.bindings = [
            replace(binding, storage=remap.get(binding.storage, binding.storage))
            for binding in self.bindings
        ]
        self.pool.storages = [storage for storage in self.pool.storages if storage not in remap]

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
                storage = self.scalar_storage(expr)
                if storage.id not in seen_vars:
                    seen_vars.add(storage.id)
                    reads.append(Access(storage, "R"))

        self.tir.stmt_functor.post_order_visit(expression, visit)
        return reads

    def buffer_access(self, buffer, indices, mode):
        storage = self.memory_storage(buffer.data, buffer.scope())
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
        """Decode an access pointer; callers must prove operation contiguity.

        CopyNode::Lower uses product(region extents), not a physical span,
        for copy access pointers. operation() discards those tentative ranges.
        """
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
        return Access(self.memory_storage(var), mode, start, count, alias=var)

    def bind(self, node, expression, var=None, kind="let"):
        if var is None:
            raise ExtractionError("A predicate needs an actual scalar binding before extraction; "
                                  "normalize_conditions handles runtime IfThenElse conditions")
        storage = self.scalar_storage(var)
        task_id = self.add_task(node, "PIPE_S", self.core or "UNRESOLVED",
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
        core = self.operation_core(normalize_operation(name), config.pipe)
        prior_core = self.core
        self.core = core
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
        # Copies may stride across rows or convert fractal layouts. Their
        # pointer extent is a logical element count (src/op/ascend.cc), so no
        # contiguous physical span follows from it. This includes PIPE_V UB
        # copies, MTE im2col, and FIX transfers. PIPE_M matrix instructions can
        # similarly read padded/fractal tiles or hardware-rounded K tails.
        # Until helper stride/layout analysis proves actual intervals, retain
        # whole-storage conflicts for transfers and matrix accesses.
        normalized_name = normalize_operation(name)
        if "copy" in normalized_name.lower() or config.pipe in ("PIPE_MTE1", "PIPE_MTE2",
                                                                "PIPE_MTE3", "PIPE_FIX", "PIPE_M"):
            accesses = [
                replace(access, offset=None, extent=None, intervals=None) for access in accesses
            ]
        self.add_task(node, config.pipe, core, accesses)
        self.core = prior_core

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
            if (not isinstance(node.condition, T.Var) or
                    not any(var.same_as(node.condition) for var in self.snapshot_vars)):
                raise ExtractionError("Runtime IfThenElse needs an actual snapshot LetStmt; "
                                      "call normalize_conditions before extraction")
            snapshot = self.scalar_storage(node.condition)
            previous = self.guard
            self.guard += ((snapshot, True),)
            self.walk(node.then_case)
            if node.else_case is not None:
                self.guard = previous + ((snapshot, False),)
                self.walk(node.else_case)
            self.guard = previous
        elif isinstance(node, T.LetStmt):
            kind = "guard" if any(var.same_as(node.var) for var in self.snapshot_vars) else "let"
            self.bind(node, node.value, node.var, kind)
            self.walk(node.body)
        elif isinstance(node, T.AttrStmt):
            stage, core = self.stage, self.core
            snapshots = self.snapshot_vars
            if node.attr_key == CONDITION_SNAPSHOT:
                if (not isinstance(node.node, T.Var) or str(node.node.dtype) != "bool" or
                        not isinstance(node.body, T.LetStmt) or
                        not node.body.var.same_as(node.node) or
                        not isinstance(node.body.body, T.IfThenElse) or
                        not node.body.body.condition.same_as(node.node)):
                    raise ExtractionError("Malformed tl.ascend_condition_snapshot wrapper")
                self.snapshot_vars += (node.node,)
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
            self.snapshot_vars = snapshots
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
            self.memory_storage(node.buffer_var)
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
            self.add_task(node, "PIPE_S", self.core or "UNRESOLVED",
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
