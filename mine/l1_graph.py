"""L1 · Graph semantics: y = relu(x @ w + b) as a Luminal-style primitive graph — your implementation (run ./check.sh l1).

Luminal records every tensor operation into a static DAG instead of running it (lazy, ahead of
time). High-level ops are *spelled* with a small primitive set (README: "Everything in Luminal
boils down to 15 primitive ops"; src/logical/ir.rs `LogicalOp` has the current full list):

  matmul(a, b)  = broadcast both to [M, N, K]  ->  Mul  ->  ReduceSum(K)     (src/frontend/matmul.rs)
  relu(x)       = maximum(x, 0) = Select(LessThan(x, 0), 0, x)                (src/frontend/unary.rs, binary.rs)
  exp(x)        = Exp2(x * log2(e))                                           (src/frontend/unary.rs)
  a + b[N]      = Add(a, Expand(b))  — broadcasting is a *view*, not a copy

This file is a ~150-line Python model of that design: a Graph of nodes with shapes and dtypes,
a GraphTensor API that records, a node-semantics table, and a reference runtime (numpy) that
executes the DAG in topological order — the analogue of Luminal's ReferenceRuntime.
"""
from dataclasses import dataclass, field

import numpy as np

PRIMITIVES = {"Input", "Constant", "Add", "Mul", "Recip", "Exp2", "Log2", "Sqrt", "Sin", "LessThan",
              "Select", "ReduceSum", "ReduceMax", "Expand", "Permute", "Cast"}
MOVEMENT = {"Expand", "Permute"}          # views: change indexing, move no data


@dataclass
class Node:
    id: int
    op: str
    inputs: list
    shape: tuple
    dtype: str = "f32"
    attrs: dict = field(default_factory=dict)


class Graph:
    def __init__(self):
        self.nodes = []
        self.outputs = []

    def add(self, op, inputs, shape, dtype="f32", **attrs):
        assert op in PRIMITIVES, op
        n = Node(len(self.nodes), op, [t.id for t in inputs], tuple(shape), dtype, attrs)
        self.nodes.append(n)
        return Tensor(self, n.id)

    def tensor(self, shape, name, dtype="f32"):
        return self.add("Input", [], shape, dtype, name=name)

    def constant(self, value, dtype="f32"):
        return self.add("Constant", [], (), dtype, value=float(value))

    # ---- inspection
    def topo_order(self):
        """Node ids such that every input comes before its user (ids already are; we verify)."""
        raise NotImplementedError("Graph.topo_order")

    def op_counts(self):
        raise NotImplementedError("Graph.op_counts")

    def semantics_table(self):
        """[(id, op, input ids, shape, dtype, note)] — the L1 deliverable."""
        raise NotImplementedError("Graph.semantics_table")

    def materialized_bytes(self, elem_bytes=4):
        """Bytes a naive executor would allocate: every non-view, non-input node gets a buffer."""
        raise NotImplementedError("Graph.materialized_bytes")


@dataclass(frozen=True)
class Tensor:
    g: Graph
    id: int

    @property
    def shape(self):
        return self.g.nodes[self.id].shape

    @property
    def dtype(self):
        return self.g.nodes[self.id].dtype

    def output(self):
        self.g.outputs.append(self.id)
        return self

    # ---- movement
    def expand(self, axis, size):
        s = list(self.shape)
        s.insert(axis, size)
        return self.g.add("Expand", [self], s, self.dtype, axis=axis, size=size)

    def expand_to(self, shape):
        """Right-aligned numpy-style broadcast by inserting leading / size-1-free axes."""
        t = self
        for axis in range(len(shape) - len(self.shape)):
            t = t.expand(axis, shape[axis])
        assert t.shape == tuple(shape), (t.shape, shape)
        return t

    def permute(self, axes):
        return self.g.add("Permute", [self], [self.shape[a] for a in axes], self.dtype, axes=tuple(axes))

    # ---- binary (shapes must already agree, like Luminal: broadcasting is explicit)
    def _bin(self, op, other, dtype=None):
        if not isinstance(other, Tensor):
            other = self.g.constant(other).expand_to(self.shape) if self.shape else self.g.constant(other)
        assert self.shape == other.shape, f"{op}: {self.shape} vs {other.shape}"
        return self.g.add(op, [self, other], self.shape, dtype or self.dtype)

    def __add__(self, o):
        return self._bin("Add", o)

    def __mul__(self, o):
        return self._bin("Mul", o)

    def lt(self, o):
        return self._bin("LessThan", o, "bool")

    def select(self, a, b):
        """self is the boolean condition."""
        assert self.shape == a.shape == b.shape
        return self.g.add("Select", [self, a, b], a.shape, a.dtype)

    def sum(self, axis):
        s = list(self.shape)
        s.pop(axis)
        return self.g.add("ReduceSum", [self], s, self.dtype, axis=axis)

    # ---- high-level ops, spelled with primitives
    def matmul(self, w):
        """[M, K] @ [K, N]: both operands become [M, N, K] views, then Mul and ReduceSum(K)."""
        raise NotImplementedError("Tensor.matmul")

    def relu(self):
        raise NotImplementedError("Tensor.relu")

    def exp(self):
        raise NotImplementedError("Tensor.exp")

    def exp2(self):
        return self.g.add("Exp2", [self], self.shape, self.dtype)


def build_mlp(M, K, N):
    """y = relu(x @ w + b). Returns (graph, x, w, b, y)."""
    raise NotImplementedError("build_mlp")


# ---------------------------------------------------------------- reference runtime
def run(g, feeds):
    """Execute in topological order with numpy. feeds: {input name: array}. Returns {node id: array}."""
    raise NotImplementedError("run")
