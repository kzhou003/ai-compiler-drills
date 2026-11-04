"""L2 · E-graphs and equality saturation from scratch — your implementation (run ./check.sh l2).

Luminal's search space is an e-graph built by egglog (src/egglog_core/egglog_preamble.egg holds
the rules). An e-graph stores *many equivalent programs at once*:

  e-node   an operator applied to e-classes      ("+", c1, c2)
  e-class  a set of e-nodes known to be equal    {a+b, b+a}
  rewrite  lhs -> rhs: when lhs matches in class C, add rhs and union it into C (nothing is deleted)
  rebuild  restore congruence: if a ≡ b then f(a) ≡ f(b)
  saturate apply all rules until nothing new appears (or a node / iteration budget runs out)
  extract  pick one e-node per class to minimize a cost -> one concrete program

Terms are tuples: ("+", x, y), leaves are strings or numbers. Pattern variables start with "?".
"""
from itertools import count


class EGraph:
    def __init__(self):
        self.parent = {}            # union-find over class ids
        self.classes = {}           # canonical id -> set of canonical e-nodes
        self.hashcons = {}          # canonical e-node -> class id
        self._ids = count()
        self.pending = []

    # ---------------------------------------------------------- union-find
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def canon(self, enode):
        op, *kids = enode
        return (op, *(self.find(k) for k in kids))

    # ---------------------------------------------------------- building
    def add_enode(self, enode):
        enode = self.canon(enode)
        if enode in self.hashcons:
            return self.find(self.hashcons[enode])
        cid = next(self._ids)
        self.parent[cid] = cid
        self.classes[cid] = {enode}
        self.hashcons[enode] = cid
        return cid

    def add(self, term):
        if isinstance(term, tuple):
            return self.add_enode((term[0], *(self.add(t) for t in term[1:])))
        return self.add_enode((term,))

    def union(self, a, b):
        raise NotImplementedError("EGraph.union")

    def rebuild(self):
        """Re-canonicalize every e-node; merge classes whose e-nodes became identical (congruence)."""
        raise NotImplementedError("EGraph.rebuild")

    # ---------------------------------------------------------- queries
    def equiv(self, t1, t2):
        raise NotImplementedError("EGraph.equiv")

    def num_classes(self):
        return len(self.classes)

    def num_nodes(self):
        return sum(len(v) for v in self.classes.values())

    # ---------------------------------------------------------- e-matching
    def ematch(self, pattern, cid, subst=None):
        """All substitutions {?var: class id} under which `pattern` matches something in class cid."""
        raise NotImplementedError("EGraph.ematch")

    def instantiate(self, pattern, subst):
        raise NotImplementedError("EGraph.instantiate")

    # ---------------------------------------------------------- saturation
    def run(self, rules, iter_limit=20, node_limit=10_000):
        """Equality saturation. rules: [(name, lhs, rhs)]. Returns a report dict."""
        raise NotImplementedError("EGraph.run")

    # ---------------------------------------------------------- extraction
    def extract(self, cid, cost):
        """Cheapest term in class cid. cost(op, child_costs) -> number. Bottom-up fixpoint
        (classes may be cyclic, e.g. x*1 = x)."""
        raise NotImplementedError("EGraph.extract")


# ---------------------------------------------------------------- rule sets
COMM_ASSOC = [
    ("comm-add", ("+", "?a", "?b"), ("+", "?b", "?a")),
    ("assoc-add", ("+", ("+", "?a", "?b"), "?c"), ("+", "?a", ("+", "?b", "?c"))),
]
ALGEBRA = COMM_ASSOC + [
    ("mul-one", ("*", "?a", 1), "?a"),
    ("add-zero", ("+", "?a", 0), "?a"),
    ("factor", ("+", ("*", "?a", "?b"), ("*", "?a", "?c")), ("*", "?a", ("+", "?b", "?c"))),
    ("exp2-log2", ("exp2", ("log2", "?a")), "?a"),
]


def ast_size(op, kids):
    return 1 + sum(kids)


def op_cost(op, kids):
    """Rough kernel cost: multiplies cost more than adds; leaves are free."""
    return {"*": 4, "+": 1, "exp2": 8, "log2": 8}.get(op, 0) + sum(kids)


def evaluate(term, env, dtype=float):
    """Evaluate a term with Python floats or numpy scalars — to show which rewrites change bits."""
    import math
    if isinstance(term, tuple):
        op, *k = term
        v = [evaluate(t, env, dtype) for t in k]
        return {"+": lambda a, b: dtype(a + b), "*": lambda a, b: dtype(a * b),
                "exp2": lambda a: dtype(2.0 ** a), "log2": lambda a: dtype(math.log2(a))}[op](*v)
    return dtype(env[term]) if isinstance(term, str) else dtype(term)
