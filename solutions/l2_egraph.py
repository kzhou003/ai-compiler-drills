"""L2 · E-graphs and equality saturation from scratch — reference solution.

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
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if len(self.classes[a]) < len(self.classes[b]):
            a, b = b, a
        self.parent[b] = a
        self.classes[a] |= self.classes.pop(b)
        self.pending.append(a)
        return True

    def rebuild(self):
        """Re-canonicalize every e-node; merge classes whose e-nodes became identical (congruence)."""
        changed = True
        while changed:
            changed = False
            new_hash = {}
            for cid in list(self.classes):
                cid = self.find(cid)
                if cid not in self.classes:
                    continue
                self.classes[cid] = {self.canon(n) for n in self.classes[cid]}
            for cid, nodes in list(self.classes.items()):
                for n in nodes:
                    other = new_hash.get(n)
                    if other is not None and self.find(other) != self.find(cid):
                        self.union(other, cid)
                        changed = True
                    new_hash[n] = self.find(cid)
            self.hashcons = {self.canon(n): self.find(c) for n, c in new_hash.items()}
        self.pending.clear()

    # ---------------------------------------------------------- queries
    def equiv(self, t1, t2):
        return self.find(self.add(t1)) == self.find(self.add(t2))

    def num_classes(self):
        return len(self.classes)

    def num_nodes(self):
        return sum(len(v) for v in self.classes.values())

    # ---------------------------------------------------------- e-matching
    def ematch(self, pattern, cid, subst=None):
        """All substitutions {?var: class id} under which `pattern` matches something in class cid."""
        subst = dict(subst or {})
        cid = self.find(cid)
        if isinstance(pattern, str) and pattern.startswith("?"):
            if pattern in subst:
                return [subst] if self.find(subst[pattern]) == cid else []
            subst[pattern] = cid
            return [subst]
        if not isinstance(pattern, tuple):
            return [subst] if (pattern,) in self.classes[cid] else []
        out = []
        op, *pkids = pattern
        for n in self.classes[cid]:
            if n[0] != op or len(n) - 1 != len(pkids):
                continue
            partial = [subst]
            for p, k in zip(pkids, n[1:]):
                partial = [s2 for s in partial for s2 in self.ematch(p, k, s)]
            out.extend(partial)
        return out

    def instantiate(self, pattern, subst):
        if isinstance(pattern, str) and pattern.startswith("?"):
            return subst[pattern]
        if not isinstance(pattern, tuple):
            return self.add_enode((pattern,))
        return self.add_enode((pattern[0], *(self.instantiate(p, subst) for p in pattern[1:])))

    # ---------------------------------------------------------- saturation
    def run(self, rules, iter_limit=20, node_limit=10_000):
        """Equality saturation. rules: [(name, lhs, rhs)]. Returns a report dict."""
        history = []
        for it in range(iter_limit):
            before = (self.num_classes(), self.num_nodes())
            matches = []
            for name, lhs, rhs in rules:
                for cid in list(self.classes):
                    for s in self.ematch(lhs, cid):
                        matches.append((name, cid, rhs, s))
            unions = 0
            for name, cid, rhs, s in matches:
                unions += self.union(cid, self.instantiate(rhs, s))
            self.rebuild()
            history.append({"iter": it, "matches": len(matches), "unions": unions,
                            "classes": self.num_classes(), "nodes": self.num_nodes()})
            if unions == 0 and (self.num_classes(), self.num_nodes()) == before:
                return {"stop": "saturated", "history": history}
            if self.num_nodes() > node_limit:
                return {"stop": "node_limit", "history": history}
        return {"stop": "iter_limit", "history": history}

    # ---------------------------------------------------------- extraction
    def extract(self, cid, cost):
        """Cheapest term in class cid. cost(op, child_costs) -> number. Bottom-up fixpoint
        (classes may be cyclic, e.g. x*1 = x)."""
        best = {}
        changed = True
        while changed:
            changed = False
            for c, nodes in self.classes.items():
                for n in nodes:
                    if all(self.find(k) in best for k in n[1:]):
                        cc = cost(n[0], [best[self.find(k)][0] for k in n[1:]])
                        if c not in best or cc < best[c][0]:
                            best[c] = (cc, n)
                            changed = True

        def build(c):
            _, n = best[self.find(c)]
            return n[0] if len(n) == 1 else (n[0], *(build(k) for k in n[1:]))
        c = self.find(cid)
        return best[c][0], build(c)


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
