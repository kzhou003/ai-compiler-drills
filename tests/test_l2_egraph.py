import numpy as np

from drill import load

l2 = load("l2_egraph")
E = l2.EGraph


def test_hashcons_shares_subterms():
    g = E()
    a = g.add(("+", "a", "b"))
    b = g.add(("+", "a", "b"))
    assert a == b and g.num_classes() == 3 and g.num_nodes() == 3


def test_union_and_congruence():
    g = E()
    fa, fb = g.add(("f", "a")), g.add(("f", "b"))
    assert g.find(fa) != g.find(fb)
    g.union(g.add("a"), g.add("b"))
    g.rebuild()
    assert g.find(fa) == g.find(fb)          # a = b  =>  f(a) = f(b)


def test_ematch():
    g = E()
    root = g.add(("+", ("+", "a", "b"), "c"))
    subs = g.ematch(("+", ("+", "?x", "?y"), "?z"), root)
    assert len(subs) == 1 and g.find(subs[0]["?x"]) == g.find(g.add("a"))
    assert g.ematch(("+", "?x", "?x"), root) == []
    assert len(g.ematch(("+", "?x", "?x"), g.add(("+", "a", "a")))) == 1


def test_three_term_saturation_counts():
    g = E()
    g.add(("+", ("+", "a", "b"), "c"))
    rep = g.run(l2.COMM_ASSOC)
    assert rep["stop"] == "saturated"
    # classes: a, b, c, a+b, a+c, b+c, a+b+c ; e-nodes: 3 leaves + 2 per pair + 6 at the top
    assert g.num_classes() == 7 and g.num_nodes() == 15
    assert g.equiv(("+", "a", ("+", "b", "c")), ("+", ("+", "a", "b"), "c"))
    assert g.equiv(("+", ("+", "c", "a"), "b"), ("+", "b", ("+", "c", "a")))
    assert not g.equiv(("+", "a", "b"), ("+", "a", "c"))


def test_blowup_is_exponential():
    sizes = []
    for n in (3, 4, 5, 6):
        g = E()
        t = "x0"
        for i in range(1, n):
            t = ("+", t, f"x{i}")
        g.add(t)
        assert g.run(l2.COMM_ASSOC, iter_limit=30, node_limit=10 ** 6)["stop"] == "saturated"
        assert g.num_classes() == 2 ** n - 1
        assert g.num_nodes() == 3 ** n - 2 ** (n + 1) + 1 + n
        sizes.append(g.num_nodes())
    assert sizes == [15, 54, 185, 608]


def test_budgets_stop_the_search():
    g = E()
    t = "x0"
    for i in range(1, 8):
        t = ("+", t, f"x{i}")
    g.add(t)
    assert g.run(l2.COMM_ASSOC, iter_limit=2)["stop"] == "iter_limit"
    g2 = E()
    g2.add(t)
    assert g2.run(l2.COMM_ASSOC, iter_limit=50, node_limit=500)["stop"] == "node_limit"


def test_extraction_picks_the_cheaper_program():
    g = E()
    root = g.add(("+", ("*", "x", "y"), ("*", "x", "z")))
    g.run(l2.ALGEBRA)
    cost, term = g.extract(root, l2.op_cost)
    assert cost == 5 and term[0] == "*" and g.equiv(term, ("*", "x", ("+", "y", "z")))
    g = E()
    root = g.add(("*", ("exp2", ("log2", "x")), 1))
    g.run(l2.ALGEBRA)
    assert g.extract(root, l2.op_cost) == (0, "x")
    assert g.extract(root, l2.ast_size) == (1, "x")


def test_reassociation_is_not_bit_exact():
    env = {"a": 1e8, "b": -1e8, "c": 1.0}
    left = l2.evaluate(("+", ("+", "a", "b"), "c"), env, np.float32)
    right = l2.evaluate(("+", "a", ("+", "b", "c")), env, np.float32)
    assert left == 1.0 and right == 0.0      # equal in the e-graph, different in fp32
