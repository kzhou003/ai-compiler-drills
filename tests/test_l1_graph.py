import numpy as np
import pytest

from drill import load

l1 = load("l1_graph")


def test_recording_does_not_compute():
    g = l1.Graph()
    x = g.tensor((2, 3), "x")
    y = x + x
    assert isinstance(y, l1.Tensor) and len(g.nodes) == 2 and g.nodes[1].op == "Add"


def test_matmul_is_broadcast_mul_reduce():
    g = l1.Graph()
    a, b = g.tensor((5, 3), "a"), g.tensor((3, 7), "b")
    c = a.matmul(b).output()
    assert c.shape == (5, 7)
    ops = [g.nodes[i].op for i in g.topo_order()]
    assert ops.count("Mul") == 1 and ops.count("ReduceSum") == 1 and ops.count("Permute") == 1
    mul = next(n for n in g.nodes if n.op == "Mul")
    assert mul.shape == (5, 7, 3)                     # the [M, N, K] product
    with pytest.raises(AssertionError):
        a.matmul(g.tensor((4, 7), "bad"))


def test_mlp_graph_shape_and_counts():
    g, x, w, b, y = l1.build_mlp(2, 3, 4)
    assert y.shape == (2, 4) and y.dtype == "f32"
    assert g.op_counts() == {"Input": 3, "Expand": 5, "Permute": 1, "Mul": 1, "ReduceSum": 1, "Add": 1,
                             "Constant": 1, "LessThan": 1, "Select": 1}
    table = g.semantics_table()
    lt = next(r for r in table if r[1] == "LessThan")
    assert lt[4] == "bool"
    assert any("view" in r[5] for r in table if r[1] == "Expand")
    assert all(set(g.nodes[r[0]].op for r in table) <= l1.PRIMITIVES for _ in [0])


def test_reference_runtime_matches_numpy():
    rng = np.random.default_rng(0)
    for M, K, N in [(2, 3, 4), (1, 1, 1), (7, 13, 5)]:
        g, x, w, b, y = l1.build_mlp(M, K, N)
        X, W, B = rng.standard_normal((M, K)), rng.standard_normal((K, N)), rng.standard_normal(N)
        v = l1.run(g, {"x": X, "w": W, "b": B})
        np.testing.assert_allclose(v[y.id], np.maximum(X @ W + B, 0), rtol=1e-5, atol=1e-5)


def test_relu_is_select_and_exp_is_exp2():
    g = l1.Graph()
    x = g.tensor((3,), "x")
    r = x.relu().output()
    assert g.nodes[r.id].op == "Select"
    e = x.exp().output()
    assert g.nodes[e.id].op == "Exp2"
    v = l1.run(g, {"x": np.array([-1.0, 0.0, 2.0])})
    np.testing.assert_allclose(v[r.id], [0, 0, 2])
    np.testing.assert_allclose(v[e.id], np.exp([-1.0, 0.0, 2.0]), rtol=1e-6)


def test_naive_execution_materializes_the_product():
    g, *_ = l1.build_mlp(64, 256, 64)
    # Mul [64, 64, 256] alone is 4 MiB of fp32 — that's what fusion must avoid
    assert g.materialized_bytes() >= 64 * 64 * 256 * 4
    assert g.materialized_bytes() == (64 * 64 * 256 + 4 * 64 * 64) * 4
