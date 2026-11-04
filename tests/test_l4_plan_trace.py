import pytest

from drill import load

l4 = load("l4_plan_trace")
B = l4.SlabBuffer


def test_toposort_and_reachability():
    dag = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    assert l4.toposort(dag) == ["a", "b", "c", "d"]
    after = l4.reachability(dag)
    assert after["a"] == {"b", "c", "d"} and after["b"] == {"d"} and after["d"] == set()
    with pytest.raises(ValueError):
        l4.toposort({"a": ["b"], "b": ["a"]})


def test_chain_reuses_memory():
    dag = {"a": ["b"], "b": ["c"], "c": []}
    bufs = [B("p", 1000, ["a"]), B("q", 100, ["b"]), B("r", 300, ["c"]), B("s", 10, ["a", "c"])]
    off = l4.plan_slab(dag, bufs, 10_000)
    assert off == {"p": 0, "q": 0, "r": 0, "s": 1024}       # p, q, r never live together
    assert l4.slab_bytes(bufs, off) == 1280
    assert l4.naive_bytes(bufs) == 1024 + 256 + 512 + 256


def test_unordered_branches_cannot_alias():
    dag = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    bufs = [B("p", 256, ["b"]), B("q", 256, ["c"]), B("r", 256, ["d"])]
    off = l4.plan_slab(dag, bufs, 10_000)
    assert off["p"] != off["q"]                              # b and c may run concurrently
    assert off["r"] == 0                                     # d is after both
    assert l4.check_no_overlap(dag, bufs, off)


def test_alignment_and_capacity():
    dag = {"a": []}
    assert l4.aligned(1, 256) == 256 and l4.aligned(0, 256) == 256 and l4.aligned(257, 256) == 512
    with pytest.raises(MemoryError, match="capacity"):
        l4.plan_slab(dag, [B("x", 300, ["a"]), B("y", 300, ["a"])], capacity=768)
    with pytest.raises(ValueError):
        l4.plan_slab(dag, [B("x", 1, ["a"])], 100, alignment=0)


def test_mlp_plans():
    dag, bufs, launches = l4.mlp_plan(fused=True)
    off = l4.plan_slab(dag, bufs, 64 << 20)
    assert len(launches) == 1 and launches[0][1] == "cublasLtMatmul"
    assert l4.check_no_overlap(dag, bufs, off)
    dag2, bufs2, launches2 = l4.mlp_plan(fused=False)
    off2 = l4.plan_slab(dag2, bufs2, 64 << 20)
    assert [l[0] for l in launches2] == ["lt_matmul", "add_bias", "relu"]
    assert off2["t1"] == 0 and off2["y"] == 16384            # x is dead by then: its bytes are reused
    assert l4.slab_bytes(bufs2, off2) < l4.naive_bytes(bufs2)
    assert l4.check_no_overlap(dag2, bufs2, off2)


def test_phases_and_achieved():
    ph = l4.phase_breakdown([("graph_compile", 0.4), ("search", 3.0), ("prepare", 0.1),
                             ("execute", 0.0005), ("execute", 0.0005)])
    assert ph["execute"] == pytest.approx(0.001)
    assert ph["share"]["search"] == pytest.approx(3.0 / 3.501)
    with pytest.raises(ValueError):
        l4.phase_breakdown([("lunch", 1.0)])
    f, frac = l4.achieved(2 * 64 * 64 * 256, 10e-6, 40e12)
    assert f == pytest.approx(2 * 64 * 64 * 256 / 10e-6) and frac < 0.01   # tiny GEMM: launch-bound
