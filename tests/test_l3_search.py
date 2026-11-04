import pytest

from drill import load

l3 = load("l3_search")
S = l3.mlp_space()


def test_space():
    assert l3.space_size(S) == 3 * 2 * 2 * 2
    gs = l3.all_genomes(S)
    assert len(gs) == 24 and len({l3.fingerprint(g) for g in gs}) == 24


def test_rank_insert_keeps_best_k_sorted():
    r = []
    for ns, name in [(50, "a"), (30, "b"), (70, "c"), (10, "d"), (60, "e")]:
        l3.rank_insert(r, ns, name, keep=3)
    assert r == [(10, "d"), (30, "b"), (50, "a")]
    l3.rank_insert(r, 50, "tie", keep=3)        # cannot displace an equal worst
    assert r[-1] == (50, "a")


def test_true_optimum():
    m = l3.mlp_measure()
    best = min(l3.all_genomes(S), key=m)
    assert m(best) == 95 and best["matmul"] == "cublaslt" and best["epilogue"] == "fused"


def test_genetic_search_budget_and_curve():
    m = l3.mlp_measure()
    r = l3.genetic_search(S, m, budget=10, seed=0)
    assert r["measured"] == 10 and len(r["curve"]) == 10
    assert all(a >= b for a, b in zip(r["curve"], r["curve"][1:]))     # best-so-far never gets worse
    assert r["ranked"][0][0] == 95 and len(r["ranked"]) == 4
    small = l3.genetic_search(S, m, budget=3, seed=1)
    assert small["measured"] == 3


def test_proxy_is_correlated_but_not_exact():
    m = l3.mlp_measure()
    gs = l3.all_genomes(S)
    rho = l3.spearman([l3.mlp_proxy(g) for g in gs], [m(g) for g in gs])
    assert 0.7 < rho < 0.95
    assert l3.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert l3.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_finalists_walk_past_invalid_candidates():
    m = l3.mlp_measure()
    r = l3.genetic_search(S, m, budget=24, seed=0)
    rank, ns, g, rejected = l3.select_finalist(r["ranked"], l3.mlp_validate())
    assert rank == 3 and ns == 103 and g["accumulate"] == "fp32" and len(rejected) == 2
    assert all("fp16" in why for _, why in rejected)


def test_proxy_top_k_can_all_be_invalid():
    ranked = l3.proxy_search(S, l3.mlp_proxy, l3.mlp_measure(), k=3)
    with pytest.raises(RuntimeError, match="no finalist"):
        l3.select_finalist(ranked, l3.mlp_validate())


def test_primitive_loop_breaks_the_arena_budget():
    v = l3.mlp_validate(budget_bytes=1 << 20)
    g = {"matmul": "primitive_loop", "epilogue": "separate", "w_layout": "row_major", "accumulate": "fp32"}
    assert "arena" in v(g)


def test_break_even():
    # 2 s of search to save 30 ns per call vs 0.1 s of search
    assert l3.break_even_runs(2.0, 87e-9, 0.1, 117e-9) == pytest.approx(1.9 / 30e-9)
    assert l3.break_even_runs(1.0, 120e-9, 0.1, 117e-9) == float("inf")
