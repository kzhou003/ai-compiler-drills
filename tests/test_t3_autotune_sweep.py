import pytest

from drill import load
from solutions.k1_gemm_model import RTX3070 as G, TileConfig as C, occupancy

t3 = load("t3_autotune_sweep")
S = 4096


def test_space_and_prune():
    cs = t3.config_space()
    assert len(cs) == 4 * 4 * 2 * 2 * 3
    alive = t3.early_prune(cs, G, S, S, S)
    assert len(alive) == 146 and all(occupancy(G, c)["valid"] for c in alive)
    skinny = t3.early_prune(cs, G, 100, S, S)
    assert max(c.BM for c in skinny) == 128          # 256-row tiles are pruned for M = 100


def test_perf_model_wave_quantization():
    c = C(128, 128, 32, 4, 3)                          # 2 CTAs / SM (smem-limited) -> 92 per wave
    assert occupancy(G, c)["blocks"] == 2
    t = t3.perf_model(G, S, S, S, c)
    assert t == pytest.approx(0.004873388571428571)
    # the same tile on 92 tiles (exactly one wave) vs 93 tiles (a second, almost empty wave)
    one = t3.perf_model(G, 128 * 92, 128, 4096, c)
    two = t3.perf_model(G, 128 * 93, 128, 4096, c)
    assert two > one


def test_top_k_is_sorted_by_model():
    alive = t3.early_prune(t3.config_space(), G, S, S, S)
    m = lambda c: t3.perf_model(G, S, S, S, c)
    k = t3.top_k(alive, m, 5)
    assert len(k) == 5 and [m(c) for c in k] == sorted(m(c) for c in k)
    assert all(m(k[-1]) <= m(c) for c in alive if c not in k)


def test_quantiles_match_do_bench_defaults():
    med, p20, p80 = t3.quantiles(list(range(11)))
    assert (med, p20, p80) == (5, 2, 8)


def test_sweep_pick_and_csv():
    m = t3.fake_measure(G, S, S, S, seed=0)
    cands = [C(128, 256, 32, 8, 2), C(128, 128, 32, 4, 3), C(64, 64, 32, 4, 2)]
    rows = t3.sweep(cands, m, repeats=15)
    assert len(rows) == 3 and all(r["p20"] <= r["median"] <= r["p80"] for r in rows)
    best = t3.pick_best(rows)
    assert best["median"] <= min(r["median"] for r in rows) * 1.1
    text = t3.to_csv(rows)
    assert text.splitlines()[0].startswith("BM,BN,BK,num_warps,num_stages,median")
    assert len(text.strip().splitlines()) == 4


def test_pick_best_prefers_less_smem_within_noise():
    rows = [{"median": 1.00, "p20": 0.95, "p80": 1.05, "smem": 98304},
            {"median": 1.02, "p20": 0.97, "p80": 1.07, "smem": 32768},
            {"median": 1.50, "p20": 1.45, "p80": 1.55, "smem": 8192}]
    assert t3.pick_best(rows)["smem"] == 32768


def test_autotune_end_to_end():
    best, rows, pruned = t3.autotune(G, S, S, S, t3.fake_measure(G, S, S, S, seed=0), k=8)
    assert pruned == 46 and len(rows) == 8
    assert (best["BM"], best["BN"]) in {(128, 256), (256, 128)}


def test_bound():
    assert t3.bound(G, 64, 64, 64, C(32, 32, 32)) == "launch"
    # no L2 in the model: a 4096^3 fp16 GEMM with 128x128 tiles re-reads 2 GB -> "memory"
    assert t3.bound(G, S, S, S, C(128, 128, 32)) == "memory"
