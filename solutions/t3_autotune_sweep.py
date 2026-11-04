"""T3 · GEMM tile / resource sweep, the way triton.autotune does it — reference solution.

python/triton/runtime/autotuner.py runs, per new key:
    configs -> early_config_prune(configs, nargs)       # drop configs that cannot work
            -> perf_model + top_k                        # keep only the k most promising
            -> do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))  # measure the survivors
            -> pick the smallest median
We build the same pipeline in Python. The pruning and the perf model reuse K1's resource model
(solutions.k1_gemm_model, given), and `measure` is injectable: on the RTX 3070 it launches the
real Triton kernel, in tests it is a seeded fake.
"""
import csv
import io
import random
import statistics
from itertools import product
from math import ceil

from solutions.k1_gemm_model import RTX3070, TileConfig, occupancy, roofline, smem_bytes, tiled_bytes, gemm_flops


def config_space(BM=(32, 64, 128, 256), BN=(32, 64, 128, 256), BK=(32, 64), num_warps=(4, 8), num_stages=(2, 3, 4)):
    return [TileConfig(*c) for c in product(BM, BN, BK, num_warps, num_stages)]


def early_prune(configs, gpu, M, N, K):
    """Drop configs that cannot launch (K1 occupancy invalid) or whose tile is pointlessly larger
    than the problem (BM > 2 * M rounds a 100-row problem up to a 256-row tile)."""
    keep = []
    for c in configs:
        if not occupancy(gpu, c)["valid"]:
            continue
        if c.BM > max(32, 2 * M) or c.BN > max(32, 2 * N) or c.BK > max(16, K):
            continue
        keep.append(c)
    return keep


def perf_model(gpu, M, N, K, c, in_bytes=2):
    """Predicted seconds = max(compute with wave quantization, DRAM traffic) + launch overhead.

    Compute: tiles run in waves of sm_count * blocks_per_sm CTAs; a partly empty last wave
    still costs a full wave, so time = waves * per_wave * tile_flops / peak."""
    occ = occupancy(gpu, c, in_bytes)
    tiles = ceil(M / c.BM) * ceil(N / c.BN)
    per_wave = gpu.sm_count * occ["blocks"]
    waves = ceil(tiles / per_wave)
    t_compute = waves * per_wave * (2 * c.BM * c.BN * K) / gpu.peak_flops
    t_memory = tiled_bytes(M, N, K, c, in_bytes) / gpu.peak_bw
    return max(t_compute, t_memory) + gpu.launch_overhead


def top_k(configs, model, k):
    """The k configs with the smallest predicted time (stable on ties)."""
    return sorted(configs, key=model)[:k]


def quantiles(samples, qs=(0.5, 0.2, 0.8)):
    s = sorted(samples)
    out = []
    for q in qs:
        pos = (len(s) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        out.append(s[lo] + (s[hi] - s[lo]) * (pos - lo))
    return out


def sweep(configs, measure, repeats=25):
    """Measure each config `repeats` times. Rows: cfg fields + median/p20/p80/cv."""
    rows = []
    for c in configs:
        xs = [measure(c) for _ in range(repeats)]
        med, p20, p80 = quantiles(xs)
        mean = statistics.fmean(xs)
        rows.append({"BM": c.BM, "BN": c.BN, "BK": c.BK, "num_warps": c.num_warps, "num_stages": c.num_stages,
                     "median": med, "p20": p20, "p80": p80, "cv": statistics.pstdev(xs) / mean, "smem": smem_bytes(c)})
    return rows


def pick_best(rows):
    """Smallest median; if another row's interval [p20, p80] overlaps the winner's, prefer the
    one with less shared memory (more occupancy headroom, same speed within noise)."""
    best = min(rows, key=lambda r: r["median"])
    ties = [r for r in rows if r["p20"] <= best["p80"] and best["p20"] <= r["p80"]]
    return min(ties, key=lambda r: (r["smem"], r["median"]))


def bound(gpu, M, N, K, c):
    """'memory' | 'compute' | 'launch' from the K1 roofline of this config's real traffic."""
    return roofline(gpu, gemm_flops(M, N, K), tiled_bytes(M, N, K, c))["bound"]


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def autotune(gpu, M, N, K, measure, configs=None, k=8, repeats=25):
    """The whole autotuner: prune -> perf_model top-k -> sweep -> pick. Returns (best_row, rows, n_pruned)."""
    configs = configs or config_space()
    alive = early_prune(configs, gpu, M, N, K)
    cand = top_k(alive, lambda c: perf_model(gpu, M, N, K, c), k)
    rows = sweep(cand, measure, repeats)
    return pick_best(rows), rows, len(configs) - len(alive)


def fake_measure(gpu, M, N, K, seed=0, noise=0.03):
    """A seeded stand-in for a real launch: the perf model times a config-specific efficiency
    factor (what the model does not know) times multiplicative noise."""
    rng = random.Random(seed)

    def m(c):
        eff = 1.0 + 0.15 * ((hash((c.BM, c.BN, c.BK, c.num_warps, c.num_stages)) % 7) / 7)
        return perf_model(gpu, M, N, K, c) * eff * (1 + rng.gauss(0, noise))
    return m
