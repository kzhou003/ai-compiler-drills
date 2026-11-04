"""T3 · GEMM tile / resource sweep, the way triton.autotune does it — your implementation (run ./check.sh t3).

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
    raise NotImplementedError("early_prune")


def perf_model(gpu, M, N, K, c, in_bytes=2):
    """Predicted seconds = max(compute with wave quantization, DRAM traffic) + launch overhead.

    Compute: tiles run in waves of sm_count * blocks_per_sm CTAs; a partly empty last wave
    still costs a full wave, so time = waves * per_wave * tile_flops / peak."""
    raise NotImplementedError("perf_model")


def top_k(configs, model, k):
    """The k configs with the smallest predicted time (stable on ties)."""
    raise NotImplementedError("top_k")


def quantiles(samples, qs=(0.5, 0.2, 0.8)):
    raise NotImplementedError("quantiles")


def sweep(configs, measure, repeats=25):
    """Measure each config `repeats` times. Rows: cfg fields + median/p20/p80/cv."""
    raise NotImplementedError("sweep")


def pick_best(rows):
    """Smallest median; if another row's interval [p20, p80] overlaps the winner's, prefer the
    one with less shared memory (more occupancy headroom, same speed within noise)."""
    raise NotImplementedError("pick_best")


def bound(gpu, M, N, K, c):
    """'memory' | 'compute' | 'launch' from the K1 roofline of this config's real traffic."""
    raise NotImplementedError("bound")


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def autotune(gpu, M, N, K, measure, configs=None, k=8, repeats=25):
    """The whole autotuner: prune -> perf_model top-k -> sweep -> pick. Returns (best_row, rows, n_pruned)."""
    raise NotImplementedError("autotune")


def fake_measure(gpu, M, N, K, seed=0, noise=0.03):
    """A seeded stand-in for a real launch: the perf model times a config-specific efficiency
    factor (what the model does not know) times multiplicative noise."""
    rng = random.Random(seed)

    def m(c):
        eff = 1.0 + 0.15 * ((hash((c.BM, c.BN, c.BK, c.num_warps, c.num_stages)) % 7) / 7)
        return perf_model(gpu, M, N, K, c) * eff * (1 + rng.gauss(0, noise))
    return m
