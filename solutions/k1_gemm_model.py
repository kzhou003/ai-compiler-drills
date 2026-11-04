"""K1 · GEMM tile & resource model — reference solution.

Pure Python. Given a GEMM shape and a tile configuration (what Triton calls
BLOCK_M / BLOCK_N / BLOCK_K / num_warps / num_stages), predict *before running anything*:

  * how many FLOPs and how many bytes the problem needs at minimum,
  * how many bytes the tiled kernel really moves (A and B are re-read once per tile column/row),
  * how much shared memory and roughly how many registers one CTA needs,
  * how many CTAs fit on one SM (occupancy) and which resource limits it,
  * how many waves of CTAs the grid needs and how full the last wave is,
  * a roofline estimate: is this config memory-, compute- or launch-bound?

These are exactly the numbers you compare against Nsight Compute in T3, and the numbers a
compiler's cost model (Luminal's search, Triton's autotuner pruning) reasons about.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class GPU:
    name: str
    sm_count: int
    smem_per_sm: int            # bytes of shared memory per SM
    smem_per_block: int         # max bytes one CTA may use
    regs_per_sm: int            # 32-bit registers per SM
    max_threads_per_sm: int
    max_blocks_per_sm: int
    max_regs_per_thread: int
    peak_flops: float           # FLOP/s for the dtype being modelled (e.g. FP16 Tensor Core)
    peak_bw: float              # bytes/s of DRAM bandwidth
    launch_overhead: float = 5e-6   # seconds; below ~this a kernel is launch-bound


# Approximate public specs. RTX 3070 = GA104, SM86 (the roadmap's starting GPU).
RTX3070 = GPU("RTX 3070 (SM86)", sm_count=46, smem_per_sm=100 * 1024, smem_per_block=99 * 1024,
              regs_per_sm=65536, max_threads_per_sm=1536, max_blocks_per_sm=16, max_regs_per_thread=255,
              peak_flops=40e12, peak_bw=448e9)


@dataclass(frozen=True)
class TileConfig:
    BM: int
    BN: int
    BK: int
    num_warps: int = 4
    num_stages: int = 3


def gemm_flops(M, N, K):
    """One multiply + one add per (m, n, k)."""
    return 2 * M * N * K


def gemm_min_bytes(M, N, K, in_bytes=2, out_bytes=2):
    """Read A and B once, write C once: the lower bound on DRAM traffic."""
    return (M * K + K * N) * in_bytes + M * N * out_bytes


def arithmetic_intensity(flops, nbytes):
    return flops / nbytes


def tiled_bytes(M, N, K, cfg: TileConfig, in_bytes=2, out_bytes=2):
    """Bytes a tiled kernel loads from DRAM (ignoring L2 hits).

    Each CTA owns a BM x BN tile of C and walks all of K, so the whole of A is read once per
    column of tiles (ceil(N / BN) times) and the whole of B once per row of tiles."""
    a_reads = M * K * ceil(N / cfg.BN)
    b_reads = K * N * ceil(M / cfg.BM)
    return (a_reads + b_reads) * in_bytes + M * N * out_bytes


def smem_bytes(cfg: TileConfig, in_bytes=2):
    """num_stages copies of one A tile (BM x BK) and one B tile (BK x BN)."""
    return cfg.num_stages * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * in_bytes


def threads(cfg: TileConfig):
    return cfg.num_warps * 32


def regs_per_thread(cfg: TileConfig, overhead=40):
    """Rough estimate: the fp32 accumulator tile spread over the CTA's threads, plus a fixed
    overhead for addresses, loop counters and operand fragments. Real numbers come from
    `ptxas -v` / Nsight; this is the first-order term that grows with the tile."""
    return cfg.BM * cfg.BN // threads(cfg) + overhead


def occupancy(gpu: GPU, cfg: TileConfig, in_bytes=2):
    """CTAs that can be resident on one SM, and which resource is the limit.

    Returns {"blocks": int, "limiter": str, "valid": bool, "reason": str}."""
    t = threads(cfg)
    regs = regs_per_thread(cfg)
    smem = smem_bytes(cfg, in_bytes)
    if smem > gpu.smem_per_block:
        return {"blocks": 0, "limiter": "smem", "valid": False,
                "reason": f"needs {smem} B shared memory, a CTA may use {gpu.smem_per_block} B"}
    if regs > gpu.max_regs_per_thread:
        return {"blocks": 0, "limiter": "regs", "valid": False,
                "reason": f"~{regs} registers/thread > {gpu.max_regs_per_thread}: will spill"}
    limits = {
        "threads": gpu.max_threads_per_sm // t,
        "regs": gpu.regs_per_sm // (regs * t),
        "smem": gpu.smem_per_sm // smem if smem else gpu.max_blocks_per_sm,
        "blocks": gpu.max_blocks_per_sm,
    }
    limiter = min(limits, key=lambda k: (limits[k], k))
    blocks = limits[limiter]
    return {"blocks": blocks, "limiter": limiter, "valid": blocks > 0,
            "reason": "" if blocks > 0 else f"{limiter} allows 0 CTAs per SM"}


def waves(gpu: GPU, M, N, cfg: TileConfig, blocks_per_sm):
    """How many rounds of CTAs the grid needs, and how full the last round is (0..1]."""
    tiles = ceil(M / cfg.BM) * ceil(N / cfg.BN)
    per_wave = gpu.sm_count * blocks_per_sm
    n = ceil(tiles / per_wave)
    last = tiles - (n - 1) * per_wave
    return {"tiles": tiles, "per_wave": per_wave, "waves": n, "tail_fill": last / per_wave}


def roofline(gpu: GPU, flops, nbytes):
    """Lower-bound time and the bound that sets it."""
    t_compute = flops / gpu.peak_flops
    t_memory = nbytes / gpu.peak_bw
    t = max(t_compute, t_memory)
    if t < gpu.launch_overhead:
        bound = "launch"
    else:
        bound = "compute" if t_compute >= t_memory else "memory"
    return {"time": max(t, gpu.launch_overhead), "t_compute": t_compute, "t_memory": t_memory, "bound": bound}


def analyze(gpu: GPU, M, N, K, cfg: TileConfig, in_bytes=2, out_bytes=2):
    """Everything above for one (shape, config) pair."""
    occ = occupancy(gpu, cfg, in_bytes)
    flops = gemm_flops(M, N, K)
    moved = tiled_bytes(M, N, K, cfg, in_bytes, out_bytes)
    out = {
        "flops": flops,
        "min_bytes": gemm_min_bytes(M, N, K, in_bytes, out_bytes),
        "tiled_bytes": moved,
        "intensity": arithmetic_intensity(flops, moved),
        "smem": smem_bytes(cfg, in_bytes),
        "regs": regs_per_thread(cfg),
        "occupancy": occ,
        "roofline": roofline(gpu, flops, moved),
    }
    if occ["valid"]:
        out["waves"] = waves(gpu, M, N, cfg, occ["blocks"])
    return out
