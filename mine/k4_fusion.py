"""K4 · Fusion: fewer launches and less DRAM traffic vs more register pressure — your implementation (run ./check.sh k4).

A chain of ops is a list of stages. Unfused, every stage is its own kernel: it reads its inputs from
DRAM and writes its output back. Fused, a group of consecutive stages runs in one kernel: only the
group's external inputs are read and only its last output is written; the intermediates live in
registers. That is the Triton view (one kernel, an epilogue) and the Luminal view (a graph rewrite
that merges nodes) of the same trade-off.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    out_elems: int          # elements written by this stage
    ext_in_elems: int       # elements read from outside the chain (weights, bias, residual, ...)
    flops: int
    live_regs: int = 0      # registers per thread this stage keeps live inside a fused kernel


def chain_mlp_epilogue(M, N, K):
    """GEMM followed by bias, GELU, residual add — the canonical fused epilogue."""
    return [
        Stage("matmul", M * N, M * K + K * N, 2 * M * N * K, live_regs=128),
        Stage("bias", M * N, N, M * N, live_regs=8),
        Stage("gelu", M * N, 0, 8 * M * N, live_regs=16),
        Stage("residual", M * N, M * N, M * N, live_regs=8),
    ]


def groups_of(stages, fuse):
    """fuse: list of group sizes summing to len(stages), e.g. [1, 1, 1, 1] (unfused) or [4]."""
    assert sum(fuse) == len(stages) and all(g > 0 for g in fuse)
    out, i = [], 0
    for g in fuse:
        out.append(stages[i:i + g])
        i += g
    return out


def traffic(stages, fuse, elem_bytes=2):
    """DRAM bytes, launches, and peak live registers for a fusion plan."""
    raise NotImplementedError("traffic")


def time_estimate(stages, fuse, peak_flops, peak_bw, launch=5e-6, elem_bytes=2, reg_limit=255, spill_penalty=2.0):
    """Roofline per group + launch overhead. If a fused group's live registers exceed reg_limit,
    its memory time is multiplied by spill_penalty (local-memory spills)."""
    raise NotImplementedError("time_estimate")


def best_plan(stages, peak_flops, peak_bw, **kw):
    """Try every way to cut the chain into consecutive groups (2^(n-1) plans)."""
    raise NotImplementedError("best_plan")
