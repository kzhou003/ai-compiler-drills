"""K4 · Fusion: fewer launches and less DRAM traffic vs more register pressure — reference solution.

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
    total, regs, prev_out = 0, 0, 0
    grps = groups_of(stages, fuse)
    for grp in grps:
        reads = sum(s.ext_in_elems for s in grp) + prev_out    # + the previous group's output
        total += (reads + grp[-1].out_elems) * elem_bytes
        prev_out = grp[-1].out_elems
        regs = max(regs, sum(s.live_regs for s in grp))
    return {"bytes": total, "launches": len(grps), "peak_regs": regs}


def time_estimate(stages, fuse, peak_flops, peak_bw, launch=5e-6, elem_bytes=2, reg_limit=255, spill_penalty=2.0):
    """Roofline per group + launch overhead. If a fused group's live registers exceed reg_limit,
    its memory time is multiplied by spill_penalty (local-memory spills)."""
    t = 0.0
    prev_out = None
    for grp in groups_of(stages, fuse):
        reads = sum(s.ext_in_elems for s in grp) + (prev_out or 0)
        nbytes = (reads + grp[-1].out_elems) * elem_bytes
        regs = sum(s.live_regs for s in grp)
        t_mem = nbytes / peak_bw * (spill_penalty if regs > reg_limit else 1.0)
        t_cmp = sum(s.flops for s in grp) / peak_flops
        t += max(t_mem, t_cmp) + launch
        prev_out = grp[-1].out_elems
    return t


def best_plan(stages, peak_flops, peak_bw, **kw):
    """Try every way to cut the chain into consecutive groups (2^(n-1) plans)."""
    n = len(stages)
    plans = []
    for mask in range(1 << (n - 1)):
        sizes, cur = [], 1
        for i in range(n - 1):
            if mask >> i & 1:
                sizes.append(cur)
                cur = 1
            else:
                cur += 1
        sizes.append(cur)
        plans.append((time_estimate(stages, sizes, peak_flops, peak_bw, **kw), sizes))
    return min(plans)
