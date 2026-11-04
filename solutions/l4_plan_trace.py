"""L4 · Backend trace: from the selected plan to GPU launches and one memory slab — reference solution.

Once search picked a genome, the backend turns the plan into:

  1. a DAG of kernels / library calls (the LLIR the README's LUMINAL_LOG_LLIR prints),
  2. buffer lifetimes: the DAG nodes at which each buffer is read or written,
  3. one device slab with an offset per buffer — src/buffer/arena.rs `plan_slab`:
       * transitive closure of the DAG ("u happens-before v"),
       * two buffers may share bytes only if *every* use of one precedes *every* use of the other,
       * first fit in request order, sizes rounded up to the alignment (256 B on CUDA),
  4. an execution timeline, split into phases (graph compile / search / prepare / execute —
     what LUMINAL_CUDA_PROFILE_RECAPTURE breaks TTFT/TPOT into).
"""
from dataclasses import dataclass


@dataclass
class SlabBuffer:
    id: str
    bytes: int
    uses: list          # DAG node names


def toposort(dag):
    """dag: {node: [successors]}. Kahn's algorithm with sorted tie-break (deterministic)."""
    nodes = set(dag) | {s for v in dag.values() for s in v}
    indeg = {n: 0 for n in nodes}
    for v in dag.values():
        for s in v:
            indeg[s] += 1
    ready = sorted(n for n in nodes if indeg[n] == 0)
    out = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for s in sorted(dag.get(n, [])):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
                ready.sort()
    if len(out) != len(nodes):
        raise ValueError("slab lifetime graph must be acyclic")
    return out


def reachability(dag):
    """after[u] = set of nodes strictly reachable from u."""
    after = {}
    for n in reversed(toposort(dag)):
        acc = set()
        for s in dag.get(n, []):
            acc.add(s)
            acc |= after[s]
        after[n] = acc
    return after


def aligned(n, a):
    return -(-max(n, 1) // a) * a          # empty buffers still reserve one aligned unit


def plan_slab(dag, buffers, capacity, alignment=256):
    """Port of arena.rs plan_slab (FirstFit). Returns {buffer id: offset}; raises on overflow."""
    if alignment <= 0:
        raise ValueError("slab alignment must be positive")
    after = reachability(dag)
    precedes = lambda a, b: all(v in after[u] for u in a.uses for v in b.uses)
    sizes = [aligned(b.bytes, alignment) for b in buffers]
    offsets = []
    for i, b in enumerate(buffers):
        occupied = sorted((offsets[j], sizes[j]) for j in range(i)
                          if not precedes(b, buffers[j]) and not precedes(buffers[j], b))
        off = 0
        for start, n in occupied:
            if off + sizes[i] <= start:
                break
            off = max(off, start + n)
        if off + sizes[i] > capacity:
            raise MemoryError(f"slab capacity {capacity} exceeded while placing request {i} ({b.bytes} bytes)")
        offsets.append(off)
    return {b.id: o for b, o in zip(buffers, offsets)}


def slab_bytes(buffers, offsets, alignment=256):
    """High-water mark: what finalists.rs compares to the device budget."""
    return max((offsets[b.id] + aligned(b.bytes, alignment) for b in buffers), default=0)


def naive_bytes(buffers, alignment=256):
    """One allocation per buffer, no reuse."""
    return sum(aligned(b.bytes, alignment) for b in buffers)


def check_no_overlap(dag, buffers, offsets, alignment=256):
    """Invariant: buffers whose lifetimes are not ordered never share a byte."""
    after = reachability(dag)
    precedes = lambda a, b: all(v in after[u] for u in a.uses for v in b.uses)
    for i, a in enumerate(buffers):
        for b in buffers[i + 1:]:
            if precedes(a, b) or precedes(b, a):
                continue
            a0, a1 = offsets[a.id], offsets[a.id] + aligned(a.bytes, alignment)
            b0, b1 = offsets[b.id], offsets[b.id] + aligned(b.bytes, alignment)
            if a0 < b1 and b0 < a1:
                return False
    return True


# ---------------------------------------------------------------- the MLP plan
def mlp_plan(M=64, K=256, N=64, fused=True, elem=4):
    """The selected plan for relu(x @ w + b) as (dag, buffers, launches).
    fused=True : one cuBLASLt call with the RELU_BIAS epilogue.
    fused=False: cuBLASLt matmul, then an add-bias kernel, then a relu kernel."""
    if fused:
        dag = {"upload": ["lt_matmul_relu_bias"], "lt_matmul_relu_bias": ["readback"], "readback": []}
        bufs = [SlabBuffer("x", M * K * elem, ["upload", "lt_matmul_relu_bias"]),
                SlabBuffer("w", K * N * elem, ["upload", "lt_matmul_relu_bias"]),
                SlabBuffer("b", N * elem, ["upload", "lt_matmul_relu_bias"]),
                SlabBuffer("workspace", 1 << 20, ["lt_matmul_relu_bias"]),
                SlabBuffer("y", M * N * elem, ["lt_matmul_relu_bias", "readback"])]
        launches = [("lt_matmul_relu_bias", "cublasLtMatmul", 2 * M * N * K)]
    else:
        dag = {"upload": ["lt_matmul"], "lt_matmul": ["add_bias"], "add_bias": ["relu"],
               "relu": ["readback"], "readback": []}
        bufs = [SlabBuffer("x", M * K * elem, ["upload", "lt_matmul"]),
                SlabBuffer("w", K * N * elem, ["upload", "lt_matmul"]),
                SlabBuffer("b", N * elem, ["upload", "add_bias"]),
                SlabBuffer("workspace", 1 << 20, ["lt_matmul"]),
                SlabBuffer("t0", M * N * elem, ["lt_matmul", "add_bias"]),
                SlabBuffer("t1", M * N * elem, ["add_bias", "relu"]),
                SlabBuffer("y", M * N * elem, ["relu", "readback"])]
        launches = [("lt_matmul", "cublasLtMatmul", 2 * M * N * K),
                    ("add_bias", "kernel", M * N), ("relu", "kernel", M * N)]
    return dag, bufs, launches


# ---------------------------------------------------------------- phases
PHASES = ("graph_compile", "search", "prepare", "execute")


def phase_breakdown(events):
    """events: [(phase, seconds)] possibly repeated. Returns {phase: total, 'share': {phase: frac}}."""
    tot = {p: 0.0 for p in PHASES}
    for p, s in events:
        if p not in tot:
            raise ValueError(f"unknown phase {p}")
        tot[p] += s
    all_s = sum(tot.values())
    tot["share"] = {p: (tot[p] / all_s if all_s else 0.0) for p in PHASES}
    return tot


def achieved(flops, seconds, peak_flops):
    """Achieved FLOP/s and fraction of peak — compare theory with the profiled kernel time."""
    f = flops / seconds
    return f, f / peak_flops
