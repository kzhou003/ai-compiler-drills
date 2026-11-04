"""L4 · Backend trace: from the selected plan to GPU launches and one memory slab — your implementation (run ./check.sh l4).

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
    raise NotImplementedError("toposort")


def reachability(dag):
    """after[u] = set of nodes strictly reachable from u."""
    raise NotImplementedError("reachability")


def aligned(n, a):
    return -(-max(n, 1) // a) * a          # empty buffers still reserve one aligned unit


def plan_slab(dag, buffers, capacity, alignment=256):
    """Port of arena.rs plan_slab (FirstFit). Returns {buffer id: offset}; raises on overflow."""
    raise NotImplementedError("plan_slab")


def slab_bytes(buffers, offsets, alignment=256):
    """High-water mark: what finalists.rs compares to the device budget."""
    raise NotImplementedError("slab_bytes")


def naive_bytes(buffers, alignment=256):
    """One allocation per buffer, no reuse."""
    raise NotImplementedError("naive_bytes")


def check_no_overlap(dag, buffers, offsets, alignment=256):
    """Invariant: buffers whose lifetimes are not ordered never share a byte."""
    raise NotImplementedError("check_no_overlap")


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
    raise NotImplementedError("phase_breakdown")


def achieved(flops, seconds, peak_flops):
    """Achieved FLOP/s and fraction of peak — compare theory with the profiled kernel time."""
    raise NotImplementedError("achieved")
