"""L3 · Search cost vs measured latency: a Luminal-style genome search — reference solution.

After saturation the e-graph holds many programs. Luminal's CUDA backend does not trust a cost
formula to choose between them; it *measures*. The pieces (crates/luminal_cuda_lite/src/search.rs,
finalists.rs, src/extraction.rs):

  Genome      one producer choice per e-class (which e-node implements this class)
  GA          a genetic search over genomes; each candidate is compiled + run, metric = measured ns
  rank_insert keep only the best `keep` genomes, sorted by measured ns
  finalists   walk the ranking; a candidate that fails validation (e.g. its arena exceeds the
              device memory budget) is rejected and the next rank is tried

Here the "e-graph" is a dict {class: [option, ...]} and `measure(genome)` is injectable.
`mlp_space()` is a small but honest model of y = relu(x @ w + b) on a GPU.
"""
import random
from itertools import product
from math import prod


def space_size(space):
    return prod(len(v) for v in space.values())


def all_genomes(space):
    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in product(*(space[k] for k in keys))]


def fingerprint(genome):
    return tuple(sorted(genome.items()))


def random_genome(space, rng):
    return {k: rng.choice(v) for k, v in sorted(space.items())}


def mutate(genome, space, rng, p=0.3):
    g = dict(genome)
    for k in sorted(space):
        if rng.random() < p:
            g[k] = rng.choice(space[k])
    return g


def crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in sorted(a)}


def rank_insert(ranked, nanos, genome, keep):
    """Port of search.rs rank_insert: ranked is [(nanos, genome)] ascending, at most `keep` long."""
    keep = max(keep, 1)
    if len(ranked) >= keep and nanos >= ranked[-1][0]:
        return
    pos = next((i for i, (m, _) in enumerate(ranked) if nanos < m), len(ranked))
    ranked.insert(pos, (nanos, genome))
    del ranked[keep:]


def genetic_search(space, measure, budget, population=6, keep=4, seed=0, mutation=0.3):
    """Measure at most `budget` distinct genomes. Returns ranked finalists, the number of distinct
    measurements, and the best-so-far curve (one entry per measurement)."""
    rng = random.Random(seed)
    seen, ranked, curve = {}, [], []

    def evaluate(g):
        fp = fingerprint(g)
        if fp in seen or len(seen) >= budget:
            return seen.get(fp)
        ns = measure(g)
        seen[fp] = ns
        rank_insert(ranked, ns, g, keep)
        curve.append(ranked[0][0])
        return ns

    pop = [random_genome(space, rng) for _ in range(population)]
    for g in pop:
        evaluate(g)
    stale = 0
    while len(seen) < min(budget, space_size(space)) and stale < 50:
        parents = [g for _, g in ranked] or pop
        a, b = rng.choice(parents), rng.choice(parents)
        child = mutate(crossover(a, b, rng), space, rng, mutation)
        before = len(seen)
        evaluate(child)
        stale = stale + 1 if len(seen) == before else 0
    return {"ranked": ranked, "measured": len(seen), "curve": curve}


def proxy_search(space, proxy, measure, k):
    """The cheap alternative: rank *every* genome by a cost proxy, measure only the top k."""
    cand = sorted(all_genomes(space), key=proxy)[:k]
    ranked = []
    for g in cand:
        rank_insert(ranked, measure(g), g, k)
    return ranked


def spearman(xs, ys):
    """Rank correlation (no tie correction needed for the tests; ties get average rank)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy)


def select_finalist(ranked, validate):
    """finalists.rs: walk ranks 1, 2, ...; return (rank, nanos, genome, rejected[]) of the first
    candidate that validates. validate(genome) -> None if OK, else a reason string."""
    rejected = []
    for rank, (ns, g) in enumerate(ranked, 1):
        why = validate(g)
        if why is None:
            return rank, ns, g, rejected
        rejected.append((rank, why))
    raise RuntimeError(f"no finalist validated: {rejected}")


def break_even_runs(search_seconds_a, latency_a, search_seconds_b, latency_b):
    """How many executions until spending more search time (a) pays off over b. inf if never."""
    if latency_a >= latency_b:
        return float("inf")
    return (search_seconds_a - search_seconds_b) / (latency_b - latency_a)


# ---------------------------------------------------------------- the workload
def mlp_space():
    """Decision points for y = relu(x @ w + b) after saturation (simplified)."""
    return {
        "matmul": ["primitive_loop", "tiled_kernel", "cublaslt"],
        "epilogue": ["separate", "fused"],
        "w_layout": ["row_major", "col_major"],
        "accumulate": ["fp32", "fp16"],
    }


MATMUL_NS = {"primitive_loop": 2400, "tiled_kernel": 130, "cublaslt": 95}
FUSE_SAVING = {"primitive_loop": 0, "tiled_kernel": 25, "cublaslt": 22}     # cuBLASLt: EPILOGUE_RELU_BIAS


def mlp_measure(noise=0.0, seed=0):
    """'Measured' ns: what a GPU would show, including effects a FLOP proxy cannot see."""
    rng = random.Random(seed)

    def m(g):
        ns = MATMUL_NS[g["matmul"]]
        ns += 30 if g["epilogue"] == "separate" else 30 - FUSE_SAVING[g["matmul"]]
        if g["matmul"] == "tiled_kernel" and g["w_layout"] == "row_major":
            ns += 20                      # B tile loads along K are strided for this kernel
        if g["accumulate"] == "fp16":
            ns -= 8
        return ns * (1 + rng.gauss(0, noise)) if noise else ns
    return m


def mlp_proxy(g):
    """A FLOP/byte proxy: knows the primitive loop is bad and fusion saves a pass, but cannot
    tell a hand-written tile from cuBLASLt, nor see layout effects."""
    base = {"primitive_loop": 2000, "tiled_kernel": 100, "cublaslt": 100}[g["matmul"]]
    return base + (30 if g["epilogue"] == "separate" else 10) + (-5 if g["accumulate"] == "fp16" else 0)


def mlp_validate(budget_bytes=8 << 20, M=64, N=64, K=256, max_err_ulps=None):
    """Reject plans whose intermediates exceed the arena budget, and fp16 accumulation (precision)."""
    def v(g):
        if g["matmul"] == "primitive_loop" and M * N * K * 4 > budget_bytes:
            return f"arena {M * N * K * 4} B > budget {budget_bytes} B"
        if g["accumulate"] == "fp16":
            return "fp16 accumulation fails the K=%d tolerance" % K
        return None
    return v
