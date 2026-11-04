"""T4 · Correctness bug minimization: fuzz -> fail -> shrink -> regression test — your implementation (run ./check.sh t4).

The loop every compiler contributor runs before opening an issue:

  1. random differential test against a reference (fixed seed, edge-heavy shapes, several dtypes,
     non-contiguous strides);
  2. keep the first failing input;
  3. shrink it — smallest shape, simplest dtype, contiguous if possible — while it still fails;
  4. print it as a pytest regression (or a lit test once you know which pass is wrong).

`ddmin` is Zeller's delta debugging for lists (e.g. a list of IR lines or of ops in a graph);
`shrink_case` is the structured version for (shape, dtype, layout) cases.
"""
import random

import numpy as np

from solutions.k3_numerics import check_close, edge_sizes, to_bf16, tolerance


# ---------------------------------------------------------------- generic list shrinking
def ddmin(items, fails):
    """Smallest sub-list (1-minimal) for which fails(sub) is still True. fails(items) must be True."""
    raise NotImplementedError("ddmin")


def shrink_int(n, fails, lo=1):
    """Smallest m in [lo, n] with fails(m) (assumes fails(n)); halve first, then step down."""
    raise NotImplementedError("shrink_int")


# ---------------------------------------------------------------- structured cases
DTYPE_ORDER = ["float32", "float16", "bfloat16"]       # simplest first


def shrink_case(case, fails):
    """case = {'shape': [..], 'dtype': str, 'contiguous': bool}. Shrink until nothing changes."""
    raise NotImplementedError("shrink_case")


def random_case(rng, block=64):
    sizes = edge_sizes(block, limit=4 * block)
    return {"shape": [rng.choice(sizes), rng.choice(sizes)], "dtype": rng.choice(DTYPE_ORDER),
            "contiguous": rng.random() < 0.7}


def fuzz(check, n=200, seed=0, block=64):
    """Run `check(case) -> bool ok` on n random cases; return (first failing case or None, tried)."""
    raise NotImplementedError("fuzz")


# ---------------------------------------------------------------- the op under test
NP = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}   # numpy has no bf16: values are rounded with to_bf16


def make_input(case, seed=0):
    m, n = case["shape"]
    rng = np.random.default_rng(seed)
    if case["contiguous"]:
        x = rng.standard_normal((m, n))
    else:
        x = rng.standard_normal((n, m)).T            # same values pattern, transposed strides
    x = x.astype(NP[case["dtype"]])
    return to_bf16(x) if case["dtype"] == "bfloat16" else x


def row_sum_reference(x):
    return np.asarray(x, np.float64).sum(axis=1)


def differential_check(impl, case, seed=0):
    """impl(x) vs float64 reference with a K3 tolerance for this dtype and row length."""
    raise NotImplementedError("differential_check")


def buggy_row_sum(x, BLOCK=64):
    """A 'tiled kernel' with a classic bug: the single-block path is masked, the multi-block path
    walks only full blocks and forgets the tail."""
    m, n = x.shape
    acc = np.zeros(m, np.float32)
    if n <= BLOCK:
        acc += x.astype(np.float32).sum(axis=1)
    else:
        for k0 in range(0, n - BLOCK + 1, BLOCK):     # BUG: drops n % BLOCK trailing columns
            acc += x[:, k0:k0 + BLOCK].astype(np.float32).sum(axis=1)
    return acc


def to_pytest(case, name="test_row_sum_regression", impl="row_sum", seed=0):
    """Emit a standalone regression test for the minimized case."""
    raise NotImplementedError("to_pytest")
