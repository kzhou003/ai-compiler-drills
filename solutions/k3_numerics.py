"""K3 · Numerical correctness: ULPs, tolerances, reassociation, edge shapes — reference solution.

Compilers reorder floating-point work all the time (split-K, tree reductions, fused epilogues,
e-graph rewrites like (a+b)+c -> a+(b+c)). The result is *different*, not necessarily *wrong*.
To tell the two apart you need: a distance in ULPs, a tolerance that scales with the dtype and the
reduction length, and inputs that hit the edges (non-divisible shapes, size 1, huge/tiny values).
"""
import math

import numpy as np

EPS = {"float32": 2.0 ** -23, "float16": 2.0 ** -10, "bfloat16": 2.0 ** -7}


def to_bf16(x):
    """Round float32 values to bfloat16 (round-to-nearest-even), returned as float32 values."""
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    rounding = ((bits >> 16) & 1) + 0x7FFF          # ties go to the even mantissa
    out = ((bits + rounding) >> 16) << 16
    return out.astype(np.uint32).view(np.float32)


def _ordered_int(a, dtype):
    """Map floats to integers whose order and spacing follow the float line (sign-magnitude ->
    two's-complement trick), so that |int(a) - int(b)| is the number of representable values
    between a and b."""
    if dtype == "float32":
        i = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    if dtype == "float16":
        i = np.asarray(a, np.float16).view(np.int16).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFF), i)
    if dtype == "bfloat16":
        i = (to_bf16(a).view(np.uint32) >> 16).astype(np.int64)
        return np.where(i >= 0x8000, -(i & 0x7FFF), i)
    raise ValueError(dtype)


def ulp_diff(a, b, dtype="float32"):
    """Number of representable `dtype` values between a and b (0 means identical)."""
    return np.abs(_ordered_int(a, dtype) - _ordered_int(b, dtype))


def sum_sequential(xs):
    acc = np.float32(0)
    for x in np.asarray(xs, np.float32):
        acc = np.float32(acc + x)
    return acc


def sum_pairwise(xs):
    """Tree reduction, the order a GPU block reduction (or tl.sum) uses."""
    v = list(np.asarray(xs, np.float32))
    if not v:
        return np.float32(0)
    while len(v) > 1:
        nxt = [np.float32(v[i] + v[i + 1]) for i in range(0, len(v) - 1, 2)]
        if len(v) % 2:
            nxt.append(v[-1])
        v = nxt
    return v[0]


def tolerance(dtype, k=1, c=2.0):
    """(rtol, atol) for comparing a result computed in `dtype` with a high-precision reference.

    The error of a length-k reduction grows roughly like sqrt(k) * eps when the rounding errors
    are random, so rtol = c * eps * sqrt(k). atol covers values near zero, where relative error
    is meaningless; using the same number is a reasonable default for O(1) data."""
    eps = EPS[dtype]
    rtol = c * eps * math.sqrt(max(k, 1))
    return rtol, rtol


def check_close(actual, expected, rtol, atol):
    """Like torch.testing.assert_close, but returns a report instead of raising.

    passes iff |actual - expected| <= atol + rtol * |expected| everywhere (NaNs must match)."""
    a = np.asarray(actual, np.float64)
    e = np.asarray(expected, np.float64)
    if a.shape != e.shape:
        return {"ok": False, "reason": f"shape {a.shape} != {e.shape}"}
    nan_a, nan_e = np.isnan(a), np.isnan(e)
    if (nan_a != nan_e).any():
        idx = tuple(int(i) for i in np.argwhere(nan_a != nan_e)[0])
        return {"ok": False, "reason": f"NaN mismatch at {idx}", "index": idx}
    diff = np.where(nan_a, 0.0, np.abs(a - e))
    allowed = atol + rtol * np.abs(np.where(nan_e, 0.0, e))
    bad = diff > allowed
    rel = diff / np.maximum(np.abs(e), 1e-30)
    worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(diff - allowed)), diff.shape)) if diff.size else ()
    return {"ok": not bad.any(), "mismatched": int(bad.sum()), "total": int(diff.size),
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "max_rel": float(np.where(nan_e, 0.0, rel).max()) if diff.size else 0.0, "index": worst}


def edge_sizes(block, limit=None):
    """Sizes that break tiled kernels: 1, block-1, block, block+1, 2*block+1, a prime, and
    (optionally) a large non-multiple. Sorted and de-duplicated."""
    s = {1, block - 1, block, block + 1, 2 * block + 1, 97}
    if limit:
        s.add(limit - (limit % block) + block // 2)
    return sorted(x for x in s if x >= 1)


def edge_values(dtype="float16"):
    """Values that stress a dtype: zeros of both signs, the smallest normal, the largest finite,
    values whose square overflows, and +/-inf / NaN."""
    fi = np.finfo(np.float16 if dtype == "float16" else np.float32)
    return np.array([0.0, -0.0, float(fi.tiny), float(fi.max), -float(fi.max), math.sqrt(float(fi.max)) * 2,
                     1.0, -1.0, math.inf, -math.inf, math.nan], dtype=np.float32)
