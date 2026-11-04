"""L5 · Differential test: ReferenceRuntime vs CUDA, written as a regression test — your implementation (run ./check.sh l5).

Luminal ships two runtimes for the same compiled graph: `luminal_reference` (a CPU interpreter
that defines the semantics, with explicit narrow-float and tf32 emulation: narrow_float.rs,
tf32.rs) and `luminal_cuda_lite` (the real device). A backend contribution is judged by
"does CUDA agree with Reference within a tolerance I can justify?".

This drill builds that harness in Python around one op (softmax over the last axis, a classic
place for backend bugs: max-subtraction, masked tails, fp16 overflow):

  reference_softmax   float64, the semantics
  device_softmax      a 'device' implementation: tiled in blocks, fp32 accumulation, output in
                      the storage dtype — with switchable, realistic bugs
  tolerance_for       rtol/atol from the dtype's eps and the reduction length (K3), plus a
                      documented rationale string that goes into the test
  run_differential    random shapes (edge-heavy), dtypes, seeds -> list of failures
  emit_rust_test      the regression test in the style of crates/luminal_cuda_lite/tests
"""
import math
import random

import numpy as np

from solutions.k3_numerics import EPS, check_close, edge_sizes, to_bf16

STORAGE = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}


def store(x, dtype):
    x = np.asarray(x, np.float32).astype(STORAGE[dtype])
    return to_bf16(x) if dtype == "bfloat16" else x


def reference_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def device_softmax(x, dtype, block=64, bug=None):
    """Two passes over the row in blocks of `block`: online max+sum, then normalize.
    bug: None | 'no_max_sub' (overflows for large inputs) | 'tail' (last partial block ignored in
    the sum) | 'fp16_acc' (accumulate in the storage dtype)."""
    x = store(x, dtype).astype(np.float32)
    rows, n = x.shape
    acc_t = np.float16 if bug == "fp16_acc" else np.float32
    out = np.empty_like(x)
    for r in range(rows):
        m = np.float32(-np.inf) if bug != "no_max_sub" else np.float32(0)
        s = acc_t(0)
        stop = n if bug != "tail" else (n // block) * block or n
        for k0 in range(0, stop, block):
            blk = x[r, k0:min(k0 + block, stop)]
            if bug != "no_max_sub":
                m_new = max(m, blk.max())
                s = acc_t(s * acc_t(np.exp(m - m_new)) + acc_t(np.exp(blk - m_new).sum()))
                m = m_new
            else:
                with np.errstate(over="ignore"):
                    s = acc_t(s + acc_t(np.exp(blk).sum()))
        with np.errstate(over="ignore", invalid="ignore"):
            out[r] = np.exp(x[r] - m) / np.float32(s)
    return store(out, dtype)


def tolerance_for(dtype, n):
    """(rtol, atol, rationale). Softmax outputs are in [0, 1]; the error comes from (a) rounding
    the *output* to the storage dtype (half an ulp: eps/2 relative) and (b) the length-n sum in
    fp32 (~sqrt(n) * eps32). atol guards the tiny probabilities where relative error is noise."""
    raise NotImplementedError("tolerance_for")


def make_input(rows, n, dtype, seed, scale=3.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, n)) * scale
    return store(x, dtype)


def run_differential(impl, cases=60, seed=0, block=64, dtypes=("float32", "float16", "bfloat16"), scale=3.0):
    """Returns (failures, tried). failures: [{'rows','n','dtype','seed','report'}]."""
    raise NotImplementedError("run_differential")


def smallest_failure(fails):
    raise NotImplementedError("smallest_failure")


RUST_DTYPE = {"float32": "F32", "float16": "F16", "bfloat16": "Bf16"}


def emit_rust_test(f, name="softmax_matches_reference"):
    """A Rust regression test sketch: same graph compiled for both runtimes, compared with the
    justified tolerance. (Adapt the API calls to the current crate before submitting.)"""
    raise NotImplementedError("emit_rust_test")
