"""K2 · A benchmark harness you can trust — your implementation (run ./check.sh k2).

The same rules apply to a Triton kernel, a Luminal graph or a cuBLAS call:

  * warm up first (JIT compile, autotune, caches, clocks) and never time the warmup;
  * synchronize before reading the clock (GPU launches are asynchronous);
  * repeat, and report the median and the spread, not the best run;
  * keep compile / search time separate from steady-state run time;
  * only call a change faster when the difference is larger than the noise.

Everything takes an injectable `timer` and `sync`, so tests drive it with a fake clock and a
real run passes `time.perf_counter` and `torch.cuda.synchronize`.
"""
import statistics
import time


def percentile(xs, q):
    """Linear-interpolation percentile, q in [0, 100] (same as numpy's default)."""
    raise NotImplementedError("percentile")


def summarize(samples):
    """Robust statistics of a list of timings."""
    raise NotImplementedError("summarize")


def bench(fn, warmup=10, repeats=50, timer=time.perf_counter, sync=lambda: None):
    """Time fn() `repeats` times after `warmup` untimed calls.

    sync() runs before *starting* the clock (drain earlier async work) and before *stopping* it
    (wait for this call's work). Returns summarize(samples) plus the raw samples."""
    raise NotImplementedError("bench")


def compile_vs_run(make_fn, repeats=20, timer=time.perf_counter, sync=lambda: None):
    """Separate one-time cost (building / compiling / searching + the first call) from
    steady-state cost. make_fn() returns the callable; that construction is part of the
    one-time cost, exactly like `torch.compile`, a Triton JIT or a Luminal search."""
    raise NotImplementedError("compile_vs_run")


def outliers(samples, k=1.5):
    """Indices outside Tukey's fences [q1 - k*IQR, q3 + k*IQR]."""
    raise NotImplementedError("outliers")


def compare(base, new):
    """Is `new` faster than `base`? Both are summarize() results.

    speedup = base.median / new.median. The verdict is "faster" / "slower" only when the
    interquartile ranges do not overlap; otherwise "within noise"."""
    raise NotImplementedError("compare")


class FakeClock:
    """Deterministic clock for tests and for reasoning about the harness: each fn() call
    advances time by the next duration in `durations` (cycled)."""

    def __init__(self, durations):
        self.t = 0.0
        self.durations = list(durations)
        self.i = 0
        self.calls = 0

    def __call__(self):
        return self.t

    def work(self):
        self.t += self.durations[self.i % len(self.durations)]
        self.i += 1
        self.calls += 1
