"""K2 · A benchmark harness you can trust — reference solution.

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
    s = sorted(xs)
    if not s:
        raise ValueError("no samples")
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(samples):
    """Robust statistics of a list of timings."""
    med = statistics.median(samples)
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    q1, q3 = percentile(samples, 25), percentile(samples, 75)
    return {
        "n": len(samples), "median": med, "mean": mean, "stdev": sd, "min": min(samples), "max": max(samples),
        "p10": percentile(samples, 10), "p90": percentile(samples, 90),
        "q1": q1, "q3": q3, "iqr": q3 - q1, "cv": sd / mean if mean else 0.0,
    }


def bench(fn, warmup=10, repeats=50, timer=time.perf_counter, sync=lambda: None):
    """Time fn() `repeats` times after `warmup` untimed calls.

    sync() runs before *starting* the clock (drain earlier async work) and before *stopping* it
    (wait for this call's work). Returns summarize(samples) plus the raw samples."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        sync()
        t0 = timer()
        fn()
        sync()
        samples.append(timer() - t0)
    out = summarize(samples)
    out["samples"] = samples
    return out


def compile_vs_run(make_fn, repeats=20, timer=time.perf_counter, sync=lambda: None):
    """Separate one-time cost (building / compiling / searching + the first call) from
    steady-state cost. make_fn() returns the callable; that construction is part of the
    one-time cost, exactly like `torch.compile`, a Triton JIT or a Luminal search."""
    sync()
    t0 = timer()
    fn = make_fn()
    fn()
    sync()
    first = timer() - t0
    steady = bench(fn, warmup=2, repeats=repeats, timer=timer, sync=sync)
    return {"first_call": first, "steady_median": steady["median"],
            "one_time_cost": first - steady["median"], "steady": steady}


def outliers(samples, k=1.5):
    """Indices outside Tukey's fences [q1 - k*IQR, q3 + k*IQR]."""
    q1, q3 = percentile(samples, 25), percentile(samples, 75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return [i for i, x in enumerate(samples) if x < lo or x > hi]


def compare(base, new):
    """Is `new` faster than `base`? Both are summarize() results.

    speedup = base.median / new.median. The verdict is "faster" / "slower" only when the
    interquartile ranges do not overlap; otherwise "within noise"."""
    speedup = base["median"] / new["median"]
    overlap = not (new["q3"] < base["q1"] or base["q3"] < new["q1"])
    if overlap:
        verdict = "within noise"
    else:
        verdict = "faster" if speedup > 1 else "slower"
    return {"speedup": speedup, "verdict": verdict}


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
