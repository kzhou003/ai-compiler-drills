import pytest

from drill import load

k2 = load("k2_bench_harness")


def test_percentile_matches_numpy_linear():
    xs = [5, 1, 4, 2, 3]
    assert k2.percentile(xs, 0) == 1 and k2.percentile(xs, 100) == 5 and k2.percentile(xs, 50) == 3
    assert k2.percentile([1, 2, 3, 4], 25) == pytest.approx(1.75)


def test_summarize_basic():
    s = k2.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0 and s["min"] == 1.0 and s["max"] == 100.0
    assert s["q1"] == 2.0 and s["q3"] == 4.0 and s["iqr"] == 2.0
    assert s["mean"] == 22.0          # the mean is dragged by the outlier, the median is not


def test_bench_skips_warmup_and_syncs():
    # warmup calls take 10 s (compilation), steady calls take 1 s
    clock = k2.FakeClock([10.0] * 3 + [1.0] * 100)
    syncs = []
    r = k2.bench(clock.work, warmup=3, repeats=20, timer=clock, sync=lambda: syncs.append(clock.t))
    assert r["n"] == 20 and r["median"] == 1.0 and max(r["samples"]) == 1.0
    assert clock.calls == 23
    assert len(syncs) == 1 + 2 * 20     # once after warmup, before and after every timed call


def test_compile_vs_run():
    clock = k2.FakeClock([1.0])
    built = []

    def make():
        clock.t += 7.0                   # "compilation"
        built.append(1)
        return clock.work

    r = k2.compile_vs_run(make, repeats=5, timer=clock)
    assert built == [1]
    assert r["first_call"] == pytest.approx(8.0)
    assert r["steady_median"] == pytest.approx(1.0)
    assert r["one_time_cost"] == pytest.approx(7.0)


def test_outliers():
    xs = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 5.0]
    assert k2.outliers(xs) == [6]


def test_compare_needs_more_than_noise():
    base = k2.summarize([10, 11, 12, 10.5, 11.5])
    clearly = k2.summarize([5, 5.5, 6, 5.2, 5.8])
    noisy = k2.summarize([9.5, 11, 12.5, 10, 11.8])
    assert k2.compare(base, clearly)["verdict"] == "faster"
    assert k2.compare(base, clearly)["speedup"] == pytest.approx(11 / 5.5)
    assert k2.compare(base, noisy)["verdict"] == "within noise"
    assert k2.compare(clearly, base)["verdict"] == "slower"
