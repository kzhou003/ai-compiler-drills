import numpy as np

from drill import load

l5 = load("l5_differential")


def impl(bug=None):
    return lambda x, d: l5.device_softmax(x, d, bug=bug)


def test_reference_is_stable():
    y = l5.reference_softmax(np.array([[1000.0, 1000.0], [0.0, -np.inf]]))
    np.testing.assert_allclose(y, [[0.5, 0.5], [1.0, 0.0]])


def test_correct_device_passes_everywhere():
    fails, tried = l5.run_differential(impl(None), cases=60, seed=0)
    assert tried == 60 and fails == []
    fails, _ = l5.run_differential(impl(None), cases=30, seed=1, scale=40.0)
    assert fails == []


def test_tail_bug_found_and_minimized():
    fails, _ = l5.run_differential(impl("tail"), cases=60, seed=0)
    assert len(fails) == 26
    f = l5.smallest_failure(fails)
    assert f["rows"] == 1 and f["n"] % 64 != 0 and f["n"] > 64


def test_fp16_accumulation_bug_found():
    fails, _ = l5.run_differential(impl("fp16_acc"), cases=60, seed=0)
    assert len(fails) == 11 and all(f["report"]["max_abs"] < 1e-2 for f in fails)   # small but real


def test_overflow_bug_needs_large_inputs():
    # N(0, 3) inputs never overflow exp(): the missing max-subtraction is invisible ...
    fails, _ = l5.run_differential(impl("no_max_sub"), cases=60, seed=0)
    assert fails == []
    # ... until the test includes large logits
    fails, _ = l5.run_differential(impl("no_max_sub"), cases=60, seed=0, scale=40.0)
    assert len(fails) == 44


def test_tolerance_rationale():
    rtol, atol, why = l5.tolerance_for("float16", 65)
    assert rtol > 2 * 2 ** -10 and atol < rtol
    assert "float16" in why and "n=65" in why
    r32, _, _ = l5.tolerance_for("float32", 65)
    assert r32 < rtol < l5.tolerance_for("bfloat16", 65)[0]


def test_emit_rust_test():
    fails, _ = l5.run_differential(impl("tail"), cases=60, seed=0)
    src = l5.emit_rust_test(l5.smallest_failure(fails))
    assert src.startswith("#[test]") and "softmax(1)" in src and "DType::F32" in src
    assert "tolerance:" in src and src.count("{") == src.count("}")
