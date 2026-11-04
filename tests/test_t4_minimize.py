import numpy as np

from drill import load

t4 = load("t4_minimize")


def test_ddmin_finds_the_two_culprits():
    items = list(range(20))
    fails = lambda xs: 3 in xs and 17 in xs
    assert sorted(t4.ddmin(items, fails)) == [3, 17]
    assert t4.ddmin(list("abcdefgh"), lambda xs: "e" in xs) == ["e"]


def test_ddmin_ir_lines():
    ir = ["%0 = tt.get_program_id x", "%1 = tt.make_range", "%2 = tt.load %p", "%3 = tt.dot %a, %b",
          "%4 = ttg.convert_layout %3", "tt.store %q, %4", "tt.return"]
    crash = lambda lines: any("tt.dot" in l for l in lines) and any("convert_layout" in l for l in lines)
    assert t4.ddmin(ir, crash) == ["%3 = tt.dot %a, %b", "%4 = ttg.convert_layout %3"]


def test_shrink_int():
    assert t4.shrink_int(1000, lambda m: m >= 65) == 65
    assert t4.shrink_int(1000, lambda m: m % 64 != 0 and m > 64) == 65
    assert t4.shrink_int(7, lambda m: True) == 1


def test_reference_impl_passes_and_bug_is_found():
    good = lambda x: x.astype(np.float32).sum(axis=1)
    for case in [{"shape": [3, 129], "dtype": "float16", "contiguous": False},
                 {"shape": [1, 1], "dtype": "float32", "contiguous": True}]:
        assert t4.differential_check(good, case)
    bad, tried = t4.fuzz(lambda c: t4.differential_check(t4.buggy_row_sum, c), n=200, seed=0)
    assert bad is not None and bad["shape"][1] > 64 and bad["shape"][1] % 64 != 0


def test_shrink_case_to_minimal():
    check = lambda c: t4.differential_check(t4.buggy_row_sum, c)
    start = {"shape": [129, 255], "dtype": "bfloat16", "contiguous": False}
    assert not check(start)
    m = t4.shrink_case(start, lambda c: not check(c))
    assert m == {"shape": [1, 65], "dtype": "float32", "contiguous": True}


def test_to_pytest_runs():
    code = t4.to_pytest({"shape": [1, 65], "dtype": "float32", "contiguous": True}, impl="buggy")
    ns = {"np": np, "buggy": t4.buggy_row_sum}
    exec(code, ns)
    try:
        ns["test_row_sum_regression"]()
    except AssertionError:
        pass
    else:
        raise AssertionError("the emitted regression test should fail on the buggy kernel")
    ns["buggy"] = lambda x: x.astype(np.float32).sum(axis=1)
    exec(code, ns)
    ns["test_row_sum_regression"]()
