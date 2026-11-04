import math

import numpy as np
import pytest

from drill import load

k3 = load("k3_numerics")


def test_bf16_rounding():
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, 3.14159265], np.float32)
    got = k3.to_bf16(x)
    assert got[0] == 1.0
    assert got[1] == 1.0                      # exactly halfway -> even mantissa (1.0)
    assert got[2] == np.float32(1.0 + 2 ** -6)  # halfway between 1+2^-7 and 1+2^-6 -> even (1+2^-6)
    assert abs(got[3] - 3.140625) < 1e-7      # bf16 keeps 8 significant bits


def test_ulp_diff():
    one = np.float32(1.0)
    nxt = np.nextafter(one, np.float32(2.0))
    assert k3.ulp_diff(one, nxt) == 1
    assert k3.ulp_diff(one, one) == 0
    assert k3.ulp_diff(np.float32(-0.0), np.float32(0.0)) == 0
    tiny = np.nextafter(np.float32(0), np.float32(1))
    assert k3.ulp_diff(-tiny, tiny) == 2      # across zero
    assert k3.ulp_diff(np.float16(1.0), np.float16(1.0 + 2 ** -10), "float16") == 1
    assert k3.ulp_diff(1.0, 1.0 + 2 ** -7, "bfloat16") == 1


def test_reassociation_changes_the_bits():
    xs = np.array([1e8, 1.0, -1e8, 1.0] * 4, np.float32)
    seq, tree = k3.sum_sequential(xs), k3.sum_pairwise(xs)
    assert seq != tree                          # same math, different rounding
    assert k3.sum_pairwise([1.0, 2.0, 3.0]) == 6.0 and k3.sum_pairwise([]) == 0.0


def test_tolerance_scales():
    r1, a1 = k3.tolerance("float32", 1)
    r4, _ = k3.tolerance("float32", 4)
    assert r1 == a1 == 2 * 2 ** -23
    assert r4 == pytest.approx(2 * r1)
    assert k3.tolerance("bfloat16", 1)[0] > k3.tolerance("float16", 1)[0] > r1


def test_check_close_report():
    e = np.array([1.0, 2.0, 0.0, np.nan])
    ok = k3.check_close(e + 1e-9, e, rtol=1e-6, atol=1e-6)
    assert ok["ok"] and ok["mismatched"] == 0
    bad = k3.check_close(np.array([1.0, 2.1, 0.0, np.nan]), e, rtol=1e-3, atol=1e-6)
    assert not bad["ok"] and bad["mismatched"] == 1 and bad["index"] == (1,)
    nanmis = k3.check_close(np.array([1.0, 2.0, np.nan, np.nan]), e, 1e-3, 1e-3)
    assert not nanmis["ok"] and "NaN" in nanmis["reason"]
    assert not k3.check_close(np.zeros(3), np.zeros(4), 1, 1)["ok"]


def test_fp16_matmul_needs_a_tolerance():
    rng = np.random.default_rng(0)
    k = 512
    a, b = rng.standard_normal((64, k)), rng.standard_normal((k, 64))
    ref = a @ b
    low = (a.astype(np.float16) @ b.astype(np.float16)).astype(np.float64)
    assert not k3.check_close(low, ref, *k3.tolerance("float32", k))["ok"]
    rtol, atol = k3.tolerance("float16", k, c=4.0)
    assert k3.check_close(low, ref, rtol, atol * math.sqrt(k))["ok"]


def test_edge_sizes_and_values():
    s = k3.edge_sizes(64, limit=1000)
    assert s[0] == 1 and {63, 64, 65, 129, 97}.issubset(s) and s == sorted(set(s))
    assert any(x % 64 for x in s if x > 129)
    v = k3.edge_values("float16")
    assert np.isnan(v).any() and np.isinf(v).sum() == 2
    assert np.isinf(np.float16(v[5]) * np.float16(v[5]))    # its square overflows fp16
