import pytest

from drill import load

k4 = load("k4_fusion")
S = 4096
PF, BW = 40e12, 448e9


def test_traffic_unfused_vs_fused():
    st = k4.chain_mlp_epilogue(S, S, S)
    mn = S * S
    un = k4.traffic(st, [1, 1, 1, 1])
    # matmul: A + B + C ; bias: C + b + C' ; gelu: 2 MN ; residual: 3 MN   (x 2 bytes)
    assert un["bytes"] == (2 * mn + mn + (mn + S + mn) + 2 * mn + 3 * mn) * 2
    assert un["launches"] == 4
    fused = k4.traffic(st, [4])
    assert fused["bytes"] == (2 * mn + S + mn + mn) * 2 and fused["launches"] == 1
    assert fused["peak_regs"] == 128 + 8 + 16 + 8
    assert k4.traffic(st, [1, 3])["bytes"] == k4.traffic(st, [2, 2])["bytes"]


def test_groups_validation():
    st = k4.chain_mlp_epilogue(8, 8, 8)
    with pytest.raises(AssertionError):
        k4.groups_of(st, [2, 1])


def test_best_plan_fuses_the_epilogue():
    t, plan = k4.best_plan(k4.chain_mlp_epilogue(S, S, S), PF, BW)
    assert plan == [4]
    t_un = k4.time_estimate(k4.chain_mlp_epilogue(S, S, S), [1, 1, 1, 1], PF, BW)
    assert t < t_un


def test_register_pressure_can_make_fusion_lose():
    n = 1 << 26
    heavy = [k4.Stage("a", n, n, n, 200), k4.Stage("b", n, 0, n, 100)]
    fused = k4.time_estimate(heavy, [2], PF, BW, spill_penalty=3.0)
    split = k4.time_estimate(heavy, [1, 1], PF, BW, spill_penalty=3.0)
    assert split < fused
    assert k4.best_plan(heavy, PF, BW, spill_penalty=3.0)[1] == [1, 1]
    light = [k4.Stage("a", n, n, n, 100), k4.Stage("b", n, 0, n, 100)]
    assert k4.best_plan(light, PF, BW, spill_penalty=3.0)[1] == [2]


def test_small_problem_is_launch_bound():
    st = k4.chain_mlp_epilogue(16, 64, 64)
    t_un = k4.time_estimate(st, [1, 1, 1, 1], PF, BW)
    t_f = k4.time_estimate(st, [4], PF, BW)
    assert t_un == pytest.approx(4 * 5e-6, rel=0.05) and t_f == pytest.approx(5e-6, rel=0.2)
