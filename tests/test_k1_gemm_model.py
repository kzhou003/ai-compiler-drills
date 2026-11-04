import math

import pytest

from drill import load

k1 = load("k1_gemm_model")
G = k1.RTX3070
C = k1.TileConfig


def test_flops_and_min_bytes():
    assert k1.gemm_flops(4096, 4096, 4096) == 2 * 4096 ** 3
    # fp16 A, B, C: three 4096x4096 matrices x 2 bytes
    assert k1.gemm_min_bytes(4096, 4096, 4096) == 3 * 4096 * 4096 * 2
    assert k1.gemm_min_bytes(8, 16, 4, in_bytes=4, out_bytes=4) == (8 * 4 + 4 * 16 + 8 * 16) * 4


def test_tiled_bytes_rereads_operands():
    # 1024^3 with 128x128 tiles: A is read 1024/128 = 8 times, B 8 times
    got = k1.tiled_bytes(1024, 1024, 1024, C(128, 128, 64))
    assert got == (1024 * 1024 * 8 * 2) * 2 + 1024 * 1024 * 2
    # partial tiles count as a full re-read
    assert k1.tiled_bytes(100, 100, 10, C(64, 64, 16), in_bytes=1, out_bytes=1) == 100 * 10 * 2 * 2 + 100 * 100


def test_smem_regs_threads():
    cfg = C(128, 128, 64, num_warps=4, num_stages=3)
    assert k1.smem_bytes(cfg) == 3 * (128 * 64 + 64 * 128) * 2 == 98304
    assert k1.threads(cfg) == 128
    assert k1.regs_per_thread(cfg) == 128 * 128 // 128 + 40


def test_occupancy_limiters():
    big = k1.occupancy(G, C(128, 128, 64, 4, 3))
    assert big["valid"] and big["blocks"] == 1 and big["limiter"] == "smem"

    small = k1.occupancy(G, C(64, 64, 32, 4, 2))
    # threads 12, regs 65536 // (72*128) = 7, smem 102400 // 16384 = 6, blocks 16
    assert small["blocks"] == 6 and small["limiter"] == "smem"

    too_much_smem = k1.occupancy(G, C(128, 256, 64, 8, 3))
    assert not too_much_smem["valid"] and too_much_smem["limiter"] == "smem"

    spills = k1.occupancy(G, C(256, 256, 32, 4, 2))
    assert not spills["valid"] and spills["limiter"] == "regs"


def test_waves_and_tail():
    w = k1.waves(G, 4096, 4096, C(128, 128, 64), blocks_per_sm=1)
    assert w["tiles"] == 1024 and w["per_wave"] == 46 and w["waves"] == 23
    assert w["tail_fill"] == pytest.approx(12 / 46)
    exact = k1.waves(G, 46 * 64, 64, C(64, 64, 32), blocks_per_sm=1)
    assert exact["waves"] == 1 and exact["tail_fill"] == 1.0


def test_roofline_bounds():
    flops = k1.gemm_flops(4096, 4096, 4096)
    r = k1.roofline(G, flops, k1.gemm_min_bytes(4096, 4096, 4096))
    assert r["bound"] == "compute" and r["time"] == pytest.approx(flops / G.peak_flops)
    # a vector add: 1 flop per 6 bytes
    n = 1 << 24
    assert k1.roofline(G, n, 6 * n)["bound"] == "memory"
    # tiny problem: below the launch overhead
    tiny = k1.roofline(G, k1.gemm_flops(64, 64, 64), k1.gemm_min_bytes(64, 64, 64))
    assert tiny["bound"] == "launch" and tiny["time"] == G.launch_overhead


def test_analyze_bundle():
    a = k1.analyze(G, 2048, 2048, 2048, C(128, 128, 32, 4, 3))
    assert a["flops"] == 2 * 2048 ** 3
    assert a["intensity"] == pytest.approx(a["flops"] / a["tiled_bytes"])
    assert a["occupancy"]["valid"] and "waves" in a
    bad = k1.analyze(G, 2048, 2048, 2048, C(256, 256, 64, 4, 4))
    assert not bad["occupancy"]["valid"] and "waves" not in bad
    assert math.isclose(a["roofline"]["t_compute"], a["flops"] / G.peak_flops)
