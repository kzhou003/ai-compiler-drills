from drill import load

t2 = load("t2_coalesce_layout")
B = t2.Blocked


def test_parse_roundtrip():
    s = "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>"
    L = t2.parse_blocked(s)
    assert L == B((1, 4), (2, 16), (4, 1), (1, 0))
    assert str(L) == s


def test_reproduces_coalesce_mlir_transpose():
    # test/TritonGPU/coalesce.mlir @transpose: 64x64 f32, num-warps 4, tt.divisibility = 16
    row = t2.coalesced_layout([64, 64], order=[1, 0], elem_bits=32, contiguity=64)
    col = t2.coalesced_layout([64, 64], order=[0, 1], elem_bits=32, contiguity=64)
    assert str(row) == ("#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], "
                        "warpsPerCTA = [4, 1], order = [1, 0]}>")
    assert str(col) == ("#ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], "
                        "warpsPerCTA = [1, 4], order = [0, 1]}>")


def test_elements_per_thread_limits():
    # fp16: 128 bits -> 8 elements per thread
    assert t2.elements_per_thread([128, 64], [1, 0], 16, 64, 16, 4) == 8
    # alignment 8 bytes only -> 2 fp32
    assert t2.elements_per_thread([128, 64], [1, 0], 32, 64, 8, 4) == 2
    # contiguity 1 (gather-like pointer) -> 1
    assert t2.elements_per_thread([128, 64], [1, 0], 32, 1, 16, 4) == 1
    # tiny tensor: 64 elements over 128 threads -> 1
    assert t2.elements_per_thread([64], [0], 32, 64, 16, 4) == 1


def test_build_blocked_1d_and_small():
    assert t2.build_blocked([1024], [4], [0], 4) == B((4,), (32,), (4,), (0,))
    # 16 columns of fp16 with 8 per thread: only 2 lanes along dim 1, the rest go to rows
    L = t2.build_blocked([64, 16], [1, 8], [1, 0], 4)
    assert L == B((1, 8), (16, 2), (4, 1), (1, 0))


def test_thread_elements_and_lane_table():
    L = B((1, 4), (2, 16), (4, 1), (1, 0))
    assert t2.tile(L) == (8, 64)
    assert t2.thread_elements(L, [8, 64], 0, 0) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert t2.thread_elements(L, [8, 64], 0, 17) == [(1, 4), (1, 5), (1, 6), (1, 7)]
    assert t2.thread_elements(L, [8, 64], 3, 16)[0] == (7, 0)
    # 64x64: the tile repeats 8 times along rows -> 32 registers per thread
    regs = t2.thread_elements(L, [64, 64], 0, 0)
    assert len(regs) == 32 and regs[4] == (8, 0)
    table = t2.lane_table(L, [8, 64])
    assert len(table) == 32 and table[16][1][0] == (1, 0)


def test_broadcast_when_tensor_smaller_than_tile():
    L = B((1, 4), (2, 16), (4, 1), (1, 0))
    own = t2.owners(L, [4, 64])          # tile is 8 rows, tensor has 4
    assert len(own) == 4 * 64
    assert len(own[(0, 0)]) == 2          # warp 0 lane 0 and warp 2 lane 0
    assert own[(0, 0)] == {(0, 0), (2, 0)}


def test_coalescing():
    row = B((1, 4), (2, 16), (4, 1), (1, 0))
    n, sectors, eff = t2.first_load_sectors(row, [64, 64], 4)
    assert n == 32 * 16 and sectors == 16 and eff == 1.0
    # the same row layout reading a column-major (transposed) tensor
    n, sectors, eff = t2.first_load_sectors(row, [64, 64], 4, strides=[1, 64])
    assert sectors == 64 and eff == 0.25          # rows 0 and 1 share a sector, 64 columns do not
    # one element per lane, lanes along the slow dim -> 32 different rows
    bad = B((1, 1), (32, 1), (4, 1), (1, 0))
    assert t2.first_load_sectors(bad, [128, 64], 4)[1] == 32


def test_classify_conversion():
    shape = [64, 64]
    row = B((1, 4), (2, 16), (4, 1), (1, 0))
    assert t2.classify_conversion(row, row, shape) == "registers"
    # order also decides how lanes are numbered, so flipping it is NOT a register-only change
    flipped = B((1, 4), (2, 16), (4, 1), (0, 1))
    assert t2.classify_conversion(row, flipped, shape) != "registers"
    # each warp keeps the same rows; only which lane holds which column changes
    narrow = B((1, 2), (2, 16), (4, 1), (1, 0))
    assert t2.classify_conversion(row, narrow, shape) == "warp_shuffle"
    # lanes stacked 16 tall: warp 0 now owns rows 0-15, which warps 0-3 shared before
    tall = B((1, 4), (16, 2), (4, 1), (1, 0))
    assert t2.classify_conversion(row, tall, shape) == "shared_memory"
    col = B((4, 1), (16, 2), (1, 4), (0, 1))
    assert t2.classify_conversion(row, col, shape) == "shared_memory"
