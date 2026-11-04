#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %pid = tt.get_program_id x : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x64x32xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<3x32x64xf16, #shared, #smem, mutable>
    %2 = ttg.async_copy_global_to_local %arg3, %4 mask %5 : tensor<64x32x!tt.ptr<f16>, #blocked> -> <64x32xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %res:2 = scf.for %iv = %c0 to %c128 step %c32 iter_args(%acc = %cst, %tok = %3) -> (tensor<64x64xf32, #mma>, !ttg.async.token) : i32 {
      %w = ttg.async_wait %tok {num = 2 : i32}
      %a = ttg.local_load %7 : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %b = ttg.local_load %8 : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %d, %w : tensor<64x64xf32, #mma>, !ttg.async.token
    }
    %9 = arith.truncf %res#0 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    %10 = ttg.convert_layout %9 : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked1>
    tt.store %11, %10 : tensor<64x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
