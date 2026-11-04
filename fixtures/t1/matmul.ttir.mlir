module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>>
    %res:3 = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%acc = %cst, %pa = %3, %pb = %4) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>) : i32 {
      %a = tt.load %pa : tensor<64x32x!tt.ptr<f16>>
      %b = tt.load %pb : tensor<32x64x!tt.ptr<f16>>
      %d = tt.dot %a, %b, %acc : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
      %na = tt.addptr %pa, %cst_a : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
      %nb = tt.addptr %pb, %cst_b : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
      scf.yield %d, %na, %nb : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>
    }
    %5 = arith.truncf %res#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %6 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
    tt.store %6, %5 : tensor<64x64x!tt.ptr<f16>>
    tt.return
  }
}
