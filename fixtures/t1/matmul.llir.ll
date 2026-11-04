; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @matmul_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) readnone captures(none) %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call i32 asm sideeffect "cp.async.cg.shared.global [ $1 + 0 ], [ $2 + 0 ], 0x10, $3;", "=r,r,l,r"(ptr addrspace(3) %20, ptr addrspace(1) %21, i32 %22)
  tail call void asm sideeffect "cp.async.wait_group 2;", ""()
  tail call void @llvm.nvvm.barrier0()
  %30 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r"(ptr addrspace(3) %29)
  %40 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};", "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3"(i32 %31, i32 %32, i32 %33, i32 %34, i32 %35, i32 %36, float %37, float %38, float %39, float %41)
  %50 = fptrunc float %42 to half
  store <4 x i32> %60, ptr addrspace(1) %61, align 16
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
!nvvm.annotations = !{!0}
!0 = !{ptr @matmul_kernel, !"maxntidx", i32 128}
