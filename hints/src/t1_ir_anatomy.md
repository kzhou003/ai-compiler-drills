title: T1 讲解：逐层解剖一个 matmul 的四层 IR
lede: 把 fixtures/t1 里的 matmul 从 TTIR 读到 PTX：每一层挑出最关键的几行逐行注释，说明哪些概念在这一层第一次出现、哪些在这一层消失；再讲清楚参考解里几个正则为什么那样写，以及在 RTX 3070 上怎样拿到你自己的四层 dump。

::: callout info 怎么读这一页
对照 [[here:solutions/t1_ir_anatomy.py|t1_ir_anatomy.py]] 和 `fixtures/t1/` 下的五个文件。fixture 按真实 Triton 输出的格式整理并做了删减（省略了大部分地址计算），关键结构保留。
:::

## ① 一张总表

| 层 | 文件 | 识别特征（detect_stage） | 本层第一次出现 |
|---|---|---|---|
| TTIR | matmul.ttir.mlir | 有 `tt.func`，没有 `#ttg.` | program_id、块张量、指针张量、`tt.dot` |
| TTGIR | matmul.ttgir.mlir | `#ttg.` 或 `"ttg.num-warps"` | layout encoding、shared memory、async copy |
| LLVM IR | matmul.llir.ll | `define` + `nvptx` | 线程号 `tid.x`、`addrspace(3)`、inline asm 的 MMA |
| PTX | matmul.ptx | `.version` + `.target sm_` | `.reg` 虚拟寄存器声明 |

[[here:solutions/t1_ir_anatomy.py@L19-L33|detect_stage（19–33 行）]] 从低层往高层判断：PTX 和 LLVM IR 的特征最不会误判；TTGIR 必须先于 TTIR 判断，因为 TTGIR 里也有 `tt.func`。

## ② TTIR：只有“块级计算”

```text
%res:3 = scf.for %iv = %c0_i32 to %c128_i32 step %c32_i32 iter_args(%acc = %cst, %pa = %3, %pb = %4) -> (...)
  %a = tt.load %pa : tensor<64x32x!tt.ptr<f16>>          ← 一整块指针，一次 load 一整块
  %d = tt.dot %a, %b, %acc : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
```

类型里只有形状和元素类型：`tensor<64x32xf16>` 不说“哪个线程拿哪个元素”。K 循环是普通的 `scf.for`，累加器和两个指针块是循环携带的值（`iter_args`）。

## ③ TTGIR：分工表出现了

```text
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
%0 = ttg.local_alloc : () -> !ttg.memdesc<3x64x32xf16, #shared, #smem, mutable>   ← 3 = num_stages
%2 = ttg.async_copy_global_to_local ...                                          ← cp.async
%a = ttg.local_load %7 : ... -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
%d = tt.dot %a, %b, %acc : ... -> tensor<64x64xf32, #mma>
%10 = ttg.convert_layout %9 : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked1>
```

::: flow TTGIR 里数据的路径
cfg cw=200 nw=172
n g 0 0.6 "global（#blocked）" "合并访存的 layout" k=acc
n s 1 0.6 "shared（#shared）" "3 级缓冲 · swizzle" k=teal
n d 2 0 "dot_op opIdx 0 / 1" "MMA 要的操作数分布" k=vio
n m 2 1.2 "#mma 累加器" "tt.dot 的结果" k=vio
n b 3 0.6 "#blocked1" "store 用" k=pass
e g -> s
e s -> d
e d -> m
e m -> b
:::

每一次 layout 改变都对应真实的数据搬运：global → shared 是 `cp.async`，shared → dot_op 是 `ldmatrix`，`#mma → #blocked1` 是一次 `convert_layout`（T2 判断它要不要过 shared memory）。`versionMajor = 2` 是 Ampere 的 `mma.sync`；在 Hopper 上同一个位置会是 `versionMajor = 3` 和 `ttng.warp_group_dot`（[[triton:test/TritonGPU/accelerate-matmul.mlir|accelerate-matmul.mlir]] 里的 CHECK）。

::: callout analogy 施工图上的图例
TTIR 是没有图例的户型图。TTGIR 文件开头的 `#blocked = ...`、`#mma = ...` 就是图例：后面每个类型里的 `, #mma>` 都在说“这块墙按 #mma 的分工砌”。读 TTGIR 的第一步永远是先读文件头的图例。
:::

## ④ LLVM IR 和 PTX：每个线程自己的指令

```text
%5  = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()                      ← 线程号出现，块张量消失
%30 = ... asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];" (ptr addrspace(3) %29)
%40 = ... asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, ..."
```

Triton 在 LLVM IR 里用 inline asm 发出 MMA，所以 [[here:solutions/t1_ir_anatomy.py@L101-L122|ptx_instructions / mma_shape（101–122 行）]] 对 LLVM IR 和 PTX 用的是同一组正则。`m16n8k16`：一条指令由整个 warp 合作完成，A 是 16×16、B 是 16×8，累加到 16×8 的 fp32；`row.col` 表示 A 行主序、B 列主序；`.f32.f16.f16.f32` 依次是 D、A、B、C 的类型。

## ⑤ concept_diff 的结果

[[here:solutions/t1_ir_anatomy.py@L126-L151|CONCEPTS / concepts / concept_diff（126–151 行）]] 对四个 fixture 算出：

| 变化 | 新增 | 消失 |
|---|---|---|
| TTIR → TTGIR | async_copy, layout_encoding, shared_memory | — |
| TTGIR → LLVM IR | mma_instruction, thread_id | block_tensor, high_level_dot, layout_encoding, pointer_tensor, program_id |
| LLVM IR → PTX | virtual_registers | — |

这张表就是 roadmap 要的“IR 注释版”的骨架。

## ⑥ 几个正则的细节

- `_OP`（[[here:solutions/t1_ir_anatomy.py@L33|33 行]]）：左边可能是 `%9 = `、`%res:2 = `、`%a, %b = `；op 名可能带引号（generic 形式 `"tt.reduce"(...)`）；op 名后面跟空白、括号、`{`、`<` 或行尾。
- `layout_aliases`：`#smem = #ttg.shared_memory` 没有 `<{...}>`，所以 body 部分是可选的。
- `_encoding_of`：`tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>` 里 encoding 本身含逗号，只能按**第一个**逗号切。

## ⑦ 源码与资料索引

- [[triton:third_party/nvidia/backend/compiler.py@def make_ttgir|make_ttgir 的 pass 列表]] · [[triton:third_party/nvidia/backend/compiler.py@def make_llir|make_llir]]
- [[triton:python/triton/compiler/compiler.py@class CompiledKernel|CompiledKernel.asm]]
- [[triton:test/Conversion/tritongpu_to_llvm.mlir@nvvm.ldmatrix|tritongpu_to_llvm.mlir 里的 ldmatrix / mma CHECK]]
- [[triton:README.md@MLIR_ENABLE_DUMP|README：MLIR_ENABLE_DUMP、TRITON_KERNEL_DUMP]]
- [PTX ISA：warp-level MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)

## ⑧ 自测

::: answer 一个 64×64×32 的 K 步、4 个 warp（warpsPerCTA = [2, 2]），需要多少条 m16n8k16？
每个 warp 负责 32×32 的输出块：M 方向 32/16 = 2，N 方向 32/8 = 4，K 方向 32/16 = 2，共 2·4·2 = 16 条，整个 CTA 64 条。fixture 的 PTX 只保留了 2 条示意；真实 dump 里你可以用 `ptx_instructions` 数出来对照。
:::

::: answer 为什么 vector add 的 TTIR 和 TTGIR 的 op 集合完全一样？
`convert_to_ttgpuir` 只给类型加 encoding；vector add 没有 dot、没有循环流水，后续 pass 也没有引入 shared memory 或 layout 转换。所以变的只有类型和 module 属性，`mlir_ops` 的计数完全相同（测试 `test_same_ops_ttir_vs_ttgir`）。
:::
