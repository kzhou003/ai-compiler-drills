lede: Triton 把一个 Python 函数编译成一个 GPU kernel。这一页画出它的全部阶段、每个阶段新增的概念，以及源码里对应的目录，T1–T5 都在这张地图上。
# Triton 编译器全景

## 一条流水线，五个阶段

NVIDIA 后端的阶段定义在 [[triton:third_party/nvidia/backend/compiler.py@def make_ttir|third_party/nvidia/backend/compiler.py]]：`make_ttir` → `make_ttgir` → `make_llir` → `make_ptx` → `make_cubin`。每一步的输出都能在 `kernel[grid](...)` 返回的 `CompiledKernel.asm` 里看到。

::: flow @triton.jit → cubin
cfg cw=245 nw=175 rh=100
n py 0 0 "@triton.jit 函数" "Python AST" k=acc
n ttir 1 0 "TTIR" "tt.dot，还没有 layout" k=warn
n ttgir 2 0 "TTGIR" "#blocked / #mma / smem" k=vio
n llir 3 0 "LLVM IR" "nvptx64，inline asm mma" k=teal
n ptx 3 1.25 "PTX" "mma.sync / ldmatrix" k=pass
n cubin 2 1.25 "cubin" "ptxas 生成" k=dark
e py -> ttir "AST"
e ttir -> ttgir "layout"
e ttgir -> llir "to LLVM"
e llir -> ptx "NVPTX"
e ptx -> cubin "ptxas"
cap: 最重要的一步是 TTIR → TTGIR：从这里开始，每个张量类型都带上 layout encoding（`tensor<64x64xf32, #blocked>`），编译器开始决定“哪个线程拿哪个元素”。T1 让你在四层 IR 里把这些概念逐个找出来。
:::

| 阶段 | 关键概念 | 源码入口 |
|---|---|---|
| Frontend | `@triton.jit`、AST → IR、`constexpr` 特化 | [[triton:python/triton/runtime/jit.py|runtime/jit.py]] · [[triton:python/triton/compiler/code_generator.py|compiler/code_generator.py]] · [[triton:python/triton/compiler/compiler.py|compiler/compiler.py]] |
| TTIR | program instance、block tensor、mask、`tt.dot`、canonicalization | [[triton:include/triton/Dialect/Triton/IR/TritonOps.td|TritonOps.td]] |
| TTGIR | Blocked / Slice / MMA / DotOperand / Shared layout，`ttg.convert_layout` | [[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td|TritonGPUAttrDefs.td]] · `lib/Dialect/TritonGPU/Transforms/` |
| Lowering | TritonGPU → LLVM/NVVM → PTX | `lib/Conversion/`、`third_party/nvidia/lib/` |
| Testing | lit + FileCheck、Python 单测 | `test/`（[[triton:test/TritonGPU/coalesce.mlir|coalesce.mlir]] 等） · [[triton:python/test/unit/language/test_core.py|test_core.py]] |

## TTGIR 上的 pass 链

`make_ttgir` 里的 pass 顺序本身就是一份“编译器做了哪些优化”的目录（摘自 compiler.py，SM86 会走到的部分）：

```text
convert_to_ttgpuir   给每个张量一个默认 layout（num_warps、threads-per-warp=32）
coalesce             为 load/store 选合并访存的 BlockedEncoding        ← T2 复刻这一个
remove_layout_conversions   删掉多余的 ttg.convert_layout（反复跑几次）
optimize_thread_locality
accelerate_matmul    tt.dot 换成 MMA layout（Ampere：nvidia_mma v2）    ← T1 在 IR 里找它
optimize_dot_operands
pipeline             按 num_stages 做软件流水（cp.async + async_wait）
prefetch / reduce_data_duplication / reorder_instructions ...
```

::: callout analogy 装修工程的图纸层次
TTIR 是**户型图**：哪里有墙、哪里是厨房（load、dot、store），不管谁来施工。TTGIR 是**施工分工图**：每面墙分给哪个班组、每人负责哪几块砖（layout），哪些材料要先运到工地仓库（shared memory）。LLVM IR / PTX 是**每个工人的操作单**：第几步拿哪块砖、用哪台机器（`mma.sync`）。`convert_layout` 就是两个班组之间交接材料——同一班组内递一下（shuffle）很便宜，跨班组要先放到仓库再取（shared memory）就贵了。
:::

## 读源码的顺序

1. **Kernel semantics**：官方 tutorials 和 [Triton Puzzles](https://github.com/srush/Triton-Puzzles)：program instance、block tensor、mask、reduction、dot。
2. **Frontend**：`@triton.jit` → `JITFunction.run` → `compile()`，看 `constexpr` 如何变成特化。
3. **TTIR**：Triton dialect 的 op 定义、canonicalization。
4. **TTGIR**：重点。layout 定义、[[triton:lib/Dialect/TritonGPU/Transforms/Coalesce.cpp|Coalesce.cpp]]、[[triton:lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp|RemoveLayoutConversions.cpp]]、[[triton:lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp|AccelerateMatmul.cpp]]。
5. **Lowering**：选 `tl.dot` 和 `convert_layout` 两条路径追到 PTX。
6. **Testing**：lit/FileCheck（T5）、Python 测试、最小复现（T4）。

::: callout info Gluon：自己写 layout
Gluon 是 Triton 里更低层的编程模型：layout 不再由编译器推断，而是你在代码里写出来。T2 算的 BlockedLayout，在 Gluon 里就是 `ttgl.BlockedLayout(size_per_thread=..., threads_per_warp=..., warps_per_cta=..., order=...)`。入门看 [[triton:python/tutorials/gluon/01-intro.py|gluon/01-intro.py]] 和 [[triton:python/tutorials/gluon/02-layouts.py|gluon/02-layouts.py]]。
:::

::: answer 自测：同一个 vector-add 的 TTIR 和 TTGIR，op 集合会变吗？
不会（T1 的测试就断言了这一点）。变的是**类型**：`tensor<1024xf32>` 变成 `tensor<1024xf32, #blocked>`，文件开头多了 `#blocked = #ttg.blocked<{sizePerThread = [4], ...}>` 和 `"ttg.num-warps" = 4` 之类的 module 属性。对 matmul 则不同：TTGIR 会多出 `ttg.local_alloc`、`ttg.async_copy_global_to_local`、`ttg.local_load` 和 `ttg.convert_layout`，这些是 pipeline 和 MMA layout 带来的新 op。
:::
