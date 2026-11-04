lede: Luminal 是一个“记录整张图、在等价程序里实测搜索、再生成执行计划”的 inference compiler。这一页画出它从 tensor API 到 CUDA launch 的全部环节，以及每个环节在源码里的位置，L1–L5 都在这张地图上。
# Luminal 全景

## 核心理念（README）

- **RISC 风格**：一切都化成一小组 primitive ops。README 列了 15 个：Unary `Log2, Exp2, Sin, Sqrt, Recip`；Binary `Add, Mul, Mod, LessThan`；Other `SumReduce, MaxReduce, Iota, Gather, Scatter, Cast`。当前 main 的 [[luminal:src/logical/ir.rs@pub enum LogicalOp|LogicalOp]] 枚举比这更长（还有 `Select`、`IndexMapApply`、各种 scan 等），读的时候以源码为准。
- **Compile everything**：`x + y` 不计算，只记录到 DAG；`execute()` 时才跑。
- **Search**：“The best heuristic is no heuristic”。不用一次性、破坏性的改写规则，而是把所有等价程序放进 e-graph，再**实测**挑最快的。
- **First-class dynamism**：形状可以是符号表达式（`(s, 4096)`）。
- **Validated against PyTorch** 与 ReferenceRuntime。

::: flow Luminal：从 tensor API 到 GPU
cfg cw=240 nw=178 rh=100
n api 0 0 "Tensor API" "cx.tensor / matmul / relu" k=acc
n hl 1 0 "logical graph" "primitive ops + shape" k=warn
n eg 2 0 "e-graph" "egglog rewrites" k=vio
n se 3 0 "search" "genome · 实测 ns" k=teal
n fin 3 1.25 "finalists" "验证 / 内存预算" k=pass
n pl 2 1.25 "执行计划" "LLIR + arena slab" k=pass
n rt 1 1.25 "runtime" "CUDA launch / graph" k=dark
e api -> hl "记录"
e hl -> eg "饱和"
e eg -> se "抽取候选"
e se -> fin "排名"
e fin -> pl
e pl -> rt
cap: L1 画出 logical graph；L2 从零实现 e-graph 和 equality saturation；L3 实现 genome 搜索、rank_insert 与 finalists；L4 移植 arena 的 plan_slab；L5 写 Reference vs CUDA 的对拍测试。
:::

| 环节 | 要搞懂的问题 | 源码入口 |
|---|---|---|
| Tensor API / graph | 节点、shape、dtype、broadcast 是 view | [[luminal:src/graph.rs|src/graph.rs]] · [[luminal:src/frontend/matmul.rs@pub fn matmul|frontend/matmul.rs]] · [[luminal:src/frontend/unary.rs@pub fn relu|frontend/unary.rs]] |
| primitive ops | primitive 集合、high-level op 如何拆 | [[luminal:src/logical/ir.rs@pub enum LogicalOp|logical/ir.rs]] · `src/logical/op/*` |
| e-graph | e-class、rewrite、saturation、extraction；rewrite 的合法性和预算 | [[luminal:src/egglog_core/egglog_preamble.egg|egglog_preamble.egg]] · [[luminal:src/extraction.rs@pub struct Genome|extraction.rs]] · [egglog](https://github.com/egraphs-good/egglog) |
| search / cost | 候选如何生成、测量、排名 | [[luminal:crates/luminal_cuda_lite/src/search.rs@pub fn search_implementations|cuda_lite/search.rs]] · [[luminal:crates/luminal_cuda_lite/src/finalists.rs|finalists.rs]] |
| 计划 / runtime | 选中的图如何变成 buffer、slab、launch | [[luminal:src/buffer/arena.rs@pub fn plan_slab|buffer/arena.rs]] · [[luminal:crates/luminal_cuda_lite/src/runtime.rs|cuda_lite/runtime.rs]] · [[luminal:crates/luminal_cuda_lite/src/cuda_graph.rs|cuda_graph.rs]] |
| kernel 库 | cuBLASLt、FlashInfer 的接入与 dispatch 条件 | [[luminal:crates/luminal_cuda_lite/src/ops/cublaslt/README.md|ops/cublaslt/README.md]] · `ops/cublaslt/egg/*.egg` |
| 正确性 | ReferenceRuntime 的语义、窄浮点模拟 | [[luminal:crates/luminal_reference/src/runtime.rs|reference/runtime.rs]] · [[luminal:crates/luminal_reference/src/narrow_float.rs|narrow_float.rs]] |

## 为什么用 e-graph + 搜索，而不是一串 pass

README 的 “What about XLA?” 一节给出了动机：传统编译器是一大串**破坏性**（单向）的改写规则，规则只能在“确定有收益”时触发，于是越写越特殊、越写越多。e-graph 的改写是**非破坏性**的：`a+b → b+a` 只是在同一个 e-class 里多加一个 e-node，原来的程序还在。所有选择都推迟到最后，由测量决定。

::: callout analogy 一本“同义句词典”而不是一支改稿红笔
传统 pass 像编辑拿红笔改稿：改了就回不去，所以只敢做有把握的修改。e-graph 像一本同义句词典：每句话旁边列出所有等价说法（e-class），词典越写越厚但从不删东西。最后由“朗读计时”（实测搜索）决定每处用哪种说法。代价是词典可能爆炸（L2 会量出 3 项加法饱和后就有 15 个 e-node，6 项有 608 个），所以要有预算。
:::

## 看见它在做什么：环境变量

README 的 Environment flags 一节列了所有开关，L4 的端到端 trace 主要用这几个：

| 变量 | 看什么 |
|---|---|
| `LUMINAL_LOG=1` | 打开所有日志通道（`SEARCH_LOG`、`EGGLOG_LOG`、`ROLLING_LOG`） |
| `LUMINAL_LOG_LLIR=1` | 每找到一个更快的图，打印规范化的 LLIR（可以直接 `diff` 两次运行） |
| `LUMINAL_SEARCH_OP_NAMES=1` | 在 best-so-far 行后面附上候选由哪些 kernel / host op 组成 |
| `LLIR_DUMP_DIR=dir` | 把选中的 finalist 的 LLIR 写成 `.txt` 和 `.dot` |
| `LUMINAL_CUDA_PROFILE_RECAPTURE=1` | 每次 execute 的 prepare / allocate / launch / sync 分解 |
| `LUMINAL_CUBLASLT_AUTOTUNE=1` | prepare 时对 cuBLASLt 算法做 autotune |

::: answer 自测：Luminal 的 matmul 为什么“不会自动 cast”？
[[luminal:src/frontend/matmul.rs@pub fn matmul|matmul 的文档注释]]写得很明确：matmul 就是两个同 dtype 操作数上的 broadcast + reduce，**从不插入 cast**；想要不同的计算精度，必须由用户写 `lhs.cast(D).matmul(rhs.cast(D))`，让 cast 成为图里可见的节点。两个 dtype 不同直接报错。这样做的好处是：精度选择在图里是显式的，搜索和对拍（L5）都不会被隐式类型转换干扰。
:::
