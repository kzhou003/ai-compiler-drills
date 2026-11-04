title: L4 讲解：plan_slab 怎样把一张图塞进一块内存
lede: 用不融合的 relu(x @ w + b) 计划，把参考解（也是 Luminal arena.rs 的移植）一步一步跑出来：传递闭包、precedes 判定、first fit 怎样把 t1 和 y 塞进 x 腾出的空间；画出 slab 的布局；再把一次执行拆成四个阶段，并对照 Luminal 的 trace 环境变量。

::: callout info 怎么读这一页
对照 [[here:solutions/l4_plan_trace.py|l4_plan_trace.py]] 和 [[luminal:src/buffer/arena.rs@pub fn plan_slab|src/buffer/arena.rs 的 plan_slab]]。M=64、K=256、N=64、fp32。
:::

## ① 输入：DAG 和 buffer 的使用点

```text
upload → lt_matmul → add_bias → relu → readback
```

| buffer | 字节 | 对齐后 | uses |
|---|---|---|---|
| x | 65,536 | 65,536 | upload, lt_matmul |
| w | 65,536 | 65,536 | upload, lt_matmul |
| b | 256 | 256 | upload, add_bias |
| workspace | 1,048,576 | 1,048,576 | lt_matmul |
| t0 | 16,384 | 16,384 | lt_matmul, add_bias |
| t1 | 16,384 | 16,384 | add_bias, relu |
| y | 16,384 | 16,384 | relu, readback |

## ② precedes：每一个使用都在前面

[[here:solutions/l4_plan_trace.py@L46-L55|reachability（46–55 行）]] 倒着拓扑序求闭包：`after[lt_matmul] = {add_bias, relu, readback}`。`precedes(a, b)` 要求 a 的**每个**使用都能到达 b 的**每个**使用：

| 对 | 判定 | 能共用？ |
|---|---|---|
| x → t1 | {upload, lt_matmul} 都能到 {add_bias, relu} | ✔ |
| x → t0 | lt_matmul 到不了 lt_matmul 自己 | ✘（同一个节点的输入和输出） |
| t0 → y | {lt_matmul, add_bias} 都能到 {relu, readback} | ✔ |
| b → t1 | add_bias 到不了 add_bias | ✘ |

## ③ first fit：按请求顺序放

[[here:solutions/l4_plan_trace.py@L62-L81|plan_slab（62–81 行）]]，与 Rust 版的循环一一对应：

| 请求 | 与它冲突的已放 buffer | 第一个放得下的位置 |
|---|---|---|
| x | — | 0 |
| w | x | 65,536 |
| b | x, w | 131,072 |
| workspace | x, w, b | 131,328 |
| t0 | x, w, b, workspace | 1,179,904 |
| t1 | b, t0（x、w、workspace 已经结束） | **0**（x 的位置） |
| y | 只有 t1（两者都用到 relu） | **16,384** |

最后一行要小心：y 的使用是 {relu, readback}，t1 是 {add_bias, relu}——它们在 relu 处重叠（relu 读 t1、写 y），所以**冲突**；y 只能避开 t1 的 [0, 16384)，放在 16,384（还在 x 原来的地盘里）。

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 70px 1fr 1fr 1fr 1fr 0.4fr 3fr 0.6fr; min-width: 640px; gap: 2px">
<div class="lbl">offset</div><div class="c0">x 0–64K</div><div class="c0">w 64K–128K</div><div class="cx">b</div><div class="c1" style="grid-column: span 3">workspace 128K–1.15M</div><div class="c2">t0</div>
<div class="lbl">后半段</div><div class="c3">t1 0–16K · y 16–32K</div><div class="cx" style="grid-column: span 6">（x、w、workspace 已结束，空闲）</div>
</div></div>
<p class="legend"><span class="c0"></span>输入<span class="c1"></span>cuBLASLt workspace<span class="c2"></span>t0<span class="c3"></span>复用 x 的字节</p>
:::

slab_bytes = 1,196,288 B（高水位由 t0 的末尾决定）；不复用时 1,229,056 B。workspace 占了大头，所以这个小例子里复用只省了 32 KiB；在真实模型里，几十层的激活值轮流复用同一块内存，差距是几倍。

::: callout analogy 停车场
slab 是一排车位，buffer 是车，uses 是停车时段。两辆车能先后用同一个车位，当且仅当一辆**完全开走**之后另一辆才来。first fit 就是“从最左边开始找第一个空位”。它不保证全场最省，Luminal 的注释说得很诚实：容量错误只意味着“这个算法没找到摆法”，不意味着不存在摆法。
:::

## ④ 为什么是 DAG 可达性，而不是时间线

一个直接的想法是按拓扑序给节点编号，用 `[第一次使用, 最后一次使用]` 区间判断重叠。但这会**发明**一个执行顺序：两个并行分支 b、c 在拓扑序里一前一后，区间不重叠，就会被安排共用内存——一旦运行时让它们并行（不同 stream、CUDA graph 里的并行节点），数据就互相覆盖。arena.rs 的注释特意强调“Transitive closure of the supplied dependency DAG, never an invented topological execution order”。测试 `test_unordered_branches_cannot_alias` 就是这个菱形 DAG。

::: flow 菱形 DAG：b、c 可能并行
cfg cw=180 nw=150
n a 0 0.6 "a" "" k=acc w=0.6
n b 1 0 "b · 用 p" "" k=warn
n c 1 1.2 "c · 用 q" "" k=warn
n d 2 0.6 "d · 用 r" "" k=pass
e a -> b
e a -> c
e b -> d
e c -> d
cap: p 和 q 互不先后 → 不同地址；r 在 b、c 之后 → 可以复用 p 的位置（offset 0）。
:::

## ⑤ 四个阶段

[[here:solutions/l4_plan_trace.py@L141-L156|phase_breakdown / achieved（141–156 行）]]：测试的例子里 graph_compile 0.4 s、search 3.0 s、prepare 0.1 s、两次 execute 各 0.5 ms——search 占 85.7%。这就是为什么 L3 要算 break-even，也是为什么 README 把 `LUMINAL_CUDA_PROFILE_RECAPTURE` 描述为“decomposing TTFT/TPOT into kernel time vs. runtime overhead”的工具。

`achieved(2·64·64·256, 10 µs, 40 TFLOP/s)` = 0.21 TFLOP/s，峰值的 0.5%：小 GEMM 是 launch-bound，和 K1 的 roofline 一致。

## ⑥ 源码与资料索引

- [[luminal:src/buffer/arena.rs@pub struct SlabBuffer|SlabBuffer]] · [[luminal:src/buffer/arena.rs@pub enum SlabAlgorithm|SlabAlgorithm::FirstFit]] · [[luminal:src/buffer/arena.rs@fn aligned_size|aligned_size]]
- [[luminal:crates/luminal_cuda_lite/src/arena.rs|cuda_lite/arena.rs：buffer_bytes]] · [[luminal:crates/luminal_cuda_lite/src/finalists.rs@pub arena|finalists.rs：arena.slab_bytes]]
- [[luminal:crates/luminal_cuda_lite/src/cuda_graph.rs|cuda_graph.rs]] · [[luminal:crates/luminal_cuda_lite/tests/arena_budget.rs|tests/arena_budget.rs]]
- README：`LUMINAL_LOG_LLIR`、`LLIR_DUMP_DIR`、`LUMINAL_CUDA_PROFILE_RECAPTURE`、`LUMINAL_CUDA_MEMORY_DEBUG`

## ⑦ 自测

::: answer 把请求顺序改成先放 t1、y，再放 x，slab_bytes 会变吗？
在这个例子里不变：t1、y 先放在 0 和 16,384，之后 x 和它们“严格先后”，仍然可以放在 0，最终 slab 还是 1,196,288 B（自己跑一下就能确认）。但一般会变：first fit 是按请求顺序的贪心，先放的占据低地址，顺序不同，后来者能找到的空隙就不同。Luminal 的文档说“The result follows request order”，所以请求顺序本身是计划的一部分。
:::

::: answer 融合计划（一个 cuBLASLt RELU_BIAS 调用）的 slab 为什么和不融合的一样大？
两者的高水位都由 `x + w + b + workspace + 一个 [M, N] 输出` 决定：不融合时 t1、y 复用了 x 的地盘，没有抬高高水位。融合省下的是两次 launch 和 t0/t1 的 DRAM 往返，而不是峰值内存。如果 workspace 很小、激活很大，结论会不同——用 `plan_slab` 换参数试试。
:::
