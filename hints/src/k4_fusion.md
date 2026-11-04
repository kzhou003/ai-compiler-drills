title: K4 讲解：fusion 的三笔账——字节、launch、寄存器
lede: 用 GEMM + bias + GELU + residual 这条最常见的 epilogue 链，算出四种切法的 DRAM 流量和 launch 次数；再构造一个寄存器压力过大的例子，看“全部融合”怎样输给“拆成两个 kernel”。最后把同一个问题放到 Triton 的 epilogue 和 Luminal 的 cuBLASLt epilogue 上对照。

::: callout info 怎么读这一页
对照 [[here:solutions/k4_fusion.py|k4_fusion.py]]。`Stage` 描述一个算子：写多少元素、从链外读多少元素（权重、bias、residual）、多少 FLOPs、融合时占多少寄存器。
:::

## ① 四种切法，一张表

`chain_mlp_epilogue(4096, 4096, 4096)`，fp16，MN = 4096² = 16.8 M 元素（[[here:solutions/k4_fusion.py@L41-L50|traffic（41–50 行）]]）：

| 切法 | 读写明细（单位 MN，× 2 B） | 字节 | launch | 寄存器峰值 |
|---|---|---|---|---|
| `[1, 1, 1, 1]` | mm: A+B 读 2，写 1 · bias: 读 1+b，写 1 · gelu: 读 1 写 1 · res: 读 2 写 1 | 335,552,512 | 4 | 128 |
| `[1, 3]` | mm 同上 · 一个 epilogue kernel: 读 C + b + R，写 Y | 201,334,784 | 2 | 128 |
| `[2, 2]` | mm+bias · gelu+res | 201,334,784 | 2 | 136 |
| `[4]` | 读 A、B、b、R，写 Y | 134,225,920 | 1 | 160 |

全部融合省掉了 C、C′、C″ 三次写回和三次读回：流量从 335 MB 降到 134 MB，launch 从 4 次降到 1 次。

::: flow 中间结果去了哪里
cfg cw=210 nw=180
n acc 0 0.6 "累加器（寄存器）" "fp32 · BM×BN" k=vio
n un 1 0 "不融合：写 C" "→ DRAM → 下个 kernel 读" k=warn
n fu 1 1.2 "融合：原地继续" "+b → gelu → +R" k=pass
n y 2 0.6 "写 Y 一次" "" k=teal
e acc -> un
e acc -> fu
e un -> y dash
e fu -> y
:::

::: callout analogy 流水线上的半成品
不融合像一条每道工序之间都要入库的流水线：切好的零件放回仓库，下一道工序再出库。融合是把几道工序放在同一个工位上连续做完，只入库一次。但工位的台面（寄存器）有限：工序越多，台面上同时摊开的半成品越多，放不下就得临时堆到地上（spill），来回弯腰反而更慢。
:::

## ② 时间估计：roofline + launch + spill 惩罚

[[here:solutions/k4_fusion.py@L53-L66|time_estimate（53–66 行）]] 每段算 `max(访存, 计算) + launch`；寄存器超过 255 时访存时间乘 `spill_penalty`。[[here:solutions/k4_fusion.py@L69-L82|best_plan（69–82 行）]] 用一个 n−1 位掩码枚举全部 2ⁿ⁻¹ 种切法（4 个 stage 是 8 种）。

4096³ 的例子里最优是 `[4]`（约 3.45 ms，几乎全是 GEMM 的计算时间）；16×4096×4096 的 decode 形状最优也是 `[4]`，但原因不同：那里省下的主要是 launch。

## ③ 融合会输的例子

```text
heavy = [Stage("a", n, n, n, live_regs=200), Stage("b", n, 0, n, live_regs=100)]    n = 2^26
```

| 切法 | 访存 | 寄存器 | 估计（spill_penalty = 3） |
|---|---|---|---|
| `[1, 1]` | a 读 n 写 n，b 读 n 写 n → 4n·2 B | 200 / 100 | ≈ 1.21 ms |
| `[2]` | 读 n 写 n → 2n·2 B，但 300 > 255 | 300 → spill | 2n·2 B × 3 ≈ 1.81 ms |

流量减半，却因为 spill 慢了 50%。真实的 spill 代价取决于溢出多少、访问模式，`ptxas -v` 或 `CompiledKernel.n_spills` 会告诉你有没有 spill；这里的 penalty 只是一个参数化的提醒。

## ④ 两个项目里的同一件事

| | Triton | Luminal |
|---|---|---|
| 谁决定融合 | 你：把 epilogue 写进 kernel（[[triton:python/tutorials/03-matrix-multiplication.py@leaky_relu|tutorial 03 的 ACTIVATION]]） | 搜索：融合与不融合都在 e-graph 里，实测决定 |
| GEMM + bias + relu | 一个 kernel | cuBLASLt `CUBLASLT_EPILOGUE_RELU_BIAS`（[[luminal:crates/luminal_cuda_lite/src/ops/cublaslt/device_call.rs@CUBLASLT_EPILOGUE_RELU_BIAS|device_call.rs]]），或者拆开 |
| 代价从哪看 | `n_regs`、`n_spills`、ncu 的 DRAM bytes | `SEARCH_LOG` 的 best-so-far、`LUMINAL_SEARCH_OP_NAMES` 的 kernel 组成 |

## ⑤ 源码与资料索引

- [[luminal:crates/luminal_cuda_lite/src/ops/cublaslt/README.md|Luminal cuBLASLt README]] · `ops/cublaslt/egg/approved_rules.egg`（哪些图模式可以交给 cuBLASLt）
- [cuBLASLt epilogue 枚举](https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t)
- [[triton:python/tutorials/03-matrix-multiplication.py|Triton tutorial 03]]

## ⑥ 自测

::: answer `[1, 3]` 和 `[2, 2]` 的字节数为什么一样？
两种切法都恰好有一个中间张量（MN 个元素）要写回再读回：`[1, 3]` 是 C，`[2, 2]` 是 C″。外部输入（A、B、b、R）不管怎么切都要读一次，最终的 Y 都要写一次。区别在寄存器峰值（128 vs 136）和每个 kernel 的复杂度。
:::

::: answer 为什么 decode 形状（M = 16）下融合的收益主要来自 launch？
M = 16 时 GEMM 的时间由读 33.5 MB 的权重决定（约 75 µs），后面三个 epilogue kernel 每个只搬约 128–256 KB，时间几乎全是 5 µs 的 launch。模型算出不融合 97.5 µs、全融合 80.8 µs：省下的 16.7 µs 里 15 µs 是三次 launch，流量的节省反而是次要的。这也是 Luminal 用 CUDA graph 的原因之一。
:::
