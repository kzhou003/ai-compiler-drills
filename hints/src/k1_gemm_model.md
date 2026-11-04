title: K1 讲解：GEMM 的资源账，一个配置一个配置地算
lede: 用 4096³ 的 fp16 GEMM 和 RTX 3070 做例子，把参考解的每个函数都算出具体数字：最少字节和实际字节差多少倍、98,304 B 的 shared memory 为什么把 occupancy 卡在 1、23 轮 wave 的最后一轮只有 26% 满，以及不计 L2 的模型为什么会得出“memory-bound”这个错误结论。

::: callout info 怎么读这一页
对照 [[here:solutions/k1_gemm_model.py|k1_gemm_model.py]] 读。本页所有数字都用参考解算过：`from solutions.k1_gemm_model import *; analyze(RTX3070, 4096, 4096, 4096, TileConfig(128, 128, 64, 4, 3))`。
:::

## ① 全貌：一个配置到一个预测

::: flow analyze() 里的数据流
cfg cw=225 nw=190
n in 0 0.6 "(M, N, K) + TileConfig" "4096³ · 128×128×64 · 4w · 3s" k=acc w=1.05
n fl 1 0 "gemm_flops" "1.37e11" k=warn
n by 1 1.2 "tiled_bytes" "2.18e9 B" k=warn
n oc 2 0.6 "occupancy" "1 CTA/SM · smem" k=vio
n wv 3 0 "waves" "23 轮 · 尾 26%" k=pass
n rf 3 1.2 "roofline" "memory · 4.9 ms" k=pass
e in -> fl
e in -> by
e in -> oc
e oc -> wv
e fl -> rf
e by -> rf
:::

## ② FLOPs 和字节：最少 vs 实际

[[here:solutions/k1_gemm_model.py@L50-L71|gemm_flops / gemm_min_bytes / tiled_bytes（50–71 行）]]：

| 量 | 4096³ fp16 | 怎么来的 |
|---|---|---|
| FLOPs | 137,438,953,472 | 2 · 4096³ |
| 最少字节 | 100,663,296（96 MiB） | 3 个 4096² 矩阵 × 2 B |
| 实际字节（128×128 tile，不计 L2） | 2,181,038,080（约 2.0 GiB） | A 读 ⌈4096/128⌉ = 32 遍，B 也读 32 遍，C 写 1 遍 |
| 比值 | 21.7× | ≈ (32 + 32 + 1) / 3 |

算术强度：按最少字节是 1365 FLOP/B，按实际字节只有 63 FLOP/B——低于 RTX 3070 的平衡点（40e12 / 448e9 ≈ 89）。tile 越大，A、B 重读越少：256×256 的 tile 只读 16 遍。这就是“大 tile = 高数据复用”的定量含义。

::: callout analogy 复印讲义
A 的每一行是一页讲义，每个“列 tile”是一个班。没有 L2 时，每个班都要自己去档案室复印一份完整讲义：32 个班印 32 份。L2 像走廊上的共享复印机缓存：如果 32 个班差不多同一时间来印同一页，只有第一个班真去档案室。grouped ordering（tutorial 03 的 `GROUP_SIZE_M`）就是安排各班的上课时间，让它们尽量同时要同一页。
:::

## ③ occupancy：四个上限，最小的那个说了算

[[here:solutions/k1_gemm_model.py@L90-L112|occupancy（90–112 行）]] 对 128×128×64、4 warp、3 stage：

| 上限 | 公式 | 数值 |
|---|---|---|
| threads | 1536 // 128 | 12 |
| regs | 65536 // (168 · 128) | 3（每线程 128·128/128 + 40 = 168 个寄存器） |
| smem | 102400 // 98304 | **1** |
| blocks | 16 | 16 |

limiter 是 smem，每 SM 1 个 CTA。把 `num_stages` 降到 2（65,536 B），smem 上限变成 1 但 102400 // 65536 仍是 1；降到 BK=32、3 stage（49,152 B）才变成 2。另外两个“直接不合法”的情况：smem 超过单 CTA 上限 99 KB（128×256×64、3 stage 要 147,456 B），或者估计寄存器超过 255（256×256 tile、4 warp：65536/128 + 40 = 552）。

::: flow occupancy 的判断顺序
cfg cw=230 nw=190
n s 0 0 "smem > 每 CTA 上限？" "→ invalid: smem" k=vio
n r 0 1.2 "寄存器 > 255？" "→ invalid: regs（spill）" k=vio
n m 1 0.6 "四个上限取最小" "按名字打破并列" k=warn
n o 2 0.6 "blocks · limiter" "" k=pass
e s -> r dash
e r -> m
e m -> o
:::

## ④ wave：最后一轮的尾巴

[[here:solutions/k1_gemm_model.py@L115-L121|waves（115–121 行）]]：1024 个 tile，每轮 46 个（46 SM × 1），需要 ⌈1024/46⌉ = 23 轮；前 22 轮跑掉 1012 个，最后一轮只有 12 个，填充率 12/46 ≈ 26%。整体效率 1024 / (23 · 46) = 96.8%。如果每 SM 能放 2 个 CTA（每轮 92 个），需要 12 轮，最后一轮 12/92 = 13%，整体 92.8%——occupancy 翻倍，wave 效率反而下降。T3 的 perf model 就是用“最后一轮按满算”来体现这一点的。

## ⑤ roofline：三种 bound

[[here:solutions/k1_gemm_model.py@L124-L133|roofline（124–133 行）]]：

| 例子 | t_compute | t_memory | bound |
|---|---|---|---|
| 4096³，按最少字节 | 3.44 ms | 0.22 ms | compute |
| 4096³，128×128 tile，不计 L2 | 3.44 ms | 4.87 ms | memory（模型的错觉） |
| 2²⁴ 元素向量加法（6 B/FLOP） | 0.4 µs | 225 µs | memory |
| 64³ GEMM | 0.013 µs | 0.05 µs | launch（< 5 µs） |

第二行是这道题最想让你记住的：**模型的结论取决于你放进去的假设**。真实 GPU 上 A/B 的重复读取大多在 L2 命中，到 DRAM 的接近最少字节，所以大 GEMM 是 compute-bound。用 Nsight Compute 的 “DRAM Bytes” 和 “L2 Hit Rate” 就能验证。

## ⑥ 源码与资料索引

- [[triton:python/tutorials/03-matrix-multiplication.py@GROUP_SIZE_M|tutorial 03：GROUP_SIZE_M 与 L2 复用]]
- [[triton:python/triton/runtime/autotuner.py@early_config_prune|autotuner：early_config_prune / perf_model]]
- [[triton:python/triton/compiler/compiler.py@n_regs|CompiledKernel.n_regs / n_spills]]
- [CUDA Programming Guide：occupancy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multiprocessor-level) · [Nsight Compute](https://docs.nvidia.com/nsight-compute/)

## ⑦ 自测

::: answer 把 4096³ 的 tile 换成 256×128×32、8 warp、3 stage，limiter 是什么？
smem = 3·(256·32 + 32·128)·2 = 73,728 B → 102400 // 73728 = 1；寄存器 256·128/256 + 40 = 168，65536 // (168·256) = 1；threads 1536 // 256 = 6。smem 和 regs 并列为 1，按名字排序 limiter 是 `regs`。这正是 T3 里 perf model 排第一的那类配置。
:::

::: answer 为什么 occupancy 只有 1 的配置常常最快？
大 tile 让 A、B 的重读次数减半，换来更高的算术强度；Tensor Core 指令本身延迟可以靠 warp 内的指令级并行和 `num_stages` 的软件流水来隐藏，不一定需要很多驻留 warp。occupancy 是手段，不是目标。
:::
