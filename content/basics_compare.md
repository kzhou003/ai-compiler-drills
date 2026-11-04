lede: 两个项目不是重复学习，而是同一个问题的两个层次。这一页把它们并排放：同样一个 GEMM / fusion / layout / benchmark / 正确性问题，Triton 和 Luminal 分别怎么回答。
# 两边对照：互补而不是重复

## 抽象层次

::: flow 同一个 relu(x @ w + b)，两种“程序”
cfg cw=250 nw=215 rh=92
n tk 0 0 "Triton：一个 kernel" "@triton.jit fused_mlp" k=pass
n tprog 0 1.1 "program instance" "算 C 的一个 BM×BN tile" k=pass
n tlay 0 2.2 "layout" "谁持有哪个元素" k=pass
n lg 1.6 0 "Luminal：一张图" "Mul · ReduceSum · Add · Select" k=vio
n lnode 1.6 1.1 "节点 + e-class" "每处有多种实现" k=vio
n lplan 1.6 2.2 "执行计划" "选中的实现 + 内存" k=vio
e tk -> tprog
e tprog -> tlay
e lg -> lnode
e lnode -> lplan
e tlay -> lplan "一个候选" dash
cap: Triton 的基本单位是 program instance（一个 CTA 算一块输出）；Luminal 的基本单位是图上的节点和 e-class。一个 Triton kernel 在 Luminal 看来只是某个 e-class 的一个候选实现。
:::

| 问题 | Triton 视角 | Luminal 视角 |
|---|---|---|
| 如何实现 GEMM？ | tile、layout、MMA lowering、pipeline（T1、T2、T3） | 选哪个 GEMM implementation / library / plan（L3） |
| 如何 fusion？ | 一个 kernel 内的 fusion 与寄存器代价（K4） | 图级 rewrite、fusion candidate 和计划搜索（L2、L3） |
| 如何优化 layout？ | thread / warp / shared memory layout（T2） | tensor layout、buffer / operation arrangement（L4） |
| 如何测性能？ | 单 kernel microbenchmark、PTX、Nsight Compute（K2、T3） | graph latency、search overhead、端到端 benchmark（L3、L4） |
| 如何保证正确性？ | compiler transformation / kernel 数值（T4、T5） | graph rewrite、backend、reference-vs-device（L2、L5） |

## 两种“搜索”

两个项目都在搜索，但搜索的东西不一样：

- **Triton autotune**（T3）：kernel 写好了，只搜 `BLOCK_M/N/K`、`num_warps`、`num_stages` 这几个**数值参数**；空间小（几十到几百个配置），先用 `early_config_prune` 剪掉跑不了的、再用 `perf_model` + `top_k` 只测最有希望的几个。
- **Luminal search**（L3）：搜的是**程序结构**：每个 e-class 选哪个 e-node（genome）。空间是各 e-class 选项数的乘积，可能极大，所以用遗传算法 + 实测排名 + 预算。

::: callout analogy 调一辆车 vs 规划一条路线
Triton autotune 像给一辆已经造好的车调胎压和档位比（参数）；Luminal search 像在地图上规划路线（结构）：走高速还是走小路、在哪里换乘（用 cuBLASLt 还是自己生成的 kernel、哪几步合并）。前者空间小、可以几乎穷举；后者空间组合爆炸，必须有策略和预算。两者都必须**实测**，不能只信估算。
:::

## 最好的共同实验

选同一组 GEMM / fused MLP workload：先用 Triton 写出并 profile 一个 kernel，理解它的硬件成本（K1、T3）；再在 Luminal 观察 graph / search 如何为它选择执行实现（L3、L4）。X1 就是这个实验的完整步骤。

::: answer 自测：为什么 Luminal 的搜索比 Triton autotune 更依赖“预算”？
Triton autotune 的配置空间是几个参数的笛卡尔积，通常几十个，剪枝后全部实测也只要几秒。Luminal 的空间是每个 e-class 选项数的**乘积**，而 e-graph 饱和本身就可能指数增长（L2：n 项加法的交换律 + 结合律饱和后有 3ⁿ − 2ⁿ⁺¹ + 1 + n 个 e-node）。不设迭代 / 节点 / 搜索预算，编译时间会失控。所以 L3 要把“搜索时间”和“最终 latency”放在一起报告，并算 break-even：多花的搜索时间要多少次执行才能赚回来。
:::
