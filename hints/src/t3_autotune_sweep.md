title: T3 讲解：一个 autotuner 的四道工序
lede: 跟着 4096³ fp16 GEMM 走一遍参考解的 autotune：192 个配置怎样被剪到 146 个、perf model 为什么把 8 warp 的大 tile 排在前面、wave 量化让多一个 tile 的问题慢了 41%、以及 pick_best 为什么没有选 median 最小的那个配置。最后对照 triton.autotune 的源码。

::: callout info 怎么读这一页
对照 [[here:solutions/t3_autotune_sweep.py|t3_autotune_sweep.py]]。数字来自 `autotune(RTX3070, 4096, 4096, 4096, fake_measure(..., seed=0))`；`fake_measure` = perf model × 一个模型不知道的效率因子 × 3% 噪声。
:::

## ① 四道工序

::: flow 192 → 146 → 8 → 1
cfg cw=225 nw=185
n sp 0 0.6 "config_space" "4·4·2·2·3 = 192" k=acc
n pr 1 0 "early_prune" "−34 smem · −12 regs" k=warn
n pm 1 1.2 "perf_model · top_k" "按预测时间取 8" k=vio
n sw 2 0.6 "sweep" "每个 25 次 · 分位数" k=teal
n pk 3 0.6 "pick_best" "噪声内选省 smem 的" k=pass
e sp -> pr
e pr -> pm
e pm -> sw
e sw -> pk
:::

## ② early_prune：不能跑的先去掉

[[here:solutions/t3_autotune_sweep.py@L26-L36|early_prune（26–36 行）]] 复用 K1 的 `occupancy`：46 个配置被剪掉，其中 34 个是 shared memory 超过每 CTA 99 KB（例如 256×256×64 任意 stage），12 个是估计寄存器超过 255（例如 256×256、4 warp：65536/128 + 40 = 552）。另一条规则针对形状：M = 100 时 256 行的 tile 有 60% 是 padding，直接去掉。

在 triton.autotune 里这就是 `prune_configs_by={"early_config_prune": fn}`（[[triton:python/triton/runtime/autotuner.py@def prune_configs|Autotuner.prune_configs]]）：函数签名是 `(configs, named_args, **kwargs)`，返回保留的配置。

## ③ perf_model：wave 量化

[[here:solutions/t3_autotune_sweep.py@L39-L50|perf_model（39–50 行）]] 的计算时间按“整轮”算：

| 问题 | tile 数 | 每轮（2 CTA/SM × 46） | 轮数 | 预测 |
|---|---|---|---|---|
| M = 128·92，N = 128，K = 4096 | 92 | 92 | 1 | 0.442 ms |
| M = 128·93 | 93 | 92 | 2 | 0.622 ms（访存时间成为下界） |

多一个 tile（+1.1% 的工作）预测慢了 41%。真实 GPU 上第二轮那一个 CTA 独占整个 GPU，其余 91 个位置空转——这就是为什么大厂的 GEMM 库会用 stream-K / split-K 这类方法把最后一轮的工作摊开。

::: callout analogy 电梯
每轮 92 个 CTA 像一部能载 92 人的电梯。92 个人一趟送完；93 个人就得跑两趟，第二趟只载一个人，时间却和满载一样长。
:::

## ④ sweep 和 pick_best：最快的不一定被选中

top 8 的实测（单位 ms，fake_measure 的结果）：

| BM×BN×BK | warps | stages | median | [p20, p80] | smem |
|---|---|---|---|---|---|
| 128×256×64 | 8 | 2 | **3.760** | [3.686, 3.880] | 98,304 |
| 256×128×32 | 8 | 3 | 3.761 | [3.683, 3.795] | 73,728 |
| 256×128×32 | 8 | 2 | 3.877 | [3.765, 3.980] | **49,152** |
| 128×256×32 | 8 | 2 | 3.894 | [3.838, 4.061] | 49,152 |
| … | | | | | |

[[here:solutions/t3_autotune_sweep.py@L81-L86|pick_best（81–86 行）]]：median 最小的是第一行；与它的 [p20, p80] 重叠的有前 7 行，其中 smem 最少（49,152）且 median 更小的是 **256×128×32、8 warp、2 stage**。它比“最快”的慢 3%，但在噪声范围内，而且只用一半的 shared memory——换到别的形状或别的 GPU 上更不容易出问题。triton.autotune 本身只取最小 median（`min(timings, key=timings.get)`）；这条“噪声内选省资源”的规则是本题加的，PR 讨论里经常用到。

## ⑤ 和 triton.autotune 逐项对照

| 本题 | triton.autotune | 位置 |
|---|---|---|
| `early_prune` | `prune_configs_by["early_config_prune"]` | [[triton:python/triton/runtime/autotuner.py@prune_configs_by|autotuner.py]] |
| `perf_model` + `top_k` | `prune_configs_by["perf_model"]`、`["top_k"]` | 同上 |
| `quantiles((0.5, 0.2, 0.8))` | `do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))` | [[triton:python/triton/runtime/autotuner.py@L184|L184]] |
| `bench` 的 warmup / repeats | `do_bench(warmup=25, rep=100)`（毫秒，不是次数） | [[triton:python/triton/testing.py@def do_bench|testing.py]] |
| 结果缓存 | 按 `key` 缓存；`cache_results=True` 时写磁盘 | autotuner.py |

## ⑥ bound：模型的盲区

`bound(G, 4096, 4096, 4096, 128×128×32)` 返回 `"memory"`——因为 `tiled_bytes` 不计 L2。读 CSV 时如果 ncu 的 DRAM 字节远小于 `tiled_bytes`，说明 L2 在帮你；如果接近，说明 grouped ordering 没起作用（`GROUP_SIZE_M = 1`）。这是 T3 性能分析里最值得写的一段。

## ⑦ 源码与资料索引

- [[triton:python/triton/runtime/autotuner.py@class Autotuner|class Autotuner]] · [[triton:python/triton/runtime/autotuner.py@def prune_configs|prune_configs]]
- [[triton:python/tutorials/03-matrix-multiplication.py@triton.Config|tutorial 03 的配置列表]]
- [[triton:python/triton/testing.py@def do_bench|do_bench]] · [Nsight Compute](https://docs.nvidia.com/nsight-compute/)

## ⑧ 自测

::: answer top_k = 8 时 perf model 排前 5 的配置预测时间完全相同（3.709 ms），为什么？
它们都是 8 warp、每 SM 1 个 CTA 的 128×256 或 256×128 tile：tile 数、每轮 CTA 数、轮数都一样，访存也相同，`max(计算, 访存)` 当然相同。模型分不清 BK 和 num_stages 的差别——正是实测要回答的问题。所以 top_k 不能小于“模型无法区分”的那一组的大小。
:::

::: answer 什么时候应该关掉 autotune，直接写死配置？
key 的取值很多（比如动态 batch）、每次重新调优的成本（几百毫秒到几秒）不可接受时；或者需要确定性（每次运行同一配置，便于复现数值）时。常见做法是离线扫好一张“形状 → 配置”表，运行时查表——Luminal 把这件事前移到编译期搜索。
:::
