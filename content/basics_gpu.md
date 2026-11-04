lede: 两个编译器最终都要回答同一组硬件问题：活怎么分给 CTA / warp / lane，数据从哪层内存来，一个 SM 上能同时放几个 CTA。这一页是后面所有题的公共词汇表。
# GPU 执行模型与内存层次（速查）

## 执行层次

| 层 | 是什么 | Triton 里对应 | 典型数字（RTX 3070, SM86） |
|---|---|---|---|
| grid | 一次 launch 的全部 CTA | `kernel[grid](...)` | 任意 |
| CTA（block） | 跑在一个 SM 上的一组线程，共享 shared memory | 一个 **program instance**（`tl.program_id`） | `num_warps × 32` 个线程 |
| warp | 32 个 lane 同步执行同一条指令 | 由 layout 的 `warpsPerCTA` 安排 | 32 lanes |
| lane / thread | 拥有自己的寄存器 | 由 `threadsPerWarp`、`sizePerThread` 安排 | 最多 255 个 32-bit 寄存器 |
| SM | 执行 CTA 的硬件单元 | — | 46 个 SM；每 SM 64K 寄存器、最多 100 KB shared memory、1536 线程 |

::: flow 一个 GEMM CTA 的数据分工（BM × BN 输出 tile）
cfg cw=265 nw=190
n c 0 0.6 "C 的一个 tile" "BM × BN，归这个 CTA" k=acc
n k 1 0 "K 循环" "每步搬 A[BM×BK]、B[BK×BN]" k=warn
n s 1 1.2 "shared memory" "num_stages 份缓冲" k=teal
n w 2 0 "4 个 warp" "各管 C tile 的一块" k=vio
n r 2 1.2 "寄存器" "fp32 累加器 BM·BN/线程数" k=pass
n tc 3 0.6 "Tensor Core" "mma.sync m16n8k16" k=dark w=0.85
e c -> k
e k -> s "cp.async"
e s -> w "ldmatrix"
e w -> r
e r -> tc anim
cap: K1 会把每一项都算成数字：这个 CTA 读多少字节、用多少 shared memory、每线程多少寄存器、一个 SM 能放几个这样的 CTA、整个 grid 要几轮（wave）。
:::

## 内存层次和三个常见瓶颈

| 层 | 容量 | 访问方式 | 要注意 |
|---|---|---|---|
| 寄存器 | 每 SM 256 KB | 线程私有 | 用超了就 **spill** 到 local memory（其实在 DRAM） |
| shared memory | 每 SM 最多 100 KB | CTA 内共享，32 个 bank | **bank conflict**：同一 warp 的多个 lane 访问同一 bank 的不同地址 |
| L2 | 4 MB | 全 GPU 共享 | GEMM 的 A/B 重复读取大多在这里命中 |
| DRAM | 8 GB，448 GB/s | 以 32 B sector 为单位 | **coalescing**：一个 warp 的访问落在越少的 sector 越好 |

::: callout analogy 仓库、货架、工作台
DRAM 是城外的大仓库（大、远、按整箱 32 B 发货），L2 是城里的中转站，shared memory 是车间的公共货架（一个班组共用，取货口有 32 个窗口，同一个窗口排队就是 bank conflict），寄存器是每个工人手边的工作台（最快，但台面很小，放不下就只能堆回仓库——spill）。**coalescing** 就是让一个班组 32 个人要的货尽量装在同几箱里一起运。
:::

## 三种 bound 和 roofline

一个 kernel 的时间下界是 `max(FLOPs / 峰值算力, 字节数 / 峰值带宽)`，再加上固定的 launch 开销（几微秒）。

- **compute-bound**：算术强度（FLOP/Byte）高于机器平衡点（RTX 3070 FP16 Tensor Core 约 40 TFLOP/s ÷ 448 GB/s ≈ 90 FLOP/B）。大 GEMM 属于这类。
- **memory-bound**：elementwise、reduction、decode attention。优化方向是少搬数据（fusion）、搬得整齐（coalescing、向量化）。
- **launch-bound**：问题太小，算和搬都只要几微秒，launch 本身占大头。优化方向是 fusion、CUDA graph（Luminal 的 `cuda_graph.rs`）。

::: answer 自测：一个 [4096] fp32 向量加法是哪种 bound？
读 2 × 16 KB、写 16 KB，共 48 KB；448 GB/s 下约 0.1 µs，远小于约 5 µs 的 launch 开销。所以是 **launch-bound**，测出来的时间几乎全是 launch。要看到 memory-bound 行为，向量至少要几百万元素（K2 的 benchmark 题会让你测出这个拐点）。
:::

::: answer 自测：为什么 occupancy 不是越高越好？
occupancy 只说明一个 SM 上同时驻留多少 warp，用来隐藏访存延迟。GEMM 常常故意用大 tile（每线程更多寄存器、更多 shared memory）换取更高的数据复用，occupancy 降到 1 个 CTA/SM 反而更快。K1 和 T3 会让你看到：最快的配置往往 occupancy 不是最高的。
:::
