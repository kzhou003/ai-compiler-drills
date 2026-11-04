title: T2 讲解：复刻 coalesce pass，再看懂一次 layout 转换
lede: 先逐行手算 Triton 的 getNumElementsPerThread 和 BlockedEncodingAttr builder，复现 coalesce.mlir 期望的两个 layout；再用一个 16×16 的例子画出完整的“谁持有哪个元素”图，数合并访存的 sector；最后用同一张图解释 convert_layout 的三档代价：寄存器重命名、warp shuffle、shared memory。

::: callout info 怎么读这一页
对照 [[here:solutions/t2_coalesce_layout.py|t2_coalesce_layout.py]]。本页所有 layout 和数字都由参考解算出：`coalesced_layout([16, 16], [1, 0], 32, 16, 16, 2)`。
:::

## ① 输入：编译器知道指针的什么

coalesce pass 不看你的 Python 代码，只看 **AxisInfo 分析**给每个指针张量算出的三样东西：

| 属性 | 含义 | 例子（`x_ptr + row*stride + tl.arange(0, 64)`） |
|---|---|---|
| contiguity | 沿某一维连续多少个地址 | 列方向 64 |
| divisibility | 起始地址整除多少字节 | `tt.divisibility = 16` → 16 |
| order | 按 contiguity 从大到小排的维度 | `[1, 0]`（列最快） |

`tt.divisibility = 16` 就是 kernel 参数上的注解：Triton 在 JIT 特化时检查指针和整数参数是否 16 的倍数，是的话就带上这个属性（这也是为什么同一个 kernel 换一个没对齐的指针会触发重新编译）。

## ② 第一步：perThread

[[here:solutions/t2_coalesce_layout.py@L72-L79|elements_per_thread（72–79 行）]] = 真实的 [[triton:lib/Dialect/TritonGPU/Transforms/Utility.cpp@unsigned getNumElementsPerThread|getNumElementsPerThread]] + [[triton:lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp@perThread = std::min<int>|CoalesceUtils 里的均分上限]]：

| 项 | 64×64 fp32，4 warp | 128×64 fp16，4 warp | 64 元素，4 warp |
|---|---|---|---|
| divisibility / elem_bytes | 16 / 4 = 4 | 16 / 2 = 8 | 4 |
| min(contiguity, shape) | 64 | 64 | 64 |
| 128 / elem_bits | 4 | 8 | 4 |
| numElems / numThreads | 4096 / 128 = 32 | 8192 / 128 = 64 | 64 / 128 → 1 |
| **perThread** | **4** | **8** | **1** |

## ③ 第二步：分 lane 和 warp

[[here:solutions/t2_coalesce_layout.py@L51-L69|build_blocked（51–69 行）]] 逐行对应 [[triton:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td@unsigned remainingLanes|TritonGPUAttrDefs.td 的 builder]]。对 coalesce.mlir 的 64×64 fp32（spt = [1, 4]，order = [1, 0]）：

| 步骤 | dim | threads_cta | tpw | wpc | 剩余 lanes / warps |
|---|---|---|---|---|---|
| 循环（最快维） | 1 | clamp(128, 1, 64/4 = 16) = 16 | clamp(16, 1, 32) = 16 | clamp(16/16, 1, 4) = 1 | 2 / 4 |
| 收尾（最慢维） | 0 | — | 32 / 16 = 2 | 4 / 1 = 4 | — |

结果 `sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1]`，和 [[triton:test/TritonGPU/coalesce.mlir|coalesce.mlir]] 第一条 CHECK 一字不差。直觉：先把最快维“喂饱”（64 列 / 每线程 4 列 = 16 个 lane），剩下的 lane 和所有 warp 都堆到慢维上。

::: flow builder 的分配顺序
cfg cw=225 nw=195
n s 0 0.6 "spt [1, 4]" "每线程 4 列" k=pass
n l 1 0 "dim 1：16 个 lane" "64 / 4 = 16，正好喂饱" k=teal
n w 1 1.2 "dim 1：1 个 warp" "16 / 16 = 1" k=vio
n r 2 0.6 "dim 0：剩下的" "2 个 lane · 4 个 warp" k=warn
n o 3 0.6 "tile = [8, 64]" "1·2·4 × 4·16·1" k=acc
e s -> l
e l -> w dash
e w -> r
e l -> r
e r -> o
:::

::: callout analogy 排座位
一排座位（最快维）有 64 个，每人占 4 个（spt），所以一排坐 16 人；一个班 32 人，剩下 16 人坐第二排（tpw = [2, 16]）。4 个班（warp）没法再横着放，就一个班接一个班往后排（wpc = [4, 1]）。整个阵型 8 排 × 64 座就是一个 CTA tile，张量更大时就把阵型往下复制。
:::

## ④ 一张完整的归属图（16×16 fp32，2 个 warp）

`coalesced_layout([16, 16], [1, 0], 32, 16, 16, num_warps=2)` 得到 `[1, 4] / [8, 4] / [2, 1]`，tile 正好 16×16。每格写持有它的 lane，颜色是 warp；紫色是 warp 1 的 lane 5，它的 4 个寄存器是 `(9, 4) (9, 5) (9, 6) (9, 7)`：

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 30px repeat(16, minmax(30px, 1fr)); min-width: 574px; gap: 2px">
<div class="lbl"></div><div class="lbl" style="text-align:center">0</div><div class="lbl" style="text-align:center">1</div><div class="lbl" style="text-align:center">2</div><div class="lbl" style="text-align:center">3</div><div class="lbl" style="text-align:center">4</div><div class="lbl" style="text-align:center">5</div><div class="lbl" style="text-align:center">6</div><div class="lbl" style="text-align:center">7</div><div class="lbl" style="text-align:center">8</div><div class="lbl" style="text-align:center">9</div><div class="lbl" style="text-align:center">10</div><div class="lbl" style="text-align:center">11</div><div class="lbl" style="text-align:center">12</div><div class="lbl" style="text-align:center">13</div><div class="lbl" style="text-align:center">14</div><div class="lbl" style="text-align:center">15</div>
<div class="lbl">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">0</div><div class="c0">1</div><div class="c0">1</div><div class="c0">1</div><div class="c0">1</div><div class="c0">2</div><div class="c0">2</div><div class="c0">2</div><div class="c0">2</div><div class="c0">3</div><div class="c0">3</div><div class="c0">3</div><div class="c0">3</div>
<div class="lbl">1</div><div class="c0">4</div><div class="c0">4</div><div class="c0">4</div><div class="c0">4</div><div class="c0">5</div><div class="c0">5</div><div class="c0">5</div><div class="c0">5</div><div class="c0">6</div><div class="c0">6</div><div class="c0">6</div><div class="c0">6</div><div class="c0">7</div><div class="c0">7</div><div class="c0">7</div><div class="c0">7</div>
<div class="lbl">2</div><div class="c0">8</div><div class="c0">8</div><div class="c0">8</div><div class="c0">8</div><div class="c0">9</div><div class="c0">9</div><div class="c0">9</div><div class="c0">9</div><div class="c0">10</div><div class="c0">10</div><div class="c0">10</div><div class="c0">10</div><div class="c0">11</div><div class="c0">11</div><div class="c0">11</div><div class="c0">11</div>
<div class="lbl">3</div><div class="c0">12</div><div class="c0">12</div><div class="c0">12</div><div class="c0">12</div><div class="c0">13</div><div class="c0">13</div><div class="c0">13</div><div class="c0">13</div><div class="c0">14</div><div class="c0">14</div><div class="c0">14</div><div class="c0">14</div><div class="c0">15</div><div class="c0">15</div><div class="c0">15</div><div class="c0">15</div>
<div class="lbl">4</div><div class="c0">16</div><div class="c0">16</div><div class="c0">16</div><div class="c0">16</div><div class="c0">17</div><div class="c0">17</div><div class="c0">17</div><div class="c0">17</div><div class="c0">18</div><div class="c0">18</div><div class="c0">18</div><div class="c0">18</div><div class="c0">19</div><div class="c0">19</div><div class="c0">19</div><div class="c0">19</div>
<div class="lbl">5</div><div class="c0">20</div><div class="c0">20</div><div class="c0">20</div><div class="c0">20</div><div class="c0">21</div><div class="c0">21</div><div class="c0">21</div><div class="c0">21</div><div class="c0">22</div><div class="c0">22</div><div class="c0">22</div><div class="c0">22</div><div class="c0">23</div><div class="c0">23</div><div class="c0">23</div><div class="c0">23</div>
<div class="lbl">6</div><div class="c0">24</div><div class="c0">24</div><div class="c0">24</div><div class="c0">24</div><div class="c0">25</div><div class="c0">25</div><div class="c0">25</div><div class="c0">25</div><div class="c0">26</div><div class="c0">26</div><div class="c0">26</div><div class="c0">26</div><div class="c0">27</div><div class="c0">27</div><div class="c0">27</div><div class="c0">27</div>
<div class="lbl">7</div><div class="c0">28</div><div class="c0">28</div><div class="c0">28</div><div class="c0">28</div><div class="c0">29</div><div class="c0">29</div><div class="c0">29</div><div class="c0">29</div><div class="c0">30</div><div class="c0">30</div><div class="c0">30</div><div class="c0">30</div><div class="c0">31</div><div class="c0">31</div><div class="c0">31</div><div class="c0">31</div>
<div class="lbl">8</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">0</div><div class="c1">1</div><div class="c1">1</div><div class="c1">1</div><div class="c1">1</div><div class="c1">2</div><div class="c1">2</div><div class="c1">2</div><div class="c1">2</div><div class="c1">3</div><div class="c1">3</div><div class="c1">3</div><div class="c1">3</div>
<div class="lbl">9</div><div class="c1">4</div><div class="c1">4</div><div class="c1">4</div><div class="c1">4</div><div class="c3">5</div><div class="c3">5</div><div class="c3">5</div><div class="c3">5</div><div class="c1">6</div><div class="c1">6</div><div class="c1">6</div><div class="c1">6</div><div class="c1">7</div><div class="c1">7</div><div class="c1">7</div><div class="c1">7</div>
<div class="lbl">10</div><div class="c1">8</div><div class="c1">8</div><div class="c1">8</div><div class="c1">8</div><div class="c1">9</div><div class="c1">9</div><div class="c1">9</div><div class="c1">9</div><div class="c1">10</div><div class="c1">10</div><div class="c1">10</div><div class="c1">10</div><div class="c1">11</div><div class="c1">11</div><div class="c1">11</div><div class="c1">11</div>
<div class="lbl">11</div><div class="c1">12</div><div class="c1">12</div><div class="c1">12</div><div class="c1">12</div><div class="c1">13</div><div class="c1">13</div><div class="c1">13</div><div class="c1">13</div><div class="c1">14</div><div class="c1">14</div><div class="c1">14</div><div class="c1">14</div><div class="c1">15</div><div class="c1">15</div><div class="c1">15</div><div class="c1">15</div>
<div class="lbl">12</div><div class="c1">16</div><div class="c1">16</div><div class="c1">16</div><div class="c1">16</div><div class="c1">17</div><div class="c1">17</div><div class="c1">17</div><div class="c1">17</div><div class="c1">18</div><div class="c1">18</div><div class="c1">18</div><div class="c1">18</div><div class="c1">19</div><div class="c1">19</div><div class="c1">19</div><div class="c1">19</div>
<div class="lbl">13</div><div class="c1">20</div><div class="c1">20</div><div class="c1">20</div><div class="c1">20</div><div class="c1">21</div><div class="c1">21</div><div class="c1">21</div><div class="c1">21</div><div class="c1">22</div><div class="c1">22</div><div class="c1">22</div><div class="c1">22</div><div class="c1">23</div><div class="c1">23</div><div class="c1">23</div><div class="c1">23</div>
<div class="lbl">14</div><div class="c1">24</div><div class="c1">24</div><div class="c1">24</div><div class="c1">24</div><div class="c1">25</div><div class="c1">25</div><div class="c1">25</div><div class="c1">25</div><div class="c1">26</div><div class="c1">26</div><div class="c1">26</div><div class="c1">26</div><div class="c1">27</div><div class="c1">27</div><div class="c1">27</div><div class="c1">27</div>
<div class="lbl">15</div><div class="c1">28</div><div class="c1">28</div><div class="c1">28</div><div class="c1">28</div><div class="c1">29</div><div class="c1">29</div><div class="c1">29</div><div class="c1">29</div><div class="c1">30</div><div class="c1">30</div><div class="c1">30</div><div class="c1">30</div><div class="c1">31</div><div class="c1">31</div><div class="c1">31</div><div class="c1">31</div>
</div></div>
<p class="legend"><span class="c0"></span>warp 0（行 0–7）<span class="c1"></span>warp 1（行 8–15）<span class="c3"></span>warp 1 · lane 5</p>
:::

验证公式：warp 1 → `warp_c = (1, 0)`；lane 5 → 按 order 拆，`5 % 4 = 1`（列）、`5 // 4 = 1`（行）。行 = `1·(1·8) + 1·1 = 9`，列 = `1·4 + s`（s = 0..3）= 4..7。✔

## ⑤ 合并访存：数 sector

[[here:solutions/t2_coalesce_layout.py@L140-L157|first_load_sectors（140–157 行）]]：warp 0 的第一条 `ld.global.v4.f32`，32 个 lane 各读 16 B，共 512 B。行主序时 lane 0–3 读第 0 行的 64 B（2 个 sector），8 行共 16 个 sector，`512 / (16·32) = 1.0`，完美合并。

同一个 64×64 的行 layout 去读**列主序**的张量（`strides = [1, 64]`）：每个 lane 的 4 个元素分属 4 列、地址相隔 256 B，各落在不同 sector；第 0、1 行共享 sector，64 列 → 64 个 sector，效率 0.25。这就是转置 kernel 必须在读和写之间换 layout 的原因。

## ⑥ convert_layout 的三档

[[here:solutions/t2_coalesce_layout.py@L160-L176|classify_conversion（160–176 行）]] 用“每个元素归谁”来判断；Triton 用线性布局精确判断（[[triton:lib/Analysis/Utility.cpp@bool cvtNeedsWarpShuffle|cvtNeedsWarpShuffle]]）：

| 从上图的 layout 转到 | 归属变化 | 档位 |
|---|---|---|
| 同一个 layout | 无 | registers |
| `[1, 2] / [8, 4] / [2, 1]`（每线程 2 列） | 每行仍在同一个 warp，lane 变了 | warp_shuffle |
| `[4, 1] / [4, 8] / [1, 2]`（列优先） | 第 0 行原来全在 warp 0，现在分给两个 warp | shared_memory |

::: flow 一次 shared_memory 档的转换在 PTX 里长什么样
cfg cw=200 nw=170
n r 0 0 "寄存器（src layout）" "" k=acc
n st 1 0 "st.shared" "按 src 写" k=warn
n bar 2 0 "bar.sync" "等所有 warp" k=vio
n ld 3 0 "ld.shared" "按 dst 读" k=warn
n r2 3 1.2 "寄存器（dst layout）" "" k=pass
e r -> st
e st -> bar
e bar -> ld
e ld -> r2
:::

::: callout analogy 交接班
寄存器档：工具还在你手里，只是换了个口袋。warp shuffle 档：同一组的人互相递一下。shared memory 档：工具要交给另一组，只能先放进公共工具房、等所有人放好、再去取。`remove_layout_conversions` 这个 pass 的工作，就是尽量把工具房那一档消掉。
:::

## ⑦ 源码与资料索引

- [[triton:lib/Dialect/TritonGPU/Transforms/Coalesce.cpp|Coalesce.cpp]] · [[triton:lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp|CoalesceUtils.cpp]]
- [[triton:lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp|RemoveLayoutConversions.cpp]]
- [[triton:lib/Analysis/Utility.cpp@bool cvtReordersRegisters|cvtReordersRegisters / cvtNeedsWarpShuffle / cvtNeedsSharedMemory]]
- [[triton:python/tutorials/gluon/02-layouts.py|Gluon 02-layouts（手写 layout）]] · [Linear Layouts 论文](https://arxiv.org/abs/2505.23819)

## ⑧ 自测

::: answer 把 coalesce.mlir 里指针的 tt.divisibility 从 16 改成 8，64×64 fp32 的 layout 变成什么？
`max_multiple = 8 / 4 = 2`，perThread = min(2, 64, 4, 32) = 2，spt = [1, 2]。builder：dim 1 `threads_cta = clamp(128, 1, 64/2 = 32) = 32`，tpw[1] = 32，wpc[1] = 1；dim 0：tpw = 1，wpc = 4。结果 `[1, 2] / [1, 32] / [4, 1]`：每个 warp 正好覆盖一整行 64 个元素，每条 load 只有 8 B。
:::

::: answer 为什么 `order` 翻转也会让转换变成 warp_shuffle 而不是 registers？
`order` 不仅决定寄存器编号，还决定 lane 和 warp 的编号顺序（`_delinearize` 对 lane、warp 都用 `L.order`）。所以翻转 order 会让同一个元素换到另一个 lane 上，测试 `test_classify_conversion` 专门检查了这一点。
:::
