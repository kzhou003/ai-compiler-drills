title: K3 讲解：ULP、容差和“不一样不等于错了”
lede: 浮点数在数轴上不是均匀分布的，所以“差多少算错”必须用 ULP 和随 dtype、归约长度缩放的容差来回答。这一页从位模式讲到 check_close 的每一个分支，用 [1e8, 1, −1e8, 1] 演示重排的效果，并解释 fp16 matmul 那个测试里的容差是怎么凑出来的。

::: callout info 怎么读这一页
对照 [[here:solutions/k3_numerics.py|k3_numerics.py]]。T4 和 L5 都直接 import 这里的 `check_close`、`tolerance`、`edge_sizes`、`to_bf16`。
:::

## ① 三种格式的位

::: raw
<div class="strip-scroll"><div class="cells" style="grid-template-columns: 70px repeat(32, minmax(14px, 1fr)); min-width: 620px; gap: 1px">
<div class="lbl">fp32</div><div class="c1">s</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div>
<div class="lbl">bf16</div><div class="c1">s</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div>
<div class="lbl">fp16</div><div class="c1">s</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c0">e</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="c2">m</div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div><div class="cx"></div>
</div></div>
<p class="legend"><span class="c1"></span>符号<span class="c0"></span>指数<span class="c2"></span>尾数<span class="cx"></span>不存在</p>
:::

bf16 就是 fp32 的高 16 位：指数一样宽（范围和 fp32 相同），尾数只剩 7 位。所以 [[here:solutions/k3_numerics.py@L15-L21|to_bf16（15–21 行）]] 只需对 fp32 的位模式做“加上舍入偏置再右移 16 位”：偏置是 `0x7FFF + (第 16 位)`，正好实现 round-to-nearest-even。测试里 `1 + 2⁻⁸` 恰在 1 和 `1 + 2⁻⁷` 中间，舍入到尾数为偶的 1.0；`1 + 3·2⁻⁸` 在 `1 + 2⁻⁷` 和 `1 + 2⁻⁶` 中间，舍入到 `1 + 2⁻⁶`。

## ② ULP：把浮点数排成整数

正浮点数的位模式（当作整数）随数值单调增加；负数的符号位在最高位，当作有符号整数是倒序的。[[here:solutions/k3_numerics.py@L24-L42|_ordered_int / ulp_diff（24–42 行）]] 把负数翻成 `-(i & 0x7FFFFFFF)`：

::: flow 浮点数轴 → 有序整数轴
cfg cw=160 nw=135
n a 0 0 "−tiny" "→ −1" k=acc
n z 1 0 "−0.0 / +0.0" "→ 0" k=warn
n b 2 0 "+tiny" "→ +1" k=acc
n o 3 0 "1.0" "→ 0x3F800000" k=pass
n n 4 0 "nextafter(1)" "→ 0x3F800001" k=pass
e a -> z
e z -> b
e b -> o dash
e o -> n
:::

于是 `ulp_diff(-tiny, +tiny) = 2`，`ulp_diff(-0.0, +0.0) = 0`，`ulp_diff(1, nextafter(1)) = 1`。ULP 距离和数值大小无关：在 1 附近 1 ULP 是 1.2e-7，在 1e8 附近是 8。

::: callout analogy 楼层号而不是海拔
浮点数像一栋越往上层高越高的楼。两层之间的海拔差（绝对误差）在低楼层是几厘米，在高楼层是几米；但“隔了几层”（ULP）在哪里都是同一个意思。说“结果差了 2 层”比说“差了 3 米”更能判断是不是算错了。
:::

## ③ 重排：同一组数，两个答案

`[1e8, 1, −1e8, 1] × 4` 在 fp32 里（[[here:solutions/k3_numerics.py@L45-L62|sum_sequential / sum_pairwise（45–62 行）]]）：

| 顺序 | fp32 结果 | 为什么 |
|---|---|---|
| 顺序累加 | **1.0** | `1e8 + 1` 舍入回 1e8（1e8 附近 1 ULP = 8），每组的第一个 1 都丢了，只有最后一组的最后一个 1 留下 |
| 树形归约 | **0.0** | 第一轮 `1e8 + 1 → 1e8`、`−1e8 + 1 → −1e8`，所有的 1 都丢了 |
| 精确值 | 8 | |

**任何改变求和顺序的编译器变换**（`num_warps`、split-K、e-graph 的结合律）都会带来这种差异。

## ④ 容差：rtol = c · eps · √k

[[here:solutions/k3_numerics.py@L65-L73|tolerance（65–73 行）]]。k 次舍入、每次误差约 ±eps/2 且大致随机，和的误差像随机游走，标准差 ∝ √k。c 是安全系数。

测试 `test_fp16_matmul_needs_a_tolerance`：64×512 @ 512×64，输入舍入到 fp16。

| 容差 | 值（k = 512） | 结果 |
|---|---|---|
| `tolerance("float32", 512)` | rtol = 2·2⁻²³·22.6 ≈ 5.4e-6 | 失败：输入舍入误差就有 2⁻¹¹ ≈ 4.9e-4 |
| `tolerance("float16", 512, c=4)`，atol × √512 | rtol ≈ 0.088，atol ≈ 2.0 | 通过：输出量级约 √512 ≈ 22.6，atol 要跟着放大 |

**为什么 atol 要乘 √k**：随机输入的点积结果大约是 √k 量级，有些元素恰好接近 0，相对误差在那里没有意义，所以绝对误差的门槛要和输出的典型大小同量级。

## ⑤ check_close 的分支

[[here:solutions/k3_numerics.py@L76-L95|check_close（76–95 行）]]：形状不同 → 失败；NaN 位置不同 → 失败并报下标；其余位置 `|a − e| ≤ atol + rtol·|e|`；报告 `mismatched`、`max_abs`、`max_rel` 和最差的下标（`argmax(diff − allowed)`，即“超标最多”的元素，而不是差值最大的元素）。

## ⑥ 边界输入

[[here:solutions/k3_numerics.py@L98-L112|edge_sizes / edge_values（98–112 行）]]：`edge_sizes(64, 1000)` = `[1, 63, 64, 65, 97, 129, 992]`（992 = 1000 − 1000%64 + 32，一个大的非整除尺寸）。`edge_values("float16")` 里第 6 个是 `2·√65504 ≈ 512`，它的平方 262144 超过 fp16 最大值 65504 → inf：专门用来抓“中间结果溢出”。

## ⑦ 源码与资料索引

- [[luminal:crates/luminal_reference/src/narrow_float.rs|Luminal narrow_float.rs]] · [[luminal:crates/luminal_reference/src/tf32.rs|tf32.rs]]：Reference 一侧如何精确模拟低精度
- [[luminal:src/frontend/binary.rs@pub fn maximum|Luminal maximum 的 NaN 语义]]
- [[triton:python/triton/language/core.py@def dot|tl.dot 的 input_precision]]
- [Goldberg：浮点数](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) · [torch.testing.assert_close](https://pytorch.org/docs/stable/testing.html)

## ⑧ 自测

::: answer bf16 的 `ulp_diff(1.0, 1.0 + 2⁻⁷)` 为什么是 1？
bf16 有 7 位尾数，1 附近相邻两个可表示数相差 2⁻⁷。`1 + 2⁻⁷` 正好是 1.0 的下一个 bf16 数，所以隔 1 个 ULP。同样的两个数在 fp32 里相差 2¹⁶ 个 ULP。
:::

::: answer 如果一个 kernel 的输出和参考比，只有 1 个元素超出容差，你怎么办？
看 `index` 和那里的值：是不是接近 0（atol 太小）？是不是在边界（最后一块、最后一行——tail 处理的 bug）？是不是输入恰好是边界值（溢出）？单个元素失败往往比大面积失败更有信息量：它指向一个具体的坐标，T4 的 shrink 就从这里开始。
:::
