title: L1 讲解：relu(x @ w + b) 在 Luminal 眼里是什么
lede: 把 M=2、K=3、N=4 的 MLP 交给参考解，逐行解读它记录下的 15 个节点：哪些是只改索引的 view，哪个是唯一的大中间结果，relu 为什么是 Select；再执行一遍 reference runtime，并和 Luminal 源码里真实的 matmul / relu / exp 对照。

::: callout info 怎么读这一页
对照 [[here:solutions/l1_graph.py|l1_graph.py]]。语义表由 `build_mlp(2, 3, 4)[0].semantics_table()` 生成。
:::

## ① 语义表

| id | op | 输入 | shape | dtype | 说明 |
|---|---|---|---|---|---|
| 0 | Input | — | [2, 3] | f32 | x |
| 3 | Expand | 0 | [2, 4, 3] | f32 | 第 1 维广播到 4（view） |
| 1 | Input | — | [3, 4] | f32 | w |
| 4 | Permute | 1 | [4, 3] | f32 | 转置（view） |
| 5 | Expand | 4 | [2, 4, 3] | f32 | 第 0 维广播到 2（view） |
| 6 | **Mul** | 3, 5 | **[2, 4, 3]** | f32 | 逐元素乘：M·N·K 个积 |
| 7 | ReduceSum | 6 | [2, 4] | f32 | 沿 K 求和 → x @ w |
| 2 | Input | — | [4] | f32 | b |
| 8 | Expand | 2 | [2, 4] | f32 | 广播 bias（view） |
| 9 | Add | 7, 8 | [2, 4] | f32 | + b |
| 10 | Constant | — | [] | f32 | 0.0 |
| 11, 12 | Expand × 2 | 10 | [2] → [2, 4] | f32 | 标量 0 广播成矩阵（view） |
| 13 | LessThan | 9, 12 | [2, 4] | **bool** | h < 0 |
| 14 | Select | 13, 12, 9 | [2, 4] | f32 | h < 0 ? 0 : h |

计数：Input 3、Expand 5、Permute 1、Mul 1、ReduceSum 1、Add 1、Constant 1、LessThan 1、Select 1。拓扑序（[[here:solutions/l1_graph.py@L53-L66|topo_order（53–66 行）]]）是从输出反向 DFS 的后序，所以 id 顺序和执行顺序不完全一样（id 2 的 b 在 7 之后才出现）。

## ② matmul = 两个 view + Mul + ReduceSum

::: flow 下标的视角：C[m, n] = Σ_k A[m, k] · B[k, n]
cfg cw=220 nw=185
n a 0 0 "A[m, k]" "[M, K]" k=acc
n ae 1 0 "A′[m, n, k] = A[m, k]" "Expand：不看 n" k=ghost w=1.1
n b 0 1.2 "B[k, n]" "[K, N]" k=acc
n be 1 1.2 "B′[m, n, k] = B[k, n]" "Permute + Expand" k=ghost w=1.1
n p 2.2 0.6 "P = A′ · B′" "[M, N, K]" k=warn
n c 3.2 0.6 "C = Σ_k P" "ReduceSum" k=pass w=0.8
e a -> ae
e b -> be
e ae -> p
e be -> p
e p -> c
:::

[[here:solutions/l1_graph.py@L164-L169|matmul（164–169 行）]] 和真实的 [[luminal:src/frontend/matmul.rs@fn matmul_body|matmul_body]] 是同一个想法：把两个操作数都变成 `[..., M, N, K]` 的 view，相乘，沿最后一维求和。真实代码用一个 view map 节点（`record_view_map`）直接写出每个输出坐标读哪个输入坐标（`am = [c(2, m), c(0, k)]`、`bm = [c(0, k), c(1, n)]`），而不是一串 Expand / Permute，但语义相同。

::: callout analogy 乘法口诀表
`[M, N, K]` 的 Mul 节点就像把一整张口诀表的每一格都写出来，再一行一行加起来。写在纸上（物化成 buffer）要 M·N·K 格；聪明的做法是边乘边加、只记结果（GEMM kernel）。L1 记录的是“写满整张表”的朴素版本；把它换成“边乘边加”是 e-graph 和搜索的工作。
:::

## ③ relu 为什么是 Select

Luminal 的 [[luminal:src/frontend/unary.rs@pub fn relu|relu]] 是 `maximum_f32(0.)`，而 [[luminal:src/frontend/binary.rs@pub fn maximum|maximum]] 是：

```rust
let picked = self.lt(rhs).select(rhs, self);   // a < b ? b : a
rhs.nan_overrides(picked)                      // b 是 NaN 时返回 b：maximum 传播 NaN
```

注释里解释了两件事：(1) 以前的写法是算术掩码 `(a<b)·b + (b<=a)·a`，这种“积之和”会喂给（整数）结合律 / 交换律 / 分配律的 e-graph 闭包，链起来就爆炸，所以改成原生的三元 `Select`；(2) 与 `fmax` 不同，`maximum` 要传播 NaN。参考解省略了 NaN 覆盖那一步——这是你对照真实图时会多看到的那一个 Select。

## ④ reference runtime

[[here:solutions/l1_graph.py@L192-L225|run（192–225 行）]]：按拓扑序，每个节点用 numpy 算一次。几个要点：

- `Expand` 用 `np.broadcast_to`：返回 view，不复制——与“broadcast 不搬数据”的语义一致。
- `ReduceSum` 显式 `dtype=np.float32`：累加精度是语义的一部分。
- `LessThan` 产出 bool，`Select` 用 `np.where`。

测试在 `(2,3,4)`、`(1,1,1)`、`(7,13,5)` 上与 `np.maximum(X @ W + B, 0)` 对拍，误差 1e-7 量级（fp32 求和顺序不同）。

## ⑤ 朴素执行要多少内存

[[here:solutions/l1_graph.py@L93-L101|materialized_bytes（93–101 行）]]：`build_mlp(64, 256, 64)` 中 Mul 节点 64·64·256·4 B = 4 MiB，其他四个非 view 节点各 16 KiB，共 4,259,840 B。真正需要的输出只有 16 KiB。**把 Mul + ReduceSum 融合成一个 GEMM，省下的就是这 4 MiB**——L3 里 `primitive_loop` 候选因为超出 arena 预算被拒，就是这个原因。

## ⑥ 源码与资料索引

- [[luminal:README.md|README：RISC-style architecture、Compile everything]]
- [[luminal:src/logical/ir.rs@pub enum LogicalOp|LogicalOp]] · [[luminal:src/graph.rs@pub struct Graph|Graph]]
- [[luminal:src/frontend/movement.rs@fn record_view_map|record_view_map]] · [[luminal:src/frontend/unary.rs@pub fn exp|exp = exp2(x·log2 e)]]

## ⑦ 自测

::: answer 为什么 `exp` 要写成 `exp2(x · log2(e))`？
primitive 集里只有 `Exp2`（硬件上 `ex2.approx` 是原生指令，`exp` 本来就是这样实现的）。把 `exp` 拆成乘法 + `Exp2` 之后，乘法可以和前面的运算合并（比如 softmax 里的 `x · scale · log2(e)` 合成一个常数），后端也只需要实现一种指数。
:::

::: answer 如果 b 的 shape 是 [1, 4] 而不是 [4]，参考解要怎么改？
二元 op 要求 shape 完全相等，`[1, 4]` 不能直接加到 `[2, 4]` 上。需要一个“把长度 1 的轴广播到 2”的 view（Luminal 的 `expand` 就是这样：“every size-1 axis whose target size differs broadcasts to it”），参考解的 `expand_to` 只处理插入新轴，可以加一个分支：在 `shape[d] == 1` 的轴上用 `np.broadcast_to`。
:::
