title: L3 讲解：实测搜索、代理成本和 finalists
lede: 在 24 个 genome 的 MLP 计划空间里，把参考解的遗传搜索一步一步跑出来：best-so-far 曲线怎样从 2422 ns 掉到 95 ns；FLOP 代理为什么和实测只有 0.82 的秩相关；按代理选前 3 名为什么全部被验证拒绝，而 GA 的第 3 名能通过。再对照 Luminal 的 Genome、rank_insert 和 PendingFinalist。

::: callout info 怎么读这一页
对照 [[here:solutions/l3_search.py|l3_search.py]]。“实测”来自 `mlp_measure()`：一个确定的延迟表（[[here:solutions/l3_search.py@L142-L169|142–169 行]]），模拟真实 GPU 上那些代理成本看不见的效应。
:::

## ① 空间与真值

genome = 四个 e-class 各选一个实现：matmul（primitive_loop / tiled_kernel / cublaslt）× epilogue（separate / fused）× w_layout（row / col）× accumulate（fp32 / fp16），共 24 个。按实测排序的前几名：

| 实测 ns | 代理 | matmul | epilogue | accumulate | 备注 |
|---|---|---|---|---|---|
| 95 | 105 | cublaslt | fused | fp16 | 最快，但 fp16 累加数值不合格 |
| 103 | 110 | cublaslt | fused | **fp32** | 第一个合格的 |
| 117 | 125 | cublaslt | separate | fp16 | |
| 125 | 130 | cublaslt | separate | fp32 | |
| … | | | | | 最慢的 primitive_loop：2430 ns |

`w_layout` 对 cuBLASLt 没有影响，所以每一行实际是两个 genome。

## ② GA：best-so-far 曲线

[[here:solutions/l3_search.py@L60-L87|genetic_search（60–87 行）]]，`budget=10, seed=0` 的曲线：

```text
测量序号:  1     2    3    4    5    6   7   8   9   10
best ns : 2422  127  117  117  117  95  95  95  95  95
```

第一个随机 genome 恰好是 primitive_loop；第 2 个就降到 127（tiled_kernel）；第 6 个找到最优。曲线的形状回答“还要不要继续搜”：后 5 次没有改进，再加预算的收益很小。`budget=24` 时 GA 测到第 20 个不同 genome 后，连续 50 次只产生已测过的 genome，提前停止。

::: flow GA 的一代
cfg cw=205 nw=175
n r 0 0.6 "ranked（keep=4）" "最快的 4 个" k=pass
n x 1 0 "crossover" "每个类随机取一个父代" k=warn
n mu 1 1.2 "mutate" "每个类 30% 重抽" k=warn
n fp 2 0.6 "fingerprint 去重" "测过就跳过" k=teal
n ms 3 0.6 "measure" "rank_insert" k=acc
e r -> x
e x -> mu
e mu -> fp
e fp -> ms
:::

## ③ rank_insert：和 Rust 版逐行对照

| Rust（[[luminal:crates/luminal_cuda_lite/src/search.rs@fn rank_insert|search.rs]]） | Python（[[here:solutions/l3_search.py@L50-L57|50–57 行]]） |
|---|---|
| `if ranked.len() >= keep && nanos >= worst { return; }` | `if len(ranked) >= keep and nanos >= ranked[-1][0]: return` |
| `position(\|(metric, _)\| nanos < *metric)` | `next(i for ... if nanos < m)` |
| `ranked.insert(position, ...); ranked.truncate(keep);` | `ranked.insert(pos, ...); del ranked[keep:]` |

等号的处理很重要：相等的新候选不能挤掉已有的（先到先得，结果可复现）。

## ④ 代理成本：方向对，细节错

`mlp_proxy` 知道 primitive_loop 很差、融合省一次读写、fp16 便宜一点，但它认为 tiled_kernel 和 cublaslt 一样快（都是 100），也看不见 tiled_kernel 在 row_major 的 W 上要多付 20 ns。所有 24 个 genome 上 `spearman(proxy, measured) ≈ 0.82`。

于是 `proxy_search(k=3)` 测的是代理最好的三个：cublaslt/fused/fp16（95 ns）、tiled/fused/fp16/col（127 ns）、tiled/fused/fp16/row（147 ns）——**三个全是 fp16 累加**，全部被验证拒绝（`test_proxy_top_k_can_all_be_invalid`）。

::: callout analogy 看菜单点菜
代理成本像看菜单上的价格估计哪道菜好吃：贵的大概不差，但两道同价的菜哪道更合口味，菜单说不出来。实测是真的尝一口。finalists 是“厨房说这道菜今天做不了”：排第一的不能做，就上第二道。只尝菜单上最便宜的三道，可能三道都恰好今天做不了。
:::

## ⑤ finalists：验证和排名分开

[[here:solutions/l3_search.py@L122-L131|select_finalist（122–131 行）]] 对 GA 的 ranked `[95 fp16, 95 fp16, 103 fp32, 103 fp32]`：第 1、2 名被拒（fp16 累加，K = 256 超出容差），第 3 名通过 → `(3, 103, {...fp32...}, [(1, ...), (2, ...)])`。

真实的 [[luminal:crates/luminal_cuda_lite/src/finalists.rs@pub struct PendingFinalist|PendingFinalist]]：`rank` 是 1-based 的 GA 排名，`metric` 是实测纳秒，`arena.slab_bytes` 是这个候选的设备内存高水位；“A rejected candidate advances the walk to the next measured rank”。本题的 `mlp_validate` 也包含内存检查：primitive_loop 的 `[M, N, K]` 中间结果超过预算就拒绝（L1 算过它有多大，L4 会算出 slab）。

## ⑥ break-even

[[here:solutions/l3_search.py@L134-L139|break_even_runs（134–139 行）]]：多花 1.9 s 搜索，每次执行省 30 ns，要执行 1.9 / 30e-9 ≈ 6300 万次才回本。对一个要服务上亿 token 的推理模型这很划算；对一次性的小脚本则不值得。这就是 search trade-off 报告的结论该有的样子。

## ⑦ 源码与资料索引

- [[luminal:src/extraction.rs@pub struct Genome|Genome]] · [[luminal:src/extraction.rs@pub struct SamplingSpace|SamplingSpace]] · [[luminal:src/extraction.rs@pub fn plan_fingerprint|plan_fingerprint]]
- [[luminal:crates/luminal_cuda_lite/src/search.rs@pub fn search_implementations|search_implementations]] · [[luminal:crates/luminal_cuda_lite/src/search.rs@pub struct CompileOptions|CompileOptions]]
- [[luminal:crates/luminal_cuda_lite/src/finalists.rs|finalists.rs]]
- README：`SEARCH_LOG`、`LUMINAL_SEARCH_SEED`、`SEARCH_MEMORY_MIB`、`LUMINAL_SEARCH_OP_NAMES`

## ⑧ 自测

::: answer 为什么不把数值验证（fp16 累加不合格）直接放进搜索空间，提前剪掉？
可以，而且在真实系统里常这样做（Luminal 在搜索前就有 memory pruning 的 post-pass）。但有两类验证只能事后做：依赖实测的（某个候选在设备上产生了 NaN）和依赖完整计划的（整个计划的 arena 是否放得下）。所以排名之后总要保留一个 finalist 遍历作为兜底；`keep` 太小，兜底就会失败。
:::

::: answer 代理和实测的相关系数只有 0.82，代理还有用吗？
有用：它能便宜地排除明显差的区域（primitive_loop 永远不会被代理排进前列），让实测预算花在“代理分不清”的候选上。常见做法是代理粗筛 + 实测精排，就像 T3 的 perf_model + top_k。危险的是只信代理、或者 k 取得太小。
:::
