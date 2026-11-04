title: X1 讲解：同一个 fused MLP，怎样做一次公平的两边对照
lede: X1 没有参考解，只有方法。这一页讲清楚对照实验的四个陷阱（测的不是同一个东西、编译时间混进来、只报最好的一次、形状偏袒一方），给出把 K1–L5 的工具串起来的完整脚本骨架，以及一份分析报告该写的结构。

::: callout info 怎么读这一页
对照主页的 X1 一节。这里用到的每个函数都在前面的题里写过：K1 `analyze`、K2 `bench` / `compare`、K3 `check_close`、T1 `ptx_instructions`、T3 `autotune`、L3 `break_even_runs`、L4 `phase_breakdown`。
:::

## ① 四个陷阱

| 陷阱 | 例子 | 对策 |
|---|---|---|
| 测的不是同一个东西 | Triton 的 kernel 里融合了 bias + relu，Luminal 的计时还包括输入上传 | 两边都只计“输入已在 GPU 上 → 输出在 GPU 上” |
| 一次性成本混进来 | Luminal 第一次 execute 包含 prepare / cuBLASLt 算法选择 | K2 `compile_vs_run`；Luminal 用 `LUMINAL_CUDA_PROFILE_RECAPTURE` 分开 |
| 只报最好的一次 | Triton autotune 的最小值 vs Luminal 的单次 | 都用 median + IQR，`compare` 判定 |
| 形状偏袒一方 | 只测 4096³（cuBLASLt 的主场） | decode / prefill / odd 三个形状都测 |

::: flow 对照实验的数据流
cfg cw=215 nw=185
n env 0 0.6 "环境快照" "basics_env 脚本" k=acc
n tr 1 0 "Triton 路径" "T3 autotune → 最佳 kernel" k=pass
n lu 1 1.2 "Luminal 路径" "compile → execute" k=vio
n ok 2 0.6 "正确性" "两边 vs 参考：check_close" k=warn
n tb 3 0.6 "对照表" "median · IQR · 一次性成本" k=teal
e env -> tr
e env -> lu
e tr -> ok
e lu -> ok
e ok -> tb
:::

## ② Triton 一侧的脚本骨架

```python
from solutions.k1_gemm_model import RTX3070, TileConfig, analyze
from solutions.k2_bench_harness import bench, compare
from solutions.k3_numerics import check_close, tolerance
from solutions.t1_ir_anatomy import dump_stages, ptx_instructions
from solutions.t3_autotune_sweep import autotune, to_csv

SHAPES = {"decode": (16, 4096, 4096), "prefill": (2048, 4096, 4096), "odd": (1000, 4095, 1000)}
for name, (M, K, N) in SHAPES.items():
    best, rows, _ = autotune(RTX3070, M, N, K, real_measure_for(M, N, K), k=10, repeats=5)
    k = launch_fused_mlp(best, M, N, K)                         # 返回 CompiledKernel
    assert check_close(out.cpu(), ref, *tolerance("float16", K, c=4))["ok"]
    r = bench(lambda: launch_fused_mlp(best, M, N, K), sync=torch.cuda.synchronize)
    mma = sum(ptx_instructions(dump_stages(k)["ptx"])["mma"].values())
    theory = analyze(RTX3070, M, N, K, TileConfig(best["BM"], best["BN"], best["BK"], best["num_warps"], best["num_stages"]))
```

## ③ Luminal 一侧要记录的东西

1. `cx.compile(...)` 的耗时（graph compile + search）。
2. `SEARCH_LOG` / `LUMINAL_SEARCH_OP_NAMES` 最后一行 best-so-far 的组成：是 `cublasLtMatmul` 一个调用，还是多个 kernel。
3. 稳态 execute 的 median（跳过第一次）。
4. `CUDA_EXEC_PROFILE` 里 kernel 时间和 runtime 开销的比例，填进 L4 的 `phase_breakdown`。

::: callout analogy 赛车对比
比两辆车，得在同一条赛道（形状）、同样的天气（环境快照）、从同样的起跑线（数据已在 GPU 上）开始计时，跑多圈取中位数，而且不能把“出厂调校”（编译 / 搜索）算进圈速——但要单独记下来，因为有的车调校一次要一小时。
:::

## ④ 报告结构

1. **设置**：环境快照、三个形状、测量方法（K2 的参数）。
2. **结果表**：主页 X1 的模板。
3. **解释**：每个形状一段——瓶颈在 kernel 本身（看 Triton 的 ncu 数据和 K1 的理论值）还是在组织执行（看 Luminal 的阶段分解）。
4. **差距与机会**：Luminal 选中的实现比最佳 Triton kernel 慢的形状，就是一个候选 issue：搜索空间缺了什么？代价是否被错误测量？
5. **break-even**：L3 的公式，Luminal 的搜索成本在多少次执行后摊平。

## ⑤ 源码与资料索引

- [[triton:python/tutorials/03-matrix-multiplication.py|Triton tutorial 03]] · [[triton:python/triton/testing.py@def do_bench|do_bench]]
- [[luminal:README.md|Luminal README（环境变量）]] · [[luminal:crates/luminal_cuda_lite/src/ops/cublaslt/README.md|cuBLASLt 接入说明]] · [[luminal:crates/luminal_cuda_lite/src/cuda_graph.rs|cuda_graph.rs]]

## ⑥ 自测

::: answer odd 形状（K = 4095）为什么可能让两边都慢很多？
4095 不是 8 的倍数：fp16 行的起始地址不再 16 B 对齐，Triton 的 coalesce pass 只能用更窄的向量（T2：divisibility 决定 perThread），cuBLASLt 也可能选不到最快的算法（很多算法要求 lda 对齐）。这类形状正是“非整除 / 对齐”类 bug 和性能悬崖的高发区，值得放进任何对照实验。
:::
