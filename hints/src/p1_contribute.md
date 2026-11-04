title: P1 讲解：一次完整的贡献演练（以 Luminal #370 为例）
lede: 把“第一笔贡献”拆成可以逐项打勾的步骤：如何判断一个 issue 是否适合你、在 issue 里第一次回复应该写什么、怎样把修复做得足够小、PR 描述的每一节写什么，以及 review 时的常见要求。以 Luminal #370（CUDA 13.x 编译失败）为主线，Triton 的 lit 测试贡献为辅线。

::: callout info 怎么读这一页
这是实践题，没有 check.sh。issue 的状态会变：开始前先确认 #370 是否仍然 open、有没有人已经提了 PR、当前 main 是否仍能复现。
:::

## ① 判断：这个 issue 适合我吗

::: flow 五个筛子
cfg cw=180 nw=150
n m 0 0 "模块熟吗" "练过的部分" k=acc
n h 1 0 "硬件有吗" "SM86 能复现" k=acc
n r 2 0 "能复现吗" "最小复现" k=warn
n p 3 0 "有人在做吗" "搜 PR" k=warn
n v 4 0 "有人回吗" "维护者活跃" k=pass w=0.8
e m -> h
e h -> r
e r -> p
e p -> v
:::

#370 的情况：模块是 CUDA backend 的 build（你在 basics_env 里已经碰过）；需要的是 CUDA 13.x Toolkit，不需要特定 GPU；复现就是一次 `cargo build`；范围可能很小（换一个 API），也可能牵涉 cudarc fork。

## ② 第一次回复写什么

```markdown
I can reproduce this on current main (<sha>).

- nvcc --version: Cuda compilation tools, release 13.x, V13.x.xx
- driver: 5xx.xx, GPU: RTX 3070 (sm_86)
- cargo 1.xx, command: `cargo build -p luminal_cuda_lite --features device`

Error (trimmed):
<the first error lines, with the missing symbol names>

These symbols come from <cudarc / our cuda_graph.rs>. CUDA 13 replaced them with <...>.
Would a PR that <switches to the v2 API behind a version check / bumps the cudarc rev> be welcome?
```

要点：只贴第一个错误（后面的往往是连锁反应）；说清楚符号来自哪里；**先问方向**，尤其当修法涉及依赖（cudarc 是 Luminal 自己的 fork，[[luminal:crates/luminal_cuda_lite/Cargo.toml@cudarc|Cargo.toml 里固定了 rev]]）。

::: callout analogy 先打电话再上门
给维护者的第一条回复像修理工先打电话：“我看到您家水管的问题了，原因可能是 X，我打算这样修，您看行吗？”而不是直接带着工具上门把墙拆了。对方可能告诉你：“那根管子下周就要整体换掉了”——省下你一周的工作。
:::

## ③ 修复要小

- 只改让编译通过所必需的地方；不要顺手重构、改格式、升级其他依赖。
- 如果要兼容 CUDA 12 和 13，用最小的版本分支（cudarc 的 `cuda-version-from-build-system` feature 就是为这个准备的），并说明在两个版本上都编译过。
- 附一个最小的 build 回归：CI 里能覆盖 CUDA 13 最好；不能的话，在 PR 描述里写清楚你在哪些 Toolkit 版本上手动验证过。

## ④ PR 描述逐节写什么

| 节 | #370 的例子 |
|---|---|
| Summary | `luminal_cuda_lite` 在 CUDA 13.x 上编译失败，因为 v1 graph dependency API 被移除 |
| Reproducer | commit、Toolkit 13.x、driver、`cargo build -p luminal_cuda_lite --features device` |
| Root cause | `cuda_graph.rs` 调用了 `<symbol>`，CUDA 13 中改为 `<symbol_v2>` |
| Fix | 按 Toolkit 版本选择 API；为什么不影响 CUDA 12 路径 |
| Validation | CUDA 12.x 和 13.x 下 build 通过；`cargo test -p luminal_cuda_lite` 结果；一个 example 能跑 |

（`<symbol>` 这些要用你实际看到的名字填；本页不猜具体是哪个 API。）

## ⑤ Triton 辅线：一个 lit 测试 PR

1. 找一个没被测试覆盖的情况（例如 T2 自测里的 `tt.divisibility = 8`）。
2. 在 `test/TritonGPU/coalesce.mlir` 末尾加 `// -----` 分段：最小输入 IR + `CHECK-LABEL` + 只检查 layout 的 CHECK，layout 用 `[[$NAME:#.*]]` 捕获（Triton 默认 `--enable-var-scope`，见 T5 讲解）。
3. `lit -v` 通过；故意改坏 `CoalesceUtils.cpp` 确认它会失败。
4. PR 标题写成“[TritonGPU] Add coalesce test for 8-byte aligned pointers”这类具体的描述。

## ⑥ review 的常见要求

| 要求 | 怎么回应 |
|---|---|
| “拆成两个 PR” | 照做；一个 PR 一件事，review 更快 |
| “加个测试” | 加；回复里贴上测试修复前失败的输出 |
| “性能数据？” | K2 的表：median、IQR、n、`compare` 结论 |
| “这会影响 AMD / Hopper 吗？” | 说明影响范围，或者说明为何不影响 |
| 长时间无回复 | 一周后礼貌地 ping 一次，附上 rebase 后的状态 |

## ⑦ 源码与资料索引

- [luminal #370](https://github.com/luminal-ai/luminal/issues/370) · [Luminal issues](https://github.com/luminal-ai/luminal/issues) · [Luminal PRs](https://github.com/luminal-ai/luminal/pulls)
- [[luminal:crates/luminal_cuda_lite/src/cuda_graph.rs|cuda_graph.rs]] · [[luminal:crates/luminal_cuda_lite/Cargo.toml|luminal_cuda_lite/Cargo.toml]]
- [[triton:CONTRIBUTING.md|Triton CONTRIBUTING.md]] · [[triton:test/TritonGPU/coalesce.mlir|coalesce.mlir]] · [[triton:test/lit.cfg.py@enable-var-scope|test/lit.cfg.py]]

## ⑧ 自测

::: answer 你在 CUDA 13 上编不过，但 issue 已经有人提了 PR，还能做什么？
在那个 PR 里帮忙验证：用你的环境构建它，回复“tested on Toolkit 13.x / driver / GPU, builds and example X runs”。这对维护者很有价值（他们未必有那个环境），也让你进入这个模块的讨论。然后再从 open issues 里挑一个 reference-vs-CUDA 测试（L5）作为自己的第一笔贡献。
:::
