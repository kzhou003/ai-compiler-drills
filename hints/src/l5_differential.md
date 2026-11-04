title: L5 讲解：一个能抓住真 bug 的差分测试
lede: 把三个“后端风格”的 softmax bug 依次放进参考解的差分测试：tail 被 n=97 当场抓住（误差 0.54）；fp16 累加只差 4e-5，靠 fp32 的紧容差才抓到；没减最大值的版本在 N(0,3) 输入下永远通过，直到输入放大到 40 倍。再把容差公式拆开讲，说明“理由”那一行该怎么写。

::: callout info 怎么读这一页
对照 [[here:solutions/l5_differential.py|l5_differential.py]]。`device_softmax`（[[here:solutions/l5_differential.py@L41-L64|41–64 行]]）是一个分块、在线 max/sum 的“设备实现”，`bug` 参数打开三种错误。
:::

## ① 测试的骨架

::: flow 一个 case 的生命周期
cfg cw=200 nw=172
n s 0 0.6 "rng 抽样" "rows · n · dtype · seed" k=acc
n x 1 0.6 "make_input" "舍入到存储 dtype" k=warn
n r 2 0 "reference" "float64 softmax" k=pass
n d 2 1.2 "device" "分块 · fp32 累加" k=warn
n c 3 0.6 "check_close" "tolerance_for(dtype, n)" k=vio
e s -> x
e x -> r
e x -> d
e r -> c
e d -> c
:::

关键一步是 **先舍入再分发**：两边看到的是同一组 fp16 / bf16 数，差异只来自计算，不来自输入表示。忘了这一步，fp16 的 case 会因为“参考用的是没舍入的输入”而大面积误报。

## ② 容差：每一项都有来源

[[here:solutions/l5_differential.py@L67-L77|tolerance_for（67–77 行）]]：

```text
rtol = 2·eps(dtype)            ← 输出舍入到存储 dtype（半个 ulp，留一倍余量）
     + 4·eps32·√n              ← 长度 n 的 fp32 求和（随机游走）
atol = max(eps(dtype) / n, 1e-7)   ← softmax 平均值是 1/n，接近 0 的概率看绝对误差
```

| dtype | n = 65 | n = 1024 |
|---|---|---|
| float32 | rtol 4.1e-6，atol 1e-7 | rtol 1.5e-5，atol 1e-7 |
| float16 | rtol 1.96e-3，atol 1.5e-5 | rtol 1.97e-3，atol 9.5e-7 |
| bfloat16 | rtol 1.56e-2，atol 1.2e-4 | rtol 1.56e-2，atol 7.6e-6 |

低精度时第一项占绝对主导；fp32 时第二项才是主角，n 越长越宽。这张表本身就是 PR 里“tolerance rationale”的好素材。

::: callout analogy 考试的允许误差
允许误差 = 尺子的刻度（输出 dtype 的 eps）+ 量的段数带来的累积误差（√n）。对一份用毫米刻度的尺子量了 1024 段的报告，你不会因为差了 0.5 mm 就判它错；但对一份用 0.01 mm 刻度、只量了 1 段的报告，差 0.5 mm 就该追查了。
:::

## ③ 三个 bug，三种结果

| bug | 机制 | 第一次被抓 / 数量（60 个 case，seed 0） | 最小失败 |
|---|---|---|---|
| `tail` | 只对完整的 64 块求和，丢掉 n % 64 列 | 26 个失败 | n = 97，fp32，max_abs 0.54 |
| `fp16_acc` | 求和累加器用存储 dtype | 11 个失败，max_abs 都 < 1e-2 | n = 63，fp32 输入，4e-5 |
| `no_max_sub` | 直接 exp(x)，不先减最大值 | scale = 3：**0 个**；scale = 40：44 个 | 溢出 → 0 / NaN |

- `tail` 是“结构性”错误：误差是 O(1)，任何容差都能抓住，**前提是 n 不是 64 的倍数**。edge_sizes 里的 65、97、129 就是为它准备的。
- `fp16_acc` 是“精度”错误：误差小但系统性，只有紧容差（fp32 那一行）才看得见。fp16 输入的 case 反而能过——容差已经宽到 2e-3。
- `no_max_sub` 是“数值稳定性”错误：exp(x) 在 fp32 下 x > 88.7 就溢出。N(0, 3) 的输入最大也就 10 左右，永远到不了；必须专门构造大 logits（真实模型里 attention 分数很容易到这个量级）。

## ④ 输出：一个 Rust 测试骨架

[[here:solutions/l5_differential.py@L108-L124|emit_rust_test（108–124 行）]] 把最小失败（`1 × 97`、fp32、seed）和容差理由写进注释：

```rust
#[test]
fn softmax_matches_reference_float32_1x97() {
    // minimized by l5_differential; input seed = ...
    // tolerance: float32: output rounding 2*eps=2.38e-07 + fp32 sum over n=97: ...
    let mut cx = Graph::new();
    let x = cx.tensor(vec![1, 97], DType::F32);
    let y = x.softmax(1).output();
    ...
    assert_close(&device, &reference, /*rtol=*/..., /*atol=*/...);
}
```

`random_vec_seeded`、`run_reference`、`run_cuda`、`assert_close` 是占位名，提交前要换成 `crates/luminal_cuda_lite/tests/` 里实际使用的 helper（先读 [[luminal:crates/luminal_cuda_lite/tests/cublaslt_precision_matches.rs|cublaslt_precision_matches.rs]] 这类文件学写法）。

## ⑤ Luminal 里的两个 runtime

| | ReferenceRuntime | CUDA（luminal_cuda_lite） |
|---|---|---|
| 角色 | 定义语义 | 被测对象 |
| 窄浮点 | [[luminal:crates/luminal_reference/src/narrow_float.rs|narrow_float.rs]] 精确模拟 | 硬件 |
| TF32 | [[luminal:crates/luminal_reference/src/tf32.rs|tf32.rs]] 模拟 | cuBLASLt 可能用 TF32 |
| 调试 | — | `LUMINAL_CUDA_CHECK_NONFINITE_INTERNAL=1` 检查中间 buffer 的 NaN / inf |

Reference 模拟了低精度，所以对拍时可以用比“float64 参考”更紧的容差——这是 Luminal 测试设计里值得学的一点。

## ⑥ 源码与资料索引

- [[luminal:src/frontend/unary.rs@pub fn softmax|Luminal softmax]] · [[luminal:crates/luminal_reference/src/harness.rs|reference/harness.rs]] · [[luminal:crates/luminal_reference/src/runtime.rs|reference/runtime.rs]]
- [[luminal:crates/luminal_cuda_lite/tests/cublaslt_tf32.rs|cublaslt_tf32.rs]] · [[luminal:crates/luminal_cuda_lite/tests/cublaslt_precision_matches.rs|cublaslt_precision_matches.rs]]
- K3 的 `check_close` / `edge_sizes`

## ⑦ 自测

::: answer 为什么 fp16_acc 的最小失败是 fp32 输入，而不是 fp16 输入？
因为容差跟着 dtype 走：fp16 输入的 case 容差约 2e-3，fp16 累加带来的 4e-5 级误差完全在容差内；fp32 输入的 case 容差约 4e-6，同样的误差就超了。这提醒我们：**最容易暴露精度 bug 的是高精度的测试用例**。
:::

::: answer 如果 CUDA 上 1% 的元素超出容差，但都是概率接近 0 的位置，算 bug 吗？
先看 atol：如果 atol 比这些元素的绝对误差还小，可能是 atol 定得太严（softmax 里 1e-30 的概率差 1e-35 没有意义）。但也可能是真问题，比如 underflow 处理不同（flush-to-zero）。正确做法是在报告里给出这些元素的值和误差，说明为什么调整 atol 是合理的——而不是悄悄放宽容差让测试变绿。
:::
