title: T4 讲解：从一个随机失败到十行回归测试
lede: 跟着参考解完整走一遍：seed=0 的 fuzz 第一个就撞到 [288, 65] fp16；shrink_case 用 24 次检查把它缩到 [1, 65] fp32 连续；ddmin 用 41 次检查从 20 个元素里找出 [3, 17]。每一步都讲清楚“为什么这样缩”，以及最后生成的测试为什么能直接提交。

::: callout info 怎么读这一页
对照 [[here:solutions/t4_minimize.py|t4_minimize.py]]。被测的 `buggy_row_sum`（[[here:solutions/t4_minimize.py@L131-L141|131–141 行]]）：`n ≤ 64` 时整行求和；`n > 64` 时只走完整的 64 列块，丢掉 `n % 64` 列尾巴。
:::

## ① 全流程

::: flow fuzz → shrink → test
cfg cw=215 nw=185
n fz 0 0 "fuzz(seed=0)" "第 1 个 case 就失败" k=acc
n c 1 0 "[288, 65] fp16 连续" "差了最后 1 列" k=warn
n sh 2 0 "shrink_case" "24 次检查" k=vio
n m 3 0 "[1, 65] fp32 连续" "最小" k=pass
n t 3 1.2 "to_pytest" "可直接提交" k=teal
e fz -> c
e c -> sh
e sh -> m
e m -> t
:::

## ② fuzz：边界尺寸是关键

[[here:solutions/t4_minimize.py@L89-L106|random_case / fuzz（89–106 行）]] 的尺寸只从 `edge_sizes(64, 256)` = `[1, 63, 64, 65, 97, 129, 288]` 里抽。纯随机的 `randint(1, 4096)` 也能撞到这个 bug（大多数 n 都不是 64 的倍数），但会撞到像 `[2731, 1509]` 这样的输入，缩小要多花很多次；如果 bug 只在 `n = 65` 这样的特定值出现，随机尺寸几乎永远撞不到。**边界尺寸是把“可能出错的地方”直接放进样本**。

[[here:solutions/t4_minimize.py@L124-L128|differential_check（124–128 行）]] 用 float64 参考和 K3 的容差：`tolerance(dtype, n, c=4)`，atol 再乘 `√n`。少了一列 N(0, 1) 的值，误差约为 1，远大于任何 dtype 的容差，所以这个 bug 在所有 dtype 上都可见——这也是 shrink 能一路简化到 fp32 的原因。

## ③ shrink_case：一维一维地缩

[[here:solutions/t4_minimize.py@L63-L86|shrink_case（63–86 行）]]：

| 轮次 | 动作 | 结果 |
|---|---|---|
| 1 | rows：288 → 144 → 72 → 36 → 18 → 9 → 4 → 2 → 1（每次减半仍失败） | [1, 65] |
| 1 | cols：65 → 32？通过（单块路径正确）；64？通过；停在 65 | [1, 65] |
| 1 | dtype：float32 仍失败 | fp32 |
| 1 | contiguous 已经是 True | — |
| 2 | 什么都没变 → fixpoint | [1, 65] fp32 连续 |

[[here:solutions/t4_minimize.py@L45-L60|shrink_int（45–60 行）]] 先试减半、再试减一：对 rows 这种单调的条件，减半是对数步；对 cols（`n > 64 且 n % 64 ≠ 0`），减半到 32 会通过，于是改为减一，65 → 64 也通过，停下。

::: callout analogy 剪头发
先大刀阔斧剪掉一半（减半）；剪过头了（测试通过了）就退回来，改成一点一点修（减一）；直到再剪一刀就不像样了（不再失败）。最后得到的“发型”就是最小复现：每一个保留下来的细节都是必要的。
:::

## ④ ddmin：缩一个列表

[[here:solutions/t4_minimize.py@L22-L42|ddmin（22–42 行）]] 对 `range(20)`、条件“同时含有 3 和 17”：

| n | 切成 | 结果 |
|---|---|---|
| 2 | [0..9] [10..19] | 都不单独失败；块数 = 2 不试补集 → n = 4 |
| 4 | [0..4] [5..9] [10..14] [15..19] | 单块都不失败；去掉 [5..9] 仍失败 → 保留其余 15 个，n = 3 |
| … | 继续去掉不含 3、17 的块 | … |
| 结束 | [3, 17] | 41 次检查 |

ddmin 的结果是 **1-minimal**：去掉任何一个元素都不再失败，但不保证全局最小（对“含有 3 和 17”来说两者一致）。它最常用在 IR 上：把出错前的 TTGIR 按行（或按 op）交给 ddmin，`fails` = “`triton-opt` 跑这段 IR 是否还崩溃 / 输出是否还错”。测试 `test_ddmin_ir_lines` 就是一个缩到“一个 tt.dot + 一个 convert_layout”的玩具版本。

## ⑤ to_pytest：生成的测试必须真的能跑

```python
def test_row_sum_regression():
    # minimized by t4_minimize.shrink_case; seed=0
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 65)).astype(np.float32)
    np.testing.assert_allclose(row_sum(x), x.astype(np.float64).sum(axis=1), rtol=1e-3, atol=1e-3)
```

测试 `test_to_pytest_runs` 把它 `exec` 两次：用 buggy 实现必须失败、用正确实现必须通过。“修复前失败、修复后通过”是一个回归测试唯一的价值，生成器也要被这样检验。

## ⑥ 写成 Triton 的风格

在 [[triton:python/test/unit/language/test_core.py|test_core.py]] 里，同类测试通常是：

```python
@pytest.mark.parametrize("M, N", [(1, 65), (3, 129)])
@pytest.mark.parametrize("dtype_str", ["float32", "float16"])
def test_row_sum_tail(M, N, dtype_str, device):
    x = torch.randn(M, N, dtype=getattr(torch, dtype_str), device=device)
    ...
    torch.testing.assert_close(out, x.float().sum(1), rtol=..., atol=...)
```

形状参数里保留最小案例，再加一两个“同一类”的案例（另一个非整除长度、多行），防止修复只对 65 有效。

## ⑦ 源码与资料索引

- [[triton:CONTRIBUTING.md|Triton CONTRIBUTING.md]] · [[triton:python/test/unit/language/test_core.py|test_core.py]]
- [Zeller & Hildebrandt：ddmin](https://www.st.cs.uni-saarland.de/papers/tse2002/) · [Hypothesis：shrinking](https://hypothesis.readthedocs.io/)
- [[triton:README.md@MLIR_ENABLE_DUMP|MLIR_ENABLE_DUMP：拿到每个 pass 前后的 IR]]

## ⑧ 自测

::: answer 如果 bug 只在 bf16 出现，shrink_case 会把 dtype 改成 fp32 吗？
不会。它只在“换成更简单的 dtype 后仍然失败”时才换。bf16-only 的 bug 会让 fp32、fp16 的尝试都通过，dtype 就保留为 bf16——这本身就是一条重要的诊断信息：问题多半在 bf16 的转换或累加路径上。
:::

::: answer shrink_int 假设了什么？什么时候会找不到真正的最小值？
它假设“失败”在 n 附近大致单调：从 n 往下一步一步减，第一个通过的位置就是边界。如果失败集合是零散的（比如只有 n = 65 和 n = 1000 失败），从 1000 开始减半到 500 通过、减一到 999 也通过，就会停在 1000。所以 fuzz 时从边界尺寸里抽样很重要：起点越接近“本质”，缩出来的结果越好。
:::
