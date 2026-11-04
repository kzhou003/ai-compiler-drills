title: K2 讲解：一个可以信任的 benchmark harness
lede: GPU launch 是异步的、第一次调用要编译、每次测量都有噪声。这一页用 FakeClock 把 GPU 换成一个可控的时钟，逐段讲 bench、compile_vs_run、outliers 和 compare 为什么要那样写，并给出“8% 加速到底算不算数”的判断方法。

::: callout info 怎么读这一页
对照 [[here:solutions/k2_bench_harness.py|k2_bench_harness.py]]。harness 的 `timer` 和 `sync` 都是参数：测试里传 `FakeClock`，真机上传 `time.perf_counter` 和 `torch.cuda.synchronize`。同一份代码，两种时钟。
:::

## ① 异步：不 sync 测到的是什么

::: flow CPU 和 GPU 两条时间线
cfg cw=210 nw=180
n c1 0 0 "CPU：launch" "几微秒就返回" k=acc
n c2 1 0 "CPU：读时钟" "没 sync：GPU 还没算完" k=warn
n g1 0 1.2 "GPU：排队" "" k=ghost
n g2 1 1.2 "GPU：执行 kernel" "真正的时间在这里" k=pass
n g3 2 1.2 "完成" "" k=ghost w=0.6
n c3 2 0 "sync 之后读时钟" "包含了 kernel" k=teal
e c1 -> c2
e g1 -> g2
e g2 -> g3
e g3 -> c3 dash
e c1 -> g1 dash
:::

`kernel[grid](...)` 只是把工作放进 stream 就返回。所以 [[here:solutions/k2_bench_harness.py@L42-L59|bench（42–59 行）]] 每个样本前后各 sync 一次：前面那次把上一轮遗留的工作排空，后面那次等这一轮的工作完成。warmup 之后再 sync 一次，避免 warmup 的尾巴落进第一个样本。测试数的就是这个次数：`1 + 2·repeats`。

::: callout analogy 寄快递
launch 像把包裹交给快递员：你一转身就“完成”了，但包裹还在路上。不 sync 就计时，等于把“交给快递员用了多久”当成“送到用了多久”。sync 就是等到对方签收再看表。
:::

## ② FakeClock：把 GPU 换成剧本

[[here:solutions/k2_bench_harness.py@L98-L114|FakeClock（98–114 行）]] 每次 `work()` 按剧本推进时间。测试 `test_bench_skips_warmup_and_syncs` 的剧本是 `[10, 10, 10, 1, 1, ...]`：前三次（warmup，模拟 JIT 编译）每次 10 s，之后每次 1 s。正确的 harness 报告 median = 1.0、max = 1.0；如果你把 warmup 也计进去，max 会是 10。

## ③ 统计量：为什么是 median 和 IQR

`summarize([1, 2, 3, 4, 100])`（[[here:solutions/k2_bench_harness.py@L29-L39|summarize（29–39 行）]]）：

| 统计量 | 值 | 受离群值影响？ |
|---|---|---|
| mean | 22.0 | 严重 |
| median | 3.0 | 不受影响 |
| q1 / q3 / IQR | 2.0 / 4.0 / 2.0 | 不受影响 |
| max | 100 | 就是离群值本身 |

`outliers`（[[here:solutions/k2_bench_harness.py@L77-L81|77–81 行]]）用 Tukey fences：`[q1 − 1.5·IQR, q3 + 1.5·IQR]` 之外的样本。`[1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 5.0]` 里只有 5.0（下标 6）被标出。看到离群样本先别删，先查原因：别的进程、GPU 降频、第一次访问某块内存的 page fault。

## ④ compile_vs_run：把一次性成本拆出来

[[here:solutions/k2_bench_harness.py@L62-L74|compile_vs_run（62–74 行）]]：

```text
t0 ─ make_fn()（编译 / 搜索 7 s）─ fn()（1 s）─ sync ─ first = 8 s
                                  之后 bench(fn) → steady median = 1 s
                                  one_time_cost = 8 − 1 = 7 s
```

对 Triton，`make_fn` 是第一次 `kernel[grid](...)`（JIT + autotune）；对 Luminal，是 `cx.compile(...)`（e-graph 饱和 + 搜索）。L3 的 break-even 就用这两个数。

## ⑤ compare：只在差异大于噪声时说“更快”

[[here:solutions/k2_bench_harness.py@L84-L95|compare（84–95 行）]]：

| base | new | IQR 区间 | 结论 |
|---|---|---|---|
| [10, 10.5, 11, 11.5, 12] | [5, 5.2, 5.5, 5.8, 6] | [10.5, 11.5] vs [5.2, 5.8] 不重叠 | faster，speedup 2.0 |
| [10, 10.5, 11, 11.5, 12] | [9.5, 10, 11, 11.8, 12.5] | [10.5, 11.5] vs [10, 11.8] 重叠 | within noise |

IQR 不重叠是一个保守、好解释的标准。更正式的做法是 bootstrap 置信区间或 Mann–Whitney U 检验；写 PR 时至少给出两边的 median 和 IQR，让 reviewer 自己判断。

::: flow 一个性能 PR 的证据链
cfg cw=210 nw=180
n env 0 0 "环境快照" "GPU · Toolkit · SHA" k=acc
n base 1 0 "before：bench" "median · IQR · n" k=warn
n new 1 1.2 "after：bench" "同样的参数" k=warn
n cmp 2 0.6 "compare" "faster / noise" k=vio
n pr 3 0.6 "PR 描述" "表格 + 命令" k=pass
e env -> base
e env -> new
e base -> cmp
e new -> cmp
e cmp -> pr
:::

## ⑥ 源码与资料索引

- [[triton:python/triton/testing.py@def do_bench|triton.testing.do_bench]]：CUDA event 计时、按目标时长估计重复次数、每次之间清 L2
- [[triton:python/triton/testing.py@def do_bench_cudagraph|do_bench_cudagraph]]：把多次调用捕获进一个 CUDA graph，去掉 launch 开销
- [[triton:python/triton/runtime/autotuner.py@L184|autotuner：quantiles=(0.5, 0.2, 0.8)]]
- Luminal README：`LUMINAL_CUDA_PROFILE_RECAPTURE`（prepare / allocate / launch / sync 分解）
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)：看 CPU 和 GPU 时间线上的空隙

## ⑦ 自测

::: answer 你的 harness 测出 12 µs，do_bench 测出 4 µs，谁错了？
可能都没错。`perf_counter` + `synchronize` 测的是端到端（Python 开销 + launch + kernel + 同步），`do_bench` 用 CUDA event 只量 GPU 上的时间。小 kernel 的 launch 和 Python 开销常常就有几微秒。报告时写清楚测的是哪一个；比较两个实现时用同一种方法。
:::

::: answer 为什么 do_bench 要在每次测量之间清 L2？
否则第二次起输入数据都在 L2 里，测出来的是“热缓存”性能，比真实使用（数据刚从别的 kernel 写出来，或者太大放不进 L2）乐观。是否清缓存取决于你想回答的问题：单独评估 kernel 时清，模拟一个融合流水线时不一定清。
:::
