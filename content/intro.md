lede: 先看这一节：为什么 Triton 和 Luminal 适合一起学、怎么分配精力、每道题怎么做，以及做完之后你应该能拿出什么。
# 路线与用法

## 目标：一个能被识别的技术定位

最终想形成的公开定位是：**GPU Compiler / Inference Performance Engineer — kernel lowering, execution planning, and performance correctness.** 具体来说，做完这套题和后面的贡献周期，你应该能说出两句有分量的话：

1. **Triton**：我能解释并优化一个 GPU kernel 从高层 tensor IR 到硬件指令的生成过程（TTIR → TTGIR → LLVM IR → PTX，layout、coalescing、`tl.dot` → MMA）。
2. **Luminal**：我能解释并改进一个 inference compiler 如何搜索、选择和组织 kernel 来执行整个模型（primitive graph → e-graph → 实测搜索 → 执行计划 → CUDA backend）。

两个能力叠加，比只会写 kernel 或只会读 graph compiler 更有区分度。**成功不以 PR 数量衡量**：理想结果是 Triton 有一笔 compiler 层面的实质贡献，Luminal 有一笔 backend 或性能 / 正确性贡献，并且你能用 benchmark 和 IR / profile 解释改动的价值。

## 两个项目是什么关系

::: flow 一个模型从 Python 到 GPU：两个项目各管一段
cfg cw=255 nw=190 rh=96
n model 0 0 "模型 / 计算图" "relu(x @ w + b)" k=acc
n lum 1 0 "Luminal：graph compiler" "e-graph + 实测搜索" k=vio
n plan 2 0 "执行计划" "选哪个实现、怎么排、内存" k=teal
n lib 1 1.3 "kernel 库" "cuBLASLt · FlashInfer" k=ghost
n tri 2 1.3 "Triton：kernel compiler" "一个 kernel → PTX" k=pass
n gpu 3 0.65 "GPU" "SM86 · RTX 3070" k=dark w=0.8
e model -> lum
e lum -> plan
e plan -> gpu
e lib -> lum "候选实现" dash
e tri -> lib "生成" dash
e tri -> gpu dash
cap: Triton 回答“一个 kernel 怎样生成、怎样跑快”；Luminal 回答“整张图该选哪些实现、怎样组织执行”。Luminal 不是另一个 Triton：它的候选里有现成的 kernel 库（README：cuBLASLt、FlashInfer），你在 Luminal 的贡献不一定是写 kernel。
:::

::: callout analogy 菜谱作者和餐厅主厨
Triton 像一个**把菜谱翻译成精确操作步骤的厨师**：拿到“炒一盘青菜”，决定火候、下锅顺序、每个帮厨（线程）切哪一段（layout）。Luminal 像**餐厅主厨排一整桌菜**：这道菜自己做还是用现成的半成品（cuBLASLt），哪几道合并在一个锅里（fusion），先上哪道、灶台怎么轮用（执行计划和内存）。主厨会试做几种排法、掐表选最快的（实测搜索），而不是只凭经验。
:::

| 共同问题 | Triton 视角 | Luminal 视角 | 本套题 |
|---|---|---|---|
| GEMM 怎么实现 | tile、layout、MMA lowering、pipeline | 选哪个 GEMM 实现 / 库 / 计划 | K1、T2、T3、L3 |
| fusion | 一个 kernel 内融合，寄存器代价 | 图级 rewrite、候选和计划搜索 | K4、L2、L3 |
| layout | thread / warp / shared memory layout | tensor layout、buffer 排布 | T2、L4 |
| 怎么测性能 | 单 kernel microbenchmark、PTX、Nsight Compute | graph latency、search 开销、端到端 | K2、T3、L3、L4 |
| 怎么保证正确 | compiler transformation、kernel 数值 | graph rewrite、backend、reference-vs-device | K3、T4、T5、L5 |

## 路线

```text
Part 0  基础：环境 · GPU 执行模型 · Triton 编译器全景 · Luminal 全景 · 两边对照
   │
共同基础  K1–K4  GEMM 资源账 · benchmark 方法 · 数值正确性 · fusion 取舍
   │
双主线 ├─ T  Triton   T1 IR 四层 · T2 coalesce/layout · T3 autotune · T4 最小复现 · T5 FileCheck
       └─ L  Luminal  L1 primitive graph · L2 e-graph · L3 搜索 · L4 计划与 arena · L5 对拍
   │
实战   X1 同一个 fused MLP 两边各量一遍 · P1 第一笔贡献
   │
14 周计划与进度 · 附录（PR 模板、资源）
```

**精力分配**：前 6 周 Triton 约 60%、Luminal 约 40%；之后按实际 issue / review 机会调整。不要同时在两个项目里做两个大 feature。

## 每道题怎么做

1. 读“背景”和“要用到的知识”，看图和类比。
2. 在 `mine/<题号>_*.py` 里实现（已给出数据结构、辅助函数和每个函数的文档字符串）。
3. 跑 `./check.sh <题号>`。测试里的期望值都是**按定义手算或对照真实源码得出的**，卡住时先读测试。
4. 实在卡住，依次展开三级提示；最后才看参考解。
5. 打开 [hints/](hints/index.html) 里这道题的单独讲解页：参考解逐段讲、更多图和例子、钉死版本的源码链接。

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy markdown-it-py
./check.sh k1                # 测你的 mine/k1_gemm_model.py
./check.sh k1 --solutions    # 测参考解
./check.sh all --solutions   # 全部（约 100 个测试，不到 1 秒）
```

::: callout info 为什么全部练习都是纯 Python
Triton 没有 macOS wheel，Luminal 的 CUDA 路径要按仓库当前依赖验证。所以每道题都把一个**真实的编译器机制**缩成能在任何机器上测试的 Python：T2 复刻 Triton 的 coalesce pass（输出和 `coalesce.mlir` 的 CHECK 逐字一致），T5 自己写 FileCheck，L2 从零写 e-graph，L4 移植 Luminal 的 `plan_slab`。每题末尾都有“在 RTX 3070 上对照真实结果”的一节，把纯 Python 结论和真编译器的输出对上。
:::

::: answer 自测：为什么说“Luminal 并不是另一个 Triton”？
Triton 编译的是**单个 GPU kernel**（输入是一个 `@triton.jit` 函数，输出是 PTX/cubin）。Luminal 编译的是**整张计算图**：它把模型拆成 primitive ops，用 e-graph 表示等价改写，再通过实测搜索为每一部分挑选实现——这些实现可以是它生成的 kernel，也可以是 cuBLASLt、FlashInfer 等现成库。所以两边的“编译”发生在不同的层次，贡献面也不同。
:::
