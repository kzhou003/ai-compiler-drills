lede: 官方仓库与文档、论文、profiler 工具，以及本仓库的构建方式。所有源码链接都固定在 build/sources/pins.txt 里的 commit 上。

## 官方与核心资源

| 项目 | 链接 |
|---|---|
| Triton | [源码](https://github.com/triton-lang/triton) · [文档](https://triton-lang.org/) · [[triton:CONTRIBUTING.md|CONTRIBUTING.md]] · [[triton:python/tutorials/gluon/01-intro.py|Gluon 入门]] · [Triton Puzzles](https://github.com/srush/Triton-Puzzles) |
| Luminal | [源码](https://github.com/luminal-ai/luminal) · [[luminal:README.md|README]] · [Issues](https://github.com/luminal-ai/luminal/issues) · [PRs](https://github.com/luminal-ai/luminal/pulls) · [文档](https://docs.luminalai.com) |
| egglog | [egglog](https://github.com/egraphs-good/egglog) · [egglog-python](https://github.com/egraphs-good/egglog-python) · [egglog 论文](https://arxiv.org/abs/2304.04332) · [egg 论文](https://arxiv.org/abs/2004.03082) |
| NVIDIA | [Nsight Compute](https://docs.nvidia.com/nsight-compute/) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/) · [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) · [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) |
| LLVM | [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) · [lit](https://llvm.org/docs/CommandGuide/lit.html) · [MLIR](https://mlir.llvm.org/) |
| 论文 | [Triton（MAPL 2019）](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) · [Linear Layouts](https://arxiv.org/abs/2505.23819) · [ddmin](https://www.st.cs.uni-saarland.de/papers/tse2002/) |

## 本仓库

```text
content/        index.html 的各节（Markdown + ::: 容器 + flow 图 DSL）
hints/src/      每道题的单独讲解页源文件 → hints/*.html
mine/           你的实现（由 build/gen_stubs.py 从 solutions/ 生成）
solutions/      参考解
tests/          验收测试（./check.sh <题号>）
fixtures/t1/    T1 用的四层 IR 样例
examples/       egglog 小实验（L2）
build/          build.py · build_hints.py · sources.py（固定 commit 的源码链接）· gen_stubs.py
```

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy markdown-it-py
./check.sh all --solutions
python3 build/sources.py fetch            # 下载被引用的源码文件并校验所有链接（需要 gh）
.venv/bin/python build/build.py           # index.html
.venv/bin/python build/build_hints.py     # hints/
.venv/bin/python build/gen_stubs.py       # 重新生成 mine/ 里缺失的桩文件（已有的不会覆盖）
```

## 和 gpu-kernel-drills 的关系

[gpu-kernel-drills](../gpu-kernel-drills/index.html) 练的是**写 kernel**（CUDA、Triton、FlashInfer、CuTe）；这里练的是**读和改编译器**。两边有几处直接相通：那边的 G1（BlockedLayout 计算器）是这边 T2 的前置；那边的 T4（Triton GEMM）是这边 T3 / X1 的 kernel；那边的 FlashInfer 题对应 Luminal 搜索空间里的 FlashInfer 候选。
