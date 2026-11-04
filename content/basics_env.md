lede: 先把两个仓库都跑起来，并把环境记清楚：GPU、driver、CUDA Toolkit、Python / PyTorch / Triton、Rust。后面每一个 benchmark、每一个 issue 复现都要引用这份快照。
# 统一实验环境

## 起点：RTX 3070（Ampere，SM86）

[[triton:README.md|Triton README]] 列出的 NVIDIA 支持从 Compute Capability 8.0 起，RTX 3070（SM86）在范围内。Luminal 的 CUDA 路径（`luminal_cuda_lite`）要按仓库当前依赖、CUDA Toolkit 和驱动实际验证——它的一个公开 issue 正是 **CUDA 13.x 下编译失败**（[luminal #370](https://github.com/luminal-ai/luminal/issues/370)，P1 会用它做第一笔贡献的候选）。

| 检查项 | 命令 | 用途 |
|---|---|---|
| GPU | `nvidia-smi` | 型号、compute capability、driver |
| CUDA Toolkit | `nvcc --version`、`echo $CUDA_HOME` | **不要**把 `nvidia-smi` 右上角的 “CUDA Version” 当成 Toolkit 版本 |
| Python 栈 | `python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"` | Triton build / test 与 reference |
| Rust | `rustc -V`、`cargo -V` | Luminal workspace 与 feature flags |
| Profiler | `nsys --version`、`ncu --version` | 区分 host overhead、kernel latency、memory / occupancy |

```bash
# 基础环境快照：贴进每一个 issue / PR / benchmark 报告
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version | tail -2
python --version
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
rustc -V && cargo -V
git -C triton rev-parse HEAD && git -C luminal rev-parse HEAD
```

::: flow 例：driver 版本 ≠ Toolkit 版本
cfg cw=265 nw=200
n smi 0 0 "nvidia-smi: CUDA 13.0" "driver 能支持的最高版本" k=warn
n drv 0 1.2 "driver 580.x" "内核模块 + libcuda" k=acc
n nvcc 1 0 "nvcc: release 12.8" "真正编译用的 Toolkit" k=pass
n tri 2 0 "Triton 自带 ptxas" "随 wheel 安装" k=teal
n lum 2 1.2 "Luminal build.rs" "找 CUDA_HOME 下的头文件" k=vio
e smi -> drv dash
e nvcc -> lum "头文件"
e drv -> tri "运行时"
e drv -> lum "运行时"
cap: driver 向下兼容旧 Toolkit，所以 `nvidia-smi` 显示 13.0 时，机器上的 Toolkit 完全可能是 12.x。issue #370 这类问题的第一步，就是把两者分开记录。
:::

::: callout analogy 打印机驱动和排版软件
driver 像打印机驱动：它说“我最高能理解 13.0 版本的文件格式”。Toolkit 像你的排版软件：它决定你实际生成的是哪个版本的文件。驱动新、软件旧没问题；但你说“我的排版软件是 13.0”只因为驱动这么写，就会把 bug 报告写错。
:::

## 两个仓库的最小跑通

```bash
# Triton（Linux + NVIDIA）：先用 wheel，需要改 compiler 时再源码 build
pip install triton           # 或：git clone https://github.com/triton-lang/triton && pip install -e .
python python/tutorials/01-vector-add.py

# Luminal：先 ReferenceRuntime（CPU），再 CUDA
git clone https://github.com/luminal-ai/luminal && cd luminal
cargo test -p luminal_reference          # 纯 CPU
cargo run --release -p luminal_cuda_lite --example llama3 --features device   # README 里的 CUDA 示例
```

Luminal [[luminal:README.md|README]] 的 Usage 一节就是最小例子：`Graph::new()` → `cx.tensor(...)` → `a.matmul(b).output()` → `cx.compile(ReferenceRuntime::default(), CompileOptions::default())` → `set_data` → `execute` → `get_f32`。注意 **`a.matmul(b)` 这一行什么都没算**，只是记录了一个节点（L1）。

::: callout warn CUDA build 失败时
保留完整日志，**先不要改依赖**。记录 Toolkit、driver、`cargo` 版本和 `git rev-parse HEAD`，再对照 issue 列表看是否已知问题。roadmap 的原话：“若失败，保留完整日志，不要先改依赖”。这份日志很可能就是你第一笔贡献的起点（P1）。
:::

建议用独立环境隔离 Triton 的 Python / CUDA 依赖和 Luminal 的 Rust / CUDA build。**不要为了同时跑两个项目而强行统一 CUDA Toolkit 版本**，先分别记录兼容矩阵。

## 本仓库自己的环境

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy markdown-it-py
./check.sh all --solutions                 # 约 100 个测试，全部纯 Python
python3 build/sources.py fetch             # 第一次构建页面前：按固定 commit 下载被引用的源码（需要 gh）
.venv/bin/python build/build.py            # 生成 index.html
.venv/bin/python build/build_hints.py      # 生成 hints/*.html
```

::: answer 自测：同事说“我机器是 CUDA 13，所以 Luminal 编不过”，你先问什么？
先问 `nvcc --version` 的输出和 `CUDA_HOME` 指向哪里。`nvidia-smi` 的 “CUDA Version” 是 driver 能支持的最高运行时版本，不是编译用的 Toolkit。只有确认实际编译用的是 13.x Toolkit，并在当前 main 上复现，才能把问题归到 #370 这类“CUDA 13 API 变化”的范畴。
:::
