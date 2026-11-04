# ai-compiler-drills · Triton + Luminal 双项目 GPU compiler contributor 训练

把 14 周的双项目 roadmap 扩写成可以动手做的题：16 道（约 15 小时的练习 + 两个实践页），每题有背景、要用到的知识、React-Flow 风格的图、类比、题干、验收测试、三级提示、参考解，以及一页单独的讲解（`hints/`）。

- **共同基础 K1–K4**：GEMM 资源账（occupancy、wave、roofline）· benchmark 方法 · 数值正确性（ULP、容差）· fusion 取舍
- **Triton T1–T5**：四层 IR 解剖 · 复刻 coalesce pass（输出与 `coalesce.mlir` 的 CHECK 逐字一致）· autotuner · 最小复现（ddmin）· 自己写 FileCheck
- **Luminal L1–L5**：primitive graph · 从零写 e-graph 与 equality saturation · genome 搜索与 finalists · 移植 `plan_slab` arena · Reference vs CUDA 差分测试
- **实战**：X1 同一个 fused MLP 两边对照 · P1 第一笔贡献（以 Luminal #370 演练）· 14 周计划与进度勾选

**在线阅读：https://kzhou003.github.io/ai-compiler-drills/** · [PDF 版（提示、参考解、答案全部展开）](ai-compiler-drills.pdf) · 本地打开 `index.html` 亦可；每题的讲解页在 `hints/index.html`。

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy markdown-it-py
./check.sh k1                 # 测 mine/k1_gemm_model.py（你的实现）
./check.sh k1 --solutions     # 测参考解
./check.sh all --solutions    # 全部（约 100 个测试，纯 Python，不到 1 秒）

.venv/bin/pip install egglog && .venv/bin/python examples/l2_egglog_experiment.py   # L2 的 egglog 实验

python3 build/sources.py fetch            # 按固定 commit 下载被引用的源码并校验全部链接（需要 gh）
.venv/bin/python build/build.py           # 生成 index.html
.venv/bin/python build/build_hints.py     # 生成 hints/*.html
.venv/bin/python build/gen_stubs.py       # 从 solutions/ 生成缺失的 mine/ 桩文件（不覆盖已有文件）
python3 build/make_pdf.py                 # 用 headless Chrome 导出 ai-compiler-drills.pdf
```

| 目录 | 内容 |
|---|---|
| `content/` | index.html 的各节 |
| `hints/src/` | 每题讲解页的源文件 |
| `mine/` · `solutions/` · `tests/` | 你的实现 · 参考解 · 验收测试 |
| `fixtures/t1/` | T1 用的四层 IR 样例（按真实 Triton 输出格式整理并删减） |
| `build/sources/pins.txt` | 源码链接固定的 commit：Triton、Luminal、egglog、egglog-python、PyTorch、Triton-Puzzles |

所有练习都能在没有 GPU 的机器上验收；每题末尾有一节“在 RTX 3070 / Luminal 里对照”，把纯 Python 的结论和真实编译器的输出对上。
