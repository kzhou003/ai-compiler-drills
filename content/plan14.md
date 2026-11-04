lede: 14 周的双项目计划：前 8 周学习并建立复现能力，第 9–12 周各争取一笔小而真实的贡献，第 13 周起选一个深水区、形成长期 ownership。每周都对应这里的题目和一个验收标准。
# 14 周计划与进度

## 一张图

::: flow 14 周
cfg cw=178 nw=160 rh=92
n w1 0 0 "W1–W2" "build · 环境 · frontend" k=acc
n w3 1 0 "W3–W4" "T1 L1 · T2 L2" k=warn
n w5 2 0 "W5–W6" "L3 · GEMM benchmark" k=warn
n w7 3 0 "W7–W8" "T5 L4 · T4 L5" k=vio
n w9 3 1.2 "W9" "issue 筛选" k=teal
n w10 2 1.2 "W10–W11" "Triton PR · Luminal PR" k=pass
n w12 1 1.2 "W12" "review" k=pass
n w13 0 1.2 "W13–W14" "深水区 · ownership" k=dark
e w1 -> w3
e w3 -> w5
e w5 -> w7
e w7 -> w9
e w9 -> w10
e w10 -> w12
e w12 -> w13
g -0.1 -0.25 3.95 0.85 "学习 + 复现能力" k=acc
g -0.1 0.95 3.95 2.05 "贡献 + 定方向" k=pass
:::

## 每周

| 周 | Triton | Luminal | 本仓库 | 验收 |
|---|---|---|---|---|
| 1 | fork，记录 SHA / 环境，跑通简单 kernel | fork，跑通 ReferenceRuntime 示例；尝试 CUDA build，失败保留日志 | basics_env，K2 | 两仓库都有可复现的 setup notes |
| 2 | vector add → TTIR | matmul graph → HLIR / logical graph | basics_triton、basics_luminal | 两张编译流程图 |
| 3 | operation、type、shape、canonicalization | primitive ops、shape tracker、graph traversal | **T1**、**L1**、K1 | 解释两边 IR 分别表达什么 |
| 4 | blocked layout、layout conversion | e-class、rewrite、saturation、extraction | **T2**、**L2** | 低层数据布局 vs 高层等价搜索的区别 |
| 5 | `tl.dot` → MMA / PTX | 候选生成、cost、search budget | **L3**、K4 | GEMM codegen 与 graph search 对照笔记 |
| 6 | GEMM benchmark | 同硬件、同 dtype / shape 的 graph / execute 时间 | **T3**、**X1** | benchmark 表 + 测量方法说明 |
| 7 | lit / FileCheck、Python tests | selected plan → CUDA launch | **T5**、**L4** | 两个项目各一条完整调用链 |
| 8 | 最小复现 | differential test | **T4**、**L5**、K3 | 两边各一个可复现 correctness case |
| 9 | 按模块、硬件、复现性、已有 PR、维护者响应筛 issue | 同左；#370 环境验证 | **P1** | 每个项目一个小而明确的任务 |
| 10 | 第一笔贡献：correctness / test / lowering bug，IR diff + 最小复现 | — | T4、T5 | PR 或已获反馈的 draft |
| 11 | — | 第一笔贡献：CUDA build 兼容、backend correctness 或 benchmark test | L5、P1 | PR 或维护者确认的任务 |
| 12 | 回应 review，拆分 / 重构，补 CI / format / test | 同左；性能 PR 给稳定 before / after | K2 | 至少一个 PR 进入实质 review |
| 13 | 深水区候选：layout conversion、dot lowering、register / smem 行为 | kernel selection、search cost、execution planning | — | 一页 design note |
| 14 | 总结 review、理解缺口、benchmark 结果 | 同左 | — | 明确的长期 ownership 方向 |

## 进度追踪

勾选状态保存在这个浏览器里（和题目的“标记完成”一样）。

::: raw
<div class="checklist" id="plan-checklist">
<label><input type="checkbox" data-k="env"> 记录 GPU、driver、CUDA Toolkit、Python、PyTorch、Rust 版本</label>
<label><input type="checkbox" data-k="tri-build"> Triton 本地 build / kernel execution 成功</label>
<label><input type="checkbox" data-k="lum-ref"> Luminal ReferenceRuntime 示例成功</label>
<label><input type="checkbox" data-k="lum-cuda"> Luminal CUDA backend build / execution 已验证或记录阻塞原因</label>
<label><input type="checkbox" data-k="tri-flow"> 画出 Triton frontend → TTIR → TTGIR → LLVM → PTX</label>
<label><input type="checkbox" data-k="lum-flow"> 画出 Luminal graph → logical IR → e-graph / search → LLIR → backend</label>
<label><input type="checkbox" data-k="t1t2"> 完成 Triton IR dump 与 layout 练习（T1、T2）</label>
<label><input type="checkbox" data-k="l2l3"> 完成 Luminal e-graph 与 search cost 练习（L2、L3）</label>
<label><input type="checkbox" data-k="harness"> 建立同一 workload 的 benchmark harness（K2、X1）</label>
<label><input type="checkbox" data-k="repro"> 两边各完成一个 correctness reproducer（T4、L5）</label>
<label><input type="checkbox" data-k="tasks"> 分别确认 Triton / Luminal 的贡献任务</label>
<label><input type="checkbox" data-k="tri-pr"> 提交 Triton PR 或 draft PR</label>
<label><input type="checkbox" data-k="lum-pr"> 提交 Luminal PR 或 draft PR</label>
<label><input type="checkbox" data-k="review"> 至少一个 PR 进入实质 review</label>
</div>
<script>
(function () {
  var KEY = "acd-plan";
  var st = {};
  try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  document.querySelectorAll("#plan-checklist input").forEach(function (b) {
    b.checked = !!st[b.dataset.k];
    b.addEventListener("change", function () {
      st[b.dataset.k] = b.checked;
      try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {}
    });
  });
})();
</script>
:::

::: callout analogy 先学开车，再上高速，最后选一条常跑的路线
前 8 周是驾校：每样操作（IR、layout、e-graph、搜索、测试）都要亲手做一遍，能稳定复现。第 9–12 周是上路：挑一条短而熟悉的路（小 issue）开一趟，接受教练（reviewer）的纠正。第 13 周以后，选一条你会天天跑的路线（一个模块），把它跑熟——长期 ownership 就是“这条路上的事大家会来问你”。
:::

## 最后的执行建议

前 8 周以学习和建立复现能力为主；第 9–12 周分别争取一笔小而真实的贡献；第 13 周之后，依据维护者反馈和实际兴趣，选一个项目作为主攻，另一个维持每月一次的 issue / PR 参与。
