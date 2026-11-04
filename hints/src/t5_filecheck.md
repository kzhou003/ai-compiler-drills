title: T5 讲解：FileCheck 是怎样“对答案”的
lede: 拿 coalesce.mlir 风格的一段输出，一条一条 CHECK 地走一遍参考解：pos 指针怎样前进、NOT 什么时候被检查、LABEL 怎样把输入切块、[[row_layout]] 这种变量怎样把前后两条 CHECK 绑在一起；最后讲怎样在 Triton 仓库里读、改、写一个 lit 测试。

::: callout info 怎么读这一页
对照 [[here:solutions/t5_filecheck.py|t5_filecheck.py]]，以及测试文件里的 `OUT`（一个 `@transpose` 函数和一个 `@other` 函数）。
:::

## ① 一次完整的匹配

检查串（来自 `test_real_coalesce_style_checks_pass`）：

```text
// CHECK: [[row:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16]
// CHECK: [[col:#.*]] = #ttg.blocked<{sizePerThread = [4, 1]
// CHECK-LABEL: @transpose
// CHECK: [[ptr:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[row]]>
// CHECK-NEXT: [[val:%[0-9]+]] = tt.load [[ptr]]
// CHECK-SAME: [[row]]>
// CHECK-NOT: tt.dot
// CHECK: tt.store {{.*}}, [[col]]>
```

| 步 | 指令 | 在哪里找 | 结果 |
|---|---|---|---|
| 0 | LABEL `@transpose` | 先扫一遍全部 LABEL | 块 A = 文件头 … `@transpose` 之前；块 B = `@transpose` 之后 |
| 1 | CHECK row | 块 A，从 0 开始 | 匹配第 1 行，`row = "#blocked"` |
| 2 | CHECK col | 块 A，从第 1 行之后 | 第 2 行，`col = "#blocked1"` |
| 3 | CHECK ptr | 块 B | `%8 = ttg.convert_layout ... #blocked>`，`ptr = "%8"` |
| 4 | NEXT | 下一行 | `%9 = tt.load %8 ...`，`val = "%9"` |
| 5 | SAME `[[row]]>` | 同一行、%8 之后 | `#blocked>` ✔ |
| 6 | NOT `tt.dot` | 挂起 | — |
| 7 | CHECK store | 从第 4 行之后 | `tt.store %11, %10 : ... #blocked1>`；此时检查 6 的区间：第 4 行末 … store 之前，没有 `tt.dot` ✔ |

::: flow pos 指针与挂起的 NOT
cfg cw=200 nw=170
n p1 0 0 "CHECK 匹配" "pos → 匹配末尾" k=acc
n n1 1 0 "CHECK-NOT" "挂起，不移动 pos" k=warn
n p2 2 0 "下一个 CHECK" "确定区间终点" k=acc
n chk 2 1.2 "检查 NOT" "[上次末尾, 这次开头)" k=vio
e p1 -> n1
e n1 -> p2
e p2 -> chk dash
:::

::: callout analogy 读书时用书签
pos 是一枚书签：每找到一条 CHECK，书签就夹到那里，下一条只能往后翻。NOT 像“这两枚书签之间不许出现某个词”——你得先放好下一枚书签，才知道要检查哪一段。LABEL 像章节：先把每一章的起点都找出来，每条 CHECK 只能在自己那一章里翻。
:::

## ② compile_pattern：三种片段

[[here:solutions/t5_filecheck.py@L46-L74|compile_pattern（46–74 行）]] 逐字符扫描：

| 片段 | 例子 | 生成的正则 |
|---|---|---|
| 字面 | `#ttg.blocked<{` | `\#ttg\.blocked<\{`（`re.escape`） |
| `{{…}}` | `{{.*}}` | `(?:.*)` |
| 捕获 | `[[row:#.*]]` | `(?P<row>#.*)` |
| 引用 | `[[row]]` | `re.escape(env["row"])` → `\#blocked` |
| 空白 | 连续空格 | `\s+` |

一个容易忽略的细节：`[[row:#.*]]` 里的 `.*` 是贪婪的，但它后面紧跟着字面的 ` = #ttg.blocked<{`，正则会回溯到正确的位置，`row` 最后只是 `#blocked`。

## ③ NEXT / SAME：行边界

[[here:solutions/t5_filecheck.py@L86-L89|_line_bounds（86–89 行）]] 给出某个位置所在行的 `[start, end)`。SAME 在“上一个匹配的末尾 … 行尾”里搜；NEXT 在“上一个匹配所在行的下一行”里搜。测试 `CHECK: tt.func @transpose` + `CHECK-NEXT: tt.load` 会失败：func 的下一行是 `convert_layout`。

## ④ DAG 与 COUNT

- DAG：一组连续的 `CHECK-DAG` 各自从同一个 pos 开始找，但已经被组内别的 DAG 用掉的区间不能再用；整组结束后 pos 跳到组内最靠后的匹配末尾。`CHECK-DAG: tt.store` + `CHECK-DAG: tt.load` 能过，换成普通 CHECK 就不行。
- COUNT-n：`parse_checks` 直接把它展开成 n 条普通 CHECK（[[here:solutions/t5_filecheck.py@L29-L43|parse_checks（29–43 行）]]）。真 FileCheck 的语义稍严格（n 次连续匹配，不允许中间夹别的 CHECK），对 Triton 测试的用法来说等价。

## ⑤ 变量的作用域

默认情况下变量在整个文件里有效；`--enable-var-scope` 时，普通变量在每个 LABEL 处清空，只有 `$` 开头的全局变量保留。**Triton 默认打开了它**：[[triton:test/lit.cfg.py@enable-var-scope|test/lit.cfg.py]] 里设置了 `FILECHECK_OPTS=--enable-var-scope`（“enabled by default in MLIR test”）。所以 accelerate-matmul.mlir 在文件头捕获 layout 时写成 `[[$MMA:.+]]`（这个文件里有 72 处 `[[$`），否则进入第一个 `CHECK-LABEL` 块后变量就失效了。写新测试时忘了 `$` 是很常见的失败原因；测试 `test_var_scope` 演示的就是这一点。

## ⑥ 在 Triton 仓库里：读、改、写

::: flow 一个 lit 测试的生命周期
cfg cw=210 nw=180
n f 0 0 "foo.mlir" "RUN + 输入 IR + CHECK" k=acc
n l 1 0 "lit" "替换 %s，执行 RUN" k=warn
n o 2 0 "triton-opt -pass" "输出变换后的 IR" k=vio
n fc 3 0 "FileCheck %s" "读同一文件的 CHECK" k=teal
n r 3 1.2 "PASS / FAIL" "失败时打印上下文" k=pass
e f -> l
e l -> o
e o -> fc
e fc -> r
:::

```bash
lit -v build/cmake.*/test/TritonGPU/coalesce.mlir      # 一个文件
triton-opt test/TritonGPU/coalesce.mlir -split-input-file -tritongpu-coalesce | less   # 看原始输出
```

写新测试时：输入 IR 尽量小（T4 的 ddmin）；只 CHECK 你的改动影响的东西；layout 用捕获变量而不是 `#blocked3` 这样的名字；`// -----` 分隔的每一段都要有自己的 `CHECK-LABEL`。

## ⑦ 源码与资料索引

- [[triton:test/TritonGPU/coalesce.mlir|coalesce.mlir]] · [[triton:test/TritonGPU/accelerate-matmul.mlir|accelerate-matmul.mlir]] · [[triton:test/Conversion/tritongpu_to_llvm.mlir@CHECK-COUNT-2|tritongpu_to_llvm.mlir]]
- [LLVM FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) · [lit](https://llvm.org/docs/CommandGuide/lit.html)
- [[triton:CONTRIBUTING.md|Triton CONTRIBUTING.md]]

## ⑧ 自测

::: answer 为什么 `CHECK-LABEL: @transpose` + `CHECK: arith.constant` + `CHECK-LABEL: @other` 会失败，而 arith.constant 明明在文件里？
LABEL 先把输入切成块：`@transpose` 块从它的位置一直到 `@other` 之前。`arith.constant` 只出现在 `@other` 之后，属于下一个块，第一个块里找不到。这正是 LABEL 的意义：防止一条 CHECK “偷看”到别的函数里的同名内容，让失败信息准确地指向出问题的函数。
:::

::: answer CHECK 和 CHECK-NEXT 该选哪个？
默认用 CHECK：它容忍中间插入无关的行，测试更稳。只有“紧挨着”本身就是要检查的性质时才用 NEXT（比如两条指令必须相邻、没有 barrier 插在中间）。过多的 NEXT 会让任何无害的输出变化都打破测试，reviewer 常会要求放宽。
:::
