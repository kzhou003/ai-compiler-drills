title: L2 讲解：一步一步看 e-graph 饱和
lede: 从 (a+b)+c 开始，把参考解的每一轮 equality saturation 都展开：哪些规则匹配了、加了哪些 e-node、哪些 e-class 被合并、rebuild 修复了什么，直到 7 个 e-class、15 个 e-node 时饱和；再看 extraction 怎样从一个带环的 e-class 里取出最便宜的程序，以及 Luminal 为什么只对整数开放交换律。

::: callout info 怎么读这一页
对照 [[here:solutions/l2_egraph.py|l2_egraph.py]]。e-node 写成元组 `("+", c1, c2)`，其中 c1、c2 是 e-class 编号；叶子是 `("a",)`。
:::

## ① 起点：4 个 e-class

`g.add(("+", ("+", "a", "b"), "c"))`（[[here:solutions/l2_egraph.py@L38-L51|add_enode / add（38–51 行）]]）：

| e-class | e-node |
|---|---|
| 0 | `a` |
| 1 | `b` |
| 2 | `+(0, 1)` |
| 3 | `c` |
| 4 | `+(2, 3)` |

hashcons 保证同一个 e-node 只出现一次：再 `add(("+", "a", "b"))` 会直接返回 2。

## ② 饱和的四轮

规则：comm `?a + ?b → ?b + ?a`；assoc `(?a + ?b) + ?c → ?a + (?b + ?c)`。[[here:solutions/l2_egraph.py@L126-L146|run（126–146 行）]] 每一轮“先收集所有匹配，再统一应用，再 rebuild”。

| 轮 | 发生了什么 | e-class / e-node |
|---|---|---|
| 0 | 3 个匹配：comm 给 `a+b` 加 `b+a`、给 `(a+b)+c` 加 `c+(a+b)`；assoc 加出 `a+(b+c)`，并新建类 `b+c` | 6 / 9 |
| 1 | 8 个匹配：新的 `b+c`、`a+(b+c)` 等继续匹配，出现 `c+b`、`(b+c)+a`… 以及第 7 个类 `a+c` | 7 / 13 |
| 2 | 14 个匹配，补上最后两个 e-node | 7 / 15 |
| 3 | 18 个匹配全部只产生已有的 e-node，`unions = 0`，大小不变 → **saturated** | 7 / 15 |

（这些数就是 `rep["history"]` 的内容，可以自己打印验证。）最终形状：

::: flow 饱和后的 e-graph：7 个类，15 个节点
cfg cw=150 nw=128 rh=92
n a 0 2.2 "a" "" k=acc w=0.6
n b 1 2.2 "b" "" k=acc w=0.6
n c 2 2.2 "c" "" k=acc w=0.6
n ab 0 1.1 "{a+b, b+a}" "" k=warn
n ac 1 1.1 "{a+c, c+a}" "" k=warn
n bc 2 1.1 "{b+c, c+b}" "" k=warn
n top 1 0 "{(ab)+c, c+(ab), …}" "6 个 e-node" k=vio w=1.6
e a -> ab
e b -> ab
e a -> ac
e c -> ac
e b -> bc
e c -> bc
e ab -> top
e ac -> top
e bc -> top
:::

## ③ rebuild：合并之后的连锁反应

[[here:solutions/l2_egraph.py@L53-L84|union / rebuild（53–84 行）]]。测试 `test_union_and_congruence`：

```text
f(a) 在类 X，f(b) 在类 Y          union(a, b)
→ 规范化后 f(a) 和 f(b) 都变成 f(代表元)
→ 同一个 e-node 出现在两个类里 → 必须 union(X, Y)
```

rebuild 一直循环到没有新的合并为止，因为一次 congruence 合并可能引发更上层的另一次。egg 论文的关键贡献就是把这一步“延迟并批量”做（rebuild 而不是每次 union 都修），参考解用最朴素的全量重扫来实现同样的语义。

::: callout analogy 通讯录合并
两个联系人原来是同一个人（union），那么“他的公司”“他的住址”这些引用他的条目也应该合并（congruence）。一次合并可能触发下一层合并：两家公司其实是同一家，那“公司地址”也是同一个。rebuild 就是反复扫描通讯录，直到再也找不到重复。
:::

## ④ e-matching

[[here:solutions/l2_egraph.py@L96-L124|ematch / instantiate（96–124 行）]] 对模式 `("+", ("+", "?x", "?y"), "?z")` 和类 4：遍历类 4 里所有 `+` 节点，左孩子类 2 里再找 `+` 节点……每条路径产出一个替换 `{?x: 0, ?y: 1, ?z: 3}`。同名变量必须绑定到同一个类：`("+", "?x", "?x")` 只匹配 `a+a`，不匹配 `a+b`。

## ⑤ 爆炸：n 项求和

| 项数 n | e-class = 2ⁿ − 1 | e-node = 3ⁿ − 2ⁿ⁺¹ + 1 + n |
|---|---|---|
| 3 | 7 | 15 |
| 4 | 15 | 54 |
| 5 | 31 | 185 |
| 6 | 63 | 608 |

每个非空子集都是一个 e-class；一个 k 元子集的类里，每种“分成两个非空部分并排序”的方式都是一个 e-node（2ᵏ − 2 个）。这就是 `node_limit`、`iter_limit` 必须存在的原因（`test_budgets_stop_the_search`），也是 Luminal 用 `Select` 而不是算术掩码实现 `maximum` 的原因（L1）。

## ⑥ extraction：带环也能取

[[here:solutions/l2_egraph.py@L149-L181|extract（149–181 行）]]。`x · 1` 在 ALGEBRA 规则下和 `x` 同类，于是那个类里有 `*(自己, 1)` 这样的环。自底向上不动点迭代：先给叶子定代价 0，再反复更新“所有孩子都有代价”的 e-node，直到没有变化。

| 起点 | 饱和后 | op_cost 最小 |
|---|---|---|
| `x·y + x·z`（代价 4+4+1 = 9） | 同类里有 `x·(y+z)` | **5**：`x * (y + z)`（只有一次乘法） |
| `exp2(log2(x)) · 1` | 同类里有 `x` | **0**：`x` |

## ⑦ Luminal 的规则长什么样

[[luminal:src/logical/op/add/rewrite.egg|src/logical/op/add/rewrite.egg]] 对每一种**整数** dtype 各写一条交换律：

```text
(rule ((= ?out (LogicalAdd ?lhs ?rhs)) (= (dtype-of ?out) (Int)))
 ((union ?out (LogicalAdd ?rhs ?lhs)))
 :ruleset main_ruleset :name "add/rewrite: commutativity, Int")
```

条件里的 `(= (dtype-of ?out) (Int))` 让规则在浮点上根本不触发（“fail closed on unknown dtype”）。对照你的 `COMM_ASSOC`：它对所有 `+` 都生效——这在实数上对，在浮点上不对（测试 `test_reassociation_is_not_bit_exact`：`(1e8 + −1e8) + 1 = 1`，`1e8 + (−1e8 + 1) = 0`）。

## ⑧ 源码与资料索引

- [[luminal:src/egglog_core/egglog_preamble.egg@core: additive identity|preamble：整数恒等式与常量折叠]] · [[luminal:src/logical/op/add/rewrite.egg|add/rewrite.egg]]
- [[luminal:src/extraction.rs@pub struct Genome|extraction.rs：Genome]]（L3：真实的 extraction 是搜索出来的）
- [[egglog:README.md|egglog]] · [[egglogpy:README.md|egglog-python]] · [egg 论文](https://arxiv.org/abs/2004.03082)
- 本仓库的 egglog 实验：[examples/l2_egglog_experiment.py](../examples/l2_egglog_experiment.py)

## ⑨ 自测

::: answer 如果把 run() 改成“每找到一个匹配就立刻 union”，会有什么问题？
同一轮的结果会依赖遍历顺序：先应用的规则产生的新 e-node 可能在同一轮被后面的规则匹配到，饱和的轮数和中间状态都不确定，调试和设置预算都变难。而且 union 后不 rebuild 就继续匹配，hashcons 是过期的，可能匹配到非规范的 e-node。“收集 → 应用 → rebuild”三段式是 egg / egglog 的标准做法。
:::

::: answer extraction 用 ast_size 和 op_cost 会选出不同的程序吗？
可能。对 `x·y + x·z` 两者恰好一致：ast_size 下原式 7 个节点、`x·(y+z)` 5 个；op_cost 下 9 对 5（测试里两者都返回 `x * (z + y)`）。但只要两个候选“节点数少的那个反而含有更贵的运算”（比如用一个 `exp2` 换掉两次乘法），两者就会分道扬镳。代价函数就是“你认为什么快”的编码——Luminal 不想写死它，所以 L3 改成实测。
:::
