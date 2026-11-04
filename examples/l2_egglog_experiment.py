"""L2 deliverable: the same comm + assoc saturation in egglog (pip install egglog). Run: .venv/bin/python examples/l2_egglog_experiment.py"""
from __future__ import annotations
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...

egraph = EGraph()
a, b, c = Num.var("a"), Num.var("b"), Num.var("c")
left = egraph.let("left", (a + b) + c)
right = egraph.let("right", a + (b + c))

@egraph.register
def _rules(x: Num, y: Num, z: Num):
    yield rewrite(x + y).to(y + x)
    yield rewrite((x + y) + z).to(x + (y + z))

report = egraph.run(10)
egraph.check(eq(left).to(right))
print("equal after saturation")
print(egraph.extract(left))
print(len(report.iterations), "iterations")
