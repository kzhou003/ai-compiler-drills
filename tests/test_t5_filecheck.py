import pytest

from drill import load

t5 = load("t5_filecheck")
FC = t5.filecheck
Fail = t5.CheckFailure

OUT = """#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
module {
  tt.func @transpose(%arg0: !tt.ptr<f32>) {
    %8 = ttg.convert_layout %7 : tensor<64x64x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %9 = tt.load %8 : tensor<64x64x!tt.ptr<f32>, #blocked>
    %10 = ttg.convert_layout %9 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.store %11, %10 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
  tt.func @other() {
    %0 = arith.constant 0 : i32
    tt.return
  }
}"""


def test_parse_checks():
    src = "// CHECK: a\n// CHECK-NEXT: b\n  // CHECK-COUNT-2: c\n// CHECK-NOT: d\n// CHECK-LABEL: @f\n; CHECK-SAME: e"
    kinds = [k for k, _, _ in t5.parse_checks(src)]
    assert kinds == ["CHECK", "NEXT", "CHECK", "CHECK", "NOT", "LABEL", "SAME"]
    assert t5.parse_checks("// MMA: x", prefix="MMA") == [("CHECK", "x", 1)]


def test_compile_pattern():
    rx, names = t5.compile_pattern("[[L:#.*]] = #ttg.blocked<{{.*}}order = [1, 0]", {})
    assert names == ["L"] and rx.search(OUT).group("L") == "#blocked"
    rx, _ = t5.compile_pattern("tt.load   [[P]]", {"P": "%8"})
    assert rx.search("%9 = tt.load %8 : x")


def test_real_coalesce_style_checks_pass():
    checks = """
// CHECK: [[row:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16]
// CHECK: [[col:#.*]] = #ttg.blocked<{sizePerThread = [4, 1]
// CHECK-LABEL: @transpose
// CHECK: [[ptr:%.*]] = ttg.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[row]]>
// CHECK-NEXT: [[val:%[0-9]+]] = tt.load [[ptr]]
// CHECK-SAME: [[row]]>
// CHECK-NOT: tt.dot
// CHECK: tt.store {{.*}}, [[col]]>
"""
    assert FC(checks, OUT)


def test_failures_are_reported():
    with pytest.raises(Fail, match="not found"):
        FC("// CHECK: tt.dot", OUT)
    with pytest.raises(Fail, match="CHECK-NEXT"):
        FC("// CHECK: tt.func @transpose\n// CHECK-NEXT: tt.load", OUT)
    with pytest.raises(Fail, match="CHECK-NOT"):
        FC("// CHECK: @transpose\n// CHECK-NOT: tt.load\n// CHECK: tt.store", OUT)
    with pytest.raises(Fail, match="no check strings"):
        FC("// nothing here", OUT)


def test_order_matters_unless_dag():
    with pytest.raises(Fail):
        FC("// CHECK: tt.store\n// CHECK: tt.load", OUT)
    assert FC("// CHECK-DAG: tt.store\n// CHECK-DAG: tt.load", OUT)


def test_labels_bound_the_search():
    # arith.constant exists, but only in @other: it must not satisfy a check in the @transpose block
    with pytest.raises(Fail):
        FC("// CHECK-LABEL: @transpose\n// CHECK: arith.constant\n// CHECK-LABEL: @other", OUT)
    assert FC("// CHECK-LABEL: @transpose\n// CHECK: tt.return\n// CHECK-LABEL: @other\n// CHECK: arith.constant", OUT)


def test_count_and_variables():
    assert FC("// CHECK-COUNT-2: ttg.convert_layout", OUT)
    with pytest.raises(Fail):
        FC("// CHECK-COUNT-3: ttg.convert_layout", OUT)
    with pytest.raises(Fail, match="undefined"):
        FC("// CHECK: tt.load [[NOPE]]", OUT)


def test_var_scope():
    src = "// CHECK: [[B:#blocked1]] =\n// CHECK-LABEL: @transpose\n// CHECK: tt.store {{.*}}[[B]]>"
    assert FC(src, OUT)
    with pytest.raises(Fail, match="undefined"):
        FC(src, OUT, enable_var_scope=True)
    assert FC(src.replace("[[B", "[[$B"), OUT, enable_var_scope=True)


def test_split_input_file():
    parts = t5.split_input_file("a\n// -----\nb\nc\n// -----\nd")
    assert parts == ["a", "b\nc", "d"]
