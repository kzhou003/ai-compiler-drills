from pathlib import Path

from drill import load

t1 = load("t1_ir_anatomy")
FX = Path(__file__).resolve().parent.parent / "fixtures" / "t1"


def read(name):
    return (FX / name).read_text()


def test_detect_stage():
    assert t1.detect_stage(read("add.ttir.mlir")) == "ttir"
    assert t1.detect_stage(read("add.ttgir.mlir")) == "ttgir"
    assert t1.detect_stage(read("matmul.ttgir.mlir")) == "ttgir"
    assert t1.detect_stage(read("matmul.llir.ll")) == "llir"
    assert t1.detect_stage(read("matmul.ptx")) == "ptx"


def test_ops_and_dialects():
    ops = t1.mlir_ops(read("add.ttir.mlir"))
    assert ops["tt.load"] == 2 and ops["tt.store"] == 1 and ops["tt.splat"] == 5
    assert ops["tt.addptr"] == 3 and ops["arith.addf"] == 1 and ops["tt.return"] == 1
    assert ops["tt.func"] == 1
    d = t1.dialect_histogram(read("matmul.ttgir.mlir"))
    assert d["ttg"] >= 6 and d["scf"] == 2 and d["tt"] >= 3
    mm = t1.mlir_ops(read("matmul.ttgir.mlir"))
    assert mm["tt.dot"] == 1 and mm["ttg.local_load"] == 2 and mm["ttg.convert_layout"] == 1


def test_same_ops_ttir_vs_ttgir():
    # adding layouts must not change which ops exist in this simple kernel
    assert t1.mlir_ops(read("add.ttir.mlir")) == t1.mlir_ops(read("add.ttgir.mlir"))


def test_layout_aliases():
    a = t1.layout_aliases(read("matmul.ttgir.mlir"))
    assert a["#blocked"] == {"kind": "blocked", "sizePerThread": [1, 8], "threadsPerWarp": [4, 8],
                             "warpsPerCTA": [4, 1], "order": [1, 0]}
    assert a["#mma"]["kind"] == "nvidia_mma" and a["#mma"]["versionMajor"] == 2
    assert a["#mma"]["instrShape"] == [16, 8] and a["#mma"]["warpsPerCTA"] == [2, 2]
    assert a["#shared"]["kind"] == "swizzled_shared" and a["#shared"]["vec"] == 8
    assert a["#smem"] == {"kind": "shared_memory"}
    assert t1.layout_aliases(read("add.ttir.mlir")) == {}


def test_parse_attr_body_nested():
    d = t1.parse_attr_body("opIdx = 0, parent = #mma, kWidth = 2")
    assert d == {"opIdx": 0, "parent": "#mma", "kWidth": 2}


def test_convert_layouts():
    cv = t1.convert_layouts(read("matmul.ttgir.mlir"))
    assert cv == [("#mma", "#blocked1")]
    assert t1.convert_layouts(read("add.ttgir.mlir")) == []


def test_ptx_instructions():
    p = t1.ptx_instructions(read("matmul.ptx"))
    mma = p["mma"]
    assert sum(mma.values()) == 2
    (name,) = mma
    assert name == "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    assert t1.mma_shape(name) == (16, 8, 16)
    assert sum(p["ldmatrix"].values()) == 2 and "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16" in p["ldmatrix"]
    assert p["cp_async"]["cp.async.wait_group"] == 1 and p["barrier"]["bar.sync"] == 1
    assert p["global_vec"]["st.global.v4.b32"] == 1
    # the LLVM IR carries the same instructions as inline asm
    assert t1.ptx_instructions(read("matmul.llir.ll"))["mma"] == mma.__class__({name: 1})


def test_concept_diff_tells_the_story():
    stages = [(s, read(f)) for s, f in [("ttir", "matmul.ttir.mlir"), ("ttgir", "matmul.ttgir.mlir"),
                                        ("llir", "matmul.llir.ll"), ("ptx", "matmul.ptx")]]
    diff = {(a, b): (app, gone) for a, b, app, gone in t1.concept_diff(stages)}
    app, gone = diff[("ttir", "ttgir")]
    assert "layout_encoding" in app and "shared_memory" in app and "async_copy" in app and gone == []
    app, gone = diff[("ttgir", "llir")]
    assert "mma_instruction" in app and "thread_id" in app
    assert {"block_tensor", "layout_encoding", "high_level_dot", "program_id"} <= set(gone)
    app, gone = diff[("llir", "ptx")]
    assert "virtual_registers" in app
